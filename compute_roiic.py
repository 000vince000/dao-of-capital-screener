#!/usr/bin/env python3
"""Compute ROIIC (Return On Incremental Invested Capital) for a list of tickers.

Overview
~~~~~~~~
• The script reads a universe of symbols from *normalized_austrian.csv* (semicolon-separated).
• For every ticker it downloads up to the last eight fiscal years of **annual**
  income-statement and balance-sheet data via *yahooquery*.
• It calculates **NOPAT** (EBIT × (1 − tax rate)) and pairs it with the company’s
  **Invested Capital** for each year.
           ┌─>  ΔIC / IC₀  < 10%   ?  (Filter A ─ material investment) --> set ROIIC to None
Input data ┤
           └─>  Regression slope ROIIC
                     │
                     ▼
               Winsorise ±40%        (Filter B ─ kill spikes)
                     │
                     ▼
           ( ROIIC   –   WACC   ≥ 0 )   ?   →   PASS / FAIL

• A straight-line regression is then fitted separately to the NOPAT and
  Invested-Capital series:  slope(NOPAT) / slope(InvestedCapital) = ROIIC.
  – At least **four** annual data points are required, otherwise ROIIC is
    recorded as *None*.
  – If the invested-capital slope is **zero** the ratio is undefined ⇒ ROIIC = None.
  – A *negative* capital slope **is allowed** and will yield a negative ROIIC.
• Optionally the most recent year can be *supplemented* from
  *current_baseline_data.csv* to include fresh data not yet covered by Yahoo.
• Results are written to *roiic_top.csv* with columns:
    symbol; roiic; data_points_used

Command-line flags (see --help for full list)
-------------------------------------------
--input       Path to normalized screener CSV (default normalized_austrian.csv)
--baseline    Optional baseline file to enrich the time series with one extra
              year (default current_baseline_data.csv)
--output      Destination CSV (default roiic_top.csv)
--rate-limit  Seconds to sleep between API calls (default 1.0)
--max-count   Upper bound of tickers processed (default 100)

Typical usage
-------------
$ python compute_roiic.py --input normalized_austrian.csv \
                         --baseline current_baseline_data.csv \
                         --output roiic_top.csv
"""

import argparse
import pathlib
import sys
from typing import List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from yahooquery import Ticker

# Suppress yahooquery warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="yahooquery")

# Import rate limiting utilities
from data_fetch_utils import fetch_with_backoff, RateLimitExceeded, BASE_DELAY_SEC


def fetch_annual_data(symbol: str, delay_ref: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch annual income statement and balance sheet data for a single symbol."""
    ticker = Ticker(symbol, asynchronous=False)
    
    # Fetch annual income statement for EBIT and tax data
    income_annual = fetch_with_backoff(
        lambda: ticker.income_statement(frequency="a"),
        desc=f"annual income {symbol}",
        delay_ref=delay_ref,
    )
    
    # Fetch annual balance sheet for InvestedCapital
    balance_annual = fetch_with_backoff(
        lambda: ticker.balance_sheet(frequency="a"),
        desc=f"annual balance {symbol}",
        delay_ref=delay_ref,
    )
    
    return income_annual, balance_annual


def compute_nopat_and_invested_capital(income_df: pd.DataFrame, balance_df: pd.DataFrame) -> pd.DataFrame:
    """Compute NOPAT and InvestedCapital for annual data points."""
    # Filter for the symbol and merge on asOfDate
    if income_df.empty or balance_df.empty:
        return pd.DataFrame()
    
    # Filter for annual data only (periodType == '12M')
    if 'periodType' in income_df.columns:
        income_annual_only = income_df[income_df['periodType'] == '12M']
    else:
        income_annual_only = income_df
        
    if 'periodType' in balance_df.columns:
        balance_annual_only = balance_df[balance_df['periodType'] == '12M']
    else:
        balance_annual_only = balance_df
    
    # Get the most recent 8 years of data (to have enough points)
    income_recent = income_annual_only.sort_values("asOfDate").tail(8)
    balance_recent = balance_annual_only.sort_values("asOfDate").tail(8)
    
    # Clean data before merging (drop rows with NaN in required columns)
    income_clean = income_recent[["asOfDate", "EBIT", "TaxRateForCalcs"]].dropna()
    balance_clean = balance_recent[["asOfDate", "InvestedCapital"]].dropna()
    
    # Merge on asOfDate
    merged = pd.merge(
        income_clean,
        balance_clean,
        on="asOfDate",
        how="inner"
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Compute NOPAT = EBIT * (1 - TaxRate)
    merged["nopat"] = merged["EBIT"] * (1 - merged["TaxRateForCalcs"])
    
    # Extract year from asOfDate for regression
    merged["year"] = pd.to_datetime(merged["asOfDate"]).dt.year
    
    return merged[["year", "nopat", "InvestedCapital"]].dropna()


def compute_roiic_slope(data: pd.DataFrame) -> Optional[float]:
    """Compute ROIIC using regression slopes: slope(NOPAT) / slope(InvestedCapital)."""
    if len(data) < 4:  # Need at least 4 points for meaningful regression
        return None
    
    # Sort by year to ensure proper time series
    data = data.sort_values("year")
    
    years = data["year"].values
    nopat = data["nopat"].values
    invested_capital = data["InvestedCapital"].values

    # ------------------------------------------------------------------
    # Filter A – ensure the capital base moved by at least 10 %
    # Otherwise incremental returns are not meaningful.
    # ------------------------------------------------------------------
    ic_first = invested_capital[0]
    ic_last = invested_capital[-1]
    # Protect against divide-by-zero
    if ic_first == 0 or abs(ic_last - ic_first) / abs(ic_first) < 0.10:
        return None

    # Fit linear regression: y = slope * x + intercept
    try:
        nopat_slope, _, _, _, _ = stats.linregress(years, nopat)
        ic_slope, _, _, _, _ = stats.linregress(years, invested_capital)
        
        # Avoid division by zero (but negative denominators are valid)
        if ic_slope == 0:
            return None
            
        roiic_raw = nopat_slope / ic_slope

        # ------------------------------------------------------------------
        # Filter B – winsorise extreme ROIIC values to ±40 %
        # ------------------------------------------------------------------
        roiic = max(min(roiic_raw, 0.40), -0.40)

        return roiic
        
    except (ValueError, ZeroDivisionError):
        return None


def process_ticker(symbol: str, delay_ref: List[float], baseline_data: Optional[pd.DataFrame] = None) -> Tuple[str, Optional[float], int]:
    """Process a single ticker to compute ROIIC."""
    try:
        print(f"  Processing {symbol}...", flush=True)
        
        # Fetch historical annual data
        income_annual, balance_annual = fetch_annual_data(symbol, delay_ref)
        
        if income_annual.empty or balance_annual.empty:
            print(f"    · No annual data available for {symbol}")
            return symbol, None, 0
            

        
        # Filter for this symbol (if symbol column exists, otherwise use all data)
        if "symbol" in income_annual.columns:
            symbol_income = income_annual[income_annual["symbol"] == symbol]
        else:
            symbol_income = income_annual  # Single symbol data
            
        if "symbol" in balance_annual.columns:
            symbol_balance = balance_annual[balance_annual["symbol"] == symbol]
        else:
            symbol_balance = balance_annual  # Single symbol data
        
        # Compute historical NOPAT and InvestedCapital
        historical_data = compute_nopat_and_invested_capital(symbol_income, symbol_balance)
        
        if historical_data.empty:
            print(f"    · Could not compute historical metrics for {symbol}")
            return symbol, None, 0
        
        # Optionally add most recent year from baseline data if available
        if baseline_data is not None:
            baseline_row = baseline_data[baseline_data["symbol"] == symbol]
            if not baseline_row.empty and "nopat" in baseline_row.columns:
                # Extract year from baseline asOfDate
                baseline_year = pd.to_datetime(baseline_row["asOfDate"].iloc[0]).year
                baseline_nopat = baseline_row["nopat"].iloc[0]
                baseline_ic = baseline_row["InvestedCapital"].iloc[0]
                
                # Add to historical data if not already present
                if baseline_year not in historical_data["year"].values:
                    new_row = pd.DataFrame({
                        "year": [baseline_year],
                        "nopat": [baseline_nopat],
                        "InvestedCapital": [baseline_ic]
                    })
                    historical_data = pd.concat([historical_data, new_row], ignore_index=True)
        
        # Compute ROIIC using slope method
        roiic = compute_roiic_slope(historical_data)
        data_points = len(historical_data)
        
        if roiic is not None:
            print(f"    · ROIIC: {roiic:.4f} (using {data_points} data points)")
        else:
            print(f"    · Could not compute ROIIC for {symbol} ({data_points} data points)")
        
        return symbol, roiic, data_points
        
    except Exception as e:
        print(f"    · Error processing {symbol}: {e}")
        return symbol, None, 0


def main():
    parser = argparse.ArgumentParser(description="Compute ROIIC using regression slopes on annual data")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("normalized_austrian.csv"),
        help="Input file containing ticker symbols (default: normalized_austrian.csv)"
    )
    parser.add_argument(
        "--baseline",
        type=pathlib.Path,
        default=pathlib.Path("current_baseline_data.csv"),
        help="Optional baseline data to supplement with recent year (default: current_baseline_data.csv)"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("roiic_top.csv"),
        help="Output CSV file (default: roiic_top.csv)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls (default: 1.0)"
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=100,
        help="Maximum number of tickers to process (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Load input data
    if not args.input.exists():
        print(f"❌ Input file {args.input} not found")
        sys.exit(1)
    
    input_df = pd.read_csv(args.input, sep=";")
    if "symbol" not in input_df.columns:
        print(f"❌ 'symbol' column not found in {args.input}")
        sys.exit(1)
    
    # Load baseline data if available
    baseline_data = None
    if args.baseline.exists():
        try:
            baseline_data = pd.read_csv(args.baseline, sep=";")
            print(f"✓ Loaded baseline data from {args.baseline}")
        except Exception as e:
            print(f"⚠️  Could not load baseline data: {e}")
    
    # Get unique tickers
    tickers = input_df["symbol"].dropna().unique().tolist()[:args.max_count]
    print(f"Processing {len(tickers)} tickers for ROIIC computation...")
    
    # Process each ticker
    delay_ref = [float(args.rate_limit)]
    results = []
    
    for i, symbol in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}]", end=" ")
        symbol_clean = str(symbol).strip()
        
        try:
            symbol_result, roiic, data_points = process_ticker(symbol_clean, delay_ref, baseline_data)
            results.append({
                "symbol": symbol_result,
                "roiic": roiic,
                "data_points_used": data_points
            })
        except RateLimitExceeded:
            print("❌ Rate limit exceeded, stopping early")
            break
        except Exception as e:
            print(f"    · Unexpected error for {symbol_clean}: {e}")
            results.append({
                "symbol": symbol_clean,
                "roiic": None,
                "data_points_used": 0
            })
    
    # Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out symbols with no ROIIC computed
    valid_results = results_df[results_df["roiic"].notna()]
    
    print(f"\n📊 Successfully computed ROIIC for {len(valid_results)} out of {len(results)} tickers")
    
    # Sort by ROIIC descending (higher is better)
    if not valid_results.empty:
        valid_results = valid_results.sort_values("roiic", ascending=False)
        print(f"Top 10 ROIIC values:")
        for _, row in valid_results.head(10).iterrows():
            print(f"  {row['symbol']}: {row['roiic']:.4f} ({row['data_points_used']} points)")
    
    # Save results
    results_df.to_csv(args.output, sep=";", index=False)
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()