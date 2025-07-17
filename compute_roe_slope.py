#!/usr/bin/env python3
"""Compute 5-year ROE slope for a list of tickers.

This mirrors *compute_roic_slope.py* but uses Return on Equity (ROE):
    ROE = Net Income / Total Shareholder Equity

The script reads an input CSV (default *austrian.csv*), determines the ticker
universe, fetches annual financial statements via *yahooquery*, computes ROE
for up to five recent fiscal years, fits a simple linear regression (year index
vs ROE) if ≥2 data points, and writes the slope / stats per ticker to an output
CSV.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from yahooquery import Ticker

# --------------------------------------------------------------------------------------
# ROE computation helpers
# --------------------------------------------------------------------------------------


def _annual_roe(balance: pd.DataFrame, income: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by fiscal year (str) → ROE value.

    Aligns *balance* and *income* on `asOfDate`. Both frames must contain *symbol*.
    """

    merged = pd.merge(
        balance,
        income,
        on=["symbol", "asOfDate"],
        suffixes=("_bs", "_is"),
        how="outer",
    )

    # Identify usable column names
    ni_col: Optional[str] = None
    for cand in ["NetIncome", "NetIncomeLoss", "netIncome"]:
        if cand in merged.columns:
            ni_col = cand
            break

    equity_col: Optional[str] = None
    for cand in [
        "TotalShareholderEquity",
        "StockholdersEquity",
        "totalStockholderEquity",
    ]:
        if cand in merged.columns:
            equity_col = cand
            break

    if ni_col is None or equity_col is None:
        return pd.Series(dtype=float)

    merged["roe"] = merged[ni_col] / merged[equity_col].replace(0, np.nan)
    merged = merged.sort_values("asOfDate")
    return merged.set_index("asOfDate")["roe"].dropna()


def compute_roe_slope(symbol: str) -> Tuple[Optional[float], int, Optional[float]]:
    """Return (slope, year_count, avg_roe) for last 5 fiscal years.

    slope : float | None – linear slope of ROE vs time index if ≥2 points
    year_count : int      – number of non-null observations (≤5)
    avg_roe : float | None – arithmetic mean of available ROE values
    """
    try:
        ticker = Ticker(symbol)
        bs = ticker.balance_sheet(frequency="a")  # annual
        inc = ticker.income_statement(frequency="a")

        if "symbol" not in bs.columns:
            bs = bs.reset_index()
        if "symbol" not in inc.columns:
            inc = inc.reset_index()
    except Exception as exc:
        print(f"  · Yahoo error {symbol}: {exc}", file=sys.stderr)
        return None, 0, None

    # Filter for full year (12M) periods
    bs = bs[bs.get("periodType", "12M") == "12M"] if "periodType" in bs.columns else bs
    inc = inc[inc.get("periodType", "12M") == "12M"] if "periodType" in inc.columns else inc

    bs = bs[bs["symbol"] == symbol]
    inc = inc[inc["symbol"] == symbol]

    roe_hist = _annual_roe(bs, inc)
    if roe_hist.empty:
        return None, 0, None

    roe_series = roe_hist.tail(5)
    year_count = len(roe_series)
    avg = float(roe_series.mean()) if year_count else None

    slope_val: Optional[float] = None
    if year_count >= 2:
        y = roe_series.values
        x = np.arange(len(y))
        try:
            slope_val, _ = np.polyfit(x, y, 1)
            slope_val = float(slope_val)
        except Exception:
            slope_val = None

    return slope_val, year_count, avg


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 5-year ROE slope for tickers.")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("austrian.csv"),
        help="Input CSV with at least a 'symbol' column (default: austrian.csv)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("roe_slope.csv"),
        help="Destination CSV filename (default: roe_slope.csv)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated tickers; if omitted uses all symbols from --input.",
    )
    args = parser.parse_args()

    # Determine universe
    if args.tickers:
        tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not args.input.exists():
            print(f"Input file {args.input} not found and --tickers not provided.", file=sys.stderr)
            sys.exit(1)
        df_in = pd.read_csv(args.input, sep=";")
        if "symbol" not in df_in.columns:
            print("Input CSV lacks 'symbol' column.", file=sys.stderr)
            sys.exit(1)
        tickers = df_in["symbol"].dropna().unique().tolist()

    # Skip symbols already processed
    existing_df = None
    if args.output.exists():
        try:
            existing_df = pd.read_csv(args.output, sep=";")
            done = set(existing_df["symbol"].dropna().astype(str).unique())
            remaining = [t for t in tickers if t not in done]
            skipped = len(tickers) - len(remaining)
            if skipped:
                print(f"✓ {skipped} tickers already present in {args.output.name}; skipping.")
            tickers = remaining
        except Exception as exc:
            print(f"⚠️  Could not read existing output file: {exc} – recomputing all.")

    rows = []
    for sym in tickers:
        print(f"Processing {sym}…", flush=True)
        slope, yr_count, avg = compute_roe_slope(sym)
        rows.append({
            "symbol": sym,
            "roeSlope": slope,
            "yearsAvailable": yr_count,
            "averageROE": avg,
        })

    # Merge with prior results
    if existing_df is not None and not existing_df.empty:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="symbol", keep="first")
        out_df = combined
    else:
        out_df = pd.DataFrame(rows)

    out_df.to_csv(args.output, sep=";", index=False)
    print(f"Saved ROE slopes → {args.output.resolve()}")


if __name__ == "__main__":
    main() 