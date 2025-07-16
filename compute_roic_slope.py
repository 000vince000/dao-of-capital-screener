#!/usr/bin/env python3
"""Compute 5-year ROIC slope for a list of tickers.

Reads an existing CSV (default: austrian.csv) produced by `austrian_stock_screener.py`,
fetches annual financial statements from Yahoo via `yahooquery`, computes ROIC for the
last five fiscal years, then fits a simple linear regression (year index 0-4 vs ROIC)
and stores the slope per ticker in an output CSV.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from yahooquery import Ticker


# --------------------------------------------------------------------------------------
# ROIC computation helpers
# --------------------------------------------------------------------------------------

def _annual_roic(balance: pd.DataFrame, income: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by fiscal year (str) → ROIC value.

    Both inputs must be filtered to a **single** symbol and contain matching fiscal
    periods. The function aligns on `asOfDate`.
    """
    # Align on asOfDate (outer join) to keep periods present in either table
    merged = pd.merge(
        balance,
        income,
        on=["symbol", "asOfDate"],
        suffixes=("_bs", "_is"),
        how="outer",
    )

    # Required columns may be missing; skip if so.
    required_cols = ["EBIT", "TaxRateForCalcs", "InvestedCapital"]
    for col in required_cols:
        if col not in merged.columns:
            return pd.Series(dtype=float)

    merged["nopat"] = merged["EBIT"] * (1 - merged["TaxRateForCalcs"].fillna(0))
    merged["roic"] = merged["nopat"] / merged["InvestedCapital"].replace(0, np.nan)

    merged = merged.sort_values("asOfDate")
    roic_series = merged.set_index("asOfDate")["roic"]
    return roic_series.dropna()


def compute_roic_slope(symbol: str):
    """Return (slope, year_count, avg_roic) for the last 5 fiscal years (max 10).

    slope : float | None – linear slope of ROIC vs year index (0 oldest) if ≥2 data points
    year_count : int      – number of non-null ROIC observations (max 5)
    avg_roic : float | None – arithmetic mean of the available ROIC values
    """
    try:
        ticker = Ticker(symbol)
        bs = ticker.balance_sheet(frequency="a")  # annual
        inc = ticker.income_statement(frequency="a")

        # Ensure 'symbol' is a column (yahooquery often returns it as index)
        if "symbol" not in bs.columns:
            bs = bs.reset_index()
        if "symbol" not in inc.columns:
            inc = inc.reset_index()
    except Exception as exc:
        print(f"  · Yahoo error {symbol}: {exc}", file=sys.stderr)
        return None

    # Filter for full-year periods (12M)
    bs = bs[bs["periodType"] == "12M"] if "periodType" in bs.columns else bs
    inc = inc[inc["periodType"] == "12M"] if "periodType" in inc.columns else inc

    # Keep only rows for this symbol; some calls return multi-symbol frames
    bs = bs[bs["symbol"] == symbol]
    inc = inc[inc["symbol"] == symbol]

    roic_hist = _annual_roic(bs, inc)
    if roic_hist.empty:
        return None, 0, None

    roic_series = roic_hist.tail(5)  # up to 5 most recent
    year_count = len(roic_series)
    avg = float(roic_series.mean()) if year_count > 0 else None

    slope_val: float | None = None
    if year_count >= 2:
        y = roic_series.values
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
    parser = argparse.ArgumentParser(description="Compute 5-year ROIC slope for tickers.")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("austrian.csv"),
        help="Input CSV containing at least a 'symbol' column (default: austrian.csv)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("roic_slope.csv"),
        help="Destination CSV filename (default: roic_slope.csv)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to process; if omitted use all symbols in --input.",
    )
    args = parser.parse_args()

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

    rows = []
    for sym in tickers:
        print(f"Processing {sym}…", flush=True)
        slope, yr_count, avg = compute_roic_slope(sym)
        rows.append({
            "symbol": sym,
            "roicSlope": slope,
            "yearsAvailable": yr_count,
            "averageROIC": avg,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, sep=";", index=False)
    print(f"Saved slopes → {args.output.resolve()}")


if __name__ == "__main__":
    main() 