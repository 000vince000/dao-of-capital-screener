#!/usr/bin/env python3
"""Pipeline driver that orchestrates the full screening workflow.

Steps:
1. Run `austrian_stock_screener.py` to refresh `austrian.csv`.
2. Run `fetch_wacc.py` to refresh `wacc_top.csv`, with `austrian.csv` as input.
3. Run `normalized_austrian_screener.py` to refresh `normalized_austrian.csv`, with `austrian.csv` and `wacc_top.csv` as input.
4. Sort `normalized_austrian.csv` by `sumRanks` ascending and pick the top *N* tickers (default 50).
5. Call `compute_roic_slope.py` and `compute_roe_scope.py` for this subset.
6. Compute "projectedReturn24Months" as FCF Yield+(ROIC−WACC)+2×slope, where slope is the slope of the ROIC if roeFlag is TRUE or ROE slope if roeFlag is FALSE
7. Rank the whole list by projectedReturn24Months and sort as such.
8. Merge the key metrics into a concise overview CSV (default: top50_overview.csv).

This script assumes it is executed from the project root where the individual
Python modules reside.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full screener pipeline and produce summary CSV.")
    p.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of tickers to keep after ranking (default: 50)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("top50_overview.csv"),
        help="Destination CSV filename (default: top50_overview.csv)",
    )
    p.add_argument(
        "--skip-screener",
        action="store_true",
        help="Skip running austrian_stock_screener.py if austrian.csv already exists.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # current interpreter (inside venv if activated)


def _run_script(script: str, *args: str):
    """Run *script* with given *args* in subprocess and abort on non-zero exit."""
    cmd = [PYTHON, str(PROJECT_ROOT / script), *args]
    print(f"→ Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    austrian_csv = PROJECT_ROOT / "austrian.csv"

    # ------------------------------------------------------------------
    # 1. Run the screener (unless skipped)
    # ------------------------------------------------------------------
    if not args.skip_screener or not austrian_csv.exists():
        _run_script("austrian_stock_screener.py")
    else:
        print("✓ Skipping screener step – austrian.csv already present.")

    if not austrian_csv.exists():
        print("❌ Expected austrian.csv was not created.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Pick top-N tickers by sumRanks
    # ------------------------------------------------------------------
    df_base = pd.read_csv(austrian_csv, sep=";")
    if "sumRanks" not in df_base.columns:
        print("❌ Column 'sumRanks' not found in austrian.csv", file=sys.stderr)
        sys.exit(1)

    df_sorted = df_base.sort_values("sumRanks", ascending=True)
    top_df = df_sorted.head(args.top)
    top_tickers: List[str] = top_df["symbol"].dropna().astype(str).tolist()

    if not top_tickers:
        print("❌ No tickers found after ranking step.", file=sys.stderr)
        sys.exit(1)

    ticker_str = ",".join(top_tickers)

    # ------------------------------------------------------------------
    # 3. Refresh WACC for entire universe before ranking normalization
    # ------------------------------------------------------------------
    _run_script("fetch_wacc.py", "--input", str(austrian_csv), "--output", "wacc_top.csv")

    # ------------------------------------------------------------------
    # 4. Produce normalized screener with excess returns
    # ------------------------------------------------------------------
    _run_script("normalized_austrian_screener.py", "--input", str(austrian_csv), "--wacc-file", "wacc_top.csv", "--output", "normalized_austrian.csv")

    # ------------------------------------------------------------------
    # 5. Load normalized CSV and pick top-N
    # ------------------------------------------------------------------
    norm_df = pd.read_csv("normalized_austrian.csv", sep=";")
    if "faustmannRank" not in norm_df.columns and "sanitizedFaustmannRatio" in norm_df.columns:
        norm_df["faustmannRank"] = norm_df["sanitizedFaustmannRatio"].rank(method="min", ascending=True)
    df_sorted = norm_df.sort_values("sumRanks", ascending=True)
    top_df = df_sorted.head(args.top)
    top_tickers = top_df["symbol"].dropna().astype(str).tolist()

    ticker_str = ",".join(top_tickers)

    # ------------------------------------------------------------------
    # 6. Compute ROIC and ROE slopes for subset
    # ------------------------------------------------------------------
    _run_script("compute_roic_slope.py", "--tickers", ticker_str, "--output", "roic_slope_top.csv")
    _run_script("compute_roe_slope.py", "--tickers", ticker_str, "--output", "roe_slope_top.csv")

    # ------------------------------------------------------------------
    # 7. Merge selected metrics into summary CSV
    # ------------------------------------------------------------------
    df_base_subset = norm_df[
        [
            "symbol",
            "roic",
            "MarketCap",
            "sanitizedFaustmannRatio",
            "sumRanks",
            "fcfYield",
            "excessReturn",
            "excessReturnRank",
            "faustmannRank",
            "industry",
            "roeFlag",
            "roe",
        ]
    ]

    df_wacc = pd.read_csv("wacc_top.csv", sep=";")
    df_roe_slope = pd.read_csv("roe_slope_top.csv", sep=";")
    df_roic_slope = pd.read_csv("roic_slope_top.csv", sep=";")
    merged = (
        top_df[["symbol"]]  # ensure ordering of top tickers
        .merge(df_base_subset, on="symbol", how="left")
        .merge(df_wacc[["symbol", "wacc", "costOfEquity"]], on="symbol", how="left")
        .merge(df_roic_slope, on="symbol", how="left")
        .merge(df_roe_slope, on="symbol", how="left")
    )

    # Compute projectedReturn24Months
    merged["normalizedSlope"] = np.where(
        merged["roeFlag"], merged.get("roeSlope"), merged.get("roicSlope")
    )
    merged["projectedReturn24Months"] = (
        merged["fcfYield"] + merged["excessReturn"] + 2 * merged["normalizedSlope"]
    )

    # Select and reorder columns
    cols_order = [
        "symbol",
        "industry",
        "roeFlag",
        "MarketCap",
        "roic",
        "roe",
        "excessReturn",
        "excessReturnRank",
        "faustmannRank",
        "sumRanks",
        "fcfYield",
        "wacc",
        "costOfEquity",
        "roicSlope",
        "roeSlope",
        "normalizedSlope",
        "projectedReturn24Months",
    ]
    # Ensure all expected columns exist
    for col in cols_order:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[cols_order]

    merged = merged.sort_values("projectedReturn24Months", ascending=False)

    merged.to_csv(args.output, sep=";", index=False)
    print(f"✓ Saved summary → {args.output.resolve()}")


if __name__ == "__main__":
    main() 