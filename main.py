#!/usr/bin/env python3
"""Pipeline driver that orchestrates the full screening workflow.

Steps:
1. Run `current_baseline_data.py` to refresh `current_baseline_data.csv`.
2. Run `fetch_wacc.py` to refresh `wacc_top.csv`, with `current_baseline_data.csv` as input.
3. Run `normalized_austrian_screener.py` to refresh `normalized_austrian.csv`, with `current_baseline_data.csv` and `wacc_top.csv` as input.
4. Sort `normalized_austrian.csv` by `rankingScore` ascending and pick the top *N* tickers (default 50).
5. Run `compute_roiic.py` to refresh `roiic_top.csv`, with `normalized_austrian.csv` as input.
6. Compute "projectedReturn24Months" as Value Anchor + Quality Spread, with a Growth Gate
7. Rank the whole list by projectedReturn24Months and sort as such.
8. Merge the key metrics into a concise overview CSV (default: top50_overview.csv)

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
        help="Skip running current_baseline_data.py if austrian.csv already exists.",
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

    current_baseline_data_csv = PROJECT_ROOT / "current_baseline_data.csv"
    # ------------------------------------------------------------------
    # 1. Run the screener (unless skipped)
    # ------------------------------------------------------------------
    if not args.skip_screener or not current_baseline_data_csv.exists():
        _run_script("current_baseline_data.py")
    else:
        print("✓ Skipping screener step – current_baseline_data.csv already present.")

    if not current_baseline_data_csv.exists():
        print("❌ Expected current_baseline_data.csv was not created.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Pick top-N tickers by sumRanks
    # ------------------------------------------------------------------
    df_base = pd.read_csv(current_baseline_data_csv, sep=";")
    
    top_tickers: List[str] = top_df["symbol"].dropna().astype(str).tolist()

    if not top_tickers:
        print("❌ No tickers found after ranking step.", file=sys.stderr)
        sys.exit(1)

    ticker_str = ",".join(top_tickers)

    # ------------------------------------------------------------------
    # 3. Refresh WACC for entire universe before ranking normalization
    # ------------------------------------------------------------------
    _run_script("fetch_wacc.py", "--input", str(current_baseline_data_csv), "--output", "wacc_top.csv")

    # ------------------------------------------------------------------
    # 4. Produce normalized screener with excess returns
    # ------------------------------------------------------------------
    _run_script("normalized_austrian_screener.py", "--input", str(current_baseline_data_csv), "--wacc-file", "wacc_top.csv", "--output", "normalized_austrian.csv")

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
            "opCashFlowYield",
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
        merged["opCashFlowYield"] + merged["excessReturn"] + 2 * merged["normalizedSlope"]
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
        "opCashFlowYield",
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