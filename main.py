#!/usr/bin/env python3
"""Pipeline driver that orchestrates the full screening workflow.

Steps:
1. Run `current_baseline_data.py` to refresh `current_baseline_data.csv`.
2. Run `fetch_wacc.py` to refresh `wacc_top.csv`, with `current_baseline_data.csv` as input.
3. Run `normalized_austrian_screener.py` to refresh `normalized_austrian.csv`, with `current_baseline_data.csv` and `wacc_top.csv` as input.
4. Sort `normalized_austrian.csv` by `rankingScore` ascending and pick the top *N* tickers (default 50).
5. Run `compute_roiic.py` to refresh `roiic_top.csv`, with `normalized_austrian.csv` as input.
6. Compute Growth Gate as roiic - wacc
7. Merge the key metrics into a concise overview CSV (default: top50_overview.csv)

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
        help="Skip running current_baseline_data.py if current_baseline_data.csv already exists.",
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
    # 2. Process all tickers (no pre-filtering needed since ranking happens later)
    # ------------------------------------------------------------------
    df_base = pd.read_csv(current_baseline_data_csv, sep=";")
    
    # Get all available tickers for WACC processing
    all_tickers: List[str] = df_base["symbol"].dropna().astype(str).tolist()

    if not all_tickers:
        print("❌ No tickers found in baseline data.", file=sys.stderr)
        sys.exit(1)

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
    # Data is already sorted by rankingScore ascending (best first)
    top_df = norm_df.head(args.top)
    top_tickers = top_df["symbol"].dropna().astype(str).tolist()

    ticker_str = ",".join(top_tickers)

    # ------------------------------------------------------------------
    # 6. Compute ROIIC for the full normalized dataset  
    # ------------------------------------------------------------------
    _run_script("compute_roiic.py", "--input", "normalized_austrian.csv", "--output", "roiic_top.csv")

    # ------------------------------------------------------------------
    # 7. Merge selected metrics into summary CSV
    # ------------------------------------------------------------------
    # Load ROIIC data
    df_roiic = pd.read_csv("roiic_top.csv", sep=";")
    
    # Merge top tickers with ROIIC data
    merged = top_df.merge(df_roiic[["symbol", "roiic"]], on="symbol", how="left")
    
    # Compute Growth Gate as roiic - wacc (if both available)
    if "wacc" in merged.columns and "roiic" in merged.columns:
        merged["growthGate"] = merged["roiic"] - merged["wacc"]

    # Select and reorder columns
    cols_order = [
        "symbol",
        "industry",
        "MarketCap",
        "roic",        
        "wacc",
        "roiic",
        "valueMetric",
        "valueMetricRank",
        "excessReturn",
        "excessReturnRank",
        "rankingScore",
        "growthGate",
    ]
    # Ensure all expected columns exist
    for col in cols_order:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[cols_order]

    merged = merged.sort_values("growthGate", ascending=False)

    merged.to_csv(args.output, sep=";", index=False)
    print(f"✓ Saved summary → {args.output.resolve()}")


if __name__ == "__main__":
    main() 