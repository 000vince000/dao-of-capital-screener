#!/usr/bin/env python3
"""Pipeline driver that orchestrates the full screening workflow.

Steps:
1. Run `current_baseline_data.py` to refresh `current_baseline_data.csv`.
2. Run `fetch_wacc.py` to refresh `wacc_top.csv`, with `current_baseline_data.csv` as input.
3. Run `normalized_austrian_screener.py` to refresh `normalized_austrian.csv`, with `current_baseline_data.csv` and `wacc_top.csv` as input.
4. Sort `normalized_austrian.csv` by `rankingScore` ascending and pick the top *N* tickers (default 50).
5. Run `compute_roiic.py` to refresh `roiic_top.csv`, with `normalized_austrian.csv` as input.
6. Compute `growthGate` as roiic - wacc. This is a DIRECTIONAL signal only (is
   reinvestment trending better or worse than the cost of capital) — per
   Mauboussin & Callahan, "Return on Invested Capital" (Morgan Stanley
   Counterpoint Global, Oct 2022), ROIIC should not be compared to WACC as a
   true measure of economic value the way ROIC can be: it overstates value
   creation when positive and understates it when negative, ignoring the
   return on the (much larger) existing capital base. It is NOT used to rank
   or filter the final output; `rankingScore`, built from `excessReturn`
   (roic - wacc), is the WACC-anchored ranking metric.
7. Merge the key metrics into a concise overview CSV, sorted by `rankingScore`
   ascending (best first).

Every CSV this pipeline writes — including the intermediate stage outputs —
is dated and saved under `artifacts/` as `<run-date>_<name>.csv` rather than
overwriting a fixed filename in the repo root. This preserves a historical
record of each run's results (e.g. this year's screen vs. last year's) and
keeps the repo root free of stale, regenerable data. `artifacts/` is
git-ignored — it's a local record, not something committed.

This script assumes it is executed from the project root where the individual
Python modules reside.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
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
        default=None,
        help="Destination CSV path (default: artifacts/<today>_top50_overview.csv)",
    )
    p.add_argument(
        "--skip-screener",
        action="store_true",
        help="Skip running current_baseline_data.py if today's dated baseline CSV already exists.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUN_DATE = date.today().isoformat()
PYTHON = sys.executable  # current interpreter (inside venv if activated)


def _dated(name: str) -> Path:
    """Return artifacts/<RUN_DATE>_<name> — the dated output path for this run."""
    return ARTIFACTS_DIR / f"{RUN_DATE}_{name}"


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

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    current_baseline_data_csv = _dated("current_baseline_data.csv")
    # ------------------------------------------------------------------
    # 1. Run the screener (unless skipped)
    # ------------------------------------------------------------------
    if not args.skip_screener or not current_baseline_data_csv.exists():
        _run_script("current_baseline_data.py", "--output", str(current_baseline_data_csv))
    else:
        print(f"✓ Skipping screener step – {current_baseline_data_csv.name} already present.")

    if not current_baseline_data_csv.exists():
        print(f"❌ Expected {current_baseline_data_csv} was not created.", file=sys.stderr)
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
    wacc_csv = _dated("wacc_top.csv")
    wacc_failed_csv = _dated("wacc_failed.csv")
    if not wacc_csv.exists():
        _run_script(
            "fetch_wacc.py",
            "--input", str(current_baseline_data_csv),
            "--output", str(wacc_csv),
            "--failed-output", str(wacc_failed_csv),
        )
    else:
        print(f"✓ Skipping WACC fetch – {wacc_csv.name} already exists.")

    # ------------------------------------------------------------------
    # 4. Produce normalized screener with excess returns
    # ------------------------------------------------------------------
    normalized_csv = _dated("normalized_austrian.csv")
    if not normalized_csv.exists():
        _run_script(
            "normalized_austrian_screener.py",
            "--input", str(current_baseline_data_csv),
            "--wacc-file", str(wacc_csv),
            "--output", str(normalized_csv),
        )
    else:
        print(f"✓ Skipping normalized screener – {normalized_csv.name} already exists.")

    # ------------------------------------------------------------------
    # 5. Load normalized CSV and pick top-N
    # ------------------------------------------------------------------
    norm_df = pd.read_csv(normalized_csv, sep=";")
    # Data is already sorted by rankingScore ascending (best first)
    top_df = norm_df.head(args.top)
    top_tickers = top_df["symbol"].dropna().astype(str).tolist()

    ticker_str = ",".join(top_tickers)

    # ------------------------------------------------------------------
    # 6. Compute ROIIC for the full normalized dataset
    # ------------------------------------------------------------------
    roiic_csv = _dated("roiic_top.csv")
    if not roiic_csv.exists():
        _run_script(
            "compute_roiic.py",
            "--input", str(normalized_csv),
            "--baseline", str(current_baseline_data_csv),
            "--output", str(roiic_csv),
        )
    else:
        print(f"✓ Skipping ROIIC computation – {roiic_csv.name} already exists.")

    # ------------------------------------------------------------------
    # 7. Merge selected metrics into summary CSV
    # ------------------------------------------------------------------
    # Load ROIIC data
    df_roiic = pd.read_csv(roiic_csv, sep=";")
    
    # Merge top tickers with ROIIC data
    merged = top_df.merge(df_roiic[["symbol", "roiic"]], on="symbol", how="left")
    
    # Compute Growth Gate as roiic - wacc (if both available).
    # NOTE: directional context only, not a value-creation measure — see the
    # module docstring. Never sort or filter on this column; use rankingScore.
    if "wacc" in merged.columns and "roiic" in merged.columns:
        merged["growthGate"] = merged["roiic"] - merged["wacc"]

    # Merge with baseline data to get missing columns (industry, MarketCap)
    # Note: EBIT and EnterpriseValue are already in normalized_austrian.csv
    baseline_df = pd.read_csv(current_baseline_data_csv, sep=";")
    
    # Only include columns that actually exist in baseline data and are missing from merged
    baseline_cols = ["symbol"]
    missing_cols = ["industry", "MarketCap"]  # These should be added to baseline data
    for col in missing_cols:
        if col in baseline_df.columns:
            baseline_cols.append(col)
    
    # Merge with baseline data for missing columns
    if len(baseline_cols) > 1:  # Only merge if we have additional columns
        merged = merged.merge(baseline_df[baseline_cols], on="symbol", how="left")

        # ------------------------------------------------------------------
        # Consolidate potential duplicate columns created by the merge
        # (e.g. industry_x / industry_y). Prefer values from the original
        # normalized file but fall back to baseline data when missing.
        # ------------------------------------------------------------------
        def _consolidate(col: str):
            x, y = f"{col}_x", f"{col}_y"
            if x in merged.columns and y in merged.columns:
                merged[col] = merged[x].combine_first(merged[y])
                merged.drop(columns=[x, y], inplace=True)
            elif x in merged.columns:
                merged.rename(columns={x: col}, inplace=True)
            elif y in merged.columns:
                merged.rename(columns={y: col}, inplace=True)

        for _c in ("industry", "MarketCap"):
            _consolidate(_c)

    # Select and reorder columns
    cols_order = [
        "symbol",
        "industry", 
        "MarketCap",
        "EBIT",
        "EnterpriseValue",
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

    # Sort by rankingScore (WACC-anchored via excessReturn), not growthGate —
    # see module docstring on why roiic - wacc isn't a valid ranking metric.
    merged = merged.sort_values("rankingScore", ascending=True)

    output_path = args.output or _dated("top50_overview.csv")
    merged.to_csv(output_path, sep=";", index=False)
    print(f"✓ Saved summary → {output_path.resolve()}")


if __name__ == "__main__":
    main() 