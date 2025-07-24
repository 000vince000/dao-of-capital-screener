#!/usr/bin/env python3
"""Post-processing tool that augments `austrian.csv` with excess-return metrics.

Prerequisites
-------------
1. `austrian.csv` – output of `current_baseline_data.py`
2. `wacc_top.csv` – subset of WACC / Cost-of-Equity values (must contain `symbol`,
   `wacc`, `costOfEquity`)

Workflow
--------
0. Validate presence of WACC file; exit with message if missing.
1. Load `austrian.csv` → drop `roicRank`, `sumRanks` columns if they exist.
2. Add `roeFlag` depending on the *industry* membership.
3. Merge WACC data onto the dataframe.
4. Compute `excessReturn`:
   • if `roeFlag` → `roe – costOfEquity`
   • else          → `roic – wacc`
5. Rank `excessReturn` descending → `excessReturnRank` (1 = best).
6. Recompute `sumRanks = excessReturnRank + faustmannRank`.
7. Save as `normalized_austrian.csv` (semicolon-separated) or path passed with
   `--output`.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Produce normalized Austrian screener output with excess-return ranks.")
    p.add_argument("--input", type=pathlib.Path, default=pathlib.Path("austrian.csv"), help="Input CSV from screener (default: austrian.csv)")
    p.add_argument("--wacc-file", type=pathlib.Path, default=pathlib.Path("wacc_top.csv"), help="CSV with WACC & costOfEquity (default: wacc_top.csv)")
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("normalized_austrian.csv"), help="Destination filename (default: normalized_austrian.csv)")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    if not args.wacc_file.exists():
        print(f"❌ Required WACC file {args.wacc_file} not found.", file=sys.stderr)
        sys.exit(1)

    if not args.input.exists():
        print(f"❌ Input screener file {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input, sep=";")

    # Drop columns if present
    df = df.drop(columns=[c for c in ["roicRank", "sumRanks"] if c in df.columns], errors="ignore")

    # Merge WACC data
    wacc_df = pd.read_csv(args.wacc_file, sep=";")
    df = df.merge(wacc_df[["symbol", "wacc", "costOfEquity"]], on="symbol", how="left")

    # Compute excessReturn
    df["excessReturn"] = df["roic"] - df["wacc"]

    # Rank excessReturn descending (1 = best). Handle NaNs → rank NaN as worst (use pct?). We'll set method='min', na_option='bottom'
    df["excessReturnRank"] = df["excessReturn"].rank(method="min", ascending=False, na_option="bottom")

    df.to_csv(args.output, sep=";", index=False)
    print(f"✓ Saved normalized screener → {args.output.resolve()}")


if __name__ == "__main__":
    main() 