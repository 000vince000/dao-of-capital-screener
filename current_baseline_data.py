#!/usr/bin/env python3
"""A refactored stock screener based on the logic from the
`AustrianStockScreener10Q.ipynb` notebook.

High-level workflow
-------------------
1. Scrape a list of tickers (Russell-1000 constituents by default).
2. For every ticker collect the most recent quarterly balance-sheet and income-statement
   plus market-cap and industry information using the `yahooquery` package.
3. Compute derived metrics (NOPAT, ROIC, debt, preferred equity, net-worth).
4. Persist the combined dataframe as CSV.

Run `python current_baseline_data.py --help` for usage instructions.
"""
from __future__ import annotations

import argparse
import pathlib
import time
import pickle
from typing import List, Tuple

import pandas as pd
from yahooquery import Ticker

# Centralised network helpers
from data_fetch_utils import (
    fetch_russell_1000_tickers,
    fetch_with_backoff,
    RateLimitExceeded,
    BASE_DELAY_SEC,
)

# --------------------------------------------------------------------------------------
# Constants & configuration
# --------------------------------------------------------------------------------------

# Wikipedia page that holds the Russell-1000 index constituents. This was the source in
# the original notebook. Feel free to replace with any other wiki or CSV that contains
# a column of tickers.
RUSSELL_1000_WIKI = "https://en.wikipedia.org/wiki/Russell_1000_Index"

# Sleep duration (seconds) between successive Yahoo queries to stay friendly to the API.
DEFAULT_RATE_LIMIT_SEC = 0.5

# Maximum number of tickers to process in a single run. The notebook capped this at
# 1,000 which equals the full Russell-1000 universe. You can override this via CLI.
DEFAULT_MAX_COUNT = 1000

DEFAULT_INDUSTRIES_RELYING_ON_ROE: List[str] = [
    "Asset Management",
    "Insurance - Diversified",
    "Insurance - Life",
    "Insurance - Property & Casualty",
    "Insurance - Specialty",
    "Insurance Brokers",
    "Credit Services",
    "REIT - Diversified",
    "REIT - Residential",
    "REIT - Office",
    "REIT - Retail",
    "REIT - Industrial",
    "REIT - Specialty",
    "REIT - Mortgage",
    "REIT - Healthcare Facilities",
    "REIT - Hotel & Motel",
    "Utilities - Regulated Electric",
    "Utilities - Regulated Gas",
    "Utilities - Regulated Water",
    "Utilities - Diversified",
    "Oil & Gas Midstream",
]
# --------------------------------------------------------------------------------------
# Helper wrappers (scraping utilities moved to `data_fetch_utils`)
# --------------------------------------------------------------------------------------


def _compute_financial_metrics(balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame, cash_flow: pd.DataFrame, details: pd.DataFrame, profile: pd.DataFrame, valuation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Orchestrate sub-steps to compute final metrics for a single symbol."""

    # ------------------------------------------------------------------
    # Helper pipeline steps (defined inline for locality)
    # ------------------------------------------------------------------

    def _merge_core() -> pd.DataFrame:
        core = pd.merge(balance_sheet, income_stmt, on=["symbol"], how="inner", suffixes=("_bs", "_is"))
        return core

    def _attach_op_cf(frame: pd.DataFrame) -> None:
        if not cash_flow.empty and "OperatingCashFlow" in cash_flow.columns:
            frame["opCashFlow"] = cash_flow.loc[cash_flow["symbol"].isin(frame["symbol"]), "OperatingCashFlow"].values
        else:
            frame["opCashFlow"] = pd.NA

    def _attach_market_industry(frame: pd.DataFrame) -> None:
        if not details.empty:
            mc = details[["marketCap"]].rename(columns={"marketCap": "MarketCap"})
            mc.index.name = "symbol"
            frame.update(mc, overwrite=False)
        if not profile.empty:
            ind = profile[["industry"]]
            frame.update(ind, overwrite=False)

        # Normalize date column
        if "asOfDate" not in frame.columns:
            for alt in ("asOfDate_bs", "asOfDate_x", "asOfDate_is", "asOfDate_y"):
                if alt in frame.columns:
                    frame["asOfDate"] = frame[alt]
                    break

    def _compute_core_metrics(frame: pd.DataFrame) -> None:
        frame["nopat"] = frame["EBIT"] * (1 - frame["TaxRateForCalcs"])
        frame["roic"] = frame["nopat"] / frame["InvestedCapital"]

        debt_cols = [
            "CurrentDeferredLiabilities",
            "LongTermDebtAndCapitalLeaseObligation",
            "NonCurrentDeferredLiabilities",
            "OtherNonCurrentLiabilities",
        ]
        for col in debt_cols:
            if col not in frame.columns:
                frame[col] = 0
        frame["totalDebt"] = frame[debt_cols].fillna(0).sum(axis=1)

        frame["preferredequity"] = frame.get("CapitalStock", 0) - frame.get("CommonStock", 0)

        # Net worth and Faustmann ratio removed per latest spec.

    def _attach_enterprise_value(frame: pd.DataFrame) -> None:
        ev_val = pd.NA
        if valuation is not None and not valuation.empty and "EnterpriseValue" in valuation.columns:
            candidate = valuation["EnterpriseValue"].iloc[0]
            if pd.notna(candidate):
                ev_val = candidate
        if pd.isna(ev_val):
            ev_val = frame["MarketCap"] + frame["totalDebt"] + frame["preferredequity"] - frame.get("CashAndCashEquivalents", 0)
        frame["EnterpriseValue"] = ev_val
        frame["opCashFlowYield"] = frame["opCashFlow"] / frame["EnterpriseValue"]

    def _compute_roe(frame: pd.DataFrame) -> None:
        ni_col = next((c for c in ["NetIncome", "NetIncomeLoss", "netIncome"] if c in frame.columns), None)
        eq_col = next((c for c in ["TotalShareholderEquity", "StockholdersEquity", "totalStockholderEquity"] if c in frame.columns), None)
        if ni_col and eq_col:
            frame["roe"] = frame[ni_col] / frame[eq_col].replace(0, pd.NA)
            frame.rename(columns={ni_col: "NetIncome", eq_col: "TotalShareholderEquity"}, inplace=True)
        else:
            frame["roe"] = pd.NA

    def _validate_and_trim(frame: pd.DataFrame) -> pd.DataFrame:
        cols_keep = [
            "symbol", "asOfDate", "EBIT", "InvestedCapital", "roic", "MarketCap",
            "CashAndCashEquivalents", "totalDebt", "preferredequity",
            "opCashFlow", "opCashFlowYield", "industry", "EnterpriseValue",
            "NetIncome", "TotalShareholderEquity", "roe",
        ]
        existing = [c for c in cols_keep if c in frame.columns]
        return frame[existing]

    # ------------------------------------------------------------------
    # Orchestrate
    # ------------------------------------------------------------------

    merged = _merge_core()
    if merged.empty:
        return merged

    try:
        _attach_op_cf(merged)
        _attach_market_industry(merged)
        _compute_core_metrics(merged)
        _attach_enterprise_value(merged)
        _compute_roe(merged)
    except KeyError:
        return pd.DataFrame()

    return _validate_and_trim(merged)


# Rate-limit utilities are now imported from `data_fetch_utils`.


def process_universe(
    tickers: List[str],
    *,
    max_count: int,
    delay_ref: list[float],
    batch_size: int,
    processed: set[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Iterate over *tickers* in batches and return (metrics_df, retry_list)."""

    # ------------------------------------------------------------------
    # Nested helpers
    # ------------------------------------------------------------------

    def _fetch_batch_data(tck: Ticker, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Download and pre-filter all Yahoo slices for *tck* (a batch Ticker)."""

        bs_q = fetch_with_backoff(
            lambda: tck.balance_sheet(frequency="q"),
            desc=f"batch balance-sheet {label}",
            delay_ref=delay_ref,
        )
        bs_q = (
            bs_q[bs_q["periodType"] == "3M"].sort_values("asOfDate").groupby("symbol").tail(1).reset_index()
        )

        inc_q = fetch_with_backoff(
            lambda: tck.income_statement(frequency="q"),
            desc=f"batch income-statement {label}",
            delay_ref=delay_ref,
        )
        inc_q = (
            inc_q[inc_q["periodType"] == "TTM"].sort_values("asOfDate").groupby("symbol").tail(1).reset_index()
        )

        cf_q = fetch_with_backoff(
            lambda: tck.cash_flow(frequency="q"),
            desc=f"batch cash-flow {label}",
            delay_ref=delay_ref,
        )
        cf_q = (
            cf_q[cf_q["periodType"] == "TTM"].sort_values("asOfDate").groupby("symbol").tail(1).reset_index()
        )

        details_dict = fetch_with_backoff(
            lambda: tck.summary_detail,
            desc=f"batch summary_detail {label}",
            delay_ref=delay_ref,
        )
        profile_dict = fetch_with_backoff(
            lambda: tck.summary_profile,
            desc=f"batch summary_profile {label}",
            delay_ref=delay_ref,
        )

        valuation_all = fetch_with_backoff(
            lambda: tck.valuation_measures,
            desc=f"batch valuation_measures {label}",
            delay_ref=delay_ref,
        )
        valuation_all = valuation_all.sort_values("asOfDate").groupby("symbol").tail(1).reset_index()

        details_df = pd.DataFrame.from_dict(details_dict).T
        profile_df = pd.DataFrame.from_dict(profile_dict).T

        return bs_q, inc_q, cf_q, details_df, profile_df, valuation_all

    def _process_symbols(batch_syms: list[str], dfs: tuple[pd.DataFrame, ...]) -> tuple[pd.DataFrame, list[str]]:
        """Build metrics for each symbol in *batch_syms* using pre-fetched DataFrames."""
        bs_q, inc_q, cf_q, details_df, profile_df, valuation_all = dfs
        batch_out = pd.DataFrame()
        batch_retry: list[str] = []

        for sym in batch_syms:
            processed.add(sym)

            # Early filter: skip if industry relies on ROE
            if sym in profile_df.index:
                industry = profile_df.loc[sym].get("industry", "")
                if industry in DEFAULT_INDUSTRIES_RELYING_ON_ROE:
                    print(f"    · skipping {sym} (ROE-relying industry: {industry})")
                    continue

            bs_row = bs_q[bs_q["symbol"] == sym]
            inc_row = inc_q[inc_q["symbol"] == sym]
            cf_row = cf_q[cf_q["symbol"] == sym]

            if bs_row.empty or inc_row.empty:
                batch_retry.append(sym)
                continue

            det_row = details_df.loc[[sym]] if sym in details_df.index else pd.DataFrame()
            prof_row = profile_df.loc[[sym]] if sym in profile_df.index else pd.DataFrame()
            val_row = valuation_all[valuation_all["symbol"] == sym]

            try:
                metrics_df = _compute_financial_metrics(bs_row, inc_row, cf_row, det_row, prof_row, val_row)
            except Exception as err:
                print(f"    · metric error {sym}: {err}")
                batch_retry.append(sym)
                continue

            if metrics_df.empty:
                batch_retry.append(sym)
                continue

            batch_out = pd.concat([batch_out, metrics_df])

        return batch_out, batch_retry

    # ------------------------------------------------------------------
    # Main batching loop
    # ------------------------------------------------------------------

    overall_df = pd.DataFrame()
    overall_retry: list[str] = []

    subset = tickers[:max_count]
    total = len(subset)

    for batch_start in range(0, total, batch_size):
        batch = [s for s in subset[batch_start : batch_start + batch_size] if s not in processed]
        if not batch:
            continue

        label = f"{batch_start + 1}-{batch_start + len(batch)}"
        print(f"[Batch {label}/{total}] Processing {len(batch)} symbols…", flush=True)

        initial_delay = delay_ref[0]
        time.sleep(delay_ref[0])

        try:
            t = Ticker(batch, asynchronous=False)
            batch_dfs = _fetch_batch_data(t, label)
        except RateLimitExceeded as exc:
            raise RateLimitExceeded(str(exc), partial_df=overall_df) from exc
        except Exception as exc:
            print(f"  · batch data error: {exc}")
            overall_retry.extend(batch)
            continue

        df_batch, retry_batch = _process_symbols(batch, batch_dfs)
        overall_df = pd.concat([overall_df, df_batch])
        overall_retry.extend(retry_batch)

        # ---------- adjust delay back down if possible ----------
        if delay_ref[0] > initial_delay:
            delay_ref[0] = max(BASE_DELAY_SEC, delay_ref[0] / 4)

    return overall_df, overall_retry


# --------------------------------------------------------------------------------------
# CLI entry-point
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Austrian stock screener (Russell-1000 adaptation).")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("austrian.csv"),
        help="Destination CSV filename (default: ./austrian.csv)",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=DEFAULT_MAX_COUNT,
        help="Maximum number of tickers to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT_SEC,
        help="Initial seconds to wait between Yahoo queries (default: 2)",
    )
    parser.add_argument(
        "--save-ticker-cache",
        action="store_true",
        help="Cache the scraped ticker list to disk (russell1000tickers.pickle).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of tickers to process in each batch (default: 100)",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Load / fetch ticker universe
    # --------------------------------------------------------------
    if pathlib.Path("russell1000tickers.pickle").exists():
        tickers: List[str] = pickle.load(open("russell1000tickers.pickle", "rb"))
        print(f"Loaded {len(tickers)} cached tickers.")
    else:
        tickers = fetch_russell_1000_tickers()
        if args.save_ticker_cache:
            pickle.dump(tickers, open("russell1000tickers.pickle", "wb"))
        print(f"Fetched {len(tickers)} tickers from Wikipedia.")

    # --------------------------------------------------------------
    # Skip already processed symbols if output/temp file exists
    # --------------------------------------------------------------
    existing_df = None
    processed_symbols: set[str] = set()
    if args.output.exists():
        try:
            existing_df = pd.read_csv(args.output, sep=";")
            # Drop any index-like unnamed columns from older runs
            existing_df = existing_df.loc[:, ~existing_df.columns.str.contains("^Unnamed")]
            if "symbol" in existing_df.columns:
                processed_symbols = set(existing_df["symbol"].dropna().unique())
                print(
                    f"Detected {len(processed_symbols)} previously processed tickers "
                    f"in {args.output.name}. They will be skipped in real-time."
                )
        except Exception as exc:
            print(f"⚠️  Could not read existing output file: {exc} – proceeding as if empty.")
            processed_symbols = set()

    # --------------------------------------------------------------
    # Main processing loop
    # --------------------------------------------------------------
    delay_ref = [float(args.rate_limit)]  # mutable single-value holder

    try:
        df, retry = process_universe(
            tickers,
            max_count=args.max_count,
            delay_ref=delay_ref,
            batch_size=args.batch_size,
            processed=processed_symbols,
        )
    except RateLimitExceeded as e:
        partial = e.partial_df if e.partial_df is not None else pd.DataFrame()
        if not partial.empty:
            partial.to_csv("austrian_partial.csv", sep=";")
            print("⚠️  Rate-limit wall hit. Partial results saved → austrian_partial.csv")
        else:
            print("⚠️  Rate-limit wall hit before any data could be saved.")
        return  # terminate early

    # Retry loop (single pass)
    if retry:
        print("\n------------- RETRY PASS -------------")
        try:
            df_retry, retry2 = process_universe(
                retry,
                max_count=len(retry),
                delay_ref=delay_ref,
                batch_size=args.batch_size,
                processed=processed_symbols,
            )
        except RateLimitExceeded as e:
            combined = pd.concat([df, e.partial_df]) if e.partial_df is not None else df
            combined.to_csv("austrian_partial.csv", sep=";")
            print("⚠️  Rate-limit wall hit during retry pass. Partial results saved → austrian_partial.csv")
            return
        df = pd.concat([df, df_retry])
        if retry2:
            print(f"⚠️  Still failed for {len(retry2)} tickers: {', '.join(retry2[:10])}{' …' if len(retry2)>10 else ''}")

    # --------------------------------------------------------------
    # Finalize and save dataframe
    # --------------------------------------------------------------
    if existing_df is not None:
        combined = pd.concat([existing_df, df], ignore_index=True)
        if "symbol" in combined.columns:
            combined = combined.drop_duplicates(subset="symbol", keep="first")
        combined.to_csv(args.output, sep=";", index=False)
    else:
        df.to_csv(args.output, sep=";", index=False)
    print(f"Saved results → {args.output.resolve()}")


if __name__ == "__main__":
    main() 