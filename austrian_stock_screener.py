#!/usr/bin/env python3
"""A refactored stock screener based on the logic from the
`AustrianStockScreener10Q.ipynb` notebook.

High-level workflow
-------------------
1. Scrape a list of tickers (Russell-1000 constituents by default).
2. For every ticker collect the most recent quarterly balance-sheet and income-statement
   plus market-cap and industry information using the `yahooquery` package.
3. Compute derived metrics (NOPAT, ROIC, debt, preferred equity, net-worth,
   Faustmann ratio).
4. Persist the combined dataframe as CSV.

Run `python austrian_stock_screener.py --help` for usage instructions.
"""
from __future__ import annotations

import argparse
import pathlib
import time
import pickle
from typing import List, Tuple

import bs4 as bs  # BeautifulSoup4
import pandas as pd
import requests
from yahooquery import Ticker

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

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def fetch_russell_1000_tickers(index_url: str = RUSSELL_1000_WIKI) -> List[str]:
    """Scrape the Russell-1000 index page and return a cleaned, sorted list of tickers."""
    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()

    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable"})
    if table is None:
        raise RuntimeError("Unable to locate the Russell-1000 constituents table.")

    tickers: List[str] = []
    for row in table.find_all("tr")[1:]:  # skip header row
        ticker_cell = row.find_all("td")[1]
        ticker = ticker_cell.text.strip()
        # Yahoo Finance expects dashes instead of dots for certain tickers (e.g. BRK.B)
        tickers.append(ticker.replace(".", "-"))

    tickers.sort()
    return tickers


def _compute_financial_metrics(balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame, details: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Merge all fragments and compute the screener metrics for a single symbol."""
    # ------------------------------------------------------------------
    # Merge pieces – resembles the logic in the original notebook.
    # ------------------------------------------------------------------
    merged = pd.merge(balance_sheet, income_stmt, on=["symbol"], how="inner", suffixes=("_bs", "_is"))
    if merged.empty:
        return merged  # return empty -> will be skipped by caller

    market_cap = details[["marketCap"]].rename(columns={"marketCap": "MarketCap"})
    market_cap.index.name = "symbol"
    industry = profile[["industry"]]

    merged = pd.merge(merged, market_cap, left_on="symbol", right_index=True, how="left")
    merged = pd.merge(merged, industry, left_on="symbol", right_index=True, how="left")

    # Normalize date column
    if "asOfDate" not in merged.columns:
        for alt in ("asOfDate_bs", "asOfDate_x", "asOfDate_is", "asOfDate_y"):
            if alt in merged.columns:
                merged["asOfDate"] = merged[alt]
                break

    # Guarantee "symbol" is a proper column
    if "symbol" not in merged.columns:
        merged = merged.reset_index().rename(columns={"index": "symbol"})

    # ------------------------------------------------------------------
    # Derived columns – wrapped in try/except to avoid crashing on missing fields.
    # ------------------------------------------------------------------
    try:
        merged["nopat"] = merged["EBIT"] * (1 - merged["TaxRateForCalcs"])
        merged["roic"] = merged["nopat"] / merged["InvestedCapital"]
    except KeyError:
        # Required columns unavailable – return empty to mark failure
        return pd.DataFrame()

    # Row-wise debt aggregation
    debt_cols = [
        "CurrentDeferredLiabilities",
        "LongTermDebtAndCapitalLeaseObligation",
        "NonCurrentDeferredLiabilities",
        "OtherNonCurrentLiabilities",
    ]
    for col in debt_cols:
        if col not in merged.columns:
            merged[col] = 0
    merged["totalDebt"] = merged[debt_cols].fillna(0).sum(axis=1)

    merged["preferredequity"] = (
        merged.get("CapitalStock", 0) - merged.get("CommonStock", 0)
    )

    merged["networth"] = (
        merged.get("InvestedCapital", 0)
        + merged.get("CashAndCashEquivalents", 0)
        - merged["totalDebt"]
        - merged["preferredequity"]
    )

    merged["faustmannRatio"] = merged["MarketCap"] / merged["networth"]

    # Keep only the relevant columns (mirrors the notebook's final selection)
    cols_to_keep = [
        "symbol", "asOfDate", "EBIT", "InvestedCapital", "roic", "MarketCap",
        "CashAndCashEquivalents", "totalDebt", "preferredequity", "faustmannRatio", "industry"
    ]
    # Some columns may be missing if upstream keys failed – drop those silently.
    existing_cols = [c for c in cols_to_keep if c in merged.columns]
    return merged[existing_cols]


# --------------------------------------------------------------------------------------
# Rate-limit handling utilities
# --------------------------------------------------------------------------------------

import requests

BASE_DELAY_SEC = DEFAULT_RATE_LIMIT_SEC  # minimum pacing
MAX_BACKOFF_SEC = 120                    # 2-minute ceiling


class RateLimitExceeded(Exception):
    """Raised when repeated 429 responses exhaust the allowed back-off budget."""

    def __init__(self, message: str, partial_df: pd.DataFrame | None = None):
        super().__init__(message)
        self.partial_df = partial_df


def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristically decide whether *exc* is an HTTP-429 / rate-limit signal."""
    if isinstance(exc, requests.exceptions.HTTPError):
        try:
            return exc.response is not None and exc.response.status_code == 429
        except AttributeError:  # response may be mocked / missing
            return False

    # Fall-back to string inspection (yahooquery wraps errors in generic Exception)
    return "429" in str(exc)


def fetch_with_backoff(callable_fn, *, desc: str, delay_ref: list[float]):
    """Execute *callable_fn* with exponential back-off on HTTP-429 errors.

    Parameters
    ----------
    callable_fn : Callable[[], Any]
        Zero-argument function performing the network request.
    desc : str
        Human-readable description for logging (e.g. "AAPL balance-sheet").
    delay_ref : list[float]
        Single-element list holding the current delay (mutable so callers share
        state).
    """
    while True:
        try:
            return callable_fn()
        except Exception as exc:  # noqa: BLE001 – blanket catch to inspect 429
            if not _is_rate_limit_error(exc):
                # Propagate non-rate-limit failures unchanged
                raise

            delay = delay_ref[0]
            if delay >= MAX_BACKOFF_SEC:
                raise RateLimitExceeded(
                    f"Hit max back-off while retrieving {desc}") from exc

            print(
                f"  · 429 rate-limit on {desc}. Sleeping {delay} s (will double).",
                flush=True,
            )
            time.sleep(delay)
            delay_ref[0] = min(delay * 2, MAX_BACKOFF_SEC)
            # Loop and retry


def process_universe(
    tickers: List[str],
    *,
    max_count: int,
    delay_ref: list[float],
    batch_size: int,
    processed: set[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Iterate over *tickers* and aggregate the screener metrics.

    Returns
    -------
    result_df : DataFrame
        Combined metrics for all tickers that were processed successfully.
    retry_list : list[str]
        Symbols that failed due to data availability issues and could be retried.
    """
    df_accum = pd.DataFrame()
    retry: List[str] = []

    subset = tickers[:max_count]
    total = len(subset)

    for batch_start in range(0, total, batch_size):
        batch = [s for s in subset[batch_start : batch_start + batch_size] if s not in processed]
        if not batch:
            continue
        batch_label = f"{batch_start + 1}-{batch_start + len(batch)}"

        print(f"[Batch {batch_label}/{total}] Processing {len(batch)} symbols…", flush=True)

        initial_delay = delay_ref[0]
        time.sleep(delay_ref[0])  # respect current pacing

        try:
            ticker = Ticker(batch, asynchronous=False)

            # --------------------------------------------------------------
            # Bulk data retrieval with back-off protection
            # --------------------------------------------------------------
            bs_q_all = fetch_with_backoff(
                lambda: ticker.balance_sheet(frequency="q"),
                desc=f"batch balance-sheet {batch_label}",
                delay_ref=delay_ref,
            )
            bs_q_all = (
                bs_q_all[bs_q_all["periodType"] == "3M"]
                .sort_values("asOfDate")
                .groupby("symbol")
                .tail(1)
                .reset_index()
            )

            inc_q_all = fetch_with_backoff(
                lambda: ticker.income_statement(frequency="q"),
                desc=f"batch income-statement {batch_label}",
                delay_ref=delay_ref,
            )
            inc_q_all = (
                inc_q_all[inc_q_all["periodType"] == "TTM"]
                .sort_values("asOfDate")
                .groupby("symbol")
                .tail(1)
                .reset_index()
            )

            details_dict = fetch_with_backoff(
                lambda: ticker.summary_detail,
                desc=f"batch summary_detail {batch_label}",
                delay_ref=delay_ref,
            )
            profile_dict = fetch_with_backoff(
                lambda: ticker.summary_profile,
                desc=f"batch summary_profile {batch_label}",
                delay_ref=delay_ref,
            )

            details_df = pd.DataFrame.from_dict(details_dict).T
            profile_df = pd.DataFrame.from_dict(profile_dict).T

        except RateLimitExceeded as rl_exc:
            raise RateLimitExceeded(str(rl_exc), partial_df=df_accum) from rl_exc
        except Exception as exc:
            print(f"  · batch data error: {exc}")
            retry.extend(batch)
            continue

        # --------------------------------------------------------------
        # Per-symbol metric computation within the batch
        # --------------------------------------------------------------
        for symbol in batch:
            processed.add(symbol)  # mark so retry pass won't repeat

            bs_row = bs_q_all[bs_q_all["symbol"] == symbol]
            inc_row = inc_q_all[inc_q_all["symbol"] == symbol]

            if bs_row.empty or inc_row.empty:
                retry.append(symbol)
                continue

            details_row = details_df.loc[[symbol]] if symbol in details_df.index else pd.DataFrame()
            profile_row = profile_df.loc[[symbol]] if symbol in profile_df.index else pd.DataFrame()

            try:
                metrics_df = _compute_financial_metrics(bs_row, inc_row, details_row, profile_row)
            except Exception as m_exc:
                print(f"    · metric error {symbol}: {m_exc}")
                retry.append(symbol)
                continue

            if metrics_df.empty:
                retry.append(symbol)
                continue

            df_accum = pd.concat([df_accum, metrics_df])

        # ------------------------------------------
        # Back-trace delay if it had increased
        # ------------------------------------------
        if delay_ref[0] > initial_delay:
            delay_ref[0] = max(BASE_DELAY_SEC, delay_ref[0] / 4)

    return df_accum, retry


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
        default=100,
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
    # Finalize dataframe: rankings and sanitized ratio
    # --------------------------------------------------------------
    def _add_rankings(frame: pd.DataFrame) -> pd.DataFrame:
        tmp = frame.copy()
        # Sanitize faustmannRatio (negative → NaN)
        if "faustmannRatio" in tmp.columns:
            tmp["sanitizedFaustmannRatio"] = tmp["faustmannRatio"].where(tmp["faustmannRatio"] >= 0)
        else:
            tmp["sanitizedFaustmannRatio"] = pd.NA

        # ROIC rank high→low (1 = best)
        if "roic" in tmp.columns:
            tmp["roicRank"] = tmp["roic"].rank(method="min", ascending=False)
        else:
            tmp["roicRank"] = pd.NA

        # Faustmann rank low→high (1 = best)
        tmp["faustmannRank"] = tmp["sanitizedFaustmannRatio"].rank(method="min", ascending=True)

        # Sum ranks (skip rows with NaNs)
        tmp["sumRanks"] = tmp[["roicRank", "faustmannRank"]].sum(axis=1, min_count=2)
        return tmp

    if existing_df is not None:
        combined = pd.concat([existing_df, df], ignore_index=True)
        if "symbol" in combined.columns:
            combined = combined.drop_duplicates(subset="symbol", keep="first")
        final_df = _add_rankings(combined)
        final_df.to_csv(args.output, sep=";", index=False)
    else:
        final_df = _add_rankings(df)
        final_df.to_csv(args.output, sep=";", index=False)
    print(f"Saved results → {args.output.resolve()}")


if __name__ == "__main__":
    main() 