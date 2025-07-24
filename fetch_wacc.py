#!/usr/bin/env python3
"""Fetch Weighted Average Cost of Capital (WACC) figures from valueinvesting.io.

For each ticker symbol provided (via --tickers or read from an input CSV), this
script requests the URL
    https://valueinvesting.io/<TICKER>/valuation/wacc
parses the returned HTML, and extracts the numeric values for:
    • WACC (selected value)
    • Cost of Equity
    • Cost of Debt

The extracted metrics are written to a semicolon-separated CSV so they can be
joined with other datasets (e.g., the Austrian stock screener output).

The CLI and general control-flow mirror *compute_roic_slope.py* for a
consistent developer experience.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------------------
# HTML parsing helpers
# --------------------------------------------------------------------------------------

_PERCENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")


def _extract_percent(text: str) -> Optional[float]:
    """Return a float ratio (e.g. 0.081) from a string like "8.1%" or None if absent."""
    match = _PERCENT_RE.search(text)
    if match:
        return float(match.group(1)) / 100.0
    return None


def _find_percent_regex(page_text: str, label: str) -> Optional[float]:
    """Search *page_text* for `<label> ... <number>%` pattern and return the value."""
    pattern = re.compile(rf"{label}[^0-9]+([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)
    m = pattern.search(page_text)
    if m:
        return float(m.group(1)) / 100.0
    return None


def _parse_from_table(soup: BeautifulSoup, row_label: str) -> Optional[float]:
    """Attempt to extract a percentage value from a table row whose first cell matches *row_label*."""
    lower_label = row_label.lower()
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        first_cell_text = cells[0].get_text(strip=True).lower()
        if lower_label in first_cell_text:
            # Prefer the 2nd cell, else any cell containing a % sign.
            for cell in cells[1:]:
                val = _extract_percent(cell.get_text())
                if val is not None:
                    return val
    return None


def _parse_wacc(html: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (wacc, cost_of_equity, cost_of_debt) as decimal fractions or None if missing."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")

    wacc = _find_percent_regex(text, "WACC")
    equity = _find_percent_regex(text, "Cost of Equity")
    debt = _find_percent_regex(text, "Cost of Debt")

    # Fallback to table-based parsing if regex failed.
    if wacc is None:
        wacc = _parse_from_table(soup, "WACC")
    if equity is None:
        equity = _parse_from_table(soup, "Cost of Equity")
    if debt is None:
        debt = _parse_from_table(soup, "Cost of Debt")

    return wacc, equity, debt

# --------------------------------------------------------------------------------------
# Networking helpers
# --------------------------------------------------------------------------------------

USER_AGENT = "Mozilla/5.0 (compatible; WACC-Scraper/1.0; +https://github.com/)"
DEFAULT_TIMEOUT = 10  # seconds
RETRY_STATUS = {429, 500, 502, 503, 504}


def _http_get(url: str, session: requests.Session, *, symbol: str = "", max_retries: int = 3) -> requests.Response:
    """GET *url* with simple retry logic on 429/5xx status codes.

    The *symbol* argument is only used for clearer logging.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            # Non-retryable error → return immediately.
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            if attempt == max_retries:
                raise
            # Quietly back-off; per-symbol retries are reported by outer loop.
            sleep_s = 1.5 * attempt
            time.sleep(sleep_s)
    # Should not reach here.
    raise RuntimeError("Exhausted retries")


# --------------------------------------------------------------------------------------
# Core fetch logic
# --------------------------------------------------------------------------------------

def fetch_wacc(ticker: str, session: requests.Session) -> tuple[dict, bool]:
    """Return (row_dict, retryable) where retryable indicates if we should retry on failure."""
    url_ticker = ticker.replace("-", ".")
    url = f"https://valueinvesting.io/{url_ticker}/valuation/wacc"
    try:
        resp = _http_get(url, session=session, symbol=ticker)
    except Exception as exc:
        print(f"  · Error fetching {ticker}: {exc}", file=sys.stderr)
        return {"symbol": ticker, "wacc": None, "costOfEquity": None, "costOfDebt": None}, True

    if resp.status_code != 200:
        print(f"  · HTTP {resp.status_code} for {ticker} ({url})", file=sys.stderr)
        return {"symbol": ticker, "wacc": None, "costOfEquity": None, "costOfDebt": None}, True

    # If redirect landed at unexpected location treat as non-retryable failure
    if '/valuation/wacc' not in resp.url:
        return {"symbol": ticker, "wacc": None, "costOfEquity": None, "costOfDebt": None}, False

    wacc, equity, debt = _parse_wacc(resp.text)
    retryable = wacc is None  # parse failure shouldn't retry
    return {"symbol": ticker, "wacc": wacc, "costOfEquity": equity, "costOfDebt": debt}, retryable


# --------------------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch WACC values for given tickers from valueinvesting.io.")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("current_baseline_data.csv"),
        help="Input CSV containing at least a 'symbol' column (default: current_baseline_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("wacc_top.csv"),
        help="Destination CSV filename (default: wacc_top.csv)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to process; if omitted use all symbols in --input.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel worker threads (default: 5)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Minimum seconds between successive outbound requests (default: 1.0)",
    )
    parser.add_argument(
        "--flush-size",
        type=int,
        default=10,
        help="Flush accumulated successful rows to CSV after this many rows (default: 10)",
    )
    parser.add_argument(
        "--failed-output",
        type=pathlib.Path,
        default=pathlib.Path("wacc_failed.csv"),
        help="CSV file to write tickers where page loaded but WACC not found (default: wacc_failed.csv)",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Limit to first N tickers (for testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Determine ticker universe
    if args.tickers:
        tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not args.input.exists():
            print(f"Input file {args.input} not found and --tickers not provided.", file=sys.stderr)
            sys.exit(1)
        # Load only the 'symbol' column to reduce memory footprint
        try:
            df_in = pd.read_csv(args.input, sep=";", usecols=["symbol"])
        except ValueError:
            # 'symbol' not in CSV or other issue – fallback to full read for clearer error
            df_in = pd.read_csv(args.input, sep=";")
        if "symbol" not in df_in.columns:
            print("Input CSV lacks 'symbol' column.", file=sys.stderr)
            sys.exit(1)
        tickers = df_in["symbol"].dropna().unique().tolist()

    # Optional limit for testing
    if args.max_count is not None:
        tickers = tickers[: args.max_count]

    # Skip symbols already present in the output file (if exists)
    existing_df = None
    if args.output.exists():
        try:
            existing_df = pd.read_csv(args.output, sep=";")
        except Exception:
            # Fallback: sanitize file with extra columns (e.g., stray _retryable)
            try:
                tmp_df = pd.read_csv(
                    args.output,
                    sep=";",
                    header=0,
                    names=["symbol", "wacc", "costOfEquity", "costOfDebt", "_extra"],
                    engine="python",
                )
                tmp_df = tmp_df[["symbol", "wacc", "costOfEquity", "costOfDebt"]]
                tmp_df.to_csv(args.output, sep=";", index=False)
                existing_df = tmp_df
                print("✓ Sanitized malformed rows in existing output file.")
            except Exception as exc2:
                print(f"⚠️  Could not sanitize existing output file: {exc2} – will refetch all tickers.")
                existing_df = None

    # If we have existing_df, apply skip logic
    if existing_df is not None:
        done = set(existing_df["symbol"].dropna().astype(str).unique())
        # Also skip symbols previously marked as failed
        if args.failed_output.exists():
            try:
                fail_df = pd.read_csv(args.failed_output, sep=";")
                failed_set = set(fail_df["symbol"].dropna().astype(str).unique())
                done.update(failed_set)
            except Exception as exc:
                print(f"⚠️  Could not read failed-output file: {exc}")

        remaining = [t for t in tickers if t not in done]
        skipped = len(tickers) - len(remaining)
        if skipped:
            print(f"✓ {skipped} tickers already processed (success or failed); skipping fetch.")
        tickers = remaining

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # ------------------------------------------------------------------
    # Parallel fetch with global pacing
    # ------------------------------------------------------------------

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rate_interval = args.rate_limit
    max_workers = args.workers

    class _AdaptiveGate:
        """Adaptive global pacing gate.

        Starts with `interval` seconds between requests. Doubles interval when
        throttled, shrinks by 10 % after a streak of `success_threshold` good
        requests (no throttling).
        """

        def __init__(self, interval: float, *, min_int: float = 0.05, max_int: float = 5.0, success_threshold: int = 5):
            self.interval = interval
            self.min_int = min_int
            self.max_int = max_int
            self.success_threshold = success_threshold
            self._success_streak = 0
            self._last = 0.0  # monotonic time
            self._lock = threading.Lock()

        def wait_turn(self):
            import time as _time
            with self._lock:
                now = _time.monotonic()
                wait = self.interval - (now - self._last)
                if wait > 0:
                    _time.sleep(wait)
                self._last = _time.monotonic()

        def report_success(self):
            with self._lock:
                self._success_streak += 1
                if self._success_streak >= self.success_threshold and self.interval > self.min_int:
                    self.interval = max(self.interval * 0.85, self.min_int)  # speed up by 15%
                    self._success_streak = 0

        def report_throttled(self):
            with self._lock:
                self.interval = min(self.interval * 1.5, self.max_int)  # gentler back-off
                self._success_streak = 0

    gate = _AdaptiveGate(rate_interval)

    total = len(tickers)

    def _task(args_tuple):
        idx, symbol, tot = args_tuple
        gate.wait_turn()
        print(f"[{idx}/{tot}] {symbol}  (intvl {gate.interval:.2f}s)", flush=True)
        sess = requests.Session()
        sess.headers.update({"User-Agent": USER_AGENT})
        row, retryable = fetch_wacc(symbol, sess)
        if row.get("wacc") is None:
            if retryable:
                gate.report_throttled()
            else:
                gate.report_success()
        else:
            gate.report_success()
        row["_retryable"] = retryable
        return row

    max_passes = 20  # per-symbol retry passes
    remaining = tickers
    rows = []
    buffer_rows: list[dict] = []
    flush_size = args.flush_size
    header_written = args.output.exists()

    for attempt in range(1, max_passes + 1):
        if not remaining:
            break
        if attempt == 1:
            print(f"--- Pass 1/{max_passes} starting with {len(remaining)} tickers ---")
        else:
            print(f"--- Retry pass {attempt}/{max_passes} for {len(remaining)} tickers ---")

        current_total = len(remaining)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_sym = {pool.submit(_task, (idx, sym, current_total)): sym for idx, sym in enumerate(remaining, start=1)}
            current_rows = []
            for fut in as_completed(future_to_sym):
                sym = future_to_sym[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    print(f"  · Error in thread for {sym}: {exc}", file=sys.stderr)
                    row = {"symbol": sym, "wacc": None, "costOfEquity": None, "costOfDebt": None}
                current_rows.append(row)

        # Determine which tickers failed this pass
        successes = [r for r in current_rows if not pd.isna(r.get("wacc"))]
        # Distinguish retryable vs final failures
        retry_fail = [r["symbol"] for r in current_rows if pd.isna(r.get("wacc")) and r.get("_retryable", True)]
        final_fail_rows = [r for r in current_rows if pd.isna(r.get("wacc")) and not r.get("_retryable", True)]

        # ----------------------------------------------
        # Incremental flush of successful rows in batches
        # ----------------------------------------------
        # Buffer and flush per --flush-size
        if successes:
            buffer_rows.extend(successes)
            if len(buffer_rows) >= flush_size:
                df_flush = (
                    pd.DataFrame(buffer_rows)
                    .drop_duplicates(subset="symbol", keep="first")
                    [["symbol", "wacc", "costOfEquity", "costOfDebt"]]
                )
                df_flush.to_csv(
                    args.output,
                    sep=";",
                    index=False,
                    mode="a" if header_written else "w",
                    header=not header_written,
                )
                header_written = True
                buffer_rows.clear()

        rows.extend(successes)
        remaining = retry_fail
        # On failure pass increase interval modestly to be polite
        if remaining:
            gate.report_throttled()

        # write final_fail_rows to failed_output immediately
        if final_fail_rows:
            df_fail = pd.DataFrame(final_fail_rows)[["symbol"]]
            df_fail.to_csv(args.failed_output, sep=";", index=False, mode="a", header=not args.failed_output.exists())


    # Combine with previously fetched rows
    if existing_df is not None and not existing_df.empty:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="symbol", keep="first")
        out_df = combined
    else:
        out_df = pd.DataFrame(rows)

    # Final flush for any remaining buffered rows
    if buffer_rows:
        df_flush = (
            pd.DataFrame(buffer_rows)
            .drop_duplicates(subset="symbol", keep="first")
            [["symbol", "wacc", "costOfEquity", "costOfDebt"]]
        )
        df_flush.to_csv(
            args.output,
            sep=";",
            index=False,
            mode="a" if header_written else "w",
            header=not header_written,
        )
        header_written = True
        rows.extend(buffer_rows)

    # Final deduplication pass: read the file, drop any duplicate symbols, rewrite.
    try:
        df_all = pd.read_csv(args.output, sep=";")
        df_all = df_all[["symbol", "wacc", "costOfEquity", "costOfDebt"]]
        df_all = df_all.drop_duplicates(subset="symbol", keep="first")
        df_all.to_csv(args.output, sep=";", index=False)
    except Exception as exc:
        print(f"⚠️  Could not finalize deduplication: {exc}", file=sys.stderr)

    print(f"Saved WACC values → {args.output.resolve()}")

    if remaining:
        df_remaining = pd.DataFrame({"symbol": remaining})
        df_remaining.to_csv(args.failed_output, sep=";", index=False, mode="a", header=not args.failed_output.exists())
        print(f"⚠️  Failed to fetch WACC for {len(remaining)} tickers after {max_passes} retries. Logged to {args.failed_output}", file=sys.stderr)


if __name__ == "__main__":
    main() 