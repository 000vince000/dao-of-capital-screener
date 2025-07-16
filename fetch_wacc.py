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


def _http_get(url: str, session: requests.Session, max_retries: int = 3) -> requests.Response:
    """GET *url* with simple retry logic on 429/5xx status codes."""
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
            sleep_s = 1.5 * attempt
            print(f"  · Retry {attempt}/{max_retries} after error: {exc}. Sleeping {sleep_s:.1f}s…", file=sys.stderr)
            time.sleep(sleep_s)
    # Should not reach here.
    raise RuntimeError("Exhausted retries")


# --------------------------------------------------------------------------------------
# Core fetch logic
# --------------------------------------------------------------------------------------

def fetch_wacc(ticker: str, session: requests.Session) -> Dict[str, Optional[float]]:
    """Return a dict with WACC metrics for *ticker* (values may be None if unavailable)."""
    url = f"https://valueinvesting.io/{ticker}/valuation/wacc"
    try:
        resp = _http_get(url, session=session)
    except Exception as exc:
        print(f"  · Error fetching {ticker}: {exc}", file=sys.stderr)
        return {"symbol": ticker, "wacc": None, "costOfEquity": None, "costOfDebt": None}

    if resp.status_code != 200:
        print(f"  · HTTP {resp.status_code} for {ticker} ({url})", file=sys.stderr)
        return {"symbol": ticker, "wacc": None, "costOfEquity": None, "costOfDebt": None}

    wacc, equity, debt = _parse_wacc(resp.text)
    return {"symbol": ticker, "wacc": wacc, "costOfEquity": equity, "costOfDebt": debt}


# --------------------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch WACC values for given tickers from valueinvesting.io.")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("austrian.csv"),
        help="Input CSV containing at least a 'symbol' column (default: austrian.csv)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("wacc.csv"),
        help="Destination CSV filename (default: wacc.csv)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to process; if omitted use all symbols in --input.",
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
        df_in = pd.read_csv(args.input, sep=";")
        if "symbol" not in df_in.columns:
            print("Input CSV lacks 'symbol' column.", file=sys.stderr)
            sys.exit(1)
        tickers = df_in["symbol"].dropna().unique().tolist()

    # Skip symbols already present in the output file (if exists)
    existing_df = None
    if args.output.exists():
        try:
            existing_df = pd.read_csv(args.output, sep=";")
            done = set(existing_df["symbol"].dropna().astype(str).unique())
            remaining = [t for t in tickers if t not in done]
            skipped = len(tickers) - len(remaining)
            if skipped:
                print(f"✓ {skipped} tickers already present in {args.output.name}; skipping fetch.")
            tickers = remaining
        except Exception as exc:
            print(f"⚠️  Could not read existing output file: {exc} – will refetch all tickers.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows = []
    for sym in tickers:
        print(f"Processing {sym}…", flush=True)
        row = fetch_wacc(sym, session)
        rows.append(row)
        time.sleep(1)  # polite delay between requests

    # Combine with previously fetched rows
    if existing_df is not None and not existing_df.empty:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="symbol", keep="first")
        out_df = combined
    else:
        out_df = pd.DataFrame(rows)

    out_df.to_csv(args.output, sep=";", index=False)
    print(f"Saved WACC values → {args.output.resolve()}")


if __name__ == "__main__":
    main() 