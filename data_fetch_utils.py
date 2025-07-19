#!/usr/bin/env python3
"""Utility helpers for network-fetching Yahoo/Wikipedia data with polite rate-limit handling.

This module centralises:
1. Scraping Russell-1000 tickers from Wikipedia.
2. Generic `fetch_with_backoff` wrapper that retries HTTP-429 responses
   with exponential back-off while sharing a mutable delay reference.
3. Convenience exception `RateLimitExceeded` to abort after a back-off ceiling.

Other scripts can simply `from data_fetch_utils import fetch_russell_1000_tickers, fetch_with_backoff, RateLimitExceeded`.
"""
from __future__ import annotations

import time
from typing import Callable, Any, List

import bs4 as bs  # BeautifulSoup4
import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Default pause between requests when no rate-limit pressure is detected.
DEFAULT_RATE_LIMIT_SEC: float = 0.5

# Base delay reference value shared across requests within a run.
BASE_DELAY_SEC: float = DEFAULT_RATE_LIMIT_SEC

# Hard cap for exponential back-off (seconds).
MAX_BACKOFF_SEC: int = 120  # 2 minutes

# Wikipedia source of Russell-1000 constituents.
RUSSELL_1000_WIKI = "https://en.wikipedia.org/wiki/Russell_1000_Index"

# --------------------------------------------------------------------------------------
# Scraper helpers
# --------------------------------------------------------------------------------------

def fetch_russell_1000_tickers(index_url: str = RUSSELL_1000_WIKI) -> List[str]:
    """Return a sorted list of ticker symbols from the Russell-1000 wiki table."""
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

# --------------------------------------------------------------------------------------
# Rate-limit handling utilities
# --------------------------------------------------------------------------------------

class RateLimitExceeded(Exception):
    """Raised when repeated 429 responses exhaust the allowed back-off budget."""

    def __init__(self, message: str, partial_df: pd.DataFrame | None = None):
        super().__init__(message)
        self.partial_df = partial_df


def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristically decide whether *exc* indicates an HTTP-429 / rate-limit."""
    if isinstance(exc, requests.exceptions.HTTPError):
        try:
            return exc.response is not None and exc.response.status_code == 429
        except AttributeError:
            return False
    # yahooquery wraps errors in `Exception` so fall back to string search.
    return "429" in str(exc)


def fetch_with_backoff(callable_fn: Callable[[], Any], *, desc: str, delay_ref: List[float]):
    """Execute *callable_fn* with exponential back-off on HTTP-429 errors.

    Parameters
    ----------
    callable_fn : Callable[[], Any]
        Zero-argument function performing the network request.
    desc : str
        Human-readable description for logging (e.g. "AAPL balance-sheet").
    delay_ref : list[float]
        Mutable single-element list holding the current delay between requests.
    """
    while True:
        try:
            return callable_fn()
        except Exception as exc:  # noqa: BLE001 – examine any raised error
            if not _is_rate_limit_error(exc):
                raise  # propagate non-rate-limit failures unchanged

            delay = delay_ref[0]
            if delay >= MAX_BACKOFF_SEC:
                raise RateLimitExceeded(f"Hit max back-off while retrieving {desc}") from exc

            print(f"  · 429 rate-limit on {desc}. Sleeping {delay} s (will double).", flush=True)
            time.sleep(delay)
            delay_ref[0] = min(delay * 2, MAX_BACKOFF_SEC)
            # Retry after sleeping 