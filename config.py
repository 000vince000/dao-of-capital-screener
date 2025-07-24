#!/usr/bin/env python3
"""Configuration file for the stock screener pipeline.

This file contains configurable parameters, especially the list of industries
that rely on ROE (Return on Equity) rather than ROIC (Return on Invested Capital).
These industries are filtered out during screening to focus on operating businesses.
"""

import json
import pathlib
from typing import List, Set

# Industries that rely on ROE rather than ROIC for performance evaluation
# These will be filtered out during the screening process
ROE_RELYING_INDUSTRIES: List[str] = [
    # Asset Management
    "Asset Management",
    
    # Insurance
    "Insurance - Diversified",
    "Insurance - Life", 
    "Insurance - Property & Casualty",
    "Insurance - Specialty",
    "Insurance Brokers",
    
    # Credit/Financial Services
    "Credit Services",
    
    # REITs (Real Estate Investment Trusts)
    "REIT - Diversified",
    "REIT - Residential",
    "REIT - Office",
    "REIT - Retail", 
    "REIT - Industrial",
    "REIT - Specialty",
    "REIT - Mortgage",
    "REIT - Healthcare Facilities",
    "REIT - Hotel & Motel",
    
    # Utilities
    "Utilities - Regulated Electric",
    "Utilities - Regulated Gas",
    "Utilities - Regulated Water",
    "Utilities - Diversified",
    
    # Oil & Gas Infrastructure  
    "Oil & Gas Midstream",
]

# Default processing parameters
DEFAULT_MAX_COUNT = 1000
DEFAULT_RATE_LIMIT_SEC = 0.5
DEFAULT_BATCH_SIZE = 20

# Russell 1000 data source
RUSSELL_1000_WIKI = "https://en.wikipedia.org/wiki/Russell_1000_Index"

# Cache file for ticker list
TICKER_CACHE_FILE = "russell1000tickers.pickle"

# Cache file for known ROE tickers (for fast passthrough)
ROE_TICKERS_CACHE_FILE = "known_roe_tickers.json"


def load_known_roe_tickers() -> Set[str]:
    """Load the cached set of known ROE-relying tickers from disk."""
    cache_path = pathlib.Path(ROE_TICKERS_CACHE_FILE)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return set(data.get('roe_tickers', []))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load ROE tickers cache: {e}")
    return set()


def save_known_roe_tickers(roe_tickers: Set[str]) -> None:
    """Save the set of known ROE-relying tickers to disk for future runs."""
    cache_path = pathlib.Path(ROE_TICKERS_CACHE_FILE)
    try:
        data = {
            'roe_tickers': sorted(list(roe_tickers)),
            'industries': ROE_RELYING_INDUSTRIES
        }
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save ROE tickers cache: {e}")


def update_known_roe_tickers(new_roe_tickers: Set[str]) -> Set[str]:
    """Add newly discovered ROE tickers to the cache and return the updated set."""
    existing = load_known_roe_tickers()
    updated = existing.union(new_roe_tickers)
    
    if len(updated) > len(existing):
        save_known_roe_tickers(updated)
        print(f"Added {len(updated) - len(existing)} new ROE tickers to cache.")
    
    return updated 