# Austrian Stock Screener (Russell-1000 adaptation)

This repository hosts a standalone Python script originally derived from the
`AustrianStockScreener10Q.ipynb` Jupyter notebook. The code has been refactored
into a maintainable, CLI-driven workflow.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
The WACC scraper (`fetch_wacc.py`) depends on `requests` and `beautifulsoup4`, both already listed in `requirements.txt`. If you install packages manually, be sure to include them.

## Usage

Run the screener against the full Russell-1000 universe:

```bash
python austrian_stock_screener.py  # writes austrian.csv in the current folder
```

Key options:

* `--output PATH` – CSV destination (default: `austrian.csv`)
* `--max-count N` – limit the number of tickers processed (default: 1000)
* `--rate-limit SECONDS` – wait time between Yahoo queries (default: 2)
* `--save-ticker-cache` – persist the scraped ticker list to a local pickle so
  subsequent runs start instantly.

## Fetch WACC metrics

Retrieve Weighted Average Cost of Capital data from **valueinvesting.io**:

```bash
# Single ticker
python fetch_wacc.py --tickers PHM

# Batch scrape based on symbols present in austrian.csv
python fetch_wacc.py --input austrian.csv --output wacc.csv
```

The script writes a semicolon-separated CSV (`wacc.csv` by default) with the columns below:

| Column | Description |
|--------|-------------|
| `symbol` | Ticker symbol |
| `wacc` | Selected Weighted Average Cost of Capital (decimal, e.g. `0.081` = 8.1 %) |
| `costOfEquity` | Cost of Equity from CAPM (decimal) |
| `costOfDebt` | Pre-tax Cost of Debt (decimal) |

Example: merge the WACC data with the main screener output in pandas:

```python
import pandas as pd
base = pd.read_csv("austrian.csv", sep=";")
wacc = pd.read_csv("wacc.csv", sep=";")
merged = base.merge(wacc, on="symbol", how="left")
```

## Output format

The resulting CSV (semicolon-separated) contains the following columns for each
ticker that could be processed successfully:

| Column               | Description                                  |
|----------------------|----------------------------------------------|
| `asOfDate`           | Date of the underlying financial statements  |
| `EBIT`               | Earnings Before Interest & Taxes (last TTM)  |
| `InvestedCapital`    | Capital invested in the business             |
| `roic`               | Return on Invested Capital (NOPAT / IC)      |
| `MarketCap`          | Yahoo Finance market capitalisation         |
| `CashAndCashEquivalents` | Self-explanatory                         |
| `totalDebt`          | Sum of short- & long-term debt obligations   |
| `preferredequity`    | CapitalStock – CommonStock                   |
| `faustmannRatio`     | MarketCap / Net-worth (see notebook)         |
| `industry`           | Yahoo Finance industry classification        |
| `fcf`                | Free Cash Flow (TTM)                         |
| `fcfYield`           | Free Cash Flow yield (FCF / MarketCap)       |
| `sanitizedFaustmannRatio` | `faustmannRatio` with negatives set to NaN |
| `roicRank`           | ROIC rank (high → low; 1 = best)            |
| `faustmannRank`      | Faustmann ratio rank (low → high; 1 = best) |
| `sumRanks`           | Combined score (`roicRank` + `faustmannRank`) |

Values can be negative or zero if data is missing or the calculation fails.

---
