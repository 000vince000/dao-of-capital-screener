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

Values can be negative or zero if data is missing or the calculation fails.

---

*Inspired by **Nassim Nicholas Taleb's** capital allocation ideas & the original
notebook by @000vince000.* 