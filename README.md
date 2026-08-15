# Austrian Stock Screener (Russell-1000 adaptation)

This repository hosts a standalone Python pipeline originally derived from the
`AustrianStockScreener10Q.ipynb` Jupyter notebook. The code has been refactored
into a maintainable, CLI-driven workflow that screens the Russell-1000 universe
for companies earning returns on invested capital above their cost of capital
(an "Austrian" / capital-cycle approach), ranks them, and estimates their
Return on Incremental Invested Capital (ROIIC) as a growth-quality check.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All third-party dependencies (`pandas`, `numpy`, `scipy`, `requests`,
`beautifulsoup4`, `yahooquery`) are listed in `requirements.txt`.

## Pipeline overview

The pipeline runs in five stages, each backed by its own script and CSV
output. `main.py` orchestrates all of them end-to-end:

```bash
python main.py  # writes top50_overview.csv (and every intermediate CSV) in the current folder
```

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1. Screen universe | `current_baseline_data.py` | Russell-1000 ticker list (scraped from Wikipedia) | `current_baseline_data.csv` |
| 2. Fetch cost of capital | `fetch_wacc.py` | `current_baseline_data.csv` | `wacc_top.csv` |
| 3. Rank by value & excess return | `normalized_austrian_screener.py` | `current_baseline_data.csv` + `wacc_top.csv` | `normalized_austrian.csv` |
| 4. Estimate growth quality | `compute_roiic.py` | `normalized_austrian.csv` | `roiic_top.csv` |
| 5. Summarize | `main.py` (merge step) | outputs of stages 1–4 | `top50_overview.csv` |

Key `main.py` options:

* `--top N` – number of tickers to keep after ranking (default: `50`)
* `--output PATH` – destination CSV (default: `top50_overview.csv`)
* `--skip-screener` – reuse an existing `current_baseline_data.csv` instead of
  re-running stage 1

Each stage also skips itself automatically if its output CSV already exists,
so re-running `main.py` after a partial failure only redoes the missing
steps. Delete the relevant CSV (or the `--skip-screener` guard) to force a
refresh.

### Stage 1 — Screen the universe

```bash
python current_baseline_data.py  # writes current_baseline_data.csv in the current folder
```

For each ticker this fetches quarterly balance-sheet, income-statement,
cash-flow, market-cap and industry data via `yahooquery`, filters out
industries that structurally rely on ROE rather than ROIC (asset management,
insurance, banking, REITs, utilities, oil & gas midstream — see
`ROE_RELYING_INDUSTRIES` in `config.py`), and computes NOPAT/ROIC and related
metrics. Tickers already known to be ROE-relying are cached in
`known_roe_tickers.json` and skipped on subsequent runs.

Key options:

* `--output PATH` – CSV destination (default: `current_baseline_data.csv`)
* `--max-count N` – limit the number of tickers processed (default: 1000)
* `--rate-limit SECONDS` – initial wait time between Yahoo queries (default: 0.5)
* `--batch-size N` – number of tickers fetched per Yahoo batch call (default: 20)
* `--save-ticker-cache` – persist the scraped ticker list to a local pickle
  (`russell1000tickers.pickle`) so subsequent runs start instantly.

### Stage 2 — Fetch WACC metrics

Retrieve Weighted Average Cost of Capital data from **valueinvesting.io**:

```bash
# Single ticker
python fetch_wacc.py --tickers PHM

# Batch scrape based on symbols present in current_baseline_data.csv
python fetch_wacc.py --input current_baseline_data.csv --output wacc_top.csv
```

The script writes a semicolon-separated CSV (`wacc_top.csv` by default) with
the columns below:

| Column | Description |
|--------|-------------|
| `symbol` | Ticker symbol |
| `wacc` | Selected Weighted Average Cost of Capital (decimal, e.g. `0.081` = 8.1 %) |
| `costOfEquity` | Cost of Equity from CAPM (decimal) |
| `costOfDebt` | Pre-tax Cost of Debt (decimal) |

Other options: `--workers N` (parallel threads, default 5), `--rate-limit
SECONDS` (default 1.0), `--flush-size N` (rows between incremental CSV
flushes, default 10), `--failed-output PATH` (tickers whose page loaded but
had no WACC value, default `wacc_failed.csv`), `--max-count N`.

### Stage 3 — Rank by value & excess return

```bash
python normalized_austrian_screener.py --input current_baseline_data.csv --wacc-file wacc_top.csv --output normalized_austrian.csv
```

Merges the WACC data onto the baseline screen and computes:

* `excessReturn` = `roic - wacc`, ranked descending as `excessReturnRank` (1 = best)
* `valueMetric` (see output table below), ranked descending as `valueMetricRank` (1 = best)
* `rankingScore` = `valueMetricRank + excessReturnRank` (lower is better)

The output is sorted by `rankingScore` ascending.

### Stage 4 — Estimate ROIIC

```bash
python compute_roiic.py --input normalized_austrian.csv --output roiic_top.csv
```

For each ticker, fetches up to eight years of annual income-statement and
balance-sheet data, fits a regression of NOPAT against Invested Capital, and
reports the slope as ROIIC (Return on Incremental Invested Capital). Tickers
with insufficient data points, negligible incremental capital, or a zero
capital slope are recorded with `roiic = None`. Writes `symbol`, `roiic`,
`data_points_used` to `roiic_top.csv`.

### Stage 5 — Summary

`main.py` takes the top-N tickers by `rankingScore`, merges in ROIIC, and
computes `growthGate = roiic - wacc` (positive = the company's reinvestment
returns still clear its cost of capital). The final `top50_overview.csv` is
sorted by `growthGate` descending and contains:

`symbol`, `industry`, `MarketCap`, `EBIT`, `EnterpriseValue`, `roic`, `wacc`,
`roiic`, `valueMetric`, `valueMetricRank`, `excessReturn`,
`excessReturnRank`, `rankingScore`, `growthGate`.

## Output format — `current_baseline_data.csv`

The resulting CSV (semicolon-separated) contains the following columns for
each ticker that could be processed successfully:

| Column               | Description                                  |
|----------------------|-----------------------------------------------|
| `symbol`             | Ticker symbol                                |
| `asOfDate`           | Date of the underlying financial statements  |
| `EBIT`               | Earnings Before Interest & Taxes (last quarter) |
| `nopat`              | Net Operating Profit After Tax (`EBIT × (1 − tax rate)`) |
| `InvestedCapital`    | Capital invested in the business             |
| `roic`               | Return on Invested Capital (`nopat / InvestedCapital`) |
| `MarketCap`          | Yahoo Finance market capitalisation          |
| `CashAndCashEquivalents` | Self-explanatory                          |
| `totalDebt`          | Sum of deferred liabilities & long-term debt/lease obligations |
| `preferredequity`    | CapitalStock – CommonStock                   |
| `opCashFlow`         | Operating Cash Flow (last quarter)           |
| `EnterpriseValue`    | Yahoo `EnterpriseValue`, else `MarketCap + totalDebt + preferredequity - CashAndCashEquivalents` |
| `opCashFlowYield`    | `opCashFlow / EnterpriseValue`               |
| `NetIncome`          | Net income (last quarter)                    |
| `TotalShareholderEquity` | Total shareholder equity (last quarter)  |
| `roe`                | Return on Equity (`NetIncome / TotalShareholderEquity`) |
| `valueMetric`        | `EBIT / EnterpriseValue`                     |
| `industry`           | Yahoo Finance industry classification        |

Values can be negative, zero, or missing if data is unavailable or a
calculation fails.

---
