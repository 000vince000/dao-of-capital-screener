#!/usr/bin/env python3
"""Point-in-time backtest: apply the current screener's economics (EBIT/InvestedCapital
-> ROIC, 8yr NOPAT/InvestedCapital regression -> ROIIC, excessReturn/growthGate vs
today's WACC as a stand-in) using a historical anchor date, sourced from SEC EDGAR's
XBRL company-facts API (which retains decades of history, unlike yahooquery's ~5yr
rolling window). Then checks forward price performance from the anchor date to today.

Standalone, one-off research script - not wired into the CLI pipeline.

Known approximations (see conversation for full discussion):
  - WACC: today's value used as a stand-in for the anchor date's cost of capital
    (no free source of historical point-in-time WACC).
  - InvestedCapital: reconstructed as (LT debt + ST debt + StockholdersEquity - Cash)
    from raw us-gaap XBRL tags, which will differ somewhat from yahooquery's own
    proprietary InvestedCapital field definition used elsewhere in this repo.
  - Anchor-date MarketCap: dei:EntityCommonStockSharesOutstanding (as reported on
    the nearest 10-K) x historical closing price from Yahoo's chart endpoint.
  - Universe: a fixed, hand-picked list of large, liquid, non-ROE-industry names
    (see UNIVERSE below) rather than a full historical Russell-1000 reconstruction -
    the latter would require reconstructing index membership as of the anchor date,
    which is its own hard problem, plus ~1000 individual SEC fetches.

Usage:
    python sec_edgar_backtest.py --years-ago 5 --top 10
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

SEC_HEADERS = {"User-Agent": "dao-of-capital-screener research contact@example.com"}

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "COST", "HD", "LOW", "MCD",
    "NKE", "SBUX", "TGT", "WMT", "PG", "KO", "PEP", "JNJ", "PFE", "MRK",
    "ABBV", "UNH", "CAT", "DE", "HON", "GE", "BA", "LMT", "RTX", "XOM",
    "CVX", "ADBE", "CRM", "ORCL", "CSCO", "INTC", "QCOM", "TXN", "AMD", "IBM",
]

# ---------------------------------------------------------------------------
# yahooquery session patch (see normalize_ebit.py for the full explanation) -
# only used here for historical/current price via the chart endpoint, which
# doesn't need the crumb.
# ---------------------------------------------------------------------------
import yahooquery.session_management as sm
import yahooquery.base as _yq_base
from curl_cffi import requests as creq


def _non_impersonated_session(session=None, **kwargs):
    if session is None:
        kwargs.pop("max_workers", None)
        kwargs.pop("asynchronous", None)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        }
        session = creq.Session(**kwargs, headers=headers)
        session = sm.setup_session(session)
    return session


sm.initialize_session = _non_impersonated_session
_yq_base.initialize_session = _non_impersonated_session

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compute_roiic import compute_roiic_slope, normalize_annual_series, apply_buyback_addback  # reuse the exact same logic as the live pipeline
from fetch_wacc import fetch_wacc

_CIK_MAP: Optional[dict] = None


def _load_cik_map() -> dict:
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    _CIK_MAP = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}
    return _CIK_MAP


def _get_facts(ticker: str) -> Optional[dict]:
    cik_map = _load_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return None
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()


def _first_present(gaap: dict, tags: list[str]) -> Optional[dict]:
    for t in tags:
        if t in gaap and "USD" in gaap[t].get("units", {}):
            return gaap[t]["units"]["USD"]
    return None


def _annual_series(concept_units: list[dict]) -> pd.DataFrame:
    """10-K, full-year (~350-380 day) duration facts, deduped by fiscal year end."""
    rows = []
    for x in concept_units:
        if x.get("form") != "10-K":
            continue
        try:
            start = pd.to_datetime(x["start"])
            end = pd.to_datetime(x["end"])
        except (KeyError, ValueError):
            continue
        days = (end - start).days
        if not (350 <= days <= 380):
            continue
        rows.append({"end": end, "val": x["val"], "filed": x.get("filed")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["end", "filed"])
    return df.groupby("end", as_index=False).last()  # prefer the latest-filed restatement per FY


def _instant_series(concept_units: list[dict]) -> pd.DataFrame:
    """Balance-sheet point-in-time facts (10-K or 10-Q), one value per 'end' date."""
    rows = []
    for x in concept_units:
        if x.get("form") not in ("10-K", "10-Q"):
            continue
        try:
            end = pd.to_datetime(x["end"])
        except (KeyError, ValueError):
            continue
        rows.append({"end": end, "val": x["val"], "filed": x.get("filed")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["end", "filed"])
    return df.groupby("end", as_index=False).last()


def _historical_price(symbol: str, target_date: date) -> Optional[float]:
    from yahooquery import Ticker

    t = Ticker(symbol, asynchronous=False)
    start = (target_date - timedelta(days=10)).isoformat()
    end = (target_date + timedelta(days=10)).isoformat()
    hist = t.history(start=start, end=end, interval="1d")
    if not isinstance(hist, pd.DataFrame) or hist.empty or "close" not in hist.columns:
        return None
    hist = hist.reset_index()
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    hist["delta"] = (hist["date"] - pd.Timestamp(target_date)).abs()
    return float(hist.sort_values("delta").iloc[0]["close"])


def process_ticker(ticker: str, anchor_date: date, wacc: float) -> dict:
    facts = _get_facts(ticker)
    if not facts or "us-gaap" not in facts.get("facts", {}):
        return {"symbol": ticker, "status": "no_sec_data"}
    gaap = facts["facts"]["us-gaap"]

    pretax_units = _first_present(gaap, [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ])
    tax_units = _first_present(gaap, ["IncomeTaxExpenseBenefit"])
    tax_annual = _annual_series(tax_units) if tax_units else pd.DataFrame()
    pretax_annual = _annual_series(pretax_units) if pretax_units else pd.DataFrame()

    ebit_units = _first_present(gaap, ["OperatingIncomeLoss"])
    ebit_annual = _annual_series(ebit_units) if ebit_units else pd.DataFrame()
    if ebit_annual.empty:
        # Fallback: not every filer tags a distinct "operating income" subtotal.
        # EBIT ~= pretax income + interest expense (adds back financing cost).
        interest_units = _first_present(gaap, ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"])
        if pretax_annual.empty or interest_units is None:
            return {"symbol": ticker, "status": "no_ebit_tag_or_fallback"}
        interest_annual = _annual_series(interest_units)
        ebit_annual = pretax_annual.merge(interest_annual, on="end", suffixes=("_pretax", "_interest"))
        ebit_annual["val"] = ebit_annual["val_pretax"] + ebit_annual["val_interest"]
        ebit_annual = ebit_annual[["end", "val"]]
    if ebit_annual.empty:
        return {"symbol": ticker, "status": "no_annual_ebit"}

    equity_units = _first_present(gaap, ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"])
    cash_units = _first_present(gaap, ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"])
    ltdebt_units = _first_present(gaap, ["LongTermDebtNoncurrent", "LongTermDebt"])
    stdebt_units = _first_present(gaap, ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"])

    equity_i = _instant_series(equity_units) if equity_units else pd.DataFrame()
    cash_i = _instant_series(cash_units) if cash_units else pd.DataFrame()
    ltdebt_i = _instant_series(ltdebt_units) if ltdebt_units else pd.DataFrame()
    stdebt_i = _instant_series(stdebt_units) if stdebt_units else pd.DataFrame()

    if equity_i.empty:
        return {"symbol": ticker, "status": "no_balance_sheet_data"}

    def _lookup(df: pd.DataFrame, on_or_before: pd.Timestamp) -> float:
        if df.empty:
            return 0.0
        sub = df[df["end"] <= on_or_before]
        return float(sub.iloc[-1]["val"]) if not sub.empty else 0.0

    def invested_capital_at(dt: pd.Timestamp) -> float:
        eq = _lookup(equity_i, dt)
        cash = _lookup(cash_i, dt)
        lt = _lookup(ltdebt_i, dt)
        st = _lookup(stdebt_i, dt)
        return (lt + st) + eq - cash

    def tax_rate_for_fy(fy_end: pd.Timestamp) -> Optional[float]:
        pre = pretax_annual[pretax_annual["end"] == fy_end]
        tax = tax_annual[tax_annual["end"] == fy_end]
        if pre.empty or tax.empty or pre.iloc[0]["val"] == 0:
            return None
        rate = float(tax.iloc[0]["val"]) / float(pre.iloc[0]["val"])
        if rate < -0.5 or rate > 0.6:  # sanity clamp, same spirit as the live pipeline
            return None
        return rate

    def pretax_for_fy(fy_end: pd.Timestamp) -> Optional[float]:
        pre = pretax_annual[pretax_annual["end"] == fy_end]
        return float(pre.iloc[0]["val"]) if not pre.empty else None

    # --- anchor year: most recent FY10-K ending on/before anchor_date ---
    anchor_ts = pd.Timestamp(anchor_date)
    eligible = ebit_annual[ebit_annual["end"] <= anchor_ts]
    if eligible.empty:
        return {"symbol": ticker, "status": "no_data_before_anchor"}
    anchor_fy_end = eligible.iloc[-1]["end"]

    # --- Build one annual window (up to 8 FYs ending at-or-before the anchor
    # year) shared by both the anchor-year ROIC snapshot and the ROIIC
    # regression, and normalize it ONCE via the same shared function the live
    # pipeline uses, so the two figures stay internally consistent. ---
    rd_units = _first_present(gaap, ["ResearchAndDevelopmentExpense"])
    sga_units = _first_present(gaap, ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"])
    restructuring_units = _first_present(gaap, ["RestructuringCharges"])
    rd_annual = _annual_series(rd_units) if rd_units else pd.DataFrame()
    sga_annual = _annual_series(sga_units) if sga_units else pd.DataFrame()
    restructuring_annual = _annual_series(restructuring_units) if restructuring_units else pd.DataFrame()

    def _val_for(series_df: pd.DataFrame, fy_end: pd.Timestamp) -> Optional[float]:
        if series_df.empty:
            return None
        row = series_df[series_df["end"] == fy_end]
        return float(row.iloc[0]["val"]) if not row.empty else None

    hist_window = ebit_annual[ebit_annual["end"] <= anchor_fy_end].tail(8).copy().rename(columns={"val": "EBIT"})
    hist_window["InvestedCapital"] = hist_window["end"].apply(invested_capital_at)
    hist_window["PretaxIncome"] = hist_window["end"].apply(pretax_for_fy)
    hist_window["TaxRateForCalcs"] = hist_window["end"].apply(lambda e: tax_rate_for_fy(e))
    hist_window["ResearchAndDevelopment"] = hist_window["end"].apply(lambda e: _val_for(rd_annual, e))
    hist_window["SellingGeneralAndAdministration"] = hist_window["end"].apply(lambda e: _val_for(sga_annual, e))
    hist_window["RestructuringAndMergernAcquisition"] = hist_window["end"].apply(lambda e: _val_for(restructuring_annual, e))

    # .apply() over a column of Python None/float results in an object-dtype
    # column, which silently breaks downstream numpy/scipy arithmetic in
    # confusing ways - coerce to real numeric dtype before anything else.
    for col in ["EBIT", "PretaxIncome", "TaxRateForCalcs", "ResearchAndDevelopment",
                "SellingGeneralAndAdministration", "RestructuringAndMergernAcquisition"]:
        hist_window[col] = pd.to_numeric(hist_window[col], errors="coerce")

    # Fill in-window missing/None tax rates with the window's own median before normalizing,
    # so normalize_annual_series() has something usable for every row (mirrors the live
    # pipeline's tolerance for gaps rather than dropping rows outright).
    fallback_rate = hist_window["TaxRateForCalcs"].dropna().median()
    if pd.isna(fallback_rate):
        fallback_rate = 0.21
    hist_window["TaxRateForCalcs"] = hist_window["TaxRateForCalcs"].fillna(fallback_rate)

    normalized = normalize_annual_series(hist_window)
    hist_window["EBIT_normalized"] = normalized["EBIT_normalized"].values
    hist_window["TaxRateForCalcs_normalized"] = normalized["TaxRateForCalcs_normalized"].values
    hist_window["flagged"] = normalized["flagged"].values
    hist_window["flag_reasons"] = normalized["flag_reasons"].values
    hist_window["nopat"] = hist_window["EBIT_normalized"] * (1 - hist_window["TaxRateForCalcs_normalized"])
    hist_window["year"] = hist_window["end"].dt.year

    # --- Buyback-adjusted (organic) InvestedCapital, for the ROIIC regression
    # ONLY (TODO.md #4) - the anchor-year ROIC/excessReturn snapshot below
    # deliberately keeps using the real, unadjusted InvestedCapital, since
    # that reflects the business's actual current capital base. The buyback
    # addback only cleans up the *trend* the regression fits, not the level
    # of capital the company is genuinely operating with today. ---
    buyback_units = _first_present(gaap, ["PaymentsForRepurchaseOfCommonStock"])
    buyback_annual = _annual_series(buyback_units) if buyback_units else pd.DataFrame()
    hist_window = hist_window.sort_values("end").reset_index(drop=True)
    hist_window["RepurchaseOfCapitalStock"] = hist_window["end"].apply(lambda e: _val_for(buyback_annual, e))
    hist_window = apply_buyback_addback(hist_window)

    anchor_row = hist_window[hist_window["end"] == anchor_fy_end].iloc[0]
    anchor_ebit = float(anchor_row["EBIT_normalized"])
    anchor_tax = float(anchor_row["TaxRateForCalcs_normalized"])
    anchor_ic = float(anchor_row["InvestedCapital"])  # unadjusted - real capital base at the anchor date
    if anchor_ic <= 0:
        return {"symbol": ticker, "status": "bad_invested_capital"}

    anchor_nopat = anchor_ebit * (1 - anchor_tax)
    anchor_roic = anchor_nopat / anchor_ic

    regression_input = hist_window[["year", "nopat", "InvestedCapital_organic"]].rename(
        columns={"InvestedCapital_organic": "InvestedCapital"}
    )
    roiic, roiic_reason = compute_roiic_slope(regression_input, with_reason=True)

    excess_return = anchor_roic - wacc
    growth_gate = (roiic - wacc) if roiic is not None else None
    flagged_years = hist_window[hist_window["flagged"]][["year", "flag_reasons"]]
    flagged_desc = "; ".join(f"{int(r.year)}: {r.flag_reasons}" for r in flagged_years.itertuples()) or "(none)"

    price_anchor = _historical_price(ticker, anchor_fy_end.date())
    price_now = _historical_price(ticker, date.today())
    fwd_return = None
    if price_anchor and price_now:
        fwd_return = (price_now / price_anchor) - 1

    return {
        "symbol": ticker,
        "status": "ok",
        "anchor_fy_end": str(anchor_fy_end.date()),
        "anchor_ebit": anchor_ebit,
        "anchor_ic": anchor_ic,
        "anchor_tax": anchor_tax,
        "roic": anchor_roic,
        "wacc": wacc,
        "excessReturn": excess_return,
        "roiic": roiic,
        "roiic_reason": roiic_reason,
        "growthGate": growth_gate,
        "data_points_used": len(hist_window),
        "flagged_years": flagged_desc,
        "price_anchor": price_anchor,
        "price_now": price_now,
        "fwd_return": fwd_return,
    }


def main():
    p = argparse.ArgumentParser(description="Point-in-time backtest of the screener's economics using SEC EDGAR history.")
    p.add_argument("--years-ago", type=float, default=5.0)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--tickers", type=str, help="Comma-separated override for UNIVERSE")
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("sec_backtest.csv"))
    args = p.parse_args()

    anchor_date = date.today() - timedelta(days=int(args.years_ago * 365.25))
    universe = [s.strip() for s in args.tickers.split(",")] if args.tickers else UNIVERSE
    print(f"Anchor date: ~{anchor_date} ({args.years_ago} years ago). Universe: {len(universe)} tickers.\n")

    wacc_sess = requests.Session()
    wacc_sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; WACC-Scraper/1.0; +https://github.com/)"})

    rows = []
    for i, sym in enumerate(universe, 1):
        print(f"[{i}/{len(universe)}] {sym}...", flush=True)
        wacc_row, _ = fetch_wacc(sym, wacc_sess)
        wacc = wacc_row.get("wacc")
        time.sleep(0.8)
        if wacc is None:
            print("    · no WACC, skipping")
            rows.append({"symbol": sym, "status": "no_wacc"})
            continue
        try:
            r = process_ticker(sym, anchor_date, wacc)
        except Exception as exc:
            print(f"    · error: {exc}")
            r = {"symbol": sym, "status": f"error: {exc}"}
        rows.append(r)
        if r.get("status") == "ok":
            fwd = r["fwd_return"]
            fwd_str = f"{fwd:+.1%}" if fwd is not None else "n/a"
            print(f"    · roic={r['roic']:.3f} wacc={wacc:.3f} excessReturn={r['excessReturn']:+.3f} "
                  f"roiic={r['roiic']} growthGate={r['growthGate']} fwd_return={fwd_str}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, sep=";", index=False)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("\nNo tickers processed successfully.")
        return

    ok["valueMetricRank"] = ok["excessReturn"].rank(ascending=False)  # proxy: no EnterpriseValue reconstructed here
    passed = ok[(ok["excessReturn"] > 0)]
    if "growthGate" in passed.columns:
        qualifiers = passed[(passed["growthGate"].isna()) | (passed["growthGate"] > 0)]
    else:
        qualifiers = passed
    qualifiers = qualifiers.sort_values("excessReturn", ascending=False)

    print(f"\n=== {len(qualifiers)}/{len(ok)} names would have qualified (excessReturn>0 and growthGate>=0 or n/a) as of ~{anchor_date} ===")
    print(qualifiers[["symbol", "excessReturn", "roiic", "growthGate", "fwd_return"]].to_string(index=False))

    if not qualifiers.empty and qualifiers["fwd_return"].notna().any():
        qual_avg = qualifiers["fwd_return"].dropna().mean()
        all_avg = ok["fwd_return"].dropna().mean()
        print(f"\nQualifiers avg forward return: {qual_avg:+.1%}  |  Full universe avg: {all_avg:+.1%}")

    print(f"\n✓ Saved → {args.output.resolve()}")


if __name__ == "__main__":
    main()
