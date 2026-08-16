#!/usr/bin/env python3
"""Flag and normalize one-time/unusual items in quarterly EBIT before computing ROIC.

Standalone by design — NOT wired into current_baseline_data.py or
compute_roiic.py. Raw GAAP EBIT can swing wildly on a single quarter's
acquired-IPR&D charge, restructuring, or other one-off item, which can flip
an `excessReturn`/thesis verdict without any real change in the business's
capital-cycle economics (see: MRK, Q1-Q2 2026, ~$14B addback). Normalizing
that automatically requires judgment calls (how big a spike counts, whether
a step-change is "real"), so this stays a separate, explicitly-run check —
run it on names the raw pipeline flags as borderline/broken, sanity-check
the flagged quarters yourself, and treat the normalized number as a second
opinion rather than a drop-in replacement for the raw one.

Heuristics used to flag a quarter's EBIT as distorted:
  1. Yahoo's own `TotalUnusualItems` field is nonzero for that quarter
     (already-tagged one-off items; always added back).
  2. Any of {ResearchAndDevelopment, SellingGeneralAndAdministration,
     RestructuringAndMergernAcquisition} exceeds `--spike-multiple` times
     the median of that same quarter's prior clean quarters (default 2x).
  3. `PretaxIncome` is negative, which makes `TaxRateForCalcs` (a raw
     TaxProvision/PretaxIncome ratio) meaningless — the quarter's tax rate
     is replaced with the median of clean quarters when reconstructing NOPAT.

Usage:
    python normalize_ebit.py --tickers MRK,PPC,LULU
    python normalize_ebit.py --input current_baseline_data.csv --wacc-file wacc_top.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import List, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# yahooquery session patch — scoped to this script only.
#
# In some sandboxed/proxied environments, curl_cffi's browser-TLS-impersonation
# mode (which yahooquery uses by default) gets its connection reset mid-handshake
# by a TLS-terminating egress proxy. A plain (non-impersonated) curl_cffi
# session works, but it can't complete Yahoo's crumb handshake (Set-Cookie
# appears to get stripped by such proxies), so crumb-gated endpoints
# (summary_detail/summary_profile/price) stay unreachable — this script only
# needs balance_sheet/income_statement/valuation_measures, none of which
# require a crumb, so it's unaffected. Deliberately NOT applied to
# current_baseline_data.py/compute_roiic.py: those rely on the crumb-gated
# endpoints for MarketCap/industry, and in an environment where the normal
# impersonated session works fine, this patch would only make things worse.
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


def _patch_yahooquery_session() -> None:
    sm.initialize_session = _non_impersonated_session
    _yq_base.initialize_session = _non_impersonated_session


# ---------------------------------------------------------------------------
# Normalization logic
# ---------------------------------------------------------------------------

SPIKE_PRONE_COLS = [
    "ResearchAndDevelopment",
    "SellingGeneralAndAdministration",
    "RestructuringAndMergernAcquisition",
]


def _flag_and_normalize(inc_q: pd.DataFrame, spike_multiple: float) -> dict:
    """Given a symbol's quarterly ('3M') income-statement rows (>=2, ideally
    5+, sorted by asOfDate ascending), return a dict describing raw vs
    normalized TTM EBIT/tax-rate over the most recent 4 quarters.
    """
    q = inc_q.sort_values("asOfDate").reset_index(drop=True)
    if len(q) < 2:
        return {"status": "insufficient_quarters"}

    last4 = q.tail(4)
    prior_pool = q.iloc[: max(0, len(q) - 4)]  # quarters before the TTM window, for a clean baseline

    flagged_quarters = []
    addback_total = 0.0
    raw_ebit_ttm = last4["EBIT"].sum()

    # Clean-quarter tax rate: PretaxIncome > 0 only, prefer quarters outside the TTM window
    clean_tax_pool = q[q["PretaxIncome"] > 0]["TaxRateForCalcs"]
    if clean_tax_pool.empty:
        clean_tax_pool = q["TaxRateForCalcs"]
    normalized_tax_rate = float(clean_tax_pool.median())

    for _, row in last4.iterrows():
        reasons = []
        addback = 0.0

        unusual = row.get("TotalUnusualItems")
        if pd.notna(unusual) and unusual != 0:
            addback += -float(unusual)  # unusual items are already netted into EBIT; subtract to remove
            reasons.append(f"TotalUnusualItems={unusual:,.0f}")

        # Build a running clean baseline from everything before this row (prior_pool + earlier last4 rows
        # not already flagged), so a second bad quarter doesn't get judged against an already-inflated baseline.
        baseline_source = pd.concat([prior_pool, last4[last4["asOfDate"] < row["asOfDate"]]])
        for col in SPIKE_PRONE_COLS:
            if col not in row or pd.isna(row[col]):
                continue
            baseline_vals = baseline_source[col].dropna()
            baseline_vals = baseline_vals[baseline_vals > 0]
            if len(baseline_vals) < 2:
                continue
            baseline = float(baseline_vals.median())
            if baseline <= 0:
                continue
            if row[col] > spike_multiple * baseline:
                excess = float(row[col]) - baseline
                addback += excess
                reasons.append(f"{col}={row[col]:,.0f} vs baseline {baseline:,.0f} (+{excess:,.0f})")

        if reasons:
            flagged_quarters.append({
                "asOfDate": str(row["asOfDate"]),
                "reasons": "; ".join(reasons),
                "addback": addback,
            })
            addback_total += addback

    normalized_ebit_ttm = raw_ebit_ttm + addback_total
    raw_tax_rate = float(last4["TaxRateForCalcs"].iloc[-1])  # matches what the live pipeline uses (latest TTM row)

    return {
        "status": "ok",
        "raw_ebit_ttm": raw_ebit_ttm,
        "normalized_ebit_ttm": normalized_ebit_ttm,
        "raw_tax_rate": raw_tax_rate,
        "normalized_tax_rate": normalized_tax_rate,
        "flagged_quarters": flagged_quarters,
        "addback_total": addback_total,
    }


def process_ticker(symbol: str, spike_multiple: float) -> dict:
    from yahooquery import Ticker

    t = Ticker(symbol, asynchronous=False)
    inc = t.income_statement(frequency="q")
    bs = t.balance_sheet(frequency="q")
    if not isinstance(inc, pd.DataFrame) or inc.empty:
        return {"symbol": symbol, "status": "no_data"}
    inc_q = inc[inc["periodType"] == "3M"]
    if inc_q.empty:
        return {"symbol": symbol, "status": "no_quarterly_data"}

    bs_3m = bs[bs["periodType"] == "3M"].sort_values("asOfDate") if isinstance(bs, pd.DataFrame) else pd.DataFrame()
    invested_capital = float(bs_3m["InvestedCapital"].iloc[-1]) if not bs_3m.empty and "InvestedCapital" in bs_3m.columns else None

    result = _flag_and_normalize(inc_q, spike_multiple)
    result["symbol"] = symbol
    result["invested_capital"] = invested_capital
    return result


def main():
    p = argparse.ArgumentParser(description="Flag and normalize one-time items in quarterly EBIT before computing ROIC.")
    p.add_argument("--tickers", type=str, help="Comma-separated tickers")
    p.add_argument("--input", type=pathlib.Path, help="CSV with a 'symbol' column (alternative to --tickers)")
    p.add_argument("--wacc-file", type=pathlib.Path, help="Semicolon-separated CSV with symbol;wacc (default: fetch fresh from valueinvesting.io)")
    p.add_argument("--spike-multiple", type=float, default=2.0, help="Flag an expense line if it exceeds this multiple of its trailing baseline (default: 2.0)")
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("normalized_ebit.csv"))
    args = p.parse_args()

    if args.tickers:
        tickers: List[str] = [s.strip() for s in args.tickers.split(",") if s.strip()]
    elif args.input:
        df_in = pd.read_csv(args.input, sep=";")
        tickers = df_in["symbol"].dropna().astype(str).unique().tolist()
    else:
        print("Provide --tickers or --input", file=sys.stderr)
        sys.exit(1)

    wacc_map = {}
    if args.wacc_file and args.wacc_file.exists():
        wacc_df = pd.read_csv(args.wacc_file, sep=";")
        wacc_map = dict(zip(wacc_df["symbol"], wacc_df["wacc"]))

    _patch_yahooquery_session()

    from fetch_wacc import fetch_wacc
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; WACC-Scraper/1.0; +https://github.com/)"})

    rows = []
    for sym in tickers:
        print(f"Processing {sym}...", flush=True)
        r = process_ticker(sym, args.spike_multiple)
        if r.get("status") != "ok":
            print(f"  · {sym}: {r.get('status')}")
            rows.append({"symbol": sym, "status": r.get("status")})
            continue

        wacc = wacc_map.get(sym)
        if wacc is None:
            wacc_row, _ = fetch_wacc(sym, sess)
            wacc = wacc_row.get("wacc")
            time.sleep(1.0)

        ic = r["invested_capital"]
        raw_nopat = r["raw_ebit_ttm"] * (1 - r["raw_tax_rate"]) if r["raw_tax_rate"] is not None else None
        norm_nopat = r["normalized_ebit_ttm"] * (1 - r["normalized_tax_rate"])

        raw_roic = raw_nopat / ic if ic else None
        norm_roic = norm_nopat / ic if ic else None
        raw_excess = (raw_roic - wacc) if (raw_roic is not None and wacc is not None) else None
        norm_excess = (norm_roic - wacc) if (norm_roic is not None and wacc is not None) else None

        flagged_desc = " | ".join(f"{fq['asOfDate']}: {fq['reasons']}" for fq in r["flagged_quarters"]) or "(none flagged)"

        row = {
            "symbol": sym,
            "status": "ok",
            "wacc": wacc,
            "raw_ebit_ttm": r["raw_ebit_ttm"],
            "normalized_ebit_ttm": r["normalized_ebit_ttm"],
            "addback_total": r["addback_total"],
            "raw_tax_rate": r["raw_tax_rate"],
            "normalized_tax_rate": r["normalized_tax_rate"],
            "invested_capital": ic,
            "raw_roic": raw_roic,
            "normalized_roic": norm_roic,
            "raw_excessReturn": raw_excess,
            "normalized_excessReturn": norm_excess,
            "flagged_quarters": flagged_desc,
        }
        rows.append(row)
        verdict_raw = "BROKEN" if (raw_excess is not None and raw_excess < 0) else "intact"
        verdict_norm = "BROKEN" if (norm_excess is not None and norm_excess < 0) else "intact"
        print(f"  · raw excessReturn={raw_excess:.4f} ({verdict_raw})  normalized={norm_excess:.4f} ({verdict_norm})  addback=${r['addback_total']:,.0f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, sep=";", index=False)
    print(f"\n✓ Saved → {args.output.resolve()}")


if __name__ == "__main__":
    main()
