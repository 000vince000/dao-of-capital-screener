#!/usr/bin/env python3
"""Rebuild NOPAT and InvestedCapital from verified yahooquery building blocks and
compare against the current pipeline's EBIT-field-based calculation.

Standalone by design — NOT wired into current_baseline_data.py or main.py. See
`normalize_ebit.py` for the precedent of running a second-opinion calculation
side by side with the live pipeline before deciding whether to fold it in.

Why this exists
----------------
Mauboussin & Callahan, "Return on Invested Capital" (Morgan Stanley
Counterpoint Global, Oct 2022) — see README.md References — defines NOPAT and
InvestedCapital from explicit operating-statement/balance-sheet building
blocks. The live pipeline instead uses yahooquery's own `EBIT` and
`InvestedCapital` fields directly. A live audit (this session) confirmed
yahooquery's `EBIT` field equals `PretaxIncome + InterestExpense`, which
wrongly folds in non-operating income/expense — for MSFT this inflated NOPAT
~8.9% and overstated ROIC ~2.3 points versus using true operating income.

Verified field mapping (see the project's plan file for the full audit)
-------------------------------------------------------------------------
NOPAT = EBITA - cash taxes, built as:
  * EBIT (operating)              -> OperatingIncome (NOT yahooquery's EBIT field)
  * + Amortization of acquired
      intangibles                 -> NOT AVAILABLE. yahooquery only exposes
                                      combined D&A (Depreciation ==
                                      DepreciationAndAmortization ==
                                      DepreciationAmortizationDepletion, all
                                      identical); no intangible-only split
                                      exists. Excluded, logged.
  * + Embedded operating-lease
      interest                    -> NOT AVAILABLE. No lease-interest-expense
                                      field found. Excluded, logged.
  * Tax provision                 -> TaxProvision
  * Deferred taxes                -> DeferredIncomeTax (annual cash_flow
                                      statement). IMPORTANT: this is reported
                                      in the standard cash-flow add-back
                                      convention (positive = non-cash portion
                                      of the tax provision, i.e. book tax
                                      expense exceeded cash tax paid) -
                                      verified by reconciling MSFT's full
                                      operating-cash-flow identity in this
                                      session. That is the OPPOSITE sign from
                                      how the paper's own Exhibit 2 lists its
                                      "deferred taxes" cash-tax-buildup line,
                                      so it is SUBTRACTED here, not added:
                                          cash_taxes = TaxProvision
                                                        - DeferredIncomeTax
                                                        + tax_shield
  * Tax shield                    -> max(InterestExpense - InterestIncome, 0)
                                      x STATUTORY_TAX_RATE. The paper wants a
                                      marginal/statutory rate here, not the
                                      reported effective rate.
  * + Goodwill/intangible
      impairment add-back         -> AssetImpairmentCharge (annual cash_flow
                                      statement). The paper: "It is standard
                                      to add back goodwill and intangible
                                      impairment charges... management should
                                      be held accountable for past capital
                                      allocation decisions." Cross-checked
                                      against real filings this session, not
                                      assumed from the field name: KHC's 2025
                                      value ($9.306B) matches their reported
                                      $6.7B goodwill + $2.6B intangible
                                      impairment ($9.3B combined) almost
                                      exactly; WBD's 2024 value ($9.603B) is
                                      close to their reported $9.1B Networks
                                      goodwill impairment (the ~5% gap is
                                      plausibly other smaller impairments
                                      bundled into the same field). Caveat:
                                      the field is named generically ("asset"
                                      impairment, not "goodwill" impairment)
                                      so it may occasionally include non-
                                      goodwill/intangible write-downs (e.g.
                                      PP&E) - not confirmed goodwill-only,
                                      just confirmed goodwill-dominated for
                                      the two names checked. NaN in this
                                      field means "no impairment reported
                                      that year" (confirmed: MSFT, which
                                      hasn't had one recently, returns NaN
                                      here) - a legitimate zero, not missing
                                      data, so it safely defaults to 0.0
                                      (unlike DeferredIncomeTax/interest
                                      fields above, where NaN genuinely means
                                      "unknown").

InvestedCapital (operating approach - the side the paper itself recommends):
  * Current assets - NIBCLs       -> CurrentAssets - (CurrentLiabilities - CurrentDebt)
  * - necessary-cash strip        -> compares CashCashEquivalentsAndShortTermInvestments
                                      (broad - matches the paper's "cash and
                                      marketable securities" language) against
                                      NECESSARY_CASH_PCT_OF_REVENUE x TotalRevenue;
                                      only the excess above that is stripped
  * + Net PP&E                    -> NetPPE
  * + Goodwill                    -> Goodwill
  * + Acquired intangibles, net   -> OtherIntangibleAssets
  * + Operating lease ROU asset   -> NOT AVAILABLE. Balance-sheet `Leases`
                                      field is ambiguous (likely mirrors
                                      CapitalLeaseObligations, a liability,
                                      not a ROU asset). Excluded, logged.
  * + Other long-term operating
      assets                      -> OtherNonCurrentAssets (catch-all; the
                                      paper's specific exclusions like
                                      non-consolidated subs/overfunded pension
                                      aren't separable from this field)

Every row's `adjustments_skipped` column names which of the above were left
out due to unavailable data - never silently defaulted to zero without saying
so.

Both the current-pipeline-style figures and the rebuilt figures are computed
from the SAME annual-period data in this script (unlike the live pipeline,
which uses quarterly TTM) so the comparison isolates the effect of formula
choice, not data-period differences.

Usage:
    python build_nopat_ic.py --tickers MSFT,AAPL,LULU
    python build_nopat_ic.py --input current_baseline_data.csv --output nopat_ic_comparison.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Optional

import pandas as pd
from yahooquery import Ticker

from data_fetch_utils import fetch_with_backoff, RateLimitExceeded, BASE_DELAY_SEC

# ---------------------------------------------------------------------------
# Named constants (documented judgment calls, not implicit)
# ---------------------------------------------------------------------------

# Paper's steady-state rule of thumb for necessary operating cash (up to 5%
# for high-growth names - not auto-applied here, single default for now).
NECESSARY_CASH_PCT_OF_REVENUE = 0.02

# US federal statutory rate, used only for the interest tax-shield leg of the
# cash-tax build (the paper wants a marginal/statutory rate here, not the
# reported effective rate).
STATUTORY_TAX_RATE = 0.21

ADJUSTMENTS_ALWAYS_SKIPPED = (
    "amortization_addback,lease_interest_addback,lease_rou_asset,"
    "tax_provision_unusual_item_adjustment"
)
# asset_impairment_addback is NOT in this list — it's applied (see
# compute_rebuilt_nopat), with the per-row dollar amount surfaced in the
# "asset_impairment_addback" output column so it's auditable, not hidden.


def _safe_num(value, default: float = 0.0) -> float:
    """Coerce a possibly-missing/NaN yahooquery field to *default*.

    `value or default` is NOT safe for this: float('nan') is truthy in
    Python, so a present-but-NaN field would silently pass through as NaN
    and contaminate downstream arithmetic instead of falling back cleanly.
    """
    if value is None or pd.isna(value):
        return default
    return float(value)


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------


def _fetch_annual_statements(symbol: str, delay_ref: List[float]):
    t = Ticker(symbol, asynchronous=False)
    inc = fetch_with_backoff(lambda: t.income_statement(frequency="a"), desc=f"{symbol} income", delay_ref=delay_ref)
    bs = fetch_with_backoff(lambda: t.balance_sheet(frequency="a"), desc=f"{symbol} balance sheet", delay_ref=delay_ref)
    cf = fetch_with_backoff(lambda: t.cash_flow(frequency="a"), desc=f"{symbol} cash flow", delay_ref=delay_ref)
    return inc, bs, cf


def _latest_12m_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if "periodType" in df.columns:
        df = df[df["periodType"] == "12M"]
    if df.empty or "asOfDate" not in df.columns:
        return None
    df = df.copy()
    df["asOfDate"] = pd.to_datetime(df["asOfDate"])
    return df.sort_values("asOfDate").iloc[-1]


def _latest_common_year(inc: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame):
    row_inc = _latest_12m_row(inc)
    row_bs = _latest_12m_row(bs)
    row_cf = _latest_12m_row(cf)
    if row_inc is None or row_bs is None or row_cf is None:
        return None, None, None

    # Align to the most recent fiscal year end common to all three statements.
    dates_inc = set(pd.to_datetime(inc[inc["periodType"] == "12M"]["asOfDate"])) if "periodType" in inc.columns else set()
    dates_bs = set(pd.to_datetime(bs[bs["periodType"] == "12M"]["asOfDate"])) if "periodType" in bs.columns else set()
    dates_cf = set(pd.to_datetime(cf[cf["periodType"] == "12M"]["asOfDate"])) if "periodType" in cf.columns else set()
    common = dates_inc & dates_bs & dates_cf
    if not common:
        return None, None, None
    year = max(common)

    row_inc = inc[(inc["periodType"] == "12M") & (pd.to_datetime(inc["asOfDate"]) == year)].iloc[0]
    row_bs = bs[(bs["periodType"] == "12M") & (pd.to_datetime(bs["asOfDate"]) == year)].iloc[0]
    row_cf = cf[(cf["periodType"] == "12M") & (pd.to_datetime(cf["asOfDate"]) == year)].iloc[0]
    return row_inc, row_bs, row_cf


# ---------------------------------------------------------------------------
# Formula building blocks (pure functions, unit-tested independently)
# ---------------------------------------------------------------------------


def compute_current_regime(row_inc: pd.Series, row_bs: pd.Series) -> dict:
    """Reproduce the live pipeline's formula: nopat = EBIT * (1 - TaxRateForCalcs),
    roic = nopat / InvestedCapital (yahooquery's own fields, unmodified)."""
    ebit = row_inc.get("EBIT")
    tax_rate = row_inc.get("TaxRateForCalcs")
    ic = row_bs.get("InvestedCapital")

    if pd.isna(ebit) or pd.isna(tax_rate) or pd.isna(ic) or ic == 0:
        return {"nopat_current": None, "invested_capital_current": None, "roic_current": None}

    nopat = ebit * (1 - tax_rate)
    return {
        "nopat_current": nopat,
        "invested_capital_current": ic,
        "roic_current": nopat / ic,
    }


def compute_cash_taxes(tax_provision: float, deferred_income_tax: float, net_interest_expense: float,
                        statutory_rate: float = STATUTORY_TAX_RATE) -> float:
    """cash_taxes = TaxProvision - DeferredIncomeTax + tax_shield.

    DeferredIncomeTax is SUBTRACTED: it is reported in the standard
    cash-flow-statement add-back convention (positive = book tax expense
    exceeded cash tax paid), the opposite sign from the paper's own Exhibit 2
    "deferred taxes" cash-tax-buildup line. See module docstring.
    """
    tax_shield = max(net_interest_expense, 0.0) * statutory_rate
    return tax_provision - deferred_income_tax + tax_shield


def compute_rebuilt_nopat(row_inc: pd.Series, row_cf: pd.Series) -> dict:
    operating_income = row_inc.get("OperatingIncome")
    tax_provision = row_inc.get("TaxProvision")
    deferred_income_tax = row_cf.get("DeferredIncomeTax")
    interest_expense = _safe_num(row_inc.get("InterestExpense"))
    interest_income = _safe_num(row_inc.get("InterestIncome"))
    # NaN here means "no impairment reported that year" (a legitimate zero,
    # confirmed against MSFT), unlike the NaN checks below which mean
    # "genuinely unknown" — see module docstring.
    asset_impairment_addback = _safe_num(row_cf.get("AssetImpairmentCharge"))

    if pd.isna(operating_income) or pd.isna(tax_provision) or pd.isna(deferred_income_tax):
        return {"nopat_rebuilt": None, "cash_taxes": None, "asset_impairment_addback": None}

    net_interest_expense = interest_expense - interest_income
    cash_taxes = compute_cash_taxes(tax_provision, deferred_income_tax, net_interest_expense)
    nopat = operating_income - cash_taxes + asset_impairment_addback
    return {
        "nopat_rebuilt": nopat,
        "cash_taxes": cash_taxes,
        "asset_impairment_addback": asset_impairment_addback,
    }


def compute_rebuilt_invested_capital(row_bs: pd.Series, row_inc: pd.Series,
                                      necessary_cash_pct: float = NECESSARY_CASH_PCT_OF_REVENUE) -> Optional[float]:
    current_assets = row_bs.get("CurrentAssets")
    current_liabilities = row_bs.get("CurrentLiabilities")
    current_debt = _safe_num(row_bs.get("CurrentDebt"))
    net_ppe = row_bs.get("NetPPE")
    goodwill = _safe_num(row_bs.get("Goodwill"))
    intangibles = _safe_num(row_bs.get("OtherIntangibleAssets"))
    other_lt_assets = _safe_num(row_bs.get("OtherNonCurrentAssets"))
    revenue = row_inc.get("TotalRevenue")

    actual_cash = row_bs.get("CashCashEquivalentsAndShortTermInvestments")
    if pd.isna(actual_cash):
        actual_cash = _safe_num(row_bs.get("CashAndCashEquivalents"))

    if pd.isna(current_assets) or pd.isna(current_liabilities) or pd.isna(net_ppe) or pd.isna(revenue):
        return None

    nibcls = current_liabilities - current_debt
    nwc = current_assets - nibcls

    necessary_cash = necessary_cash_pct * revenue
    excess_cash = max(actual_cash - necessary_cash, 0.0)
    nwc_adjusted = nwc - excess_cash

    return nwc_adjusted + net_ppe + goodwill + intangibles + other_lt_assets


def compute_rebuilt_regime(row_inc: pd.Series, row_bs: pd.Series, row_cf: pd.Series) -> dict:
    nopat_result = compute_rebuilt_nopat(row_inc, row_cf)
    ic = compute_rebuilt_invested_capital(row_bs, row_inc)

    nopat = nopat_result["nopat_rebuilt"]
    roic = (nopat / ic) if (nopat is not None and ic not in (None, 0)) else None

    return {
        "nopat_rebuilt": nopat,
        "invested_capital_rebuilt": ic,
        "roic_rebuilt": roic,
        "asset_impairment_addback": nopat_result.get("asset_impairment_addback"),
        "adjustments_skipped": ADJUSTMENTS_ALWAYS_SKIPPED,
    }


# ---------------------------------------------------------------------------
# Per-ticker orchestration
# ---------------------------------------------------------------------------


def process_ticker(symbol: str, delay_ref: List[float]) -> dict:
    try:
        inc, bs, cf = _fetch_annual_statements(symbol, delay_ref)
    except RateLimitExceeded:
        raise
    except Exception as exc:
        return {"symbol": symbol, "status": f"fetch_error: {exc}"}

    if not isinstance(inc, pd.DataFrame) or not isinstance(bs, pd.DataFrame) or not isinstance(cf, pd.DataFrame):
        return {"symbol": symbol, "status": "malformed_data"}

    row_inc, row_bs, row_cf = _latest_common_year(inc, bs, cf)
    if row_inc is None:
        return {"symbol": symbol, "status": "no_common_fiscal_year"}

    result = {"symbol": symbol, "status": "ok", "fiscal_year_end": str(pd.to_datetime(row_inc["asOfDate"]).date())}
    result.update(compute_current_regime(row_inc, row_bs))
    result.update(compute_rebuilt_regime(row_inc, row_bs, row_cf))

    if result.get("roic_current") is not None and result.get("roic_rebuilt") is not None:
        result["roic_delta"] = result["roic_rebuilt"] - result["roic_current"]
    else:
        result["roic_delta"] = None

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Compare current EBIT-field-based NOPAT/InvestedCapital/ROIC against a "
                    "rebuild from verified yahooquery building blocks (Mauboussin & Callahan "
                    "operating-approach methodology)."
    )
    p.add_argument("--tickers", type=str, help="Comma-separated tickers")
    p.add_argument("--input", type=pathlib.Path, help="CSV with a 'symbol' column (alternative to --tickers)")
    p.add_argument("--rate-limit", type=float, default=BASE_DELAY_SEC, help="Seconds to wait between API calls")
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("nopat_ic_comparison.csv"))
    args = p.parse_args()

    if args.tickers:
        tickers: List[str] = [s.strip() for s in args.tickers.split(",") if s.strip()]
    elif args.input:
        df_in = pd.read_csv(args.input, sep=";")
        tickers = df_in["symbol"].dropna().astype(str).unique().tolist()
    else:
        print("Provide --tickers or --input", file=sys.stderr)
        sys.exit(1)

    delay_ref = [float(args.rate_limit)]
    rows = []
    for sym in tickers:
        print(f"Processing {sym}...", flush=True)
        try:
            row = process_ticker(sym, delay_ref)
        except RateLimitExceeded:
            print("❌ Rate limit exceeded, stopping early", flush=True)
            break
        rows.append(row)

        if row.get("status") == "ok":
            print(
                f"  · roic_current={row['roic_current']:.2%}  roic_rebuilt={row['roic_rebuilt']:.2%}  "
                f"delta={row['roic_delta']:+.2%}"
                if row["roic_current"] is not None and row["roic_rebuilt"] is not None
                else "  · one or both ROIC values could not be computed (missing fields)"
            )
        else:
            print(f"  · {row.get('status')}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, sep=";", index=False)
    print(f"\n✓ Saved → {args.output.resolve()}")


if __name__ == "__main__":
    main()
