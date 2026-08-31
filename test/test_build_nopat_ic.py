#!/usr/bin/env python3
"""Golden-number regression tests for build_nopat_ic.py.

Seeded with Mauboussin & Callahan's own worked Microsoft FY2022 exhibits
(Exhibit 2 for cash taxes, Exhibit 4 for invested capital) so the formulas
are verified against the paper's own arithmetic, independent of live data
drift — yahooquery no longer retains MSFT's FY2022 figures (confirmed live
in this session), so this can't be checked against live data directly.

Where this repo's build deliberately omits a paper adjustment (amortization
of acquired intangibles, operating-lease interest/ROU asset — both
unavailable via yahooquery, see build_nopat_ic.py's module docstring), the
expected values below are the paper's own numbers MINUS that omitted piece,
not the paper's full headline number. Comments call out the delta in each
case so the gap is documented, not hidden.
"""

import unittest

import pandas as pd

import build_nopat_ic as m


class TestComputeCashTaxes(unittest.TestCase):
    def test_matches_paper_exhibit2_msft_fy2022(self):
        # Paper's Exhibit 2 (MSFT FY2022, $ billions): tax provision 11,
        # deferred taxes +6 (ADDED in the paper's own layout), tax shield 0,
        # cash taxes = 11 + 6 + 0 = 17.
        #
        # This repo's compute_cash_taxes takes `deferred_income_tax` in the
        # standard cash-flow-statement add-back convention, which is the
        # OPPOSITE sign from the paper's own "deferred taxes" cash-tax-buildup
        # line (verified empirically against MSFT's live operating-cash-flow
        # identity in this session) — so reproducing the paper's +6 effect
        # here means passing deferred_income_tax=-6, since the function
        # SUBTRACTS it: cash_taxes = provision - deferred_income_tax + shield.
        cash_taxes = m.compute_cash_taxes(
            tax_provision=11e9,
            deferred_income_tax=-6e9,
            net_interest_expense=0.0,
        )
        self.assertAlmostEqual(cash_taxes, 17e9, delta=1e6)

    def test_tax_shield_only_applies_to_net_interest_expense_not_income(self):
        # A company with net interest INCOME (negative net interest expense)
        # should get no tax shield — max(..., 0) clips it.
        cash_taxes_with_income = m.compute_cash_taxes(
            tax_provision=100.0, deferred_income_tax=0.0, net_interest_expense=-50.0
        )
        self.assertEqual(cash_taxes_with_income, 100.0)

        cash_taxes_with_expense = m.compute_cash_taxes(
            tax_provision=100.0, deferred_income_tax=0.0, net_interest_expense=50.0,
            statutory_rate=0.21,
        )
        self.assertAlmostEqual(cash_taxes_with_expense, 100.0 + 50.0 * 0.21)


class TestComputeRebuiltNopat(unittest.TestCase):
    def test_matches_paper_cash_taxes_but_excludes_ebita_addback(self):
        # Paper's Exhibit 2 MSFT FY2022: EBIT (operating income) = 83,
        # EBITA = 87 (EBIT + amortization 2 + lease payments 1, +1 rounding
        # in the paper's own table), cash taxes = 17, headline NOPAT = 70.
        #
        # This build uses OperatingIncome (83) directly as the EBIT base and
        # does NOT add back amortization/lease interest (unavailable via
        # yahooquery, see adjustments_skipped) — so the expected NOPAT here
        # is 83 - 17 = 66, not the paper's headline 70. The 4 (=70-66)
        # difference is exactly the excluded EBITA add-back, not a bug.
        row_inc = pd.Series({
            "OperatingIncome": 83e9,
            "TaxProvision": 11e9,
            "InterestExpense": 0.0,
            "InterestIncome": 0.0,
        })
        row_cf = pd.Series({"DeferredIncomeTax": -6e9})

        result = m.compute_rebuilt_nopat(row_inc, row_cf)
        self.assertAlmostEqual(result["cash_taxes"], 17e9, delta=1e6)
        self.assertAlmostEqual(result["nopat_rebuilt"], 66e9, delta=1e6)

    def test_missing_required_field_returns_none_not_zero(self):
        row_inc = pd.Series({"OperatingIncome": 83e9, "TaxProvision": float("nan"),
                              "InterestExpense": 0.0, "InterestIncome": 0.0})
        row_cf = pd.Series({"DeferredIncomeTax": -6e9})
        result = m.compute_rebuilt_nopat(row_inc, row_cf)
        self.assertIsNone(result["nopat_rebuilt"])

    def test_asset_impairment_charge_is_added_back(self):
        # WBD's real FY2024 AssetImpairmentCharge ($9.603B), cross-checked
        # this session against their reported $9.1B Networks-segment
        # goodwill impairment (see build_nopat_ic.py docstring). Confirms
        # the add-back is wired in and the dollar amount is surfaced, not
        # just silently folded into NOPAT.
        row_inc = pd.Series({
            "OperatingIncome": 10e9, "TaxProvision": 1e9,
            "InterestExpense": 0.0, "InterestIncome": 0.0,
        })
        row_cf = pd.Series({"DeferredIncomeTax": 0.0, "AssetImpairmentCharge": 9.603e9})

        result = m.compute_rebuilt_nopat(row_inc, row_cf)
        self.assertAlmostEqual(result["asset_impairment_addback"], 9.603e9, delta=1e6)
        # nopat = operating_income - cash_taxes + addback = 10e9 - 1e9 + 9.603e9
        self.assertAlmostEqual(result["nopat_rebuilt"], 18.603e9, delta=1e6)

    def test_no_impairment_year_defaults_cleanly_to_zero(self):
        # NaN/missing AssetImpairmentCharge means "no impairment reported
        # that year" (confirmed against MSFT, which returns NaN here) - a
        # legitimate zero, not missing data, so this must NOT return None.
        row_inc = pd.Series({
            "OperatingIncome": 10e9, "TaxProvision": 1e9,
            "InterestExpense": 0.0, "InterestIncome": 0.0,
        })
        row_cf = pd.Series({"DeferredIncomeTax": 0.0})  # no AssetImpairmentCharge key at all

        result = m.compute_rebuilt_nopat(row_inc, row_cf)
        self.assertEqual(result["asset_impairment_addback"], 0.0)
        self.assertAlmostEqual(result["nopat_rebuilt"], 9e9, delta=1e6)


class TestComputeRebuiltInvestedCapital(unittest.TestCase):
    def test_matches_paper_exhibit4_msft_fy2022_excluding_lease_rou(self):
        # Paper's Exhibit 4 (MSFT FY2022, $ billions, operating approach):
        # current assets 69 (cash already set to 2% of revenue = 4, per the
        # exhibit's own footnote), NIBCLs 92, current portion of LT debt 3
        # (from the financing side) => current liabilities = 92 + 3 = 95.
        # NWC = 69 - 95 - (-3) = 69 - 92 = -23 (matches the exhibit exactly).
        # + PP&E 74, + operating lease ROU 13, + goodwill 68,
        # + intangibles 11, + other 22 = 165 (the exhibit's headline total).
        #
        # This build has no ROU-asset field (unavailable, see
        # adjustments_skipped), so the expected total here is 165 - 13 = 152.
        #
        # Since the exhibit's own "current assets" already reflects the
        # necessary-cash assumption (not full actual cash), actual_cash is
        # fed here equal to necessary_cash so the strip is a documented
        # no-op for this test — see test_excess_cash_is_actually_stripped
        # below for a case where the strip has a real effect.
        row_bs = pd.Series({
            "CurrentAssets": 69e9,
            "CurrentLiabilities": 95e9,
            "CurrentDebt": 3e9,
            "NetPPE": 74e9,
            "Goodwill": 68e9,
            "OtherIntangibleAssets": 11e9,
            "OtherNonCurrentAssets": 22e9,
            "CashCashEquivalentsAndShortTermInvestments": 4e9,  # == necessary cash below
        })
        row_inc = pd.Series({"TotalRevenue": 198e9})  # 2% x 198 ~= 4, matching the exhibit's own footnote

        ic = m.compute_rebuilt_invested_capital(row_bs, row_inc)
        self.assertAlmostEqual(ic, 152e9, delta=1e8)

    def test_excess_cash_is_actually_stripped(self):
        # Clean synthetic numbers (not tied to the paper) to validate the
        # necessary-cash strip actually reduces NWC when actual cash exceeds
        # the necessary-cash threshold.
        row_bs = pd.Series({
            "CurrentAssets": 1000.0,   # includes 300 of actual cash
            "CurrentLiabilities": 400.0,
            "CurrentDebt": 0.0,
            "NetPPE": 500.0,
            "Goodwill": 0.0,
            "OtherIntangibleAssets": 0.0,
            "OtherNonCurrentAssets": 0.0,
            "CashCashEquivalentsAndShortTermInvestments": 300.0,
        })
        row_inc = pd.Series({"TotalRevenue": 1000.0})  # necessary cash = 2% x 1000 = 20

        ic = m.compute_rebuilt_invested_capital(row_bs, row_inc)
        # nwc = 1000 - 400 = 600; excess_cash = 300 - 20 = 280; nwc_adj = 320
        # ic = 320 + 500 = 820
        self.assertAlmostEqual(ic, 820.0, delta=1e-6)

    def test_missing_required_field_returns_none(self):
        row_bs = pd.Series({"CurrentAssets": float("nan")})
        row_inc = pd.Series({"TotalRevenue": 1000.0})
        self.assertIsNone(m.compute_rebuilt_invested_capital(row_bs, row_inc))


if __name__ == "__main__":
    unittest.main()
