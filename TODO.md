# ROIIC improvement backlog

Context: `compute_roiic.py`'s ROIIC regression looked weak in the SEC-EDGAR
point-in-time backtest (`sec_edgar_backtest.py`, `--years-ago 5`). Diagnosis
and a 5-step improvement plan came out of that review; steps 1, 3, and 4 are
built. Steps 2/5 are parked here for later — not abandoned, just not proven
to move the needle yet (see "why paused" below).

Backtest scorecard across all three built fixes (5yr anchor, n=32 clean
names; none of these gaps clear statistical significance — see the harness
power-limit note below, don't over-read the ranking):

| Version | qual-vs-disqual gap | p-value | ROIIC computable | clamped |
|---|---|---|---|---|
| Original | +101.1% | 0.308 | 29/32 | 12/29 (41%) |
| +#1 normalize | +80.8% | 0.393 | 29/32 | 13/29 (45%) |
| +#1+#4 buyback | +103.9% | 0.269 | 31/32 | 2/31 (6%) |
| +#1+#4+#3 Theil-Sen | +128.3% | 0.197 | 31/32 | 4/31 (13%) |

## Done

- **#1 — Normalize one-time items in the annual EBIT/tax series before the
  regression.** Implemented as `normalize_annual_series()` in
  `compute_roiic.py`, wired into `compute_nopat_and_invested_capital()`
  (`normalize=True` by default) and mirrored in `sec_edgar_backtest.py`'s
  `process_ticker()` so both the live pipeline and the backtest stay
  consistent. Flags TotalUnusualItems and R&D/SG&A/restructuring spikes
  >2x trailing baseline; corrects the tax rate when PretaxIncome went
  negative. Verified against real filings (MRK's IPR&D charge, GE's 2023
  one-time gain, a PG data-quality bug it also caught). All existing unit
  tests still pass.

- **#4 — Buyback-adjusted (organic) InvestedCapital.** A share buyback
  shrinks equity/cash with zero effect on the operating business, which
  distorted the ΔIC trend the regression relies on. Implemented as
  `apply_buyback_addback()` in `compute_roiic.py`: adds back cumulative
  `RepurchaseOfCapitalStock` (yahooquery) / `PaymentsForRepurchaseOfCommonStock`
  (SEC) to InvestedCapital before it feeds `compute_roiic_slope()` — applied
  **only** to the regression window, deliberately not to the anchor-year
  ROIC/excessReturn snapshot (a bug where it leaked into the snapshot was
  caught and fixed in `sec_edgar_backtest.py`: it had inflated AAPL's
  anchor InvestedCapital to $512B, cutting its real ~42% ROIC down to a
  fake ~11%). Wired into `compute_roiic.py`'s live batch path (added a
  `cash_flow` fetch alongside income/balance) and mirrored in
  `sec_edgar_backtest.py`. Verified against real filings: AAPL went from
  an unmeasurable ROIIC (flat IC, masked by ~$90B/year buybacks) to a real
  +5.3%; GE went from a strong -29.6% (partly buyback, partly unrelated
  spinoff noise this fix doesn't touch) to unmeasurable, which is the more
  honest outcome given the fix only corrects the buyback piece. In the
  5-year backtest, ROIIC's ±40% clamp rate dropped from 41-45% of computed
  values down to 6% — the clearest evidence either fix worked, since the
  aggregate qualifiers-vs-disqualified return gap remains statistically
  insignificant either way (p≈0.27-0.39 throughout, per the harness
  power-limit note below) and shouldn't be over-read turn to turn.
  Considered and rejected for now: rebuilding InvestedCapital from the
  operating/asset side (PP&E + working capital) — more comprehensive (also
  fixes M&A distortion) but would force `current_baseline_data.py`'s ROIC
  snapshot to change definition too for consistency; bigger blast radius
  than warranted right now.

- **#3 — Theil-Sen instead of OLS for the regression slope.** Swapped
  `stats.linregress` for `stats.theilslopes` in `compute_roiic_slope()`
  (both `nopat` and `InvestedCapital` vs. `year`) - same target quantity as
  before (marginal NOPAT per unit of marginal InvestedCapital), just the
  median of every pairwise slope instead of a squared-error-minimizing fit,
  so a single noisy year can't dominate the estimate. Chosen deliberately
  as the *primary* estimator, not just a diagnostic, after concluding it's
  actually more explainable than OLS once broken down: a pairwise slope
  between two years literally *is* two-point incremental ROIC (a concept
  already used throughout this project), whereas OLS's "minimize squared
  error" doesn't map to any financial intuition - the fear that Theil-Sen
  is "abstract" was really unfamiliarity with the method, not the concept
  underneath it. `sec_edgar_backtest.py` picked this up automatically
  (imports `compute_roiic_slope` directly, no separate wiring needed).
  Verified against GE: previously -8.2% (OLS+buyback-adjusted), now +3.1%
  (Theil-Sen) - a concrete example of the median-of-pairwise-slopes
  correctly down-weighting one outlier year-pair that was dominating the
  OLS fit. All existing unit tests still pass.

## Backlog (not built)

- **#2 — Confidence flag alongside `growthGate`.** Surface whether the
  ROIIC regression hit the ±40% winsorization clamp (or expose the spread
  between the low/high Theil-Sen slope bounds `stats.theilslopes` already
  returns - free extra info now that we're using it), so a noisy/clamped
  reading can be weighted down instead of trusted at face value. Cheap to
  add - do it whenever the clamp rate (now 4/31, 13%) becomes annoying.

- **#5 — Lag InvestedCapital vs. NOPAT in the regression.** Fit
  InvestedCapital(t-1) → NOPAT(t) instead of the current same-year fit,
  since new investment typically takes 1-3 years to mature. Cheap (an
  8-year series still gives 7 lagged pairs), but a single fixed lag is a
  simplification — the right lag varies by business (near-zero for
  software, multi-year for capital-heavy industrials/semis).

- **Multi-anchor/larger-universe backtest harness.** Needed before any of
  #2-5 above can be judged by backtest results rather than by
  spot-checking against filings. Currently `sec_edgar_backtest.py` tests
  exactly one anchor date against ~40 hand-picked tickers (n≈32 clean) —
  a power calculation off the observed return variance (qualifiers' fwd
  returns had a 372% std dev, driven by outliers like NVDA +1601%) shows
  ~484 names needed to detect a 100pp gap at 80% power, ~1,930 for a more
  modest 50pp gap. Building a harness that could actually resolve "did
  this fix help" means scaling toward full-Russell-1000-across-multiple-
  anchors, which reintroduces SEC rate limits, XBRL tag-inconsistency
  dropout (~15-20%), and InvestedCapital-denominator artifacts at 10-25x
  today's scale — a large, dedicated build, not a quick follow-up.
  **Until this exists, judge #2-5 by verifying against real company
  filings (as done for #1), not by backtest score.**

## Reference

- `normalize_ebit.py` — standalone one-time-item checker for the trailing
  TTM EBIT feeding `current_baseline_data.py`'s ROIC/excessReturn (the
  quarterly counterpart to `normalize_annual_series()`'s annual/ROIIC use).
- `sec_edgar_backtest.py` — point-in-time backtest via SEC EDGAR XBRL data
  (deeper history than yahooquery's ~5yr rolling window allows).
