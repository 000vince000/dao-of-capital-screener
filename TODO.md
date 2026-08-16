# ROIIC improvement backlog

Context: `compute_roiic.py`'s ROIIC regression looked weak in the SEC-EDGAR
point-in-time backtest (`sec_edgar_backtest.py`, `--years-ago 5`). Diagnosis
and a 5-step improvement plan came out of that review; only step 1 has been
built. Steps 2-5 are parked here for later — not abandoned, just not proven
to move the needle yet (see "why paused" below).

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

## Backlog (not built)

- **#2 — Confidence flag alongside `growthGate`.** Surface whether the
  ROIIC regression hit the ±40% winsorization clamp (or expose R²), so a
  noisy/clamped reading can be weighted down instead of trusted at face
  value. Cheap to add once the others are in — do it last.

- **#3 — Theil-Sen instead of OLS for the regression slope.** Same target
  quantity as today (marginal NOPAT per unit of marginal InvestedCapital),
  just estimated as the median of pairwise slopes instead of least-squares
  — far less sensitive to a single outlier year. `scipy.stats.theilslopes`
  is close to a drop-in replacement for the existing `stats.linregress`
  calls in `compute_roiic_slope()`.

- **#4 — Buyback-adjusted (organic) InvestedCapital.** A share buyback
  shrinks equity/cash with zero effect on the operating business, which
  distorts the ΔIC trend the regression relies on. Fix: add back
  cumulative `RepurchaseOfCapitalStock` (SEC) / equivalent cash-flow field
  (yahooquery) to the equity series before computing InvestedCapital per
  year, rather than the current all-or-nothing `ΔIC/IC₀ < 10%` filter that
  just drops capital-light/buyback-heavy names (IT, NTAP, BBWI, WMT, PFE
  all lost their ROIIC signal entirely this way in testing). Considered and
  rejected for now: rebuilding InvestedCapital from the operating/asset
  side (PP&E + working capital) — more comprehensive (also fixes M&A
  distortion) but would force `current_baseline_data.py`'s ROIC snapshot
  to change definition too for consistency; bigger blast radius than
  warranted right now.

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
