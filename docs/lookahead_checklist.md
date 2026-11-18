# Lookahead-bias checklist

**Owner:** Salah · **Week 5** · Code: `src/walk_forward.py` · Evidence: `notebooks/13_lookahead_check.ipynb`

Our system is meant to warn at the close of day *t* using only what was published by then. Any feature, label or split that uses later information makes backtest results look better than a live system could do. Go through this list before you add a feature or report a model result.

## Features

- [ ] **Trailing windows only.** Rolling calculations end on day *t* (`rolling(...)` is trailing by default). Never use `center=True`, `shift(-k)` or full-sample statistics in a feature. Use `src.windows` for anything computed on stock windows.
- [ ] **Pass the truncation test.** Recompute the feature with all data after *t* deleted; the value at *t* must not change. Section 1 of `13_lookahead_check.ipynb` shows how.
- [ ] **Delay late-published data.** `STLFSI4` and `NFCI` are dated Fridays but published the next Wednesday or Thursday. Use `wf.lag_by_release(panel)` (5 trading days). Do the same for any new FRED or economic series: check its release day.
- [ ] **Text data uses the day after filing.** SEC filings map to the first trading day on or after the filing date, then shift one more trading day (filings after the 4 p.m. close keep that day's date).
- [ ] **Join snapshots backward.** For features computed every 5 or 20 days, use `pd.merge_asof(..., direction="backward")` so a date only sees snapshots dated on or before it.
- [ ] **No full-sample scaling or thresholds.** Z-scores, percentiles, PCA and anomaly thresholds must use past data only (rolling or expanding), or be fit inside each training fold.
- [ ] **Today's universe, not today's survivors.** Use point-in-time membership (`sp500_membership.parquet`); never pick companies or sectors because of how they did later.

## Labels and targets

- [ ] **Stress labels are targets only.** `stress`, `vol_centered`, `episode_id` and `stress_day_raw` look 10 trading days ahead. Never use them, or anything derived from them, as a feature.
- [ ] **Do not use `daily_panel["stress"]`.** It is an old placeholder (STLFSI4 > 1, not lagged), not the official label.
- [ ] **Record how far each label reaches.** `wf.stress_target(stress, horizon)` gives `label_end`: the last day whose returns the label uses (*t* + horizon + 10).
- [ ] **Early warning means a future target.** Predict stress in days *t*+1 to *t*+*h* and drop days already in stress (`in_stress == 0`). A same-day target only measures detection.

## Splits and evaluation

- [ ] **Walk forward, never shuffle.** Use `wf.walk_forward_splits`: train on everything before a test year, test on that year, refit, move on. No random or k-fold splits.
- [ ] **Purge overlapping labels.** Drop training rows whose `label_end` is on or after the test start (the splitter does this). Check with `wf.check_folds`.
- [ ] **Fit everything inside the fold.** Scalers, feature selection, thresholds and hyperparameters are chosen on training data only. A `make_pipeline(StandardScaler(), model)` refit per fold is fine.
- [ ] **Choose the alert threshold before the test year.** Pick it on training data (or fix it in advance), not on the test results.
- [ ] **Compare models on the same test days.** Same folds, same rows, same metrics.
- [ ] **Report the base rate next to PR-AUC.** A PR-AUC only means something relative to the share of positive days.

## Known limitations we accept (state them in the report)

- Return cleaning (`src/returns.py`) uses a day or two of future data to spot data errors (spikes that reverse next day, stale prices) and manual overrides chosen from the full history; it removes 0.215% of member-day returns.
- FRED serves revised history, not the values first published. NFCI is re-estimated weekly. A stricter test would use ALFRED vintages.
- Episode names and the 25% stress threshold were chosen after looking at the full history.
