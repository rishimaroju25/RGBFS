# Data dictionary

**Owner:** Rishi (data pipeline) · **Last updated:** Week 2

Everything under `data/` is created by code and is not committed to GitHub (it is too large). Run the steps in **How to rebuild** below, or copy the team's shared `data/processed/` folder from Google Drive.

## How to rebuild

From the repo root, in this order:

| Step | Command | Creates |
|---|---|---|
| 1 | `python scripts/download_sp500_prices.py` | `data/raw/prices/`, `data/processed/sp500_*` |
| 2 | run `01_data_collection.ipynb` | `data/processed/daily_panel.parquet` |
| 3 | run `notebooks/stress_labels.ipynb` | `stress_labels_daily.parquet`, `docs/stress_periods.csv` |
| 4 | run `notebooks/04_returns_and_windows.ipynb` | `returns_*`, `features/connectedness_60d.parquet` |
| 5 | run `notebooks/02_sec_filings.ipynb` | `data/processed/sec/` |
| 6 | run `notebooks/13_anomaly_detection.ipynb` (after step 4) | `features/anomaly_network_daily.parquet`, `notebooks/outputs/rishi_week4/` |
| 7 | run `notebooks/14_anomaly_flags_in_models.ipynb` | `notebooks/outputs/rishi_week5/` |
| 8 | run `notebooks/15_alert_lead_time.ipynb` | `notebooks/outputs/rishi_week6/` (read by the dashboard alerts panel) |

**Keep `data/raw/prices/`.** Yahoo drops a symbol's history after a company is acquired or delisted (8 tickers already lost theirs, see `survivorship_bias.md`), so a fresh download later can permanently lose stocks.

## Stock prices and membership (`src/universe.py`, `src/prices.py`)

| File | Shape | Contents |
|---|---|---|
| `sp500_close.parquet` | trading days x tickers | Split- and dividend-adjusted close from Yahoo Finance, 2005 onward. Includes `^GSPC` (index level) and `SPY`. **Not masked**: has prices on days a stock was not in the index |
| `sp500_volume.parquet` | trading days x tickers | Daily share volume |
| `sp500_membership.parquet` | trading days x tickers | `True` if the ticker was in the S&P 500 that day (point-in-time, from fja05680/sp500) |
| `sp500_coverage_report.csv` | one row per ticker | `member_days`, `priced_member_days`, `coverage`, first and last price date, `no_data`, `late_start`, `check_ticker_reuse`, `prices_from` (renamed ticker filled from its current symbol) |

## Clean returns (`src/returns.py`, Week 2)

| File | Shape | Contents |
|---|---|---|
| `returns_daily.parquet` | trading days x tickers | Daily **simple** returns, `close_t / close_t-1 - 1`. NaN on every day the stock was not in the index, and on removed days (rules below) |
| `returns_market.parquet` | trading days x 4 | `^GSPC_ret`, `SPY_ret`, `ew_member_ret` (equal-weighted average of member returns), `n_returns` (stocks with a return that day) |
| `returns_quality.csv` | one row per removed or flagged return | `date`, `ticker`, `raw_return`, `reason` |
| `returns_summary.csv` | measure, value | Counts for each rule |

**Cleaning rules**, in order:

1. **Member days only.** Returns are computed from the full price history, then kept only on days the stock was in the index. A stock's first day in the index has a real return (its price the day before joining is known).
2. **No filling across gaps.** If yesterday's price is missing, today's return is NaN, not a multi-day return.
3. **Stale prices:** 5 or more identical closes in a row on member days are removed (`stale_removed`).
4. **Spikes:** a one-day move above 40% that is back within 10% of its starting price on the **next** trading day is a data blip; both legs are removed (`spike_removed`). Slower reversals and moves that do not reverse are kept (`large_move_kept`). A 3-day version of this rule was tried first and wrongly removed AIG's real moves during its September 2008 bailout.
5. **Repeated jumps:** a ticker with 3+ moves above 40% within a year is listed as `repeated_jumps_review` for a manual check (usually a reused symbol). Not removed automatically.
6. **Overrides:** `docs/return_overrides.csv` (ticker, start, end, action, note) wins over every rule. Current entries: CPWR and EP dropped (bad prices found in the 2011 EDA), and the eight large 2020 and 2022 moves the team confirmed as real.

Why simple returns: the cross-sectional measures (correlation, dispersion) are nearly identical with log returns at daily frequency, and simple returns average correctly across stocks for the equal-weighted portfolio.

## Rolling windows (`src/windows.py`, Week 2)

| Function | Use |
|---|---|
| `windows(returns, window=60, step=5)` | Yields each trailing window (end date, start date, block of returns) |
| `correlation_at(returns, date, window=60)` | Stocks x stocks correlation matrix for one window (network prototype) |
| `rolling_apply(returns, func, window, step)` | Runs any `func(block) -> dict` on every window; one row per end date |
| `connectedness(returns, window, step)` | `avg_corr`, `pc1_share`, `absorption`, `n_stocks` per window |

Rules: windows are **trailing** (a value dated *t* uses days up to *t* only); a stock is used only if it has a return on **every** day of the window (so stocks entering or leaving the index mid-window are excluded); windows with fewer than 100 such stocks are skipped. The tests in `tests/test_returns_windows.py` check that changing future returns never changes past feature values.

Saved feature: `data/processed/features/connectedness_60d.parquet` (60-day windows every 5 trading days, 2005 onward).

## Clusters (`src/clustering.py`, Week 3)

| File | Contents |
|---|---|
| `features/clusters_market_adjusted_20d.parquet` | Every 20 trading days: n_stocks, n_nodes, n_edges, density, n_communities, modularity, largest_share, sector_ari, stability (ARI vs the previous window). Built from residual correlations after removing the equal-weighted market move, strongest 5% of positive links, weighted Louvain (seed 2302) |
| `features/cluster_memberships_market_adjusted_20d.parquet` | date, ticker, community |

Built by `notebooks/08_cluster_stability.ipynb`, which also compares this setting with Grace's original one.

## Centrality concentration (`src/centrality.py`, Week 4)

| File | Contents |
|---|---|
| `features/centrality_60d.parquet` | Every 5 trading days (same dates as `connectedness_60d`): strength_gini, strength_top10, degree_central, evc_top10, raw_evc_eff_share, n_stocks. Same graph as the cluster features (market-adjusted residual correlations, strongest 5% of positive links), computed over every stock in the window including unlinked ones. `raw_evc_eff_share` uses the raw correlation matrix and is almost the same as `avg_corr` (rank correlation 0.95), so do not use both as model features |
| `features/centrality_hubs_60d.parquet` | date, rank, ticker, strength for the 10 most connected stocks in each window |

Built by `notebooks/10_centrality_concentration.ipynb`.

## Anomalies and alerts (`src/anomaly.py`, `src/alerts.py`, Weeks 4 to 6)

| File | Contents |
|---|---|
| `features/anomaly_network_daily.parquet` | One row per trading day. `z_<metric>`: trailing one-year z-score of avg_corr, pc1_share, absorption and the 20-day changes in avg_corr and absorption (current window excluded from its own baseline). `z_composite`: mean z-score signed toward stress. `z_flag`: any metric z >= 2. `if_score`, `if_flag`: walk-forward Isolation Forest (refit yearly on past rows; threshold = 95th percentile of training scores). `if_flag_stress`: IF flag only when z_composite > 0. `alert`: z_flag or if_flag_stress. `asof`: date of the window the row comes from (carried forward at most 10 trading days). NaN during warm-up (about 1 year for z-scores, 3 years for IF) |
| `notebooks/outputs/rishi_week5/model_alerts_daily.csv` | Walk-forward logistic regression, target = an episode starts within the next 20 trading days (days inside episodes excluded). `<model>|prob`, `<model>|alert` (above the 90th percentile of training probabilities) for baseline (VIX, 20-day VIX change, 20-day realized vol), + network z-scores, + Isolation Forest score |
| `notebooks/outputs/rishi_week6/signals_daily.csv` | 0/1 alert per signal per day: network anomaly, network z-score only, VIX > 25, VIX z >= 2, STLFSI4 > 0 and > 1 (shifted 4 trading days for publication lag), and the two model alerts |
| `notebooks/outputs/rishi_week6/signal_summary.csv` | Per signal and per start date (`start`, `start_trailing`): episodes warned early / late / missed, median lead (trading days), false alarm events per year, day-level precision, recall and false-alarm rate |
| `notebooks/outputs/rishi_week6/lead_times_by_episode_*.csv` | Lead time in trading days for every episode and signal. Window: 60 trading days before the start (never before the previous episode's end) to 10 after |

The stress labels are only used to score alerts, never as inputs.

## Market and stress series

| File | Contents |
|---|---|
| `daily_panel.parquet` | FRED series on the trading-day calendar: `STLFSI4` and `NFCI` (weekly, forward-filled by observation date; **shift by the release lag before modeling**: `src.walk_forward.lag_by_release` uses 5 trading days, `src/alerts.py` uses 4 for STLFSI4), `VIXCLS`, `BAMLH0A0HYM2` (only from late 2023 in our download), `T10Y2Y`; SPY and sector ETF prices and log returns; `avg_corr_60d` and `absorption_250d` (sector ETFs, trailing). **`stress` is an old placeholder (STLFSI4 > 1, not lagged), not the official label: do not use it; use `stress_labels_daily.parquet`** |
| `stress_labels_daily.parquet` | `vol_centered`, `stress_day_raw`, `episode_id`, `stress` (1 inside an episode). **Uses future data by design: evaluation target only, never a feature** |
| `docs/stress_periods.csv` | One row per episode: start (centered), start_trailing, end, peak volatility, drawdown, name |

## Text (`src/sec_filings.py`)

| File | Contents |
|---|---|
| `sec/filings.csv` | One row per filing: ticker, CIK, form, filing date, accession, document |
| `sec/paragraphs.parquet` | One row per cleaned paragraph: ticker, form, filing_date, accession, paragraph_id, section (Item 1A, Item 7, ...), text |

## Known gaps

* About 23% of member-days have no price (62% coverage in 2008, 92% in 2022), mostly firms that failed or were acquired. Stress in 2008 is likely understated. Details in `survivorship_bias.md`.
* Failed firms' final "delisting" return is never in Yahoo data.
* Text covers 5 technology firms in 2022 so far.
