# Proposal: what counts as a stress period

**Owner:** Salah · **Status:** proposed, needs team agreement (Week 1) · **Code:** `src/stress_labels.py`, `notebooks/stress_labels.ipynb` · **Table:** `docs/stress_periods.csv`

## The rule

| Step | Proposal |
|---|---|
| Measure | Realized volatility of the S&P 500 index (`^GSPC`): standard deviation of daily returns over a **centered 21-trading-day window** (10 days before, the day itself, 10 days after), annualized |
| Stress day | Volatility **at least 25%** |
| Episode | Stress days less than **20 trading days** apart are merged into one episode |
| Minimum length | Episodes shorter than **10 trading days** are dropped |

On data from 2005-01-03 to 2026-10-26 this gives **13 episodes** covering 11.5% of labeled days (table in `docs/stress_periods.csv`). Every key date from the four EDA periods falls inside an episode: Lehman (2008-09-15), the first trading day after the S&P downgrade (2011-08-08), the largest 2020 daily fall (2020-03-16), the day the S&P 500 entered a bear market (2022-06-13) and the 2022 closing low (2022-10-12).

## Why these choices

**Built from index prices only, not VIX or STLFSI4.** The project question is whether our signals warn *earlier than standard stress indexes*. If the labels came from VIX or STLFSI4, those indexes would match the labels by construction and the Week 5/6 comparison would be circular. We use VIX and STLFSI4 only to *validate* the episodes (Section 5 of the notebook, once `daily_panel.parquet` is available).

**Realized volatility.** It is the same quantity Grace's forecasting target uses (future realized volatility), so the labels and the target measure the same thing.

**A fixed 25% level, not a percentile.** A percentile of the full sample changes every time we add data, which would move past labels. A fixed level does not. For scale, 25% is about twice the 2005 to 2026 median (12.9%) and close to the 90th percentile (26.5%).

**A centered window, with both start dates reported.** A trailing 20-day window starts every episode about **10 trading days late** (checked for all 13 episodes in Section 3 of the notebook). That delay would add roughly two weeks of fake lead time to every alert in Rishi's Week 6 lead-time test. The centered window has the **opposite bias**: because it looks 10 days ahead, an episode can start up to 10 trading days *before* volatility actually rises. For example, the COVID-19 episode starts on 2020-02-14, three trading days before the S&P 500's pre-crash closing high on 2020-02-19 (`notebooks/eda_2020.ipynb`, Section 3). An alert during those days counts as "inside the episode" rather than as a lead, so the centered start makes measured lead time look **shorter** than it is.

Neither start is "the" right one: the centered start is an early bound and the trailing start a late bound. `docs/stress_periods.csv` therefore has both (`start` and `start_trailing`, 9 to 10 trading days apart), and the Week 6 lead-time test should report lead time against both. For 2011 the bounds are 2011-07-25 (first trading day after the debt talks collapsed) and 2011-08-08 (first trading day after the downgrade).

**Merge gap and minimum length.** These stop one crisis from splitting into many short episodes and remove one-week blips. Changing either one moves the count by only one episode (Section 3).

## Rules for using the labels

1. **The labels use future data by design.** The centered window looks 10 trading days ahead. Labels are evaluation targets only: never use `stress`, `vol_centered`, `episode_id` or anything derived from them as a model feature. This goes on the lookahead checklist (Week 5).
2. The first and last 10 trading days of the data cannot be labeled and are left missing. Do not treat them as calm days.
3. Episode names in the table are descriptive. Verify them against a primary source before citing them in the report.

## Sensitivity

| Variant | Episodes | Share of days labeled stress |
|---|---|---|
| **Proposed (centered 21d, 25%)** | **13** | **11.5%** |
| Threshold 20% | 17 | 22.4% |
| Threshold 30% | 8 | 7.3% |
| Trailing 20d window, 25% | 14 | 11.5% (each episode starts ~10 days later) |
| Merge gap 10 days | 14 | 11.3% |
| Min length 5 days | 14 | 11.7% |

At 30% the rule drops the 2007, Bear Stearns, Feb 2018 and Feb–Mar 2022 episodes. At 20% it doubles the stress share and adds several mild episodes.

## Decisions needed from the team

1. **Agree to report lead time against both start dates** (centered and trailing), as described above.
2. **Agree on the 25% threshold**, or pick 30% if we want only severe episodes. 25% gives Week 6 more episodes to measure lead time and false alarms on.
3. **2022 as one episode or three?** The rule gives three (Feb–Mar, Apr–Jun, Sep–Nov). Keeping three is consistent with the rule; merging them would need a longer merge gap for every period.
4. **2011 framing.** The rule labels 2011-07-25 to 2011-12-08 as one episode covering both the U.S. debt ceiling and downgrade and the euro-area debt crisis. The 2011 EDA notebook uses the same framing.
5. **Benchmark validation.** `01_data_collection.ipynb` could not run before (its loader file was in the wrong folder; fixed). Done (2026-10-28 run): average VIX is 35.6 inside episodes vs 17.0 outside, every episode's peak VIX is above 30, and STLFSI4 averages 1.84 inside vs -0.28 outside. Five episodes (2015, Feb 2018, Q4 2018, Feb-Mar 2022 and 2025) never reach STLFSI4 = 1, so the `STLFSI4 > 1` placeholder in `01_data_collection.ipynb` would miss them.
