"""Anomaly detection on network (connectedness) metrics, without looking ahead.

Owner: Rishi (Anomaly detection lead). Week 4 prototype, used in Weeks 5 to 7.

Input: a table of network metrics with one row per window end date, such as
data/processed/features/connectedness_60d.parquet (avg_corr, pc1_share,
absorption, n_stocks every 5 trading days). A value dated t uses returns up to
t only, so it is known at the close of t.

Two detectors:

1. Trailing z-scores. Each metric is compared with its own recent history:
   z_t = (x_t - mean of the previous `lookback` values) / their std.
   The current value is NOT part of its own baseline, so a sudden jump shows up
   at full size. A metric "fires" when its z-score passes `threshold` in the
   stress direction (up for correlation, for example).

2. Walk-forward Isolation Forest (Liu, Ting and Zhou, 2008). Fitted only on
   rows before each refit date and refitted every `refit_every` rows. The flag
   threshold is a high quantile of the scores on the training rows. Isolation
   Forest flags unusual rows in ANY direction, including unusually calm
   markets, so `detect` only keeps its flags when the z-score composite points
   toward stress.

Rules (checked in tests/test_anomaly.py): changing future rows never changes
past scores, z-scores, or flags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# +1: higher than usual means more stress. -1: lower than usual means more stress.
DIRECTIONS = {
    "avg_corr": 1,
    "pc1_share": 1,
    "absorption": 1,
    "avg_corr_chg": 1,
    "absorption_chg": 1,
    "network_density": 1,   # Brandon's thresholded-network density, if merged in
}
BASE_METRICS = ["avg_corr", "pc1_share", "absorption"]
CHANGE_METRICS = ["avg_corr", "absorption"]

STEP_LAGS = 4        # 4 rows x 5 trading days = change over about 20 trading days
LOOKBACK = 52        # about one year of 5-day rows
MIN_PERIODS = 26     # about six months before a z-score is reported
Z_THRESHOLD = 2.0
IF_MIN_TRAIN = 156   # about three years of 5-day rows before the first Isolation Forest fit
IF_REFIT_EVERY = 52  # refit about once a year
IF_QUANTILE = 0.95   # flag scores above the 95th percentile of training scores
SEED = 2302


def add_changes(feat: pd.DataFrame, cols=CHANGE_METRICS, lags: int = STEP_LAGS) -> pd.DataFrame:
    """Add `<col>_chg`: change over the last `lags` rows (backward-looking)."""
    out = feat.copy()
    for c in cols:
        out[f"{c}_chg"] = out[c].diff(lags)
    return out


def trailing_zscores(feat: pd.DataFrame, lookback: int = LOOKBACK,
                     min_periods: int = MIN_PERIODS) -> pd.DataFrame:
    """z-score of each column against its previous `lookback` values (current row excluded)."""
    prev = feat.shift(1)
    mean = prev.rolling(lookback, min_periods=min_periods).mean()
    std = prev.rolling(lookback, min_periods=min_periods).std()
    return (feat - mean) / std.replace(0, np.nan)


def signed(z: pd.DataFrame, directions: dict = DIRECTIONS) -> pd.DataFrame:
    """Flip columns so that positive always means 'toward stress'."""
    cols = [c for c in z.columns if c in directions]
    return z[cols] * pd.Series({c: directions[c] for c in cols})


def zscore_flags(z: pd.DataFrame, directions: dict = DIRECTIONS,
                 threshold: float = Z_THRESHOLD, min_count: int = 1) -> pd.DataFrame:
    """Composite (mean signed z), number of metrics past the threshold, and a flag.

    The flag is NaN where no metric has a z-score yet (warm-up period).
    """
    s = signed(z, directions)
    n_fired = (s >= threshold).sum(axis=1)
    ready = s.notna().any(axis=1)
    return pd.DataFrame({
        "z_composite": s.mean(axis=1),
        "z_n_fired": n_fired.where(ready),
        "z_flag": (n_fired >= min_count).astype(float).where(ready),
    })


def walk_forward_iforest(X: pd.DataFrame, min_train: int = IF_MIN_TRAIN,
                         refit_every: int = IF_REFIT_EVERY, quantile: float = IF_QUANTILE,
                         seed: int = SEED, n_estimators: int = 200) -> pd.DataFrame:
    """Isolation Forest scores where every row is scored by a model fitted on earlier rows only.

    Rows with missing values are skipped (score NaN). Returns `if_score` (higher =
    more unusual), `if_threshold` (from the training rows) and `if_flag`.
    `fit_end` is the last training date, so the no-look-ahead rule can be checked.
    """
    X = X.dropna()
    out = pd.DataFrame(np.nan, index=X.index, columns=["if_score", "if_threshold", "if_flag"])
    out["fit_end"] = pd.NaT
    for start in range(min_train, len(X), refit_every):
        train = X.iloc[:start]
        test = X.iloc[start:start + refit_every]
        mu, sd = train.mean(), train.std().replace(0, 1.0)
        model = IsolationForest(n_estimators=n_estimators, random_state=seed)
        model.fit(((train - mu) / sd).to_numpy())
        train_scores = -model.score_samples(((train - mu) / sd).to_numpy())
        thr = float(np.quantile(train_scores, quantile))
        scores = -model.score_samples(((test - mu) / sd).to_numpy())
        out.loc[test.index, "if_score"] = scores
        out.loc[test.index, "if_threshold"] = thr
        out.loc[test.index, "if_flag"] = (scores > thr).astype(float)
        out.loc[test.index, "fit_end"] = train.index[-1]
    return out


def detect(feat: pd.DataFrame, base_cols=BASE_METRICS, directions: dict = DIRECTIONS,
           lookback: int = LOOKBACK, min_periods: int = MIN_PERIODS,
           z_threshold: float = Z_THRESHOLD, if_kwargs: dict | None = None) -> pd.DataFrame:
    """Run both detectors on a network-metrics table (one row per window end date).

    Returns one row per input date with:
      z_<metric>        trailing z-score of each metric and of its 20-day change
      z_composite       mean z-score, signed so positive = toward stress
      z_n_fired, z_flag z-score detector
      if_score, if_threshold, if_flag  walk-forward Isolation Forest
      if_flag_stress    Isolation Forest flag kept only when z_composite > 0
      alert             1 if z_flag or if_flag_stress, NaN during warm-up
    """
    feat = add_changes(feat[list(base_cols)].astype(float), cols=[c for c in CHANGE_METRICS if c in base_cols])
    z = trailing_zscores(feat, lookback, min_periods)
    zs = zscore_flags(z, directions, z_threshold)
    iso = walk_forward_iforest(z.dropna(), **(if_kwargs or {}))
    out = pd.concat([z.add_prefix("z_"), zs, iso], axis=1)
    out["if_flag_stress"] = (out["if_flag"].eq(1) & out["z_composite"].gt(0)).astype(float).where(out["if_flag"].notna())
    flags = out[["z_flag", "if_flag_stress"]]
    out["alert"] = flags.max(axis=1).where(flags.notna().any(axis=1))
    return out


def to_daily(table: pd.DataFrame, calendar: pd.DatetimeIndex, max_stale: int = 10) -> pd.DataFrame:
    """Carry each row forward onto trading days until the next row arrives.

    A row dated t is used from t onward (never before t). Values older than
    `max_stale` trading days are dropped to NaN rather than carried forever.
    """
    calendar = pd.DatetimeIndex(calendar).sort_values()
    tab = table.sort_index().copy()
    cols = list(tab.columns)
    tab["asof"] = pd.DatetimeIndex(tab.index)
    tab["_row_pos"] = calendar.searchsorted(tab["asof"])
    cal = pd.DataFrame({"_date": calendar, "_cal_pos": np.arange(len(calendar))})
    m = pd.merge_asof(cal, tab.reset_index(drop=True), left_on="_date", right_on="asof",
                      direction="backward")
    stale = m["asof"].isna() | (m["_cal_pos"] - m["_row_pos"] > max_stale)
    m.loc[stale, cols + ["asof"]] = np.nan
    return m.set_index(calendar)[cols + ["asof"]]
