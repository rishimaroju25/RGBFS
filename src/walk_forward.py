"""Walk-forward train/test splits and point-in-time helpers (Week 5).

Owners: Salah with Fatima (Evaluation / backtesting).
Checklist: docs/lookahead_checklist.md · Audit: notebooks/13_lookahead_check.ipynb

The rule every split here enforces
----------------------------------
A training row may only be used if everything it depends on, features AND
label, was known before the first day of the test block it is used for.

Labels are the hard part. Our stress labels (src/stress_labels.py) use a
centered 21-day volatility window, so the label on day t uses returns up to
t + 10 trading days. An early-warning target "stress within the next h days"
reaches t + h + 10. A plain chronological split lets the last training rows
see returns from inside the test block. `stress_target` records how far each
label reaches (`label_end`), and `walk_forward_splits` drops ("purges") every
training row whose `label_end` is on or after the test start.

Typical use
-----------
    from src import walk_forward as wf
    tgt = wf.stress_target(labels["stress"], horizon=20)
    X = features.join(tgt, how="inner").dropna()
    for fold in wf.walk_forward_splits(X.index, X["label_end"], first_test_year=2010):
        model.fit(X.loc[fold.train, cols], X.loc[fold.train, "y"])
        p = model.predict_proba(X.loc[fold.test, cols])[:, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

LABEL_REACH = 10   # trading days a stress label looks ahead (centered 21-day window)

# Trading days to delay weekly FRED stress indexes before using them as features.
# Both are dated the Friday that ends the week. NFCI is published the following
# Wednesday (Chicago Fed); STLFSI4 the following Wednesday or Thursday (FRED).
# Five trading days puts each value on the next Friday, after both releases.
RELEASE_LAG = {"STLFSI4": 5, "NFCI": 5}


def lag_by_release(panel: pd.DataFrame, lags: dict[str, int] = RELEASE_LAG) -> pd.DataFrame:
    """Copy of `panel` with each listed column shifted forward by its release lag."""
    out = panel.copy()
    for col, k in lags.items():
        if col in out:
            out[col] = out[col].shift(k)
    return out


def stress_target(stress: pd.Series, horizon: int = 0, reach: int = LABEL_REACH) -> pd.DataFrame:
    """Target table indexed like `stress` (0/1 with NaN where unlabeled).

    horizon = 0  -> y = stress on day t (same-day detection)
    horizon = h  -> y = 1 if any of days t+1 .. t+h is a stress day (early warning)

    Columns: y, in_stress (label on day t, for dropping days already in stress
    in an early-warning test), label_end (last trading day whose returns the
    label uses). y is NaN if any day it needs is unlabeled.
    """
    s = stress.astype(float)
    idx = s.index
    vals = s.to_numpy()
    n = len(vals)
    y = np.full(n, np.nan)
    end_pos = np.arange(n) + horizon + reach
    for i in range(n):
        if horizon == 0:
            y[i] = vals[i]
        elif i + horizon < n:
            block = vals[i + 1:i + horizon + 1]
            y[i] = np.nan if np.isnan(block).any() else float(block.max())
    label_end = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
    ok = end_pos < n
    label_end.iloc[np.flatnonzero(ok)] = idx[end_pos[ok]]
    y[~ok] = np.nan
    return pd.DataFrame({"y": y, "in_stress": vals, "label_end": label_end}, index=idx)


@dataclass
class Fold:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    purged: int            # training rows dropped because their label reaches the test block


def walk_forward_splits(index: pd.DatetimeIndex, label_end: pd.Series, first_test_year: int,
                        last_test_year: int | None = None, min_train: int = 750) -> Iterator[Fold]:
    """Expanding-window folds with one calendar year per test block.

    For the test block starting on day T, training rows are every row dated
    before T whose label_end is also before T (purging). Years with fewer than
    `min_train` usable training rows are skipped. The model is refit once per
    year, which is fine for the small models we use.
    """
    index = pd.DatetimeIndex(index).sort_values()
    label_end = pd.Series(label_end, index=index)
    last = last_test_year or index.max().year
    for year in range(first_test_year, last + 1):
        test = index[index.year == year]
        if len(test) == 0:
            continue
        t0 = test[0]
        before = index[index < t0]
        keep = label_end.loc[before] < t0
        train = before[keep.to_numpy()]
        if len(train) < min_train:
            continue
        yield Fold(t0, test[-1], train, test, purged=int((~keep).sum()))


def fold_table(folds: list[Fold]) -> pd.DataFrame:
    """One row per fold: dates and sizes, for printing and for checks."""
    return pd.DataFrame([{
        "test_year": f.test_start.year, "train_start": f.train[0].date(), "train_end": f.train[-1].date(),
        "test_start": f.test_start.date(), "test_end": f.test_end.date(),
        "n_train": len(f.train), "n_test": len(f.test), "purged": f.purged,
    } for f in folds])


def check_folds(folds: list[Fold], label_end: pd.Series) -> None:
    """Raise if any fold could leak: overlap, wrong order, or a training label reaching the test block."""
    seen = set()
    for f in folds:
        assert f.train.max() < f.test_start, "training row on or after test start"
        assert label_end.loc[f.train].max() < f.test_start, "training label reaches into test block"
        assert not seen.intersection(f.test), "test blocks overlap"
        seen.update(f.test)
