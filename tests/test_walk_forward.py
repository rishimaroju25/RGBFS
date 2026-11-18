"""Tests for src/walk_forward.py on small synthetic examples."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import walk_forward as wf  # noqa: E402


def _stress(n=30):
    idx = pd.bdate_range("2020-01-01", periods=n)
    s = pd.Series(0.0, index=idx)
    s.iloc[15:18] = 1.0
    return s


def test_same_day_target_and_reach():
    s = _stress()
    t = wf.stress_target(s, horizon=0, reach=2)
    assert t["y"].iloc[15] == 1 and t["y"].iloc[14] == 0
    assert t["label_end"].iloc[0] == s.index[2]
    assert t["y"].iloc[-2:].isna().all()          # label would need data past the end


def test_early_warning_target():
    s = _stress()
    t = wf.stress_target(s, horizon=3, reach=0)
    assert t["y"].iloc[12] == 1                    # day 15 is within 13..15
    assert t["y"].iloc[11] == 0                    # 12..14 has no stress
    assert t["in_stress"].iloc[16] == 1
    assert t["label_end"].iloc[12] == s.index[15]


def test_nan_in_window_gives_nan():
    s = _stress()
    s.iloc[20] = np.nan
    t = wf.stress_target(s, horizon=3, reach=0)
    assert np.isnan(t["y"].iloc[18])


def test_release_lag_shifts_only_listed_columns():
    idx = pd.bdate_range("2020-01-03", periods=10)
    p = pd.DataFrame({"STLFSI4": np.arange(10.0), "VIXCLS": np.arange(10.0)}, index=idx)
    q = wf.lag_by_release(p)
    assert q["STLFSI4"].iloc[5] == 0 and q["STLFSI4"].iloc[:5].isna().all()
    assert (q["VIXCLS"] == p["VIXCLS"]).all()


def test_splits_purge_and_order():
    idx = pd.bdate_range("2015-01-01", "2019-12-31")
    s = pd.Series(0.0, index=idx)
    t = wf.stress_target(s, horizon=20, reach=10)
    folds = list(wf.walk_forward_splits(idx, t["label_end"], first_test_year=2017, min_train=100))
    assert [f.test_start.year for f in folds] == [2017, 2018, 2019]
    wf.check_folds(folds, t["label_end"])
    assert all(f.purged == 30 for f in folds)      # horizon + reach rows dropped before each test start
    assert len(folds[1].train) > len(folds[0].train)
