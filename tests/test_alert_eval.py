"""Tests for src/alert_eval.py on a small hand-built calendar."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import alert_eval as ae  # noqa: E402

IDX = pd.bdate_range("2020-01-01", periods=100)


def _stress():
    s = pd.Series(0.0, index=IDX)
    s.iloc[50:60] = 1.0          # one episode starting on day 50
    return s


def _alert(days):
    a = pd.Series(False, index=IDX)
    a.iloc[days] = True
    return a


def test_true_warning_false_alarm_and_during_stress():
    r = ae.score_alerts(_alert([10, 40, 41, 55]), _stress(), horizon=20, merge_gap=2)
    ev = r.events.set_index(r.events["start"].map(IDX.get_loc))
    assert ev.loc[10, "false_alarm"]                       # no episode within 20 days
    assert ev.loc[40, "true_warning"] and ev.loc[40, "lead"] == 10
    assert ev.loc[40, "alert_days"] == 2                   # days 40-41 merged
    assert ev.loc[55, "during_stress"] and not ev.loc[55, "false_alarm"]
    assert r.episodes["caught"].all()


def test_merge_gap():
    r = ae.score_alerts(_alert([10, 13, 30]), _stress(), merge_gap=2)
    assert len(r.events) == 2                              # 10 and 13 merge (gap of 2), 30 separate


def test_summary_counts_and_quiet_days():
    r = ae.score_alerts(_alert([10, 40]), _stress(), horizon=20)
    s = ae.summarize(r)
    assert s["warnings"] == 2 and s["false_alarms"] == 1 and s["episodes_caught"] == 1
    # quiet days: calm and not within 20 days before day 50 -> 100 - 10 stress - 20 = 70
    assert r.days["quiet"].sum() == 70
    assert np.isclose(s["day_false_positive_rate"], 1 / 70)


def test_period_filter():
    r = ae.score_alerts(_alert([10, 40]), _stress(), horizon=20)
    early = ae.summarize(r, end=IDX[30])
    assert early["warnings"] == 1 and early["episodes"] == 0


def test_episode_starts():
    assert list(ae.episode_starts(_stress())) == [IDX[50]]


def test_after_stress_false_alarm():
    r = ae.score_alerts(_alert([65, 90]), _stress(), horizon=20)
    ev = r.events.set_index(r.events["start"].map(IDX.get_loc))
    assert ev.loc[65, "false_alarm"] and ev.loc[65, "after_stress"]     # 6 days after the episode ends
    assert ev.loc[90, "false_alarm"] and not ev.loc[90, "after_stress"]


def test_long_alert_is_not_a_warning():
    r = ae.score_alerts(_alert(list(range(5, 45))), _stress(), horizon=20)
    assert r.events["false_alarm"].iloc[0] and not r.episodes["caught"].iloc[0]
