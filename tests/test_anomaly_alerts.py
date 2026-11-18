"""Tests for src/anomaly.py and src/alerts.py (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from src import alerts as al
from src import anomaly as an
from tests import synthetic


@pytest.fixture(scope="module")
def fake():
    return synthetic.make()


def test_zscore_excludes_current_value():
    s = pd.DataFrame({"avg_corr": [0.2, 0.21, 0.19, 0.2, 0.21, 0.19, 0.2, 0.6]})
    z = an.trailing_zscores(s, lookback=7, min_periods=5)
    assert z.iloc[-1, 0] > 10          # the jump is measured against the calm past
    assert z.iloc[:4].isna().all().all()  # warm-up


def test_detect_has_no_lookahead(fake):
    feat = fake["connectedness"]
    base = an.detect(feat)
    cut = feat.index[400]
    changed = feat.copy()
    changed.loc[changed.index > cut, ["avg_corr", "absorption"]] *= 3
    after = an.detect(changed)
    cols = ["z_avg_corr", "z_composite", "z_flag", "if_score", "if_flag", "alert"]
    pd.testing.assert_frame_equal(base.loc[:cut, cols], after.loc[:cut, cols])


def test_iforest_trains_only_on_past(fake):
    out = an.detect(fake["connectedness"])
    scored = out.dropna(subset=["if_score"])
    assert len(scored) > 0
    assert (pd.to_datetime(scored["fit_end"]) < scored.index).all()


def test_detect_finds_fake_buildups(fake):
    out = an.detect(fake["connectedness"])
    daily = an.to_daily(out[["alert"]], fake["panel"].index)
    lt = al.lead_times(daily["alert"], fake["episodes"])
    assert (lt.result == "early").sum() >= 4


def test_to_daily_never_uses_a_value_before_its_date():
    cal = pd.bdate_range("2020-01-01", periods=30)
    tab = pd.DataFrame({"x": [1.0, 2.0]}, index=[cal[5], cal[10]])
    d = an.to_daily(tab, cal, max_stale=3)
    assert d.loc[cal[:5], "x"].isna().all()
    assert (d.loc[cal[5:9], "x"] == 1).all()
    assert d.loc[cal[9], "x"] != d.loc[cal[9], "x"]  # stale after 3 days -> NaN
    assert (d.loc[cal[10:14], "x"] == 2).all() and d.loc[cal[14:], "x"].isna().all()


def test_alert_events_merges_short_gaps():
    cal = pd.bdate_range("2020-01-01", periods=30)
    f = pd.Series(0.0, index=cal)
    f.iloc[[2, 3, 6, 20]] = 1
    ev = al.alert_events(f, merge_gap=5)
    assert len(ev) == 2
    assert ev.iloc[0].start == cal[2] and ev.iloc[0].end == cal[6]


def _one_episode(cal, s, e):
    return pd.DataFrame({"episode_id": [1], "name": ["x"], "start": [cal[s]], "end": [cal[e]],
                         "start_trailing": [cal[s + 5]]})


def test_lead_times_early_late_missed():
    cal = pd.bdate_range("2020-01-01", periods=200)
    ep = _one_episode(cal, 100, 120)
    f = pd.Series(0.0, index=cal)
    f.iloc[85] = 1
    assert al.lead_times(f, ep).lead_days.iloc[0] == 15
    f[:] = 0
    f.iloc[103] = 1
    r = al.lead_times(f, ep).iloc[0]
    assert r.result == "late" and r.lead_days == -3
    f[:] = 0
    f.iloc[10] = 1   # far too early: outside the 60-day window
    assert al.lead_times(f, ep).result.iloc[0] == "missed"
    assert al.lead_times(f, ep, start_col="start_trailing").start.iloc[0] == cal[105]


def test_lead_window_stops_at_previous_episode():
    cal = pd.bdate_range("2020-01-01", periods=300)
    eps = pd.DataFrame({"episode_id": [1, 2], "name": ["a", "b"], "start": [cal[50], cal[120]],
                        "end": [cal[100], cal[150]], "start_trailing": [cal[55], cal[125]]})
    f = pd.Series(0.0, index=cal)
    f.iloc[95] = 1   # inside episode 1, should not count as a warning for episode 2
    lt = al.lead_times(f, eps)
    assert lt.result.iloc[1] == "missed"


def test_masks_target_and_eligible():
    cal = pd.bdate_range("2020-01-01", periods=200)
    m = al.episode_masks(cal, _one_episode(cal, 100, 120), horizon=20)
    assert m.target.iloc[80:100].all() and m.target.iloc[79] == 0 and m.target.iloc[100] == 0
    assert m.in_episode.iloc[100:121].all()
    assert m.eligible.iloc[100:121].sum() == 0
    assert m.eligible.iloc[-20:].sum() == 0


def test_walk_forward_classifier_purges_future_labels(fake):
    cal = fake["panel"].index
    masks = al.episode_masks(cal, fake["episodes"])
    data = masks.join(fake["panel"][["VIXCLS"]])
    pred = al.walk_forward_classifier(data, ["VIXCLS"], first_test_year=2010)
    assert not pred.empty
    pos = pd.Series(np.arange(len(cal)), index=cal)
    for year, g in pred.groupby("fold"):
        first = pos[g.index.min()]
        assert pos[g.train_last.iloc[0]] + al.HORIZON < first
