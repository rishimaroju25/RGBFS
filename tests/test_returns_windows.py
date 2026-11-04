"""Tests for src/returns.py and src/windows.py on small hand-built data.

Run from the repo root:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import returns as rt  # noqa: E402
from src import windows as wn  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", periods=12)
NO_OVERRIDES = pd.DataFrame(columns=["ticker", "start", "end", "action", "note"])


def frame(**cols):
    return pd.DataFrame(cols, index=DAYS, dtype=float)


def members(cols, off=None):
    m = pd.DataFrame(True, index=DAYS, columns=cols)
    for t, days in (off or {}).items():
        m.loc[m.index[days], t] = False
    return m


def run(close, mem, overrides=NO_OVERRIDES):
    return rt.clean_returns(close, mem, overrides)


def test_member_days_only_and_first_day_has_return():
    close = frame(A=np.linspace(100, 111, 12))
    mem = members(["A"], off={"A": [0, 1, 2]})          # joins on day 3
    r = run(close, mem).returns["A"]
    assert r.iloc[:3].isna().all()                       # not a member yet
    assert r.iloc[3] == pytest.approx(close["A"].iloc[3] / close["A"].iloc[2] - 1)  # uses pre-join price


def test_leaving_index_masks_returns():
    close = frame(A=np.linspace(100, 111, 12))
    mem = members(["A"], off={"A": list(range(8, 12))})  # leaves after day 7
    r = run(close, mem).returns["A"]
    assert r.iloc[8:].isna().all() and r.iloc[1:8].notna().all()


def test_no_fill_across_missing_price():
    px = np.linspace(100, 111, 12)
    px[5] = np.nan
    r = run(frame(A=px), members(["A"])).returns["A"]
    assert np.isnan(r.iloc[5]) and np.isnan(r.iloc[6])   # day after the gap is not a 2-day return
    assert r.iloc[7] == pytest.approx(px[7] / px[6] - 1)


def test_stale_run_removed_short_run_kept():
    px = np.array([100, 101, 102, 102, 102, 102, 102, 103, 104, 104, 105, 106], float)
    res = run(frame(A=px), members(["A"]))
    r = res.returns["A"]
    assert r.iloc[3:7].isna().all()                      # five equal closes -> stale
    assert r.iloc[9] == 0.0                              # two equal closes is normal
    assert res.summary["stale_removed"] == 4


def test_spike_removed_real_crash_kept():
    spike = np.array([100, 101, 102, 300, 103, 104, 105, 106, 107, 108, 109, 110], float)
    crash = np.array([100, 101, 102, 50, 51, 52, 53, 52, 51, 52, 53, 54], float)
    res = run(frame(S=spike, C=crash), members(["S", "C"]))
    assert res.returns["S"].iloc[3:5].isna().all()       # both legs of the spike gone
    assert res.returns["S"].iloc[5] == pytest.approx(104 / 103 - 1)
    assert res.returns["C"].iloc[3] == pytest.approx(-0.5098, abs=1e-3)  # real crash kept
    reasons = dict(zip(res.quality["ticker"] + res.quality["reason"], res.quality["reason"]))
    assert "Sspike_removed" in reasons and "Clarge_move_kept" in reasons


def test_overrides_drop_range_and_keep_day():
    spike = np.array([100, 101, 102, 300, 103, 104, 105, 106, 107, 108, 109, 110], float)
    ov = pd.DataFrame({"ticker": ["S", "B"], "start": [DAYS[3], DAYS[0]], "end": [DAYS[3], DAYS[11]],
                       "action": ["keep", "drop"], "note": ["", ""]})
    res = run(frame(S=spike, B=np.linspace(50, 60, 12)), members(["S", "B"]), ov)
    assert res.returns["S"].iloc[3] == pytest.approx(300 / 102 - 1)   # kept by override
    assert res.returns["B"].isna().all()                               # whole range dropped


def test_market_columns():
    close = frame(A=np.linspace(100, 111, 12), **{"^GSPC": np.linspace(3000, 3110, 12)})
    m = run(close, members(["A"])).market
    assert {"^GSPC_ret", "ew_member_ret", "n_returns"} <= set(m.columns)
    assert m["ew_member_ret"].iloc[2] == pytest.approx(close["A"].iloc[2] / close["A"].iloc[1] - 1)


# ------------------------------------------------------------------ windows
def make_returns(n_days=80, n_stocks=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    common = rng.standard_normal((n_days, 1))
    data = 0.6 * common + rng.standard_normal((n_days, n_stocks))
    return pd.DataFrame(data / 100, index=idx, columns=[f"T{i}" for i in range(n_stocks)])


def test_windows_are_trailing_and_complete():
    r = make_returns()
    r.iloc[70, 0] = np.nan                               # T0 missing one day
    ws = list(wn.windows(r, window=20, step=10, min_stocks=100))
    assert ws[0].end == r.index[19] and ws[0].start == r.index[0]
    for w in ws:
        assert w.block.index[-1] == w.end and len(w.block) == 20
        assert w.block.notna().all().all()
        if w.start <= r.index[70] <= w.end:
            assert "T0" not in w.block.columns          # incomplete stock left out


def test_feature_does_not_change_when_future_changes():
    r = make_returns()
    before = wn.connectedness(r, window=20, step=5, min_stocks=100)
    r2 = r.copy()
    r2.iloc[60:] = r2.iloc[60:] * 5 + 0.01               # change only the future
    after = wn.connectedness(r2, window=20, step=5, min_stocks=100)
    safe = before.index < r.index[60]
    pd.testing.assert_frame_equal(before[safe], after[safe])


def test_min_stocks_skips_windows():
    r = make_returns(n_stocks=50)
    assert wn.connectedness(r, window=20, step=5, min_stocks=100).empty


def test_correlation_at_and_metrics():
    r = make_returns()
    c = wn.correlation_at(r, r.index[50], window=30)
    assert c.shape == (120, 120) and np.allclose(np.diag(c), 1)
    m = wn.connectedness_metrics(r.iloc[:30])
    assert 0 < m["avg_corr"] < 1 and 0 < m["pc1_share"] <= 1 and 0 < m["absorption"] <= 1
    with pytest.raises(ValueError):
        wn.correlation_at(r, r.index[5], window=30)


def test_quality_report_lists_only_flagged_cells():
    spike = np.array([100, 101, 102, 300, 103, 104, 105, 106, 107, 108, 109, 110], float)
    res = run(frame(S=spike, N=np.linspace(50, 60, 12)), members(["S", "N"]))
    q = res.quality
    assert set(q["ticker"]) == {"S"} and set(q["reason"]) == {"spike_removed"} and len(q) == 2
