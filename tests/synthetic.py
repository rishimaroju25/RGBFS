"""Synthetic stand-ins for the real data files, for tests and dry runs only.

Nothing here is real market data. The fake episodes have network metrics that
start rising before the episode, so the pipeline has something to find.

    python tests/synthetic.py /tmp/rgbfs_fake   # writes a fake data/ and docs/ tree
    RGBFS_DATA_ROOT=/tmp/rgbfs_fake jupyter nbconvert --execute ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPISODES = [("2008-09-02", "2009-04-30"), ("2011-07-25", "2011-11-30"), ("2015-08-12", "2015-09-10"),
            ("2018-12-11", "2019-01-09"), ("2020-02-14", "2020-06-25"), ("2022-04-20", "2022-06-28")]


def calendar(start="2005-01-03", end="2026-10-30") -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def episodes_table(cal: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for i, (s, e) in enumerate(EPISODES, 1):
        s_pos = cal.searchsorted(pd.Timestamp(s))
        rows.append({"episode_id": i, "start": cal[s_pos], "end": pd.Timestamp(e),
                     "start_trailing": cal[s_pos + 8], "name": f"Fake episode {i}"})
    return pd.DataFrame(rows)


def stress_signal(cal: pd.DatetimeIndex, eps: pd.DataFrame, ramp: int = 30) -> np.ndarray:
    """0 in calm times, ramps up over `ramp` days before each start, 1 inside episodes."""
    x = np.zeros(len(cal))
    for _, ep in eps.iterrows():
        s, e = cal.searchsorted(ep.start), cal.searchsorted(ep.end)
        x[max(0, s - ramp):s] = np.maximum(x[max(0, s - ramp):s], np.linspace(0, 0.8, min(ramp, s)))
        x[s:e + 1] = 1.0
    return x


def make(seed: int = 0):
    rng = np.random.default_rng(seed)
    cal = calendar()
    eps = episodes_table(cal)
    sig = stress_signal(cal, eps)
    n = len(cal)
    noise = pd.Series(rng.normal(0, 1, n)).rolling(10, min_periods=1).mean().to_numpy()
    avg_corr = 0.25 + 0.30 * sig + 0.03 * noise
    conn_days = cal[59::5]
    conn = pd.DataFrame({
        "avg_corr": avg_corr[59::5],
        "pc1_share": (avg_corr * 1.05 + 0.01)[59::5],
        "absorption": (0.45 + 0.35 * sig + 0.03 * noise)[59::5],
        "n_stocks": 450.0,
    }, index=pd.DatetimeIndex(conn_days, name="date"))
    vol = 0.12 + 0.30 * np.r_[np.zeros(5), sig[:-5]] + 0.02 * np.abs(rng.normal(size=n))
    spy_ret = rng.normal(0, 1, n) * vol / np.sqrt(252)
    vix = 100 * vol + rng.normal(0, 1.5, n)
    stlfsi = pd.Series(-0.5 + 2.5 * np.r_[np.zeros(10), sig[:-10]] + rng.normal(0, 0.2, n), index=cal)
    stlfsi = stlfsi.where(cal.dayofweek == 4).ffill()
    panel = pd.DataFrame({"VIXCLS": vix, "STLFSI4": stlfsi.to_numpy(), "SPY_ret": spy_ret,
                          "SPY": 100 * np.exp(np.cumsum(spy_ret))}, index=cal)
    stress = pd.DataFrame({"stress": (sig == 1).astype(float)}, index=cal)
    stress.iloc[-10:] = np.nan
    market = pd.DataFrame({"n_returns": 470 + rng.integers(-20, 20, n), "ew_member_ret": spy_ret,
                           "^GSPC_ret": spy_ret}, index=cal)
    return {"connectedness": conn, "panel": panel, "stress": stress, "episodes": eps, "market": market}


def write(root) -> Path:
    root = Path(root)
    d = make()
    (root / "data/processed/features").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    d["connectedness"].to_parquet(root / "data/processed/features/connectedness_60d.parquet")
    d["panel"].to_parquet(root / "data/processed/daily_panel.parquet")
    d["stress"].to_parquet(root / "data/processed/stress_labels_daily.parquet")
    d["market"].to_parquet(root / "data/processed/returns_market.parquet")
    d["episodes"].to_csv(root / "docs/stress_periods.csv", index=False)
    return root


if __name__ == "__main__":
    print(write(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rgbfs_fake"))
