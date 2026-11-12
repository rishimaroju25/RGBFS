"""Tests for src/clustering.py on synthetic data with planted sector groups."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import clustering as cl  # noqa: E402


def planted(n_days=120, n_sectors=5, per=30, market=2.0, sector=0.8, seed=0):
    """Returns with a strong market factor plus a weaker factor per sector."""
    rng = np.random.default_rng(seed)
    mkt = rng.standard_normal((n_days, 1)) * market
    cols, data, sectors = [], [], {}
    for s in range(n_sectors):
        f = rng.standard_normal((n_days, 1)) * sector
        x = mkt * rng.uniform(0.8, 1.2, per) + f + rng.standard_normal((n_days, per))
        data.append(x)
        for i in range(per):
            t = f"S{s}_{i}"
            cols.append(t)
            sectors[t] = f"sector{s}"
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(np.hstack(data) / 100, index=idx, columns=cols), sectors


def labels_for(block, setting):
    return cl.communities(cl.build_graph(cl.adjusted_corr(block, setting.adjust), setting), setting)


def test_market_adjustment_recovers_sectors_when_market_dominates():
    r, sectors = planted()
    block = r.iloc[:60]
    old = cl.sector_agreement(labels_for(block, cl.BASELINE), sectors)
    new = cl.sector_agreement(labels_for(block, cl.RECOMMENDED), sectors)
    assert new > 0.8, new                 # planted groups found
    assert new > old + 0.3, (old, new)    # and much better than the old setting


def test_residual_corr_removes_market():
    r, _ = planted(sector=0.0)            # market only, no groups
    raw = cl.adjusted_corr(r, "none").to_numpy()
    res = cl.adjusted_corr(r, "market").to_numpy()
    off = ~np.eye(len(raw), dtype=bool)
    assert raw[off].mean() > 0.6 and abs(res[off].mean()) < 0.05


def test_fixed_density_keeps_network_size_constant():
    calm, _ = planted(market=0.3)
    stress, _ = planted(market=3.0)
    s = cl.Setting("raw, top 5%", adjust="none", density=0.05)
    n_pairs = 150 * 149 / 2
    for r in (calm, stress):
        g = cl.build_graph(cl.adjusted_corr(r, "none"), s)
        assert abs(g.number_of_edges() / n_pairs - 0.05) < 0.005


def test_stability_and_results_repeat():
    r, sectors = planted(n_days=200)
    a = labels_for(r.iloc[:60], cl.RECOMMENDED)
    b = labels_for(r.iloc[20:80], cl.RECOMMENDED)
    assert cl.stability(a, b) > 0.7                         # same planted groups, overlapping windows
    assert labels_for(r.iloc[:60], cl.RECOMMENDED) == a     # fixed seed: same answer every run


def test_run_over_time_columns():
    r, sectors = planted(n_days=200, per=25)
    feats, members = cl.run_over_time(r, cl.RECOMMENDED, window=60, step=20, sectors=sectors, min_stocks=100)
    assert {"n_communities", "modularity", "sector_ari", "stability"} <= set(feats.columns)
    assert feats["stability"].iloc[1:].notna().all() and feats["sector_ari"].min() > 0.7
    assert set(members.columns) == {"date", "ticker", "community"}
