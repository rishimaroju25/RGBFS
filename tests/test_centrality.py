"""Tests for src/centrality.py on small synthetic examples."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import centrality as ce  # noqa: E402


def test_gini_and_top_share_extremes():
    assert abs(ce.gini(np.ones(50))) < 1e-12
    assert ce.gini(np.r_[np.zeros(49), 1.0]) > 0.95
    assert abs(ce.top_share(np.ones(100)) - 0.10) < 1e-12
    assert ce.top_share(np.r_[np.zeros(90), np.ones(10)]) == 1.0


def test_freeman_star_and_ring():
    n = 20
    star = np.r_[n - 1, np.ones(n - 1)]
    ring = np.full(n, 2)
    assert abs(ce.freeman_centralization(star) - 1.0) < 1e-12
    assert ce.freeman_centralization(ring) == 0.0


def _returns(hub_group: bool, n_days=60, n=150, seed=0):
    """Market factor plus noise; optionally one 15-stock group sharing an extra factor."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_days, 1)) + rng.standard_normal((n_days, n))
    if hub_group:
        x[:, :15] += 2.0 * rng.standard_normal((n_days, 1))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(x / 100, index=idx, columns=[f"S{i}" for i in range(n)])


def test_concentrated_group_raises_concentration():
    even, _ = ce.window_metrics(_returns(False))
    hub, strength = ce.window_metrics(_returns(True))
    assert hub["strength_gini"] > even["strength_gini"] + 0.1
    assert hub["evc_top10"] > even["evc_top10"] + 0.2
    assert set(strength.nlargest(10).index) <= {f"S{i}" for i in range(15)}


def test_over_time_shapes():
    r = pd.concat([_returns(False, n_days=80, seed=s) for s in range(1)])
    feats, hubs = ce.centrality_over_time(r, window=60, step=10, min_stocks=100)
    assert len(feats) == 3
    assert set(hubs["rank"]) == set(range(1, ce.N_HUBS + 1))
