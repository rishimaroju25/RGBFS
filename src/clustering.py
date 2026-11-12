"""Clustering settings and stability checks for stock networks.

Owner: Rishi (Week 3: support on clustering; test stability, help choose settings).
Used by Grace's clustering notebooks and notebooks/08_cluster_stability.ipynb.

Why this exists
---------------
The Week 2/3 clustering used raw correlations with a fixed threshold (|corr| > 0.55).
On the real data that had two problems (notebooks/08_cluster_stability.ipynb):

1. **It mostly measured the market, not groups.** In stress, almost every pair
   passes 0.55 (63% of pairs on 2008-10-15), so the network is one blob. The
   number of communities and modularity then track network density with rank
   correlation -0.92 and -0.95, and density is just the overall correlation
   level we already have in features/connectedness_60d.parquet.
2. **Clusters reshuffled a lot** from one window to the next.

Two standard fixes, both optional here:

* **Remove the market move first** (`adjust="market"`): regress each stock's
  returns on the equal-weighted market return inside the window and correlate
  the residuals. What is left is how stocks move *relative to the market*,
  which is where sector and style groups show up.
* **Fixed density instead of a fixed threshold** (`density=0.05`): keep the
  strongest 5% of positive links every window, so the network has the same
  size in calm and stress periods and the cluster measures stop being a copy
  of the correlation level.

Communities use Louvain on the weighted graph (link weight = correlation),
with a fixed seed so results repeat.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

SEED = 2302


@dataclass
class Setting:
    """One way of building the network and finding communities."""
    name: str
    adjust: str = "market"        # "market" = residuals after removing the market move; "none" = raw returns
    density: float | None = 0.05  # keep this share of strongest positive links; None = use `threshold`
    threshold: float = 0.55       # used only when density is None
    absolute: bool = False        # True = link on |corr| (old setting); False = positive links only
    method: str = "louvain"       # "louvain" (weighted) or "greedy" (unweighted greedy modularity, old setting)
    resolution: float = 1.0


# The original Week 2/3 setting, so it can be compared side by side
BASELINE = Setting("baseline: raw, |corr| > 0.55, greedy", adjust="none", density=None,
                   threshold=0.55, absolute=True, method="greedy")
RECOMMENDED = Setting("market-adjusted, top 5% links, Louvain")


def adjusted_corr(block: pd.DataFrame, adjust: str = "market") -> pd.DataFrame:
    """Correlation matrix of a window, optionally after removing the market move.

    `block` is days x stocks with complete data (as from src.windows).
    """
    x = block.to_numpy(dtype=float)
    if adjust == "market":
        m = x.mean(axis=1)
        mc = m - m.mean()
        xc = x - x.mean(axis=0)
        beta = mc @ xc / (mc @ mc)
        x = xc - np.outer(mc, beta)
    elif adjust != "none":
        raise ValueError("adjust must be 'market' or 'none'")
    c = np.corrcoef(x, rowvar=False)
    return pd.DataFrame(c, index=block.columns, columns=block.columns)


def build_graph(corr: pd.DataFrame, setting: Setting) -> nx.Graph:
    """Weighted graph from a correlation matrix. Isolated stocks are dropped."""
    c = corr.to_numpy(copy=True)
    np.fill_diagonal(c, np.nan)
    iu = np.triu_indices(len(c), k=1)
    vals = c[iu]
    score = np.abs(vals) if setting.absolute else vals
    if setting.density is not None:
        k = max(1, int(round(setting.density * len(vals))))
        cut = np.partition(np.nan_to_num(score, nan=-np.inf), len(vals) - k)[len(vals) - k]
        keep = score >= cut
        if not setting.absolute:
            keep &= vals > 0
    else:
        keep = score > setting.threshold
    keep &= ~np.isnan(vals)
    names = corr.index.to_numpy()
    g = nx.Graph()
    g.add_weighted_edges_from(zip(names[iu[0][keep]], names[iu[1][keep]], np.abs(vals[keep])))
    return g


def communities(g: nx.Graph, setting: Setting) -> dict:
    """ticker -> community number (1 = largest)."""
    if g.number_of_edges() == 0:
        return {}
    if setting.method == "louvain":
        groups = nx.community.louvain_communities(g, weight="weight", resolution=setting.resolution, seed=SEED)
    elif setting.method == "greedy":
        groups = nx.community.greedy_modularity_communities(g, resolution=setting.resolution)
    else:
        raise ValueError("method must be 'louvain' or 'greedy'")
    groups = sorted(groups, key=lambda s: (-len(s), sorted(map(str, s))[0]))
    return {t: i for i, grp in enumerate(groups, start=1) for t in grp}


def stability(labels_a: dict, labels_b: dict, min_common: int = 50) -> float:
    """Adjusted Rand index between two windows' communities, on stocks in both.

    1 = identical grouping, about 0 = no more alike than random.
    """
    common = sorted(set(labels_a) & set(labels_b))
    if len(common) < min_common:
        return np.nan
    return adjusted_rand_score([labels_a[t] for t in common], [labels_b[t] for t in common])


def sector_agreement(labels: dict, sectors: dict, min_matched: int = 10) -> float:
    """Adjusted Rand index between communities and sector labels."""
    pairs = [(labels[t], sectors[t]) for t in labels if t in sectors]
    if len(pairs) < min_matched:
        return np.nan
    a, b = zip(*pairs)
    return adjusted_rand_score(b, a)


def run_over_time(returns: pd.DataFrame, setting: Setting, window: int = 60, step: int = 20,
                  sectors: dict | None = None, min_stocks: int = 100,
                  start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Communities every `step` trading days. Returns (features, memberships).

    features: one row per window end date with n_stocks, n_edges, density,
    n_communities, modularity, largest_share, sector_ari and stability
    (ARI against the previous window). memberships: date, ticker, community.
    Uses src.windows so every window is trailing and point-in-time.
    """
    from src import windows as wn

    feats, members, prev = [], [], None
    for w in wn.windows(returns, window=window, step=step, min_stocks=min_stocks, start=start, end=end):
        g = build_graph(adjusted_corr(w.block, setting.adjust), setting)
        labels = communities(g, setting)
        groups = {}
        for t, k in labels.items():
            groups.setdefault(k, set()).add(t)
        mod = nx.community.modularity(g, list(groups.values()), weight="weight") if groups else np.nan
        feats.append({
            "date": w.end, "n_stocks": w.n_stocks, "n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges(),
            "density": nx.density(g) if g.number_of_nodes() > 1 else np.nan,
            "n_communities": len(groups), "modularity": mod,
            "largest_share": max(map(len, groups.values())) / len(labels) if labels else np.nan,
            "sector_ari": sector_agreement(labels, sectors) if sectors else np.nan,
            "stability": stability(prev, labels) if prev is not None else np.nan,
        })
        members += [{"date": w.end, "ticker": t, "community": k} for t, k in labels.items()]
        prev = labels
    return pd.DataFrame(feats).set_index("date"), pd.DataFrame(members)
