"""Centrality concentration features for the stock network (Week 4).

Owner: Salah. Notebook: notebooks/10_centrality_concentration.ipynb
Output: data/processed/features/centrality_60d.parquet (+ centrality_hubs_60d.parquet)

Question these features answer
------------------------------
Is influence in the market network spread across many stocks, or concentrated
in a few hubs? Stress is often said to build in one group first (for example
banks in 2007-08) before it spreads, which would show up as rising
concentration *before* the whole market moves together.

Network definition
------------------
The graph is the team's recommended setting from src/clustering.py
(`RECOMMENDED`): each stock's returns minus the equal-weighted market move,
then the strongest 5% of positive links, weighted by correlation. Using the
same graph as the shared cluster features keeps the network features
consistent. The market adjustment matters here: on raw correlations nearly
every pair links up in stress, every stock becomes a hub, and concentration
just mirrors the average correlation we already have.

One measure (`raw_evc_eff_share`) uses the raw correlation matrix with no
graph at all, as a check that does not depend on the network definition.

All measures are computed over every stock in the window, including stocks
with no links, so values are comparable across windows (unlike density after
dropping isolates).

Measures (one row per window end date)
--------------------------------------
strength_gini      Gini coefficient of weighted degree (0 = all equal, 1 = one hub)
strength_top10     share of total weighted degree held by the top 10% of stocks
degree_central     Freeman degree centralization of the unweighted graph (0 = even, 1 = star)
evc_top10          share of eigenvector-centrality mass (squared loadings) in the top 10% of stocks
raw_evc_eff_share  effective number of stocks in the leading eigenvector of the RAW
                   correlation matrix, divided by n (1 = every stock equally, low = a few)

All windows are trailing (src.windows), so a value dated t uses data up to t only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import clustering as cl
from src import windows as wn

SETTING = cl.RECOMMENDED
TOP_SHARE = 0.10
N_HUBS = 10


def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative values (0 = equal, ->1 = concentrated)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


def top_share(x: np.ndarray, share: float = TOP_SHARE) -> float:
    """Share of the total held by the largest `share` of entries."""
    x = np.asarray(x, dtype=float)
    if x.sum() == 0:
        return np.nan
    k = max(1, int(round(share * len(x))))
    return np.sort(x)[-k:].sum() / x.sum()


def freeman_centralization(degree: np.ndarray) -> float:
    """Freeman degree centralization: 0 for a regular graph, 1 for a star."""
    n = len(degree)
    if n < 3:
        return np.nan
    return (degree.max() - degree).sum() / ((n - 1) * (n - 2))


def leading_vector(m: np.ndarray) -> np.ndarray:
    """Leading eigenvector of a symmetric matrix, squared so it sums to 1."""
    _, vecs = np.linalg.eigh(m)
    v = vecs[:, -1]
    return v ** 2 / (v ** 2).sum()


def window_metrics(block: pd.DataFrame, setting: cl.Setting = SETTING) -> tuple[dict, pd.Series]:
    """Measures for one window, plus each stock's weighted degree (for the hub table)."""
    tickers = block.columns
    n = len(tickers)

    corr = cl.adjusted_corr(block, setting.adjust)
    g = cl.build_graph(corr, setting)

    # Weighted adjacency over ALL stocks in the window (isolated stocks get zero rows).
    pos = {t: i for i, t in enumerate(tickers)}
    a = np.zeros((n, n))
    for u, v, w in g.edges(data="weight"):
        a[pos[u], pos[v]] = a[pos[v], pos[u]] = w
    strength = a.sum(axis=1)
    degree = (a > 0).sum(axis=1)

    raw = np.corrcoef(block.to_numpy(dtype=float), rowvar=False)
    w_raw = leading_vector(raw)

    out = {
        "strength_gini": gini(strength),
        "strength_top10": top_share(strength),
        "degree_central": freeman_centralization(degree),
        "evc_top10": top_share(leading_vector(a)) if a.any() else np.nan,
        "raw_evc_eff_share": 1.0 / (w_raw ** 2).sum() / n,
    }
    return out, pd.Series(strength, index=tickers)


def centrality_over_time(returns: pd.DataFrame, window: int = wn.WINDOW, step: int = wn.STEP,
                         min_stocks: int = wn.MIN_STOCKS, start: str | None = None,
                         end: str | None = None, setting: cl.Setting = SETTING,
                         n_hubs: int = N_HUBS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features, hubs).

    features: one row per window end date with the measures above and n_stocks.
    hubs: date, rank, ticker, strength for the `n_hubs` most connected stocks.
    """
    rows, hubs = {}, []
    for w in wn.windows(returns, window, step, min_stocks, start, end):
        m, strength = window_metrics(w.block, setting)
        m["n_stocks"] = w.n_stocks
        rows[w.end] = m
        top = strength.nlargest(n_hubs)
        hubs += [{"date": w.end, "rank": r, "ticker": t, "strength": s}
                 for r, (t, s) in enumerate(top.items(), start=1)]
    feats = pd.DataFrame.from_dict(rows, orient="index").astype(float)
    feats.index.name = "date"
    return feats, pd.DataFrame(hubs)
