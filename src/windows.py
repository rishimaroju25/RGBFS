"""Rolling windows over the cleaned returns, without looking ahead.

Owner: Rishi (Data pipeline, Week 2). Used by network (Brandon), clustering
(Grace) and anomaly detection.

Rules every window follows
--------------------------
* **Trailing only.** The window ending on day t uses days t-window+1 .. t.
  A feature dated t never uses data after t, so it is safe for forecasting.
* **Point-in-time stocks.** A stock is in a window only if it has a return
  on every day of that window. Its returns are already NaN outside its index
  membership (src/returns.py), so a stock that joined or left the S&P 500
  inside the window is left out, and so is a stock with a missing price.
* **Enough stocks.** Windows with fewer than `min_stocks` usable stocks are
  skipped, because correlations from a handful of stocks are noisy.

Main functions
--------------
windows(returns, window, step)        yields Window(end, block) objects
correlation_at(returns, end, window)  correlation matrix for one window (network prototype)
rolling_apply(returns, func, ...)     runs any function on every window -> DataFrame
connectedness(returns, ...)           avg pairwise correlation, PC1 share, absorption ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd

WINDOW = 60       # trading days, matches the EDA notebooks
STEP = 5          # compute every 5th trading day by default
MIN_STOCKS = 100


@dataclass
class Window:
    end: pd.Timestamp        # last day in the window (the date the result is stamped with)
    start: pd.Timestamp      # first day in the window
    block: pd.DataFrame      # window x stocks, complete data only

    @property
    def n_stocks(self) -> int:
        return self.block.shape[1]


def _complete(block: pd.DataFrame) -> pd.DataFrame:
    block = block.loc[:, block.notna().all()]
    return block.loc[:, block.std() > 0]


def windows(returns: pd.DataFrame, window: int = WINDOW, step: int = STEP,
            min_stocks: int = MIN_STOCKS, start: str | None = None,
            end: str | None = None) -> Iterator[Window]:
    """Yield trailing windows. `start`/`end` limit which END dates are produced;
    earlier history is still used as each window's lookback."""
    idx = returns.index
    ends = np.arange(window - 1, len(idx), step)
    if start is not None:
        ends = ends[idx[ends] >= pd.Timestamp(start)]
    if end is not None:
        ends = ends[idx[ends] <= pd.Timestamp(end)]
    vals = returns.to_numpy()
    cols = returns.columns
    for e in ends:
        raw = vals[e - window + 1:e + 1]
        ok = ~np.isnan(raw).any(axis=0)
        if ok.sum() < min_stocks:
            continue
        block = pd.DataFrame(raw[:, ok], index=idx[e - window + 1:e + 1], columns=cols[ok])
        block = block.loc[:, block.std() > 0]
        if block.shape[1] < min_stocks:
            continue
        yield Window(end=idx[e], start=idx[e - window + 1], block=block)


def correlation_at(returns: pd.DataFrame, end: str | pd.Timestamp, window: int = WINDOW,
                   min_stocks: int = MIN_STOCKS) -> pd.DataFrame:
    """Correlation matrix (stocks x stocks) for the window ending on `end`.

    If `end` is not a trading day, the last trading day on or before it is used.
    """
    idx = returns.index
    pos = idx.searchsorted(pd.Timestamp(end), side="right") - 1
    if pos < window - 1:
        raise ValueError(f"Not enough history before {end} for a {window}-day window")
    block = _complete(returns.iloc[pos - window + 1:pos + 1])
    if block.shape[1] < min_stocks:
        raise ValueError(f"Only {block.shape[1]} stocks have complete data in that window (min {min_stocks})")
    return block.corr()


def rolling_apply(returns: pd.DataFrame, func: Callable[[pd.DataFrame], dict], window: int = WINDOW,
                  step: int = STEP, min_stocks: int = MIN_STOCKS, start: str | None = None,
                  end: str | None = None) -> pd.DataFrame:
    """Run `func(block) -> dict of numbers` on every window; one row per window end date.

    Adds n_stocks automatically. Example:
        rolling_apply(rets, lambda b: {"mean_vol": b.std().mean()})
    """
    rows = {}
    for w in windows(returns, window, step, min_stocks, start, end):
        out = dict(func(w.block))
        out["n_stocks"] = w.n_stocks
        rows[w.end] = out
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "date"
    return df.astype(float)


def connectedness_metrics(block: pd.DataFrame, ar_share: float = 0.2) -> dict:
    """Co-movement measures for one window.

    avg_corr:   mean pairwise correlation
    pc1_share:  largest eigenvalue of the correlation matrix / number of stocks
    absorption: share of total variance explained by the top `ar_share` of
                the covariance matrix's eigenvectors (Kritzman et al., 2011 use
                about one fifth of the assets; this version uses plain, not
                exponentially weighted, covariance)
    """
    x = block.to_numpy()
    c = np.corrcoef(x, rowvar=False)
    n = c.shape[0]
    eig_c = np.linalg.eigvalsh(c)[::-1]
    eig_v = np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1]
    k = max(1, int(round(n * ar_share)))
    return {
        "avg_corr": (c.sum() - n) / (n * (n - 1)),
        "pc1_share": eig_c[0] / n,
        "absorption": eig_v[:k].sum() / eig_v.sum(),
    }


def connectedness(returns: pd.DataFrame, window: int = WINDOW, step: int = STEP,
                  min_stocks: int = MIN_STOCKS, start: str | None = None,
                  end: str | None = None) -> pd.DataFrame:
    """avg_corr, pc1_share, absorption and n_stocks for every window end date."""
    return rolling_apply(returns, connectedness_metrics, window, step, min_stocks, start, end)
