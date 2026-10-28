"""Stress-period labels for the Market Stress Early-Warning System.

Owner: Salah (Stress labels & benchmarks). Rule proposal: docs/stress_label_rule.md

Rule (proposed, needs team agreement)
-------------------------------------
1. Daily realized volatility of the S&P 500 index (^GSPC): standard deviation
   of daily simple returns over a CENTERED 21-trading-day window (10 days
   before, the day itself, 10 days after), annualized with sqrt(252).
2. A day is a stress day if that volatility is at least 25% (fixed level).
3. Stress days less than 20 trading days apart are merged into one episode.
4. Episodes shorter than 10 trading days are dropped.

Why these choices (details in the doc):
* Built only from ^GSPC prices, never from VIX or STLFSI4, because Week 5/6
  tests whether our signals beat those indexes. Labels built from them would
  make that test circular.
* A fixed 25% level instead of a sample percentile, so adding new data never
  changes past labels.
* A centered window so episode start dates are not pushed late by a trailing
  window, which would inflate measured alert lead time.

LOOKAHEAD WARNING: labels deliberately use future returns (the centered
window looks 10 days ahead). They are evaluation targets only. Never use
`stress`, `vol_centered` or anything derived from them as a model feature.
The last 10 trading days of the data cannot be labeled and are left missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLD = 0.25    # annualized realized volatility
WINDOW = 21         # trading days, centered
MERGE_GAP = 20      # trading days between stress days that still count as one episode
MIN_LENGTH = 10     # trading days; shorter episodes are dropped

# Descriptive names for episodes found with the default rule on data through
# 2026-10. Names are for readability only; verify against a primary source
# before citing them in the report. Keyed by episode start date.
EPISODE_NAMES = {
    "2007-08-03": "Aug 2007 credit and quant-fund stress",
    "2008-03-04": "Bear Stearns",
    "2008-08-29": "Global financial crisis (Lehman)",
    "2010-04-26": "Flash crash and first euro-area debt stress",
    "2011-07-25": "U.S. debt ceiling, S&P downgrade and euro-area debt crisis",
    "2015-08-12": "China devaluation selloff",
    "2018-02-01": "Feb 2018 volatility spike",
    "2018-12-11": "Q4 2018 selloff",
    "2020-02-14": "COVID-19 crash",
    "2022-02-23": "Russia's invasion of Ukraine",
    "2022-04-20": "2022 rate-hike bear market (first leg)",
    "2022-09-19": "2022 rate-hike bear market (second leg)",
    "2025-03-20": "Spring 2025 tariff selloff",
}


def realized_vol(prices: pd.Series, window: int = WINDOW, centered: bool = True) -> pd.Series:
    """Annualized realized volatility of daily simple returns."""
    rets = prices.pct_change(fill_method=None)
    return rets.rolling(window, center=centered).std() * np.sqrt(252)


def labels_from_flag(flag: pd.Series, vol: pd.Series, merge_gap: int = MERGE_GAP,
                     min_length: int = MIN_LENGTH) -> pd.DataFrame:
    """Turn a daily stress-day flag into episodes (merge close days, drop short runs)."""
    raw = flag.fillna(False).astype(bool)
    pos = np.flatnonzero(raw.to_numpy())
    episode = pd.Series(np.nan, index=flag.index)
    eid = 0
    if len(pos):
        start = prev = pos[0]
        for p in list(pos[1:]) + [None]:
            if p is not None and p - prev <= merge_gap:
                prev = p
                continue
            if prev - start + 1 >= min_length:
                eid += 1
                episode.iloc[start:prev + 1] = eid
            if p is not None:
                start = prev = p

    return pd.DataFrame({
        "vol_centered": vol,
        "stress_day_raw": raw.astype(float).where(vol.notna()),
        "episode_id": episode,
        "stress": episode.notna().astype(float).where(vol.notna()),
    })


def daily_labels(prices: pd.Series, threshold: float = THRESHOLD, window: int = WINDOW,
                 merge_gap: int = MERGE_GAP, min_length: int = MIN_LENGTH,
                 centered: bool = True) -> pd.DataFrame:
    """One row per trading day: vol, raw stress-day flag, episode id and final label.

    `stress` is 1 inside a kept episode, 0 outside, and missing where the
    volatility window is incomplete (start and end of the data).
    `centered=False` gives the trailing-window version, for comparison only.
    """
    prices = prices.dropna()
    vol = realized_vol(prices, window=window, centered=centered)
    return labels_from_flag(vol >= threshold, vol, merge_gap, min_length)


def episode_table(labels: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """One row per episode with dates, length and severity measures."""
    prices = prices.dropna()
    peak_252 = prices.rolling(252, min_periods=1).max()
    drawdown = prices / peak_252 - 1
    rows = []
    for eid, block in labels.dropna(subset=["episode_id"]).groupby("episode_id"):
        s, e = block.index[0], block.index[-1]
        dd = drawdown.loc[s:e]
        rows.append({
            "episode_id": int(eid),
            "start": s.date(),
            "end": e.date(),
            "trading_days": len(block),
            "peak_vol": round(block["vol_centered"].max(), 3),
            "peak_vol_date": block["vol_centered"].idxmax().date(),
            "max_drawdown_252d": round(dd.min(), 3),
            "trough_date": dd.idxmin().date(),
            "name": EPISODE_NAMES.get(str(s.date()), "unnamed: check and add a name"),
        })
    return pd.DataFrame(rows).set_index("episode_id")
