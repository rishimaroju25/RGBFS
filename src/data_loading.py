"""Data collection helpers for the Market Stress Early-Warning System.

Sources
-------
* FRED (Federal Reserve Bank of St. Louis): stress indexes and market series.
  Uses the official FRED API when FRED_API_KEY is set (in a .env file or the
  environment); otherwise falls back to FRED's public CSV download link,
  which needs no key.
* Yahoo Finance via the `yfinance` package: daily prices for SPY and the
  nine original Select Sector SPDR ETFs (all trade back to Dec 1998).

Every loader caches its raw download in data/raw/ so the notebook does not
hit the APIs again on every run. Delete a file in data/raw/ to refresh it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# FRED series used in the first pass. Descriptions are FRED's series titles.
FRED_SERIES = {
    "STLFSI4": "St. Louis Fed Financial Stress Index (weekly)",
    "NFCI": "Chicago Fed National Financial Conditions Index (weekly)",
    "VIXCLS": "CBOE Volatility Index: VIX (daily)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread (daily)",
    "T10Y2Y": "10-Year Treasury minus 2-Year Treasury yield spread (daily)",
}

# SPY plus the nine original sector ETFs. XLRE (2015) and XLC (2018) are left
# out so the panel has full history back to 1999.
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
MARKET_ETF = "SPY"


# ---------------------------------------------------------------- FRED ----
def _fred_api(series_id: str, api_key: str, start: str) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    obs = pd.DataFrame(resp.json()["observations"])
    s = pd.to_numeric(obs["value"], errors="coerce")  # FRED marks missing as "."
    s.index = pd.to_datetime(obs["date"])
    return s


def _fred_csv(series_id: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    # First column is the date (header has been both "DATE" and
    # "observation_date" over time), second is the value.
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s[s.index >= start]


def load_fred(series_ids=None, start: str = "1999-01-01", refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame with one column per FRED series (native frequency)."""
    series_ids = list(series_ids or FRED_SERIES)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    api_key = os.getenv("FRED_API_KEY", "").strip()

    cols = {}
    for sid in series_ids:
        cache = RAW_DIR / f"fred_{sid}.csv"
        if cache.exists() and not refresh:
            s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            s = _fred_api(sid, api_key, start) if api_key else _fred_csv(sid, start)
            s.rename(sid).to_csv(cache)
        cols[sid] = s.rename(sid)
    out = pd.concat(cols.values(), axis=1).sort_index()
    out.index.name = "date"
    return out


# -------------------------------------------------------------- Prices ----
def load_prices(tickers=None, start: str = "1999-01-01", refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted close prices from Yahoo Finance, one column per ticker."""
    tickers = list(tickers or [MARKET_ETF] + SECTOR_ETFS)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "etf_prices.csv"
    if cache.exists() and not refresh:
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
        missing = set(tickers) - set(px.columns)
        if not missing:
            return px[tickers]

    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data; check your connection and try again.")
    px = raw["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px[tickers].sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.index.name = "date"
    px.to_csv(cache)
    return px


# ------------------------------------------------------- Align & label ----
def build_daily_panel(fred: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Put everything on the trading-day calendar of the price data.

    Weekly series (STLFSI4, NFCI) are forward-filled onto trading days, so each
    day carries the latest weekly reading dated on or before it. NOTE: this
    uses the observation date, not the publication date. Before any
    forecasting, shift weekly series by their release lag to avoid look-ahead
    bias.
    """
    days = prices.index
    fred_daily = fred.reindex(fred.index.union(days)).ffill().reindex(days)
    returns = np.log(prices).diff().add_suffix("_ret")
    panel = pd.concat([prices, returns, fred_daily], axis=1)
    return panel.iloc[1:]  # drop first row (no return)


def stress_flag(stlfsi: pd.Series, threshold: float = 1.0) -> pd.Series:
    """Preliminary binary stress label.

    STLFSI4 is constructed so that zero is roughly normal conditions and
    positive values are above-average stress. The threshold here is a
    placeholder for the group to decide (context questions 1 and 2).
    """
    return (stlfsi > threshold).astype(int).rename("stress")


# ------------------------------------------------- First-look features ----
def rolling_avg_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Mean pairwise correlation across assets over a rolling window."""
    vals = returns.to_numpy()
    n = vals.shape[1]
    iu = np.triu_indices(n, k=1)
    out = np.full(len(returns), np.nan)
    for t in range(window - 1, len(returns)):
        w = vals[t - window + 1 : t + 1]
        if np.isnan(w).any():
            continue
        out[t] = np.corrcoef(w, rowvar=False)[iu].mean()
    return pd.Series(out, index=returns.index, name=f"avg_corr_{window}d")


def absorption_ratio(returns: pd.DataFrame, window: int = 250, n_components: int | None = None) -> pd.Series:
    """Share of total return variance explained by the top principal components.

    Simplified version of Kritzman, Li, Page & Rigobon (2011): plain rolling
    covariance (the paper uses exponential weighting and a longer window) and
    about one fifth of the assets as components by default.
    """
    vals = returns.to_numpy()
    n = vals.shape[1]
    k = n_components or max(1, round(n / 5))
    out = np.full(len(returns), np.nan)
    for t in range(window - 1, len(returns)):
        w = vals[t - window + 1 : t + 1]
        if np.isnan(w).any():
            continue
        eig = np.linalg.eigvalsh(np.cov(w, rowvar=False))[::-1]
        out[t] = eig[:k].sum() / eig.sum()
    return pd.Series(out, index=returns.index, name=f"absorption_ratio_{window}d")
