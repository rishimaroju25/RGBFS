"""Bring the cached price files up to date without losing history (Week 7).

Owner: Salah. Used by scripts/refresh_data.py.

Why not just re-run scripts/download_sp500_prices.py: that script only
downloads tickers that have no cache file yet, so cached tickers never get new
days. Deleting the cache and downloading again is worse: Yahoo removes a
company's history after it is acquired or delisted, so those stocks would be
lost for good (see docs/survivorship_bias.md).

What update_cached_prices does, per ticker that is still trading:
1. Download from a few days before the last cached date to today.
2. Check the overlapping days. Yahoo's prices are adjusted for splits and
   dividends, so a split or dividend since the last download rescales the
   whole history. If the overlap no longer matches, download the full history
   again and replace the file, but only if the new history starts no later
   than the old one (so nothing is lost). Otherwise keep the old file and
   report the ticker for a manual check.
3. If the overlap matches, append only the new days.

Tickers whose cache ended more than `stale_days` before the newest cache date
are treated as delisted and left alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from src import prices as px
from src import universe as uni

OVERLAP_DAYS = 7          # calendar days of overlap used to check the adjustment
TOLERANCE = 1e-3          # max relative difference on overlapping closes (0.1%)
STALE_DAYS = 30           # calendar days behind the newest cache = delisted, skip


@dataclass
class UpdateReport:
    appended: dict = field(default_factory=dict)       # ticker -> new rows
    rebuilt: list = field(default_factory=list)        # full history re-downloaded (split or dividend)
    kept_old: list = field(default_factory=list)       # adjustment changed but full history was shorter: check by hand
    no_new_data: list = field(default_factory=list)
    skipped_delisted: list = field(default_factory=list)
    failed: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"appended": len(self.appended), "rows_added": sum(self.appended.values()),
                "rebuilt": len(self.rebuilt), "kept_old_check_by_hand": len(self.kept_old),
                "no_new_data": len(self.no_new_data), "skipped_delisted": len(self.skipped_delisted),
                "failed": len(self.failed)}


def _read(t: str) -> pd.DataFrame:
    df = pd.read_parquet(px._cache_path(t))
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def _download(downloader, yahoo, start, end):
    raw = downloader(yahoo, start=start, end=end, auto_adjust=True, actions=False, group_by="column",
                     threads=True, progress=False, multi_level_index=True)
    frames = px._split_download(raw, yahoo)
    for yt, df in frames.items():
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "Date"
    return frames


def merge_update(old: pd.DataFrame, new: pd.DataFrame, tolerance: float = TOLERANCE) -> tuple[str, pd.DataFrame]:
    """Combine a cached history with a recent download.

    Returns (status, frame): "append" with old + new days, "same" with old
    unchanged, or "adjusted" (overlap does not match; caller should
    re-download the full history) with old unchanged.
    """
    new = new.dropna(subset=["Close"])
    overlap = old.index.intersection(new.index)
    if len(overlap):
        ratio = new.loc[overlap, "Close"] / old.loc[overlap, "Close"]
        if (ratio - 1).abs().max() > tolerance:
            return "adjusted", old
    fresh = new[new.index > old.index.max()]
    if fresh.empty:
        return "same", old
    return "append", pd.concat([old, fresh[old.columns.intersection(fresh.columns)]])


def update_cached_prices(tickers: list[str], end: str | None = None, batch_size: int = 50,
                         pause: float = 2.0, stale_days: int = STALE_DAYS, downloader=None) -> UpdateReport:
    """Append new days to every cached ticker that is still trading (see module docstring)."""
    if downloader is None:
        import yfinance as yf

        downloader = yf.download
    rep = UpdateReport()
    cached = {t: _read(t) for t in tickers if px._cache_path(t).exists()}
    if not cached:
        return rep
    newest = max(df.index.max() for df in cached.values())
    live = {}
    for t, df in cached.items():
        if df.index.max() < newest - pd.Timedelta(days=stale_days):
            rep.skipped_delisted.append(t)
        else:
            live[t] = df
    start = (min(df.index.max() for df in live.values()) - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")

    names = sorted(live)
    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        ymap = {uni.to_yahoo(t): t for t in batch}
        try:
            frames = _download(downloader, list(ymap), start, end)
        except Exception as exc:  # network errors, rate limits
            print(f"  batch {i // batch_size + 1}: download failed ({exc!r})")
            rep.failed += batch
            continue
        for yt, t in ymap.items():
            if yt not in frames:
                rep.failed.append(t)
                continue
            status, merged = merge_update(live[t], frames[yt])
            if status == "append":
                rep.appended[t] = len(merged) - len(live[t])
                merged.to_parquet(px._cache_path(t))
            elif status == "same":
                rep.no_new_data.append(t)
            else:
                try:
                    full = _download(downloader, [yt], px.START, end).get(yt)
                except Exception:
                    full = None
                old = live[t]
                if full is not None and not full.empty and full.index.min() <= old.index.min() + pd.Timedelta(days=7):
                    full.to_parquet(px._cache_path(t))
                    rep.rebuilt.append(t)
                else:
                    rep.kept_old.append(t)
                time.sleep(pause / 4)
        print(f"  {min(i + batch_size, len(names))}/{len(names)} checked")
        time.sleep(pause)
    return rep
