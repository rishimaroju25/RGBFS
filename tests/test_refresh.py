"""Tests for src/refresh.py with a fake downloader (no network)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import prices as px  # noqa: E402
from src import refresh as rf  # noqa: E402


def _frame(dates, close):
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                         "Volume": np.ones(len(dates))}, index=pd.DatetimeIndex(dates, name="Date"))


def test_merge_update_append_same_adjusted():
    old = _frame(pd.bdate_range("2026-01-01", periods=10), np.arange(10.0) + 1)
    new = _frame(pd.bdate_range("2026-01-08", periods=8), np.arange(5.0, 13.0) + 1)
    status, merged = rf.merge_update(old, new)
    assert status == "append" and len(merged) == 13 and merged.index.is_monotonic_increasing
    assert rf.merge_update(old, old.iloc[-3:])[0] == "same"
    split = new.assign(Close=new["Close"] / 2)
    assert rf.merge_update(old, split)[0] == "adjusted"


def _fake(histories):
    """Downloader stub: returns the slice of each full history between start and end."""
    def dl(tickers, start=None, end=None, **kw):
        parts = {}
        for t in tickers:
            h = histories[t]
            h = h[h.index >= pd.Timestamp(start)]
            parts[t] = h
        return pd.concat(parts, axis=1).swaplevel(0, 1, axis=1)
    return dl


def test_update_cached_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(px, "RAW_PRICE_DIR", tmp_path)
    days = pd.bdate_range("2026-01-01", periods=40)
    full = {"AAA": _frame(days, np.arange(40.0) + 10),
            "BBB": _frame(days, np.arange(40.0) + 10),
            "OLD": _frame(days[:5], np.ones(5))}
    full["AAA"].iloc[:30].to_parquet(px._cache_path("AAA"))                       # normal: append 10 days
    full["BBB"].iloc[:30].assign(Close=lambda d: d["Close"] * 2).to_parquet(px._cache_path("BBB"))  # split since: rebuild
    full["OLD"].to_parquet(px._cache_path("OLD"))                                 # delisted: skip
    rep = rf.update_cached_prices(["AAA", "BBB", "OLD"], downloader=_fake(full), pause=0)
    assert rep.appended == {"AAA": 10}
    assert rep.rebuilt == ["BBB"] and rep.skipped_delisted == ["OLD"]
    assert len(pd.read_parquet(px._cache_path("AAA"))) == 40
    assert np.allclose(pd.read_parquet(px._cache_path("BBB"))["Close"], full["BBB"]["Close"])
