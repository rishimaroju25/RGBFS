"""Download and assemble daily prices for point-in-time S&P 500 members.

Pipeline
--------
1. `src.universe` gives every ticker that was in the S&P 500 at any point
   since START (about 980 tickers for 2005 onward).
2. `download_prices` pulls daily split- and dividend-adjusted prices from
   Yahoo Finance with `yfinance`, in batches, and caches one parquet file per
   ticker in data/raw/prices/. Re-running skips tickers already cached, so an
   interrupted download can simply be restarted.
3. `assemble` builds wide tables (dates x tickers) of adjusted close and
   volume, the membership mask on the same dates, and a coverage report that
   shows which members Yahoo has no data for.

Known limitation: Yahoo Finance mostly keeps history only for symbols that
still trade. Many firms that went bankrupt or were acquired (for example
Lehman Brothers, Bear Stearns, Washington Mutual) will come back empty. The
coverage report measures that gap; docs/survivorship_bias.md explains how we
handle it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import universe as uni

ROOT = Path(__file__).resolve().parents[1]
RAW_PRICE_DIR = ROOT / "data" / "raw" / "prices"
PROCESSED_DIR = ROOT / "data" / "processed"
EMPTY_LOG = ROOT / "data" / "raw" / "prices_empty.csv"

START = "2005-01-01"
MIN_ROWS = 20  # a cached ticker with fewer rows than this is treated as a broken download
BENCHMARKS = ["^GSPC", "SPY"]  # index level and the ETF; used for EDA comparisons


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("/", "_")
    return RAW_PRICE_DIR / f"{safe}.parquet"


def _cached_rows(ticker: str) -> int:
    """Number of rows in a ticker's cache file (0 if it does not exist)."""
    p = _cache_path(ticker)
    if not p.exists():
        return 0
    import pyarrow.parquet as pq

    return pq.ParquetFile(p).metadata.num_rows


def _read_empty_log() -> set[str]:
    if EMPTY_LOG.exists():
        return set(pd.read_csv(EMPTY_LOG)["ticker"])
    return set()


def _split_download(raw: pd.DataFrame, yahoo_tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Turn yfinance's (field, ticker) column MultiIndex into one frame per ticker."""
    out = {}
    if raw is None or raw.empty:
        return out
    if not isinstance(raw.columns, pd.MultiIndex):  # defensive: single-level columns
        raw = pd.concat({yahoo_tickers[0]: raw}, axis=1).swaplevel(0, 1, axis=1)
    fields = raw.columns.get_level_values(0)
    tick_level = 1
    if "Close" not in set(fields):  # group_by="ticker" layout
        tick_level = 0
    for yt in yahoo_tickers:
        try:
            sub = raw.xs(yt, axis=1, level=tick_level)
        except KeyError:
            continue
        sub = sub.dropna(how="all")
        if sub.empty or "Close" not in sub:
            continue
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub]
        out[yt] = sub[cols].astype("float64")
    return out


def download_prices(
    tickers: list[str],
    start: str = START,
    end: str | None = None,
    batch_size: int = 50,
    pause: float = 2.0,
    max_retries: int = 3,
    downloader=None,
    retry_empty: bool = False,
) -> dict:
    """Download and cache daily prices. `tickers` use S&P-style symbols (BRK.B).

    `downloader` defaults to `yfinance.download`; tests pass a stub.
    `retry_empty=True` tries again for tickers logged as empty on earlier runs.
    Returns a summary dict with counts of cached, downloaded and empty tickers.

    Why a ticker is retried on its own before being logged as empty: when
    Yahoo rate-limits a batch, yfinance can return empty columns instead of
    raising an error. Without the retry, healthy tickers were logged as "no
    data" and skipped on every later run.

    Cached files with fewer than MIN_ROWS rows are downloaded again: some
    tickers in the first download (for example AVB, EQR, EA) came back with a
    single day of prices, which made long-time members look unpriced.
    """
    if downloader is None:
        import yfinance as yf

        downloader = yf.download

    RAW_PRICE_DIR.mkdir(parents=True, exist_ok=True)
    empty = set() if retry_empty else _read_empty_log()
    old_empty = _read_empty_log()
    short = [t for t in tickers if 0 < _cached_rows(t) < MIN_ROWS]
    for t in short:
        _cache_path(t).unlink()
    todo = [t for t in tickers if not _cache_path(t).exists() and t not in empty]
    print(f"{len(tickers)} tickers: {len(tickers) - len(todo)} already done, {len(todo)} to download")
    if short:
        print(f"  re-downloading {len(short)} tickers whose cached prices had under {MIN_ROWS} rows: "
              + ", ".join(short[:20]) + (" ..." if len(short) > 20 else ""))

    new_empty = []
    n_saved = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        ymap = {uni.to_yahoo(t): t for t in batch}
        raw = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = downloader(
                    list(ymap),
                    start=start,
                    end=end,
                    auto_adjust=True,  # adjusted for splits and dividends
                    actions=False,
                    group_by="column",
                    threads=True,
                    progress=False,
                    multi_level_index=True,
                )
                break
            except Exception as exc:  # network errors, rate limits
                wait = pause * 5 * attempt
                print(f"  batch {i // batch_size + 1}: attempt {attempt} failed ({exc!r}); waiting {wait:.0f}s")
                time.sleep(wait)
        if raw is None:
            print(f"  batch {i // batch_size + 1}: giving up for now; re-run the script to retry")
            continue
        frames = _split_download(raw, list(ymap))
        missing = [yt for yt in ymap if yt not in frames]
        if missing:
            # Second chance, one ticker at a time, so a rate-limited batch
            # does not get logged as permanently empty.
            time.sleep(pause)
            for yt in missing:
                try:
                    one = downloader(
                        [yt], start=start, end=end, auto_adjust=True, actions=False,
                        group_by="column", threads=False, progress=False, multi_level_index=True,
                    )
                    frames.update(_split_download(one, [yt]))
                except Exception as exc:
                    print(f"  {yt}: single retry failed ({exc!r})")
                time.sleep(pause / 4)
        for yt, t in ymap.items():
            if yt in frames:
                frames[yt].to_parquet(_cache_path(t))
                n_saved += 1
            else:
                new_empty.append(t)
        done = min(i + batch_size, len(todo))
        print(f"  {done}/{len(todo)} processed ({n_saved} saved, {len(new_empty)} empty)")
        time.sleep(pause)

    # Rewrite the log: keep earlier empties that were not retried, drop any
    # that now have prices, add the new ones.
    still_empty = {t for t in old_empty if not _cache_path(t).exists()} | set(new_empty)
    if still_empty or EMPTY_LOG.exists():
        pd.DataFrame({"ticker": sorted(still_empty)}).to_csv(EMPTY_LOG, index=False)
    recovered = len(old_empty - still_empty)
    still_short = [t for t in tickers if 0 < _cached_rows(t) < MIN_ROWS]
    if still_short:
        print(f"  still under {MIN_ROWS} rows after download (check by hand): {', '.join(still_short)}")
    return {"requested": len(tickers), "downloaded": n_saved, "empty_new": len(new_empty),
            "empty_total": len(still_empty), "recovered_from_empty_log": recovered,
            "short_redownloaded": len(short), "still_short": len(still_short)}


def load_cached(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        p = _cache_path(t)
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            out[t] = df[~df.index.duplicated(keep="last")]
    return out


def assemble(changes: pd.DataFrame, start: str = START, end: str | None = None, calendar_ticker: str = "^GSPC"):
    """Build wide price tables, membership mask and coverage report from the cache.

    Returns (close, volume, membership, report). `close` and `volume` hold ALL
    downloaded prices, including days when a stock was not in the index. Use
    `membership` (or `masked_close`) to keep only member days.
    """
    tickers = uni.tickers_in_period(changes, start, end)
    data = load_cached(tickers + BENCHMARKS + sorted(set(uni.TICKER_ALIASES.values())))

    # Renamed tickers: give the old symbol the current symbol's history. The
    # membership mask keeps each symbol to its own member days, so the
    # company is never counted twice on the same day.
    aliased = {}
    for old, new in uni.TICKER_ALIASES.items():
        if old in tickers and old not in data and new in data:
            data[old] = data[new]
            aliased[old] = new

    if calendar_ticker in data:
        dates = data[calendar_ticker].index
    else:  # fall back to the union of all stock dates
        dates = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    dates = dates[(dates >= pd.Timestamp(start)) & ((dates <= pd.Timestamp(end)) if end else True)]

    close = pd.DataFrame({t: d["Close"] for t, d in data.items()}).reindex(dates)
    volume = pd.DataFrame({t: d["Volume"] for t, d in data.items() if "Volume" in d}).reindex(dates)
    membership = uni.membership_matrix(changes, dates)[tickers]

    stock_close = close.reindex(columns=tickers)
    has_price = stock_close.notna()
    spells = uni.membership_spells(changes).groupby("ticker")["n_spells"].first()
    member_days = membership.sum()
    priced_member_days = (membership & has_price).sum()
    first_px = stock_close.apply(lambda s: s.first_valid_index())
    last_px = stock_close.apply(lambda s: s.last_valid_index())
    first_member = membership.idxmax().where(member_days > 0)
    report = pd.DataFrame({
        "member_days": member_days,
        "priced_member_days": priced_member_days,
        "coverage": (priced_member_days / member_days.replace(0, np.nan)).round(4),
        "first_price": first_px,
        "last_price": last_px,
        "first_member_day": first_member,
        "n_spells": spells.reindex(tickers).fillna(1).astype(int),
    })
    report["no_data"] = report["priced_member_days"] == 0
    # Prices that start well after membership began usually mean Yahoo has the
    # symbol's newer history only (or the symbol now belongs to another firm).
    gap = (report["first_price"] - report["first_member_day"]).dt.days
    report["late_start"] = (gap > 30) & ~report["no_data"]
    report["check_ticker_reuse"] = report["n_spells"] > 1
    report["prices_from"] = pd.Series(aliased).reindex(report.index)
    report.index.name = "ticker"
    return close, volume, membership, report.sort_values("member_days", ascending=False)


def masked_close(close: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Adjusted close for index members only; NaN on days a stock was not a member.

    Columns are every ticker in `membership`, so members Yahoo has no data for
    appear as all-NaN columns instead of silently disappearing.
    """
    return close.reindex(columns=membership.columns).where(membership)


def coverage_by_day(close: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Per trading day: number of members, members with a price, and the share priced."""
    cols = membership.columns
    has = close.reindex(columns=cols).notna() & membership
    out = pd.DataFrame({"members": membership.sum(axis=1), "priced": has.sum(axis=1)})
    out["share_priced"] = out["priced"] / out["members"]
    return out


def save_processed(close, volume, membership, report) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    close.to_parquet(PROCESSED_DIR / "sp500_close.parquet")
    volume.to_parquet(PROCESSED_DIR / "sp500_volume.parquet")
    membership.to_parquet(PROCESSED_DIR / "sp500_membership.parquet")
    report.to_csv(PROCESSED_DIR / "sp500_coverage_report.csv")
