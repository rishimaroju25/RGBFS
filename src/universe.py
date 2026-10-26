"""Point-in-time S&P 500 membership.

Source
------
fja05680/sp500 on GitHub (MIT license):
https://github.com/fja05680/sp500

The file "S&P 500 Historical Components & Changes (Updated).csv" has one row
per index change date (1996 onward) with the full comma-separated list of
member tickers on that date. Tickers are the ones in use on that date, and
bankrupt or delisted firms appear under their later OTC symbols (for example
Lehman Brothers is LEHMQ, Washington Mutual is WAMUQ, Fannie Mae is FNMA).

The maintainer updates the file by hand every couple of months from the
Wikipedia S&P 500 page, so the most recent index changes can be missing.
Check `membership_last_date()` against today's date before trusting the end
of the series.

Why this matters: using today's S&P 500 list for 2005 to 2008 would keep only
the firms that survived, which removes exactly the firms that failed during
the stress periods we study (survivorship bias). See docs/survivorship_bias.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MEMBERSHIP_CACHE = RAW_DIR / "sp500_membership_changes.csv"


def load_membership_changes(refresh: bool = False, path: str | Path | None = None) -> pd.DataFrame:
    """Return the raw change table: one row per change date, `tickers` as a list.

    Downloads once from GitHub and caches to data/raw/. Pass `path` to read a
    local copy instead (useful offline).
    """
    if path is None:
        if refresh or not MEMBERSHIP_CACHE.exists():
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            pd.read_csv(MEMBERSHIP_URL).to_csv(MEMBERSHIP_CACHE, index=False)
        path = MEMBERSHIP_CACHE
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["tickers"] = df["tickers"].str.split(",").apply(lambda xs: sorted({x.strip() for x in xs if x.strip()}))
    return df.sort_values("date").reset_index(drop=True)


def membership_last_date(changes: pd.DataFrame) -> pd.Timestamp:
    """Date of the most recent change recorded in the source file."""
    return changes["date"].max()


def membership_matrix(changes: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean DataFrame (dates x tickers): True when the ticker was a member.

    Each change row is the full member list from that date until the next
    change, so membership on any date is the most recent row on or before it.
    Dates before the first change row are all False.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    all_tickers = sorted({t for ts in changes["tickers"] for t in ts})
    rows = pd.DataFrame(False, index=changes["date"], columns=all_tickers)
    for d, ts in zip(changes["date"], changes["tickers"]):
        rows.loc[d, ts] = True
    rows = rows[~rows.index.duplicated(keep="last")]
    out = rows.reindex(rows.index.union(dates)).ffill().reindex(dates)
    return out.fillna(False).astype(bool)


def tickers_in_period(changes: pd.DataFrame, start: str, end: str | None = None) -> list[str]:
    """Every ticker that was a member at any point between start and end."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end else changes["date"].max()
    # Include the last change row before `start`: it describes membership on `start`.
    before = changes[changes["date"] <= start].tail(1)
    during = changes[(changes["date"] > start) & (changes["date"] <= end)]
    rows = pd.concat([before, during])
    return sorted({t for ts in rows["tickers"] for t in ts})


def membership_spells(changes: pd.DataFrame) -> pd.DataFrame:
    """One row per continuous membership spell: ticker, start, end (NaT = still a member)."""
    dates = pd.DatetimeIndex(changes["date"])
    mat = membership_matrix(changes, dates)
    records = []
    for t in mat.columns:
        on = mat[t]
        starts = on.index[on & ~on.shift(1, fill_value=False)]
        # `end` is the first change date on which the ticker is no longer listed.
        ends = on.index[~on & on.shift(1, fill_value=False)]
        for s in starts:
            later = ends[ends > s]
            records.append({"ticker": t, "start": s, "end": later[0] if len(later) else pd.NaT})
    spells = pd.DataFrame(records).sort_values(["ticker", "start"]).reset_index(drop=True)
    spells["n_spells"] = spells.groupby("ticker")["ticker"].transform("size")
    return spells


def to_yahoo(ticker: str) -> str:
    """Yahoo Finance uses '-' where the S&P list uses '.', e.g. BRK.B -> BRK-B."""
    return ticker.replace(".", "-")


def from_yahoo(ticker: str) -> str:
    return ticker.replace("-", ".")
