"""Clean daily returns for point-in-time S&P 500 members.

Owner: Rishi (Data pipeline, Week 2). Used by every feature notebook.

What "clean" means here
-----------------------
1. **Returns come from unmasked prices, then are masked to member days.**
   A stock's first day in the index gets a real return (its price the day
   before joining is known), and days outside the index are NaN.
2. **No filling across gaps.** If yesterday's price is missing, today's
   return is NaN instead of a multi-day return disguised as one day.
3. **Stale prices are removed.** Five or more identical closes in a row on
   member days usually means Yahoo kept repeating a dead quote. Those
   zero returns would make a stock look perfectly calm and pull
   correlations toward zero.
4. **Price spikes are removed, real crashes are kept.** A one-day move above
   40% is flagged. If the price is back within 10% of its pre-jump level on
   the very next trading day, it is a one-day data blip and both legs are
   set to NaN. Anything slower is kept: real crisis stocks can swing hard
   for days (AIG fell 45% and recovered over two days during its September
   2008 bailout; a 3-day version of this rule wrongly removed that). Large
   moves that do not reverse (OXY -52% on 2020-03-09) are kept. Every flag
   is listed in the quality report so it can be checked by hand.
5. **Repeated jumps are flagged for review.** A ticker with 3 or more
   flagged moves within a year usually has prices from a different security
   (a reused symbol), as CPWR and EP did in 2011. These are listed, not
   removed automatically.
6. **Manual overrides win.** `docs/return_overrides.csv` lists rows the team
   has checked: ticker, start, end, action (`keep` or `drop`), note. A row
   applies to every member day from start to end.

What it cannot fix: firms Yahoo has no prices for (mostly failed or
acquired firms) and the final "delisting" return of a failed firm, which is
never in the data. See docs/survivorship_bias.md.

Lookahead note: the spike rule looks 1 day ahead to decide whether a
past price was an error. That is data cleaning of the historical record,
not a model feature, but a real-time system would only know about a spike
a day later. Keep this in mind for the Week 5 lookahead checklist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OVERRIDES = ROOT / "docs" / "return_overrides.csv"

JUMP = 0.40          # one-day absolute return that gets flagged
REVERT_DAYS = 1      # a spike must reverse on the next trading day (3 days removed real AIG 2008 moves)
REVERT_BAND = 0.10   # "reversed" = back within 10% of the pre-jump price
REPEAT_JUMPS = 3     # flagged moves within REPEAT_WINDOW days that trigger a review flag
REPEAT_WINDOW = 252
STALE_RUN = 5        # identical closes in a row that count as stale
BENCHMARKS = ("^GSPC", "SPY")


@dataclass
class ReturnsResult:
    returns: pd.DataFrame          # dates x tickers, member days only, cleaned
    quality: pd.DataFrame          # one row per removed or flagged return
    summary: pd.Series             # counts for the docs and the notebook
    market: pd.DataFrame           # ^GSPC return and equal-weighted member return


def load_overrides(path: Path | str = OVERRIDES) -> pd.DataFrame:
    """Team-checked decisions: columns ticker, start, end, action (keep/drop), note."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "start", "end", "action", "note"])
    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"].fillna(df["start"]))
    df["action"] = df["action"].str.strip().str.lower()
    bad = set(df["action"]) - {"keep", "drop"}
    if bad:
        raise ValueError(f"{path}: action must be keep or drop, found {sorted(bad)}")
    return df


def _stale_mask(close: pd.DataFrame, membership: pd.DataFrame, run: int = STALE_RUN) -> pd.DataFrame:
    """True on member days that sit inside a run of `run`+ identical closes."""
    same = close.eq(close.shift(1)) & close.notna()
    out = pd.DataFrame(False, index=close.index, columns=close.columns)
    for t in close.columns:
        s = same[t].to_numpy()
        if not s.any():
            continue
        # run id for consecutive True values; a run of k "same" days means k+1 equal closes
        grp = np.cumsum(~s)
        lengths = pd.Series(s).groupby(grp).transform("sum").to_numpy()
        stale = s & (lengths + 1 >= run)
        out[t] = stale
    return out & membership.reindex_like(close).fillna(False).astype(bool)


def _spike_mask(close: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag large moves; mark the ones that reverse quickly as spikes.

    Returns (flagged, spike): boolean frames. A spike also blanks the return
    on the day the price comes back, so neither leg of the error survives.
    """
    flagged = raw.abs() > JUMP
    spike = pd.DataFrame(False, index=raw.index, columns=raw.columns)
    rows, cols = np.nonzero(flagged.to_numpy())
    px = close.to_numpy()
    for i, j in zip(rows, cols):
        if i == 0 or np.isnan(px[i - 1, j]):
            continue
        before = px[i - 1, j]
        ahead = px[i + 1:i + 1 + REVERT_DAYS, j]
        back = np.where(np.abs(ahead / before - 1) <= REVERT_BAND)[0]
        if len(back):
            k = i + 1 + back[0]
            spike.iat[i, j] = True
            spike.iloc[i + 1:k + 1, j] = True  # returns until the price is back
    return flagged, spike


def clean_returns(close: pd.DataFrame, membership: pd.DataFrame,
                  overrides: pd.DataFrame | None = None) -> ReturnsResult:
    """Build the cleaned, member-days-only daily simple returns.

    `close` is the adjusted close table from data/processed/sp500_close.parquet
    (dates x tickers, may include ^GSPC and SPY). `membership` is the boolean
    point-in-time table from sp500_membership.parquet.
    """
    tickers = membership.columns
    px = close.reindex(columns=tickers).astype(float)
    mem = membership.reindex(index=px.index).fillna(False).astype(bool)

    raw = px.pct_change(fill_method=None)              # NaN after any gap: no filling
    stale = _stale_mask(px, mem)
    flagged, spike = _spike_mask(px, raw)

    drop = stale | spike
    keep_extra = pd.DataFrame(False, index=raw.index, columns=raw.columns)
    overrides = load_overrides() if overrides is None else overrides
    applied = []
    for r in overrides.itertuples(index=False):
        if r.ticker not in drop.columns:
            continue
        days = drop.index[(drop.index >= r.start) & (drop.index <= r.end)]
        if not len(days):
            continue
        if r.action == "drop":
            drop.loc[days, r.ticker] = True
        else:
            drop.loc[days, r.ticker] = False
            keep_extra.loc[days, r.ticker] = flagged.loc[days, r.ticker] | stale.loc[days, r.ticker] | spike.loc[days, r.ticker]
        applied.append((r.ticker, r.start, r.end, r.action))

    returns = raw.where(mem & ~drop)

    # Quality report: every removed or flagged member-day return
    def _rows(mask, reason):
        # pick flagged cells directly (stack() keeps NaN in pandas 3, so it can't be used as a filter)
        i, j = np.nonzero((mask & mem).to_numpy())
        return pd.DataFrame({"date": raw.index[i], "ticker": raw.columns[j],
                             "raw_return": raw.to_numpy()[i, j], "reason": reason})
    big_kept = flagged & ~spike & ~stale
    # Tickers with repeated jumps inside a year: likely a reused symbol, check by hand
    n_jumps = (flagged & mem).astype(int).rolling(REPEAT_WINDOW, min_periods=1).sum()
    repeat = (n_jumps >= REPEAT_JUMPS) & flagged & ~drop
    quality = pd.concat([
        _rows(spike & ~keep_extra, "spike_removed"),
        _rows(stale & ~keep_extra, "stale_removed"),
        _rows(big_kept & ~drop, "large_move_kept"),
        _rows(keep_extra, "override_keep"),
        _rows(drop & ~spike & ~stale, "override_drop"),
        _rows(repeat, "repeated_jumps_review"),
    ], ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

    member_days = int(mem.to_numpy().sum())
    priced = int((mem & px.notna()).to_numpy().sum())
    summary = pd.Series({
        "member_days": member_days,
        "member_days_with_price": priced,
        "member_days_with_return": int(returns.notna().to_numpy().sum()),
        "returns_nan_after_gap": int((mem & px.notna() & raw.isna() & px.shift(1).isna()).to_numpy().sum()),
        "stale_removed": int((stale & mem).to_numpy().sum()),
        "spike_removed": int((spike & mem).to_numpy().sum()),
        "large_moves_kept": int((big_kept & mem & ~drop).to_numpy().sum()),
        "override_rows_applied": len(applied),
        "tickers_to_review_repeated_jumps": int(repeat.any().sum()),
    })

    market = pd.DataFrame(index=px.index)
    for b in BENCHMARKS:
        if b in close:
            market[f"{b}_ret"] = close[b].pct_change(fill_method=None)
    market["ew_member_ret"] = returns.mean(axis=1, skipna=True)
    market["n_returns"] = returns.notna().sum(axis=1)
    return ReturnsResult(returns, quality, summary, market)


def save(result: ReturnsResult, out_dir: Path | str = PROCESSED_DIR) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "returns_daily.parquet", out_dir / "returns_market.parquet",
             out_dir / "returns_quality.csv", out_dir / "returns_summary.csv"]
    result.returns.to_parquet(paths[0])
    result.market.to_parquet(paths[1])
    result.quality.to_csv(paths[2], index=False)
    result.summary.rename("value").to_csv(paths[3], index_label="measure")
    return paths


def load_returns(start: str | None = None, end: str | None = None,
                 proc: Path | str = PROCESSED_DIR) -> pd.DataFrame:
    """Load the saved cleaned returns (run notebooks/04_returns_and_windows.ipynb first)."""
    path = Path(proc) / "returns_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebooks/04_returns_and_windows.ipynb first.")
    return pd.read_parquet(path).loc[start:end]
