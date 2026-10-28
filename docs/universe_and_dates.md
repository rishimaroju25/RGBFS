# Proposal: stock universe and date range

**Owner:** Rishi · **Status:** adopted for Week 1 (marked done in the team tracker, 2026-10-26). Raise objections at the next check-in.

## Proposal

| Decision | Proposal |
|---|---|
| Universe | **Point-in-time S&P 500**: on each date, only the stocks that were in the index on that date |
| Membership source | [fja05680/sp500](https://github.com/fja05680/sp500) (MIT license), historical member lists from 1996 |
| Date range | **2005-01-01 to the latest trading day** |
| Prices | Daily, split- and dividend-adjusted close from Yahoo Finance (`yfinance`) |
| Benchmarks | `^GSPC` (index level) and `SPY` (ETF), downloaded with the stocks |
| Existing sector ETF panel | Keep it. It still feeds the stress label work and is a cross-check for the stock-level network |

## Why point-in-time membership

If we used today's S&P 500 list, 2008 would contain only the companies that
survived to today. The firms that failed or were absorbed during the crisis,
the ones most connected to the stress we are trying to detect, would be
missing. That would make 2008 look calmer and less connected than it was.

From the membership file itself: **495 tickers were in the index on
2005-01-03, and only 223 of those tickers are in the latest list
(2026-08-18).** Some of the gap is ticker changes rather than companies
leaving, but either way a "current members" universe would be very different
from what investors actually held in 2005 to 2009.

## Why 2005 onward

* It covers all four EDA periods: 2008 (Rishi), 2011 (Salah), 2020 (Grace), 2022 (Fatima).
* Our longest planned rolling window is 250 trading days (absorption ratio,
  following Kritzman et al., 2011). Starting in January 2005, a 250-day window
  is full by early 2006, more than a year before stress began building in mid
  2007, so the windows are warmed up before the first episode.
* About 980 tickers were members at some point since 2005, which is a
  manageable download.
* Trade-off: it leaves out the 2000 to 2002 dot-com decline. We can add it
  later by changing one start date if the team wants a fifth episode.

## Known limitations (details in `docs/survivorship_bias.md`)

1. Yahoo Finance usually has no history for firms that went bankrupt or were
   acquired, so some point-in-time members will have no prices. The download
   script measures exactly how many.
2. The membership file is maintained by hand and its last update is
   2026-08-18; any index changes since then are not in it yet.
3. A few tickers have been reused by different companies over time, and these
   are flagged for a manual check.

## What I need from the team

* Agree (or push back) on the universe and the 2005 start date.
* Everyone runs `python scripts/download_sp500_prices.py` once, or we share
  the `data/processed/` files through Google Drive. They are too large to
  commit to GitHub.
