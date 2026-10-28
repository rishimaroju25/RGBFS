# How we handle survivorship bias

**Owner:** Rishi · **Applies to:** every analysis that uses stock-level data

## What it is

Survivorship bias happens when a dataset includes only the companies that are
still around today. For our project it is especially dangerous: the companies
that disappear during a crisis (bankruptcies, emergency mergers, delistings)
are often the ones at the center of the stress. Dropping them makes stress
periods look milder, makes the market look less connected, and can make an
early-warning signal look better or worse than it really is.

## What we do about it

### 1. Use point-in-time membership, not today's list
`src/universe.py` loads the historical S&P 500 member lists from
[fja05680/sp500](https://github.com/fja05680/sp500). For each trading day,
`membership_matrix()` says which tickers were in the index **on that day**.
We download prices for every ticker that was a member at any point since
2005 (about 980), not just today's ~500.

### 2. Only use a stock on days it was in the index
`masked_close()` sets prices to missing on days a stock was not a member.
Every cross-sectional calculation (correlations, networks, clusters,
equal-weighted returns) should start from the masked prices, so a stock
enters our universe when it joins the index and leaves when it is removed.

### 3. Measure the gap we cannot close
Yahoo Finance mostly keeps history only for symbols that still trade, so
many failed or acquired firms (Lehman Brothers, Bear Stearns, Washington
Mutual and others) will have no prices. We do not hide this:

* `data/processed/sp500_coverage_report.csv` lists every member with the
  share of its member-days that have a price (`coverage`) and a `no_data` flag.
* `coverage_by_day()` gives the share of index members we have prices for on
  each day. The 2008 EDA notebook plots it.

**Results from our download** (Rishi's rerun on 2026-10-28 with `--retry-empty` and the renamed-ticker fix; the first run on 2026-10-26 gave 76.3% of member-days priced):

| Measure | Value |
|---|---|
| Members since 2005 | 977 |
| Members with no Yahoo data while in the index | 342 |
| Share of member-days with a price, 2005 onward | 77.1% |
| Lowest daily share priced (and date) | 53.3% on 2005-01-03 |
| Tickers flagged for possible symbol reuse (more than one membership spell) | 51 |
| Share priced, 2007 to 2009 (2008 EDA) | 61.6% |
| Share priced during Sep 2008 to Mar 2009 (2008 EDA) | 62.6% |
| Share priced, 2011 (2011 EDA) | 67.2% |
| Share priced, 2020 (2020 EDA) | 88.0% |
| Share priced, 2022 (2022 EDA) | 92.3% |

Coverage is lowest at the start of the sample because the earliest members are the ones most likely to have been acquired or to have failed since. Coverage rises over time (62% in 2008 to 92% in 2022) because the more recent the period, the fewer of its members have since disappeared from Yahoo.

### 3b. Gaps we have fixed since the first download

* **Renamed tickers.** The membership file stores each company under the symbol
  it used on each date, but Yahoo keeps the full history under the company's
  current symbol only (for example `ABC` is now `COR`). These showed up as
  "no data" even though the company never left the index (found in the 2011
  EDA). `TICKER_ALIASES` in `src/universe.py` maps old to current symbols and
  the coverage report's `prices_from` column shows which ones were filled. Add
  more as we find them.
* **Rate-limited downloads.** When Yahoo rate-limits a batch, `yfinance` can
  return empty columns instead of an error. The first version of the script
  logged those tickers as permanently empty. It now retries each missing
  ticker on its own first. To retry tickers logged by an earlier run:
  `python scripts/download_sp500_prices.py --retry-empty`.

### 4. Treat our 2008 results as a likely understatement
The missing firms are concentrated among those that failed, and failing
firms tend to have extreme returns right before they disappear. So the
stocks we lose are probably among the most stressed. Our measures of
volatility, dispersion and connectedness in 2008 are therefore more likely to
**understate** stress than overstate it. We will say this in the limitations
section and check whether conclusions change when we restrict to stocks with
full coverage.

### 5. Check reused tickers by hand
Some symbols have belonged to different companies at different times.
`check_ticker_reuse` in the coverage report flags tickers with more than one
membership spell, and `late_start` flags tickers whose prices start well
after their membership began (often a sign that Yahoo only has the newer
company's history). Before final results, someone should look at the flagged
tickers that matter for our stress windows.

## Limitations we accept

* **Membership file freshness.** The file is updated by hand from Wikipedia
  every couple of months; its last change date is printed every time the
  download runs. Recent index changes may be missing.
* **Membership file accuracy.** The maintainer notes that the pre-2001 lists
  may be missing a few symbols. We start in 2005, after that window, but we
  have not independently audited the file.
* **No delisting returns.** Even with a perfect member list, the last price
  move of a failed firm (often close to -100%) is not in Yahoo data. A
  survivorship-free source such as CRSP would fix this, but we do not
  currently have access to it.
* **Symbols that lost their history recently.** Eight tickers (AVB, EA, EQR, LEG, PSKY, QRVO, SATS, WBD) return only their last few trading days from Yahoo, starting between 2026-07-29 and 2026-10-20, with both `yf.download` and `yf.Ticker().history` (checked 2026-10-28). This looks like Yahoo keeping only the final days of recently acquired, merged or delisted symbols; the corporate events have not been verified. They are long-time members, so they lower 2020 and 2022 coverage by a little over 1 percentage point. Because any future delisting can erase a symbol's history the same way, **keep `data/raw/prices/` (the download cache) and share it on Google Drive** rather than re-downloading from scratch.
* **Adjusted prices.** Yahoo's adjusted close is recomputed after each new
  split or dividend, so a re-download can change past values slightly. We
  record the download date in the README.
