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

**Results from our download** (fill in after running the script):

| Measure | Value |
|---|---|
| Members since 2005 | |
| Members with no Yahoo data | |
| Share of member-days with a price | |
| Lowest daily share priced (and date) | |
| Share priced during Sep 2008 to Mar 2009 | |

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
* **Adjusted prices.** Yahoo's adjusted close is recomputed after each new
  split or dividend, so a re-download can change past values slightly. We
  record the download date in the README.
