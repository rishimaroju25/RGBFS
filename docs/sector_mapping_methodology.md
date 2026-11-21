# Sector Mapping: Source, Construction, and Limitations

## Purpose

The RGBFS clustering analysis compares network-detected stock communities with industry sector labels. The file `sector_mapping.csv` provides a ticker-to-sector lookup for these comparisons. It is a reference mapping, **not** a verified point-in-time sector classification panel.

## File location and access

- **Local project path:** `data/processed/sector_mapping.csv`
- **Team Google Drive:** `Shared with me / data / processed / sector_mapping.csv`
- **Fields:** `ticker`, `name`, `sector`
- **Version used in the initial analysis:** 361 unique company/ticker records across 11 sector-label categories (including labels introduced after 2008). Check the shared file before relying on these counts; they describe the original generated mapping.

The processed CSV is shared through the team's Google Drive rather than committed to Git. To reproduce the analysis, download it to the local path above.

## Upstream data source

The mapping was derived from the publicly available `sp500-components-history` dataset:

- Repository: https://github.com/lawcal/sp500-components-history
- Source CSV: https://raw.githubusercontent.com/lawcal/sp500-components-history/main/data/components_history.csv

The source contains fields including `symbol`, `name`, `sector`, `date_added`, and `date_removed`. The mapping was constructed by filtering the source to constituents whose recorded membership interval included **2008-10-15**, then retaining the ticker, company name, and sector and removing duplicate ticker records.

The original preparation did not pin a specific upstream Git commit or snapshot. As a result, downloading the source again may yield different results if the repository changes. For exact reproducibility, use the shared `sector_mapping.csv` as the frozen input and record its checksum when preparing a release.

## Reproduction procedure

From the RGBFS project root, with pandas installed:

```bash
curl -L --fail -o data/processed/components_history.csv https://raw.githubusercontent.com/lawcal/sp500-components-history/main/data/components_history.csv
```

```python
import pandas as pd

source = pd.read_csv("data/processed/components_history.csv")
source["date_added"] = pd.to_datetime(source["date_added"], errors="coerce")
source["date_removed"] = pd.to_datetime(source["date_removed"], errors="coerce")
reference_date = pd.Timestamp("2008-10-15")

active = source.loc[
    source["date_added"].le(reference_date)
    & (
        source["date_removed"].isna()
        | source["date_removed"].gt(reference_date)
    )
].copy()

mapping = (
    active.rename(columns={"symbol": "ticker"})
    [["ticker", "name", "sector"]]
    .drop_duplicates(subset=["ticker"])
    .sort_values("ticker")
    .reset_index(drop=True)
)

mapping.to_csv("data/processed/sector_mapping.csv", index=False)
print(mapping.shape)
```

**Reproduction caveat:** This reconstructs the documented filtering logic, but it is not guaranteed to reproduce the exact shared file byte-for-byte because the upstream version was not pinned and source records may require additional cleaning. If duplicate tickers carry conflicting sector labels, inspect them rather than silently accepting the first entry.

## How the mapping is used

Join the mapping to network or community membership data on ticker. Retain unmatched stocks and label their sector as `Unmatched` in displays. Report the fraction of stocks with a matched sector for each comparison or snapshot. Sector alignment statistics (such as adjusted Rand index or normalized mutual information) should be interpreted alongside mapping coverage.

## Historical and methodological limitations

1. **Not point-in-time sector history.** Membership filtering at 2008-10-15 does not establish that the `sector` values are the classifications used on that date. The source may apply later sector labels to historical constituents.
2. **Sector definitions changed.** Real Estate became a standalone GICS sector in 2016, and Communication Services was restructured in 2018. Those labels should not be presented as exact 2008 classifications.
3. **One mapping across the full period.** Applying a single ticker-to-sector label to snapshots from 2005 through 2026 ignores reclassifications, constituent turnover, ticker changes, mergers, and delistings.
4. **Incomplete coverage.** Stocks absent from the mapping remain unmatched; alignment statistics are computed only on matched observations and may not represent the full network.
5. **Comparison is descriptive.** Weak alignment between network communities and sector labels does not by itself establish whether the clustering algorithm is correct or incorrect.

## Recommended reporting language

> Sector labels are drawn from a static ticker-to-sector reference constructed using historical S&P 500 membership records. The labels are approximate and have not been verified as point-in-time GICS classifications for each snapshot. Coverage varies by date, and unmatched tickers are excluded from sector-agreement calculations. Results should therefore be interpreted as descriptive rather than historically exact sector validation.

## Suggested future improvement

For rigorous historical evaluation, replace the static mapping with a dated security identifier-to-sector panel from a licensed or independently verified point-in-time source. Use stable identifiers where possible, apply sector labels as of each network snapshot date, and document the source version and coverage.
