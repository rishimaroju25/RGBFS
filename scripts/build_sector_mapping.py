"""Rebuild docs/sector_mapping.csv, the ticker -> sector lookup used by the clustering notebooks.

Source: components_history.csv from the public lawcal/sp500-components-history
dataset (https://github.com/lawcal/sp500-components-history), as described in
notebooks/06_clustering_analysis.ipynb. Rows are filtered to stocks that were
S&P 500 members on 2008-10-15, keeping one row per ticker.

The committed docs/sector_mapping.csv was built from commit 580c2b7 of that
dataset (361 tickers). It matches all 274 sector labels in Grace's saved
Week 3 results. One label per ticker for all years is an approximation: GICS
sector definitions changed in 2016 and 2018.

    git clone --depth 1 https://github.com/lawcal/sp500-components-history /tmp/sp500ch
    python scripts/build_sector_mapping.py /tmp/sp500ch/data/components_history.csv
"""
import sys
from pathlib import Path

import pandas as pd

AS_OF = pd.Timestamp("2008-10-15")
OUT = Path(__file__).resolve().parents[1] / "docs" / "sector_mapping.csv"


def build(history_csv) -> pd.DataFrame:
    h = pd.read_csv(history_csv, dtype=str)
    for c in ["date_added", "date_removed"]:
        h[c] = pd.to_datetime(h[c], errors="coerce")
    members = h[(h.date_added <= AS_OF) & (h.date_removed.isna() | (h.date_removed > AS_OF))]
    out = (members.rename(columns={"symbol": "ticker"})[["ticker", "name", "sector"]]
           .dropna(subset=["ticker", "sector"]).drop_duplicates("ticker").sort_values("ticker"))
    return out.reset_index(drop=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    table = build(sys.argv[1])
    table.to_csv(OUT, index=False)
    print(f"Wrote {len(table)} tickers to {OUT}")
