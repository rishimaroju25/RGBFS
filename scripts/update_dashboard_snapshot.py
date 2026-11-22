"""Copy the small processed files the dashboard reads into dashboard/snapshot/.

Run after scripts/refresh_data.py so the deployed dashboard shows the new days:
    python scripts/update_dashboard_snapshot.py
"""
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SNAP = ROOT / "dashboard" / "snapshot"
FILES = [
    "features/clusters_market_adjusted_20d.parquet",
    "features/cluster_memberships_market_adjusted_20d.parquet",
    "features/anomaly_network_daily.parquet",
    "features/sec_sentiment_filings.parquet",
    "features/sec_sentiment_market_daily.parquet",
    "features/connectedness_60d.parquet",
    "network_features.parquet",
    "returns_market.parquet",
    "stress_labels_daily.parquet",
]

if __name__ == "__main__":
    SNAP.mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        src = PROC / rel
        if src.exists():
            shutil.copy2(src, SNAP / src.name)
            print("copied", rel)
        else:
            print("missing (kept old snapshot):", rel)
    panel = PROC / "daily_panel.parquet"
    if panel.exists():
        pd.read_parquet(panel)[["VIXCLS", "STLFSI4"]].to_parquet(SNAP / "daily_panel.parquet")
        print("copied daily_panel.parquet (VIXCLS, STLFSI4)")
