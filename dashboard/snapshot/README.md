# Dashboard data snapshot

Small copies of the processed data files the dashboard panels read, so the
dashboard works on Streamlit Community Cloud and for teammates who have not
copied `data/` from the shared Drive. `data/` itself is never on GitHub.

Each panel reads `data/processed/<file>` when it exists and falls back to the
copy here (see `data_file()` in `dashboard/compat.py`).

| File | Source | Used by |
|---|---|---|
| `clusters_market_adjusted_20d.parquet` | notebook 08 (`src/clustering.py`, recommended setting) | Network & Clusters |
| `cluster_memberships_market_adjusted_20d.parquet` | notebook 08 | Network & Clusters |
| `anomaly_network_daily.parquet` | notebook 13 | Data & Alerts |
| `sec_sentiment_filings.parquet` | `scripts/build_sentiment.py` / notebook 04_sentiment_full_corpus (50 firms, 2006 on) | Sentiment |
| `sec_sentiment_market_daily.parquet` | same | Sentiment |
| `sec_sentiment_daily.parquet` | Fatima's original 2022 notebook (5 tech firms); kept only as the fallback view until the 50-firm files are added | Sentiment |
| `connectedness_60d.parquet`, `network_features.parquet` | notebook 04, `scripts/refresh_data.py` | Network & Clusters |
| `returns_market.parquet` | `src/returns.py` (notebook 04) | Data & Alerts, Volatility Forecasts |
| `stress_labels_daily.parquet` | `src/stress_labels.py` | Network & Clusters |
| `daily_panel.parquet` | `src/data_loading.py`, **VIXCLS and STLFSI4 columns only** | Data & Alerts |

Snapshot of the team data through 2026-10-28 (sentiment: 2022 sample until the 50-firm build is run). After
`python scripts/refresh_data.py`, refresh it with
`python scripts/update_dashboard_snapshot.py`.
