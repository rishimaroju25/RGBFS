# Deploying the dashboard

**Owner:** Salah · **Week 7** · App: `dashboard/app.py`

## Run it locally

From the repo root:

```
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

It opens at http://localhost:8501. To try one panel on its own: `python -m streamlit run dashboard/test_forecasts.py` (or `test_alerts.py`).

## Deploy on Streamlit Community Cloud (free)

1. Go to share.streamlit.io and sign in with GitHub. The account must have access to `rishimaroju25/RGBFS` (Rishi can deploy it himself, or add the deployer as a collaborator).
2. Click **Create app**, then choose to deploy from a GitHub repository.
3. Repository `rishimaroju25/RGBFS`, branch `main`, main file path `dashboard/app.py`.
4. Optional: choose an app URL, for example `rgbfs-market-stress`.
5. Click **Deploy**. The first build takes a few minutes. Every push to `main` redeploys the app.

Community Cloud installs `dashboard/requirements.txt` (it takes precedence over the root file because it sits next to `app.py`), so the build skips torch and transformers.

## What works without the data folder

`data/` is never on GitHub. The dashboard falls back to a small committed snapshot in `dashboard/snapshot/` (about 0.8 MB, team data through 2026-10-28; see its README), so every page works on the deployed app and for teammates without the Drive data. When `data/processed/` exists locally, it is used instead.

| Panel | Reads |
|---|---|
| Overview | `notebooks/outputs/` summary tables |
| Data & alerts | `notebooks/outputs/rishi_week6/`, plus snapshot anomaly scores, VIX and coverage |
| Network & clusters | snapshot `network_features.parquet`, `connectedness_60d.parquet`, cluster files |
| Sentiment | snapshot `sec_sentiment_daily.parquet` |
| Volatility forecasts | `notebooks/outputs/grace_week6/`, snapshot `returns_market.parquet` |

After a data refresh, run `python scripts/update_dashboard_snapshot.py` and push, so the deployed app shows the new days.

## Test on another computer (checklist)

On a computer that has never run the project:

- [ ] Open the deployed URL in a browser: every page loads, no red error boxes except the known cluster one
- [ ] Clone the repo, `pip install -r requirements.txt`, copy `data/` from the Drive, run `python -m streamlit run dashboard/app.py`
- [ ] Every page loads locally, and the forecast panel shows "Forecast after the close on <last data day>"
- [ ] Change the date range and model choices on each panel; charts update
- [ ] Download buttons give a CSV
- [ ] Note the computer, OS, Python and Streamlit versions and any errors in the PR or group chat

## Refreshing the data

`python scripts/refresh_data.py` updates prices, the FRED panel, returns, stress labels and features to the latest trading day (add `--notebooks` to re-run the forecast and alert notebooks the dashboard reads). It backs up `data/processed` first and never deletes old price history. After a refresh, send `data/processed` to Rishi for the shared Drive and push the updated notebook outputs so the deployed app shows the new days.
