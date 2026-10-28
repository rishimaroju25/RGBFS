# RGBFS
DTSC-2302 Final Project: **Market Stress Early-Warning System**

Can signals from market networks, financial text sentiment, anomaly detection, and time-series forecasting warn of U.S. equity market stress earlier than standard financial stress indexes do?

Team: Rishi, Grace, Brandon, Fatima, Salah.

## Setup (once)

Run these from the repo folder (Windows PowerShell, macOS or Linux):

```bash
git pull
python -m pip install -r requirements.txt
```

Optional `.env` file in the repo root (it is gitignored, never commit it):

```
FRED_API_KEY=your_key            # optional; FRED works without it
SEC_USER_AGENT=RGBFS student project your_email@charlotte.edu   # needed for the SEC notebook
```

## Get the data

The data is too large for GitHub (`data/` is gitignored). Get it one of two ways:

* **Download it yourself** (several minutes, about 1,000 tickers):
  ```bash
  python scripts/download_sp500_prices.py
  python -m jupyter nbconvert --to notebook --execute --inplace 01_data_collection.ipynb
  ```
  The first command writes `data/processed/sp500_*.{parquet,csv}`. The second writes `data/processed/daily_panel.parquet` (FRED stress indexes, VIX, sector ETFs). If the download reports many empty tickers, run `python scripts/download_sp500_prices.py --retry-empty`.
* **Or copy the team's shared `data/processed/` folder** from Google Drive into the repo, so the path is `RGBFS/data/processed/`.

## Layout

| Path | What it is |
|---|---|
| `src/universe.py` | Point-in-time S&P 500 membership, renamed-ticker map (`TICKER_ALIASES`) |
| `src/prices.py` | Downloads and assembles daily prices, membership mask, coverage report |
| `src/data_loading.py` | FRED series, sector ETF prices, daily panel, first network features |
| `src/stress_labels.py` | Stress-period labels (rule in `docs/stress_label_rule.md`) |
| `src/sec_filings.py` | SEC EDGAR 10-K/10-Q download and paragraph cleaning |
| `scripts/download_sp500_prices.py` | Runs the price download end to end |
| `01_data_collection.ipynb` | FRED + ETF panel, preliminary label, first-look features |
| `notebooks/eda_2008.ipynb` | EDA: 2008 financial crisis (Rishi) |
| `notebooks/eda_2011.ipynb` | EDA: 2011 debt ceiling, downgrade and euro-area stress (Salah) |
| `notebooks/eda_2020.ipynb` | EDA: 2020 COVID-19 crash (Grace) |
| `notebooks/eda_2022.ipynb` | EDA: 2022 rate-hike bear market (Fatima) |
| `notebooks/stress_labels.ipynb` | Builds the stress episode table (Salah) |
| `notebooks/02_sec_filings.ipynb` | SEC filing text for sentiment (Fatima) |
| `docs/` | Decisions and write-ups: universe and dates, survivorship bias, stress label rule, stress periods table, text source comparison |
| `Group_Project.ipynb`, root `*.csv` | Brandon's benchmark downloads (`*_daily.csv` are the daily files) |

All four EDA notebooks have the same sections (coverage, market overview, VIX/STLFSI4, dispersion and breadth, connectedness, timing table, robustness), so the stress periods can be compared directly.

## Data sources
* S&P 500 point-in-time membership: [fja05680/sp500](https://github.com/fja05680/sp500) (MIT license)
* Yahoo Finance (via `yfinance`): daily prices for S&P 500 members, `^GSPC`, `SPY`, sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY)
* FRED: STLFSI4, NFCI, VIXCLS, BAMLH0A0HYM2, T10Y2Y
* SEC EDGAR: 10-K and 10-Q filings
