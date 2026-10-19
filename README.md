# RGBFS
DTSC-2302 Final Project: **Market Stress Early-Warning System**

Can signals from market networks, financial text sentiment, anomaly detection, and time-series forecasting warn of U.S. equity market stress earlier than standard financial stress indexes do?

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # optional: add a FRED API key
```
Then run `01_data_collection.ipynb` top to bottom. It downloads the data into `data/raw/` and writes `data/processed/daily_panel.parquet`.

## Layout
| Path | What it is |
|---|---|
| `01_data_collection.ipynb` | Loads FRED stress indexes and sector ETF prices, aligns them, builds a preliminary stress label and first network features |
| `src/data_loading.py` | Reusable loaders and feature functions |
| `data.ipynb` | Earlier API scratch notebook |

[View project code](https://github.com/rishimaroju25/RGBFS/blob/main/Group_Project.ipynb)

## Data sources
* FRED: STLFSI4, NFCI, VIXCLS, BAMLH0A0HYM2, T10Y2Y
* Yahoo Finance (via `yfinance`): SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY


