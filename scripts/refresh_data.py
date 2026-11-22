"""Refresh the project data to the latest trading day (Week 7).

Owner: Salah. Run from the repo root:

    python scripts/refresh_data.py                         # everything except notebooks
    python scripts/refresh_data.py --steps labels features # only some steps
    python scripts/refresh_data.py --notebooks             # also re-run the model notebooks the dashboard reads

Steps, in order (each uses the same code as the notebook that first built the file):

  prices    append new days to data/raw/prices/ (src/refresh.py), download new index
            members, rebuild sp500_close/volume/membership/coverage   (scripts/download_sp500_prices.py)
  panel     FRED series and sector ETFs -> daily_panel.parquet          (01_data_collection.ipynb)
  returns   returns_daily/market/quality/summary                       (04_returns_and_windows.ipynb)
  labels    stress_labels_daily.parquet, stress_periods.csv            (stress_labels.ipynb)
  features  features/connectedness_60d, clusters_*, centrality_* (if src/centrality.py exists)
                                                                       (04, 08, 10_centrality notebooks)
  sentiment new SEC filings for the 50 firms, FinBERT on new paragraphs only, sec_sentiment_* features
                                                                       (scripts/build_sentiment.py)
  --notebooks  re-runs 04_sentiment_full_corpus, 10_volatility_forecasting, 11_walk_forward_forecasting, 13, 14 and 15
            so the dashboard's forecast and alert outputs include the new days.

Safety:
* data/processed is copied to data/backup/<date-time>/ first (skip with --no-backup).
* The prices step stops before writing if the rebuilt price table would have
  fewer stocks than the current one (an incomplete data/raw/prices/ folder, for
  example a partial Drive download). Use --force only if that is intended.
* Old price history is never deleted (see src/refresh.py).

After a refresh, send the new data/processed folder to Rishi for the shared
Drive and note the new last date in the Drive's "Current version" line.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import clustering as cl  # noqa: E402
from src import data_loading as dl  # noqa: E402
from src import prices as px  # noqa: E402
from src import refresh as rf  # noqa: E402
from src import returns as rt  # noqa: E402
from src import stress_labels as sl  # noqa: E402
from src import universe as uni  # noqa: E402
from src import windows as wn  # noqa: E402

PROC = ROOT / "data" / "processed"
FEATURES = PROC / "features"
STEPS = ["prices", "panel", "returns", "labels", "features", "sentiment"]
NOTEBOOKS = [
    "notebooks/04_sentiment_full_corpus.ipynb",
    "notebooks/10_volatility_forecasting.ipynb",
    "notebooks/11_walk_forward_forecasting.ipynb",
    "notebooks/13_anomaly_detection.ipynb",
    "notebooks/14_anomaly_flags_in_models.ipynb",
    "notebooks/15_alert_lead_time.ipynb",
]


def step(name):
    print(f"\n=== {name} ({datetime.now():%H:%M:%S})")


def backup():
    if not PROC.exists():
        return
    dest = ROOT / "data" / "backup" / datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copytree(PROC, dest / "processed", ignore=shutil.ignore_patterns("sec"))
    print(f"Backed up data/processed to {dest.relative_to(ROOT)}")


def refresh_prices(end, refresh_membership, force):
    changes = uni.load_membership_changes(refresh=refresh_membership)
    tickers = uni.tickers_in_period(changes, px.START, end)
    wanted = tickers + px.BENCHMARKS + sorted(set(uni.TICKER_ALIASES.values()) - set(tickers))
    print(f"{len(wanted)} tickers; appending new days to cached prices")
    rep = rf.update_cached_prices(wanted, end=end)
    print("Update:", rep.summary())
    if rep.kept_old:
        print("  Adjustment changed but the full re-download was shorter; old file kept, check by hand:",
              ", ".join(rep.kept_old))
    print("Downloading tickers with no cache yet (new index members):")
    print(px.download_prices(wanted, end=end))

    close, volume, membership, report = px.assemble(changes, start=px.START, end=end)
    old_path = PROC / "sp500_close.parquet"
    if old_path.exists():
        old_n = pd.read_parquet(old_path).notna().any().sum()
        new_n = close.notna().any().sum()
        if new_n < old_n and not force:
            raise SystemExit(f"Stopped: the rebuilt price table has {new_n} priced series, the current one {old_n}. "
                             "data/raw/prices/ is probably incomplete (get the full raw folder from the Drive). "
                             "Nothing was written. Use --force to override.")
    px.save_processed(close, volume, membership, report)
    print(f"sp500_close: {close.shape[1]} series, last day {close.index.max().date()}")


def refresh_panel():
    fred = dl.load_fred(start="1999-01-01", refresh=True)
    etf = dl.load_prices(start="1999-01-01", refresh=True)
    panel = dl.build_daily_panel(fred, etf)
    panel["stress"] = dl.stress_flag(panel["STLFSI4"], threshold=1.0)   # old placeholder, kept for compatibility
    sector_rets = panel[[f"{t}_ret" for t in dl.SECTOR_ETFS]]
    panel["avg_corr_60d"] = dl.rolling_avg_correlation(sector_rets, window=60)
    panel["absorption_250d"] = dl.absorption_ratio(sector_rets, window=250)
    panel.to_parquet(PROC / "daily_panel.parquet")
    print(f"daily_panel: last day {panel.index.max().date()}")


def refresh_returns():
    close = pd.read_parquet(PROC / "sp500_close.parquet")
    membership = pd.read_parquet(PROC / "sp500_membership.parquet")
    res = rt.clean_returns(close, membership)
    rt.save(res)
    print(f"returns_daily: last day {res.returns.index.max().date()}")
    return res.returns


def refresh_labels():
    gspc = pd.read_parquet(PROC / "sp500_close.parquet")["^GSPC"].dropna()
    labels = sl.daily_labels(gspc)
    episodes = sl.episode_table(labels, gspc)
    labels.to_parquet(PROC / "stress_labels_daily.parquet")
    episodes.to_csv(PROC / "stress_periods.csv")
    episodes.to_csv(ROOT / "docs" / "stress_periods.csv")
    unnamed = episodes["name"].str.startswith("unnamed").sum()
    print(f"stress labels: last labeled day {labels['stress'].dropna().index.max().date()}, "
          f"{len(episodes)} episodes" + (f" ({unnamed} new, add a name in src/stress_labels.py)" if unnamed else ""))


def network_features(rets, window=60, threshold=0.55, freq="W"):
    """Brandon's weekly network table (notebook 05, Issue 7): density of the graph with an
    edge where the 60-day correlation is above 0.55 (all stocks counted), and the average
    pairwise correlation. Same calculation without networkx (density = share of pairs linked)."""
    import numpy as np

    rows = []
    for d in rets.resample(freq).last().dropna(how="all").index:
        if len(rets.loc[:d]) < window:
            continue
        c = wn.correlation_at(rets, end=d, window=window).to_numpy()
        iu = np.triu_indices(len(c), k=1)
        rows.append({"date": d, "network_density": float((c[iu] > threshold).mean()),
                     "avg_correlation": float(c[iu].mean())})
    return pd.DataFrame(rows).set_index("date")


def refresh_features(rets):
    FEATURES.mkdir(parents=True, exist_ok=True)
    conn = wn.connectedness(rets)
    conn.to_parquet(FEATURES / "connectedness_60d.parquet")
    print(f"connectedness_60d: last window {conn.index.max().date()}")

    net = network_features(rets)
    net.to_parquet(PROC / "network_features.parquet")
    print(f"network_features: last week {net.index.max().date()}")

    sec = pd.read_csv(ROOT / "docs" / "sector_mapping.csv")
    sectors = sec.dropna(subset=["sector"]).set_index("ticker")["sector"].to_dict()
    feats, members = cl.run_over_time(rets, cl.RECOMMENDED, window=60, step=20, sectors=sectors)
    feats.to_parquet(FEATURES / "clusters_market_adjusted_20d.parquet")
    members.to_parquet(FEATURES / "cluster_memberships_market_adjusted_20d.parquet")
    print(f"clusters: last window {feats.index.max().date()}")

    try:
        from src import centrality as ce
    except ImportError:
        print("centrality: src/centrality.py not on this branch, skipped")
        return
    cent, hubs = ce.centrality_over_time(rets)
    cent.to_parquet(FEATURES / "centrality_60d.parquet")
    hubs.to_parquet(FEATURES / "centrality_hubs_60d.parquet", index=False)
    print(f"centrality: last window {cent.index.max().date()}")


def refresh_sentiment(end):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_sentiment", ROOT / "scripts" / "build_sentiment.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    try:
        bs.main(["--end", end] if end else [])
    except RuntimeError as exc:          # no SEC_USER_AGENT in .env
        print(f"sentiment: skipped ({exc})")
    except ImportError as exc:           # transformers or torch missing
        print(f"sentiment: skipped, install transformers and torch first ({exc})")


def run_notebooks():
    import nbformat
    from nbclient import NotebookClient

    for rel in NOTEBOOKS:
        path = ROOT / rel
        if not path.exists():
            print(f"  {rel}: not found, skipped")
            continue
        t0 = time.time()
        nb = nbformat.read(path, as_version=4)
        try:
            NotebookClient(nb, timeout=1800, kernel_name="python3",
                           resources={"metadata": {"path": str(path.parent)}}).execute()
            nbformat.write(nb, path)
            print(f"  {rel}: done in {time.time() - t0:.0f}s")
        except Exception as exc:
            print(f"  {rel}: FAILED ({type(exc).__name__}); later notebooks may use old outputs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS, help="which steps to run (default: all)")
    ap.add_argument("--notebooks", action="store_true", help="also re-run the model notebooks the dashboard reads")
    ap.add_argument("--end", default=None, help="exclusive end date for prices (default: today)")
    ap.add_argument("--refresh-membership", action="store_true", help="download the latest S&P 500 membership file")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--force", action="store_true", help="write prices even if fewer stocks than before")
    args = ap.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)
    if not args.no_backup:
        backup()
    rets = None
    if "prices" in args.steps:
        step("prices")
        refresh_prices(args.end, args.refresh_membership, args.force)
    if "panel" in args.steps:
        step("daily panel (FRED and sector ETFs)")
        refresh_panel()
    if "returns" in args.steps:
        step("returns")
        rets = refresh_returns()
    if "labels" in args.steps:
        step("stress labels")
        refresh_labels()
    if "features" in args.steps:
        step("features")
        refresh_features(rets if rets is not None else rt.load_returns())
    if "sentiment" in args.steps:
        step("sentiment (SEC filings, FinBERT)")
        refresh_sentiment(args.end)
    if args.notebooks:
        step("notebooks")
        run_notebooks()
    snap = ROOT / "scripts" / "update_dashboard_snapshot.py"
    if snap.exists():
        step("dashboard snapshot")
        import runpy
        runpy.run_path(str(snap), run_name="__main__")
    print("\nDone. Push the notebook outputs and dashboard/snapshot, and send data/processed to Rishi for the shared Drive.")


if __name__ == "__main__":
    main()
