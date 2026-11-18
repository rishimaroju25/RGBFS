"""Data & alerts panel (Week 7, Rishi).

Shows the current alert status, the alert history against the labeled stress
episodes, lead times by episode, and data coverage.

Reads (run notebooks 13, 14 and 15 first):
  notebooks/outputs/rishi_week6/signals_daily.csv           alert flags by signal (required)
  notebooks/outputs/rishi_week6/signal_summary.csv          one row per signal (required)
  notebooks/outputs/rishi_week6/lead_times_by_episode_*.csv lead times (required)
  docs/stress_periods.csv                                   labeled episodes (required)
  data/processed/features/anomaly_network_daily.parquet     z-scores and IF score (optional)
  data/processed/daily_panel.parquet                        VIX (optional)
  data/processed/returns_market.parquet                     stocks with a return each day (optional)

Set RGBFS_DATA_ROOT to point at a different data folder (used for test runs).
"""
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(os.environ.get("RGBFS_DATA_ROOT", Path(__file__).resolve().parents[2]))
NETWORK = "Network anomaly (combined)"


def _read_optional_parquet(path, cols=None):
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df[[c for c in cols if c in df.columns]] if cols else df


@st.cache_data(ttl=300)
def load_data(root):
    root = Path(root)
    out = root / "notebooks" / "outputs" / "rishi_week6"
    signals = pd.read_csv(out / "signals_daily.csv", index_col="date", parse_dates=True)
    summary = pd.read_csv(out / "signal_summary.csv")
    leads = {col: pd.read_csv(out / f"lead_times_by_episode_{col}.csv", dtype=str).fillna("")
             for col in ["start", "start_trailing"]}
    episodes = pd.read_csv(root / "docs" / "stress_periods.csv", parse_dates=["start", "end", "start_trailing"])
    anom = _read_optional_parquet(root / "data/processed/features/anomaly_network_daily.parquet",
                                  ["z_composite", "if_score", "alert", "asof"])
    vix = _read_optional_parquet(root / "data/processed/daily_panel.parquet", ["VIXCLS"])
    market = _read_optional_parquet(root / "data/processed/returns_market.parquet", ["n_returns"])
    return signals, summary, leads, episodes, anom, vix, market


def _shade(fig, episodes, lo, hi):
    for ep in episodes.itertuples():
        if ep.end >= lo and ep.start <= hi:
            fig.add_vrect(x0=max(ep.start, lo), x1=min(ep.end, hi), fillcolor="red", opacity=0.12,
                          line_width=0, layer="below")


def _last_on(series):
    on = series.index[series.eq(1)]
    return on[-1] if len(on) else None


def render_alerts_panel(root=ROOT):
    st.header("Data & alerts")
    st.caption("Early-warning alerts from network anomalies compared with VIX and STLFSI4. "
               "Red shading marks the labeled stress episodes, which are defined after the fact "
               "and are only used to score the alerts.")
    try:
        signals, summary, leads, episodes, anom, vix, market = load_data(str(root))
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Unable to load alert data: {exc}. Run notebooks 13, 14 and 15 first.")
        return
    if signals.empty:
        st.warning("Alert data is empty.")
        return

    # ---- current status
    last_day = signals.index.max()
    st.subheader(f"Status on {last_day:%Y-%m-%d}")
    names = list(signals.columns)
    cols = st.columns(min(4, len(names)))
    for i, name in enumerate(names[:4]):
        s = signals[name]
        now = s.loc[:last_day].dropna()
        state = "No data" if now.empty else ("ALERT" if now.iloc[-1] == 1 else "Quiet")
        prev = _last_on(s)
        cols[i].metric(name, state, f"last alert {prev:%Y-%m-%d}" if prev is not None else "never alerted",
                       delta_color="off")
    if anom is not None and anom["z_composite"].notna().any():
        z = anom["z_composite"].dropna()
        st.caption(f"Latest network z composite: {z.iloc[-1]:.2f} (window ending "
                   f"{pd.Timestamp(anom['asof'].dropna().iloc[-1]):%Y-%m-%d}). Alert threshold is 2.")

    # ---- timeline
    st.subheader("Alert history")
    lo_all, hi_all = signals.index.min().date(), signals.index.max().date()
    bounds = st.date_input("Date range", (lo_all, hi_all), min_value=lo_all, max_value=hi_all, key="alerts_range")
    if len(bounds) != 2:
        st.info("Select both dates.")
        return
    lo, hi = map(pd.Timestamp, bounds)
    chosen = st.multiselect("Signals to show", names, default=[n for n in names if n in (NETWORK, "VIX z >= 2", "STLFSI4 > 1 (lagged)")] or names[:3],
                            key="alerts_signals")
    line_options = [o for o, ok in [("Network z composite", anom is not None), ("VIX", vix is not None)] if ok]
    line = st.radio("Line", line_options, horizontal=True, key="alerts_line") if line_options else None

    fig = go.Figure()
    if line == "Network z composite":
        z = anom.loc[lo:hi, "z_composite"]
        fig.add_scatter(x=z.index, y=z, mode="lines", name="z composite", line=dict(width=1))
        fig.add_hline(y=2, line_dash="dash", line_color="gray")
    elif line == "VIX":
        v = vix.loc[lo:hi, "VIXCLS"]
        fig.add_scatter(x=v.index, y=v, mode="lines", name="VIX", line=dict(width=1))
    for i, name in enumerate(chosen):
        on = signals.loc[lo:hi, name]
        on = on.index[on.eq(1)]
        fig.add_scatter(x=on, y=[f"{name}"] * len(on), mode="markers", marker=dict(symbol="line-ns-open", size=10),
                        name=name, yaxis="y2")
    _shade(fig, episodes, lo, hi)
    fig.update_layout(height=420 + 25 * len(chosen), margin=dict(t=30),
                      yaxis=dict(title=line or ""),
                      yaxis2=dict(overlaying="y", side="right", type="category", showgrid=False),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    # ---- lead times
    st.subheader("Lead time by episode")
    which = st.radio("Episode start date", ["start", "start_trailing"], horizontal=True, key="alerts_start",
                     help="start uses a centered volatility window (early bound); start_trailing a trailing window (late bound).")
    st.caption("Trading days of warning before the episode start. Negative = late, 'missed' = no alert "
               "within 60 days before to 10 days after, blank = no data yet, * = the signal was already on when the window opened.")
    st.dataframe(leads[which], width="stretch", hide_index=True)

    st.subheader("Signal comparison")
    show = summary[summary.start_col.eq(which)].drop(columns=["start_col"])
    keep = ["signal", "episodes_scored", "warned_early", "late", "missed", "median_lead_days",
            "false_alarms_per_year", "day_precision", "day_recall", "day_false_alarm_rate"]
    st.dataframe(show[[c for c in keep if c in show]].round(3), width="stretch", hide_index=True)
    st.caption("A signal that fires often can 'warn' many episodes by chance: read lead times together with false alarms per year.")
    st.download_button("Download signal comparison CSV", show.to_csv(index=False), "signal_comparison.csv",
                       "text/csv", key="alerts_download_summary")

    # ---- data
    st.subheader("Data coverage")
    if market is not None and "n_returns" in market:
        n = market["n_returns"]
        figc = go.Figure(go.Scatter(x=n.index, y=n, mode="lines", line=dict(width=1), name="stocks"))
        figc.update_layout(height=280, margin=dict(t=20), yaxis_title="Stocks with a usable return")
        st.plotly_chart(figc, width="stretch")
    fresh = {"Alert signals": signals.index.max()}
    if anom is not None:
        fresh["Network windows"] = pd.Timestamp(anom["asof"].dropna().max()) if anom["asof"].notna().any() else None
    if vix is not None:
        fresh["VIX"] = vix["VIXCLS"].last_valid_index()
    if market is not None:
        fresh["Stock returns"] = market.index.max()
    fresh["Stress labels (last episode end)"] = episodes["end"].max()
    st.dataframe(pd.DataFrame({"source": list(fresh), "latest date": [f"{d:%Y-%m-%d}" if d is not None else "" for d in fresh.values()]}),
                 hide_index=True)
    st.caption("Prices: Yahoo Finance. VIX and STLFSI4: FRED. STLFSI4 is shifted 5 trading days for publication lag.")
