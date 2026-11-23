"""Volatility forecast panel (Week 7, Salah).
 
Shows the latest 20-day volatility forecast, the walk-forward forecasts
against what actually happened, and forecast accuracy overall, by year and
in calm vs stress periods.
 
Reads:
  notebooks/outputs/grace_week6/walk_forward_predictions.csv     forecasts by date (required)
  notebooks/outputs/grace_week6/model_comparison_results.csv     overall accuracy (optional)
  notebooks/outputs/grace_week6/annual_model_performance.csv     accuracy by year (optional)
  notebooks/outputs/grace_week6/stress_period_performance.csv    calm vs stress accuracy (optional)
  docs/stress_periods.csv                                        labeled episodes (optional)
  data/processed/returns_market.parquet                          SPY returns, for today's forecast (optional)
 
The required files are committed to GitHub, so the panel works on a deployed
copy without the data folder. With the data folder it also shows a forecast
for the latest trading day (HAR-style, refit on every matured target).
 
Set RGBFS_DATA_ROOT to point at a different data folder (used for test runs).
"""
import os
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
try:
    from dashboard.compat import data_file, full_width
except ImportError:  # loaded by file path, or another "dashboard" package is installed
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("rgbfs_compat", Path(__file__).resolve().parents[1] / "compat.py")
    _compat = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_compat)
    full_width = _compat.full_width
    data_file = _compat.data_file
 
ROOT = Path(os.environ.get("RGBFS_DATA_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path("notebooks") / "outputs" / "grace_week6"
MODELS = ["Persistence", "HAR-style", "Volatility + clusters"]
VOL = ["vol_5", "vol_20", "vol_60"]
COLORS = {"Persistence": "#1f77b4", "HAR-style": "#d62728", "Volatility + clusters": "#2ca02c"}
STRESS_LEVEL = 0.25      # annualized volatility that defines a stress day (docs/stress_label_rule.md)
HORIZON = 20
 
 
def _read_csv(path, **kw):
    path = Path(path)
    return pd.read_csv(path, **kw) if path.exists() else None


def _output(root, name):
    """notebooks/outputs/grace_week6/<name>, or the committed copy in dashboard/snapshot/.

    Grace's CSV outputs moved to the shared Drive (2026-11-22), so the snapshot
    copy keeps this panel working on Streamlit Cloud and fresh clones.
    """
    local = Path(root) / OUT / name
    return local if local.exists() else Path(__file__).resolve().parents[1] / "snapshot" / name
 
 
@st.cache_data(ttl=300)
def load_data(root):
    root = Path(root)
    pred = pd.read_csv(_output(root, "walk_forward_predictions.csv"), parse_dates=["date", "target_end"])
    pred = pred.set_index("date").sort_index()
    overall = _read_csv(_output(root, "model_comparison_results.csv"))
    annual = _read_csv(_output(root, "annual_model_performance.csv"))
    by_stress = _read_csv(_output(root, "stress_period_performance.csv"))
    episodes = _read_csv(root / "docs" / "stress_periods.csv", parse_dates=["start", "end"])
    market_path = data_file(root, "returns_market.parquet")
    spy = None
    if market_path.exists():
        m = pd.read_parquet(market_path)
        if "SPY_ret" in m:
            spy = pd.to_numeric(m["SPY_ret"], errors="coerce")
            spy.index = pd.to_datetime(spy.index)
    return pred, overall, annual, by_stress, episodes, spy
 
 
def live_forecast(pred, spy):
    """HAR-style forecast for the latest trading day with SPY returns.
 
    Same features as Grace's notebooks (root mean squared SPY return over the last
    5, 20 and 60 days, annualized), refit by least squares on every forecast whose
    20-day target has already ended. Returns None if the inputs are missing.
    """
    if spy is None or spy.dropna().empty:
        return None
    r = spy.dropna()
    feats = {f"vol_{w}": np.sqrt(252 * (r ** 2).rolling(w, min_periods=w).mean()) for w in (5, 20, 60)}
    feats = pd.DataFrame(feats).dropna()
    if feats.empty:
        return None
    day = feats.index[-1]
    if day <= pred.index[-1]:
        return None          # nothing newer than the backtest
    train = pred[(pred["target_end"] <= day)].dropna(subset=VOL + ["target"])
    X = np.column_stack([np.ones(len(train)), train[VOL].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, train["target"].to_numpy(), rcond=None)
    x = np.r_[1.0, feats.loc[day, VOL].to_numpy()]
    return {"date": day, "Persistence": float(feats.loc[day, "vol_20"]), "HAR-style": float(x @ coef),
            "train_n": len(train)}
 
 
def _shade(fig, episodes, lo, hi):
    if episodes is None:
        return
    for ep in episodes.itertuples():
        if ep.end >= lo and ep.start <= hi:
            fig.add_vrect(x0=max(ep.start, lo), x1=min(ep.end, hi), fillcolor="red", opacity=0.12,
                          line_width=0, layer="below")
 
 
def _status(v):
    return "above the stress level" if v >= STRESS_LEVEL else "below the stress level"
 
 
def render_forecasts_panel(root=ROOT):
    st.header("20-day volatility forecasts")
    st.caption("Each forecast is made after the close and predicts annualized S&P 500 (SPY) volatility over the "
               f"next {HORIZON} trading days. A value of {STRESS_LEVEL:.0%} or more matches the level that defines "
               "a stress day. Forecasts are walk-forward: each model is refit only on outcomes known at the time.")
    try:
        pred, overall, annual, by_stress, episodes, spy = load_data(str(root))
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Unable to load forecast data: {exc}. Run notebook 11_walk_forward_forecasting first.")
        return
    if pred.empty:
        st.warning("Forecast data is empty.")
        return
 
    # ---- latest forecast
    live = live_forecast(pred, spy)
    if live is not None:
        st.subheader(f"Forecast after the close on {live['date']:%Y-%m-%d}")
        cols = st.columns(3)
        for col, m in zip(cols, ["HAR-style", "Persistence"]):
            col.metric(m, f"{live[m]:.1%}")
        cols[2].metric("Stress level", f"{STRESS_LEVEL:.0%}")
        st.markdown(f"HAR-style forecast is **{_status(live['HAR-style'])}**.")
        st.caption(f"HAR-style refit on {live['train_n']:,} forecasts whose outcome is known. Persistence assumes the "
                   "last 20 days' volatility continues. The clusters model needs cluster features and appears in the "
                   "backtest only.")
    else:
        last = pred.index[-1]
        st.subheader(f"Latest backtest forecast: {last:%Y-%m-%d}")
        cols = st.columns(len(MODELS) + 1)
        for col, m in zip(cols, MODELS):
            col.metric(m, f"{pred.loc[last, m]:.1%}")
        cols[-1].metric(f"Actual, to {pred.loc[last, 'target_end']:%Y-%m-%d}", f"{pred.loc[last, 'target']:.1%}")
        st.caption("Today's forecast needs data/processed/returns_market.parquet (not on GitHub). "
                   "Backtest forecasts are shown only once their 20-day outcome is known.")
 
    # ---- forecast vs actual
    st.subheader("Forecasts vs what happened")
    lo_all, hi_all = pred.index.min().date(), pred.index.max().date()
    default_lo = max(lo_all, (pred.index.max() - pd.DateOffset(years=3)).date())
    bounds = st.date_input("Date range", (default_lo, hi_all), min_value=lo_all, max_value=hi_all,
                           key="forecast_range")
    if len(bounds) != 2:
        st.info("Select both dates.")
        return
    lo, hi = map(pd.Timestamp, bounds)
    chosen = st.multiselect("Models", MODELS, default=["Persistence", "HAR-style"], key="forecast_models")
    d = pred.loc[lo:hi]
    fig = go.Figure()
    fig.add_scatter(x=d.index, y=d["target"], mode="lines", name="Actual (next 20 days)",
                    line=dict(color="#f5b800", width=2))
    for m in chosen:
        fig.add_scatter(x=d.index, y=d[m], mode="lines", name=m, line=dict(width=1, color=COLORS[m]))
    fig.add_hline(y=STRESS_LEVEL, line_dash="dash", line_color="gray", annotation_text="stress level")
    _shade(fig, episodes, lo, hi)
    fig.update_layout(height=430, margin=dict(t=30), yaxis=dict(title="Annualized volatility", tickformat=".0%"),
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    full_width(st.plotly_chart, fig)
    st.caption("Plotted on the date each forecast was made: the yellow line is the volatility that then happened "
               "over the next 20 trading days. Red shading marks labeled stress episodes.")
 
    # ---- accuracy
    st.subheader("Accuracy")
    if overall is not None:
        show = overall.sort_values("RMSE")[["model", "n", "MAE", "RMSE", "R2"]]
        full_width(st.dataframe, show.round(4), hide_index=True)
        st.caption("Lower MAE and RMSE are better (in volatility points, 0.01 = 1%). R2 above 0 means better than "
                   "predicting the average.")
    if annual is not None:
        fig = go.Figure()
        for m in MODELS:
            a = annual[annual["model"] == m]
            fig.add_scatter(x=a["year"], y=a["RMSE"], mode="lines+markers", name=m, line=dict(color=COLORS[m]))
        fig.update_layout(height=320, margin=dict(t=30), yaxis=dict(title="RMSE by year"),
                          legend=dict(orientation="h", y=-0.2))
        full_width(st.plotly_chart, fig)
    if by_stress is not None:
        t = by_stress.assign(period=by_stress["stress"].map({0.0: "Calm", 1.0: "Stress"}))
        t = t.pivot(index="model", columns="period", values="RMSE").reindex(MODELS)
        st.markdown("**RMSE in calm vs stress periods**")
        full_width(st.dataframe, t.round(4))
        st.caption("All models miss by much more during stress, when volatility jumps faster than any of them expect.")
 
    st.download_button("Download forecasts CSV", pred.reset_index().to_csv(index=False),
                       "walk_forward_predictions.csv", "text/csv", key="forecast_download")
 