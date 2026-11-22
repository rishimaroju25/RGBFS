import sys
import streamlit as st
import pandas as pd
from pathlib import Path

# Load panels straight from their files, so no other package or old copy of the
# project on this computer can be picked up instead (dashboard/panels/<name>.py).
import importlib.util
import traceback

ROOT = Path(__file__).resolve().parents[1]
PANELS = Path(__file__).resolve().parent / "panels"


def _load_panel(name, func):
    """Return (function, None) or (None, error text) for dashboard/panels/<name>.py."""
    path = PANELS / f"{name}.py"
    try:
        spec = importlib.util.spec_from_file_location(f"rgbfs_panel_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func), None
    except Exception:  # noqa: BLE001
        return None, (f"Tried {path} (exists: {path.exists()}).\n\n"
                      f"Python: {sys.executable}\n\n{traceback.format_exc(limit=3)}")


render_alerts_panel, ALERTS_ERROR = _load_panel("alerts", "render_alerts_panel")

# Add your loader here:
render_network_panel, NETWORK_ERROR = _load_panel("network", "render_network_panel")
render_clusters_panel, CLUSTERS_ERROR = _load_panel("clusters", "render_clusters_panel")
render_sentiment_panel, SENTIMENT_ERROR = _load_panel("sentiment_panel", "show_sentiment_panel")
render_forecasts_panel, FORECASTS_ERROR = _load_panel("forecasts", "render_forecasts_panel")

# 1. Page Configuration
st.set_page_config(
    page_title="Market Stress Early-Warning System", 
    page_icon="📉",
    layout="wide"
)

# 2. Sidebar Navigation
st.sidebar.title("Market Stress Dashboard")
st.sidebar.write("Developed by Team RGBFS")
page = st.sidebar.radio(
    "Navigation", 
    ["Overview", "Data & Alerts", "Network & Clusters", "Sentiment Anomalies", "Volatility Forecasts"]
)

# 3. Page Layouts & Panels
if page == "Overview":
    st.title("Market Stress Early-Warning System")
    st.write("**Question:** can signals from market networks, SEC filing sentiment, anomaly detection and "
             "volatility forecasting warn of U.S. stock market stress *earlier* than the standard indicators "
             "(VIX and the St. Louis Fed Financial Stress Index)? S&P 500 stocks, 2005 to 2026, "
             "13 labeled stress episodes.")
    OUTS = ROOT / "notebooks" / "outputs"

    def _csv(rel, **kw):
        path = OUTS / rel
        return pd.read_csv(path, **kw) if path.exists() else None

    st.subheader("Current status")
    signals = _csv("rishi_week6/signals_daily.csv", index_col="date", parse_dates=True)
    if signals is not None and not signals.empty:
        last = signals.index.max()
        cols = [c for c in ["Network anomaly (combined)", "VIX > 25", "VIX z >= 2", "STLFSI4 > 1 (lagged)"] if c in signals]
        metric_cols = st.columns(len(cols))
        for col, name in zip(metric_cols, cols):
            now = signals[name].loc[:last].dropna()
            col.metric(name, "No data" if now.empty else ("ALERT" if now.iloc[-1] == 1 else "Quiet"))
        st.caption(f"Signals as of {last:%Y-%m-%d}, the last day in the team data snapshot. Details on the Data & Alerts page.")

    st.subheader("Key results so far")
    summary = _csv("rishi_week6/signal_summary.csv")
    if summary is not None:
        s0 = summary[summary.start_col.eq("start")].set_index("signal")
        rows = []
        for name in ["Network anomaly (combined)", "VIX > 25", "VIX z >= 2", "STLFSI4 > 1 (lagged)"]:
            if name in s0.index:
                r = s0.loc[name]
                rows.append({"Signal": name, "Warned early": int(r.warned_early), "Late": int(r.late),
                             "Missed": int(r.missed), "Episodes scored": int(r.episodes_scored),
                             "False alarms per year": round(float(r.false_alarms_per_year), 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Warned early = first alert in the 60 trading days before an episode starts. "
                   "Network measures mostly rise *with* stress, not before it.")
    perf = _csv("grace_week6/model_comparison_results.csv")
    if perf is not None:
        best = perf.sort_values("RMSE").iloc[0]
        st.write(f"**Volatility forecasting (20 days ahead, walk-forward):** best model is {best['model']} "
                 f"(RMSE {best['RMSE']:.3f}, R² {best['R2']:.2f}); adding cluster features did not help. "
                 "Details on the Volatility Forecasts page.")
    flags = _csv("grace_week4/sentiment_anomaly_flags.csv")
    if flags is not None and "anomaly_flag" in flags:
        n_flag = int(flags["anomaly_flag"].astype(str).str.lower().eq("true").sum())
        n_eval = int(flags["evaluable"].astype(str).str.lower().eq("true").sum()) if "evaluable" in flags else len(flags)
        st.write(f"**SEC filing sentiment (5 tech companies, 2022):** {n_flag} of {n_eval} evaluable filing-section "
                 "changes were flagged as unusual. Details on the Sentiment page.")

elif page == "Data & Alerts":
    if render_alerts_panel is None:
        st.error("The Data & Alerts panel could not load. Send the text below to Rishi. "
                 "Usually `python -m pip install -r requirements.txt` and a restart fixes it.")
        st.code(ALERTS_ERROR)
    else:
        render_alerts_panel()

elif page == "Network & Clusters":
    if render_network_panel is None:
        st.error("The network panel could not load. Send the text below to Brandon.")
        st.code(NETWORK_ERROR)
    else:
        render_network_panel()
    st.divider()
    if render_clusters_panel is None:
        st.error("The clusters panel could not load. Send the text below to Grace.")
        st.code(CLUSTERS_ERROR)
    else:
        render_clusters_panel()

elif page == "Sentiment Anomalies":
    if render_sentiment_panel is None:
        st.error("The sentiment panel could not load. Send the text below to Fatima.")
        st.code(SENTIMENT_ERROR)
    else:
        render_sentiment_panel()

elif page == "Volatility Forecasts":
    if render_forecasts_panel is None:
        st.error("The Volatility Forecasts panel could not load. Send the text below to Salah.")
        st.code(FORECASTS_ERROR)
    else:
        render_forecasts_panel()
