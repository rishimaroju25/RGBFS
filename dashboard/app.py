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
    st.title("System Overview")
    st.write("Welcome to the RGBFS Market Stress Early-Warning System.")
    
    # High-level metric placeholders
    st.subheader("Current Market Pulse")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Current Market Volatility", value="18.5%", delta="+1.2%")
    col2.metric(label="Active Sentiment Anomalies", value="3 flags", delta="-1")
    col3.metric(label="Network Fragility", value="High", delta="Warning", delta_color="inverse")

elif page == "Data & Alerts":
    if render_alerts_panel is None:
        st.error("The Data & Alerts panel could not load. Send the text below to Rishi. "
                 "Usually `python -m pip install -r requirements.txt` and a restart fixes it.")
        st.code(ALERTS_ERROR)
    else:
        render_alerts_panel()

elif page == "Network & Clusters":
    st.title("Network & Community Structure")
    st.write("This panel will integrate Grace and Brandon's network and clustering data.")
    st.info("Placeholder: 60-day rolling network visualizations, density, and modularity charts will go here.")

elif page == "Sentiment Anomalies":
    st.title("SEC Filing Sentiment Anomalies")
    st.write("This panel will display Fatima and Rishi's NLP anomaly flags.")
    st.info("Placeholder: Lead-time table and sentiment drop alerts will go here.")

elif page == "Volatility Forecasts":
    st.title("20-Day Volatility Forecasts")
    st.write("This panel will display Grace's model comparison and Salah's walk-forward backtesting results.")
    st.info("Placeholder: Model 1, Model 2, and Model 3 accuracy metrics and actual vs. predicted charts will go here.")