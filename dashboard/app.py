import sys
import streamlit as st
import pandas as pd
from pathlib import Path

# Make the repo importable when run as: python -m streamlit run dashboard/app.py
ROOT = Path(__file__).resolve().parents[1]
# Always put the repo first, so another installed package named "dashboard" cannot shadow this one
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
_loaded = sys.modules.get("dashboard")
if _loaded is not None and not str(getattr(_loaded, "__file__", "") or "").startswith(str(ROOT)):
    for _name in [m for m in sys.modules if m == "dashboard" or m.startswith("dashboard.")]:
        del sys.modules[_name]

# If a panel cannot load, show why on its page instead of crashing the whole app
try:
    from dashboard.panels.alerts import render_alerts_panel
    ALERTS_ERROR = None
except Exception as exc:  # noqa: BLE001
    render_alerts_panel = None
    ALERTS_ERROR = f"{type(exc).__name__}: {exc}"

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
        st.error(f"The Data & Alerts panel could not load ({ALERTS_ERROR}). "
                 "Run `git pull`, then `python -m pip install -r requirements.txt`, and restart the dashboard.")
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