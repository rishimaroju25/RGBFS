import sys
import streamlit as st
import pandas as pd
from pathlib import Path

# Make the repo importable when run as: python -m streamlit run dashboard/app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.panels.alerts import render_alerts_panel

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