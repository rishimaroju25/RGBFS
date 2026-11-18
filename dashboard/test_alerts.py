import sys
from pathlib import Path

import streamlit as st

# Locate the RGBFS project root
ROOT = Path(__file__).resolve().parents[1]

# Make project modules accessible
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.panels.alerts import render_alerts_panel

st.set_page_config(page_title="RGBFS - Data & Alerts", page_icon="🚨", layout="wide")

# Run with:  python -m streamlit run dashboard/test_alerts.py
render_alerts_panel()
