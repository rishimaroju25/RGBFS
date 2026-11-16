import sys
from pathlib import Path

import streamlit as st

# Locate the RGBFS project root
ROOT = Path(__file__).resolve().parents[1]

# Make project modules accessible
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.panels.clusters import render_clusters_panel

# Configure the dashboard
st.set_page_config(
    page_title="RGBFS - Clusters Dashboard",
    page_icon="📊",
    layout="wide"
)

# Display the clusters panel
render_clusters_panel()