import streamlit as st
import pandas as pd
from pathlib import Path

def render_network_panel():
    st.title("Network & Community Structure")
    st.write("This panel tracks the 60-day rolling network density and modularity.")
    
    @st.cache_data
    def load_data():
        # Navigate up from dashboard/panels/network.py to the RGBFS root
        root = Path(__file__).resolve().parents[2]
        try:
            cluster_df = pd.read_parquet(root / "data" / "processed" / "features" / "clusters_market_adjusted_20d.parquet")
        except:
            cluster_df = None
        return cluster_df

    cluster_df = load_data()
    
    if cluster_df is not None:
        st.subheader("Network Density Over Time")
        st.line_chart(cluster_df['density'])
        
        st.subheader("Modularity (Community Strength)")
        st.line_chart(cluster_df['modularity'])
    else:
        st.warning("Network data not loaded yet. Make sure 'clusters_market_adjusted_20d.parquet' exists in the processed features folder.")