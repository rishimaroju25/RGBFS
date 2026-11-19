import streamlit as st
import pandas as pd

def render_network_panel(cluster_df):
    st.title("Network & Community Structure")
    st.write("This panel tracks the 60-day rolling network density and modularity.")
    
    if cluster_df is not None:
        st.subheader("Network Density Over Time")
        st.line_chart(cluster_df['density'])
        
        st.subheader("Modularity (Community Strength)")
        st.line_chart(cluster_df['modularity'])
    else:
        st.warning("Network data not loaded yet.")