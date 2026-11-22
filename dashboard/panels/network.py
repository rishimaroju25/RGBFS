"""Network panel (Week 7, Brandon; snapshot fallback and real density added by Rishi).

Shows how tightly U.S. stocks move together over time:
  * network density: share of stock pairs whose 60-day correlation is above 0.55
    (notebook 05, weekly), and the average pairwise correlation
  * absorption ratio and PC1 share from the 60-day connectedness windows (notebook 04)

Reads data/processed/network_features.parquet and
data/processed/features/connectedness_60d.parquet, falling back to the committed
copies in dashboard/snapshot/.
"""
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(ttl=300)
def load_data(root):
    root = Path(root)
    out = {}
    net_path = data_file(root, "network_features.parquet")
    if net_path.exists():
        net = pd.read_parquet(net_path)
        if "date" in net.columns:
            net = net.set_index("date")
        net.index = pd.to_datetime(net.index)
        out["network"] = net.sort_index()
    conn_path = data_file(root, "features/connectedness_60d.parquet")
    if conn_path.exists():
        conn = pd.read_parquet(conn_path)
        conn.index = pd.to_datetime(conn.index)
        out["connectedness"] = conn.sort_index()
    ep_path = root / "docs" / "stress_periods.csv"
    if ep_path.exists():
        out["episodes"] = pd.read_csv(ep_path, parse_dates=["start", "end"])
    return out


def _line(series_by_name, episodes, y_title, height=320):
    fig = go.Figure()
    for name, s in series_by_name.items():
        fig.add_scatter(x=s.index, y=s, mode="lines", name=name, line=dict(width=1.3))
    if episodes is not None:
        lo = min(s.index.min() for s in series_by_name.values())
        hi = max(s.index.max() for s in series_by_name.values())
        for ep in episodes.itertuples():
            if ep.end >= lo and ep.start <= hi:
                fig.add_vrect(x0=ep.start, x1=ep.end, fillcolor="red", opacity=0.12, line_width=0, layer="below")
    fig.update_layout(height=height, margin=dict(t=20, b=10), yaxis_title=y_title,
                      legend=dict(orientation="h", y=1.08))
    return fig


def render_network_panel(root=ROOT):
    st.title("Network & Community Structure")
    st.write("How tightly S&P 500 stocks move together. In a stress episode, stocks start to move as one "
             "block: correlations jump and the network becomes dense. Red shading marks the 13 labeled "
             "stress episodes (2005 to 2026).")
    d = load_data(str(root))
    if d.get("network") is None and d.get("connectedness") is None:
        st.warning("Network data not found in data/processed or dashboard/snapshot.")
        return
    episodes = d.get("episodes")

    net = d.get("network")
    if net is not None:
        latest = net.dropna().iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Network density (latest week)", f"{latest['network_density']:.1%}")
        c2.metric("Average correlation (60 days)", f"{latest['avg_correlation']:.2f}")
        c3.metric("Median density, 2005 to 2026", f"{net['network_density'].median():.1%}")
        st.subheader("Network density over time")
        st.caption("Share of stock pairs whose 60-day return correlation is above 0.55 (weekly, notebook 05). "
                   "For example, about 2% in calm October 2017 versus over 60% in October 2008.")
        full_width(st.plotly_chart, _line({"Network density": net["network_density"]}, episodes, "Density"))
        st.subheader("Average pairwise correlation")
        full_width(st.plotly_chart, _line({"Average correlation": net["avg_correlation"]}, episodes, "Correlation"))

    conn = d.get("connectedness")
    if conn is not None:
        st.subheader("Market concentration: absorption ratio and PC1 share")
        st.caption("Absorption ratio (Kritzman et al., 2011): share of total variance explained by the top "
                   "principal components. PC1 share: share explained by the first component alone. Both rise "
                   "when one common factor drives the market (60-day windows every 5 trading days, notebook 04).")
        cols = {k: conn[k] for k in ["absorption", "pc1_share"] if k in conn}
        full_width(st.plotly_chart, _line(cols, episodes, "Share of variance"))

    st.caption("Our tests (notebooks 13 to 15) found that these measures rise at the same time as stress, "
               "not before it, so they describe the size of a crisis better than they predict its start.")
