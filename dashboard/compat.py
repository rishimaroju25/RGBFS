"""Small helpers so the dashboard runs on old and new Streamlit versions.

Newer Streamlit sizes elements with width="stretch"; older versions (such as
the one bundled with Anaconda) only accept a number there and use
use_container_width=True instead. full_width() tries the new style first and
falls back to the old one.
"""
import streamlit as st


def full_width(element, *args, **kwargs):
    """Call a Streamlit element (st.dataframe, st.plotly_chart, ...) at full container width."""
    try:
        return element(*args, width="stretch", **kwargs)
    except (TypeError, ValueError, st.errors.StreamlitAPIException):
        return element(*args, use_container_width=True, **kwargs)


from pathlib import Path as _Path

SNAPSHOT = _Path(__file__).resolve().parent / "snapshot"


def data_file(root, rel):
    """Path to data/processed/<rel>, or to the committed copy in dashboard/snapshot/.

    The local data folder wins when it exists, so a refreshed `data/` is always
    shown. The snapshot keeps the panels working on Streamlit Community Cloud
    and on computers without the shared data. Returns the data/processed path
    (which may not exist) if neither is found.
    """
    local = _Path(root) / "data" / "processed" / rel
    if local.exists():
        return local
    snap = SNAPSHOT / _Path(rel).name
    return snap if snap.exists() else local
