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
