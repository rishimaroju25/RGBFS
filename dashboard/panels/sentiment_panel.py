"""SEC filing sentiment panel (Week 7, Fatima).

Reads data/processed/features/sec_sentiment_daily.parquet (notebook
04_sentiment_full_corpus), falling back to the committed copy in dashboard/snapshot/.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

try:
    from dashboard.compat import data_file, full_width
except ImportError:  # loaded by file path, or another "dashboard" package is installed
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("rgbfs_compat", Path(__file__).resolve().parents[1] / "compat.py")
    _compat = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_compat)
    full_width = _compat.full_width
    data_file = _compat.data_file

# Location of the sentiment dataset (local data folder first, then the dashboard snapshot)
DATA_PATH = data_file(Path(__file__).resolve().parents[2], "features/sec_sentiment_daily.parquet")


@st.cache_data
def load_sentiment_data():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def show_sentiment_panel():
    st.title("SEC Filing Sentiment Analysis")

    st.write(
        "FinBERT sentiment scores from SEC filings of "
        "five technology companies during 2022."
    )

    if not DATA_PATH.exists():
        st.warning("Sentiment dataset not found.")
        return

    df = load_sentiment_data()

    st.subheader("Sentiment Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Companies", df["ticker"].nunique())
    col2.metric("Trading Days", df["date"].nunique())
    col3.metric(
        "Available Sentiment Scores",
        int(df["average_sentiment"].notna().sum())
    )

    st.caption(
        "The dataset includes MD&A and Risk Factors "
        "sections from SEC filings."
    )




    # Filters for company and filing section
    st.subheader("Explore Sentiment Data")

    company = st.selectbox(
        "Select Company",
        ["All Companies"] + sorted(df["ticker"].dropna().unique().tolist())
    )

    filing_section = st.selectbox(
        "Select Filing Section",
        ["All Sections"] + sorted(df["section_group"].dropna().unique().tolist())
    )

    # Apply selected filters
    filtered_df = df.copy()

    if company != "All Companies":
        filtered_df = filtered_df[filtered_df["ticker"] == company]

    if filing_section != "All Sections":
        filtered_df = filtered_df[
            filtered_df["section_group"] == filing_section
        ]

    st.write("Matching Records:", len(filtered_df))
    full_width(st.dataframe, filtered_df, hide_index=True)


    # Average sentiment over time
    st.subheader("Sentiment Trends Over Time")

    sentiment_trend = (
        filtered_df.groupby("date")["average_sentiment"]
        .mean()
        .dropna()
        .sort_index()
    )

    if not sentiment_trend.empty:
        st.line_chart(sentiment_trend)
    else:
        st.info("No sentiment scores available for the selected filters.")


    # Average tone change over time
    st.subheader("Sentiment Tone Changes")

    tone_trend = (
        filtered_df.groupby("date")["tone_change"]
        .mean()
        .dropna()
        .sort_index()
    )

    if not tone_trend.empty:
        st.line_chart(tone_trend)
    else:
        st.info("No tone change data available for the selected filters.")

    st.caption(
        "Tone change measures the difference in sentiment "
        "between comparable SEC filing sections. "
        "Positive values indicate improving sentiment, "
        "while negative values indicate declining sentiment."
    )


    # Show data coverage for selected filters
    st.subheader("Sentiment Data Coverage")

    total_records = len(filtered_df)
    available_scores = filtered_df["average_sentiment"].notna().sum()

    coverage = (
        available_scores / total_records * 100
        if total_records > 0 else 0
    )

    col1, col2 = st.columns(2)

    col1.metric("Total Records", total_records)
    col2.metric("Sentiment Coverage", f"{coverage:.1f}%")

    st.caption(
        "Coverage represents the percentage of records "
        "with available FinBERT sentiment scores."
    )




