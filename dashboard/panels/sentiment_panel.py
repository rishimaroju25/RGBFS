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

import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[2]
# Location of the sentiment dataset (local data folder first, then the dashboard snapshot)
DATA_PATH = data_file(REPO, "features/sec_sentiment_daily.parquet")
FILINGS_PATH = data_file(REPO, "features/sec_sentiment_filings.parquet")       # 50 firms, 2006 on
MARKET_PATH = data_file(REPO, "features/sec_sentiment_market_daily.parquet")


@st.cache_data
def load_sentiment_data():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def show_sentiment_panel_2022():
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






# ---------------------------------------------------------------------------
# 50-firm view (scripts/build_sentiment.py). Falls back to the 2022 view above
# until the 50-firm files exist.

@st.cache_data
def load_full():
    filings = pd.read_parquet(FILINGS_PATH)
    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    market = pd.read_parquet(MARKET_PATH)
    market.index = pd.to_datetime(market.index)
    episodes_path = REPO / "docs" / "stress_periods.csv"
    episodes = pd.read_csv(episodes_path, parse_dates=["start", "end"]) if episodes_path.exists() else None
    return filings, market, episodes


def _shade(fig, episodes, lo, hi):
    if episodes is None:
        return
    for ep in episodes.itertuples():
        if ep.end >= lo and ep.start <= hi:
            fig.add_vrect(x0=ep.start, x1=ep.end, fillcolor="red", opacity=0.12, line_width=0, layer="below")


def show_sentiment_panel():
    if not (FILINGS_PATH.exists() and MARKET_PATH.exists()):
        show_sentiment_panel_2022()
        return
    filings, market, episodes = load_full()
    st.title("SEC Filing Sentiment")
    st.write(f"FinBERT tone of MD&A and new Risk Factors paragraphs in 10-K and 10-Q filings of "
             f"{filings['ticker'].nunique()} large S&P 500 companies across all sectors, "
             f"{filings['filing_date'].min():%Y} to {filings['filing_date'].max():%Y}. "
             "Red shading marks the labeled stress episodes.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Companies", filings["ticker"].nunique())
    c2.metric("Filings", filings["accession"].nunique())
    c3.metric("Latest filing", f"{filings['filing_date'].max():%Y-%m-%d}")

    st.subheader("Market-wide filing tone")
    st.caption("Average over companies of each company's latest filing (used from the trading day after it is "
               "filed; a company drops out once its latest filing is about two quarters old). -1 = every "
               "paragraph negative, +1 = every paragraph positive.")
    m = market.dropna(how="all")
    fig = go.Figure()
    for col, name in [("mdna_sentiment", "MD&A tone"), ("risk_sentiment", "Risk Factors tone (new paragraphs)")]:
        if col in m:
            fig.add_scatter(x=m.index, y=m[col], mode="lines", name=name, line=dict(width=1.3))
    _shade(fig, episodes, m.index.min(), m.index.max())
    fig.update_layout(height=380, margin=dict(t=20), yaxis_title="Average tone", legend=dict(orientation="h", y=1.1))
    full_width(st.plotly_chart, fig)

    st.subheader("One company")
    col_a, col_b = st.columns(2)
    company = col_a.selectbox("Company", sorted(filings["ticker"].unique()), key="sent_company")
    section = col_b.selectbox("Section", ["MD&A", "Risk Factors"], key="sent_section")
    f = filings[(filings["ticker"] == company) & (filings["section_group"] == section)].sort_values("filing_date")
    figc = go.Figure()
    for form, symbol in [("10-K", "diamond"), ("10-Q", "circle")]:
        g = f[f["form"] == form]
        figc.add_scatter(x=g["filing_date"], y=g["average_sentiment"], mode="markers+lines", name=form,
                         marker=dict(symbol=symbol, size=7), line=dict(width=1))
    _shade(figc, episodes, f["filing_date"].min(), f["filing_date"].max())
    figc.update_layout(height=320, margin=dict(t=20), yaxis_title="Filing tone", legend=dict(orientation="h", y=1.1))
    full_width(st.plotly_chart, figc)
    st.caption("Tone change compares each filing with the previous filing of the same form. "
               "Risk Factors filings with no new paragraphs have no tone (gaps).")
    show = f[["filing_date", "form", "paragraphs", "average_sentiment", "tone_change", "negative_share"]].copy()
    show["filing_date"] = show["filing_date"].dt.date
    full_width(st.dataframe, show.sort_values("filing_date", ascending=False).round(3), hide_index=True)
