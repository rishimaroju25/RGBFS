"""SEC filing sentiment for the early-warning study (Fatima's Week 3 pipeline, scaled up).

Owner: Fatima (NLP lead). Generalized on 2026-11-21 from
notebooks/04_sentiment_full_corpus.ipynb so the same rules run on 50 firms
from 2006 to today instead of 5 firms in 2022:

1. Keep MD&A (10-K Item 7, 10-Q Item 2) and Risk Factors (Item 1A)
   paragraphs of at least 15 words.                       (Fatima, Step 2)
2. Drop Risk Factors paragraphs repeated from the company's earlier filings
   (exact or >= 90% similar).                              (Fatima, Step 4.5)
   Change: compared with the previous `RF_LOOKBACK` filings (about a year)
   instead of every earlier filing, with fast pre-filters, because the
   all-pairs check grows with the square of 20 years of text. This is done
   BEFORE scoring, so FinBERT never scores text that would be dropped anyway.
3. FinBERT (ProsusAI/finbert) label per paragraph, -1 / 0 / +1.  (Step 3)
   Change: scores are cached by text hash, so a crash or a later refresh
   only scores new paragraphs.
4. Filing-level average per company, filing and section; tone change =
   difference from the previous filing of the same form.   (Step 5)
5. Each filing is used from the first trading day on or after its filing
   date, then delayed one more trading day.                 (Steps 8 and 11)
6. New: a market-wide daily index (average over firms) for the lead-time tests.

No step uses information from after the date it is reported on.
"""
from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = "2006-01-01"
MIN_WORDS = 15
RF_SIMILAR = 0.90
RF_LOOKBACK = 4          # earlier filings (about one year) a risk factor is compared with
MODEL = "ProsusAI/finbert"
SCORE_MAP = {"negative": -1, "neutral": 0, "positive": 1}

# 50 large companies that were S&P 500 members for all of 2006 to 2026 (checked
# against data/processed/sp500_membership.parquet), spread over the 11 sectors.
# Includes Fatima's original five (AAPL, AMZN, GOOGL, MSFT, NVDA).
FIRMS = {
    "AAPL": "information_technology", "MSFT": "information_technology", "NVDA": "information_technology",
    "ORCL": "information_technology", "CSCO": "information_technology", "INTC": "information_technology",
    "QCOM": "information_technology",
    "GOOGL": "communication_services", "CMCSA": "communication_services", "DIS": "communication_services",
    "T": "communication_services", "VZ": "communication_services",
    "JPM": "financials", "BAC": "financials", "WFC": "financials", "C": "financials", "GS": "financials",
    "MS": "financials", "AXP": "financials",
    "JNJ": "health_care", "UNH": "health_care", "PFE": "health_care", "MRK": "health_care",
    "ABT": "health_care", "LLY": "health_care", "AMGN": "health_care",
    "AMZN": "consumer_discretionary", "HD": "consumer_discretionary", "MCD": "consumer_discretionary",
    "NKE": "consumer_discretionary", "LOW": "consumer_discretionary",
    "GE": "industrials", "CAT": "industrials", "HON": "industrials", "UPS": "industrials",
    "BA": "industrials", "UNP": "industrials",
    "PG": "consumer_staples", "KO": "consumer_staples", "PEP": "consumer_staples",
    "WMT": "consumer_staples", "COST": "consumer_staples",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "NEE": "utilities", "DUK": "utilities",
    "SHW": "materials", "APD": "materials",
    "SPG": "real_estate",
}
# Companies whose filings before a reorganization sit under an older SEC CIK.
# The SEC ticker file only maps the current CIK. (To verify against EDGAR: the
# build prints filings per firm and year, so a gap shows up in coverage.csv.)
PREDECESSOR_CIKS = {
    "GOOGL": [1288776],   # Google Inc., before the 2015 Alphabet holding company
    "DIS": [1001039],     # The Walt Disney Company before the 2019 Fox deal holding company
    "XOM": [34088],       # Exxon Mobil Corp; the SEC ticker file now points XOM at a new CIK (2115436) with only recent filings
}


# ------------------------------------------------------------ 1. sections ----
def select_sections(paras: pd.DataFrame, min_words: int = MIN_WORDS) -> pd.DataFrame:
    """Fatima's Step 2: MD&A and Risk Factors paragraphs with at least `min_words` words."""
    df = paras.dropna(subset=["text"]).copy()
    df["clean_text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["word_count"] = df["clean_text"].str.split().str.len()
    df = df[df["word_count"] >= min_words]
    risk = df["section"].eq("Item 1A")
    mdna = (df["form"].eq("10-K") & df["section"].eq("Item 7")) | (df["form"].eq("10-Q") & df["section"].eq("Item 2"))
    df = df[risk | mdna].copy()
    df["section_group"] = np.where(df["section"].eq("Item 1A"), "Risk Factors", "MD&A")
    return df.reset_index(drop=True)


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ----------------------------------------------- 2. repeated risk factors ----
def _similar(a: str, b: str, threshold: float) -> bool:
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return sm.real_quick_ratio() >= threshold and sm.quick_ratio() >= threshold and sm.ratio() >= threshold


def mark_new_risk_factors(df: pd.DataFrame, lookback: int = RF_LOOKBACK,
                          threshold: float = RF_SIMILAR) -> pd.Series:
    """True for MD&A paragraphs and for Risk Factors paragraphs not repeated from the
    company's previous `lookback` filings (exact text or similarity >= `threshold`).

    Only earlier filings are used, so a paragraph's flag never depends on the future.
    """
    is_new = pd.Series(True, index=df.index)
    rf = df[df["section_group"].eq("Risk Factors")]
    for _, g in rf.groupby("ticker"):
        history: list[list[str]] = []                      # paragraphs of each earlier filing
        order = g[["filing_date", "accession"]].drop_duplicates().sort_values(["filing_date", "accession"])
        for acc in order["accession"]:
            cur = g[g["accession"].eq(acc)]
            recent = list(dict.fromkeys(t for filing in history[-lookback:] for t in filing))
            exact = set(recent)
            # Candidate search: an inverted index of 3-word phrases. Two paragraphs that are
            # >= 90% similar share far more than half their phrases, so only those go on to
            # the exact SequenceMatcher check.
            shingles = [_shingles(t) for t in recent]
            index: dict[str, list[int]] = {}
            for i, sh in enumerate(shingles):
                for s in sh:
                    index.setdefault(s, []).append(i)
            for idx, text in zip(cur.index, cur["clean_text"]):
                if text in exact:
                    is_new[idx] = False
                    continue
                sh = _shingles(text)
                counts: dict[int, int] = {}
                for s in sh:
                    for i in index.get(s, ()):
                        counts[i] = counts.get(i, 0) + 1
                cands = [i for i, c in sorted(counts.items(), key=lambda kv: -kv[1])
                         if c >= 0.5 * min(len(sh), len(shingles[i]))][:10]
                if any(_similar(text, recent[i], threshold) for i in cands):
                    is_new[idx] = False
            history.append(cur["clean_text"].tolist())
    return is_new


def _shingles(text: str) -> set[str]:
    w = text.lower().split()
    return {" ".join(w[i:i + 3]) for i in range(len(w) - 2)} or {text.lower()}


# ------------------------------------------------------------- 3. scoring ----
def finbert_scorer(model: str = MODEL, batch_size: int = 16, device: int = -1):
    """Return f(texts) -> list of (label, confidence) using Hugging Face FinBERT.

    The model loads on the first call, so a run where every paragraph is already
    in the cache never needs transformers or torch.
    """
    state = {}

    def score(texts):
        if "clf" not in state:
            from transformers import pipeline
            state["clf"] = pipeline("text-classification", model=model, tokenizer=model, device=device)
        out = state["clf"](list(texts), truncation=True, max_length=512, batch_size=batch_size)
        return [(p["label"].lower(), round(float(p["score"]), 4)) for p in out]
    return score


def score_texts(texts: pd.Series, scorer, cache_path: Path | None = None, chunk: int = 512,
                progress: bool = True) -> pd.DataFrame:
    """Labels for each text, reusing and extending a parquet cache keyed by text hash.

    The cache is written after every `chunk` new texts, so an interrupted run
    resumes where it stopped.
    """
    keys = texts.map(text_key)
    cache = pd.DataFrame(columns=["key", "sentiment", "confidence"])
    if cache_path is not None and Path(cache_path).exists():
        cache = pd.read_parquet(cache_path)
    known = set(cache["key"])
    todo = pd.Series(texts.values, index=keys.values)
    todo = todo[~todo.index.duplicated()]
    todo = todo[~todo.index.isin(known)]
    if progress:
        print(f"Scoring: {len(texts):,} paragraphs, {len(todo):,} new, {len(texts) - len(todo):,} from cache")
    for start in range(0, len(todo), chunk):
        part = todo.iloc[start:start + chunk]
        res = scorer(part.tolist())
        new = pd.DataFrame({"key": part.index, "sentiment": [r[0] for r in res], "confidence": [r[1] for r in res]})
        cache = pd.concat([cache, new], ignore_index=True)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            cache.to_parquet(cache_path, index=False)
        if progress:
            print(f"  {min(start + chunk, len(todo)):,} / {len(todo):,}", flush=True)
    lookup = cache.drop_duplicates("key", keep="last").set_index("key")
    out = lookup.reindex(keys.values)
    out.index = texts.index
    return out


# ------------------------------------------------------ 4. filing summary ----
FILING_KEYS = ["ticker", "accession", "filing_date", "form", "section_group"]


def filing_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Fatima's Step 5: average score of the new paragraphs per filing and section;
    filings with no new paragraphs keep NaN; tone change vs the previous filing of
    the same company, section and form."""
    s = scored.copy()
    s["sentiment_score"] = s["sentiment"].map(SCORE_MAP)
    kept = s[s["is_new"]]
    summ = (kept.groupby(FILING_KEYS).agg(paragraphs=("sentiment_score", "size"),
                                          average_sentiment=("sentiment_score", "mean"),
                                          negative_share=("sentiment_score", lambda v: float((v < 0).mean())))
            .reset_index())
    summ = s[FILING_KEYS].drop_duplicates().merge(summ, on=FILING_KEYS, how="left")
    summ["filing_date"] = pd.to_datetime(summ["filing_date"])
    summ = summ.sort_values(["ticker", "section_group", "form", "filing_date"])
    summ["tone_change"] = summ.groupby(["ticker", "section_group", "form"])["average_sentiment"].diff()
    return summ.reset_index(drop=True)


# ------------------------------------------------ 5. trading-day features ----
def align_to_trading_days(summ: pd.DataFrame, trading_days) -> pd.DataFrame:
    """Fatima's Step 8: first trading day on or after the filing date."""
    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).dropna().unique().sort_values()
    out = summ.copy()
    pos = days.searchsorted(pd.to_datetime(out["filing_date"]), side="left")
    ok = pos < len(days)
    out["trading_date"] = pd.NaT
    out.loc[ok, "trading_date"] = days.take(pos[ok]).to_numpy()
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    return out


DAILY_COLS = ["average_sentiment", "tone_change", "paragraphs", "negative_share", "source_filing_trading_date"]


def daily_features(aligned: pd.DataFrame, trading_days, start: str = START) -> pd.DataFrame:
    """Fatima's Steps 9 and 11: latest filing per company and section carried forward
    over trading days, then delayed one trading day. Same columns as before, plus
    negative_share."""
    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).dropna().unique().sort_values()
    dates = pd.DataFrame({"date": days[days >= pd.Timestamp(start)].astype("datetime64[ns]")})
    tables = []
    a = aligned.dropna(subset=["trading_date"]).copy()
    a["trading_date"] = a["trading_date"].astype("datetime64[ns]")
    for (ticker, section), g in a.groupby(["ticker", "section_group"]):
        g = g.sort_values("trading_date").drop_duplicates("trading_date", keep="last")
        d = pd.merge_asof(dates, g[["trading_date", "average_sentiment", "tone_change", "paragraphs", "negative_share"]],
                          left_on="date", right_on="trading_date", direction="backward")
        d = d.rename(columns={"trading_date": "source_filing_trading_date"})
        d[DAILY_COLS] = d[DAILY_COLS].shift(1)            # one more trading day of delay
        d["ticker"], d["section_group"] = ticker, section
        tables.append(d)
    if not tables:
        return pd.DataFrame(columns=["date", "ticker", "section_group"] + DAILY_COLS)
    out = pd.concat(tables, ignore_index=True)
    return out[["date", "ticker", "section_group"] + DAILY_COLS]


def market_daily(daily: pd.DataFrame, max_age: int = 130) -> pd.DataFrame:
    """Market-wide index: average over firms of each firm's latest (already delayed)
    filing sentiment, per section. A firm's value counts only while its latest filing
    is at most `max_age` trading days old (about two quarters), so a firm that stops
    filing drops out instead of being carried forever."""
    d = daily.dropna(subset=["source_filing_trading_date"]).copy()
    days = pd.DatetimeIndex(sorted(daily["date"].unique()))
    pos = pd.Series(np.arange(len(days)), index=days)
    src = pd.DatetimeIndex(d["source_filing_trading_date"])
    age = pos.reindex(pd.DatetimeIndex(d["date"])).to_numpy() - days.searchsorted(src)
    d = d[age <= max_age]
    rows = []
    for section, tag in [("MD&A", "mdna"), ("Risk Factors", "risk")]:
        s = d[d["section_group"].eq(section)]
        g = s.groupby("date")
        rows.append(pd.DataFrame({
            f"{tag}_sentiment": g["average_sentiment"].mean(),
            f"{tag}_tone_change": g["tone_change"].mean(),
            f"{tag}_negative_share": g["negative_share"].mean(),
            f"{tag}_firms": g["average_sentiment"].count(),
        }))
    out = pd.concat(rows, axis=1).reindex(days)
    out.index.name = "date"
    return out


def coverage(filings: pd.DataFrame) -> pd.DataFrame:
    """Filings per firm and year, to spot gaps (a firm with no filings in a year)."""
    f = filings.copy()
    f["year"] = pd.to_datetime(f["filing_date"]).dt.year
    return f.pivot_table(index="ticker", columns="year", values="accession", aggfunc="count", fill_value=0)
