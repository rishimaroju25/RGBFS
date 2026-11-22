"""Tests for src/sentiment.py and scripts/build_sentiment.py (fake SEC data, fake scorer)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src import sec_filings as sec
from src import sentiment as se

ROOT = Path(__file__).resolve().parents[1]


def fake_scorer(calls=None):
    def score(texts):
        if calls is not None:
            calls.append(len(texts))
        out = []
        for t in texts:
            if "loss" in t or "decline" in t:
                out.append(("negative", 0.9))
            elif "growth" in t or "record" in t:
                out.append(("positive", 0.8))
            else:
                out.append(("neutral", 0.7))
        return out
    return score


def para(ticker, form, date, acc, section, text, pid=1):
    return {"ticker": ticker, "form": form, "filing_date": date, "accession": acc,
            "paragraph_id": pid, "section": section, "text": text}


FILLER = " ".join(["word"] * 14)


def test_select_sections_follows_fatima_rules():
    df = pd.DataFrame([
        para("A", "10-K", "2010-02-01", "a1", "Item 7", "revenue growth " + FILLER),
        para("A", "10-Q", "2010-05-01", "a2", "Item 2", "revenue growth " + FILLER),
        para("A", "10-K", "2010-02-01", "a1", "Item 2", "not MD&A in a 10-K " + FILLER),
        para("A", "10-K", "2010-02-01", "a1", "Item 1A", "short risk"),
        para("A", "10-K", "2010-02-01", "a1", "Item 1A", "competition may hurt us " + FILLER),
    ])
    out = se.select_sections(df)
    assert out["section_group"].tolist() == ["MD&A", "MD&A", "Risk Factors"]


def test_repeated_risk_factors_dropped_only_against_earlier_filings():
    base = "our business faces intense competition and pricing pressure in every market " + FILLER
    df = se.select_sections(pd.DataFrame([
        para("A", "10-K", "2010-02-01", "a1", "Item 1A", base, 1),
        para("A", "10-Q", "2010-05-01", "a2", "Item 1A", base, 1),                         # exact repeat
        para("A", "10-Q", "2010-08-01", "a3", "Item 1A", base.replace("every", "each"), 1),  # near repeat
        para("A", "10-Q", "2010-11-01", "a4", "Item 1A", "a new cyber attack caused a loss " + FILLER, 1),
        para("B", "10-K", "2010-02-01", "b1", "Item 1A", base, 1),                          # other firm: new
        para("A", "10-K", "2010-02-01", "a1", "Item 7", base, 2),                           # MD&A: always kept
    ]))
    new = se.mark_new_risk_factors(df)
    got = dict(zip(df["accession"] + df["section_group"], new))
    assert got["a1Risk Factors"] and not got["a2Risk Factors"] and not got["a3Risk Factors"]
    assert got["a4Risk Factors"] and got["b1Risk Factors"] and got["a1MD&A"]


def test_score_cache_resumes(tmp_path):
    texts = pd.Series(["record growth " + FILLER, "a loss " + FILLER, "neutral text " + FILLER])
    calls = []
    first = se.score_texts(texts, fake_scorer(calls), cache_path=tmp_path / "c.parquet", chunk=2, progress=False)
    assert sum(calls) == 3 and first["sentiment"].tolist() == ["positive", "negative", "neutral"]
    calls.clear()
    again = se.score_texts(pd.concat([texts, pd.Series(["another decline " + FILLER])], ignore_index=True),
                           fake_scorer(calls), cache_path=tmp_path / "c.parquet", progress=False)
    assert sum(calls) == 1 and again["sentiment"].iloc[3] == "negative"


def test_daily_features_have_no_lookahead():
    days = pd.bdate_range("2010-01-04", periods=40)
    summ = pd.DataFrame({"ticker": ["A"], "accession": ["a1"], "filing_date": [days[10] + pd.Timedelta(hours=1)],
                         "form": ["10-Q"], "section_group": ["MD&A"], "paragraphs": [5],
                         "average_sentiment": [0.4], "negative_share": [0.1], "tone_change": [float("nan")]})
    summ["filing_date"] = pd.to_datetime(summ["filing_date"].dt.normalize() + pd.Timedelta(days=1))  # filed on a weekday evening -> next day
    aligned = se.align_to_trading_days(summ, days)
    daily = se.daily_features(aligned, days, start="2010-01-01").set_index("date")
    first = daily["average_sentiment"].first_valid_index()
    assert first > aligned["trading_date"].iloc[0]           # one trading day after it is aligned
    assert first > summ["filing_date"].iloc[0]


def test_market_index_drops_stale_firms():
    days = pd.bdate_range("2010-01-04", periods=300)
    summ = pd.DataFrame({"ticker": ["A", "B"], "accession": ["a1", "b1"], "filing_date": [days[5], days[5]],
                         "form": ["10-Q", "10-Q"], "section_group": ["MD&A", "MD&A"], "paragraphs": [5, 5],
                         "average_sentiment": [0.5, -0.5], "negative_share": [0.1, 0.6], "tone_change": [None, None]})
    daily = se.daily_features(se.align_to_trading_days(summ, days), days, start="2010-01-01")
    m = se.market_daily(daily, max_age=130)
    assert m["mdna_firms"].iloc[20] == 2 and m["mdna_sentiment"].iloc[20] == 0
    assert pd.isna(m["mdna_sentiment"].iloc[250])            # both filings too old by then


# ---------------------------------------------------------------- end to end
def _html(mdna, risk):
    p = "".join(f"<p>{t}</p>" for t in mdna)
    r = "".join(f"<p>{t}</p>" for t in risk)
    return (f"<html><body><p>Item 1A. Risk Factors</p>{r}<p>Item 2. Management's Discussion and Analysis</p>{p}"
            f"<p>Item 7. Management's Discussion and Analysis</p>{p}</body></html>")


def _fake_sec():
    filings = {320193: [("10-K", "2015-10-28", "0000320193-15-000001"), ("10-Q", "2016-01-27", "0000320193-16-000001"),
                        ("10-Q", "2016-04-27", "0000320193-16-000002")],
               1288776: [("10-K", "2014-02-03", "0001288776-14-000001")],
               1652044: [("10-K", "2016-02-11", "0001652044-16-000001")]}
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"},
               "1": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet"}}

    def fetch_json(url):
        if url == sec.TICKERS_URL:
            return tickers
        cik = int(url.split("CIK")[1][:10])
        rows = filings.get(cik, [])
        return {"filings": {"recent": {"form": [r[0] for r in rows], "filingDate": [r[1] for r in rows],
                                       "reportDate": [r[1] for r in rows], "accessionNumber": [r[2] for r in rows],
                                       "primaryDocument": ["doc.htm"] * len(rows)}, "files": []}}

    def fetch_text(url):
        acc = url.split("/")[-2]
        if acc.endswith("000002"):
            return _html(["sales fell and we reported a loss " + FILLER], ["supply chain risk persists " + FILLER])
        return _html(["record revenue growth this year " + FILLER], ["supply chain risk persists " + FILLER])
    return fetch_json, fetch_text


def test_build_end_to_end(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("build_sentiment", ROOT / "scripts" / "build_sentiment.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    monkeypatch.setattr(bs, "SEC_DIR", tmp_path / "sec")
    monkeypatch.setattr(bs, "FEATURES", tmp_path / "features")
    monkeypatch.setattr(sec, "SEC_RAW", tmp_path / "raw")
    fetch_json, fetch_text = _fake_sec()
    paras = bs.download(["AAPL", "GOOGL"], "2006-01-01", "2026-12-31", fetch_json, fetch_text, log=lambda *a: None)
    assert set(paras["accession"].str[:10]) == {"0000320193", "0001288776", "0001652044"}   # predecessor CIK read
    days = pd.bdate_range("2013-01-01", "2016-12-30")
    summ, daily, market = bs.build_features(paras, fake_scorer(), days, start="2013-01-01", log=lambda *a: None)
    aapl = summ[(summ.ticker == "AAPL") & (summ.section_group == "MD&A") & (summ.form == "10-Q")].sort_values("filing_date")
    assert aapl["average_sentiment"].tolist() == [1.0, -1.0] and aapl["tone_change"].iloc[-1] == -2.0
    risk = summ[(summ.ticker == "AAPL") & (summ.section_group == "Risk Factors")]
    assert risk["average_sentiment"].isna().sum() == 2       # repeated risk factors -> NaN, as in Fatima's rule
    assert (tmp_path / "features" / "sec_sentiment_market_daily.parquet").exists()
    again = bs.download(["AAPL", "GOOGL"], "2006-01-01", "2026-12-31", fetch_json,
                        lambda url: pytest.fail("re-downloaded a saved filing"), log=lambda *a: None)
    assert len(again) == len(paras)
