"""Build the SEC filing sentiment features for 50 firms, 2006 to today.

Owner: Fatima (NLP). Uses src/sec_filings.py (download and cleaning) and
src/sentiment.py (her Week 3 rules, scaled up). Run from the repo root:

    python scripts/build_sentiment.py                 # download new filings, score, rebuild features
    python scripts/build_sentiment.py --skip-download # rescore/rebuild from filings already downloaded
    python scripts/build_sentiment.py --firms AAPL MSFT --start 2020-01-01   # a quick small run

Needs SEC_USER_AGENT in .env (name or project plus a contact email) and, for
scoring, `pip install transformers torch` (FinBERT downloads once, about 440 MB).

Incremental: filings already in data/processed/sec/paragraphs_full.parquet are
not downloaded again, and paragraphs already in the FinBERT cache are not
scored again, so a rerun after an interruption, or a later refresh, only does
the new part. The first full run downloads several thousand filings and scores
a few hundred thousand paragraphs on the CPU: expect several hours.

Writes
  data/processed/sec/filings_full.csv, paragraphs_full.parquet, problems_full.csv, coverage_full.csv
  data/processed/sec/finbert_cache.parquet                       (scores by text hash)
  data/processed/features/sec_sentiment_scored.parquet           (one row per kept paragraph)
  data/processed/features/sec_sentiment_filings.parquet          (one row per filing and section)
  data/processed/features/sec_sentiment_daily.parquet            (firm x section x trading day; dashboard)
  data/processed/features/sec_sentiment_market_daily.parquet     (market-wide index; notebooks 14 and 15)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import sec_filings as sec  # noqa: E402
from src import sentiment as se  # noqa: E402

PROC = ROOT / "data" / "processed"
SEC_DIR = PROC / "sec"
FEATURES = PROC / "features"


def companies(tickers, fetch_json=sec.get_json) -> pd.DataFrame:
    """ticker -> every CIK to read (current one from the SEC ticker file + predecessors)."""
    cur = sec.cik_map(list(tickers), fetch_json=fetch_json)[["ticker", "cik"]]
    extra = [{"ticker": t, "cik": c} for t, ciks in se.PREDECESSOR_CIKS.items() if t in tickers for c in ciks]
    return pd.concat([cur, pd.DataFrame(extra, columns=["ticker", "cik"])], ignore_index=True).drop_duplicates()


def download(tickers, start, end, fetch_json=sec.get_json, fetch_text=sec.get_text, log=print):
    """New filings and their paragraphs, appended to the saved corpus."""
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    para_path, fil_path, prob_path = SEC_DIR / "paragraphs_full.parquet", SEC_DIR / "filings_full.csv", SEC_DIR / "problems_full.csv"
    old_paras = pd.read_parquet(para_path) if para_path.exists() else pd.DataFrame()
    old_fil = pd.read_csv(fil_path) if fil_path.exists() else pd.DataFrame()
    have = set(old_paras["accession"]) if not old_paras.empty else set()
    filings, paras, problems = [], [], []
    for _, co in companies(tickers, fetch_json).iterrows():
        try:
            f = sec.list_filings(co["cik"], start, end, fetch_json=fetch_json)
        except Exception as exc:  # noqa: BLE001
            problems.append({"ticker": co["ticker"], "accession": None, "problem": f"listing failed for CIK {co['cik']}: {exc!r}"})
            continue
        f.insert(0, "ticker", co["ticker"])
        filings.append(f)
        new = f[~f["accession"].isin(have)]
        log(f"{co['ticker']} (CIK {co['cik']}): {len(f)} filings, {len(new)} new")
        for _, row in new.iterrows():
            try:
                html = sec.download_filing(row["cik"], row["accession"], row["primary_document"], fetch_text=fetch_text)
            except Exception as exc:  # noqa: BLE001
                problems.append({"ticker": row["ticker"], "accession": row["accession"], "problem": f"download failed: {exc!r}"})
                continue
            p = sec.filing_paragraphs(html)
            if p.empty:
                problems.append({"ticker": row["ticker"], "accession": row["accession"], "problem": "no paragraphs extracted"})
                continue
            paras.append(p.assign(ticker=row["ticker"], form=row["form"], filing_date=row["filing_date"],
                                  accession=row["accession"]))
    cols = ["ticker", "form", "filing_date", "accession", "paragraph_id", "section", "text"]
    new_paras = pd.concat(paras, ignore_index=True)[cols] if paras else pd.DataFrame(columns=cols)
    all_paras = pd.concat([old_paras, new_paras], ignore_index=True) if not old_paras.empty else new_paras
    all_fil = pd.concat([old_fil] + filings, ignore_index=True) if filings or not old_fil.empty else pd.DataFrame()
    if not all_fil.empty:
        all_fil = all_fil.drop_duplicates("accession", keep="last").sort_values(["ticker", "filing_date"])
    all_paras.to_parquet(para_path, index=False)
    all_fil.to_csv(fil_path, index=False)
    pd.DataFrame(problems, columns=["ticker", "accession", "problem"]).to_csv(prob_path, index=False)
    if not all_fil.empty:
        se.coverage(all_fil).to_csv(SEC_DIR / "coverage_full.csv")
    log(f"Corpus: {all_paras['accession'].nunique() if len(all_paras) else 0} filings, {len(all_paras):,} paragraphs; "
        f"{len(problems)} problems (see data/processed/sec/problems_full.csv)")
    return all_paras


def build_features(paras: pd.DataFrame, scorer, trading_days, start=se.START, log=print):
    sel = se.select_sections(paras)
    sel["is_new"] = se.mark_new_risk_factors(sel)
    log(f"Paragraphs: {len(sel):,} MD&A and Risk Factors; {int((~sel['is_new']).sum()):,} repeated risk factors dropped")
    keep = sel[sel["is_new"]].copy()
    scores = se.score_texts(keep["clean_text"], scorer, cache_path=SEC_DIR / "finbert_cache.parquet")
    sel = sel.join(scores[["sentiment", "confidence"]], how="left")
    FEATURES.mkdir(parents=True, exist_ok=True)
    sel.drop(columns=["text"]).to_parquet(FEATURES / "sec_sentiment_scored.parquet", index=False)
    summ = se.align_to_trading_days(se.filing_summary(sel), trading_days)
    summ.to_parquet(FEATURES / "sec_sentiment_filings.parquet", index=False)
    daily = se.daily_features(summ, trading_days, start=start)
    daily.to_parquet(FEATURES / "sec_sentiment_daily.parquet", index=False)
    market = se.market_daily(daily)
    market.to_parquet(FEATURES / "sec_sentiment_market_daily.parquet")
    log(f"Features: {summ['accession'].nunique()} filings, {daily['ticker'].nunique()} firms, "
        f"daily {daily['date'].min():%Y-%m-%d} to {daily['date'].max():%Y-%m-%d}")
    return summ, daily, market


def trading_calendar():
    panel = PROC / "daily_panel.parquet"
    if not panel.exists():
        raise SystemExit("data/processed/daily_panel.parquet not found: run scripts/refresh_data.py --steps panel first")
    return pd.DatetimeIndex(pd.read_parquet(panel).index)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--firms", nargs="+", default=list(se.FIRMS), help="tickers (default: the 50 in src/sentiment.py)")
    ap.add_argument("--start", default=se.START)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--skip-download", action="store_true", help="use only filings already downloaded")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)
    if args.skip_download:
        paras = pd.read_parquet(SEC_DIR / "paragraphs_full.parquet")
    else:
        paras = download(args.firms, args.start, args.end)
    paras = paras[paras["ticker"].isin(args.firms)]
    build_features(paras, se.finbert_scorer(batch_size=args.batch_size), trading_calendar(), start=args.start)


if __name__ == "__main__":
    main()
