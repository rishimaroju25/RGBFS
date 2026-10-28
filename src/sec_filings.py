"""SEC EDGAR filings: find, download and clean 10-K and 10-Q text.

Owner: Fatima (NLP / sentiment). Used by notebooks/02_sec_filings.ipynb.

What it does
------------
1. `cik_map()` turns tickers into SEC company IDs (CIKs).
2. `list_filings()` finds 10-K and 10-Q filings in a date range. It reads the
   company's recent-filings list AND the older index files listed under
   `filings.files`, because older filings (for example Alphabet's 2022
   filings) are only in those older files.
3. `download_filing()` fetches the main filing document and caches it under
   data/raw/sec/, so re-running does not hit the SEC again.
4. `filing_paragraphs()` turns the filing HTML into clean paragraphs: it drops
   the hidden inline-XBRL header (the "nvda-20221030 0001045810 ..." codes),
   hidden elements, tables of numbers and short lines like checkboxes and page
   numbers, and tags each paragraph with the filing section it sits in
   ("Item 1A", "Item 7", ...).

SEC rules (https://www.sec.gov/os/accessing-edgar-data): send a User-Agent that
names you and gives a contact email, and stay under 10 requests per second.
Set SEC_USER_AGENT in a .env file in the repo root, for example
    SEC_USER_AGENT=RGBFS student project your_email@charlotte.edu

Bug fixed from the first draft: the archive URL must use the COMPANY's CIK.
The first draft took the CIK from the first part of the accession number,
which is the ID of whoever submitted the filing. Companies that file through
an agent (Microsoft in 2022) got 404 errors.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEC_RAW = ROOT / "data" / "raw" / "sec"
PAUSE = 0.15  # seconds between requests, well under the SEC's 10 per second

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[.:\-\u2013\u2014]?\s*(.{0,150})$", re.IGNORECASE)


def user_agent() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise RuntimeError(
            "Set SEC_USER_AGENT (your name or project plus a contact email) in a .env file in the "
            "repo root, e.g. SEC_USER_AGENT=RGBFS student project your_email@charlotte.edu"
        )
    return ua


def _get(url: str, session=None, retries: int = 3):
    """GET with the SEC headers, a pause, and retries on rate limits or server errors."""
    import requests

    http = session or requests
    for attempt in range(1, retries + 1):
        resp = http.get(url, headers={"User-Agent": user_agent()}, timeout=30)
        time.sleep(PAUSE)
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(2 * attempt)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def get_json(url: str):
    return _get(url).json()


def get_text(url: str) -> str:
    return _get(url).text


# ------------------------------------------------------------ finding ----
def cik_map(tickers: list[str], fetch_json=get_json) -> pd.DataFrame:
    """ticker -> CIK (int) and company name, from the SEC's ticker file."""
    raw = pd.DataFrame.from_dict(fetch_json(TICKERS_URL), orient="index")
    raw["ticker"] = raw["ticker"].str.upper()
    out = raw[raw["ticker"].isin([t.upper() for t in tickers])]
    out = out.rename(columns={"cik_str": "cik", "title": "company"})[["ticker", "cik", "company"]]
    missing = sorted(set(t.upper() for t in tickers) - set(out["ticker"]))
    if missing:
        print("Not in the SEC ticker file (renamed or delisted?):", missing)
    return out.drop_duplicates("ticker").reset_index(drop=True)


def _rows(block: dict) -> pd.DataFrame:
    cols = ["form", "filingDate", "reportDate", "accessionNumber", "primaryDocument"]
    return pd.DataFrame({c: block.get(c, [None] * len(block["form"])) for c in cols})


def list_filings(cik: int, start: str, end: str, forms=("10-K", "10-Q"),
                 fetch_json=get_json) -> pd.DataFrame:
    """All filings of the given forms with filing dates in [start, end].

    Reads the recent list plus every older index file whose date range
    overlaps [start, end].
    """
    main = fetch_json(SUBMISSIONS_URL.format(name=f"CIK{int(cik):010d}.json"))
    blocks = [_rows(main["filings"]["recent"])]
    for f in main["filings"].get("files", []):
        lo, hi = f.get("filingFrom", "0000"), f.get("filingTo", "9999")
        if hi >= start and lo <= end:
            blocks.append(_rows(fetch_json(SUBMISSIONS_URL.format(name=f["name"]))))
    df = pd.concat(blocks, ignore_index=True)
    df = df[df["form"].isin(forms) & df["filingDate"].between(start, end)]
    df = df.drop_duplicates("accessionNumber").sort_values("filingDate")
    df.insert(0, "cik", int(cik))
    return df.rename(columns={"filingDate": "filing_date", "reportDate": "report_date",
                              "accessionNumber": "accession", "primaryDocument": "primary_document"}
                     ).reset_index(drop=True)


def filing_url(cik: int, accession: str, primary_document: str) -> str:
    """Archive URL. Uses the company's CIK, never the accession prefix."""
    return ARCHIVE_URL.format(cik=int(cik), acc=accession.replace("-", ""), doc=primary_document)


def download_filing(cik: int, accession: str, primary_document: str, fetch_text=get_text) -> str:
    """Filing HTML, cached in data/raw/sec/<cik>/<accession>.htm."""
    path = SEC_RAW / str(int(cik)) / f"{accession}.htm"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    html = fetch_text(filing_url(cik, accession, primary_document))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html


# ----------------------------------------------------------- cleaning ----
def _is_hidden(tag) -> bool:
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def filing_paragraphs(html: str, min_words: int = 15) -> pd.DataFrame:
    """Clean paragraphs from a filing's HTML, with the section each one is in.

    Drops the inline-XBRL header, hidden elements, scripts, styles and tables,
    then keeps block-level text with at least `min_words` words.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    drop = soup.find_all(["script", "style", "table"])
    drop += [t for t in soup.find_all(True) if t.name.lower() == "ix:header" or _is_hidden(t)]
    for tag in drop:
        if not tag.decomposed:  # skip tags already removed with a parent
            tag.decompose()

    rows, section = [], "front matter"
    for block in soup.find_all(["p", "div", "li"]):
        if block.find(["p", "div", "li"]):  # only innermost blocks, so text is not counted twice
            continue
        text = re.sub(r"\s+", " ", block.get_text(" ", strip=True)).strip()
        if not text:
            continue
        m = ITEM_RE.match(text)
        if m and len(text.split()) <= 15:
            section = f"Item {m.group(1).upper()}"
            continue
        if len(text.split()) >= min_words and not re.search(r"[☐☒]", text):
            rows.append({"section": section, "text": text})
    out = pd.DataFrame(rows, columns=["section", "text"])
    out.insert(0, "paragraph_id", range(1, len(out) + 1))
    return out


def build_corpus(tickers: list[str], start: str, end: str, forms=("10-K", "10-Q"),
                 min_words: int = 15, fetch_json=get_json, fetch_text=get_text):
    """Download and clean every matching filing for `tickers`.

    Returns (filings, paragraphs, problems): one row per filing, one row per
    paragraph, and one row per filing that failed or produced no paragraphs.
    """
    companies = cik_map(tickers, fetch_json=fetch_json)
    filings, paras, problems = [], [], []
    for _, co in companies.iterrows():
        try:
            f = list_filings(co["cik"], start, end, forms, fetch_json=fetch_json)
        except Exception as exc:
            problems.append({"ticker": co["ticker"], "accession": None, "problem": f"listing failed: {exc!r}"})
            continue
        f.insert(0, "ticker", co["ticker"])
        f.insert(2, "company", co["company"])
        filings.append(f)
        for _, row in f.iterrows():
            try:
                html = download_filing(row["cik"], row["accession"], row["primary_document"], fetch_text=fetch_text)
            except Exception as exc:
                problems.append({"ticker": row["ticker"], "accession": row["accession"],
                                 "problem": f"download failed: {exc!r}"})
                continue
            p = filing_paragraphs(html, min_words=min_words)
            if p.empty:
                problems.append({"ticker": row["ticker"], "accession": row["accession"],
                                 "problem": "no paragraphs extracted"})
                continue
            p = p.assign(ticker=row["ticker"], form=row["form"],
                         filing_date=row["filing_date"], accession=row["accession"])
            paras.append(p)
    filings = pd.concat(filings, ignore_index=True) if filings else pd.DataFrame()
    paras = pd.concat(paras, ignore_index=True) if paras else pd.DataFrame()
    if not paras.empty:
        paras = paras[["ticker", "form", "filing_date", "accession", "paragraph_id", "section", "text"]]
    return filings, paras, pd.DataFrame(problems, columns=["ticker", "accession", "problem"])
