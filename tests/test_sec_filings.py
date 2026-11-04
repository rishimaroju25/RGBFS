"""Tests for the offline parts of src/sec_filings.py (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import sec_filings as sf  # noqa: E402

LONG = "Our business could be harmed by higher interest rates, inflation and weaker demand in many of the markets where we operate."


def test_headings_inside_tables_are_kept():
    # Amazon-style layout: item headings sit in small tables
    html = f"""<html><body>
    <table><tr><td>Item 1A.</td><td>Risk Factors</td></tr></table>
    <p>{LONG}</p>
    <table><tr><td>Item 7.</td><td>Management's Discussion and Analysis</td></tr></table>
    <p>{LONG} Revenue grew.</p>
    <table><tr><td>Net sales</td><td>469,822</td><td>513,983</td></tr></table>
    </body></html>"""
    p = sf.filing_paragraphs(html)
    assert list(p["section"]) == ["Item 1A", "Item 7"]
    assert not p["text"].str.contains("469,822").any()     # number tables still removed


def test_plain_headings_and_hidden_xbrl():
    html = f"""<html><body><div style="display:none"><ix:header>aapl-20220625 0000320193 us-gaap:Revenue</ix:header></div>
    <div>Item 2. Management's Discussion and Analysis</div><p>{LONG}</p></body></html>"""
    p = sf.filing_paragraphs(html)
    assert list(p["section"]) == ["Item 2"] and not p["text"].str.contains("us-gaap").any()
