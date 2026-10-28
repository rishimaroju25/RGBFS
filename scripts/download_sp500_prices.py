"""Download daily prices for every stock that was in the S&P 500 since 2005.

Run from the repo root:

    python scripts/download_sp500_prices.py            # 2005-01-01 to today
    python scripts/download_sp500_prices.py --start 2005-01-01 --end 2026-10-23
    python scripts/download_sp500_prices.py --retry-empty  # retry tickers logged as empty

What it does
  1. Downloads the point-in-time S&P 500 membership file (fja05680/sp500) to
     data/raw/ the first time.
  2. Downloads daily adjusted prices from Yahoo Finance for every ticker that
     was a member at any point in the date range, plus ^GSPC and SPY.
     Prices are cached per ticker in data/raw/prices/, so if the run stops
     (network error, rate limit) just run it again and it picks up where it
     left off. Each ticker missing from a batch is retried on its own before
     it is listed in data/raw/prices_empty.csv; run with --retry-empty to try
     those again.
  3. Writes data/processed/:
       sp500_close.parquet       adjusted close, dates x tickers
       sp500_volume.parquet      volume, dates x tickers
       sp500_membership.parquet  True when a ticker was in the index that day
       sp500_coverage_report.csv which members have prices and which do not

The full run is roughly 1,000 tickers in batches of 50, with a pause between
batches to avoid Yahoo's rate limits. Expect it to take several minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import prices as px  # noqa: E402
from src import universe as uni  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=px.START)
    ap.add_argument("--end", default=None, help="exclusive end date (yfinance convention); default today")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between batches")
    ap.add_argument("--refresh-membership", action="store_true", help="re-download the membership file")
    ap.add_argument("--retry-empty", action="store_true",
                    help="try again for tickers logged as empty on earlier runs (data/raw/prices_empty.csv)")
    args = ap.parse_args()

    changes = uni.load_membership_changes(refresh=args.refresh_membership)
    last = uni.membership_last_date(changes)
    print(f"Membership file covers {changes['date'].min().date()} to {last.date()}")
    print("  (changes after that date are not in the file yet; see docs/survivorship_bias.md)")

    tickers = uni.tickers_in_period(changes, args.start, args.end)
    print(f"{len(tickers)} tickers were S&P 500 members at some point since {args.start}")

    summary = px.download_prices(
        tickers + px.BENCHMARKS + sorted(set(uni.TICKER_ALIASES.values()) - set(tickers)),
        start=args.start, end=args.end, batch_size=args.batch_size, pause=args.pause,
        retry_empty=args.retry_empty,
    )
    print("Download summary:", summary)

    close, volume, membership, report = px.assemble(changes, start=args.start, end=args.end)
    px.save_processed(close, volume, membership, report)

    cov = px.coverage_by_day(close, membership)
    weighted = report["priced_member_days"].sum() / report["member_days"].sum()
    print()
    print(f"Saved data/processed/ tables: {close.shape[0]} trading days x {close.shape[1]} series")
    print(f"Tickers with no Yahoo data while in the index: {int(report['no_data'].sum())} of {len(report)}")
    print(f"Share of member-days with a price: {weighted:.1%}")
    print(f"Lowest daily share of members priced: {cov['share_priced'].min():.1%} "
          f"on {cov['share_priced'].idxmin().date()}")
    print(f"Tickers to check for symbol reuse (more than one membership spell): "
          f"{int(report['check_ticker_reuse'].sum())}")
    if report["prices_from"].notna().any():
        print("Renamed tickers filled from their current symbol: "
              + ", ".join(f"{o}->{n}" for o, n in report["prices_from"].dropna().items()))


if __name__ == "__main__":
    main()
