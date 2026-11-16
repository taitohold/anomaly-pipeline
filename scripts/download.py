"""Download daily bars for all tickers into data/raw/bars.csv.

Run with: uv run --env-file .env python scripts/download.py [--feed sip|iex]
"""

import argparse
import datetime
import os
from pathlib import Path

from anomaly.download import TICKERS, fetch_bars, missing_days

OUT = Path("data/raw/bars.csv")
START = "2016-01-01"
# Pinned so a rerun downloads the same days and reproduces the README's numbers.
# This is the end date of the original download (2026-11-14 was a Saturday, so
# the last bar is 2026-11-13).
END = "2026-11-14"

parser = argparse.ArgumentParser()
parser.add_argument("--feed", default="sip", choices=["sip", "iex"])
args = parser.parse_args()

if OUT.exists():
    raise SystemExit(f"{OUT} already exists. Raw data is never overwritten; move it first.")

key_id = os.environ.get("ALPACA_API_KEY_ID")
secret = os.environ.get("ALPACA_API_SECRET_KEY")
if not key_id or not secret:
    raise SystemExit("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in .env (see .env.example).")

bars = fetch_bars(TICKERS, START, END, args.feed, key_id, secret)

OUT.parent.mkdir(parents=True, exist_ok=True)
bars.to_csv(OUT, index=False)
OUT.with_name("bars_source.txt").write_text(
    f"Alpaca /v2/stocks/bars, timeframe=1Day, adjustment=all, feed={args.feed}, "
    f"{START} to {END}, downloaded {datetime.date.today()}\n"
)

print(f"Wrote {len(bars):,} bars for {bars['symbol'].nunique()} stocks to {OUT}")
print(f"Dates: {bars['date'].min()} to {bars['date'].max()}")
gaps = missing_days(bars)
print("Stocks missing trading days:", gaps[gaps > 0].to_dict() or "none")
