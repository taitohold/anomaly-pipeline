"""Download earnings filing times from SEC EDGAR into data/raw/earnings.csv.

Run with: uv run --env-file .env python scripts/download_earnings.py
Needs SEC_USER_AGENT in .env: your name and a contact email, as the SEC requires.
"""

import os
from pathlib import Path

import pandas as pd

from anomaly.download import TICKERS
from anomaly.earnings import fetch_earnings_filings, reaction_dates

OUT = Path("data/raw/earnings.csv")
SINCE = "2023-01-01"  # covers the validation and test periods

if OUT.exists():
    raise SystemExit(f"{OUT} already exists. Raw data is never overwritten; move it first.")
user_agent = os.environ.get("SEC_USER_AGENT")
if not user_agent:
    raise SystemExit("Set SEC_USER_AGENT in .env, e.g. SEC_USER_AGENT=Jane Doe jane@example.com")

filings = fetch_earnings_filings(TICKERS, user_agent, SINCE)
OUT.parent.mkdir(parents=True, exist_ok=True)
filings.to_csv(OUT, index=False)

print(f"Wrote {len(filings)} earnings filings for {filings['symbol'].nunique()} stocks to {OUT}")
per_stock = filings["symbol"].value_counts().reindex(TICKERS, fill_value=0)
print("Fewest filings:", per_stock.nsmallest(5).to_dict())

# Check the time-zone assumption: US earnings come out before the 9:30 open or
# after the 4pm close, so Eastern times should cluster there. If they cluster
# 4 to 5 hours later instead, the times are really UTC.
hours = pd.to_datetime(filings["accepted"].str.replace("Z", "", regex=False)).dt.hour
print("Filings by hour of day:", hours.value_counts().sort_index().to_dict())
print(f"After the close (Eastern): {reaction_dates(filings)['after_close'].mean():.0%}")
