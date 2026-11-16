"""Earnings release days from SEC EDGAR, used to mark real events in the test set.

Companies announce quarterly results in an 8-K filing with item 2.02
("Results of Operations and Financial Condition"). Those days cause most of
the big real moves in large stocks. A detector that flags one is probably
right, so precision is also reported with alarms on these days set aside.
This is an outside source that does not depend on any detector.
"""

import time

import numpy as np
import pandas as pd
import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
MARKET_CLOSE = "16:00"
MASK_DAYS = 1  # also mask 1 trading day either side of the reaction day


def fetch_earnings_filings(
    tickers: list[str], user_agent: str, since: str, session=requests, pause: float = 0.2
) -> pd.DataFrame:
    """Return symbol and accepted (filing time) for every 8-K item 2.02 since `since`.

    The SEC requires a User-Agent naming who is making the requests, and asks
    for at most 10 requests per second; `pause` keeps well under that.
    EDGAR keeps each company's most recent filings in one table and older ones
    in extra files, which are fetched only if they reach back past `since`.
    """
    headers = {"User-Agent": user_agent}

    def get(url):
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        time.sleep(pause)
        return response.json()

    cik_by_ticker = {row["ticker"]: row["cik_str"] for row in get(TICKERS_URL).values()}
    rows = []
    for ticker in tickers:
        cik = cik_by_ticker[ticker.replace(".", "-")]  # EDGAR writes BRK.B as BRK-B
        submissions = get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
        tables = [submissions["filings"]["recent"]]
        for extra in submissions["filings"].get("files", []):
            if extra["filingTo"] >= since:
                tables.append(get(SUBMISSIONS_URL.format(name=extra["name"])))
        for table in tables:
            for form, items, accepted in zip(
                table["form"], table["items"], table["acceptanceDateTime"]
            ):
                if form == "8-K" and "2.02" in items.split(",") and accepted[:10] >= since:
                    rows.append({"symbol": ticker, "accepted": accepted})
    return pd.DataFrame(rows, columns=["symbol", "accepted"]).drop_duplicates()


def reaction_dates(filings: pd.DataFrame) -> pd.DataFrame:
    """Add "after_close": whether the filing came after the 4pm market close,
    in which case the market first reacts on the next trading day.

    EDGAR's acceptanceDateTime is UTC (the "Z"). It is converted to US
    Eastern time, which handles daylight saving. Checked on the first download:
    in UTC, filings clustered at 10-12h and 20-21h, which is 6-8am and 4-5pm
    Eastern, the usual before-open and after-close release times."""
    out = filings.copy()
    accepted = (
        pd.to_datetime(out["accepted"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    out["filed"] = accepted.dt.normalize()
    out["after_close"] = accepted.dt.strftime("%H:%M") >= MARKET_CLOSE
    return out


def earnings_mask(filings: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Trading days (symbol, pos) within MASK_DAYS of each earnings reaction day.

    `positions` comes from evaluate.with_positions and gives each stock's
    trading days. The reaction day is the filing day, or the next trading day
    if the filing came after the close."""
    days = []
    filings = reaction_dates(filings)
    for symbol, stock in positions.groupby("symbol"):
        dates = stock["date"].to_numpy()
        pos = stock["pos"].to_numpy()
        for row in filings[filings["symbol"] == symbol].itertuples():
            i = np.searchsorted(
                dates, np.datetime64(row.filed), side="right" if row.after_close else "left"
            )
            for j in range(i - MASK_DAYS, i + MASK_DAYS + 1):
                if 0 <= j < len(dates):
                    days.append({"symbol": symbol, "pos": pos[j]})
    return pd.DataFrame(days, columns=["symbol", "pos"]).drop_duplicates()
