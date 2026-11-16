"""Download daily bars from Alpaca's market data REST API."""

import pandas as pd
import requests

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "V",
    "MA", "UNH", "JNJ", "PG", "XOM", "CVX", "HD", "KO", "PEP", "WMT",
    "COST", "DIS", "NFLX", "ADBE", "CRM", "INTC", "CSCO", "BAC", "PFE", "MRK",
]  # fmt: skip

# Alpaca's short field names -> the column names used in data/raw/bars.csv
FIELDS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}


def fetch_bars(
    symbols: list[str],
    start: str,
    end: str,
    feed: str,
    key_id: str,
    secret: str,
    session=requests,
) -> pd.DataFrame:
    """Fetch split- and dividend-adjusted daily bars for all symbols.

    Alpaca returns at most 10,000 bars per request and a next_page_token when
    there are more, so this keeps requesting until the token is empty.
    `session` is anything with a requests-style get(); tests pass a fake one.
    """
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start,
        "end": end,
        "adjustment": "all",
        "feed": feed,
        "limit": 10000,
    }
    rows = []
    while True:
        response = session.get(BARS_URL, headers=headers, params=params, timeout=30)
        if response.status_code == 403:
            raise PermissionError(
                f"Alpaca refused feed={feed!r}: {response.text}. "
                "Your plan may only allow the IEX feed; rerun with --feed iex."
            )
        response.raise_for_status()
        body = response.json()
        for symbol, bars in (body.get("bars") or {}).items():
            for bar in bars:
                rows.append(
                    {"symbol": symbol, "t": bar["t"], **{v: bar[k] for k, v in FIELDS.items()}}
                )
        token = body.get("next_page_token")
        if not token:
            break
        params["page_token"] = token

    df = pd.DataFrame(rows, columns=["symbol", "t", *FIELDS.values()])
    # Daily bars are stamped at midnight New York time; keep just the date.
    df["date"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York").dt.date
    df = df.drop(columns="t")[["symbol", "date", *FIELDS.values()]]
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def missing_days(bars: pd.DataFrame) -> pd.Series:
    """Count, per stock, the trading days other stocks have but it lacks."""
    all_days = set(bars["date"])
    return bars.groupby("symbol")["date"].agg(lambda d: len(all_days - set(d)))
