import datetime

import pandas as pd
import pytest

from anomaly.download import fetch_bars, missing_days


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code
        self.text = str(body)

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Returns the given responses in order and records each request's params."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers, params, timeout):
        self.calls.append(dict(params))
        return self.responses.pop(0)


def bar(t, close):
    return {"t": t, "o": close, "h": close, "l": close, "c": close, "v": 1000}


def test_follows_page_tokens_until_done():
    session = FakeSession(
        [
            FakeResponse(
                {"bars": {"AAA": [bar("2024-01-02T05:00:00Z", 10.0)]}, "next_page_token": "p2"}
            ),
            FakeResponse(
                {"bars": {"BBB": [bar("2024-01-02T05:00:00Z", 20.0)]}, "next_page_token": None}
            ),
        ]
    )
    df = fetch_bars(["AAA", "BBB"], "2024-01-01", "2024-01-05", "sip", "k", "s", session)

    assert len(session.calls) == 2
    assert "page_token" not in session.calls[0]
    assert session.calls[1]["page_token"] == "p2"
    assert session.calls[0]["adjustment"] == "all"
    assert list(df["symbol"]) == ["AAA", "BBB"]
    assert list(df["close"]) == [10.0, 20.0]


def test_converts_timestamps_to_new_york_dates():
    # 05:00 UTC in January is midnight in New York
    session = FakeSession([FakeResponse({"bars": {"AAA": [bar("2024-01-02T05:00:00Z", 1.0)]}})])
    df = fetch_bars(["AAA"], "2024-01-01", "2024-01-05", "sip", "k", "s", session)
    assert df.loc[0, "date"] == datetime.date(2024, 1, 2)


def test_forbidden_feed_gives_a_clear_error():
    session = FakeSession([FakeResponse({"message": "forbidden"}, status_code=403)])
    with pytest.raises(PermissionError, match="--feed iex"):
        fetch_bars(["AAA"], "2024-01-01", "2024-01-05", "sip", "k", "s", session)


def test_missing_days_counts_gaps_per_stock():
    bars = pd.DataFrame({"symbol": ["AAA", "AAA", "AAA", "BBB", "BBB"], "date": [1, 2, 3, 1, 3]})
    assert missing_days(bars).to_dict() == {"AAA": 0, "BBB": 1}
