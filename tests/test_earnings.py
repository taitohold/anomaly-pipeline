import pandas as pd

from anomaly.earnings import earnings_mask, fetch_earnings_filings
from anomaly.evaluate import with_positions


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, pages):
        self.pages = pages
        self.urls = []

    def get(self, url, headers, timeout):
        assert headers["User-Agent"] == "test agent"
        self.urls.append(url)
        return FakeResponse(self.pages[url.rsplit("/", 1)[-1]])


def table(rows):
    form, items, accepted = zip(*rows)
    return {"form": list(form), "items": list(items), "acceptanceDateTime": list(accepted)}


def test_keeps_only_8k_item_202_since_the_start_date():
    pages = {
        "company_tickers.json": {"0": {"cik_str": 1067983, "ticker": "BRK-B"}},
        "CIK0001067983.json": {
            "filings": {
                "recent": table(
                    [
                        ("8-K", "2.02,9.01", "2025-05-03T08:00:00.000Z"),  # keep
                        ("8-K", "5.02", "2025-04-01T16:05:00.000Z"),  # other 8-K item
                        ("10-Q", "", "2025-05-03T08:01:00.000Z"),  # other form
                    ]
                ),
                "files": [
                    {"name": "old-1.json", "filingTo": "2025-01-10"},
                    {"name": "old-2.json", "filingTo": "2019-12-31"},  # too old: not fetched
                ],
            }
        },
        "old-1.json": table(
            [
                ("8-K", "2.02", "2025-01-08T16:30:00.000Z"),  # keep
                ("8-K", "2.02", "2024-10-30T16:30:00.000Z"),  # before since
            ]
        ),
    }
    session = FakeSession(pages)
    out = fetch_earnings_filings(["BRK.B"], "test agent", "2025-01-01", session, pause=0)

    assert sorted(out["accepted"]) == ["2025-01-08T16:30:00.000Z", "2025-05-03T08:00:00.000Z"]
    assert set(out["symbol"]) == {"BRK.B"}
    assert not any(u.endswith("old-2.json") for u in session.urls)


def test_mask_starts_the_next_day_for_after_close_filings():
    days = pd.bdate_range("2025-03-03", "2025-03-14")  # Mon 3rd .. Fri 14th
    positions = with_positions(pd.DataFrame({"symbol": "AAA", "date": days}))
    filings = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "accepted": [
                "2025-03-05T16:05:00.000Z",  # Wed after close -> reacts Thu 6th (pos 3)
                "2025-03-12T07:30:00.000Z",  # Wed before open -> reacts Wed 12th (pos 7)
            ],
        }
    )
    mask = earnings_mask(filings, positions)
    assert sorted(mask["pos"]) == [2, 3, 4, 6, 7, 8]
