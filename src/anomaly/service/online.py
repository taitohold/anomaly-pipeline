"""Score bars one at a time as they arrive, using the same code as offline.

Train/serve skew, where the live system computes features slightly differently
from the training code, is one of the most common ML production bugs. To rule
it out, the online scorer keeps the last HISTORY bars of each stock and runs
the exact same build_features and detector functions on them.
"""

import json
from collections import defaultdict, deque

import pandas as pd

from anomaly.features import build_features

# The autoencoder needs 30 feature rows, and each feature row needs 20 earlier
# bars (log return plus a 20-day lookback): 30 + 20 = 50 bars.
HISTORY = 50
BAR_FIELDS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def encode_bar(bar: dict) -> bytes:
    """One bar as a JSON message. Dates travel as YYYY-MM-DD strings."""
    out = {k: bar[k] for k in BAR_FIELDS}
    out["date"] = pd.Timestamp(out["date"]).strftime("%Y-%m-%d")
    return json.dumps(out).encode()


def decode_bar(message: bytes) -> dict:
    bar = json.loads(message)
    bar["date"] = pd.Timestamp(bar["date"])
    return bar


class OnlineScorer:
    """Keeps a rolling window of bars per stock and scores each new day.

    `detector` is any function from a features frame to a Series of scores,
    the same interface the offline evaluation uses.
    """

    def __init__(self, detector, history: int = HISTORY):
        self.detector = detector
        self.bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=history))

    def add(self, bar: dict) -> float | None:
        """Add one bar and return its day's score, or None while the stock is
        still warming up. A bar that is not newer than the stock's latest one
        (a duplicate or a replay) is ignored and also returns None."""
        window = self.bars[bar["symbol"]]
        if window and bar["date"] <= window[-1]["date"]:
            return None
        window.append(bar)
        features = build_features(pd.DataFrame(list(window)))
        if features.empty:
            return None
        score = self.detector(features).iloc[-1]  # features are sorted by date
        return None if pd.isna(score) else float(score)
