import numpy as np
import pandas as pd
import pytest
import torch

from anomaly.detectors import AutoencoderDetector, zscore_scores
from anomaly.features import build_features
from anomaly.model import LSTMAutoencoder
from anomaly.service.online import OnlineScorer, decode_bar, encode_bar
from anomaly.split import assign_split, fit_scaler
from anomaly.synthetic import make_bars


@pytest.fixture
def stream_bars():
    """About 9 months of bars for 3 stocks, in the order a replay sends them."""
    bars = make_bars(["AAA", "BBB", "CCC"], "2022-03-01", "2022-12-30", seed=5)
    return bars.sort_values(["date", "symbol"]).reset_index(drop=True)


def offline_scores(bars, detector):
    features = build_features(bars)
    return features.assign(score=detector(features)).set_index(["symbol", "date"])["score"]


def online_scores(bars, detector):
    scorer = OnlineScorer(detector)
    out = {}
    for bar in bars.to_dict("records"):
        score = scorer.add(bar)
        if score is not None:
            out[(bar["symbol"], bar["date"])] = score
    return pd.Series(out)


def assert_same_scores(bars, detector, rtol):
    offline = offline_scores(bars, detector).dropna()
    online = online_scores(bars, detector)
    assert set(online.index) == set(offline.index), "online and offline score different days"
    np.testing.assert_allclose(online.loc[offline.index], offline, rtol=rtol, atol=1e-12)


def test_online_zscore_matches_offline_exactly(stream_bars):
    assert_same_scores(stream_bars, zscore_scores, rtol=1e-9)


def test_online_autoencoder_matches_offline(stream_bars):
    """The model runs in float32, and scoring one window at a time instead of
    thousands at once changes the rounding: measured, up to 2.4e-7 relative,
    about one float32 step. 1e-5 allows for that and nothing more."""
    features = build_features(stream_bars)
    features["split"] = assign_split(features["date"])  # all 2022: training rows
    torch.manual_seed(0)
    detector = AutoencoderDetector(LSTMAutoencoder(), fit_scaler(features))
    assert_same_scores(stream_bars, detector.score, rtol=1e-5)


def test_duplicates_and_old_bars_are_ignored(stream_bars):
    scorer = OnlineScorer(zscore_scores)
    records = stream_bars.to_dict("records")
    scores = [scorer.add(b) for b in records]
    replayed = [scorer.add(b) for b in records]  # the whole stream again
    assert any(s is not None for s in scores)
    assert all(s is None for s in replayed)
    assert len(scorer.bars["AAA"]) == 50


def test_messages_round_trip():
    bar = {
        "symbol": "AAA",
        "date": pd.Timestamp("2025-03-04"),
        "open": 1.5,
        "high": 2.0,
        "low": 1.0,
        "close": 1.75,
        "volume": 1e6,
    }
    assert decode_bar(encode_bar(bar)) == bar
