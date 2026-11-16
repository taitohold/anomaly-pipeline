import numpy as np
import pandas as pd
import pytest
import torch

from anomaly.dataset import WINDOW
from anomaly.detectors import AutoencoderDetector
from anomaly.features import FEATURES
from anomaly.model import LSTMAutoencoder, newest_day_error
from anomaly.split import fit_scaler


def test_output_has_the_same_shape_as_the_input():
    model = LSTMAutoencoder()
    x = torch.randn(5, WINDOW, len(FEATURES))
    assert model(x).shape == x.shape


def test_parameter_count_matches_the_plan():
    # encoder LSTM 4*32*(3+32) + 8*32 = 4736, to_code 32*8 + 8 = 264,
    # decoder LSTM 4*32*(8+32) + 8*32 = 5376, to_output 32*3 + 3 = 99
    n = sum(p.numel() for p in LSTMAutoencoder().parameters())
    assert n == 4736 + 264 + 5376 + 99 == 10_475


def test_score_is_the_error_on_the_last_day_only():
    torch.manual_seed(0)
    model = LSTMAutoencoder()
    x = torch.randn(4, WINDOW, len(FEATURES))
    expected = ((model(x) - x)[:, -1, :] ** 2).mean(dim=1)
    torch.testing.assert_close(newest_day_error(model, x), expected.detach())


@pytest.fixture
def detector(features):
    torch.manual_seed(0)
    return AutoencoderDetector(LSTMAutoencoder(), fit_scaler(features))


def test_every_day_after_the_first_window_gets_a_score(features, detector):
    scores = detector.score(features)
    for _, stock in features.sort_values("date").groupby("symbol"):
        assert scores[stock.index[: WINDOW - 1]].isna().all()
        assert np.isfinite(scores[stock.index[WINDOW - 1 :]]).all()


def test_score_matches_scoring_the_window_directly(features, detector):
    stock = features[features["symbol"] == "BBB"].sort_values("date")
    scaler = detector.scaler.loc["BBB"]
    x = np.stack([(stock[f] - scaler[f"{f}_mean"]) / scaler[f"{f}_std"] for f in FEATURES], axis=1)
    i = 400
    window = torch.tensor(x[i - WINDOW + 1 : i + 1], dtype=torch.float32).unsqueeze(0)
    expected = newest_day_error(detector.model, window).item()
    assert detector.score(features)[stock.index[i]] == pytest.approx(expected, rel=1e-5)


def test_scores_never_look_ahead(features, detector):
    cutoff = pd.Timestamp("2024-03-28")
    changed = features.copy()
    changed.loc[changed["date"] > cutoff, FEATURES] *= 10.0
    keep = features["date"] <= cutoff
    pd.testing.assert_series_equal(detector.score(features)[keep], detector.score(changed)[keep])


def test_row_order_of_the_input_does_not_matter(features, detector):
    shuffled = features.sample(frac=1.0, random_state=0)
    a = detector.score(features)
    b = detector.score(shuffled).reindex(a.index)
    pd.testing.assert_series_equal(a, b)
