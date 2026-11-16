import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from anomaly.dataset import WindowDataset
from anomaly.detectors import AutoencoderDetector
from anomaly.features import FEATURES
from anomaly.model import LSTMAutoencoder
from anomaly.split import apply_scaler, fit_scaler
from anomaly.train import (
    TrainConfig,
    load_checkpoint,
    mean_loss,
    save_checkpoint,
    set_seed,
    train,
)


def quiet(_):
    pass


@pytest.fixture
def scaled(features):
    return apply_scaler(features, fit_scaler(features))


@pytest.fixture
def datasets(scaled):
    return WindowDataset(scaled, "train"), WindowDataset(scaled, "val")


def test_same_seed_gives_identical_training(datasets):
    # The seed must be set before the model is built too: building it draws
    # the random starting weights.
    config = TrainConfig(epochs=2, batch_size=64, seed=3)
    set_seed(3)
    a = train(LSTMAutoencoder(), *datasets, config, log=quiet)
    set_seed(3)
    b = train(LSTMAutoencoder(), *datasets, config, log=quiet)
    assert a == b


def test_loss_falls_when_learning(datasets):
    history = train(LSTMAutoencoder(), *datasets, TrainConfig(epochs=5, batch_size=32), log=quiet)
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_stops_early_when_validation_never_improves(datasets):
    # With a learning rate of 0 the weights never change, so epoch 1 stays the best.
    config = TrainConfig(epochs=50, lr=0.0, patience=3)
    history = train(LSTMAutoencoder(), *datasets, config, log=quiet)
    assert len(history) == 1 + 3
    assert [h["best"] for h in history] == [True, False, False, False]


def test_keeps_the_best_epoch_not_the_last(datasets):
    model = LSTMAutoencoder()
    history = train(model, *datasets, TrainConfig(epochs=6, batch_size=32), log=quiet)
    best = min(h["val_loss"] for h in history)
    assert mean_loss(model, DataLoader(datasets[1], batch_size=256), "cpu") == pytest.approx(best)


def test_can_overfit_a_tiny_subset(datasets):
    """If the model cannot memorise 4 windows, something is broken.

    Synthetic returns are pure noise, so memorising them is slow: measured,
    loss reaches 45% of its start after 500 steps and 0.8% after 3,000.
    About 20 seconds."""
    tiny = Subset(datasets[0], range(0, 4 * 30, 30))  # 4 non-overlapping windows
    config = TrainConfig(epochs=3000, batch_size=4, lr=3e-3, patience=3000)
    set_seed(0)
    history = train(LSTMAutoencoder(), tiny, tiny, config, log=quiet)
    assert history[-1]["train_loss"] < 0.05 * history[0]["train_loss"]


def test_checkpoint_round_trip_gives_identical_scores(tmp_path, features):
    torch.manual_seed(0)
    model, scaler = LSTMAutoencoder(), fit_scaler(features)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, scaler, TrainConfig())

    loaded_model, loaded_scaler = load_checkpoint(path)
    pd.testing.assert_frame_equal(loaded_scaler, scaler)
    a = AutoencoderDetector(model, scaler).score(features)
    b = AutoencoderDetector(loaded_model, loaded_scaler).score(features)
    pd.testing.assert_series_equal(a, b)


def test_model_scaling_comes_from_training_rows_only(features):
    garbage = features.copy()
    garbage.loc[garbage["split"] != "train", FEATURES] = 1e6
    pd.testing.assert_frame_equal(fit_scaler(features), fit_scaler(garbage))
