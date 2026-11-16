import numpy as np
import pandas as pd
import pytest

from anomaly.features import FEATURES
from anomaly.split import apply_scaler, assign_split, fit_scaler


def test_split_boundaries():
    dates = pd.Series(pd.to_datetime(["2022-12-30", "2023-01-03", "2024-12-31", "2025-01-02"]))
    assert list(assign_split(dates)) == ["train", "val", "val", "test"]


def test_scaler_ignores_validation_and_test_rows(features):
    """Replace every non-training value with garbage: the fitted stats must not move."""
    garbage = features.copy()
    held_out = garbage["split"] != "train"
    garbage.loc[held_out, FEATURES] = 1e6

    pd.testing.assert_frame_equal(fit_scaler(features), fit_scaler(garbage))


def test_scaled_training_data_has_mean_0_std_1(features):
    scaled = apply_scaler(features, fit_scaler(features))
    train = scaled[scaled["split"] == "train"]
    for _, stock in train.groupby("symbol"):
        np.testing.assert_allclose(stock[FEATURES].mean(), 0.0, atol=1e-9)
        np.testing.assert_allclose(stock[FEATURES].std(), 1.0, atol=1e-9)


def test_apply_scaler_rejects_unknown_stock(features):
    scaler = fit_scaler(features)
    unknown = features.copy()
    unknown.loc[0, "symbol"] = "ZZZ"
    with pytest.raises(ValueError):
        apply_scaler(unknown, scaler)
