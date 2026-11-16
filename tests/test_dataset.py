import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from anomaly.dataset import WINDOW, WindowDataset
from anomaly.features import FEATURES
from anomaly.split import SPLITS


def test_rejects_unknown_split(features):
    with pytest.raises(ValueError):
        WindowDataset(features, "training")


@pytest.mark.parametrize("split", SPLITS)
def test_length_is_one_window_per_possible_end_day(features, split):
    rows_per_stock = features[features["split"] == split].groupby("symbol").size()
    expected = int((rows_per_stock - WINDOW + 1).clip(lower=0).sum())
    ds = WindowDataset(features, split)
    assert len(ds) == expected
    assert len(ds.index) == expected


def test_items_are_float32_days_by_features(features):
    item = WindowDataset(features, "train")[0]
    assert isinstance(item, torch.Tensor)
    assert item.dtype == torch.float32
    assert item.shape == (WINDOW, len(FEATURES))


@pytest.mark.parametrize("i", [0, 1, 250, -1])
def test_item_holds_the_rows_it_says_it_came_from(features, i):
    ds = WindowDataset(features, "val")
    i = i % len(ds)
    symbol, end = ds.index[i]
    stock = features[features["symbol"] == symbol].sort_values("date")
    expected = stock[stock["date"] <= end].tail(WINDOW)[FEATURES].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(ds[i].numpy(), expected)


@pytest.mark.parametrize("split", SPLITS)
def test_every_window_stays_inside_its_split(features, split):
    """Look up each window's days in the full frame: all must carry this split."""
    ds = WindowDataset(features, split)
    by_symbol = {s: g.sort_values("date") for s, g in features.groupby("symbol")}
    for symbol, end in ds.index:
        stock = by_symbol[symbol]
        days = stock[stock["date"] <= end].tail(WINDOW)
        assert len(days) == WINDOW
        assert (days["split"] == split).all(), f"{symbol} window ending {end} leaves {split}"


def test_first_val_window_ends_window_minus_one_days_into_val(features):
    ds = WindowDataset(features, "val")
    for symbol, stock in features.groupby("symbol"):
        val_days = stock[stock["split"] == "val"].sort_values("date")["date"]
        first_end = min(end for s, end in ds.index if s == symbol)
        assert first_end == val_days.iloc[WINDOW - 1]


def test_row_order_of_the_input_does_not_matter(features):
    shuffled = features.sample(frac=1.0, random_state=0)
    a = WindowDataset(features, "train")
    b = WindowDataset(shuffled, "train")
    assert sorted(a.index) == sorted(b.index)
    lookup = {key: k for k, key in enumerate(b.index)}
    for k in [0, 10, len(a) - 1]:
        torch.testing.assert_close(a[k], b[lookup[a.index[k]]])


def test_window_longer_than_the_split_gives_no_items(features):
    assert len(WindowDataset(features, "test", window=10_000)) == 0


def test_works_with_a_dataloader(features):
    loader = DataLoader(WindowDataset(features, "train"), batch_size=64, shuffle=True)
    batch = next(iter(loader))
    assert batch.shape == (64, WINDOW, len(FEATURES))
