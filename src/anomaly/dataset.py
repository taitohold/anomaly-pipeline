"""Sliding windows of features, served to PyTorch one window at a time."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from anomaly.features import FEATURES
from anomaly.split import SPLITS

WINDOW = 30


class WindowDataset(Dataset):
    """Every run of `window` consecutive days of one stock, within one split.

    Item i is a float32 tensor of shape (window, len(FEATURES)): one row per
    day, oldest first, with the columns in the order of FEATURES.

    A window is included only if every one of its rows has split == `split`,
    so no window ever straddles train/val or val/test. Windows never mix two
    stocks. The input may arrive in any row order.

    After construction, self.index[i] is (symbol, date of the window's last
    row) for item i, so scores can be mapped back to a stock and a day.
    """

    def __init__(self, features: pd.DataFrame, split: str, window: int = WINDOW):
        """
        features: columns symbol, date, split, and every name in FEATURES
                  (already scaled).
        split:    "train", "val", or "test". Anything else raises ValueError.
        window:   number of consecutive days per item.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.window = window
        self.index: list[tuple[str, pd.Timestamp]] = []
        self._arrays: list[np.ndarray] = []  # one (days, features) array per stock
        self._starts: list[tuple[int, int]] = []  # item i -> (which array, first row)

        # All the slow work happens once, here, so __getitem__ is just a slice.
        rows = features[features["split"] == split].sort_values(["symbol", "date"])
        for symbol, stock in rows.groupby("symbol"):
            # The split is by date, so one stock's rows in one split are an
            # unbroken run of days: every window inside it stays in the split.
            values = stock[FEATURES].to_numpy(dtype=np.float32)
            dates = stock["date"].tolist()
            array_id = len(self._arrays)
            self._arrays.append(values)
            # A run of n days holds n - window + 1 windows (none if n < window).
            for start in range(len(values) - window + 1):
                self._starts.append((array_id, start))
                self.index.append((symbol, dates[start + window - 1]))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, i: int) -> torch.Tensor:
        array_id, start = self._starts[i]
        # from_numpy shares memory with the array instead of copying it.
        return torch.from_numpy(self._arrays[array_id][start : start + self.window])
