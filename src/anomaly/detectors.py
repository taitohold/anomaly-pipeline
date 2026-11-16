"""The three baseline detectors. Each gives one anomaly score per row: higher is
more unusual. Scores are returned as a Series aligned to the input's index."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest

from anomaly.features import FEATURES
from anomaly.split import apply_scaler, fit_scaler

LOOKBACK = 20
MAD_TO_STD = 1.4826  # makes the MAD comparable to a standard deviation for normal data


def _previous_windows(x: np.ndarray, lookback: int) -> np.ndarray:
    """Row k holds x[k : k + lookback], the days before row k + lookback."""
    return sliding_window_view(x[:-1], lookback)


def _rolling_z(x: np.ndarray, lookback: int = LOOKBACK) -> np.ndarray:
    """|x - mean| / std, using only the `lookback` values before each row."""
    out = np.full(len(x), np.nan)
    if len(x) > lookback:
        w = _previous_windows(x, lookback)
        out[lookback:] = np.abs(x[lookback:] - w.mean(axis=1)) / w.std(axis=1, ddof=1)
    return out


def _rolling_robust_z(x: np.ndarray, lookback: int = LOOKBACK) -> np.ndarray:
    """|x - median| / (1.4826 x MAD), using only the `lookback` values before each row."""
    out = np.full(len(x), np.nan)
    if len(x) > lookback:
        w = _previous_windows(x, lookback)
        median = np.median(w, axis=1)
        mad = np.median(np.abs(w - median[:, None]), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[lookback:] = np.abs(x[lookback:] - median) / (MAD_TO_STD * mad)
    return out


def _per_stock(features: pd.DataFrame, column: str, fn) -> pd.Series:
    """Apply fn to each stock's column, in date order, and line results up with the input."""
    out = pd.Series(np.nan, index=features.index)
    for _, stock in features.sort_values("date").groupby("symbol"):
        out.loc[stock.index] = fn(stock[column].to_numpy())
    return out


def zscore_scores(features: pd.DataFrame) -> pd.Series:
    """Detector 1, the original rule: z-score of today's return against the
    mean and standard deviation of the previous 20 returns. Price only."""
    return _per_stock(features, "log_return", _rolling_z)


def robust_scores(features: pd.DataFrame) -> pd.Series:
    """Detector 2: the same idea with median and MAD, which a single past
    outlier cannot drag around. Applied to all three features; the score is
    the worst of the three."""
    per_feature = [_per_stock(features, f, _rolling_robust_z) for f in FEATURES]
    return pd.concat(per_feature, axis=1).max(axis=1, skipna=True)


class IsolationForestDetector:
    """Detector 3: scikit-learn's isolation forest over the three features of
    each day, scaled per stock with training-period statistics.

    An isolation forest builds random trees that split the data at random
    values. Unusual points end up alone after few splits, so a short average
    path means a high anomaly score.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        """Fit on training rows only (split == "train"), whatever else df holds."""
        self.scaler = fit_scaler(features)
        train = apply_scaler(features[features["split"] == "train"], self.scaler)
        self.forest = IsolationForest(n_estimators=200, random_state=self.seed)
        self.forest.fit(train[FEATURES].to_numpy())
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        x = apply_scaler(features, self.scaler)[FEATURES].to_numpy()
        # score_samples is high for normal points, so flip the sign.
        return pd.Series(-self.forest.score_samples(x), index=features.index)
