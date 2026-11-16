"""Split rows by time and scale features using the training period only."""

import pandas as pd

from anomaly.features import FEATURES

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2024-12-31")
SPLITS = ["train", "val", "test"]


def assign_split(dates: pd.Series) -> pd.Series:
    """Label each date "train" (to 2022), "val" (2023-2024), or "test" (2025 on)."""
    dates = pd.to_datetime(dates)
    split = pd.Series("test", index=dates.index)
    split[dates <= VAL_END] = "val"
    split[dates <= TRAIN_END] = "train"
    return split


def fit_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """Return the mean and standard deviation of each feature, per stock.

    Only rows whose "split" column is "train" are used, even if df holds
    every split. That is what keeps validation and test data out of scaling.
    The result has one row per symbol and columns like "log_return_mean".
    """
    train = df[df["split"] == "train"]
    stats = train.groupby("symbol")[FEATURES].agg(["mean", "std"])
    stats.columns = [f"{feature}_{stat}" for feature, stat in stats.columns]
    return stats


def apply_scaler(df: pd.DataFrame, scaler: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with each feature standardised by its stock's stats."""
    missing = set(df["symbol"]) - set(scaler.index)
    if missing:
        raise ValueError(f"no training statistics for: {sorted(missing)}")
    out = df.copy()
    stats = scaler.loc[out["symbol"]].reset_index(drop=True)
    for feature in FEATURES:
        mean = stats[f"{feature}_mean"].to_numpy()
        std = stats[f"{feature}_std"].to_numpy()
        out[feature] = (out[feature].to_numpy() - mean) / std
    return out
