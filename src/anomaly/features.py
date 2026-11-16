"""Turn raw daily bars into the three features every detector uses."""

import numpy as np
import pandas as pd

LOOKBACK = 20
FEATURES = ["log_return", "volatility", "rel_volume"]


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute features per stock, using only each row's own past.

    - log_return: log of today's close over yesterday's close.
    - volatility: standard deviation of the last 20 log returns, today included.
    - rel_volume: log of today's volume over the median volume of the 20 days
      before today. Today is left out so a volume surge cannot hide itself.

    The first 20 rows of each stock cannot have full features and are dropped.
    Returns columns: symbol, date, log_return, volatility, rel_volume.
    """
    if (bars["close"] <= 0).any() or (bars["volume"] <= 0).any():
        raise ValueError("close and volume must be positive to take logs")

    df = bars.sort_values(["symbol", "date"]).reset_index(drop=True)
    by_symbol = df.groupby("symbol")

    df["log_return"] = by_symbol["close"].transform(lambda c: np.log(c).diff())
    df["volatility"] = df.groupby("symbol")["log_return"].transform(
        lambda r: r.rolling(LOOKBACK).std()
    )
    past_median = by_symbol["volume"].transform(lambda v: v.shift(1).rolling(LOOKBACK).median())
    df["rel_volume"] = np.log(df["volume"] / past_median)

    out = df[["symbol", "date", *FEATURES]].dropna()
    return out.reset_index(drop=True)
