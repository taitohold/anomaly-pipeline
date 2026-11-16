"""Fake daily bars for tests and CI, so they never need an API key or real data."""

import numpy as np
import pandas as pd


def make_bars(symbols: list[str], start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Return random-walk daily bars in the same columns as data/raw/bars.csv.

    One row per symbol per weekday from start to end. Prices follow a random
    walk of log returns; volume varies randomly around a per-symbol level.
    The same seed always gives the same bars.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for symbol in symbols:
        log_returns = rng.normal(0.0, 0.015, len(dates))
        close = 100.0 * np.exp(np.cumsum(log_returns))
        open_ = close * np.exp(rng.normal(0.0, 0.005, len(dates)))
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.005, len(dates))))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.005, len(dates))))
        volume = rng.lognormal(np.log(1e7), 0.3, len(dates)).round()
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
