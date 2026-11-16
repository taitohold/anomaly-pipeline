"""Plant labelled anomalies into the test period of raw daily bars.

Real anomalies have no labels, so detectors are scored on planted ones.
Injection works on raw prices and volume, before features are built, so the
rolling features react to a planted event the way they would to a real one.
"""

import numpy as np
import pandas as pd

from anomaly.split import VAL_END

KINDS = ["spike", "level_shift", "vol_burst", "volume_surge"]

# Spike and level shift sizes are in units of the stock's recent daily
# volatility (sigma). Burst sizes multiply returns; surge sizes multiply volume.
SIZES = {
    "spike": [2, 3, 4, 6, 8],
    "level_shift": [2, 3, 4, 6, 8],
    "vol_burst": [2, 3, 5],
    "volume_surge": [3, 5, 10],
}

EVENTS_PER_STOCK = 6
MIN_HISTORY = 30  # test days before the first event, so every detector has full history
SPACING = 50  # minimum days between two event starts in one stock
TAIL = 12  # room after the last start: a 10-day burst plus 2 days to catch it
SIGMA_LOOKBACK = 20


def inject_anomalies(
    bars: pd.DataFrame,
    seed: int,
    events_per_stock: int = EVENTS_PER_STOCK,
    kinds: list[str] = KINDS,
    after: pd.Timestamp = VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (changed copy of bars, table of planted events).

    Only rows dated after `after` (the test period) are changed; the input is
    left untouched. The four kinds:

    - spike:        one day's close moves by size x sigma, up or down. Only that
                    day changes, so the next day's return reverses the move.
    - level_shift:  every close from the start day on moves by size x sigma.
    - vol_burst:    daily returns are multiplied by size for 5 to 10 days.
    - volume_surge: volume is multiplied by size for 1 to 3 days; prices are
                    untouched.

    Open, high and low move by the same ratio as the close. Kinds are spread
    evenly across all events; sizes and days are drawn at random from the seed.

    The events table has one row per event: symbol, kind, size, direction
    (+1/-1 for spike and level shift, 0 otherwise), sigma, start, end.
    """
    rng = np.random.default_rng(seed)
    out = bars.sort_values(["symbol", "date"]).reset_index(drop=True)
    out["volume"] = out["volume"].astype(float)

    groups = out.groupby("symbol").indices
    kind_plan = list(np.resize(kinds, len(groups) * events_per_stock))
    rng.shuffle(kind_plan)

    events = []
    for symbol, rows in groups.items():
        dates = out["date"].to_numpy()[rows]
        close = out["close"].to_numpy()[rows]
        volume = out["volume"].to_numpy()[rows].copy()
        log_price = np.log(close)
        original_returns = np.diff(log_price)  # original_returns[j] is the return on row j + 1

        lo = np.searchsorted(dates, np.datetime64(after), side="right") + MIN_HISTORY
        span = len(rows) - TAIL - lo + 1
        slot = span // events_per_stock
        if slot < SPACING:
            raise ValueError(f"{symbol}: test period too short for {events_per_stock} events")

        for k in range(events_per_stock):
            t = lo + k * slot + int(rng.integers(0, slot - SPACING + 1))
            kind = kind_plan.pop()
            size = rng.choice(SIZES[kind]).item()
            sigma = np.std(original_returns[t - 1 - SIGMA_LOOKBACK : t - 1], ddof=1)
            direction = 0
            end = t

            if kind == "spike":
                direction = int(rng.choice([-1, 1]))
                log_price[t] += direction * size * sigma
            elif kind == "level_shift":
                direction = int(rng.choice([-1, 1]))
                log_price[t:] += direction * size * sigma
            elif kind == "vol_burst":
                length = int(rng.integers(5, 11))
                returns = log_price[t : t + length] - log_price[t - 1 : t + length - 1]
                extra = np.cumsum((size - 1) * returns)
                log_price[t : t + length] += extra
                log_price[t + length :] += extra[-1]
                end = t + length - 1
            else:  # volume_surge
                length = int(rng.integers(1, 4))
                volume[t : t + length] *= size
                end = t + length - 1

            events.append(
                {
                    "symbol": symbol,
                    "kind": kind,
                    "size": size,
                    "direction": direction,
                    "sigma": sigma,
                    "start": pd.Timestamp(dates[t]),
                    "end": pd.Timestamp(dates[end]),
                }
            )

        ratio = np.exp(log_price) / close
        for column in ["open", "high", "low", "close"]:
            out.loc[rows, column] = out[column].to_numpy()[rows] * ratio
        out.loc[rows, "volume"] = volume

    return out, pd.DataFrame(events)
