"""Turn detector scores into alarms and score them against planted events.

Days are counted in trading days per stock ("positions"), not calendar days.
"""

import numpy as np
import pandas as pd

CATCH_AFTER = 2  # an alarm up to 2 trading days after an event's last day still catches it
DAYS_PER_MONTH = 21  # trading days


def with_positions(features: pd.DataFrame) -> pd.DataFrame:
    """Sorted copy with "pos": each row's trading-day number within its stock."""
    out = features.sort_values(["symbol", "date"]).copy()
    out["pos"] = out.groupby("symbol").cumcount()
    return out


def find_alarms(frame: pd.DataFrame) -> pd.DataFrame:
    """Group flagged days into alarms: a run of consecutive flagged days in one
    stock is one alarm. `frame` has symbol, pos, flag, sorted by symbol and pos,
    with no gaps in pos. Returns symbol, a_start, a_end (positions)."""
    flag = frame["flag"].to_numpy(dtype=bool)
    symbol = frame["symbol"].to_numpy()
    pos = frame["pos"].to_numpy()
    same_as_prev = np.r_[False, symbol[1:] == symbol[:-1]]
    same_as_next = np.r_[symbol[:-1] == symbol[1:], False]
    prev_flag = np.r_[False, flag[:-1]] & same_as_prev
    next_flag = np.r_[flag[1:], False] & same_as_next
    starts = flag & ~prev_flag
    ends = flag & ~next_flag
    return pd.DataFrame({"symbol": symbol[starts], "a_start": pos[starts], "a_end": pos[ends]})


def alarm_rate(frame: pd.DataFrame) -> float:
    """Alarms per stock per month (21 trading days) in `frame`."""
    return len(find_alarms(frame)) * DAYS_PER_MONTH / len(frame)


def calibrate_threshold(frame: pd.DataFrame, target: float) -> float:
    """Lowest threshold whose alarm rate (flag = score > threshold) is at most
    `target`. `frame` has symbol, pos, score. Missing scores never flag.

    Found by binary search over the observed scores. More alarms at a lower
    threshold is the usual pattern, though raising a threshold can split one
    alarm into two, so the result is checked rather than assumed."""
    candidates = np.unique(frame["score"].dropna().to_numpy())
    candidates = candidates[np.isfinite(candidates)]

    def rate(threshold: float) -> float:
        return alarm_rate(frame.assign(flag=frame["score"] > threshold))

    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if rate(candidates[mid]) <= target:
            hi = mid
        else:
            lo = mid + 1
    threshold = float(candidates[lo])
    if rate(threshold) > target:
        raise RuntimeError("could not reach the target alarm rate")
    return threshold


def to_positions(events: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Add start_pos and end_pos to events, from their start and end dates."""
    lookup = positions.set_index(["symbol", "date"])["pos"]
    out = events.copy()
    out["start_pos"] = lookup.loc[list(zip(out["symbol"], out["start"]))].to_numpy()
    out["end_pos"] = lookup.loc[list(zip(out["symbol"], out["end"]))].to_numpy()
    return out


def match(events: pd.DataFrame, alarms: pd.DataFrame, catch_after: int = CATCH_AFTER):
    """Mark which events were caught and which alarms were true.

    An event is caught if any alarm in the same stock overlaps the days from
    its start to `catch_after` days after its end. An alarm is true if it
    catches at least one event; otherwise it is a false alarm.
    Returns (events with "caught", alarms with "true").
    """
    ev = events.reset_index(drop=True).rename_axis("ev_id").reset_index()
    al = alarms.reset_index(drop=True).rename_axis("al_id").reset_index()
    pairs = ev[["ev_id", "symbol", "start_pos", "end_pos"]].merge(
        al[["al_id", "symbol", "a_start", "a_end"]], on="symbol"
    )
    hit = pairs[
        (pairs["a_start"] <= pairs["end_pos"] + catch_after)
        & (pairs["a_end"] >= pairs["start_pos"])
    ]
    ev["caught"] = ev["ev_id"].isin(hit["ev_id"])
    al["true"] = al["al_id"].isin(hit["al_id"])
    return ev.drop(columns="ev_id"), al.drop(columns="al_id")


def touches(alarms: pd.DataFrame, days: pd.DataFrame) -> pd.Series:
    """For each alarm, whether any of `days` (symbol, pos) falls inside it."""
    al = alarms.reset_index(drop=True).rename_axis("al_id").reset_index()
    pairs = al.merge(days[["symbol", "pos"]], on="symbol")
    inside = pairs[(pairs["pos"] >= pairs["a_start"]) & (pairs["pos"] <= pairs["a_end"])]
    return al["al_id"].isin(inside["al_id"]).set_axis(alarms.index)
