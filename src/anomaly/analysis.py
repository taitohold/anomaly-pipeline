"""Error analysis: who catches what, what false alarms look like, and which
real days the model flags."""

import pandas as pd

ONLY = "only {}"


def overlap_counts(caught: pd.DataFrame, detectors: list[str]) -> pd.DataFrame:
    """Per kind, how many events each combination of detectors caught.

    `caught` has one row per (event_id, detector) with columns event_id, kind,
    detector, caught. Columns of the result: events, all, none, one "only X"
    per detector, and "some" (caught by more than one but not all). Each
    event lands in exactly one column, so they add up to `events`.
    """
    wide = caught.pivot_table(
        index=["event_id", "kind"], columns="detector", values="caught", aggfunc="first"
    )[detectors].astype(bool)
    n_caught = wide.sum(axis=1)
    category = pd.Series("some", index=wide.index)
    category[n_caught == len(detectors)] = "all"
    category[n_caught == 0] = "none"
    for d in detectors:
        category[(n_caught == 1) & wide[d]] = ONLY.format(d)
    columns = ["all", *[ONLY.format(d) for d in detectors], "some", "none"]
    table = (
        category.rename("category")
        .reset_index()
        .groupby(["kind", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=columns, fill_value=0)
    )
    table.insert(0, "events", table.sum(axis=1))
    return table


def reliable_size(recall: pd.DataFrame, level: float = 0.9) -> dict:
    """Smallest size from which a detector catches at least `level` of events
    at that size and every larger one. `recall` has kind, size, caught, events
    for one detector. None if even the largest size falls short."""
    out = {}
    for kind, rows in recall.groupby("kind"):
        rows = rows.sort_values("size", ascending=False)
        smallest = None
        for row in rows.itertuples():
            if row.caught / row.events < level:
                break
            smallest = row.size
        out[kind] = smallest
    return out


def agreement(alarms: pd.DataFrame, others: pd.DataFrame, slack: int = 1) -> pd.Series:
    """For each alarm, whether `others` has an alarm in the same stock within
    `slack` trading days of it. Both have symbol, a_start, a_end."""
    a = alarms.reset_index(drop=True).rename_axis("al_id").reset_index()
    pairs = a.merge(others[["symbol", "a_start", "a_end"]], on="symbol", suffixes=("", "_o"))
    near = pairs[
        (pairs["a_start_o"] <= pairs["a_end"] + slack)
        & (pairs["a_end_o"] >= pairs["a_start"] - slack)
    ]
    return a["al_id"].isin(near["al_id"]).set_axis(alarms.index)


def alarm_peaks(scored: pd.DataFrame, alarms: pd.DataFrame) -> pd.DataFrame:
    """Add each alarm's highest-scoring day: peak_pos and peak_score.
    `scored` has symbol, pos, score."""
    lookup = scored.set_index(["symbol", "pos"])["score"]
    peak_pos, peak_score = [], []
    for a in alarms.itertuples():
        scores = lookup.loc[[(a.symbol, p) for p in range(a.a_start, a.a_end + 1)]]
        best = scores.idxmax()
        peak_pos.append(best[1])
        peak_score.append(scores.max())
    return alarms.assign(peak_pos=peak_pos, peak_score=peak_score)
