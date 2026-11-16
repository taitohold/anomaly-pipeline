import numpy as np
import pandas as pd

from anomaly.analysis import agreement, alarm_peaks, overlap_counts, reliable_size
from anomaly.evaluate import alarm_rate, calibrate_threshold, find_alarms, match

DETECTORS = ["a", "b", "c"]


def caught_rows(pattern_by_event):
    """{event_id: (kind, {detector: caught})} -> long table."""
    return pd.DataFrame(
        [
            {"event_id": e, "kind": kind, "detector": d, "caught": c}
            for e, (kind, flags) in pattern_by_event.items()
            for d, c in flags.items()
        ]
    )


def test_overlap_puts_each_event_in_exactly_one_column():
    caught = caught_rows(
        {
            1: ("spike", {"a": True, "b": True, "c": True}),  # all
            2: ("spike", {"a": False, "b": False, "c": False}),  # none
            3: ("spike", {"a": False, "b": True, "c": False}),  # only b
            4: ("spike", {"a": True, "b": True, "c": False}),  # some
            5: ("volume_surge", {"a": False, "b": False, "c": True}),  # only c
        }
    )
    table = overlap_counts(caught, DETECTORS)
    spike = table.loc["spike"]
    assert spike["events"] == 4
    assert (spike["all"], spike["none"], spike["only b"], spike["some"]) == (1, 1, 1, 1)
    assert table.loc["volume_surge", "only c"] == 1
    assert (table.drop(columns="events").sum(axis=1) == table["events"]).all()


def test_reliable_size_needs_every_larger_size_to_pass():
    recall = pd.DataFrame(
        {
            "kind": ["spike"] * 4 + ["volume_surge"] * 2,
            "size": [2, 3, 4, 6, 3, 10],
            "caught": [50, 95, 85, 100, 10, 80],
            "events": [100] * 6,
        }
    )
    # spike: 6 passes, 4 fails (85%), so 3 does not count even though it passes
    assert reliable_size(recall) == {"spike": 6, "volume_surge": None}


def test_agreement_allows_one_day_of_slack():
    mine = pd.DataFrame(
        {"symbol": ["AAA", "AAA", "AAA"], "a_start": [10, 20, 30], "a_end": [10, 20, 30]}
    )
    theirs = pd.DataFrame(
        {"symbol": ["AAA", "AAA", "BBB"], "a_start": [11, 22, 30], "a_end": [11, 22, 30]}
    )
    assert list(agreement(mine, theirs)) == [True, False, False]


def test_alarm_peaks_finds_the_highest_day():
    scored = pd.DataFrame({"symbol": "AAA", "pos": range(6), "score": [0, 5, 9, 7, 0, 3]})
    alarms = pd.DataFrame({"symbol": ["AAA", "AAA"], "a_start": [1, 5], "a_end": [3, 5]})
    peaks = alarm_peaks(scored, alarms)
    assert list(peaks["peak_pos"]) == [2, 5]
    assert list(peaks["peak_score"]) == [9, 3]


def test_equal_rate_thresholds_give_equal_alarm_rates():
    """Two detectors with very different score scales end up with the same rate."""
    rng = np.random.default_rng(1)
    n = 21 * 24
    base = pd.DataFrame({"symbol": np.repeat(["AAA", "BBB"], n), "pos": np.tile(np.arange(n), 2)})
    small = base.assign(score=rng.standard_normal(2 * n))
    large = base.assign(score=100 * rng.exponential(size=2 * n))
    rates = []
    for f in (small, large):
        t = calibrate_threshold(f, target=0.5)
        rates.append(alarm_rate(f.assign(flag=f["score"] > t)))
    assert abs(rates[0] - rates[1]) <= 0.1
    assert max(rates) <= 0.5


def test_chance_floor_extremes():
    events = pd.DataFrame({"symbol": ["AAA"] * 2, "start_pos": [10, 40], "end_pos": [10, 41]})
    days = pd.DataFrame({"symbol": "AAA", "pos": range(60)})
    nothing = find_alarms(days.assign(flag=False))
    everything = find_alarms(days.assign(flag=True))
    assert not match(events, nothing)[0]["caught"].any()
    assert match(events, everything)[0]["caught"].all()
