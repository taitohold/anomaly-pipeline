import numpy as np
import pandas as pd
import pytest

from anomaly.evaluate import alarm_rate, calibrate_threshold, find_alarms, match, touches


def frame(flags_by_symbol):
    rows = [
        {"symbol": s, "pos": i, "flag": bool(f)}
        for s, flags in flags_by_symbol.items()
        for i, f in enumerate(flags)
    ]
    return pd.DataFrame(rows)


def test_consecutive_flags_make_one_alarm():
    alarms = find_alarms(frame({"AAA": [0, 1, 1, 0, 1, 0]}))
    assert alarms.to_dict("list") == {"symbol": ["AAA", "AAA"], "a_start": [1, 4], "a_end": [2, 4]}


def test_alarms_do_not_join_across_stocks():
    alarms = find_alarms(frame({"AAA": [0, 0, 1], "BBB": [1, 0, 0]}))
    assert len(alarms) == 2


def test_alarm_rate_is_per_stock_per_21_days():
    # 2 stocks x 21 days, 3 alarms in all -> 1.5 alarms per stock per month
    f = frame({"AAA": [1] + [0] * 19 + [1], "BBB": [0] * 10 + [1] + [0] * 10})
    assert alarm_rate(f) == pytest.approx(1.5)


def test_catch_window_runs_two_days_past_the_event():
    events = pd.DataFrame(
        {"symbol": ["AAA"] * 3, "start_pos": [10, 30, 50], "end_pos": [10, 30, 52]}
    )
    # Alarm 1 starts 2 days after event 1 (caught); alarm 2 starts 3 days after
    # event 2 (not caught); alarm 3 overlaps the start of event 3 (caught).
    alarms = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "BBB"],
            "a_start": [12, 33, 45, 10],
            "a_end": [12, 33, 50, 10],
        }
    )
    ev, al = match(events, alarms)
    assert list(ev["caught"]) == [True, False, True]
    assert list(al["true"]) == [True, False, True, False]  # BBB never matches AAA's events


def test_touches_finds_days_inside_alarms():
    alarms = pd.DataFrame({"symbol": ["AAA", "AAA"], "a_start": [5, 20], "a_end": [7, 20]})
    days = pd.DataFrame({"symbol": ["AAA", "BBB"], "pos": [6, 20]})
    assert list(touches(alarms, days)) == [True, False]


def test_calibrated_threshold_hits_but_does_not_exceed_the_target():
    rng = np.random.default_rng(0)
    n = 21 * 24  # two years of trading days
    f = pd.DataFrame(
        {
            "symbol": np.repeat(["AAA", "BBB", "CCC"], n),
            "pos": np.tile(np.arange(n), 3),
            "score": rng.standard_normal(3 * n),
        }
    )
    threshold = calibrate_threshold(f, target=1.0)
    rate = alarm_rate(f.assign(flag=f["score"] > threshold))
    assert 0.9 <= rate <= 1.0
