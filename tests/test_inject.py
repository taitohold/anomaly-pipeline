import itertools

import numpy as np
import pandas as pd
import pytest

from anomaly.inject import MIN_HISTORY, SIZES, SPACING, inject_anomalies
from anomaly.split import VAL_END
from anomaly.synthetic import make_bars


@pytest.fixture
def long_bars():
    """About 23 months of test period per stock, like the real data."""
    return make_bars(["AAA", "BBB", "CCC", "DDD"], "2024-06-01", "2026-11-13", seed=3)


def returns_of(bars, symbol):
    stock = bars[bars["symbol"] == symbol].sort_values("date")
    return pd.Series(np.diff(np.log(stock["close"].to_numpy())), index=stock["date"].iloc[1:])


def volume_of(bars, symbol):
    stock = bars[bars["symbol"] == symbol].sort_values("date")
    return pd.Series(stock["volume"].to_numpy(dtype=float), index=stock["date"])


def test_input_is_not_modified(long_bars):
    before = long_bars.copy()
    inject_anomalies(long_bars, seed=0)
    pd.testing.assert_frame_equal(long_bars, before)


def test_only_test_period_rows_change(long_bars):
    changed, _ = inject_anomalies(long_bars, seed=0)
    original = long_bars.sort_values(["symbol", "date"]).reset_index(drop=True)
    early = original["date"] <= VAL_END
    cols = ["open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(
        changed.loc[early, cols], original.loc[early, cols].astype(float), check_dtype=False
    )
    assert not changed.loc[~early, cols].equals(original.loc[~early, cols].astype(float))


def test_same_seed_same_result_different_seed_different(long_bars):
    a_bars, a_events = inject_anomalies(long_bars, seed=7)
    b_bars, b_events = inject_anomalies(long_bars, seed=7)
    pd.testing.assert_frame_equal(a_bars, b_bars)
    pd.testing.assert_frame_equal(a_events, b_events)
    _, c_events = inject_anomalies(long_bars, seed=8)
    assert not a_events.equals(c_events)


def test_events_are_placed_and_spaced_correctly(long_bars):
    _, events = inject_anomalies(long_bars, seed=0)
    assert len(events) == 4 * 6
    for symbol, stock_events in events.groupby("symbol"):
        days = long_bars[long_bars["symbol"] == symbol].sort_values("date")["date"].tolist()
        first_test = next(i for i, d in enumerate(days) if d > VAL_END)
        starts = sorted(days.index(d) for d in stock_events["start"])
        assert starts[0] >= first_test + MIN_HISTORY
        assert all(b - a >= SPACING for a, b in itertools.pairwise(starts))


def test_kinds_are_balanced_and_sizes_come_from_the_grid(long_bars):
    _, events = inject_anomalies(long_bars, seed=0)
    assert set(events["kind"].value_counts()) == {6}  # 24 events, 4 kinds
    for kind, size in zip(events["kind"], events["size"]):
        assert size in SIZES[kind]


def test_spike_moves_one_close_by_size_sigma(long_bars):
    changed, events = inject_anomalies(long_bars, seed=0, kinds=["spike"])
    for symbol, stock_events in events.groupby("symbol"):
        diff = returns_of(changed, symbol) - returns_of(long_bars, symbol)
        expected_days = set()
        for ev in stock_events.itertuples():
            day_after = diff.index[diff.index.get_loc(ev.start) + 1]
            expected_days |= {ev.start, day_after}
            assert diff[ev.start] == pytest.approx(ev.direction * ev.size * ev.sigma)
            assert diff[day_after] == pytest.approx(-diff[ev.start])
        assert set(diff[diff.abs() > 1e-12].index) == expected_days
        pd.testing.assert_series_equal(volume_of(long_bars, symbol), volume_of(changed, symbol))


def test_level_shift_changes_exactly_one_return(long_bars):
    changed, events = inject_anomalies(long_bars, seed=0, kinds=["level_shift"])
    for symbol, stock_events in events.groupby("symbol"):
        diff = returns_of(changed, symbol) - returns_of(long_bars, symbol)
        moved = diff[diff.abs() > 1e-12]
        assert set(moved.index) == set(stock_events["start"])
        for ev in stock_events.itertuples():
            assert diff[ev.start] == pytest.approx(ev.direction * ev.size * ev.sigma)


def test_vol_burst_scales_returns_only_inside_the_burst(long_bars):
    changed, events = inject_anomalies(long_bars, seed=0, kinds=["vol_burst"])
    for symbol, stock_events in events.groupby("symbol"):
        before, after = returns_of(long_bars, symbol), returns_of(changed, symbol)
        inside = pd.Series(False, index=before.index)
        for ev in stock_events.itertuples():
            window = (before.index >= ev.start) & (before.index <= ev.end)
            assert 5 <= window.sum() <= 10
            np.testing.assert_allclose(after[window], ev.size * before[window])
            inside |= window
        np.testing.assert_allclose(after[~inside], before[~inside], atol=1e-12)


def test_volume_surge_leaves_prices_alone(long_bars):
    changed, events = inject_anomalies(long_bars, seed=0, kinds=["volume_surge"])
    for symbol, stock_events in events.groupby("symbol"):
        np.testing.assert_allclose(returns_of(changed, symbol), returns_of(long_bars, symbol))
        ratio = volume_of(changed, symbol) / volume_of(long_bars, symbol)
        expected = pd.Series(1.0, index=ratio.index)
        for ev in stock_events.itertuples():
            days = (ratio.index >= ev.start) & (ratio.index <= ev.end)
            assert 1 <= days.sum() <= 3
            expected[days] = ev.size
        np.testing.assert_allclose(ratio, expected)


def test_too_short_test_period_raises():
    short = make_bars(["AAA"], "2024-06-01", "2025-03-31", seed=0)
    with pytest.raises(ValueError):
        inject_anomalies(short, seed=0)
