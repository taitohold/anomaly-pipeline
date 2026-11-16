import numpy as np
import pandas as pd
import pytest

from anomaly.features import LOOKBACK, build_features
from anomaly.synthetic import make_bars


def test_synthetic_bars_are_repeatable():
    a = make_bars(["AAA"], "2024-01-01", "2024-03-01", seed=1)
    b = make_bars(["AAA"], "2024-01-01", "2024-03-01", seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_drops_first_lookback_rows_per_stock(bars):
    out = build_features(bars)
    for symbol, n_bars in bars.groupby("symbol").size().items():
        assert (out["symbol"] == symbol).sum() == n_bars - LOOKBACK
    assert not out.isna().any().any()


def test_values_match_a_direct_calculation(bars):
    """Recompute one row's features by hand from the raw bars."""
    out = build_features(bars)
    stock = bars[bars["symbol"] == "AAA"].reset_index(drop=True)
    i = 100  # any row past the first LOOKBACK
    day = stock.loc[i, "date"]
    row = out[(out["symbol"] == "AAA") & (out["date"] == day)].iloc[0]

    returns = np.log(stock["close"].to_numpy()[1:] / stock["close"].to_numpy()[:-1])
    # returns[k] is the return on row k + 1, so row i's last 20 returns end at i - 1
    expected_vol = np.std(returns[i - LOOKBACK : i], ddof=1)
    expected_rel = np.log(stock.loc[i, "volume"] / np.median(stock["volume"][i - LOOKBACK : i]))

    assert row["log_return"] == pytest.approx(returns[i - 1])
    assert row["volatility"] == pytest.approx(expected_vol)
    assert row["rel_volume"] == pytest.approx(expected_rel)


def test_features_never_look_ahead(bars):
    """Changing bars after a date must not change any feature on or before it."""
    cutoff = pd.Timestamp("2023-06-30")
    changed = bars.copy()
    later = changed["date"] > cutoff
    changed.loc[later, "close"] *= 3.0
    changed.loc[later, "volume"] *= 50.0

    before = build_features(bars)
    after = build_features(changed)
    keep = before["date"] <= cutoff
    pd.testing.assert_frame_equal(
        before[keep].reset_index(drop=True), after[after["date"] <= cutoff].reset_index(drop=True)
    )


def test_stocks_do_not_leak_into_each_other(bars):
    """Rolling windows must restart at each stock, not run across them."""
    changed = bars.copy()
    changed.loc[changed["symbol"] == "AAA", "volume"] *= 100.0

    before = build_features(bars)
    after = build_features(changed)
    others = before["symbol"] != "AAA"
    pd.testing.assert_frame_equal(
        before[others].reset_index(drop=True),
        after[after["symbol"] != "AAA"].reset_index(drop=True),
    )


def test_rejects_non_positive_volume(bars):
    bad = bars.copy()
    bad.loc[5, "volume"] = 0
    with pytest.raises(ValueError):
        build_features(bad)
