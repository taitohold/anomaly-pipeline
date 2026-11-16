import numpy as np
import pandas as pd
import pytest

from anomaly.detectors import (
    LOOKBACK,
    MAD_TO_STD,
    IsolationForestDetector,
    robust_scores,
    zscore_scores,
)
from anomaly.features import FEATURES


def one_stock(features, symbol="AAA"):
    return features[features["symbol"] == symbol].sort_values("date")


def test_zscore_matches_a_hand_calculation(features):
    scores = zscore_scores(features)
    stock = one_stock(features)
    i = 200
    past = stock["log_return"].to_numpy()[i - LOOKBACK : i]
    today = stock["log_return"].to_numpy()[i]
    expected = abs(today - past.mean()) / past.std(ddof=1)
    assert scores[stock.index[i]] == pytest.approx(expected)


def test_robust_score_is_the_worst_feature(features):
    scores = robust_scores(features)
    stock = one_stock(features)
    i = 200
    per_feature = []
    for f in FEATURES:
        past = stock[f].to_numpy()[i - LOOKBACK : i]
        median = np.median(past)
        mad = np.median(np.abs(past - median))
        per_feature.append(abs(stock[f].to_numpy()[i] - median) / (MAD_TO_STD * mad))
    assert scores[stock.index[i]] == pytest.approx(max(per_feature))


def test_first_lookback_rows_have_no_score(features):
    for scores in (zscore_scores(features), robust_scores(features)):
        stock = one_stock(features)
        assert scores[stock.index[:LOOKBACK]].isna().all()
        assert scores[stock.index[LOOKBACK:]].notna().all()


@pytest.mark.parametrize("detector", [zscore_scores, robust_scores])
def test_scores_never_look_ahead(features, detector):
    cutoff = pd.Timestamp("2023-06-30")
    changed = features.copy()
    changed.loc[changed["date"] > cutoff, FEATURES] *= 10.0
    keep = features["date"] <= cutoff
    pd.testing.assert_series_equal(detector(features)[keep], detector(changed)[keep])


def test_isolation_forest_ignores_validation_and_test_when_fitting(features):
    garbage = features.copy()
    garbage.loc[garbage["split"] != "train", FEATURES] = 1e3
    a = IsolationForestDetector(seed=0).fit(features).score(features)
    b = IsolationForestDetector(seed=0).fit(garbage).score(features)
    pd.testing.assert_series_equal(a, b)


def test_isolation_forest_scores_an_extreme_day_higher(features):
    detector = IsolationForestDetector(seed=0).fit(features)
    row = features.iloc[[300]].copy()
    extreme = row.copy()
    extreme["log_return"] = row["log_return"] + 0.30
    extreme["rel_volume"] = row["rel_volume"] + 3.0
    assert detector.score(extreme).iloc[0] > detector.score(row).iloc[0]
