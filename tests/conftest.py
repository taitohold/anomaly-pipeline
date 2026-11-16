import pytest

from anomaly.features import build_features
from anomaly.split import assign_split
from anomaly.synthetic import make_bars

SYMBOLS = ["AAA", "BBB", "CCC"]


@pytest.fixture
def bars():
    """Fake bars that span all three splits: 2022-06 to 2025-06."""
    return make_bars(SYMBOLS, "2022-06-01", "2025-06-30", seed=0)


@pytest.fixture
def features(bars):
    """Features with a split column, as written to data/processed/."""
    df = build_features(bars)
    df["split"] = assign_split(df["date"])
    return df
