"""Setup shared by scripts/compare.py and scripts/error_analysis.py."""

from pathlib import Path

import pandas as pd

from anomaly.detectors import (
    AutoencoderDetector,
    IsolationForestDetector,
    robust_scores,
    zscore_scores,
)
from anomaly.evaluate import with_positions
from anomaly.features import build_features
from anomaly.split import assign_split
from anomaly.train import load_checkpoint

BARS = Path("data/raw/bars.csv")
EARNINGS = Path("data/raw/earnings.csv")
CHECKPOINT = Path("checkpoints/autoencoder.pt")
PASSES = 10  # injection passes, seeds 0 to 9


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw bars -> features with a split column and trading-day positions."""
    features = build_features(raw)
    features["split"] = assign_split(features["date"])
    return with_positions(features)


def build_detectors(clean: pd.DataFrame) -> dict:
    """The four detectors, each a function from a features frame to scores.
    The isolation forest is fitted on training rows; the autoencoder is loaded
    from its checkpoint."""
    forest = IsolationForestDetector(seed=0).fit(clean)
    autoencoder = AutoencoderDetector(*load_checkpoint(CHECKPOINT))
    return {
        "z-score": zscore_scores,
        "robust": robust_scores,
        "isolation forest": forest.score,
        "autoencoder": autoencoder.score,
    }
