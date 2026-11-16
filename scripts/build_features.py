"""Build features from data/raw/bars.csv into data/processed/features.csv."""

from pathlib import Path

import pandas as pd

from anomaly.features import build_features
from anomaly.split import assign_split

RAW = Path("data/raw/bars.csv")
OUT = Path("data/processed/features.csv")

bars = pd.read_csv(RAW, parse_dates=["date"])
features = build_features(bars)
features["split"] = assign_split(features["date"])

OUT.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUT, index=False)
print(f"Wrote {len(features):,} feature rows to {OUT}")
