"""Print rows and windows per split from data/processed/features.csv,
and save them to results/split_counts.csv."""

import pandas as pd

from anomaly.dataset import WINDOW, WindowDataset
from anomaly.split import SPLITS

df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])

rows = []
print(f"{'split':<6} {'first day':<11} {'last day':<11} {'rows':>8} {'windows':>8}")
for split in SPLITS:
    part = df[df["split"] == split]
    windows = len(WindowDataset(df, split))
    first, last = part["date"].min().date(), part["date"].max().date()
    print(f"{split:<6} {first!s:<11} {last!s:<11} {len(part):>8,} {windows:>8,}")
    rows.append(
        {
            "split": split,
            "first_day": first,
            "last_day": last,
            "rows": len(part),
            "windows": windows,
        }
    )
print(f"{'total':<30} {len(df):>8,}")
print(f"\nStocks: {df['symbol'].nunique()}, window: {WINDOW} days")
pd.DataFrame(rows).assign(stocks=df["symbol"].nunique()).to_csv(
    "results/split_counts.csv", index=False
)
