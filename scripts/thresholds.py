"""Write checkpoints/thresholds.json, the flag thresholds the service uses.

The z-score keeps the original pipeline's rule, |z| > 3. The autoencoder's
threshold is set on the validation period to raise the same number of alarms
as that rule, so on the dashboard both detectors flag about as often and the
comparison is fair (see DECISIONS.md #26).
"""

import json
from pathlib import Path

import pandas as pd

from anomaly.detectors import AutoencoderDetector, zscore_scores
from anomaly.evaluate import alarm_rate, calibrate_threshold
from anomaly.runs import BARS, CHECKPOINT, prepare
from anomaly.train import load_checkpoint

OUT = Path("checkpoints/thresholds.json")
Z_RULE = 3.0

clean = prepare(pd.read_csv(BARS, parse_dates=["date"]))
val = clean[clean["split"] == "val"]

z_scores = zscore_scores(clean)
model_scores = AutoencoderDetector(*load_checkpoint(CHECKPOINT)).score(clean)

z_val = val.assign(score=z_scores)
target = alarm_rate(z_val.assign(flag=z_val["score"] > Z_RULE))

model_val = val.assign(score=model_scores)
model_threshold = calibrate_threshold(model_val, target)
model_rate = alarm_rate(model_val.assign(flag=model_val["score"] > model_threshold))

thresholds = {
    "z-score": {"threshold": Z_RULE, "val_alarms_per_stock_month": round(target, 4)},
    "autoencoder": {
        "threshold": model_threshold,
        "val_alarms_per_stock_month": round(model_rate, 4),
    },
}
OUT.write_text(json.dumps(thresholds, indent=2) + "\n")

# What the service will do on 2025-2026: flagged days per detector at these
# thresholds, computed offline (the service's scores match offline, see
# tests/test_online.py and the integration test). Saved for the README.
test = clean[clean["split"] == "test"]
z_flag = z_scores[test.index] > Z_RULE
m_flag = model_scores[test.index] > model_threshold


def per_stock_month(flags) -> float:
    return round(flags.sum() * 21 / len(test), 2)


pd.DataFrame(
    [
        {"metric": "z-score flagged days", "value": int(z_flag.sum())},
        {"metric": "autoencoder flagged days", "value": int(m_flag.sum())},
        {"metric": "z-score flagged days per stock per month", "value": per_stock_month(z_flag)},
        {
            "metric": "autoencoder flagged days per stock per month",
            "value": per_stock_month(m_flag),
        },
        {"metric": "flagged by both", "value": int((z_flag & m_flag).sum())},
        {"metric": "flagged by z-score only", "value": int((z_flag & ~m_flag).sum())},
        {"metric": "flagged by autoencoder only", "value": int((m_flag & ~z_flag).sum())},
        {"metric": "validation alarms per stock per month (both)", "value": round(target, 4)},
    ]
).to_csv("results/service_flags.csv", index=False)
print(json.dumps(thresholds, indent=2))
