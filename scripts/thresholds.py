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

z_val = val.assign(score=zscore_scores(clean))
target = alarm_rate(z_val.assign(flag=z_val["score"] > Z_RULE))

model_val = val.assign(score=AutoencoderDetector(*load_checkpoint(CHECKPOINT)).score(clean))
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
print(json.dumps(thresholds, indent=2))
