"""Sanity check before full training: can the model memorise 4 real windows?

If the loss does not fall to a small fraction of where it started, the model or
the training loop is broken, and a full training run would be wasted.
"""

import time

import pandas as pd
from torch.utils.data import Subset

from anomaly.dataset import WINDOW, WindowDataset
from anomaly.model import LSTMAutoencoder
from anomaly.split import apply_scaler, fit_scaler
from anomaly.train import TrainConfig, set_seed, train

STEPS = 3000

features = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
scaled = apply_scaler(features, fit_scaler(features))
tiny = Subset(WindowDataset(scaled, "train"), range(0, 4 * WINDOW, WINDOW))

started = time.time()
set_seed(0)
config = TrainConfig(epochs=STEPS, batch_size=4, lr=3e-3, patience=STEPS)
history = train(LSTMAutoencoder(), tiny, tiny, config, log=lambda _: None)

first = history[0]["train_loss"]
for step in [1, 100, 500, 1000, 2000, STEPS]:
    loss = history[step - 1]["train_loss"]
    print(f"step {step:5d}  loss {loss:.5f}  ({loss / first:.1%} of start)")
ratio = history[-1]["train_loss"] / first
print(f"\n{'PASS' if ratio < 0.05 else 'FAIL'}: final loss is {ratio:.1%} of the start")
print(f"({time.time() - started:.0f}s)")
