"""Train the LSTM autoencoder on the training period, keeping the epoch with the
lowest validation loss. Saves checkpoints/autoencoder.pt and
results/training_history.csv.

Run with: uv run python scripts/train.py [--device cpu|mps|cuda|auto]
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from anomaly.dataset import WindowDataset
from anomaly.device import pick_device
from anomaly.model import LSTMAutoencoder
from anomaly.split import apply_scaler, fit_scaler
from anomaly.train import TrainConfig, save_checkpoint, set_seed, train

CHECKPOINT = Path("checkpoints/autoencoder.pt")
HISTORY = Path("results/training_history.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
args = parser.parse_args()
device = (
    pick_device(torch.cuda.is_available(), torch.backends.mps.is_available())
    if args.device == "auto"
    else args.device
)

features = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
scaler = fit_scaler(features)  # training rows only
scaled = apply_scaler(features, scaler)
train_ds, val_ds = WindowDataset(scaled, "train"), WindowDataset(scaled, "val")
print(f"device {device}, {len(train_ds):,} training windows, {len(val_ds):,} validation windows")

config = TrainConfig()
set_seed(config.seed)
model = LSTMAutoencoder()
started = time.time()
history = train(model, train_ds, val_ds, config, device=device)
elapsed = time.time() - started

CHECKPOINT.parent.mkdir(exist_ok=True)
save_checkpoint(CHECKPOINT, model.cpu(), scaler, config)
HISTORY.parent.mkdir(exist_ok=True)
pd.DataFrame(history).to_csv(HISTORY, index=False)

best = min(history, key=lambda h: h["val_loss"])
print(
    f"\nbest epoch {best['epoch']} of {len(history)}: val loss {best['val_loss']:.5f} "
    f"({elapsed:.0f}s, {elapsed / len(history):.1f}s per epoch)"
)
print(f"saved {CHECKPOINT} and {HISTORY}")
