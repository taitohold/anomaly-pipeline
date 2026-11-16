"""Training loop for the autoencoder, plus saving and loading checkpoints."""

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from anomaly.model import LSTMAutoencoder


@dataclass
class TrainConfig:
    epochs: int = 150  # 50 was not enough: validation loss was still falling (DECISIONS.md #23)
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 5  # stop after this many epochs without a better validation loss
    seed: int = 0


def set_seed(seed: int) -> None:
    """Seed every random number generator training touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def mean_loss(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Average reconstruction loss (MSE over the whole window) on a dataset."""
    model.eval()
    total, count = 0.0, 0
    for x in loader:
        x = x.to(device)
        total += nn.functional.mse_loss(model(x), x, reduction="sum").item()
        count += x.numel()
    return total / count


def train(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    config: TrainConfig,
    device: str = "cpu",
    log=print,
) -> list[dict]:
    """Train with Adam on reconstruction loss, checking validation after every epoch.

    Keeps the weights from the epoch with the lowest validation loss and loads
    them back into `model` before returning. Stops early once validation loss
    has not improved for `config.patience` epochs. Returns one dict per epoch.

    For a fully repeatable run, call set_seed(config.seed) before building the
    model as well: building it draws the random starting weights.
    """
    set_seed(config.seed)
    model.to(device)
    # A seeded generator fixes the shuffle order, so a rerun sees the same batches.
    shuffle = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, generator=shuffle
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_loss, best_state, since_best = float("inf"), None, 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), x)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.numel()
            count += x.numel()

        val_loss = mean_loss(model, val_loader, device)
        improved = val_loss < best_loss
        history.append(
            {"epoch": epoch, "train_loss": total / count, "val_loss": val_loss, "best": improved}
        )
        log(
            f"epoch {epoch:3d}  train {total / count:.5f}  val {val_loss:.5f}"
            + ("  *best" if improved else "")
        )
        if improved:
            best_loss, best_state, since_best = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= config.patience:
                log(f"stopping: no improvement for {config.patience} epochs")
                break

    model.load_state_dict(best_state)
    return history


def save_checkpoint(path, model: LSTMAutoencoder, scaler: pd.DataFrame, config: TrainConfig):
    """Save everything needed to score new data: weights, model settings,
    training settings, and the training-period scaling statistics."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model.config,
            "train_config": asdict(config),
            "scaler": {
                "symbols": list(scaler.index),
                "columns": list(scaler.columns),
                "values": scaler.to_numpy().tolist(),
            },
        },
        path,
    )


def load_checkpoint(path) -> tuple[LSTMAutoencoder, pd.DataFrame]:
    """Return (model with saved weights, scaler) from a checkpoint file."""
    saved = torch.load(path, weights_only=True)
    model = LSTMAutoencoder(**saved["model_config"])
    model.load_state_dict(saved["state_dict"])
    s = saved["scaler"]
    scaler = pd.DataFrame(s["values"], index=s["symbols"], columns=s["columns"])
    scaler.index.name = "symbol"
    return model, scaler
