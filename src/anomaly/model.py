"""The LSTM autoencoder: it learns to rebuild normal 30-day windows, so a window
it rebuilds badly is unusual."""

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Squeeze a (window, features) sequence into a few numbers, then rebuild it.

    encoder LSTM  -> reads the window day by day; its final hidden state
                     summarises the whole window
    to_code       -> squeezes that summary into `bottleneck` numbers
    decoder LSTM  -> is fed the same code on every day and unrolls it back
                     into a sequence
    to_output     -> maps each day's decoder state back to the features

    The bottleneck is the point: 8 numbers cannot hold 90, so the model has to
    learn the patterns common in normal data, and it rebuilds rare windows badly.
    """

    def __init__(self, n_features: int = 3, hidden: int = 32, bottleneck: int = 8):
        super().__init__()
        self.config = {"n_features": n_features, "hidden": hidden, "bottleneck": bottleneck}
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_code = nn.Linear(hidden, bottleneck)
        self.decoder = nn.LSTM(bottleneck, hidden, batch_first=True)
        self.to_output = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, features) -> reconstruction of the same shape."""
        _, (h, _) = self.encoder(x)  # h: (1, batch, hidden), the state after the last day
        code = self.to_code(h[-1])  # (batch, bottleneck)
        repeated = code.unsqueeze(1).repeat(1, x.shape[1], 1)  # (batch, window, bottleneck)
        decoded, _ = self.decoder(repeated)  # (batch, window, hidden)
        return self.to_output(decoded)


@torch.no_grad()
def newest_day_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Anomaly score per window: mean squared reconstruction error on the last
    day only, so a flag points at the day something happened (DECISIONS.md #8)."""
    model.eval()
    error = (model(x) - x)[:, -1, :] ** 2
    return error.mean(dim=1)
