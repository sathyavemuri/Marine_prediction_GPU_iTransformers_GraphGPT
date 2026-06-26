"""PyTorch Dataset for windowed residual data."""

import torch
from torch.utils.data import Dataset
import numpy as np


class ResidualWindowDataset(Dataset):
    """PyTorch Dataset wrapping residual windows."""

    def __init__(
        self,
        windows: list[dict],
        scaler=None,
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        windows : list[dict]
            List of window dictionaries with keys: 'history', 'baseline_forecast', 'residuals'
        scaler : ResidualScaler, optional
            Scaler for normalizing residuals
        """
        self.windows = windows
        self.scaler = scaler

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        dict
            Dictionary with 'history', 'targets', 'baseline' tensors
        """
        window = self.windows[idx]

        # Convert to tensors
        history = torch.FloatTensor(window["history"])  # (lookback, nodes)
        baseline = torch.FloatTensor(window["baseline_forecast"])  # (horizon, nodes)
        residuals = torch.FloatTensor(window["residuals"])  # (horizon, targets)

        # Normalize residuals if scaler provided
        if self.scaler is not None:
            residuals_scaled = self.scaler.transform(residuals.numpy())
            residuals = torch.FloatTensor(residuals_scaled)

        return {
            "history": history,
            "baseline": baseline,
            "targets": residuals,  # Normalized residuals
        }
