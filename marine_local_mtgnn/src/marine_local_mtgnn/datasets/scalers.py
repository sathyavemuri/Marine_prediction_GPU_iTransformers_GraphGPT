"""Scaling for residuals: standardization fit on training-only."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib

from ..constants import TARGET_NAMES

logger = logging.getLogger(__name__)


class ResidualScaler:
    """Fit and apply scaling to residuals (training-only fit)."""

    def __init__(self):
        """Initialize residual scaler."""
        self.mean = None
        self.std = None
        self.is_fitted = False

    def fit(self, residuals_list: list[np.ndarray]) -> None:
        """
        Fit scaler on training residuals only.

        Parameters
        ----------
        residuals_list : list[np.ndarray]
            List of residual arrays from training windows
        """
        # Stack all residuals
        all_residuals = np.vstack(residuals_list)

        # Compute mean and std per target
        self.mean = np.nanmean(all_residuals, axis=0)
        self.std = np.nanstd(all_residuals, axis=0)

        # Avoid division by zero
        self.std[self.std == 0] = 1.0

        self.is_fitted = True
        logger.info(f"Fitted scaler on {len(residuals_list)} training windows")
        logger.debug(f"Mean: {self.mean}")
        logger.debug(f"Std: {self.std}")

    def transform(self, residuals: np.ndarray) -> np.ndarray:
        """
        Standardize residuals.

        Parameters
        ----------
        residuals : np.ndarray
            Residuals of shape (..., n_targets)

        Returns
        -------
        np.ndarray
            Standardized residuals
        """
        if not self.is_fitted:
            raise ValueError("Must fit scaler before transform")

        return (residuals - self.mean) / self.std

    def inverse_transform(self, scaled_residuals: np.ndarray) -> np.ndarray:
        """
        Reverse standardization.

        Parameters
        ----------
        scaled_residuals : np.ndarray
            Standardized residuals

        Returns
        -------
        np.ndarray
            Original-scale residuals
        """
        if not self.is_fitted:
            raise ValueError("Must fit scaler before inverse_transform")

        return scaled_residuals * self.std + self.mean

    def save(self, output_dir: str | Path = "outputs") -> None:
        """
        Save scaler to joblib file.

        Parameters
        ----------
        output_dir : str | Path
            Output directory
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scaler_path = output_dir / "residual_scaler.joblib"
        joblib.dump(self, scaler_path)
        logger.info(f"Saved scaler: {scaler_path}")

        # Also save as CSV for inspection
        scaler_csv = output_dir / "residual_scaler_stats.csv"
        stats_df = pd.DataFrame({
            "target": TARGET_NAMES,
            "mean": self.mean,
            "std": self.std,
        })
        stats_df.to_csv(scaler_csv, index=False)
        logger.info(f"Saved scaler stats: {scaler_csv}")


def load_scaler(path: str | Path) -> ResidualScaler:
    """
    Load scaler from joblib file.

    Parameters
    ----------
    path : str | Path
        Path to scaler file

    Returns
    -------
    ResidualScaler
        Loaded scaler
    """
    return joblib.load(path)
