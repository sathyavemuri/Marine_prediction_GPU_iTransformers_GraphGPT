"""Visualization of results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ResultsPlotter:
    """Generate diagnostic plots for evaluation results."""

    def __init__(self, output_dir: str | Path = "outputs"):
        """
        Initialize plotter.

        Parameters
        ----------
        output_dir : str | Path
            Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_training_history(
        self,
        train_losses: list[float],
        val_losses: list[float],
    ) -> None:
        """
        Plot training and validation loss over epochs.

        Parameters
        ----------
        train_losses : list[float]
            Training losses per epoch
        val_losses : list[float]
            Validation losses per epoch
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Training History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "training_history.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved plot: {path}")

    def plot_error_by_lead_time(
        self,
        per_horizon_metrics: list[dict],
    ) -> None:
        """
        Plot error metrics by forecast lead time.

        Parameters
        ----------
        per_horizon_metrics : list[dict]
            Per-horizon metrics from evaluation
        """
        df = pd.DataFrame(per_horizon_metrics)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df["lead_time_steps"], df["mae"], "b-o", label="MAE", linewidth=2)
        ax.set_xlabel("Lead Time (15-minute steps)")
        ax.set_ylabel("MAE")
        ax.set_title("Forecast Error by Lead Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "error_by_lead_time.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved plot: {path}")

    def plot_error_by_target(
        self,
        per_target_metrics: list[dict],
    ) -> None:
        """
        Plot error metrics by target variable.

        Parameters
        ----------
        per_target_metrics : list[dict]
            Per-target metrics from evaluation
        """
        df = pd.DataFrame(per_target_metrics)

        fig, ax = plt.subplots(figsize=(14, 6))

        x_pos = np.arange(len(df))
        ax.bar(x_pos, df["mae"], alpha=0.7, label="MAE")

        ax.set_xlabel("Target")
        ax.set_ylabel("MAE")
        ax.set_title("Forecast Error by Target")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["target"], rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = self.output_dir / "error_by_target.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved plot: {path}")

    def plot_sample_forecast(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        sample_idx: int = 0,
        target_idx: int = 0,
        target_name: str = "Target",
    ) -> None:
        """
        Plot a sample forecast vs actual.

        Parameters
        ----------
        predictions : np.ndarray
            Predictions (num_samples, horizon, targets)
        actuals : np.ndarray
            Actual values (num_samples, horizon, targets)
        sample_idx : int
            Sample index to plot
        target_idx : int
            Target index to plot
        target_name : str
            Name of target
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        horizon = predictions.shape[1]
        lead_times = np.arange(1, horizon + 1)

        pred_sample = predictions[sample_idx, :, target_idx]
        actual_sample = actuals[sample_idx, :, target_idx]

        ax.plot(lead_times, actual_sample, "b-o", label="Actual", linewidth=2)
        ax.plot(lead_times, pred_sample, "r--s", label="Forecast", linewidth=2)

        ax.set_xlabel("Lead Time (15-minute steps)")
        ax.set_ylabel("Value")
        ax.set_title(f"Sample Forecast: {target_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / f"sample_forecast_target_{target_idx}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved plot: {path}")

    def plot_residuals_distribution(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> None:
        """
        Plot distribution of residuals.

        Parameters
        ----------
        predictions : np.ndarray
            Predictions
        actuals : np.ndarray
            Actual values
        """
        residuals = predictions - actuals
        residuals_flat = residuals.flatten()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(residuals_flat, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(residuals_flat), color="r", linestyle="--", linewidth=2, label=f"Mean: {np.mean(residuals_flat):.4f}")
        ax.axvline(np.median(residuals_flat), color="g", linestyle="--", linewidth=2, label=f"Median: {np.median(residuals_flat):.4f}")

        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Residuals")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = self.output_dir / "residuals_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved plot: {path}")
