"""Evaluation metrics for MTGNN forecasts."""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    target_names: list[str],
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    predictions : np.ndarray
        Predictions of shape (num_samples, horizon, num_targets)
    actuals : np.ndarray
        Actual values of shape (num_samples, horizon, num_targets)
    target_names : list[str]
        Names of targets

    Returns
    -------
    dict
        Dictionary with 'overall', 'by_target', 'by_horizon' metrics
    """
    assert predictions.shape == actuals.shape

    num_samples, horizon, num_targets = predictions.shape

    # Overall metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Skill score relative to persistence baseline
    persistence_forecast = np.repeat(actuals[:, :1, :], horizon, axis=1)
    persistence_mse = np.mean((persistence_forecast - actuals) ** 2)
    skill = 1.0 - (mse / persistence_mse) if persistence_mse > 0 else 0.0

    # Per-target metrics
    per_target_mae = np.mean(np.abs(predictions - actuals), axis=(0, 1))
    per_target_rmse = np.sqrt(np.mean((predictions - actuals) ** 2, axis=(0, 1)))

    per_target_metrics = []
    for i, name in enumerate(target_names):
        per_target_metrics.append({
            "target": name,
            "mae": float(per_target_mae[i]),
            "rmse": float(per_target_rmse[i]),
        })

    # Per-horizon (lead time) metrics
    per_horizon_mae = np.mean(np.abs(predictions - actuals), axis=(0, 2))
    per_horizon_rmse = np.sqrt(np.mean((predictions - actuals) ** 2, axis=(0, 2)))

    per_horizon_metrics = []
    for i in range(horizon):
        per_horizon_metrics.append({
            "lead_time_steps": i + 1,
            "mae": float(per_horizon_mae[i]),
            "rmse": float(per_horizon_rmse[i]),
        })

    return {
        "overall": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "skill_vs_persistence": float(skill),
        },
        "by_target": per_target_metrics,
        "by_horizon": per_horizon_metrics,
    }


def compute_physical_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    target_names: list[str],
    scaler=None,
) -> dict:
    """
    Compute metrics on physical (un-scaled) values.

    Parameters
    ----------
    predictions : np.ndarray
        Predictions (may be scaled)
    actuals : np.ndarray
        Actuals (may be scaled)
    target_names : list[str]
        Target names
    scaler : ResidualScaler, optional
        Scaler to inverse-transform if provided

    Returns
    -------
    dict
        Physical metrics
    """
    # Inverse transform if scaler provided
    if scaler is not None:
        # Reshape for scaler
        orig_shape = predictions.shape
        pred_flat = predictions.reshape(-1, orig_shape[-1])
        actual_flat = actuals.reshape(-1, actuals.shape[-1])

        predictions = scaler.inverse_transform(pred_flat).reshape(orig_shape)
        actuals = scaler.inverse_transform(actual_flat).reshape(orig_shape)

    # Compute metrics on physical scale
    return compute_metrics(predictions, actuals, target_names)


def skill_score(
    predictions: np.ndarray,
    actuals: np.ndarray,
    baseline: np.ndarray | None = None,
) -> float:
    """
    Compute skill score (1.0 - normalized_error).

    Parameters
    ----------
    predictions : np.ndarray
        Predictions
    actuals : np.ndarray
        Actual values
    baseline : np.ndarray, optional
        Baseline forecast (default: persistence)

    Returns
    -------
    float
        Skill score in [0, 1]
    """
    if baseline is None:
        # Persistence baseline
        horizon = predictions.shape[1]
        baseline = np.repeat(actuals[:, :1, :], horizon, axis=1)

    mse = np.mean((predictions - actuals) ** 2)
    baseline_mse = np.mean((baseline - actuals) ** 2)

    if baseline_mse > 0:
        return float(1.0 - mse / baseline_mse)
    else:
        return 0.0
