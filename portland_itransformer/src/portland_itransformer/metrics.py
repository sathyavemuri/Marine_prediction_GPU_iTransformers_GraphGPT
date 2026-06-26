"""Evaluation metrics for marine forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    target_names: List[str],
    horizon_buckets: List[Tuple[int, int, str]] = None,
) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    predictions : array
        [num_samples, horizon, num_targets] predicted values (scaled or physical)
    actuals : array
        [num_samples, horizon, num_targets] actual values
    target_names : list
        Names of 13 target variables
    horizon_buckets : list
        Horizon groupings: [(start_hours, end_hours, label), ...]
        Default: 0-6h, 6-24h, 24-72h, 72-168h

    Returns
    -------
    dict
        Metrics dictionary with:
        - 'overall': Overall metrics (MAE, RMSE, RMSE_norm, skill)
        - 'by_target': Per-target metrics
        - 'by_horizon': Per-horizon bucket metrics
    """
    if horizon_buckets is None:
        # Default buckets (in hours)
        cadence_hours = 0.25  # 15-minute cadence
        horizon_buckets = [
            (0, 6, '0-6h'),
            (6, 24, '6-24h'),
            (24, 72, '24-72h'),
            (72, 168, '72-168h'),
        ]

    num_samples, horizon, num_targets = predictions.shape
    assert actuals.shape == predictions.shape

    # Overall metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # RMSE normalized by range
    actual_range = np.nanmax(actuals) - np.nanmin(actuals)
    rmse_norm = rmse / (actual_range + 1e-8)

    # Skill score vs persistence baseline
    persistence_pred = np.repeat(actuals[:, :1, :], horizon, axis=1)
    persistence_mse = np.mean((persistence_pred - actuals) ** 2)
    skill = 1.0 - (mse / persistence_mse) if persistence_mse > 0 else 0.0

    overall = {
        'mse': float(mse),
        'rmse': float(rmse),
        'rmse_normalized': float(rmse_norm),
        'mae': float(mae),
        'skill_vs_persistence': float(skill),
    }

    # Per-target metrics
    per_target_mae = np.nanmean(np.abs(predictions - actuals), axis=(0, 1))
    per_target_rmse = np.sqrt(np.nanmean((predictions - actuals) ** 2, axis=(0, 1)))
    per_target_bias = np.nanmean(predictions - actuals, axis=(0, 1))

    per_target = []
    for i, name in enumerate(target_names):
        per_target.append({
            'target': name,
            'mae': float(per_target_mae[i]),
            'rmse': float(per_target_rmse[i]),
            'bias': float(per_target_bias[i]),
        })

    # Per-horizon metrics
    cadence_hours = 0.25  # 15-minute
    per_horizon = []

    for bucket_start, bucket_end, bucket_label in horizon_buckets:
        # Convert hours to steps (cadence = 15 min = 0.25 hours)
        start_step = int(bucket_start / cadence_hours)
        end_step = int(bucket_end / cadence_hours)

        if end_step > horizon:
            end_step = horizon

        if start_step >= horizon:
            continue

        bucket_pred = predictions[:, start_step:end_step, :]
        bucket_actual = actuals[:, start_step:end_step, :]

        bucket_mae = np.nanmean(np.abs(bucket_pred - bucket_actual))
        bucket_rmse = np.sqrt(np.nanmean((bucket_pred - bucket_actual) ** 2))

        # Persistence baseline for this bucket
        persistence_bucket = np.repeat(bucket_actual[:, :1, :], bucket_actual.shape[1], axis=1)
        persistence_bucket_mse = np.nanmean((persistence_bucket - bucket_actual) ** 2)
        bucket_skill = 1.0 - (np.nanmean((bucket_pred - bucket_actual) ** 2) / persistence_bucket_mse) \
            if persistence_bucket_mse > 0 else 0.0

        per_horizon.append({
            'horizon_label': bucket_label,
            'lead_time_hours': f"{bucket_start}-{bucket_end}",
            'mae': float(bucket_mae),
            'rmse': float(bucket_rmse),
            'skill': float(bucket_skill),
            'num_steps': end_step - start_step,
        })

    return {
        'overall': overall,
        'by_target': per_target,
        'by_horizon': per_horizon,
    }


def compute_baseline_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    baseline_pred: np.ndarray,
    target_names: List[str],
) -> Dict:
    """
    Compare model vs baseline forecasts.

    Parameters
    ----------
    predictions : array
        [num_samples, horizon, num_targets] model predictions
    actuals : array
        [num_samples, horizon, num_targets] actual values
    baseline_pred : array
        [num_samples, horizon, num_targets] baseline predictions
    target_names : list
        Target variable names

    Returns
    -------
    dict
        Comparison metrics
    """
    num_targets = len(target_names)

    model_mse = np.nanmean((predictions - actuals) ** 2)
    baseline_mse = np.nanmean((baseline_pred - actuals) ** 2)

    model_mae = np.nanmean(np.abs(predictions - actuals))
    baseline_mae = np.nanmean(np.abs(baseline_pred - actuals))

    comparison = {
        'model_mse': float(model_mse),
        'baseline_mse': float(baseline_mse),
        'model_mae': float(model_mae),
        'baseline_mae': float(baseline_mae),
        'mae_improvement': float(baseline_mae - model_mae),
        'mae_improvement_pct': float(100 * (baseline_mae - model_mae) / (baseline_mae + 1e-8)),
    }

    return comparison


def skill_score(
    predictions: np.ndarray,
    actuals: np.ndarray,
    baseline: np.ndarray = None,
) -> float:
    """
    Compute skill score (1.0 - normalized_error).

    Parameters
    ----------
    predictions : array
        Model predictions
    actuals : array
        Actual values
    baseline : array, optional
        Baseline forecast (default: persistence)

    Returns
    -------
    float
        Skill in [0, 1] (1.0 = perfect, 0.0 = same as baseline, <0 = worse)
    """
    if baseline is None:
        # Persistence baseline
        horizon = predictions.shape[1]
        baseline = np.repeat(actuals[:, :1, :], horizon, axis=1)

    mse = np.nanmean((predictions - actuals) ** 2)
    baseline_mse = np.nanmean((baseline - actuals) ** 2)

    if baseline_mse > 1e-8:
        return float(1.0 - mse / baseline_mse)
    else:
        return 0.0


def circular_mae(pred_deg: np.ndarray, actual_deg: np.ndarray) -> float:
    """
    Compute circular mean absolute error for directions.

    Parameters
    ----------
    pred_deg : array
        Predicted directions in degrees [0, 360)
    actual_deg : array
        Actual directions in degrees [0, 360)

    Returns
    -------
    float
        Circular MAE in degrees
    """
    diff = ((pred_deg - actual_deg + 180) % 360) - 180
    return float(np.nanmean(np.abs(diff)))


def create_metric_dataframe(
    metrics: Dict,
    split_name: str = 'test',
) -> pd.DataFrame:
    """
    Create pandas DataFrame from metrics dict.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from compute_metrics()
    split_name : str
        Name of split (train, validation, test)

    Returns
    -------
    DataFrame
        [num_targets, columns] with per-target metrics
    """
    rows = []
    for target_metric in metrics['by_target']:
        row = {
            'split': split_name,
            'target': target_metric['target'],
            'mae': target_metric['mae'],
            'rmse': target_metric['rmse'],
            'bias': target_metric.get('bias', 0.0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def create_horizon_dataframe(
    metrics: Dict,
    split_name: str = 'test',
) -> pd.DataFrame:
    """
    Create pandas DataFrame from horizon metrics.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from compute_metrics()
    split_name : str
        Name of split

    Returns
    -------
    DataFrame
        Horizon bucket metrics
    """
    rows = []
    for horizon_metric in metrics['by_horizon']:
        row = {
            'split': split_name,
            'horizon': horizon_metric['horizon_label'],
            'mae': horizon_metric['mae'],
            'rmse': horizon_metric['rmse'],
            'skill': horizon_metric['skill'],
            'num_steps': horizon_metric['num_steps'],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_metrics_summary(metrics: Dict, split_name: str = 'test'):
    """
    Print metrics summary to logger.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from compute_metrics()
    split_name : str
        Name of split
    """
    logger.info("\n" + "=" * 100)
    logger.info(f"METRICS SUMMARY: {split_name.upper()}")
    logger.info("=" * 100)

    # Overall
    overall = metrics['overall']
    logger.info("\nOVERALL:")
    logger.info(f"  MAE:   {overall['mae']:.6f}")
    logger.info(f"  RMSE:  {overall['rmse']:.6f}")
    logger.info(f"  Skill: {overall['skill_vs_persistence']:.4f}")

    # Top performers
    by_target = sorted(metrics['by_target'], key=lambda x: x['mae'])
    logger.info("\nBEST 3 TARGETS (lowest MAE):")
    for i, target_metric in enumerate(by_target[:3]):
        logger.info(
            f"  {i+1}. {target_metric['target']:30s} MAE={target_metric['mae']:.6f}"
        )

    # Worst performers
    logger.info("\nWORST 3 TARGETS (highest MAE):")
    for i, target_metric in enumerate(by_target[-3:]):
        logger.info(
            f"  {i+1}. {target_metric['target']:30s} MAE={target_metric['mae']:.6f}"
        )

    # Horizon degradation
    logger.info("\nHORIZON DEGRADATION:")
    for horizon_metric in metrics['by_horizon']:
        logger.info(
            f"  {horizon_metric['horizon_label']:10s} MAE={horizon_metric['mae']:.6f} "
            f"Skill={horizon_metric['skill']:.4f}"
        )
