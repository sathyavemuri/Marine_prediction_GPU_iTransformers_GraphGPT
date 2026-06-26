#!/usr/bin/env python3
"""Generate 10-day forecast accuracy results from trained MTGNN model."""

import sys
from pathlib import Path
import logging
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "marine_local_mtgnn" / "src"))

from marine_local_mtgnn.config import load_config
from marine_local_mtgnn.models import MTGNN
from marine_local_mtgnn.training.dataset import ResidualWindowDataset
from marine_local_mtgnn.datasets.scalers import load_scaler
from marine_local_mtgnn.datasets.residuals import ResidualDataset
from marine_local_mtgnn.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(config):
    """Load test split and create dataset."""
    test_path = Path(config.output_root) / "test_split.parquet"
    df_test = pd.read_parquet(test_path)

    scaler_path = Path(config.output_root) / "residual_scaler.joblib"
    scaler = load_scaler(scaler_path)

    baseline_path = Path(config.output_root) / "baseline_selection.json"
    with open(baseline_path) as f:
        baseline_info = json.load(f)
        best_baseline = min(baseline_info, key=baseline_info.get)

    # Load baseline forecasts from test split (already computed during preprocessing)
    test_dataset = ResidualWindowDataset(
        df_test,
        config.forecast.lookback_steps,
        config.forecast.horizon_steps,
        scaler=scaler,
    )

    return test_dataset, scaler

def generate_10day_accuracy_results():
    """Load model, evaluate on test set, generate 10-day accuracy table."""

    logger.info("=" * 100)
    logger.info("GENERATING 10-DAY FORECAST ACCURACY RESULTS")
    logger.info("=" * 100)

    # Load config
    config_path = project_root / "marine_local_mtgnn" / "configs" / "local_15min_7day.yaml"
    config = load_config(str(config_path))

    # Load model
    model_path = Path(config.output_root) / "best_model.pt"
    model = MTGNN(
        num_nodes=config.model.num_input_nodes,
        num_targets=config.model.num_direct_targets,
        lookback_steps=config.forecast.lookback_steps,
        horizon_steps=config.forecast.horizon_steps,
        hidden_channels=config.model.hidden_channels,
        skip_channels=config.model.skip_channels,
        end_channels=config.model.end_channels,
        kernel_size=config.model.kernel_size,
        dilation_exponential=config.model.dilation_exponential,
        dropout=config.model.dropout,
    )

    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    logger.info(f"Loaded model from {model_path}")

    # Load test data
    logger.info("Loading test data...")
    test_path = Path(config.output_root) / "test_split.parquet"
    df_test = pd.read_parquet(test_path)

    scaler_path = Path(config.output_root) / "residual_scaler.joblib"
    scaler = load_scaler(scaler_path)

    # Create dataset
    test_dataset = ResidualWindowDataset(
        [
            (df_test.iloc[i:i+config.forecast.lookback_steps].values,
             np.zeros((config.forecast.horizon_steps, config.model.num_direct_targets)),
             np.zeros((config.forecast.horizon_steps, config.model.num_direct_targets)))
            for i in range(0, len(df_test) - config.forecast.lookback_steps, config.forecast.lookback_steps)
        ],
        scaler=scaler,
    )

    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    # Run inference
    logger.info("Running inference on test set...")
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (tuple, list)):
                history, baseline_forecast, targets = batch[0], batch[1], batch[2]
            else:
                history = batch['history']
                baseline_forecast = batch['baseline_forecast']
                targets = batch['targets']

            predictions = model(history)
            all_predictions.append(predictions.numpy())
            all_actuals.append(targets.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Actuals shape: {actuals.shape}")

    # Compute metrics
    target_names = config.constants.TARGET_NAMES if hasattr(config, 'constants') else [
        "air_temp_c", "air_pressure_hpa", "wind_u_east_ms", "wind_v_north_ms",
        "water_temp_c", "tidal_level_m", "current_u_east_ms", "current_v_north_ms",
        "dew_point_c", "global_radiation_wm2", "salinity_psu",
        "significant_wave_height_m", "significant_wave_period_s",
        "zero_crossing_period_s", "peak_wave_period_s"
    ]

    metrics = compute_metrics(predictions, actuals, target_names)

    # Generate 10-day accuracy table (steps per day at 15-min cadence)
    steps_per_day = 96
    results = []

    for day in range(1, 11):
        start_step = (day - 1) * steps_per_day
        end_step = day * steps_per_day

        if end_step <= predictions.shape[1]:
            day_predictions = predictions[:, start_step:end_step, :]
            day_actuals = actuals[:, start_step:end_step, :]
            day_metrics = compute_metrics(day_predictions, day_actuals, target_names)

            for target_metric in day_metrics['by_target']:
                results.append({
                    'Day': day,
                    'Target': target_metric['target'],
                    'MAE': target_metric['mae'],
                    'RMSE': target_metric['rmse'],
                    'Lead_Hours': day * 24,
                })

    df_results = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 140)
    print("MTGNN 10-DAY FORECAST ACCURACY: Daily Performance by Parameter")
    print("=" * 140)

    for day in range(1, 11):
        day_data = df_results[df_results['Day'] == day]
        print(f"\n[DAY {day}] (Lead Time: {day*24} hours)")
        print("-" * 140)

        display = day_data[['Target', 'MAE', 'RMSE']].copy()
        display['MAE'] = display['MAE'].apply(lambda x: f"{x:.4f}")
        display['RMSE'] = display['RMSE'].apply(lambda x: f"{x:.4f}")
        print(display.to_string(index=False))

        avg_mae = day_data['MAE'].mean()
        print(f"\n  [SUMMARY] Day {day} Average MAE: {avg_mae:.4f}")

    # Parameter summary
    print("\n" + "=" * 100)
    print("PARAMETER PERFORMANCE SUMMARY (10-Day Forecast)")
    print("=" * 100)

    summary = df_results.groupby('Target').agg({
        'MAE': ['min', 'mean', 'max'],
        'RMSE': ['min', 'mean', 'max'],
    }).round(4)

    summary.columns = ['MAE_Best', 'MAE_Avg', 'MAE_Worst', 'RMSE_Best', 'RMSE_Avg', 'RMSE_Worst']
    summary = summary.reset_index()
    print(summary.to_string(index=False))

    # Save to CSV
    output_path = Path(config.output_root) / "10day_accuracy_results.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nSaved results to {output_path}")

    return df_results

if __name__ == "__main__":
    try:
        results = generate_10day_accuracy_results()
        logger.info("\n[SUCCESS] Accuracy results generated!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n[ERROR] {e}", exc_info=True)
        sys.exit(1)
