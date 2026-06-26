#!/usr/bin/env python3
"""Evaluate trained model and generate 10-day accuracy report."""

import sys
from pathlib import Path
import logging
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "marine_local_mtgnn" / "src"))

from marine_local_mtgnn.config import load_config
from marine_local_mtgnn.models import MTGNN
from marine_local_mtgnn.training.dataset import ResidualWindowDataset
from marine_local_mtgnn.datasets.scalers import load_scaler
from marine_local_mtgnn.datasets.residuals import ResidualDataset
from marine_local_mtgnn.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    config_path = Path(__file__).parent / "marine_local_mtgnn" / "configs" / "local_15min_7day.yaml"
    config = load_config(str(config_path))

    output_dir = Path(config.output_root)

    # Load model
    model_path = output_dir / "best_model.pt"
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
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    # Load scaler
    scaler_path = output_dir / "residual_scaler.joblib"
    scaler = load_scaler(scaler_path)

    # Load splits
    train_path = output_dir / "train_split.parquet"
    val_path = output_dir / "validation_split.parquet"
    test_path = output_dir / "test_split.parquet"

    df_train = pd.read_parquet(train_path)
    df_validation = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    # Recreate baselines and residual dataset
    from marine_local_mtgnn.baselines.selector import BaselineSelector
    baseline_selector = BaselineSelector(config)
    baseline_selector.fit_all(df_train, df_validation)

    residual_dataset_creator = ResidualDataset(config, baseline_selector)
    residual_data = residual_dataset_creator.create(df_train, df_validation, df_test)
    test_windows = residual_data["test"]["windows"]
    test_dataset = ResidualWindowDataset(test_windows, scaler=scaler)

    # Evaluate
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch in test_loader:
            history = batch["history"]
            targets = batch["targets"]

            predictions = model(history)
            all_predictions.append(predictions.detach().numpy())
            all_actuals.append(targets.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    logger.info(f"Predictions shape: {predictions.shape}, Actuals shape: {actuals.shape}")

    # Target names
    target_names = [
        "air_temp_c", "air_pressure_hpa", "wind_u_east_ms", "wind_v_north_ms",
        "water_temp_c", "tidal_level_m", "current_u_east_ms", "current_v_north_ms",
        "dew_point_c", "global_radiation_wm2", "salinity_psu",
        "significant_wave_height_m", "significant_wave_period_s",
        "zero_crossing_period_s", "peak_wave_period_s"
    ]

    # Generate 10-day results
    steps_per_day = 96
    results = []

    for day in range(1, 11):
        start_idx = (day - 1) * steps_per_day
        end_idx = min(day * steps_per_day, predictions.shape[1])

        if start_idx < predictions.shape[1]:
            day_pred = predictions[:, start_idx:end_idx, :]
            day_actual = actuals[:, start_idx:end_idx, :]

            day_metrics = compute_metrics(day_pred, day_actual, target_names)

            for target_metric in day_metrics['by_target']:
                results.append({
                    'Day': day,
                    'Target': target_metric['target'],
                    'MAE': target_metric['mae'],
                    'RMSE': target_metric['rmse'],
                })

    df_results = pd.DataFrame(results)

    # Print results
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
        print(f"[Daily Avg MAE] Day {day}: {avg_mae:.4f}")

    # Save to CSV
    output_path = output_dir / "10day_accuracy_results.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")

    return df_results

if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
