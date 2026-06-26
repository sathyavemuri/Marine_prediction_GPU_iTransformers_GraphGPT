#!/usr/bin/env python3
"""Compute skill scores (% vs persistence baseline) for each parameter per day."""

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
from marine_local_mtgnn.baselines.selector import BaselineSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

def compute_skill_score(predictions, actuals, persistence_forecast):
    """Compute skill score: 1 - (MSE_model / MSE_persistence)."""
    mse_model = np.mean((predictions - actuals) ** 2)
    mse_persistence = np.mean((persistence_forecast - actuals) ** 2)

    if mse_persistence > 0:
        skill = 1.0 - (mse_model / mse_persistence)
    else:
        skill = 0.0

    return skill * 100  # Convert to percentage

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

    # Load splits and recreate datasets
    train_path = output_dir / "train_split.parquet"
    val_path = output_dir / "validation_split.parquet"
    test_path = output_dir / "test_split.parquet"

    df_train = pd.read_parquet(train_path)
    df_validation = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    # Recreate baselines
    baseline_selector = BaselineSelector(config)
    baseline_selector.fit_all(df_train, df_validation)

    residual_dataset_creator = ResidualDataset(config, baseline_selector)
    residual_data = residual_dataset_creator.create(df_train, df_validation, df_test)
    test_windows = residual_data["test"]["windows"]
    test_dataset = ResidualWindowDataset(test_windows, scaler=scaler)

    # Run inference
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

    # Target names
    target_names = [
        "air_temp_c", "air_pressure_hpa", "wind_u_east_ms", "wind_v_north_ms",
        "water_temp_c", "tidal_level_m", "current_u_east_ms", "current_v_north_ms",
        "dew_point_c", "global_radiation_wm2", "salinity_psu",
        "significant_wave_height_m", "significant_wave_period_s",
        "zero_crossing_period_s", "peak_wave_period_s"
    ]

    # Compute skill scores per day per parameter
    steps_per_day = 96
    skill_results = []

    for day in range(1, 8):  # Days 1-7
        start_idx = (day - 1) * steps_per_day
        end_idx = min(day * steps_per_day, predictions.shape[1])

        if start_idx < predictions.shape[1]:
            day_pred = predictions[:, start_idx:end_idx, :]
            day_actual = actuals[:, start_idx:end_idx, :]

            # Persistence baseline: repeat last known value
            lookback_steps = config.forecast.lookback_steps
            persistence_pred = np.repeat(day_actual[:, :1, :], day_actual.shape[1], axis=1)

            for target_idx, target_name in enumerate(target_names):
                skill = compute_skill_score(
                    day_pred[:, :, target_idx],
                    day_actual[:, :, target_idx],
                    persistence_pred[:, :, target_idx]
                )

                skill_results.append({
                    'Day': day,
                    'Target': target_name,
                    'Skill_%': skill,
                    'Lead_Hours': day * 24,
                })

    df_skill = pd.DataFrame(skill_results)

    # Print results by day
    print("\n" + "=" * 160)
    print("MTGNN SKILL SCORE (% vs Persistence Baseline) - Each Parameter by Day")
    print("=" * 160)
    print("\nSkill Score > 0% = Better than persistence")
    print("Skill Score = 0% = Same as persistence")
    print("Skill Score < 0% = Worse than persistence\n")

    for day in range(1, 8):
        day_data = df_skill[df_skill['Day'] == day].sort_values('Skill_%', ascending=False)
        print(f"\n[DAY {day}] (Lead Time: {day*24} hours)")
        print("-" * 160)

        display = day_data[['Target', 'Skill_%']].copy()
        display['Skill_%'] = display['Skill_%'].apply(lambda x: f"{x:+.1f}%")
        print(display.to_string(index=False))

        # Summary stats for day
        avg_skill = day_data['Skill_%'].mean()
        max_skill = day_data['Skill_%'].max()
        min_skill = day_data['Skill_%'].min()
        print(f"\n  [SUMMARY DAY {day}] Avg Skill: {avg_skill:+.1f}% | Best: {max_skill:+.1f}% | Worst: {min_skill:+.1f}%")

    # Overall summary by parameter
    print("\n\n" + "=" * 160)
    print("SKILL SCORE SUMMARY BY PARAMETER (Days 1-7 Average)")
    print("=" * 160)

    param_summary = df_skill.groupby('Target')['Skill_%'].agg(['mean', 'min', 'max']).round(1)
    param_summary = param_summary.sort_values('mean', ascending=False)
    param_summary.columns = ['Avg_Skill_%', 'Min_Skill_%', 'Max_Skill_%']

    for target, row in param_summary.iterrows():
        avg_str = f"{row['Avg_Skill_%']:+.1f}%"
        min_str = f"{row['Min_Skill_%']:+.1f}%"
        max_str = f"{row['Max_Skill_%']:+.1f}%"
        print(f"{target:30s}  | Avg: {avg_str:>8s} | Min: {min_str:>8s} | Max: {max_str:>8s}")

    # Save to CSV
    output_path = output_dir / "skill_scores_by_parameter_day.csv"
    df_skill.to_csv(output_path, index=False)
    logger.info(f"\nSkill scores saved to {output_path}")

    return df_skill

if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
