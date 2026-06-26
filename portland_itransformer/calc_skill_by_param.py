#!/usr/bin/env python3
"""Calculate per-parameter skill scores vs persistence."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load metrics
metrics_df = pd.read_csv('outputs/test_metrics_by_target.csv')

# Load test data to compute persistence baseline
test_data = pd.read_parquet('data/processed/portland_preprocessed.parquet')
split_labels = np.load('data/processed/split_labels.npy')

# Import constants
import sys
sys.path.insert(0, 'src')
from portland_itransformer.constants import TARGET_FEATURES

# Get test indices
test_mask = split_labels == 2
test_subset = test_data[test_mask].reset_index(drop=True)

# Calculate skill for each parameter
results = []
for idx, row in metrics_df.iterrows():
    param = row['target']
    model_rmse = float(row['rmse'])
    model_mae = float(row['mae'])

    # Get parameter index
    if param in TARGET_FEATURES:
        param_idx = TARGET_FEATURES.index(param)
    else:
        continue

    # Get actual values
    actual_values = test_subset[param].values

    # Persistence baseline RMSE = std deviation (assuming forecasting constant value)
    pers_rmse = np.std(actual_values)

    # Skill = 1 - (MSE_model / MSE_persistence)
    mse_model = model_rmse ** 2
    mse_pers = pers_rmse ** 2

    if mse_pers > 1e-10:
        skill = 1.0 - (mse_model / mse_pers)
    else:
        skill = 0.0

    skill_pct = skill * 100.0

    results.append({
        'Parameter': param,
        'MAE': model_mae,
        'RMSE': model_rmse,
        'Persistence_RMSE': pers_rmse,
        'Skill': skill,
        'Skill_%': skill_pct,
    })

# Sort by skill
df_skill = pd.DataFrame(results).sort_values('Skill_%', ascending=False).reset_index(drop=True)

print("=" * 130)
print("SKILL SCORE VS PERSISTENCE - BY PARAMETER")
print("=" * 130)
print("\nFormula: Skill = 1 - (MSE_model / MSE_persistence)")
print("  Skill > 0% = Better than persistence baseline")
print("  Skill < 0% = Worse than persistence baseline")
print("  Persistence baseline = Forecasting constant value (std deviation of test data)\n")

print(f"{'Rank':<5} {'Parameter':<35} {'Skill_%':<10} {'Model_RMSE':<12} {'Persistence_RMSE':<18} {'Status'}")
print("-" * 130)

for idx, row in df_skill.iterrows():
    param = row['Parameter']
    skill_pct = row['Skill_%']
    model_rmse = row['RMSE']
    pers_rmse = row['Persistence_RMSE']
    status = "BETTER" if skill_pct > 0 else "WORSE"

    print(f"{idx+1:<5} {param:<35} {skill_pct:>8.2f}% {model_rmse:>11.4f} {pers_rmse:>17.4f}   {status}")

print("\n" + "=" * 130)
print("SUMMARY STATISTICS")
print("=" * 130)
print(f"\nBest skill:      {df_skill['Skill_%'].max():>8.2f}% ({df_skill.iloc[0]['Parameter']})")
print(f"Worst skill:     {df_skill['Skill_%'].min():>8.2f}% ({df_skill.iloc[-1]['Parameter']})")
print(f"Mean skill:      {df_skill['Skill_%'].mean():>8.2f}%")
print(f"Median skill:    {df_skill['Skill_%'].median():>8.2f}%")

positive = (df_skill['Skill_%'] > 0).sum()
negative = (df_skill['Skill_%'] < 0).sum()
print(f"\nParameters outperforming persistence:  {positive:2d}/13 ({100*positive/13:5.1f}%)")
print(f"Parameters underperforming persistence: {negative:2d}/13 ({100*negative/13:5.1f}%)")

# Save to CSV
df_skill.to_csv('outputs/skill_by_parameter.csv', index=False)
print(f"\nResults saved to: outputs/skill_by_parameter.csv")
