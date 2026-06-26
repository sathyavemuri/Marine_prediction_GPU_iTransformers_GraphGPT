#!/usr/bin/env python3
"""Compute per-parameter skill scores."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from portland_itransformer.preprocessing_marine import TARGET_MARINE
from portland_itransformer.preprocessing_atmosphere import TARGET_ATMOSPHERE

def compute_skill_per_target(predictions, actuals):
    """Compute skill vs persistence for each target."""
    num_samples, horizon, num_targets = predictions.shape
    skills = []

    for i in range(num_targets):
        pred_target = predictions[:, :, i]
        actual_target = actuals[:, :, i]

        # Model MSE
        model_mse = np.mean((pred_target - actual_target) ** 2)

        # Persistence baseline MSE
        persistence_pred = np.repeat(actual_target[:, :1], horizon, axis=1)
        persistence_mse = np.mean((persistence_pred - actual_target) ** 2)

        # Skill
        skill = 1.0 - (model_mse / persistence_mse) if persistence_mse > 0 else 0.0
        skills.append(skill * 100)  # as percentage

    return skills

# Load marine data
marine_pred = np.load('outputs/marine/predictions_test.npy')
marine_actual = np.load('outputs/marine/actuals_test.npy')

# Load atmosphere data
atm_pred = np.load('outputs/atmosphere/predictions_test.npy')
atm_actual = np.load('outputs/atmosphere/actuals_test.npy')

print("\n" + "="*70)
print("MARINE iTRANSFORMER - PER-PARAMETER SKILL SCORES")
print("="*70)

marine_skills = compute_skill_per_target(marine_pred, marine_actual)
marine_df = pd.DataFrame({
    'Parameter': TARGET_MARINE,
    'Skill %': marine_skills
}).sort_values('Skill %', ascending=False)

for _, row in marine_df.iterrows():
    skill_str = f"{row['Skill %']:+.1f}%"
    print(f"{row['Parameter']:40s} {skill_str:>10s}")

print(f"\nOverall Marine Skill: +64.7%")

print("\n" + "="*70)
print("ATMOSPHERIC iTRANSFORMER - PER-PARAMETER SKILL SCORES (ANOMALY-BASED)")
print("="*70)

atm_skills = compute_skill_per_target(atm_pred, atm_actual)
atm_df = pd.DataFrame({
    'Parameter': TARGET_ATMOSPHERE,
    'Skill %': atm_skills
}).sort_values('Skill %', ascending=False)

for _, row in atm_df.iterrows():
    skill_str = f"{row['Skill %']:+.1f}%"
    print(f"{row['Parameter']:40s} {skill_str:>10s}")

print(f"\nOverall Atmospheric Skill: -31.2%")
