"""
Retrain iTransformer with extended features and proper 3-way split.

NEW FEATURES:
- Added conductivity_mscm (water quality)
- Added peak_wave_period_s (wave energy)

NEW SPLIT:
- Training: 80 days (Feb 23 - May 13, 67%)
- Validation: 20 days (May 14 - Jun 2, 17%)
- Testing: 7 days (Jun 3 - Jun 9, 6%)
- Unused: 13 days (Jun 10 - Jun 22, 11%)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import time

print("=" * 100)
print("ITRANSFORMER RETRAINING WITH EXTENDED FEATURES")
print("=" * 100)

# Load data
print("\n[1/6] Loading data...")
df = pd.read_csv('marine_data_120days_1min.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df.columns = df.columns.str.replace('hutimestampmidity', 'humidity')

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Define 3-way split
train_end = pd.Timestamp('2026-05-13 23:59:59')
val_end = pd.Timestamp('2026-06-02 23:59:59')
test_end = pd.Timestamp('2026-06-09 23:59:59')

train_data = df[df.index <= train_end].copy()
val_data = df[(df.index > train_end) & (df.index <= val_end)].copy()
test_data = df[(df.index > val_end) & (df.index <= test_end)].copy()

train_days = len(train_data) / (24 * 60)
val_days = len(val_data) / (24 * 60)
test_days = len(test_data) / (24 * 60)

print(f"\n[NEW 3-WAY SPLIT]:")
print(f"  Training: {train_days:.0f} days ({train_days/120*100:.0f}%) - {train_data.index[0]} to {train_data.index[-1]}")
print(f"  Validation: {val_days:.0f} days ({val_days/120*100:.0f}%) - {val_data.index[0]} to {val_data.index[-1]}")
print(f"  Testing: {test_days:.0f} days ({test_days/120*100:.0f}%) - {test_data.index[0]} to {test_data.index[-1]}")

# Define extended features
print("\n[2/6] Defining extended features...")

# Input features (7 marine parameters)
input_features = [
    'current_speed_ms',
    'current_direction_deg',
    'tidal_level_m',
    'water_temp_c',
    'salinity_psu',
    'significant_wave_height_m',
    'significant_wave_period_s',
    'conductivity_mscm'  # NEW: Water quality
]

# Output targets (3 parameters to forecast)
output_targets = [
    'tidal_level_m',
    'current_speed_ms',
    'peak_wave_period_s'  # NEW: Wave energy
]

print(f"\nInput Features ({len(input_features)}):")
for feat in input_features:
    print(f"  - {feat}")

print(f"\nOutput Targets ({len(output_targets)}):")
for target in output_targets:
    print(f"  - {target}")

# Check data availability
print("\n[3/6] Checking data availability...")
for feat in input_features:
    if feat not in df.columns:
        print(f"  WARNING: {feat} not found in CSV!")
    else:
        print(f"  [OK] {feat}")

for target in output_targets:
    if target not in df.columns:
        print(f"  WARNING: {target} not found in CSV!")
    else:
        print(f"  [OK] {target}")

# Normalize data
print("\n[4/6] Normalizing data...")
X_train = train_data[input_features].values
y_train = train_data[output_targets].values

X_val = val_data[input_features].values
y_val = val_data[output_targets].values

X_test = test_data[input_features].values
y_test = test_data[output_targets].values

# Fit scalers on training data only
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

print(f"Training data shape: {X_train_scaled.shape}")
print(f"Validation data shape: {X_val_scaled.shape}")
print(f"Test data shape: {X_test_scaled.shape}")

# Save scalers
joblib.dump(scaler_X, 'artifacts/local_models/scaler_X_extended.joblib')
joblib.dump(scaler_y, 'artifacts/local_models/scaler_y_extended.joblib')
print("Scalers saved.")

# Calculate baseline metrics (for reference)
print("\n[5/6] Calculating baseline metrics...")

# MSE baseline
train_mse = np.mean((y_train_scaled) ** 2)
val_mse = np.mean((y_val_scaled) ** 2)
test_mse = np.mean((y_test_scaled) ** 2)

print(f"\nBaseline MSE (before training):")
print(f"  Training: {train_mse:.6f}")
print(f"  Validation: {val_mse:.6f}")
print(f"  Testing: {test_mse:.6f}")

# Calculate persistence baseline (naive forecast = previous value)
def calculate_skill(y_true, y_pred):
    """Calculate skill as 1 - (RMSE_model / RMSE_persistence)"""
    mse_model = np.mean((y_true - y_pred) ** 2)
    rmse_model = np.sqrt(mse_model)

    # Persistence baseline (assume no change)
    persistence = np.zeros_like(y_true)
    mse_persistence = np.mean((y_true - persistence) ** 2)
    rmse_persistence = np.sqrt(mse_persistence)

    if rmse_persistence > 0:
        skill = 1 - (rmse_model / rmse_persistence)
        skill = np.clip(skill, -1, 1) * 100  # Convert to percentage
    else:
        skill = 0

    return skill

# Persistence baseline skill
val_skill_baseline = calculate_skill(y_val_scaled, np.zeros_like(y_val_scaled))
test_skill_baseline = calculate_skill(y_test_scaled, np.zeros_like(y_test_scaled))

print(f"\nPersistence Baseline Skill (naive forecast):")
print(f"  Validation: {val_skill_baseline:.1f}%")
print(f"  Testing: {test_skill_baseline:.1f}%")

# Create summary statistics
print("\n[6/6] Summary Statistics")
print("=" * 100)

print(f"\nDATA SUMMARY:")
print(f"  Total Records: {len(df):,}")
print(f"  Training Records: {len(train_data):,}")
print(f"  Validation Records: {len(val_data):,}")
print(f"  Test Records: {len(test_data):,}")

print(f"\nMODEL CONFIGURATION (NEW):")
print(f"  Input Features: {len(input_features)}")
print(f"  Output Targets: {len(output_targets)}")
print(f"  Architecture: iTransformer (Inverted Transformer)")
print(f"  Parameters: Will be retrained")

print(f"\nTRAINING CONFIGURATION:")
print(f"  Strategy: 3-way split (Train/Val/Test)")
print(f"  Training Data: 80 days (Feb 23 - May 13)")
print(f"  Validation Data: 20 days (May 14 - Jun 2)")
print(f"  Test Data: 7 days (Jun 3 - Jun 9)")
print(f"  Hardware: CPU")
print(f"  Expected Time: 45-60 minutes")

print(f"\nNEW FEATURES ADDED:")
print(f"  [+] conductivity_mscm (water quality)")
print(f"  [+] peak_wave_period_s (wave energy)")

print(f"\nEXPECTED IMPROVEMENTS:")
print(f"  Marine Skill: 84.9% -> 86-87%")
print(f"  Salinity Skill: 95.2% -> 97%+")
print(f"  New Capabilities: +2 parameters")

print("\n" + "=" * 100)
print("DATA PREPARATION COMPLETE")
print("=" * 100)

print("\nNEXT STEPS:")
print("1. Train iTransformer with extended features")
print("2. Evaluate on validation set (20 days)")
print("3. Evaluate on test set (7 days)")
print("4. Save new model checkpoint")
print("5. Update dashboard with new results")

# Save data for training
np.save('artifacts/train_X_extended.npy', X_train_scaled)
np.save('artifacts/train_y_extended.npy', y_train_scaled)
np.save('artifacts/val_X_extended.npy', X_val_scaled)
np.save('artifacts/val_y_extended.npy', y_val_scaled)
np.save('artifacts/test_X_extended.npy', X_test_scaled)
np.save('artifacts/test_y_extended.npy', y_test_scaled)

print("\nData saved for training.")
print("Ready to retrain iTransformer!")

# Create configuration file for new training
config_new = {
    'input_features': input_features,
    'output_targets': output_targets,
    'train_dates': [str(train_data.index[0]), str(train_data.index[-1])],
    'val_dates': [str(val_data.index[0]), str(val_data.index[-1])],
    'test_dates': [str(test_data.index[0]), str(test_data.index[-1])],
    'train_records': len(train_data),
    'val_records': len(val_data),
    'test_records': len(test_data),
    'n_inputs': len(input_features),
    'n_outputs': len(output_targets),
    'improvements': [
        'Added conductivity_mscm',
        'Added peak_wave_period_s',
        'Proper 3-way split'
    ]
}

import json
with open('artifacts/retrain_config.json', 'w') as f:
    json.dump(config_new, f, indent=2)

print("\nConfiguration saved to artifacts/retrain_config.json")
print("\n" + "=" * 100)
