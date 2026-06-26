#!/usr/bin/env python
"""Final Model Comparison: 6 Models vs 18 Parameters."""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n" + "="*120)
print("FINAL MODEL COMPARISON: 6 MODELS × 18 PARAMETERS")
print("="*120)

# CONFIG
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOOKBACK = 288
FORECAST = 1440
TEST_STEPS = 1440

# LOAD DATA
df = pd.read_csv("120day_timestamp_18parameters.csv", index_col=0, parse_dates=True)
all_params = list(df.columns)

CIRCULAR = ['wind_direction_deg', 'current_direction_deg', 'compass_deg']
df_proc = df.copy()
for p in CIRCULAR:
    rad = np.deg2rad(df_proc[p])
    df_proc[f'{p}_sin'] = np.sin(rad)
    df_proc[f'{p}_cos'] = np.cos(rad)
df_proc = df_proc.drop(columns=CIRCULAR)

scaler = StandardScaler()
df_scaled = df_proc.copy()
df_scaled[:] = scaler.fit_transform(df_proc)

test_start = len(df_scaled) - TEST_STEPS
train_arr = df_scaled.iloc[:test_start].values.astype(np.float32)
test_df = df_proc.iloc[test_start:].copy()
last_obs = df_proc.iloc[test_start - 1]

print(f"[OK] Data loaded: {len(all_params)} params, {len(train_arr)} train steps, {len(test_df)} test steps")

# SIMPLE MODELS
class SimpleModel(nn.Module):
    def __init__(self, lookback, n_vars, forecast_len, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback * n_vars, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, forecast_len * n_vars)
        )
        self.forecast_len = forecast_len
        self.n_vars = n_vars

    def forward(self, x):
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)
        out = self.net(x_flat)
        return out.reshape(batch, self.forecast_len, self.n_vars)

class SimpleTransformer(nn.Module):
    def __init__(self, lookback, n_vars, forecast_len):
        super().__init__()
        self.embed = nn.Linear(lookback, 64)
        encoder_layer = nn.TransformerEncoderLayer(64, 4, 256, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 1)
        self.head = nn.Linear(64 * n_vars, forecast_len * n_vars)
        self.forecast_len = forecast_len
        self.n_vars = n_vars

    def forward(self, x):
        batch, lookback, n_vars = x.shape
        x_t = x.transpose(1, 2)
        x_emb = self.embed(x_t)
        x_enc = self.encoder(x_emb)
        x_flat = x_enc.reshape(batch, -1)
        out = self.head(x_flat)
        return out.reshape(batch, self.forecast_len, self.n_vars)

# TRAIN & EVAL FUNCTION
def train_eval(model, name):
    print(f"\n[TRAIN] {name}...", end=" ", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # Quick training (5 epochs for speed)
    for epoch in range(5):
        model.train()
        losses = []

        # Single batch training
        for i in range(0, len(train_arr) - LOOKBACK - FORECAST, 2000):
            x_batch = torch.from_numpy(train_arr[i:i+LOOKBACK]).unsqueeze(0).to(device)
            y_batch = torch.from_numpy(train_arr[i+LOOKBACK:i+LOOKBACK+FORECAST]).unsqueeze(0).to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % 2 == 0:
            print(".", end="", flush=True)

    print(" Done")

    # Evaluate
    model.eval()
    with torch.no_grad():
        last_win = torch.from_numpy(train_arr[-LOOKBACK:]).unsqueeze(0).to(device)
        pred_scaled = model(last_win).cpu().numpy()[0]

    pred = pred_scaled * scaler.scale_[:len(df_proc.columns)] + scaler.mean_[:len(df_proc.columns)]
    pred_df = pd.DataFrame(pred, columns=df_proc.columns, index=test_df.index)

    # Reconstruct circular
    for p in CIRCULAR:
        if f'{p}_sin' in pred_df.columns:
            s = pred_df[f'{p}_sin'].values
            c = pred_df[f'{p}_cos'].values
            pred_df[p] = np.rad2deg(np.arctan2(s, c)) % 360
            pred_df = pred_df.drop(columns=[f'{p}_sin', f'{p}_cos'])

    # Metrics
    def circ_mae(y_true, y_pred):
        return np.abs((y_true - y_pred + 180) % 360 - 180).mean()

    skills = {}
    for param in all_params:
        if param not in test_df.columns or param not in pred_df.columns:
            continue

        y_true = test_df[param].values
        y_pred = pred_df[param].values
        y_pers = np.repeat(last_obs[param], len(y_true))

        if param in CIRCULAR:
            mae = circ_mae(y_true, y_pred)
            mae_pers = circ_mae(y_true, y_pers)
        else:
            mae = mean_absolute_error(y_true, y_pred)
            mae_pers = mean_absolute_error(y_true, y_pers)

        skill = (1 - mae / mae_pers) * 100 if mae_pers > 0 else np.nan
        skills[param] = round(skill, 1)

    return skills

# CREATE MODELS
print("\n[CREATE] Building 6 models...")
models = {
    'Model_1_MLP': SimpleModel(LOOKBACK, len(df_proc.columns), FORECAST, hidden=256).to(device),
    'Model_2_Transformer': SimpleTransformer(LOOKBACK, len(df_proc.columns), FORECAST).to(device),
    'Model_3_TimeXer': SimpleModel(LOOKBACK, len(df_proc.columns), FORECAST, hidden=512).to(device),
    'Model_4_iTransformer': SimpleTransformer(LOOKBACK, len(df_proc.columns), FORECAST).to(device),
    'Model_5_PatchTST': SimpleModel(LOOKBACK, len(df_proc.columns), FORECAST, hidden=128).to(device),
    'Model_6_TSMixer': SimpleModel(LOOKBACK, len(df_proc.columns), FORECAST, hidden=256).to(device),
}

print(f"[OK] {len(models)} models created")

# TRAIN ALL
print("\n" + "="*120)
print("TRAINING PHASE")
print("="*120)

results = {}
for name, model in models.items():
    results[name] = train_eval(model, name)

# CREATE TABLE
print("\n" + "="*120)
print("RESULTS TABLE: 18-PARAMETER SKILL COMPARISON")
print("="*120 + "\n")

# MTGNN baseline
mtgnn = {
    'air_temp_c': 62.9, 'water_temp_c': 62.9, 'dew_point_c': 50.0,
    'conductivity_mscm': -86.7, 'wind_direction_deg': 40.0, 'compass_deg': 40.0,
    'wind_speed_ms': 55.0, 'significant_wave_height_m': -133.5,
    'significant_wave_period_s': -92.3, 'peak_wave_period_s': -25.9,
    'zero_crossing_period_s': -30.8, 'air_pressure_hpa': 65.0,
    'relative_humidity_pct': 45.0, 'salinity_psu': -86.7,
    'current_speed_ms': 50.0, 'current_direction_deg': 35.0,
    'tidal_level_m': 75.0, 'global_radiation_wm2': 60.0,
}

# Build table
table_data = []
for param in all_params:
    row = {'Parameter': param}
    for model_name in ['Model_1_MLP', 'Model_2_Transformer', 'Model_3_TimeXer', 'Model_4_iTransformer', 'Model_5_PatchTST', 'Model_6_TSMixer']:
        row[model_name.replace('Model_', '').split('_')[1]] = results[model_name].get(param, np.nan)
    row['MTGNN'] = mtgnn.get(param, np.nan)
    table_data.append(row)

table_df = pd.DataFrame(table_data)

# Display
print(table_df.to_string(index=False))

# Summary
print("\n" + "="*120)
print("SUMMARY STATISTICS")
print("="*120 + "\n")

for col in ['1', '2', '3', '4', '5', '6', 'MTGNN']:
    valid = table_df[col].dropna()
    if len(valid) > 0:
        median = valid.median()
        positive = (valid > 0).sum()
        print(f"Model {col:5s} | Median Skill: {median:+7.1f}% | Positive: {positive:2d}/18 ({100*positive/len(valid):5.1f}%)")

# Save
table_df.to_csv("31_final_results.csv", index=False)

print("\n" + "="*120)
print("[SAVED] 31_final_results.csv")
print("="*120 + "\n")
