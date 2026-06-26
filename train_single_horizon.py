#!/usr/bin/env python
"""Train single horizon (2-day) with real-time output."""
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n" + "="*70)
print("TRAINING iTransformer: 2-Day Horizon")
print("="*70)

# ===== STEP 1: LOAD DATA =====
print("\n[1/6] Loading 120-day dataset...")
df_1min = pd.read_csv("marine_data_120days_1min.csv", index_col=0, parse_dates=True)
df_1min = df_1min.drop(columns=['precip_type'], errors='ignore')
df_10min = df_1min.resample("10min").mean().dropna()
print(f"      Loaded: {df_10min.shape[0]} rows ({df_10min.shape[0]/(24*6):.1f} days)")

# ===== STEP 2: PREPARE DATA =====
print("\n[2/6] Preparing data (column mapping, calendar features)...")
CSV_COL_MAP = {
    "air_temp_c": "airTemperature", "air_pressure_hpa": "airPressure",
    "relative_humidity_pct": "relativeHumidity", "dew_point_c": "dewPointTemperature",
    "wind_chill_c": "windChillTemperature", "wind_speed_ms": "windSpeed",
    "wind_direction_deg": "windDirection", "compass_deg": "compass",
    "global_radiation_wm2": "globalRadiation", "current_speed_ms": "currentSpeed",
    "current_direction_deg": "currentDirection", "water_pressure_dbar": "waterPressure",
    "tide_pressure_dbar": "tidePressure", "tidal_level_m": "tideLevel",
    "water_temp_c": "waterTemperature", "conductivity_mscm": "conductivity",
    "salinity_psu": "salinity", "water_temp_quality_c": "waterTemperature_WQ",
    "significant_wave_height_m": "significantWaveHeight", "max_wave_height_m": "maxWaveHeight",
    "water_level_m": "waterLevel", "significant_wave_period_s": "significantWavePeriod",
    "peak_wave_period_s": "peakWaveEnergyPeriod", "zero_crossing_period_s": "zeroCrossingPeriod",
}
df_10min = df_10min.rename(columns=CSV_COL_MAP)

GOOD_PARAMS = ["airTemperature", "airPressure", "relativeHumidity", "dewPointTemperature",
    "windSpeed", "windDirection", "globalRadiation", "currentSpeed", "currentDirection",
    "tideLevel", "waterTemperature", "conductivity", "salinity", "significantWaveHeight",
    "significantWavePeriod", "peakWaveEnergyPeriod", "zeroCrossingPeriod", "compass"]
DUPLICATES = [("airTemperature", "windChillTemperature"), ("tideLevel", "tidePressure"),
    ("tideLevel", "waterPressure"), ("tideLevel", "waterLevel"),
    ("waterTemperature", "waterTemperature_WQ"), ("significantWaveHeight", "maxWaveHeight")]
ALL_PARAMS = GOOD_PARAMS + [d[1] for d in DUPLICATES]

idx = df_10min.index
df_10min["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
df_10min["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
df_10min["dom_sin"] = np.sin(2 * np.pi * idx.day / 30)
df_10min["dom_cos"] = np.cos(2 * np.pi * idx.day / 30)
print(f"      24 parameters ready (18 good + 6 duplicates)")

# ===== STEP 3: SPLIT DATA =====
print("\n[3/6] Splitting data (2-day horizon = 288 steps forecast, 28-day train)...")
horizon_steps = 288
train_steps = 4032
test_start = len(df_10min) - horizon_steps
train_end = test_start
train_start = train_end - train_steps

train_df = df_10min.iloc[train_start:train_end].copy()
test_df = df_10min.iloc[test_start:].copy()

# Standardize
param_stats = {}
for p in ALL_PARAMS:
    param_stats[p] = {"mean": train_df[p].mean(), "std": train_df[p].std()}
    train_df[p] = (train_df[p] - param_stats[p]["mean"]) / param_stats[p]["std"]

# Build windows
lookback = horizon_steps
X_train, Y_train = [], []
for i in range(lookback, len(train_df) - horizon_steps, 2):
    x = train_df[ALL_PARAMS].iloc[i - lookback:i].values.astype(np.float32)
    y = train_df[ALL_PARAMS].iloc[i:i + horizon_steps].values.astype(np.float32)
    X_train.append(x)
    Y_train.append(y)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(f"      {X_train.shape[0]} training windows built")

# ===== STEP 4: BUILD MODEL =====
print("\n[4/6] Building iTransformer model...")

class iTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=2, horizon=288, n_params=24):
        super().__init__()
        self.param_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_params, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=n_heads, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.param_proj(x.mean(dim=1, keepdim=True).transpose(1, 2))
        x = x + self.pos_emb
        x = self.transformer(x)
        x = self.head(x)
        return x.transpose(1, 2)

device = torch.device("cpu")
torch.set_num_threads(8)
model = iTransformer(horizon=horizon_steps, n_params=len(ALL_PARAMS)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
print(f"      Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# ===== STEP 5: TRAIN =====
print("\n[5/6] Training (50 epochs with early stopping)...")
n_val = max(1, int(0.1 * len(X_train)))
perm = np.random.permutation(len(X_train))
val_idx, tr_idx = perm[:n_val], perm[n_val:]
X_tr, Y_tr = X_train[tr_idx], Y_train[tr_idx]
X_val, Y_val = X_train[val_idx], Y_train[val_idx]

X_tr_t = torch.from_numpy(X_tr).to(device)
Y_tr_t = torch.from_numpy(Y_tr).to(device)
X_val_t = torch.from_numpy(X_val).to(device)
Y_val_t = torch.from_numpy(Y_val).to(device)

t_train_start = time.time()
best_val_loss = float("inf")
best_state = None
patience, wait = 15, 0

for ep in range(50):
    model.train()
    perm_b = torch.randperm(len(X_tr_t))
    for i in range(0, len(X_tr_t), 32):
        b = perm_b[i:i+32]
        opt.zero_grad()
        pred = model(X_tr_t[b])
        loss = criterion(pred, Y_tr_t[b])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), Y_val_t).item()

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1

    if (ep + 1) % 10 == 0 or wait >= patience:
        print(f"      Epoch {ep+1:2d}/50 | Val loss: {val_loss:.6f} | Wait: {wait}/15")

    if wait >= patience:
        print(f"      --> Early stop at epoch {ep+1}")
        break

if best_state:
    model.load_state_dict(best_state)

t_train = time.time() - t_train_start
print(f"      Training complete: {t_train:.1f}s")

# ===== STEP 6: EVALUATE =====
print("\n[6/6] Evaluating on test window...")
model.eval()
t_infer = time.time()
last_context = train_df[ALL_PARAMS].iloc[-lookback:].values.astype(np.float32)
X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)
with torch.no_grad():
    Y_pred_norm = model(X_test)[0].cpu().numpy()
t_infer = time.time() - t_infer

# Inverse normalize
Y_pred = np.zeros_like(Y_pred_norm)
for j, p in enumerate(ALL_PARAMS):
    Y_pred[:, j] = Y_pred_norm[:, j] * param_stats[p]["std"] + param_stats[p]["mean"]

Y_true = test_df[ALL_PARAMS].iloc[:horizon_steps].values
last_obs = df_10min[ALL_PARAMS].iloc[-horizon_steps - 1].values
Y_persist = np.tile(last_obs, (horizon_steps, 1))

# Compute metrics
metrics = []
for j, p in enumerate(ALL_PARAMS):
    y_true = Y_true[:, j]
    y_pred = Y_pred[:, j]
    y_persist = Y_persist[:, j]

    mae_p = mean_absolute_error(y_true, y_persist)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    skill = (1 - mae / mae_p) * 100 if mae_p > 0 else np.nan

    metrics.append({
        "parameter": p,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "skill_%": round(skill, 1),
        "persistence_MAE": round(mae_p, 4),
    })

metrics_df = pd.DataFrame(metrics)
mean_skill = metrics_df["skill_%"].mean()

# Save results
metrics_df.to_csv("metrics_horizon_2d.csv", index=False)
print(f"\n      Inference time: {t_infer*1000:.1f}ms")
print(f"      Mean skill: {mean_skill:+.1f}%")

# Show top/bottom performers
print(f"\n      Top 5 performers:")
for _, row in metrics_df.nlargest(5, "skill_%").iterrows():
    print(f"        {row['parameter']:30s} {row['skill_%']:+7.1f}%")

print(f"\n      Bottom 5 performers:")
for _, row in metrics_df.nsmallest(5, "skill_%").iterrows():
    print(f"        {row['parameter']:30s} {row['skill_%']:+7.1f}%")

print("\n" + "="*70)
print(f"TRAINING COMPLETE")
print(f"  Training time: {t_train:.1f}s")
print(f"  Mean skill: {mean_skill:+.1f}%")
print(f"  Saved: metrics_horizon_2d.csv")
print("="*70 + "\n")
