#!/usr/bin/env python
"""Train Temporal Fusion Transformer for 2-day marine forecasting."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import time

print("\n" + "="*80)
print("TEMPORAL FUSION TRANSFORMER: 2-Day Marine Forecasting")
print("Training: May 27 - June 24 | Test: June 25-26")
print("="*80)

# ===== 1. LOAD DATA =====
print("\n[1/7] Loading dataset...")
df_1min = pd.read_csv("marine_data_120days_1min.csv", index_col=0, parse_dates=True)
df_1min = df_1min.drop(columns=['precip_type'], errors='ignore')
df_10min = df_1min.resample("10min").mean().dropna()
print(f"[OK] Loaded: {df_10min.shape[0]} rows ({df_10min.shape[0]/(24*6):.1f} days)")

# Column mapping
CSV_COL_MAP = {
    "air_temp_c": "airTemperature", "air_pressure_hpa": "airPressure",
    "relative_hutimestampmidity_pct": "relativeHumidity", "dew_point_c": "dewPointTemperature",
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

GOOD_PARAMS = [
    "airTemperature", "airPressure", "relativeHumidity", "dewPointTemperature",
    "windSpeed", "windDirection", "globalRadiation", "currentSpeed", "currentDirection",
    "tideLevel", "waterTemperature", "conductivity", "salinity", "significantWaveHeight",
    "significantWavePeriod", "peakWaveEnergyPeriod", "zeroCrossingPeriod", "compass",
]

DUPLICATES = {
    "windChillTemperature": "airTemperature",
    "tidePressure": "tideLevel",
    "waterPressure": "tideLevel",
    "waterLevel": "tideLevel",
    "waterTemperature_WQ": "waterTemperature",
    "maxWaveHeight": "significantWaveHeight",
}

print(f"[OK] {len(GOOD_PARAMS)} good parameters, {len(DUPLICATES)} reconstructed")

# ===== 2. PREPARE DATA =====
print("\n[2/7] Preparing data...")
idx = df_10min.index
df_10min["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
df_10min["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
df_10min["day_sin"] = np.sin(2 * np.pi * idx.day / 30)
df_10min["day_cos"] = np.cos(2 * np.pi * idx.day / 30)
df_10min["month"] = idx.month / 12

TEMPORAL_FEATURES = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month"]

horizon_steps = 288
train_steps = 4032
lookback_steps = 288

test_start = len(df_10min) - horizon_steps
train_end = test_start
train_start = train_end - train_steps

train_df = df_10min.iloc[train_start:train_end].copy()
test_df = df_10min.iloc[test_start:].copy()
test_df_orig = df_10min.iloc[test_start:].copy()

print(f"[OK] Train: {train_steps} steps ({train_steps//144} days)")
print(f"  Test: {horizon_steps} steps ({horizon_steps//144} days)")
print(f"  Train dates: {train_df.index[0]} to {train_df.index[-1]}")
print(f"  Test dates: {test_df.index[0]} to {test_df.index[-1]}")

# ===== 3. STANDARDIZE =====
print("\n[3/7] Standardizing...")
scaler = StandardScaler()
train_df[GOOD_PARAMS] = scaler.fit_transform(train_df[GOOD_PARAMS])
test_df[GOOD_PARAMS] = scaler.transform(test_df[GOOD_PARAMS])
print(f"[OK] Standardized training data")

# ===== 4. BUILD TFT ARCHITECTURE =====
print("\n[4/7] Building Temporal Fusion Transformer...")

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_attn = self.softmax(self.fc2(torch.relu(self.fc1(x))))
        return x * x_attn


class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_params, num_temporal_features, hidden_size=128, n_heads=8, n_layers=2, horizon=288):
        super().__init__()
        self.num_params = num_params
        self.hidden_size = hidden_size
        self.horizon = horizon

        self.var_select = VariableSelectionNetwork(num_params, hidden_size)
        self.temporal_select = VariableSelectionNetwork(num_temporal_features, hidden_size)
        self.input_proj = nn.Linear(num_params + num_temporal_features, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.horizon_embed = nn.Embedding(horizon, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_params)
        )

    def forward(self, x_past, t_past, t_future):
        batch_size = x_past.size(0)

        x_selected = self.var_select(x_past)
        t_selected = self.temporal_select(t_past)
        x_combined = torch.cat([x_selected, t_selected], dim=-1)
        x_embedded = self.input_proj(x_combined)

        context = self.transformer(x_embedded)
        context_aggregate = context.mean(dim=1)

        predictions = []
        for h in range(self.horizon):
            horizon_feat = self.horizon_embed(torch.tensor(h, device=x_past.device)).unsqueeze(0).expand(batch_size, -1)
            decoder_input = torch.cat([context_aggregate, horizon_feat], dim=-1)
            pred = self.decoder(decoder_input)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)


device = torch.device('cpu')
torch.set_num_threads(8)
model = TemporalFusionTransformer(len(GOOD_PARAMS), len(TEMPORAL_FEATURES), horizon=horizon_steps).to(device)
print(f"[OK] Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# ===== 5. BUILD TRAINING DATA =====
print("\n[5/7] Building training data...")

class MarineDataset(Dataset):
    def __init__(self, df, params, temporal_features, lookback=288, horizon=288):
        self.df = df
        self.params = params
        self.temporal_features = temporal_features
        self.lookback = lookback
        self.horizon = horizon
        self.samples = [i for i in range(lookback, len(df) - horizon, 2)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i = self.samples[idx]
        x_past = self.df[self.params].iloc[i - self.lookback:i].values.astype(np.float32)
        t_past = self.df[self.temporal_features].iloc[i - self.lookback:i].values.astype(np.float32)
        y_future = self.df[self.params].iloc[i:i + self.horizon].values.astype(np.float32)
        t_future = self.df[self.temporal_features].iloc[i:i + self.horizon].values.astype(np.float32)
        return torch.from_numpy(x_past), torch.from_numpy(t_past), torch.from_numpy(t_future), torch.from_numpy(y_future)

train_dataset = MarineDataset(train_df, GOOD_PARAMS, TEMPORAL_FEATURES, lookback_steps, horizon_steps)
n_val = max(1, int(0.1 * len(train_dataset)))
perm = np.random.permutation(len(train_dataset))
val_idx, tr_idx = perm[:n_val], perm[n_val:]

train_indices = [train_dataset.samples[i] for i in tr_idx]
val_indices = [train_dataset.samples[i] for i in val_idx]

train_df_split = train_df.iloc[min(train_indices):max(train_indices)+horizon_steps].copy()
val_df_split = train_df.iloc[min(val_indices):max(val_indices)+horizon_steps].copy()

train_ds = MarineDataset(train_df_split, GOOD_PARAMS, TEMPORAL_FEATURES, lookback_steps, horizon_steps)
val_ds = MarineDataset(val_df_split, GOOD_PARAMS, TEMPORAL_FEATURES, lookback_steps, horizon_steps)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
print(f"[OK] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

# ===== 6. TRAIN =====
print("\n[6/7] Training TFT...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
patience = 15

train_losses, val_losses = [], []
best_val_loss = float('inf')
best_state = None
wait = 0

t_start = time.time()
for epoch in range(20):
    model.train()
    epoch_loss = 0
    for x_past, t_past, t_future, y_future in train_loader:
        x_past, t_past, t_future, y_future = x_past.to(device), t_past.to(device), t_future.to(device), y_future.to(device)
        optimizer.zero_grad()
        y_pred = model(x_past, t_past, t_future)
        loss = criterion(y_pred, y_future)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_past, t_past, t_future, y_future in val_loader:
            x_past, t_past, t_future, y_future = x_past.to(device), t_past.to(device), t_future.to(device), y_future.to(device)
            y_pred = model(x_past, t_past, t_future)
            val_loss += criterion(y_pred, y_future).item()

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1

    if (epoch + 1) % 5 == 0 or wait >= patience:
        print(f"Epoch {epoch+1:2d}/20 | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Wait: {wait}/15")

    if wait >= patience:
        print(f"→ Early stop at epoch {epoch+1}")
        break

if best_state:
    model.load_state_dict(best_state)

t_train = time.time() - t_start
print(f"[OK] Training complete: {t_train:.1f}s")

# ===== 7. EVALUATE =====
print("\n[7/7] Evaluating on test window...")
model.eval()

train_df_orig = df_10min.iloc[train_start:train_end].copy()
x_test = train_df[GOOD_PARAMS].iloc[-lookback_steps:].values.astype(np.float32)
t_test = train_df[TEMPORAL_FEATURES].iloc[-lookback_steps:].values.astype(np.float32)
t_future_test = test_df[TEMPORAL_FEATURES].iloc[:horizon_steps].values.astype(np.float32)

x_test_t = torch.from_numpy(x_test).unsqueeze(0).to(device)
t_test_t = torch.from_numpy(t_test).unsqueeze(0).to(device)
t_future_t = torch.from_numpy(t_future_test).unsqueeze(0).to(device)

t_infer = time.time()
with torch.no_grad():
    y_pred_norm = model(x_test_t, t_test_t, t_future_t)[0].cpu().numpy()
t_infer = time.time() - t_infer

y_pred = scaler.inverse_transform(y_pred_norm)

# Reconstruct duplicates
recon_models = {}
for dup_col, twin_col in DUPLICATES.items():
    X_train_dup = train_df_orig[twin_col].values.reshape(-1, 1)
    y_train_dup = train_df_orig[dup_col].values
    model_dup = LinearRegression()
    model_dup.fit(X_train_dup, y_train_dup)
    recon_models[dup_col] = model_dup

y_pred_all = np.hstack([y_pred, np.zeros((horizon_steps, len(DUPLICATES)))])
for k, (dup_col, twin_col) in enumerate(DUPLICATES.items()):
    twin_idx = GOOD_PARAMS.index(twin_col)
    y_pred_all[:, len(GOOD_PARAMS) + k] = recon_models[dup_col].predict(y_pred[:, twin_idx:twin_idx+1]).flatten()

# Compute metrics
y_true = test_df_orig[GOOD_PARAMS].iloc[:horizon_steps].values
y_true_all = test_df_orig[GOOD_PARAMS + list(DUPLICATES.keys())].iloc[:horizon_steps].values

last_obs = df_10min[GOOD_PARAMS].iloc[-horizon_steps - 1].values
y_persist = np.tile(last_obs, (horizon_steps, 1))
y_persist_all = np.tile(df_10min[GOOD_PARAMS + list(DUPLICATES.keys())].iloc[-horizon_steps - 1].values, (horizon_steps, 1))

mae_good = mean_absolute_error(y_true, y_pred)
mae_good_persist = mean_absolute_error(y_true, y_persist)
skill_good = (1 - mae_good / mae_good_persist) * 100 if mae_good_persist > 0 else 0

mae_all = mean_absolute_error(y_true_all, y_pred_all)
mae_all_persist = mean_absolute_error(y_true_all, y_persist_all)
skill_all = (1 - mae_all / mae_all_persist) * 100 if mae_all_persist > 0 else 0

# Print results
print("\n" + "="*80)
print("RESULTS: Temporal Fusion Transformer (2-Day Forecast)")
print("="*80)
print(f"\nGood 18 Parameters:")
print(f"  MAE: {mae_good:.4f}")
print(f"  Skill: {skill_good:+.1f}%")
print(f"\nAll 24 Parameters (incl. reconstructed):")
print(f"  MAE: {mae_all:.4f}")
print(f"  Skill: {skill_all:+.1f}%")
print(f"\nTiming:")
print(f"  Training: {t_train:.1f}s")
print(f"  Inference: {t_infer*1000:.2f}ms")

# Per-parameter metrics
metrics_list = []
all_params_list = GOOD_PARAMS + list(DUPLICATES.keys())
for j, p in enumerate(all_params_list):
    y_t = y_true_all[:, j]
    y_p = y_pred_all[:, j]
    y_pers = y_persist_all[:, j]

    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae_pers = mean_absolute_error(y_t, y_pers)
    skill = (1 - mae / mae_pers) * 100 if mae_pers > 0 else 0

    is_dup = "*" if p in DUPLICATES else " "
    metrics_list.append({
        "Parameter": p + is_dup,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Skill_%": round(skill, 1),
    })

metrics_df = pd.DataFrame(metrics_list)

print(f"\n{'='*80}")
print("TOP 5 PARAMETERS:")
print(f"{'='*80}")
for _, row in metrics_df.nlargest(5, "Skill_%").iterrows():
    print(f"  {row['Parameter']:30s} {row['Skill_%']:+7.1f}%  MAE: {row['MAE']:.4f}")

print(f"\n{'='*80}")
print("BOTTOM 5 PARAMETERS:")
print(f"{'='*80}")
for _, row in metrics_df.nsmallest(5, "Skill_%").iterrows():
    print(f"  {row['Parameter']:30s} {row['Skill_%']:+7.1f}%  MAE: {row['MAE']:.4f}")

# Save results
results_summary = pd.DataFrame([{
    "Model": "Temporal Fusion Transformer",
    "Skill_Good_18": skill_good,
    "Skill_All_24": skill_all,
    "MAE_Good_18": mae_good,
    "MAE_All_24": mae_all,
    "Training_Time_s": t_train,
    "Inference_Time_ms": t_infer * 1000,
}])

results_summary.to_csv("tft_2day_results.csv", index=False)
metrics_df.to_csv("tft_2day_metrics.csv", index=False)

print(f"\n{'='*80}")
print(f"[OK] Results saved to tft_2day_results.csv and tft_2day_metrics.csv")
print(f"{'='*80}\n")
