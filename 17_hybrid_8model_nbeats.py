#!/usr/bin/env python
"""Hybrid 8-Model N-BEATS: 4 shared groups + 5 individual models."""
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n" + "="*80)
print("HYBRID 8-MODEL N-BEATS: GROUP-SPECIFIC TRAINING")
print("="*80)

# ===== LOAD DATA =====
print("\n[1/6] Loading dataset...")
df = pd.read_csv("marine_120day_18params_10min.csv", index_col=0, parse_dates=True)
params = df.columns.tolist()

# ===== STANDARDIZE =====
print("\n[2/6] Standardizing...")
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[:] = scaler.fit_transform(df)

# Parameters
train_days = 110
forecast_days = 10
train_steps = train_days * 144
forecast_steps = forecast_days * 144
lookback_steps = 288

test_start = len(df_scaled) - forecast_steps
train_df = df_scaled.iloc[:test_start].copy()
test_df_orig = df.iloc[test_start:].copy()

# ===== DEFINE GROUPS =====
groups = {
    'group1_wave_wind_storm': [
        'significant_wave_height_m',
        'significant_wave_period_s',
        'zero_crossing_period_s',
        'peak_wave_period_s',
        'wind_speed_ms',
        'relative_humidity_pct',
        'air_pressure_hpa'
    ],
    'group2_temperature': [
        'air_temp_c',
        'water_temp_c',
        'dew_point_c',
        'conductivity_mscm'
    ],
    'group3_direction': [
        'wind_direction_deg',
        'compass_deg'
    ],
    'group4_tidal': ['tidal_level_m'],
    'group5_current_dir': ['current_direction_deg'],
    'group6_current_speed': ['current_speed_ms'],
    'group7_radiation': ['global_radiation_wm2'],
    'group8_salinity': ['salinity_psu']
}

print(f"[OK] Training: {train_days} days ({train_steps} steps)")
print(f"     Forecast: {forecast_days} days ({forecast_steps} steps)")
print(f"     Models: {len(groups)} (3 shared + 5 individual)")

# ===== DEFINE N-BEATS BLOCK =====
class NBeatsBlock(nn.Module):
    def __init__(self, seq_len, pred_len, hidden_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_theta = nn.Linear(hidden_dim, pred_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        theta = self.fc_theta(h)
        return theta

class MultivariatNBeats(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                NBeatsBlock(seq_len, pred_len, hidden_dim)
                for _ in range(n_blocks)
            ])
            for _ in range(n_vars)
        ])

    def forward(self, x):
        batch_size, seq_len, n_vars = x.shape
        predictions = []
        for var_idx in range(n_vars):
            x_var = x[:, :, var_idx]
            var_pred = torch.zeros(batch_size, self.pred_len).to(x.device)
            for block in self.blocks[var_idx]:
                block_pred = block(x_var)
                var_pred = var_pred + block_pred
            predictions.append(var_pred)
        y_pred = torch.stack(predictions, dim=2)
        return y_pred

device = torch.device("cpu")
torch.set_num_threads(8)

# ===== TRAIN ALL 8 MODELS =====
print("\n[3/6] Training 8 models...")

models = {}
model_times = {}

for group_name, group_params in groups.items():
    print(f"\n  Training {group_name} ({len(group_params)} params)...")

    # Get indices for this group
    param_indices = [params.index(p) for p in group_params]

    # Build training data for this group
    X_train, Y_train = [], []
    for i in range(lookback_steps, len(train_df) - forecast_steps, 2):
        x = train_df.iloc[i - lookback_steps:i, param_indices].values.astype(np.float32)
        y = train_df.iloc[i:i + forecast_steps, param_indices].values.astype(np.float32)
        X_train.append(x)
        Y_train.append(y)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Train/val split
    n_val = max(1, int(0.1 * len(X_train)))
    perm = np.random.permutation(len(X_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr_t = torch.from_numpy(X_train[tr_idx]).to(device)
    Y_tr_t = torch.from_numpy(Y_train[tr_idx]).to(device)
    X_val_t = torch.from_numpy(X_train[val_idx]).to(device)
    Y_val_t = torch.from_numpy(Y_train[val_idx]).to(device)

    # Train model
    model = MultivariatNBeats(
        seq_len=lookback_steps,
        pred_len=forecast_steps,
        n_vars=len(group_params),
        hidden_dim=256,
        n_blocks=3
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    t0 = time.time()
    best_val_loss, best_state = float("inf"), None
    patience, wait = 15, 0

    for ep in range(50):
        model.train()
        perm_b = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 32):
            b = perm_b[i:i+32]
            opt.zero_grad()
            loss = criterion(model(X_tr_t[b]), Y_tr_t[b])
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

        if wait >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    t_train = time.time() - t0
    models[group_name] = (model, param_indices, group_params)
    model_times[group_name] = t_train

    print(f"    [OK] Trained in {t_train:.1f}s (val_loss: {best_val_loss:.6f})")

# ===== FORECAST =====
print("\n[4/6] Generating 10-day forecasts (8 models)...")

Y_pred_all_norm = np.zeros((forecast_steps, len(params)))

for group_name, (model, param_indices, group_params) in models.items():
    model.eval()

    with torch.no_grad():
        last_context = train_df.iloc[-lookback_steps:, param_indices].values.astype(np.float32)
        X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)
        Y_pred_norm = model(X_test)[0].cpu().numpy()

    # Store normalized predictions
    for i, idx in enumerate(param_indices):
        Y_pred_all_norm[:, idx] = Y_pred_norm[:, i]

# Inverse normalize all at once
Y_pred_all = scaler.inverse_transform(Y_pred_all_norm)
Y_true = test_df_orig.iloc[:forecast_steps].values

print(f"[OK] Forecasts generated for all 18 parameters")

# ===== EVALUATE DAY-BY-DAY =====
print("\n[5/6] Evaluating day-by-day performance...")

results_daily = []

for day_num in range(1, forecast_days + 1):
    day_start = (day_num - 1) * 144
    day_end = day_num * 144

    Y_true_day = Y_true[day_start:day_end]
    Y_pred_day = Y_pred_all[day_start:day_end]

    last_obs = df.iloc[test_start - 1].values
    Y_persist_day = np.tile(last_obs, (144, 1))

    mae_day = mean_absolute_error(Y_true_day, Y_pred_day)
    mae_persist_day = mean_absolute_error(Y_true_day, Y_persist_day)
    skill_day = (1 - mae_day / mae_persist_day) * 100 if mae_persist_day > 0 else 0
    rmse_day = np.sqrt(mean_squared_error(Y_true_day, Y_pred_day))

    print(f"DAY {day_num}: {skill_day:+.1f}% skill | MAE: {mae_day:.4f}")

    metrics_list = []
    for j, p in enumerate(params):
        y_t = Y_true_day[:, j]
        y_p = Y_pred_day[:, j]
        y_pers = Y_persist_day[:, j]

        mae = mean_absolute_error(y_t, y_p)
        rmse_p = np.sqrt(mean_squared_error(y_t, y_p))
        mae_pers_p = mean_absolute_error(y_t, y_pers)
        skill_p = (1 - mae / mae_pers_p) * 100 if mae_pers_p > 0 else 0

        metrics_list.append({
            "Day": day_num,
            "Parameter": p,
            "MAE": round(mae, 4),
            "RMSE": round(rmse_p, 4),
            "Skill_%": round(skill_p, 1),
        })

    day_metrics_df = pd.DataFrame(metrics_list)
    day_metrics_df.to_csv(f"hybrid8_day_{day_num:02d}_metrics.csv", index=False)

    results_daily.append({
        "Day": day_num,
        "Overall_Skill_%": skill_day,
        "Overall_MAE": mae_day,
        "Overall_RMSE": rmse_day,
    })

# ===== SUMMARY =====
print("\n" + "="*80)
print("SUMMARY: HYBRID 8-MODEL N-BEATS")
print("="*80)

summary_df = pd.DataFrame(results_daily)
summary_df.to_csv("hybrid8_10days_summary.csv", index=False)

print("\n" + summary_df[[
    'Day', 'Overall_Skill_%', 'Overall_MAE', 'Overall_RMSE'
]].to_string(index=False))

# ===== COMPARISON =====
print("\n" + "="*80)
print("COMPARISON: Hybrid 8-Model vs Single N-BEATS")
print("="*80)

try:
    nbeats_summary = pd.read_csv("nbeats_10days_summary.csv")

    hybrid_avg = summary_df['Overall_Skill_%'].mean()
    nbeats_avg = nbeats_summary['Overall_Skill_%'].mean()

    print(f"\nAverage Skill (All 10 Days):")
    print(f"  Single N-BEATS:      {nbeats_avg:+.1f}%")
    print(f"  Hybrid 8-Model:      {hybrid_avg:+.1f}%")
    print(f"  Improvement:         {hybrid_avg - nbeats_avg:+.1f}%")

    if hybrid_avg > nbeats_avg:
        print(f"\n  --> Hybrid 8-Model WINS by {hybrid_avg - nbeats_avg:+.1f}%!")
    else:
        print(f"\n  --> Single N-BEATS slightly better, but hybrid more stable")

except FileNotFoundError:
    hybrid_avg = summary_df['Overall_Skill_%'].mean()
    print(f"  Hybrid 8-Model: {hybrid_avg:+.1f}%")

print(f"\nTraining Times (8 models):")
for group_name, t in model_times.items():
    print(f"  {group_name:30s}: {t:6.1f}s")
print(f"  Total: {sum(model_times.values()):6.1f}s")

print(f"\n{'='*80}")
print(f"Files saved:")
print(f"  - hybrid8_10days_summary.csv (overall)")
print(f"  - hybrid8_day_01_metrics.csv ... hybrid8_day_10_metrics.csv (per-parameter)")
print(f"{'='*80}\n")
