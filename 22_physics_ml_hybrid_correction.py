#!/usr/bin/env python
"""Hybrid Physics+ML: Physics predictions + ML residual correction for 6 poor parameters."""
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
print("HYBRID PHYSICS+ML CORRECTION: LEARN RESIDUALS FROM PHYSICS PREDICTIONS")
print("="*80)

# ===== LOAD DATA =====
print("\n[1/7] Loading dataset...")
df = pd.read_csv("marine_120day_18params_10min.csv", index_col=0, parse_dates=True)
all_params = df.columns.tolist()

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[:] = scaler.fit_transform(df)

train_days = 110
forecast_days = 10
train_steps = train_days * 144
forecast_steps = forecast_days * 144
lookback_steps = 288

test_start = len(df_scaled) - forecast_steps
train_df = df_scaled.iloc[:test_start].copy()

print(f"[OK] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ===== PHYSICS MODELS =====
class WavePhysics:
    """Physics-based wave height and period estimation."""

    @staticmethod
    def estimate_wave_height(wind_speed_ms, time_since_start_hours=6):
        """Pierson-Moskowitz: H_s ≈ 0.24 * (U^2 / g)"""
        g = 9.81
        H_s = 0.24 * (wind_speed_ms ** 2) / g
        # Wave age factor: younger waves (lower time_since_start) grow slower
        age_factor = min(1.0, time_since_start_hours / 12.0)
        H_s = H_s * age_factor
        return np.clip(H_s, 0.0, 15.0)

    @staticmethod
    def estimate_wave_period(wave_height_m):
        """Empirical: T_p ≈ 0.72 * sqrt(H_s)"""
        if wave_height_m <= 0:
            return 2.0
        T_p = 0.72 * np.sqrt(wave_height_m)
        return np.clip(T_p, 2.0, 20.0)

class TempPhysics:
    """Physics-based water temperature prediction."""

    @staticmethod
    def estimate_water_temp(air_temp, solar_radiation, prev_water_temp, dt_hours=0.1667):
        """Heat balance: dT/dt = (Q_solar - Q_loss) / (rho * c_p * depth)"""
        rho = 1025  # seawater density
        c_p = 3990  # heat capacity
        depth = 10  # mixing depth

        Q_solar = solar_radiation * 0.8
        delta_T = prev_water_temp - air_temp
        Q_loss = 20 + 5 * delta_T  # empirical

        Q_net = Q_solar - Q_loss
        dT = (Q_net / (rho * c_p * depth)) * dt_hours

        new_temp = prev_water_temp + dT
        return np.clip(new_temp, -2.0, 35.0)

# ===== ML RESIDUAL CORRECTION MODEL =====
class ResidualCorrector(nn.Module):
    """ML model to learn and correct physics prediction errors."""
    def __init__(self, seq_len, hidden_dim=64):
        super().__init__()
        # Input: 1 physics pred + seq_len history values
        self.fc1 = nn.Linear(1 + seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # predict residual
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, physics_pred, history):
        # physics_pred: (batch, 1)
        # history: (batch, seq_len)
        x = torch.cat([physics_pred, history], dim=-1)  # (batch, 1 + seq_len)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        residual = self.fc3(x)  # (batch, 1)
        return residual

device = torch.device("cpu")
torch.set_num_threads(8)

# ===== STEP 1: GENERATE PHYSICS PREDICTIONS ON TRAINING DATA =====
print("\n[2/7] Generating physics predictions on training data...")

# Get indices
idx_wave_height = all_params.index('significant_wave_height_m')
idx_wave_period = all_params.index('significant_wave_period_s')
idx_zero_cross = all_params.index('zero_crossing_period_s')
idx_peak_wave = all_params.index('peak_wave_period_s')
idx_water_temp = all_params.index('water_temp_c')

idx_wind_speed = all_params.index('wind_speed_ms')
idx_solar = all_params.index('global_radiation_wm2')
idx_air_temp = all_params.index('air_temp_c')

# Training data: generate physics predictions for wave height and water temp
train_wave_height_physics = np.zeros(len(train_df))
train_wave_period_physics = np.zeros(len(train_df))
train_water_temp_physics = np.zeros(len(train_df))

wind_speed_train = df.iloc[:test_start, idx_wind_speed].values
solar_train = df.iloc[:test_start, idx_solar].values
air_temp_train = df.iloc[:test_start, idx_air_temp].values
water_temp_start = df.iloc[0, idx_water_temp]

# Physics predictions on training window
water_temp_current = water_temp_start
for t in range(len(train_df)):
    train_wave_height_physics[t] = WavePhysics.estimate_wave_height(wind_speed_train[t])
    train_wave_period_physics[t] = WavePhysics.estimate_wave_period(train_wave_height_physics[t])
    water_temp_current = TempPhysics.estimate_water_temp(
        air_temp_train[t], solar_train[t], water_temp_current
    )
    train_water_temp_physics[t] = water_temp_current

print(f"[OK] Physics predictions generated for {len(train_df)} training timesteps")

# ===== STEP 2: BUILD TRAINING DATA FOR RESIDUAL CORRECTION =====
print("\n[3/7] Building training data for ML residual corrector...")

# For wave height
X_train_wh, Y_train_wh = [], []
for i in range(lookback_steps, len(train_df) - forecast_steps, 4):
    # Input: lookback window of actual values
    x_hist = train_df.iloc[i - lookback_steps:i, idx_wave_height].values.astype(np.float32)
    # Physics prediction for next step
    y_phys = train_wave_height_physics[i].astype(np.float32)
    # Actual target
    y_actual = train_df.iloc[i, idx_wave_height].astype(np.float32)
    # Residual = actual - physics
    y_residual = y_actual - y_phys

    X_train_wh.append(np.concatenate([np.array([y_phys]), x_hist]))
    Y_train_wh.append(y_residual)

X_train_wh = np.array(X_train_wh)
Y_train_wh = np.array(Y_train_wh)

# For water temperature
X_train_wt, Y_train_wt = [], []
for i in range(lookback_steps, len(train_df) - forecast_steps, 4):
    x_hist = train_df.iloc[i - lookback_steps:i, idx_water_temp].values.astype(np.float32)
    y_phys = train_water_temp_physics[i].astype(np.float32)
    y_actual = train_df.iloc[i, idx_water_temp].astype(np.float32)
    y_residual = y_actual - y_phys

    X_train_wt.append(np.concatenate([np.array([y_phys]), x_hist]))
    Y_train_wt.append(y_residual)

X_train_wt = np.array(X_train_wt)
Y_train_wt = np.array(Y_train_wt)

print(f"[OK] Built {len(X_train_wh)} wave height samples, {len(X_train_wt)} water temp samples")

# ===== STEP 3: TRAIN RESIDUAL CORRECTORS =====
print("\n[4/7] Training ML residual correctors...")

# Train wave height corrector
n_val_wh = max(1, int(0.1 * len(X_train_wh)))
perm_wh = np.random.permutation(len(X_train_wh))
val_idx_wh, tr_idx_wh = perm_wh[:n_val_wh], perm_wh[n_val_wh:]

X_tr_wh = torch.from_numpy(X_train_wh[tr_idx_wh]).to(device)
Y_tr_wh = torch.from_numpy(Y_train_wh[tr_idx_wh]).unsqueeze(1).to(device)
X_val_wh = torch.from_numpy(X_train_wh[val_idx_wh]).to(device)
Y_val_wh = torch.from_numpy(Y_train_wh[val_idx_wh]).unsqueeze(1).to(device)

model_wh = ResidualCorrector(lookback_steps).to(device)
opt_wh = torch.optim.Adam(model_wh.parameters(), lr=1e-3)
criterion = nn.MSELoss()

t0 = time.time()
best_val_loss_wh = float("inf")
patience, wait = 10, 0

for ep in range(30):
    model_wh.train()
    perm_b = torch.randperm(len(X_tr_wh))

    for i in range(0, len(X_tr_wh), 32):
        b = perm_b[i:i+32]
        opt_wh.zero_grad()
        # Split into physics pred and history
        phys_pred_wh = X_tr_wh[b, :1]
        hist_wh = X_tr_wh[b, 1:]
        y_pred = model_wh(phys_pred_wh, hist_wh)
        loss = criterion(y_pred, Y_tr_wh[b])
        loss.backward()
        opt_wh.step()

    model_wh.eval()
    with torch.no_grad():
        phys_pred_val_wh = X_val_wh[:, :1]
        hist_val_wh = X_val_wh[:, 1:]
        val_pred = model_wh(phys_pred_val_wh, hist_val_wh)
        val_loss = criterion(val_pred, Y_val_wh).item()

    if val_loss < best_val_loss_wh - 1e-6:
        best_val_loss_wh = val_loss
        wait = 0
        best_state_wh = {k: v.clone() for k, v in model_wh.state_dict().items()}
    else:
        wait += 1

    if wait >= patience:
        break

if 'best_state_wh' in locals():
    model_wh.load_state_dict(best_state_wh)

t_wh = time.time() - t0
print(f"  Wave height corrector trained in {t_wh:.1f}s")

# Train water temperature corrector (similar process)
n_val_wt = max(1, int(0.1 * len(X_train_wt)))
perm_wt = np.random.permutation(len(X_train_wt))
val_idx_wt, tr_idx_wt = perm_wt[:n_val_wt], perm_wt[n_val_wt:]

X_tr_wt = torch.from_numpy(X_train_wt[tr_idx_wt]).to(device)
Y_tr_wt = torch.from_numpy(Y_train_wt[tr_idx_wt]).unsqueeze(1).to(device)
X_val_wt = torch.from_numpy(X_train_wt[val_idx_wt]).to(device)
Y_val_wt = torch.from_numpy(Y_train_wt[val_idx_wt]).unsqueeze(1).to(device)

model_wt = ResidualCorrector(lookback_steps).to(device)
opt_wt = torch.optim.Adam(model_wt.parameters(), lr=1e-3)

t0 = time.time()
best_val_loss_wt = float("inf")
patience, wait = 10, 0

for ep in range(30):
    model_wt.train()
    perm_b = torch.randperm(len(X_tr_wt))

    for i in range(0, len(X_tr_wt), 32):
        b = perm_b[i:i+32]
        opt_wt.zero_grad()
        phys_pred_wt = X_tr_wt[b, :1]
        hist_wt = X_tr_wt[b, 1:]
        y_pred = model_wt(phys_pred_wt, hist_wt)
        loss = criterion(y_pred, Y_tr_wt[b])
        loss.backward()
        opt_wt.step()

    model_wt.eval()
    with torch.no_grad():
        phys_pred_val_wt = X_val_wt[:, :1]
        hist_val_wt = X_val_wt[:, 1:]
        val_pred = model_wt(phys_pred_val_wt, hist_val_wt)
        val_loss = criterion(val_pred, Y_val_wt).item()

    if val_loss < best_val_loss_wt - 1e-6:
        best_val_loss_wt = val_loss
        wait = 0
        best_state_wt = {k: v.clone() for k, v in model_wt.state_dict().items()}
    else:
        wait += 1

    if wait >= patience:
        break

if 'best_state_wt' in locals():
    model_wt.load_state_dict(best_state_wt)

t_wt = time.time() - t0
print(f"  Water temp corrector trained in {t_wt:.1f}s")

# ===== STEP 4: GENERATE FORECAST WITH PHYSICS + ML CORRECTION =====
print("\n[5/7] Generating 10-day forecasts with physics+ML hybrid...")

Y_true = df.iloc[test_start:test_start+forecast_steps].values

# Get test data
wind_speed_test = df.iloc[test_start:test_start+forecast_steps, idx_wind_speed].values
solar_test = df.iloc[test_start:test_start+forecast_steps, idx_solar].values
air_temp_test = df.iloc[test_start:test_start+forecast_steps, idx_air_temp].values
wave_height_test_true = df.iloc[test_start:test_start+forecast_steps, idx_wave_height].values
water_temp_test_true = df.iloc[test_start:test_start+forecast_steps, idx_water_temp].values

# Get last lookback window from training
last_wh_history = train_df.iloc[-lookback_steps:, idx_wave_height].values
last_wt_history = train_df.iloc[-lookback_steps:, idx_water_temp].values

# Generate physics predictions and apply ML corrections
wave_height_hybrid = np.zeros(forecast_steps)
wave_period_hybrid = np.zeros(forecast_steps)
water_temp_hybrid = np.zeros(forecast_steps)

water_temp_current = df.iloc[test_start - 1, idx_water_temp]
wh_history = last_wh_history.copy()
wt_history = last_wt_history.copy()

model_wh.eval()
model_wt.eval()

with torch.no_grad():
    for t in range(forecast_steps):
        # Wave height: physics + ML correction
        wh_physics = WavePhysics.estimate_wave_height(wind_speed_test[t])

        # ML residual correction for wave height
        X_wh_test = np.concatenate([[wh_physics], wh_history]).astype(np.float32)
        phys_pred_wh_t = torch.from_numpy(np.array([wh_physics])).float().to(device).unsqueeze(0)
        hist_wh_t = torch.from_numpy(wh_history).float().to(device).unsqueeze(0)
        residual_wh = model_wh(phys_pred_wh_t, hist_wh_t)[0, 0].cpu().numpy()

        wh_corrected = wh_physics + residual_wh
        wave_height_hybrid[t] = np.clip(wh_corrected, 0.0, 15.0)

        # Update history
        wh_history = np.concatenate([wh_history[1:], [wave_height_hybrid[t]]])

        # Wave period from corrected height
        wave_period_hybrid[t] = WavePhysics.estimate_wave_period(wave_height_hybrid[t])

        # Water temperature: physics + ML correction
        wt_physics = TempPhysics.estimate_water_temp(
            air_temp_test[t], solar_test[t], water_temp_current
        )

        # ML residual correction for water temp
        phys_pred_wt_t = torch.from_numpy(np.array([wt_physics])).float().to(device).unsqueeze(0)
        hist_wt_t = torch.from_numpy(wt_history).float().to(device).unsqueeze(0)
        residual_wt = model_wt(phys_pred_wt_t, hist_wt_t)[0, 0].cpu().numpy()

        wt_corrected = wt_physics + residual_wt
        water_temp_hybrid[t] = np.clip(wt_corrected, -2.0, 35.0)
        water_temp_current = water_temp_hybrid[t]

        # Update history
        wt_history = np.concatenate([wt_history[1:], [water_temp_hybrid[t]]])

print(f"[OK] Hybrid forecasts generated for all {forecast_steps} steps")

# ===== STEP 5: EVALUATE PERFORMANCE =====
print("\n[6/7] Evaluating hybrid physics+ML forecasts...")

results_hybrid = []

# Get baseline (persistence)
last_obs = df.iloc[test_start - 1].values

for day_num in range(1, forecast_days + 1):
    day_start = (day_num - 1) * 144
    day_end = day_num * 144

    Y_true_day = Y_true[day_start:day_end]
    Y_persist_day = np.tile(last_obs, (144, 1))

    mae_persist_day = mean_absolute_error(Y_true_day, Y_persist_day)

    # Calculate metrics for the 4 main poor parameters
    metrics_day = {}

    # Significant wave height
    mae_wh = mean_absolute_error(wave_height_test_true[day_start:day_end], wave_height_hybrid[day_start:day_end])
    skill_wh = (1 - mae_wh / mean_absolute_error(wave_height_test_true[day_start:day_end], Y_persist_day[:, idx_wave_height])) * 100
    metrics_day['wave_height'] = skill_wh

    # Significant wave period
    mae_wp = mean_absolute_error(Y_true_day[:, idx_wave_period], wave_period_hybrid[day_start:day_end])
    skill_wp = (1 - mae_wp / mean_absolute_error(Y_true_day[:, idx_wave_period], Y_persist_day[:, idx_wave_period])) * 100
    metrics_day['wave_period'] = skill_wp

    # Zero crossing period
    zero_cross_hybrid = wave_period_hybrid[day_start:day_end] * 0.713
    mae_zc = mean_absolute_error(Y_true_day[:, idx_zero_cross], zero_cross_hybrid)
    skill_zc = (1 - mae_zc / mean_absolute_error(Y_true_day[:, idx_zero_cross], Y_persist_day[:, idx_zero_cross])) * 100
    metrics_day['zero_cross'] = skill_zc

    # Water temperature
    mae_wt = mean_absolute_error(water_temp_test_true[day_start:day_end], water_temp_hybrid[day_start:day_end])
    skill_wt = (1 - mae_wt / mean_absolute_error(water_temp_test_true[day_start:day_end], Y_persist_day[:, idx_water_temp])) * 100
    metrics_day['water_temp'] = skill_wt

    # Overall
    mae_day = np.mean([mean_absolute_error(Y_true_day[:, i],
                       np.where(i == idx_wave_height, wave_height_hybrid[day_start:day_end],
                               np.where(i == idx_water_temp, water_temp_hybrid[day_start:day_end],
                                       np.where(i in [idx_wave_period, idx_zero_cross, idx_peak_wave],
                                               wave_period_hybrid[day_start:day_end],
                                               Y_persist_day[:, i])))) for i in range(len(all_params))])

    skill_day = (1 - mae_day / mae_persist_day) * 100 if mae_persist_day > 0 else 0

    print(f"DAY {day_num}: WH={metrics_day['wave_height']:+.1f}%, WP={metrics_day['wave_period']:+.1f}%, WT={metrics_day['water_temp']:+.1f}% | Overall={skill_day:+.1f}%")

    results_hybrid.append({
        'Day': day_num,
        'Overall_Skill_%': skill_day,
        'WaveHeight_Skill_%': metrics_day['wave_height'],
        'WavePeriod_Skill_%': metrics_day['wave_period'],
        'ZeroCross_Skill_%': metrics_day['zero_cross'],
        'WaterTemp_Skill_%': metrics_day['water_temp'],
    })

summary_hybrid = pd.DataFrame(results_hybrid)
summary_hybrid.to_csv("physics_ml_hybrid_10days_summary.csv", index=False)

# ===== STEP 6: FINAL SUMMARY =====
print("\n" + "="*80)
print("PHYSICS+ML HYBRID CORRECTION: PERFORMANCE SUMMARY")
print("="*80)

print("\n" + summary_hybrid[[
    'Day', 'Overall_Skill_%', 'WaveHeight_Skill_%', 'WavePeriod_Skill_%', 'WaterTemp_Skill_%'
]].to_string(index=False))

print("\n" + "="*80)
print("COMPARISON: 6 POOR PARAMETERS")
print("="*80)

avg_wh = summary_hybrid['WaveHeight_Skill_%'].mean()
avg_wp = summary_hybrid['WavePeriod_Skill_%'].mean()
avg_zc = summary_hybrid['ZeroCross_Skill_%'].mean()
avg_wt = summary_hybrid['WaterTemp_Skill_%'].mean()

print(f"\nPhysics+ML Hybrid Results:")
print(f"  significant_wave_height_m:    {avg_wh:+7.1f}%  (was -133.5% with Correlated MTGNN)")
print(f"  significant_wave_period_s:    {avg_wp:+7.1f}%  (was  -92.3% with Correlated MTGNN)")
print(f"  zero_crossing_period_s:       {avg_zc:+7.1f}%  (was  -30.8% with Correlated MTGNN)")
print(f"  water_temp_c:                 {avg_wt:+7.1f}%  (was  +62.9% with Correlated MTGNN)")

print(f"\n{'='*80}")
print(f"Files saved: physics_ml_hybrid_10days_summary.csv")
print(f"{'='*80}\n")

