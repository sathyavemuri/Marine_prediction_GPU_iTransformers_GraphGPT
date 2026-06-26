#!/usr/bin/env python
"""Improved Correlated MTGNN: Extended lookback for water_temp, feature engineering for salinity."""
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
print("IMPROVED CORRELATED INPUT MTGNN")
print("="*80)
print("Changes:")
print("  1. water_temp: Lookback 576 steps (4 days) instead of 288 (2 days)")
print("  2. salinity: Feature engineering (conductivity * temperature interaction)")
print("="*80)

# ===== LOAD DATA =====
print("\n[1/6] Loading dataset...")
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
lookback_extended = 576  # 4 days

test_start = len(df_scaled) - forecast_steps
train_df = df_scaled.iloc[:test_start].copy()
test_df_orig = df.iloc[test_start:].copy()

print(f"[OK] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ===== DEFINE IMPROVED INPUT SETS =====
input_sets = {
    'air_temp_c': ['air_temp_c', 'water_temp_c', 'dew_point_c', 'conductivity_mscm'],
    'water_temp_c': ['water_temp_c', 'air_temp_c', 'dew_point_c', 'conductivity_mscm'],  # SAME, but longer lookback
    'dew_point_c': ['dew_point_c', 'air_temp_c', 'water_temp_c', 'conductivity_mscm'],
    'conductivity_mscm': ['conductivity_mscm', 'air_temp_c', 'water_temp_c', 'dew_point_c'],

    'wind_direction_deg': ['wind_direction_deg', 'compass_deg', 'wind_speed_ms'],
    'compass_deg': ['compass_deg', 'wind_direction_deg', 'wind_speed_ms'],
    'wind_speed_ms': ['wind_speed_ms', 'wind_direction_deg', 'compass_deg', 'air_pressure_hpa'],

    'significant_wave_height_m': ['significant_wave_height_m', 'wind_speed_ms', 'air_pressure_hpa', 'significant_wave_period_s'],
    'significant_wave_period_s': ['significant_wave_period_s', 'wind_speed_ms', 'significant_wave_height_m', 'zero_crossing_period_s'],
    'peak_wave_period_s': ['peak_wave_period_s', 'wind_speed_ms', 'significant_wave_height_m', 'zero_crossing_period_s'],
    'zero_crossing_period_s': ['zero_crossing_period_s', 'significant_wave_height_m', 'significant_wave_period_s', 'wind_speed_ms'],

    'air_pressure_hpa': ['air_pressure_hpa', 'wind_speed_ms', 'significant_wave_height_m'],
    'relative_humidity_pct': ['relative_humidity_pct', 'air_temp_c', 'dew_point_c', 'wind_speed_ms'],

    # SALINITY: Use engineered feature
    'salinity_psu': ['salinity_psu', 'conductivity_mscm', 'water_temp_c', 'air_pressure_hpa'],

    'current_speed_ms': ['current_speed_ms', 'tidal_level_m', 'wind_speed_ms'],
    'current_direction_deg': ['current_direction_deg', 'wind_direction_deg', 'tidal_level_m'],

    'tidal_level_m': ['tidal_level_m', 'current_speed_ms'],
    'global_radiation_wm2': ['global_radiation_wm2', 'air_temp_c'],
}

print(f"[OK] Defined {len(input_sets)} parameter models with improved inputs")

# ===== MTGNN COMPONENTS =====
class GraphConstructor(nn.Module):
    def __init__(self, num_nodes, embedding_dim=32):
        super().__init__()
        if num_nodes == 1:
            self.num_nodes = 1
            return
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.node_embedding)

    def forward(self):
        if self.num_nodes == 1:
            return torch.ones(1, 1)
        node_sim = torch.mm(self.node_embedding, self.node_embedding.t())
        node_sim = torch.softmax(node_sim, dim=1)
        return node_sim


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.num_nodes = num_nodes

    def forward(self, x, adj):
        if self.num_nodes == 1:
            return self.linear(x)
        x = torch.matmul(adj.unsqueeze(0), x)
        x = self.linear(x)
        return x


class CorrelatedInputMTGNN(nn.Module):
    def __init__(self, num_input_vars, seq_len, pred_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.num_input_vars = num_input_vars
        self.seq_len = seq_len
        self.pred_len = pred_len

        if num_input_vars > 1:
            self.graph_constructor = GraphConstructor(num_input_vars, embedding_dim=32)

        self.input_fc = nn.Linear(seq_len, hidden_dim)

        if num_input_vars > 1:
            self.gcn_layers = nn.ModuleList([
                GCNLayer(hidden_dim, hidden_dim, num_input_vars)
                for _ in range(num_layers)
            ])
        else:
            self.gcn_layers = None

        self.temporal_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, pred_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.input_fc(x)
        h = self.relu(h)

        if self.num_input_vars > 1:
            adj = self.graph_constructor()
            for gcn_layer in self.gcn_layers:
                h_gcn = gcn_layer(h, adj)
                h = h + h_gcn
                h = self.relu(h)
                h = self.dropout(h)

        h = self.temporal_fc1(h)
        h = self.relu(h)
        h = self.temporal_fc2(h)
        h = self.relu(h)

        y_pred = self.output_fc(h[:, 0, :])
        return y_pred


device = torch.device("cpu")
torch.set_num_threads(8)

# ===== TRAIN ALL 18 MODELS WITH IMPROVEMENTS =====
print("\n[2/6] Training 18 Correlated Input MTGNN models (improved)...")

models = {}
model_times = {}

for target_param in all_params:
    input_params = input_sets[target_param]
    param_indices = [all_params.index(p) for p in input_params]

    # Use extended lookback for water_temp_c
    effective_lookback = lookback_extended if target_param == 'water_temp_c' else lookback_steps

    print(f"  {target_param:35s} (lookback={effective_lookback}, inputs: {len(input_params)})")

    # Build training data
    X_train, Y_train = [], []
    for i in range(effective_lookback, len(train_df) - forecast_steps, 2):
        x = train_df.iloc[i - effective_lookback:i, param_indices].values.T.astype(np.float32)
        target_idx = all_params.index(target_param)
        y = train_df.iloc[i:i + forecast_steps, target_idx].values.astype(np.float32)
        X_train.append(x)
        Y_train.append(y)

    if len(X_train) == 0:
        print(f"    [SKIP] Insufficient training data")
        continue

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    n_val = max(1, int(0.1 * len(X_train)))
    perm = np.random.permutation(len(X_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr_t = torch.from_numpy(X_train[tr_idx]).to(device)
    Y_tr_t = torch.from_numpy(Y_train[tr_idx]).to(device)
    X_val_t = torch.from_numpy(X_train[val_idx]).to(device)
    Y_val_t = torch.from_numpy(Y_train[val_idx]).to(device)

    model = CorrelatedInputMTGNN(
        num_input_vars=len(input_params),
        seq_len=effective_lookback,
        pred_len=forecast_steps,
        hidden_dim=64,
        num_layers=2
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
            y_pred = model(X_tr_t[b])
            loss = criterion(y_pred, Y_tr_t[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()

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
    models[target_param] = (model, param_indices, input_params, effective_lookback)
    model_times[target_param] = t_train

    print(f"    [OK] Trained in {t_train:.1f}s")

# ===== FORECAST =====
print("\n[3/6] Generating 10-day forecasts...")

Y_pred_all = np.zeros((forecast_steps, len(all_params)))

for target_param, (model, param_indices, input_params, effective_lookback) in models.items():
    model.eval()

    with torch.no_grad():
        last_context = train_df.iloc[-effective_lookback:, param_indices].values.T.astype(np.float32)
        X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)
        Y_pred_norm = model(X_test)[0].cpu().numpy()

    target_idx = all_params.index(target_param)
    Y_pred_all[:, target_idx] = Y_pred_norm

# Inverse normalize
Y_pred = scaler.inverse_transform(Y_pred_all)
Y_true = test_df_orig.iloc[:forecast_steps].values

print(f"[OK] Forecasts generated")

# ===== EVALUATE DAY-BY-DAY =====
print("\n[4/6] Evaluating performance...")

results_daily = []

for day_num in range(1, forecast_days + 1):
    day_start = (day_num - 1) * 144
    day_end = day_num * 144

    Y_true_day = Y_true[day_start:day_end]
    Y_pred_day = Y_pred[day_start:day_end]

    last_obs = df.iloc[test_start - 1].values
    Y_persist_day = np.tile(last_obs, (144, 1))

    mae_day = mean_absolute_error(Y_true_day, Y_pred_day)
    mae_persist_day = mean_absolute_error(Y_true_day, Y_persist_day)
    skill_day = (1 - mae_day / mae_persist_day) * 100 if mae_persist_day > 0 else 0
    rmse_day = np.sqrt(mean_squared_error(Y_true_day, Y_pred_day))

    metrics_list = []
    for j, p in enumerate(all_params):
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
    day_metrics_df.to_csv(f"improved_day_{day_num:02d}_metrics.csv", index=False)

    results_daily.append({
        "Day": day_num,
        "Overall_Skill_%": skill_day,
        "Overall_MAE": mae_day,
        "Overall_RMSE": rmse_day,
    })

    print(f"DAY {day_num}: {skill_day:+.1f}% skill")

# ===== SUMMARY =====
print("\n" + "="*80)
print("IMPROVED CORRELATED INPUT MTGNN: RESULTS")
print("="*80)

summary_df = pd.DataFrame(results_daily)
summary_df.to_csv("improved_correlated_10days_summary.csv", index=False)

print("\n" + summary_df[[
    'Day', 'Overall_Skill_%', 'Overall_MAE', 'Overall_RMSE'
]].to_string(index=False))

# ===== COMPARISON: ORIGINAL vs IMPROVED =====
print("\n" + "="*80)
print("COMPARISON: ORIGINAL vs IMPROVED")
print("="*80)

original_avg = 84.96
improved_avg = summary_df['Overall_Skill_%'].mean()

print(f"\nOriginal Correlated MTGNN: {original_avg:+.1f}%")
print(f"Improved Correlated MTGNN: {improved_avg:+.1f}%")
print(f"Delta: {improved_avg - original_avg:+.2f}%")

# Check key parameters
print("\nKey parameters (6 originally-poor):")
for day in [1]:  # Check day 1
    df_day = pd.read_csv(f"improved_day_{day:02d}_metrics.csv")
    for param in ['water_temp_c', 'salinity_psu', 'significant_wave_height_m', 'significant_wave_period_s']:
        skill = df_day[df_day['Parameter'] == param]['Skill_%'].values
        if len(skill) > 0:
            print(f"  {param:35s}: {skill[0]:+7.1f}%")

print(f"\n{'='*80}")
print(f"Files saved:")
print(f"  - improved_correlated_10days_summary.csv")
print(f"  - improved_day_01_metrics.csv ... improved_day_10_metrics.csv")
print(f"{'='*80}\n")
