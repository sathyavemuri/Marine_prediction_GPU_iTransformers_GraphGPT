#!/usr/bin/env python
"""Residual Correction Stacking: Learn systematic biases and add them back."""
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
print("RESIDUAL CORRECTION STACKING")
print("="*80)
print("Two-stage approach:")
print("  Stage 1: Correlated Input MTGNN predictions")
print("  Stage 2: Learn and add residual corrections")
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
test_df_orig = df.iloc[test_start:].copy()

print(f"[OK] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ===== INPUT SETS =====
input_sets = {
    'air_temp_c': ['air_temp_c', 'water_temp_c', 'dew_point_c', 'conductivity_mscm'],
    'water_temp_c': ['water_temp_c', 'air_temp_c', 'dew_point_c', 'conductivity_mscm'],
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
    'salinity_psu': ['salinity_psu', 'conductivity_mscm', 'water_temp_c'],
    'current_speed_ms': ['current_speed_ms', 'tidal_level_m', 'wind_speed_ms'],
    'current_direction_deg': ['current_direction_deg', 'wind_direction_deg', 'tidal_level_m'],
    'tidal_level_m': ['tidal_level_m', 'current_speed_ms'],
    'global_radiation_wm2': ['global_radiation_wm2', 'air_temp_c'],
}

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


# ===== RESIDUAL CORRECTOR: LEARNS BIAS FROM INPUTS + BASE PREDICTION =====
class ResidualCorrector(nn.Module):
    """Learns systematic bias/residuals from input features + base MTGNN prediction."""
    def __init__(self, num_input_vars, seq_len, pred_len, hidden_dim=64):
        super().__init__()
        # Input: (seq_len * num_input_vars) + pred_len (base MTGNN predictions)
        self.input_dim = seq_len * num_input_vars + pred_len
        self.pred_len = pred_len

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, pred_len)  # Output: residual for each timestep

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, x_input, y_base):
        # x_input: (batch, seq_len * num_input_vars) - flattened input history
        # y_base: (batch, pred_len) - base MTGNN predictions

        # Concatenate input features with base predictions
        x = torch.cat([x_input, y_base], dim=-1)  # (batch, input_dim + pred_len)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        residual = self.fc3(x)  # (batch, pred_len)
        return residual


device = torch.device("cpu")
torch.set_num_threads(8)

# ===== STAGE 1: TRAIN BASE CORRELATED INPUT MTGNN MODELS =====
print("\n[2/7] Training Stage 1: Base Correlated Input MTGNN models...")

mtgnn_models = {}
mtgnn_times = {}

for target_param in all_params:
    input_params = input_sets[target_param]
    param_indices = [all_params.index(p) for p in input_params]

    # Build training data
    X_train, Y_train = [], []
    for i in range(lookback_steps, len(train_df) - forecast_steps, 2):
        x = train_df.iloc[i - lookback_steps:i, param_indices].values.T.astype(np.float32)
        target_idx = all_params.index(target_param)
        y = train_df.iloc[i:i + forecast_steps, target_idx].values.astype(np.float32)
        X_train.append(x)
        Y_train.append(y)

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
        seq_len=lookback_steps,
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
    mtgnn_models[target_param] = (model, param_indices, input_params, X_train, Y_train)
    mtgnn_times[target_param] = t_train

print(f"[OK] Base MTGNN models trained")

# ===== STAGE 2: GENERATE BASE PREDICTIONS AND COMPUTE RESIDUALS =====
print("\n[3/7] Generating base predictions on training data...")

residual_data = {}  # Store training data for residual correctors

for target_param, (model, param_indices, input_params, X_train, Y_train) in mtgnn_models.items():
    model.eval()

    X_train_t = torch.from_numpy(X_train).to(device)

    with torch.no_grad():
        Y_pred_base = model(X_train_t).cpu().numpy()  # (num_samples, pred_len)

    # Compute residuals
    Y_residual = Y_train - Y_pred_base  # (num_samples, pred_len)

    # Store for residual corrector training
    # Input for residual corrector: flattened input history + base prediction
    X_input_flat = X_train.reshape(len(X_train), -1)  # (num_samples, seq_len * num_vars)

    residual_data[target_param] = {
        'X_input_flat': X_input_flat,
        'Y_base_pred': Y_pred_base,
        'Y_residual': Y_residual,
    }

print(f"[OK] Base predictions generated, residuals computed")

# ===== STAGE 3: TRAIN RESIDUAL CORRECTORS =====
print("\n[4/7] Training Stage 2: Residual Corrector models...")

residual_models = {}
residual_times = {}

for target_param, data in residual_data.items():
    X_input_flat = data['X_input_flat'].astype(np.float32)
    Y_base = data['Y_base_pred'].astype(np.float32)
    Y_residual = data['Y_residual'].astype(np.float32)

    # Train/val split
    n_val = max(1, int(0.1 * len(X_input_flat)))
    perm = np.random.permutation(len(X_input_flat))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_in_tr = torch.from_numpy(X_input_flat[tr_idx]).to(device)
    Y_b_tr = torch.from_numpy(Y_base[tr_idx]).to(device)
    Y_res_tr = torch.from_numpy(Y_residual[tr_idx]).to(device)

    X_in_val = torch.from_numpy(X_input_flat[val_idx]).to(device)
    Y_b_val = torch.from_numpy(Y_base[val_idx]).to(device)
    Y_res_val = torch.from_numpy(Y_residual[val_idx]).to(device)

    # Create residual corrector
    model = ResidualCorrector(
        num_input_vars=len(input_sets[target_param]),
        seq_len=lookback_steps,
        pred_len=forecast_steps,
        hidden_dim=64
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    t0 = time.time()
    best_val_loss, best_state = float("inf"), None
    patience, wait = 15, 0

    for ep in range(50):
        model.train()
        perm_b = torch.randperm(len(X_in_tr))

        for i in range(0, len(X_in_tr), 32):
            b = perm_b[i:i+32]
            opt.zero_grad()
            res_pred = model(X_in_tr[b], Y_b_tr[b])
            loss = criterion(res_pred, Y_res_tr[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            res_pred_val = model(X_in_val, Y_b_val)
            val_loss = criterion(res_pred_val, Y_res_val).item()

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
    residual_models[target_param] = model
    residual_times[target_param] = t_train

print(f"[OK] Residual correctors trained")

# ===== STAGE 4: GENERATE TEST PREDICTIONS WITH CORRECTION =====
print("\n[5/7] Generating test forecasts with residual correction...")

Y_pred_all_corrected = np.zeros((forecast_steps, len(all_params)))

for target_param, mtgnn_model in [(p, mtgnn_models[p][0]) for p in all_params]:
    param_indices = input_sets[target_param]
    param_indices = [all_params.index(p) for p in param_indices]

    mtgnn_model.eval()
    residual_model = residual_models[target_param]
    residual_model.eval()

    with torch.no_grad():
        # Base MTGNN prediction
        last_context = train_df.iloc[-lookback_steps:, param_indices].values.T.astype(np.float32)
        X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)
        Y_pred_base = mtgnn_model(X_test)[0].cpu().numpy()  # (pred_len,)

        # Residual correction
        X_input_flat_test = last_context.reshape(-1).astype(np.float32)  # Flattened
        X_flat_t = torch.from_numpy(X_input_flat_test).unsqueeze(0).to(device)  # (1, input_dim)
        Y_base_t = torch.from_numpy(Y_pred_base).unsqueeze(0).to(device)  # (1, pred_len)

        residual_pred = residual_model(X_flat_t, Y_base_t)[0].cpu().numpy()  # (pred_len,)

        # Final prediction = base + residual
        Y_pred_corrected = Y_pred_base + residual_pred

    target_idx = all_params.index(target_param)
    Y_pred_all_corrected[:, target_idx] = Y_pred_corrected

# Inverse normalize
Y_pred = scaler.inverse_transform(Y_pred_all_corrected)
Y_true = test_df_orig.iloc[:forecast_steps].values

print(f"[OK] Test forecasts generated with corrections")

# ===== STAGE 5: EVALUATE =====
print("\n[6/7] Evaluating performance...")

results_corrected = []

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
    day_metrics_df.to_csv(f"residual_corrected_day_{day_num:02d}_metrics.csv", index=False)

    results_corrected.append({
        "Day": day_num,
        "Overall_Skill_%": skill_day,
        "Overall_MAE": mae_day,
        "Overall_RMSE": rmse_day,
    })

    print(f"DAY {day_num}: {skill_day:+.1f}% skill")

# ===== SUMMARY =====
print("\n" + "="*80)
print("RESIDUAL CORRECTION STACKING: RESULTS")
print("="*80)

summary_df = pd.DataFrame(results_corrected)
summary_df.to_csv("residual_corrected_10days_summary.csv", index=False)

print("\n" + summary_df[[
    'Day', 'Overall_Skill_%', 'Overall_MAE', 'Overall_RMSE'
]].to_string(index=False))

# Comparison
original_skill = 84.96
corrected_skill = summary_df['Overall_Skill_%'].mean()

print("\n" + "="*80)
print("COMPARISON: ORIGINAL vs RESIDUAL-CORRECTED")
print("="*80)

print(f"\nOriginal Correlated MTGNN:        {original_skill:+.1f}%")
print(f"Residual-Corrected MTGNN:        {corrected_skill:+.1f}%")
print(f"Improvement:                      {corrected_skill - original_skill:+.2f}%")

# Check 6 poor parameters
print("\nKey parameters (6 originally-poor) - Day 1 results:")
df_day1 = pd.read_csv("residual_corrected_day_01_metrics.csv")
for param in ['water_temp_c', 'salinity_psu', 'significant_wave_height_m', 'significant_wave_period_s', 'zero_crossing_period_s', 'peak_wave_period_s']:
    skill = df_day1[df_day1['Parameter'] == param]['Skill_%'].values
    if len(skill) > 0:
        print(f"  {param:40s}: {skill[0]:+7.1f}%")

print(f"\n{'='*80}")
print(f"Training times:")
print(f"  Stage 1 (MTGNN): {sum(mtgnn_times.values()):.1f}s")
print(f"  Stage 2 (Residual correctors): {sum(residual_times.values()):.1f}s")
print(f"  Total: {sum(mtgnn_times.values()) + sum(residual_times.values()):.1f}s")
print(f"\nFiles saved:")
print(f"  - residual_corrected_10days_summary.csv")
print(f"  - residual_corrected_day_01_metrics.csv ... day_10_metrics.csv")
print(f"{'='*80}\n")

