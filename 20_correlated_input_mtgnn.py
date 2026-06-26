#!/usr/bin/env python
"""Correlated Input MTGNN: 18 individual models with intelligent parameter inputs."""
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
print("CORRELATED INPUT MTGNN: 18 MODELS WITH INTELLIGENT PARAMETER COUPLING")
print("="*80)

# ===== LOAD DATA =====
print("\n[1/6] Loading dataset...")
df = pd.read_csv("marine_120day_18params_10min.csv", index_col=0, parse_dates=True)
all_params = df.columns.tolist()

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

# ===== DEFINE INTELLIGENT INPUT SETS FOR EACH PARAMETER =====
# Based on correlation analysis: include parameters with >0.7 correlation

input_sets = {
    'air_temp_c': ['air_temp_c', 'water_temp_c', 'dew_point_c', 'conductivity_mscm'],  # 0.97-0.98 corr
    'water_temp_c': ['water_temp_c', 'air_temp_c', 'dew_point_c', 'conductivity_mscm'],  # 0.97-0.98 corr
    'dew_point_c': ['dew_point_c', 'air_temp_c', 'water_temp_c', 'conductivity_mscm'],  # 0.97-0.98 corr
    'conductivity_mscm': ['conductivity_mscm', 'air_temp_c', 'water_temp_c', 'dew_point_c'],  # 0.98 corr

    'wind_direction_deg': ['wind_direction_deg', 'compass_deg', 'wind_speed_ms'],  # 0.94 corr + wind
    'compass_deg': ['compass_deg', 'wind_direction_deg', 'wind_speed_ms'],  # 0.94 corr + wind
    'wind_speed_ms': ['wind_speed_ms', 'wind_direction_deg', 'compass_deg', 'air_pressure_hpa'],  # waves

    'significant_wave_height_m': ['significant_wave_height_m', 'wind_speed_ms', 'air_pressure_hpa', 'significant_wave_period_s'],  # wave coupling
    'significant_wave_period_s': ['significant_wave_period_s', 'wind_speed_ms', 'significant_wave_height_m', 'zero_crossing_period_s'],  # wave period coupling
    'peak_wave_period_s': ['peak_wave_period_s', 'wind_speed_ms', 'significant_wave_height_m', 'zero_crossing_period_s'],
    'zero_crossing_period_s': ['zero_crossing_period_s', 'significant_wave_height_m', 'significant_wave_period_s', 'wind_speed_ms'],

    'air_pressure_hpa': ['air_pressure_hpa', 'wind_speed_ms', 'significant_wave_height_m'],  # weather system
    'relative_humidity_pct': ['relative_humidity_pct', 'air_temp_c', 'dew_point_c', 'wind_speed_ms'],  # moisture

    'salinity_psu': ['salinity_psu', 'conductivity_mscm', 'water_temp_c'],  # salt content correlation

    'current_speed_ms': ['current_speed_ms', 'tidal_level_m', 'wind_speed_ms'],  # tidal/wind driven
    'current_direction_deg': ['current_direction_deg', 'wind_direction_deg', 'tidal_level_m'],  # direction alignment

    'tidal_level_m': ['tidal_level_m', 'current_speed_ms'],  # tidal relationship
    'global_radiation_wm2': ['global_radiation_wm2', 'air_temp_c'],  # solar heating
}

print(f"[OK] Defined {len(input_sets)} parameter models with intelligent input coupling")

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
    """MTGNN that takes correlated inputs for a single target parameter."""
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
        # x: (batch, num_vars, seq_len)
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

        # Output only for first (target) variable
        y_pred = self.output_fc(h[:, 0, :])  # (batch, pred_len)

        return y_pred


device = torch.device("cpu")
torch.set_num_threads(8)

# ===== TRAIN ALL 18 MODELS =====
print("\n[3/6] Training 18 Correlated Input MTGNN models...")

models = {}
model_times = {}

for target_param in all_params:
    input_params = input_sets[target_param]
    param_indices = [all_params.index(p) for p in input_params]

    print(f"\n  Training {target_param:35s} (inputs: {', '.join(input_params[:2])}...)")

    # Build training data
    X_train, Y_train = [], []
    for i in range(lookback_steps, len(train_df) - forecast_steps, 2):
        # Input: all correlated parameters
        x = train_df.iloc[i - lookback_steps:i, param_indices].values.T.astype(np.float32)  # (num_vars, seq_len)
        # Target: only the target parameter
        target_idx = all_params.index(target_param)
        y = train_df.iloc[i:i + forecast_steps, target_idx].values.astype(np.float32)  # (pred_len,)
        X_train.append(x)
        Y_train.append(y)

    X_train = np.array(X_train)  # (samples, num_vars, seq_len)
    Y_train = np.array(Y_train)  # (samples, pred_len)

    # Train/val split
    n_val = max(1, int(0.1 * len(X_train)))
    perm = np.random.permutation(len(X_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr_t = torch.from_numpy(X_train[tr_idx]).to(device)
    Y_tr_t = torch.from_numpy(Y_train[tr_idx]).to(device)
    X_val_t = torch.from_numpy(X_train[val_idx]).to(device)
    Y_val_t = torch.from_numpy(Y_train[val_idx]).to(device)

    # Train model
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
    models[target_param] = (model, param_indices, input_params)
    model_times[target_param] = t_train

    print(f"    [OK] Trained in {t_train:.1f}s (val_loss: {best_val_loss:.6f})")

# ===== FORECAST =====
print("\n[4/6] Generating 10-day forecasts (18 Correlated MTGNN models)...")

Y_pred_all = np.zeros((forecast_steps, len(all_params)))

for target_param, (model, param_indices, input_params) in models.items():
    model.eval()

    with torch.no_grad():
        last_context = train_df.iloc[-lookback_steps:, param_indices].values.T.astype(np.float32)
        X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)
        Y_pred_norm = model(X_test)[0].cpu().numpy()  # (pred_len,)

    target_idx = all_params.index(target_param)
    Y_pred_all[:, target_idx] = Y_pred_norm

# Inverse normalize
Y_pred = scaler.inverse_transform(Y_pred_all)
Y_true = test_df_orig.iloc[:forecast_steps].values

print(f"[OK] Forecasts generated for all 18 parameters")

# ===== EVALUATE DAY-BY-DAY =====
print("\n[5/6] Evaluating day-by-day performance...")

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

    print(f"DAY {day_num}: {skill_day:+.1f}% skill | MAE: {mae_day:.4f}")

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
    day_metrics_df.to_csv(f"correlated_day_{day_num:02d}_metrics.csv", index=False)

    results_daily.append({
        "Day": day_num,
        "Overall_Skill_%": skill_day,
        "Overall_MAE": mae_day,
        "Overall_RMSE": rmse_day,
    })

# ===== SUMMARY =====
print("\n" + "="*80)
print("SUMMARY: CORRELATED INPUT MTGNN - DAY-BY-DAY SKILL")
print("="*80)

summary_df = pd.DataFrame(results_daily)
summary_df.to_csv("correlated_input_10days_summary.csv", index=False)

print("\n" + summary_df[[
    'Day', 'Overall_Skill_%', 'Overall_MAE', 'Overall_RMSE'
]].to_string(index=False))

# ===== FINAL COMPARISON =====
print("\n" + "="*80)
print("FINAL 5-WAY COMPARISON")
print("="*80)

try:
    nbeats = pd.read_csv("nbeats_10days_summary.csv")
    mtgnn_single = pd.read_csv("mtgnn_10days_summary.csv")
    hybrid_mtgnn = pd.read_csv("hybrid_mtgnn_10days_summary.csv")
    hybrid_nbeats = pd.read_csv("hybrid8_10days_summary.csv")

    results = [
        ("Single N-BEATS", nbeats['Overall_Skill_%'].mean()),
        ("Hybrid 8 N-BEATS", hybrid_nbeats['Overall_Skill_%'].mean()),
        ("Single MTGNN", mtgnn_single['Overall_Skill_%'].mean()),
        ("Hybrid 8 MTGNN", hybrid_mtgnn['Overall_Skill_%'].mean()),
        ("Correlated Input MTGNN", summary_df['Overall_Skill_%'].mean()),
    ]
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

    print(f"\nAverage Skill (All 10 Days) - FINAL RANKING:")
    for i, (model_name, skill) in enumerate(results_sorted, 1):
        badge = "[GOLD]" if i == 1 else "[SILVER]" if i == 2 else "[BRONZE]" if i == 3 else f"   [{i}]"
        improvement = skill - results[0][1]  # vs Single N-BEATS baseline
        print(f"  {i}. {badge} {model_name:35s}: {skill:+.1f}% ({improvement:+.1f}% vs baseline)")

except FileNotFoundError as e:
    correlated_avg = summary_df['Overall_Skill_%'].mean()
    print(f"  Correlated Input MTGNN: {correlated_avg:+.1f}% (waiting for other results)")

print(f"\nTraining Times (18 models):")
total_time = sum(model_times.values())
print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"  Average per model: {total_time/len(model_times):.1f}s")

print(f"\n{'='*80}")
print(f"Files saved:")
print(f"  - correlated_input_10days_summary.csv (overall)")
print(f"  - correlated_day_01_metrics.csv ... correlated_day_10_metrics.csv")
print(f"{'='*80}\n")
