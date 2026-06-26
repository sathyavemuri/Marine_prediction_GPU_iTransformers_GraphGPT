#!/usr/bin/env python
"""MTGNN (Graph Neural Network) for 18-parameter marine forecasting."""
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n" + "="*80)
print("MTGNN (GRAPH NEURAL NETWORK): 110 DAYS TRAIN -> 10 DAYS FORECAST")
print("="*80)

# ===== LOAD DATA =====
print("\n[1/5] Loading dataset...")
df = pd.read_csv("marine_120day_18params_10min.csv", index_col=0, parse_dates=True)
params = df.columns.tolist()
print(f"[OK] Loaded: {df.shape[0]} rows, {df.shape[1]} columns (18 parameters)")

# ===== STANDARDIZE =====
print("\n[2/5] Standardizing...")
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[:] = scaler.fit_transform(df)

# Parameters
train_days = 110
forecast_days = 10
train_steps = train_days * 144
forecast_steps = forecast_days * 144
lookback_steps = 288
n_vars = len(params)

print(f"[OK] Training: {train_days} days ({train_steps} steps)")
print(f"     Forecast: {forecast_days} days ({forecast_steps} steps)")
print(f"     Variables: {n_vars} parameters")

# Split data
test_start = len(df_scaled) - forecast_steps
train_df = df_scaled.iloc[:test_start].copy()
test_df_orig = df.iloc[test_start:].copy()

# ===== SIMPLIFIED MTGNN =====
class GraphConstructor(nn.Module):
    """Learn the dependency graph between variables."""
    def __init__(self, num_nodes, embedding_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        # Learn node embeddings to capture variable characteristics
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.node_embedding)

    def forward(self):
        # Compute correlation matrix between node embeddings
        # Shape: (num_nodes, num_nodes)
        node_sim = torch.mm(self.node_embedding, self.node_embedding.t())
        node_sim = torch.softmax(node_sim, dim=1)
        return node_sim


class GCNLayer(nn.Module):
    """Graph Convolutional Layer."""
    def __init__(self, in_features, out_features, num_nodes):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.num_nodes = num_nodes

    def forward(self, x, adj):
        # x: (batch, num_nodes, features)
        # adj: (num_nodes, num_nodes)

        # Apply adjacency matrix: AX
        x = torch.matmul(adj.unsqueeze(0), x)  # (batch, num_nodes, features)

        # Linear transformation
        x = self.linear(x)

        return x


class MTGNN_Simple(nn.Module):
    """Simplified MTGNN for marine multivariate forecasting."""
    def __init__(self, num_nodes, seq_len, pred_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Graph constructor: learns dependencies between variables
        self.graph_constructor = GraphConstructor(num_nodes, embedding_dim=32)

        # Input embedding
        self.input_fc = nn.Linear(seq_len, hidden_dim)

        # GCN layers for capturing variable relationships
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, num_nodes)
            for _ in range(num_layers)
        ])

        # Temporal processing layers
        self.temporal_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.output_fc = nn.Linear(hidden_dim, pred_len)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, num_nodes, seq_len)
        batch_size = x.shape[0]

        # Embed input sequences
        h = self.input_fc(x)  # (batch, num_nodes, hidden_dim)
        h = self.relu(h)

        # Learn and apply adjacency matrix (variable dependencies)
        adj = self.graph_constructor()  # (num_nodes, num_nodes)

        # Apply GCN layers to capture variable relationships
        for gcn_layer in self.gcn_layers:
            h_gcn = gcn_layer(h, adj)  # (batch, num_nodes, hidden_dim)
            h = h + h_gcn  # Residual connection
            h = self.relu(h)
            h = self.dropout(h)

        # Temporal processing
        h = self.temporal_fc1(h)
        h = self.relu(h)
        h = self.temporal_fc2(h)
        h = self.relu(h)

        # Output: predict future values
        y_pred = self.output_fc(h)  # (batch, num_nodes, pred_len)

        return y_pred


device = torch.device("cpu")
torch.set_num_threads(8)

# ===== BUILD TRAINING DATA =====
print("\n[3/5] Building training samples...")
X_train, Y_train = [], []

for i in range(lookback_steps, len(train_df) - forecast_steps, 2):
    # Shape: (num_nodes/params, seq_len)
    x = train_df.iloc[i - lookback_steps:i].values.T.astype(np.float32)
    # Shape: (num_nodes/params, pred_len)
    y = train_df.iloc[i:i + forecast_steps].values.T.astype(np.float32)
    X_train.append(x)
    Y_train.append(y)

X_train = np.array(X_train)  # (num_samples, num_nodes, seq_len)
Y_train = np.array(Y_train)  # (num_samples, num_nodes, pred_len)

print(f"[OK] Built {len(X_train)} training samples")
print(f"     Shape: X={X_train.shape}, Y={Y_train.shape}")

# Train/val split
n_val = max(1, int(0.1 * len(X_train)))
perm = np.random.permutation(len(X_train))
val_idx, tr_idx = perm[:n_val], perm[n_val:]

X_tr_t = torch.from_numpy(X_train[tr_idx]).to(device)
Y_tr_t = torch.from_numpy(Y_train[tr_idx]).to(device)
X_val_t = torch.from_numpy(X_train[val_idx]).to(device)
Y_val_t = torch.from_numpy(Y_train[val_idx]).to(device)

# ===== TRAIN =====
print("\n[4/5] Training MTGNN...")

model = MTGNN_Simple(
    num_nodes=n_vars,
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

    if (ep + 1) % 10 == 0 or wait >= patience:
        print(f"  Epoch {ep+1:2d}/50 | Val: {val_loss:.6f} | Wait: {wait}/15")

    if wait >= patience:
        print(f"  -> Early stop at epoch {ep+1}")
        break

if best_state:
    model.load_state_dict(best_state)

t_train = time.time() - t0

# ===== FORECAST =====
print("\n[5/5] Forecasting 10 days ahead...")

model.eval()
t0 = time.time()

with torch.no_grad():
    # Shape: (1, num_nodes, seq_len)
    last_context = train_df.iloc[-lookback_steps:].values.T.astype(np.float32)
    X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)
    # Shape: (1, num_nodes, pred_len)
    Y_pred_norm_t = model(X_test)
    Y_pred_norm = Y_pred_norm_t[0].cpu().numpy()  # (num_nodes, pred_len)

t_infer = time.time() - t0

# Transpose and inverse normalize
# Y_pred_norm: (num_nodes/params, pred_len)
Y_pred_norm_transposed = Y_pred_norm.T  # (pred_len, num_nodes)
Y_pred = scaler.inverse_transform(Y_pred_norm_transposed)

Y_true = test_df_orig.iloc[:forecast_steps].values

print(f"[OK] Forecast shape: {Y_pred.shape}")
print(f"     Inference time: {t_infer*1000:.2f}ms")

# ===== EVALUATE DAY-BY-DAY =====
print("\n" + "="*80)
print("DAY-BY-DAY PERFORMANCE ANALYSIS (MTGNN)")
print("="*80)

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
    day_metrics_df.to_csv(f"mtgnn_day_{day_num:02d}_metrics.csv", index=False)

    results_daily.append({
        "Day": day_num,
        "Overall_Skill_%": skill_day,
        "Overall_MAE": mae_day,
        "Overall_RMSE": rmse_day,
    })

# ===== SUMMARY =====
print("\n" + "="*80)
print("SUMMARY: MTGNN - DAY-BY-DAY SKILL")
print("="*80)

summary_df = pd.DataFrame(results_daily)
summary_df.to_csv("mtgnn_10days_summary.csv", index=False)

print("\n" + summary_df[[
    'Day', 'Overall_Skill_%', 'Overall_MAE', 'Overall_RMSE'
]].to_string(index=False))

# ===== COMPARISON =====
print("\n" + "="*80)
print("COMPARISON: MTGNN vs Other Models")
print("="*80)

try:
    nbeats_summary = pd.read_csv("nbeats_10days_summary.csv")
    hybrid_summary = pd.read_csv("hybrid8_10days_summary.csv")

    mtgnn_avg = summary_df['Overall_Skill_%'].mean()
    nbeats_avg = nbeats_summary['Overall_Skill_%'].mean()
    hybrid_avg = hybrid_summary['Overall_Skill_%'].mean()

    print(f"\nAverage Skill (All 10 Days):")
    print(f"  Single N-BEATS:      {nbeats_avg:+.1f}%")
    print(f"  Hybrid 8-Model:      {hybrid_avg:+.1f}%")
    print(f"  MTGNN:               {mtgnn_avg:+.1f}%")

    winner = max([("N-BEATS", nbeats_avg), ("Hybrid 8-Model", hybrid_avg), ("MTGNN", mtgnn_avg)], key=lambda x: x[1])
    print(f"\n  --> WINNER: {winner[0].upper()} ({winner[1]:+.1f}%)")

except FileNotFoundError:
    mtgnn_avg = summary_df['Overall_Skill_%'].mean()
    print(f"  MTGNN: {mtgnn_avg:+.1f}% (comparison pending)")

print(f"\n{'='*80}")
print(f"Files saved:")
print(f"  - mtgnn_10days_summary.csv (overall)")
print(f"  - mtgnn_day_01_metrics.csv ... mtgnn_day_10_metrics.csv (per-parameter)")
print(f"{'='*80}\n")
