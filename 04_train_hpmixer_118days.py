#!/usr/bin/env python
"""Train HPMixer for 2-day marine forecasting on 118-day training window."""
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys

# Add HPMixer to path
sys.path.insert(0, "HPMixer")

print("\n" + "="*80)
print("STEP 2B: HPMIXER TRAINING - 118-DAY TRAINING + 2-DAY TEST")
print("="*80)

# ===== LOAD PREPARED DATA =====
print("\n[1/5] Loading prepared dataset...")
df = pd.read_csv("marine_120day_18params_10min.csv", index_col=0, parse_dates=True)
print(f"[OK] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"     Date range: {df.index[0]} to {df.index[-1]}")

params = df.columns.tolist()
print(f"[OK] Using {len(params)} parameters")

# ===== SPLIT DATA =====
print("\n[2/5] Splitting data (118 days train + 2 days test)...")
horizon = 288  # 2 days at 10-min = 288 steps
train_window = 16992  # 118 days at 10-min = 16,992 steps
lookback = 288  # Context window = 2 days

test_start = len(df) - horizon
train_end = test_start
train_start = train_end - train_window

train_df = df.iloc[train_start:train_end].copy()
test_df = df.iloc[test_start:].copy()

print(f"[OK] Train: {train_window} steps ({train_window/144:.1f} days)")
print(f"     Train dates: {train_df.index[0]} to {train_df.index[-1]}")
print(f"[OK] Test: {horizon} steps ({horizon/144:.1f} days)")
print(f"     Test dates: {test_df.index[0]} to {test_df.index[-1]}")

# ===== STANDARDIZE =====
print("\n[3/5] Standardizing...")
scaler = StandardScaler()
train_df_scaled = train_df.copy()
train_df_scaled[:] = scaler.fit_transform(train_df)

test_df_scaled = test_df.copy()
test_df_scaled[:] = scaler.transform(test_df)

print(f"[OK] Standardized")

# ===== BUILD TRAINING DATA =====
print("\n[4/5] Building training samples...")
X_train, Y_train = [], []
for i in range(lookback, len(train_df_scaled) - horizon, 2):
    x = train_df_scaled.iloc[i - lookback:i].values.astype(np.float32)
    y = train_df_scaled.iloc[i:i + horizon].values.astype(np.float32)
    X_train.append(x)
    Y_train.append(y)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(f"[OK] Built {len(X_train)} training samples")
print(f"     Shape: X={X_train.shape}, Y={Y_train.shape}")

# Split train/val
n_val = max(1, int(0.1 * len(X_train)))
perm = np.random.permutation(len(X_train))
val_idx, tr_idx = perm[:n_val], perm[n_val:]
X_tr, Y_tr = X_train[tr_idx], Y_train[tr_idx]
X_val, Y_val = X_train[val_idx], Y_train[val_idx]

device = torch.device("cpu")
torch.set_num_threads(8)

X_tr_t = torch.from_numpy(X_tr).to(device)
Y_tr_t = torch.from_numpy(Y_tr).to(device)
X_val_t = torch.from_numpy(X_val).to(device)
Y_val_t = torch.from_numpy(Y_val).to(device)

# ===== DEFINE HPMIXER =====
print("\n[5/5] Training HPMixer...")

class HPMixer(nn.Module):
    """Simplified Patch Mixer for time series forecasting."""
    def __init__(self, seq_len, pred_len, n_vars, patch_size=16, d_model=128, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.d_model = d_model

        # Input projection: (seq_len, n_vars) -> d_model
        self.input_proj = nn.Linear(seq_len, d_model)

        # Mixing blocks for each variable
        self.mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        B, L, V = x.shape

        # Project each variable independently
        x_proj = self.input_proj(x.transpose(1, 2))  # (B, n_vars, d_model)

        # Apply mixing layers
        for mixer in self.mixers:
            x_proj = x_proj + mixer(x_proj)  # Residual connection

        # Generate predictions
        y_pred = self.output_proj(x_proj)  # (B, n_vars, pred_len)
        y_pred = y_pred.transpose(1, 2)  # (B, pred_len, n_vars)

        return y_pred

# Create model
model = HPMixer(seq_len=lookback, pred_len=horizon, n_vars=len(params),
                patch_size=16, d_model=128, n_layers=2).to(device)
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

    if (ep + 1) % 10 == 0 or wait >= patience:
        print(f"     Epoch {ep+1:2d}/50 | Val: {val_loss:.6f} | Wait: {wait}/15")

    if wait >= patience:
        print(f"     -> Early stop at epoch {ep+1}")
        break

if best_state:
    model.load_state_dict(best_state)

t_train = time.time() - t0

# ===== EVALUATE =====
model.eval()
t0 = time.time()
with torch.no_grad():
    last_context = train_df_scaled.iloc[-lookback:].values.astype(np.float32)
    X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)
    Y_pred_norm = model(X_test)[0].cpu().numpy()
t_infer = time.time() - t0

# Inverse normalize
Y_pred = scaler.inverse_transform(Y_pred_norm)

# Get true values
Y_true = test_df.iloc[:horizon].values

# Persistence baseline
last_obs = df.iloc[-horizon - 1].values
Y_persist = np.tile(last_obs, (horizon, 1))

# Compute metrics
mae_pred = mean_absolute_error(Y_true, Y_pred)
mae_pers = mean_absolute_error(Y_true, Y_persist)
skill = (1 - mae_pred / mae_pers) * 100 if mae_pers > 0 else 0

rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))

# Per-parameter metrics
print("\n" + "="*80)
print("RESULTS: HPMixer 2-Day Forecast (118 days training)")
print("="*80)
print(f"\nOverall Metrics:")
print(f"  MAE: {mae_pred:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  Skill: {skill:+.1f}%")
print(f"  Persistence MAE: {mae_pers:.4f}")

print(f"\nTiming:")
print(f"  Training: {t_train:.1f}s ({t_train/60:.2f} min)")
print(f"  Inference: {t_infer*1000:.2f}ms")

# Top/bottom performers
metrics_list = []
for j, p in enumerate(params):
    y_t = Y_true[:, j]
    y_p = Y_pred[:, j]
    y_pers = Y_persist[:, j]

    mae = mean_absolute_error(y_t, y_p)
    rmse_p = np.sqrt(mean_squared_error(y_t, y_p))
    mae_pers_p = mean_absolute_error(y_t, y_pers)
    skill_p = (1 - mae / mae_pers_p) * 100 if mae_pers_p > 0 else 0

    metrics_list.append({
        "Parameter": p,
        "MAE": round(mae, 4),
        "RMSE": round(rmse_p, 4),
        "Skill_%": round(skill_p, 1),
    })

metrics_df = pd.DataFrame(metrics_list)

print(f"\n{'='*80}")
print("TOP 5 PERFORMERS:")
print(f"{'='*80}")
for _, row in metrics_df.nlargest(5, "Skill_%").iterrows():
    print(f"  {row['Parameter']:30s} {row['Skill_%']:+7.1f}%  MAE: {row['MAE']:.4f}")

print(f"\n{'='*80}")
print("BOTTOM 5 PERFORMERS:")
print(f"{'='*80}")
for _, row in metrics_df.nsmallest(5, "Skill_%").iterrows():
    print(f"  {row['Parameter']:30s} {row['Skill_%']:+7.1f}%  MAE: {row['MAE']:.4f}")

# Save results
results_df = pd.DataFrame([{
    "Model": "HPMixer",
    "Training_Days": 118,
    "Horizon_Days": 2,
    "Skill_%": skill,
    "MAE": mae_pred,
    "RMSE": rmse,
    "Training_Time_s": t_train,
    "Inference_Time_ms": t_infer * 1000,
    "Num_Parameters": len(params),
    "Training_Samples": len(X_train),
    "Train_Start": str(train_df.index[0]),
    "Train_End": str(train_df.index[-1]),
    "Test_Start": str(test_df.index[0]),
    "Test_End": str(test_df.index[-1]),
}])

results_df.to_csv("hpmixer_118day_results.csv", index=False)
metrics_df.to_csv("hpmixer_118day_metrics.csv", index=False)

print(f"\n{'='*80}")
print("Files saved:")
print(f"  - hpmixer_118day_results.csv")
print(f"  - hpmixer_118day_metrics.csv")
print(f"{'='*80}\n")
