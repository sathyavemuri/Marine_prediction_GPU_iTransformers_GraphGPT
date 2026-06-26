#!/usr/bin/env python
"""Train HPMixer for 1-15 day horizons using 14× training ratio."""
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
print("MULTI-HORIZON HPMIXER TRAINING: 1-15 Days")
print("Using 14x Training Ratio Rule")
print("="*80)

# ===== LOAD DATA =====
print("\n[Loading] marine_120day_18params_10min.csv...")
df = pd.read_csv("marine_120day_18params_10min.csv", index_col=0, parse_dates=True)
params = df.columns.tolist()
print(f"[OK] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"     Date range: {df.index[0]} to {df.index[-1]}")

# ===== STANDARDIZE (GLOBAL) =====
print("\n[Standardizing] Using global scaler for all horizons...")
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[:] = scaler.fit_transform(df)

# ===== DEFINE HPMIXER =====
class HPMixer(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, d_model=128, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model

        self.input_proj = nn.Linear(seq_len, d_model)

        self.mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        x_proj = self.input_proj(x.transpose(1, 2))  # (B, n_vars, d_model)

        for mixer in self.mixers:
            x_proj = x_proj + mixer(x_proj)

        y_pred = self.output_proj(x_proj)  # (B, n_vars, pred_len)
        y_pred = y_pred.transpose(1, 2)  # (B, pred_len, n_vars)

        return y_pred

# ===== TRAINING LOOP =====
device = torch.device("cpu")
torch.set_num_threads(8)

results_all = []

for horizon_days in range(1, 16):
    print(f"\n{'='*80}")
    print(f"HORIZON: {horizon_days}-DAY FORECAST")
    print(f"{'='*80}")

    # Calculate training window (14× rule, capped at 120 days)
    train_days = min(horizon_days * 14, 120)
    train_steps = train_days * 144
    horizon_steps = horizon_days * 144
    lookback_steps = 288  # 2 days context

    total_needed = train_steps + horizon_steps
    available = len(df_scaled)

    if total_needed > available:
        train_steps = available - horizon_steps
        train_days = train_steps // 144

    print(f"\nTraining window: {train_days} days ({train_steps} steps)")
    print(f"Forecast horizon: {horizon_days} days ({horizon_steps} steps)")

    # Split data
    test_start = len(df_scaled) - horizon_steps
    train_end = test_start
    train_start = train_end - train_steps

    train_df = df_scaled.iloc[train_start:train_end].copy()
    test_df = df_scaled.iloc[test_start:].copy()

    print(f"Train: {train_start} to {train_end}")
    print(f"Test: {test_start} to {len(df_scaled)}")

    # Build training data
    X_train, Y_train = [], []
    for i in range(lookback_steps, len(train_df) - horizon_steps, 2):
        x = train_df.iloc[i - lookback_steps:i].values.astype(np.float32)
        y = train_df.iloc[i:i + horizon_steps].values.astype(np.float32)
        X_train.append(x)
        Y_train.append(y)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    print(f"Training samples: {len(X_train)}")

    if len(X_train) < 10:
        print(f"[SKIP] Not enough training samples ({len(X_train)} < 10)")
        continue

    # Split train/val
    n_val = max(1, int(0.1 * len(X_train)))
    perm = np.random.permutation(len(X_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, Y_tr = X_train[tr_idx], Y_train[tr_idx]
    X_val, Y_val = X_train[val_idx], Y_train[val_idx]

    X_tr_t = torch.from_numpy(X_tr).to(device)
    Y_tr_t = torch.from_numpy(Y_tr).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)

    # Train
    model = HPMixer(seq_len=lookback_steps, pred_len=horizon_steps, n_vars=len(params)).to(device)
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
            print(f"  Epoch {ep+1:2d}/50 | Val: {val_loss:.6f} | Wait: {wait}/15")

        if wait >= patience:
            print(f"  -> Early stop at epoch {ep+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    t_train = time.time() - t0

    # Evaluate
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        last_context = train_df.iloc[-lookback_steps:].values.astype(np.float32)
        X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)
        Y_pred_norm = model(X_test)[0].cpu().numpy()
    t_infer = time.time() - t0

    # Inverse normalize
    Y_pred = scaler.inverse_transform(Y_pred_norm)

    # Get true values
    Y_true = df.iloc[test_start:test_start + horizon_steps].values

    # Persistence baseline
    last_obs = df.iloc[test_start - 1].values
    Y_persist = np.tile(last_obs, (horizon_steps, 1))

    # Compute metrics
    mae_pred = mean_absolute_error(Y_true, Y_pred)
    mae_pers = mean_absolute_error(Y_true, Y_persist)
    skill = (1 - mae_pred / mae_pers) * 100 if mae_pers > 0 else 0

    rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))

    # Per-parameter metrics
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
            "Horizon_Days": horizon_days,
            "Parameter": p,
            "MAE": round(mae, 4),
            "RMSE": round(rmse_p, 4),
            "Skill_%": round(skill_p, 1),
            "Persistence_MAE": round(mae_pers_p, 4),
        })

    metrics_df = pd.DataFrame(metrics_list)

    print(f"\nResults:")
    print(f"  Overall Skill: {skill:+.1f}%")
    print(f"  Overall MAE: {mae_pred:.4f}")
    print(f"  Overall RMSE: {rmse:.4f}")
    print(f"  Training Time: {t_train:.1f}s")
    print(f"  Inference Time: {t_infer*1000:.2f}ms")

    print(f"\nTop 3 Parameters:")
    for _, row in metrics_df.nlargest(3, "Skill_%").iterrows():
        print(f"  {row['Parameter']:30s} {row['Skill_%']:+7.1f}%")

    # Save results
    results_summary = pd.DataFrame([{
        "Horizon_Days": horizon_days,
        "Training_Days": train_days,
        "Training_Steps": train_steps,
        "Horizon_Steps": horizon_steps,
        "Num_Samples": len(X_train),
        "Overall_Skill_%": skill,
        "Overall_MAE": mae_pred,
        "Overall_RMSE": rmse,
        "Training_Time_s": t_train,
        "Inference_Time_ms": t_infer * 1000,
    }])

    results_all.append(results_summary)
    metrics_df.to_csv(f"horizon_{horizon_days:02d}d_metrics.csv", index=False)

# ===== SUMMARY =====
print("\n" + "="*80)
print("ALL HORIZONS TRAINED")
print("="*80)

summary_all = pd.concat(results_all, ignore_index=True)
summary_all.to_csv("all_horizons_summary.csv", index=False)

print("\n" + summary_all.to_string(index=False))

print("\n" + "="*80)
print("FILES SAVED")
print("="*80)
print("Summary: all_horizons_summary.csv")
print("Per-horizon metrics: horizon_01d_metrics.csv ... horizon_15d_metrics.csv")
print("\n")
