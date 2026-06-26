#!/usr/bin/env python
"""Conv1d Channel Mixer for 18 marine sensors (110-day training, 10-day forecast)."""
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
print("CONV1D CHANNEL MIXER: 110 DAYS TRAIN -> 10 DAYS FORECAST")
print("="*80)

# ===== LOAD DATA =====
print("\n[1/5] Loading dataset...")
df = pd.read_csv("marine_120day_18params_10min.csv", index_col=0, parse_dates=True)
params = df.columns.tolist()
print(f"[OK] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"     Date range: {df.index[0]} to {df.index[-1]}")
print(f"     Total days: {df.shape[0]/144:.1f}")

# ===== STANDARDIZE =====
print("\n[2/5] Standardizing...")
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[:] = scaler.fit_transform(df)

# Parameters
train_days = 110
forecast_days = 10
train_steps = train_days * 144  # 15,840 steps
forecast_steps = forecast_days * 144  # 1,440 steps
lookback_steps = 288  # 2 days

print(f"[OK] Training: {train_days} days ({train_steps} steps)")
print(f"     Forecast: {forecast_days} days ({forecast_steps} steps)")

# Split: Train first 110 days, test last 10 days
test_start = len(df_scaled) - forecast_steps
train_end = test_start
train_start = 0

train_df = df_scaled.iloc[train_start:train_end].copy()
test_df = df_scaled.iloc[test_start:].copy()
test_df_orig = df.iloc[test_start:].copy()

print(f"[OK] Train: rows 0 to {train_end} ({train_days} days)")
print(f"     Test: rows {test_start} to {len(df_scaled)} ({forecast_days} days)")

# ===== DEFINE CONV1D CHANNEL MIXER =====
class Conv1dChannelMixer(nn.Module):
    """
    1D CNN + Channel Mixing for multivariate sensor forecasting.

    Architecture:
    - Conv1d blocks: Capture local temporal patterns (e.g., wind→waves)
    - Channel mixing: Learn parameter interactions (e.g., temp↔salinity)
    - Residual connections: Stabilize training across all 18 sensors
    """

    def __init__(self, seq_len, pred_len, n_vars, d_model=64, n_layers=3, kernel_size=5):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model

        # Input projection: (batch, seq_len, n_vars) -> (batch, d_model, seq_len)
        self.input_proj = nn.Linear(n_vars, d_model)

        # Conv1d blocks with residual connections
        self.conv_blocks = nn.ModuleList()
        for i in range(n_layers):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model * 2, kernel_size=kernel_size,
                         padding=kernel_size//2, padding_mode='reflect'),
                nn.BatchNorm1d(d_model * 2),
                nn.GELU(),
                nn.Conv1d(d_model * 2, d_model, kernel_size=kernel_size,
                         padding=kernel_size//2, padding_mode='reflect'),
                nn.BatchNorm1d(d_model),
            ))

        # Channel mixing: Learn interdependencies between parameters
        self.channel_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            ) for _ in range(n_layers)
        ])

        # Output projection: (batch, d_model, seq_len) -> (batch, pred_len, n_vars)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_vars)
        batch_size = x.shape[0]

        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x.transpose(1, 2)   # (batch, d_model, seq_len)

        # Conv1d blocks with residual connections and channel mixing
        for i, (conv_block, channel_mixer) in enumerate(zip(self.conv_blocks, self.channel_mixers)):
            # Temporal convolution
            x_conv = conv_block(x)  # (batch, d_model, seq_len)
            x = x + x_conv  # Residual connection

            # Channel mixing (learn parameter interactions)
            x_mix = x.transpose(1, 2)  # (batch, seq_len, d_model)
            x_mix = channel_mixer(x_mix)  # (batch, seq_len, d_model)
            x_mix = x_mix.transpose(1, 2)  # (batch, d_model, seq_len)
            x = x + x_mix  # Residual connection

        # Output projection: global average pooling then project to pred_len
        x = x.mean(dim=2)  # (batch, d_model) - average across time
        y_pred = self.output_proj(x)  # (batch, pred_len)
        y_pred = y_pred.unsqueeze(-1).expand(-1, -1, self.n_vars)  # (batch, pred_len, n_vars)

        return y_pred

device = torch.device("cpu")
torch.set_num_threads(8)

# ===== BUILD TRAINING DATA =====
print("\n[3/5] Building training samples...")
X_train, Y_train = [], []
for i in range(lookback_steps, len(train_df) - forecast_steps, 2):
    x = train_df.iloc[i - lookback_steps:i].values.astype(np.float32)
    y = train_df.iloc[i:i + forecast_steps].values.astype(np.float32)
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

X_tr_t = torch.from_numpy(X_tr).to(device)
Y_tr_t = torch.from_numpy(Y_tr).to(device)
X_val_t = torch.from_numpy(X_val).to(device)
Y_val_t = torch.from_numpy(Y_val).to(device)

# ===== TRAIN =====
print("\n[4/5] Training Conv1d Channel Mixer...")

model = Conv1dChannelMixer(
    seq_len=lookback_steps,
    pred_len=forecast_steps,
    n_vars=len(params),
    d_model=64,
    n_layers=3,
    kernel_size=5
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
    last_context = train_df.iloc[-lookback_steps:].values.astype(np.float32)
    X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)
    Y_pred_norm = model(X_test)[0].cpu().numpy()
t_infer = time.time() - t0

# Inverse normalize
Y_pred = scaler.inverse_transform(Y_pred_norm)
Y_true = test_df_orig.iloc[:forecast_steps].values

print(f"[OK] Forecast shape: {Y_pred.shape}")
print(f"     Inference time: {t_infer*1000:.2f}ms")

# ===== EVALUATE DAY-BY-DAY =====
print("\n" + "="*80)
print("DAY-BY-DAY PERFORMANCE ANALYSIS (Conv1d Channel Mixer)")
print("="*80)

results_daily = []

for day_num in range(1, forecast_days + 1):
    day_start = (day_num - 1) * 144
    day_end = day_num * 144

    Y_true_day = Y_true[day_start:day_end]
    Y_pred_day = Y_pred[day_start:day_end]

    # Persistence baseline
    last_obs = df.iloc[test_start - 1].values
    Y_persist_day = np.tile(last_obs, (144, 1))

    # Overall metrics
    mae_day = mean_absolute_error(Y_true_day, Y_pred_day)
    mae_persist_day = mean_absolute_error(Y_true_day, Y_persist_day)
    skill_day = (1 - mae_day / mae_persist_day) * 100 if mae_persist_day > 0 else 0
    rmse_day = np.sqrt(mean_squared_error(Y_true_day, Y_pred_day))

    print(f"\nDAY {day_num}:")
    print(f"  Overall Skill: {skill_day:+.1f}%")
    print(f"  MAE: {mae_day:.4f} | RMSE: {rmse_day:.4f}")

    # Per-parameter metrics
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

    # Top 3 & Bottom 3
    print(f"  Top 3 parameters:")
    for _, row in day_metrics_df.nlargest(3, "Skill_%").iterrows():
        print(f"    {row['Parameter']:30s} {row['Skill_%']:+7.1f}%")

    print(f"  Bottom 3 parameters:")
    for _, row in day_metrics_df.nsmallest(3, "Skill_%").iterrows():
        print(f"    {row['Parameter']:30s} {row['Skill_%']:+7.1f}%")

    # Save daily metrics
    day_metrics_df.to_csv(f"conv1d_day_{day_num:02d}_metrics.csv", index=False)

    # Add to summary
    results_daily.append({
        "Day": day_num,
        "Overall_Skill_%": skill_day,
        "Overall_MAE": mae_day,
        "Overall_RMSE": rmse_day,
        "Training_Days": train_days,
        "Training_Steps": train_steps,
        "Training_Time_s": t_train,
    })

# ===== SUMMARY =====
print("\n" + "="*80)
print("SUMMARY: Conv1d Channel Mixer - DAY-BY-DAY SKILL")
print("="*80)

summary_df = pd.DataFrame(results_daily)
summary_df.to_csv("conv1d_10days_summary.csv", index=False)

print("\n" + summary_df[[
    'Day', 'Overall_Skill_%', 'Overall_MAE', 'Overall_RMSE'
]].to_string(index=False))

# ===== COMPARISON WITH HPMIXER =====
print("\n" + "="*80)
print("COMPARISON: Conv1d Channel Mixer vs HPMixer")
print("="*80)

try:
    hpmixer_summary = pd.read_csv("forecast_10days_summary.csv")

    print("\nAverage Skill (All 10 Days):")
    conv1d_avg = summary_df['Overall_Skill_%'].mean()
    hpmixer_avg = hpmixer_summary['Overall_Skill_%'].mean()

    print(f"  HPMixer:        {hpmixer_avg:+.1f}%")
    print(f"  Conv1d Mixer:   {conv1d_avg:+.1f}%")
    print(f"  Improvement:    {conv1d_avg - hpmixer_avg:+.1f}% {'[WINNER]' if conv1d_avg > hpmixer_avg else ''}")

    # Best day comparison
    conv1d_best = summary_df['Overall_Skill_%'].max()
    hpmixer_best = hpmixer_summary['Overall_Skill_%'].max()
    print(f"\nBest Day Skill:")
    print(f"  HPMixer:        {hpmixer_best:+.1f}%")
    print(f"  Conv1d Mixer:   {conv1d_best:+.1f}%")

    # Worst day comparison
    conv1d_worst = summary_df['Overall_Skill_%'].min()
    hpmixer_worst = hpmixer_summary['Overall_Skill_%'].min()
    print(f"\nWorst Day Skill:")
    print(f"  HPMixer:        {hpmixer_worst:+.1f}%")
    print(f"  Conv1d Mixer:   {conv1d_worst:+.1f}%")

    print(f"\nTraining Time:")
    print(f"  Both models:    ~{t_train:.0f} seconds")
    print(f"  Inference:      {t_infer*1000:.2f}ms per 10-day forecast")

except FileNotFoundError:
    print("  (HPMixer results not found for comparison)")

print(f"\n{'='*80}")
print(f"Files saved:")
print(f"  - conv1d_10days_summary.csv (overall)")
print(f"  - conv1d_day_01_metrics.csv ... conv1d_day_10_metrics.csv (per-parameter)")
print(f"{'='*80}\n")
