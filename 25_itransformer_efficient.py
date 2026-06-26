#!/usr/bin/env python
"""iTransformer - Memory Efficient Version (generator-based)."""

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
print("ITRANSFORMER: MARINE FORECASTING (MEMORY EFFICIENT)")
print("="*80)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOOKBACK = 288
FORECAST = 1440
D_MODEL = 64
N_HEADS = 8
N_LAYERS = 2
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 20
PATIENCE = 8

print(f"[CONFIG] Device: {device} | Batch: {BATCH_SIZE}")

# ===== LOAD & PREPARE =====
print("\n[1/4] Loading dataset...")
df = pd.read_csv("120day_timestamp_18parameters.csv", index_col=0, parse_dates=True)

CIRCULAR_PARAMS = ['wind_direction_deg', 'current_direction_deg', 'compass_deg']
df_processed = df.copy()
for param in CIRCULAR_PARAMS:
    rad = np.deg2rad(df_processed[param])
    df_processed[f'{param}_sin'] = np.sin(rad)
    df_processed[f'{param}_cos'] = np.cos(rad)
df_processed = df_processed.drop(columns=CIRCULAR_PARAMS)

scaler = StandardScaler()
df_scaled = df_processed.copy()
df_scaled[:] = scaler.fit_transform(df_processed)

TEST_STEPS = 10 * 144
test_start = len(df_scaled) - TEST_STEPS
train_arr = df_scaled.iloc[:test_start].values.astype(np.float32)
test_df_orig = df_processed.iloc[test_start:].copy()

print(f"[OK] Train: {len(train_arr)} steps | Test: {len(test_df_orig)} steps")

# ===== iTransformer =====
class iTransformer(nn.Module):
    def __init__(self, lookback, n_vars, forecast_len, d_model=64, n_heads=8, n_layers=2):
        super().__init__()
        self.lookback = lookback
        self.n_vars = n_vars
        self.forecast_len = forecast_len

        self.var_embed = nn.Linear(lookback, d_model)
        self.var_id = nn.Parameter(torch.randn(n_vars, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        var_embed = self.var_embed(x_t) + self.var_id.unsqueeze(0)
        enc_out = self.encoder(var_embed)
        out = self.decoder(enc_out)
        return out.transpose(1, 2)

print("\n[2/4] Building model...")
model = iTransformer(LOOKBACK, len(df_processed.columns), FORECAST, D_MODEL, N_HEADS, N_LAYERS).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] Parameters: {total_params:,}")

# ===== TRAINING (GENERATOR) =====
print("\n[3/4] Training (generator-based)...")

def data_gen(arr, lookback, forecast, batch_size, shuffle=True):
    n_windows = len(arr) - lookback - forecast + 1
    indices = np.random.permutation(n_windows) if shuffle else np.arange(n_windows)

    for i in range(0, n_windows, batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = np.array([arr[idx:idx+lookback] for idx in batch_idx], dtype=np.float32)
        Y_batch = np.array([arr[idx+lookback:idx+lookback+forecast] for idx in batch_idx], dtype=np.float32)
        yield torch.from_numpy(X_batch).to(device), torch.from_numpy(Y_batch).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=6)
loss_fn = nn.MSELoss()

best_val_loss = float('inf')
best_state = None
wait = 0
t0 = time.time()

val_arr = train_arr[max(0, len(train_arr)-20000):]

for epoch in range(EPOCHS):
    epoch_t0 = time.time()

    # Train
    model.train()
    train_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in data_gen(train_arr, LOOKBACK, FORECAST, BATCH_SIZE):
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1

    train_loss /= n_batches

    # Validate
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in data_gen(val_arr, LOOKBACK, FORECAST, BATCH_SIZE, shuffle=False):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()
            val_batches += 1

    val_loss /= val_batches if val_batches > 0 else 1
    scheduler.step(val_loss)

    epoch_time = time.time() - epoch_t0
    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | train={train_loss:.6f} | val={val_loss:.6f} | time={epoch_time:.0f}s")

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print("  Early stopping")
            break

if best_state is not None:
    model.load_state_dict(best_state)

total_time = time.time() - t0
print(f"\n[OK] Training complete | Time: {total_time:.0f}s")

# ===== EVALUATE =====
print("\n[4/4] Evaluation...")

model.eval()
with torch.no_grad():
    last_window = torch.from_numpy(train_arr[-LOOKBACK:].astype(np.float32)).unsqueeze(0).to(device)
    y_pred_scaled = model(last_window).cpu().numpy()[0]

y_pred = y_pred_scaled * scaler.scale_ + scaler.mean_

pred_df = pd.DataFrame(y_pred, columns=df_processed.columns, index=test_df_orig.index)

for param in CIRCULAR_PARAMS:
    if f'{param}_sin' in pred_df.columns:
        sin_col = pred_df[f'{param}_sin'].values
        cos_col = pred_df[f'{param}_cos'].values
        angle = np.rad2deg(np.arctan2(sin_col, cos_col)) % 360
        pred_df[param] = angle
        pred_df = pred_df.drop(columns=[f'{param}_sin', f'{param}_cos'])

actual_df = test_df_orig.copy()

# Metrics
print("\n" + "="*80)
print("SKILL METRICS (vs Persistence)")
print("="*80)

def circular_mae(y_true, y_pred):
    return np.abs((y_true - y_pred + 180) % 360 - 180).mean()

metrics_list = []
last_obs = df_processed.iloc[test_start - 1]

for param in df.columns:
    if param not in pred_df.columns:
        continue

    y_true = actual_df[param].values
    y_pred = pred_df[param].values
    y_persist = np.repeat(last_obs[param], len(y_true))

    if param in CIRCULAR_PARAMS:
        mae_model = circular_mae(y_true, y_pred)
        mae_persist = circular_mae(y_true, y_persist)
        rmse_model = np.nan
    else:
        mae_model = mean_absolute_error(y_true, y_pred)
        mae_persist = mean_absolute_error(y_true, y_persist)
        rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))

    skill = (1 - mae_model / mae_persist) * 100 if mae_persist > 0 else np.nan

    metrics_list.append({
        'Parameter': param,
        'Persistence_MAE': round(mae_persist, 4),
        'iTransformer_MAE': round(mae_model, 4),
        'iTransformer_RMSE': round(rmse_model, 4) if not np.isnan(rmse_model) else 'N/A',
        'Skill_%': round(skill, 1),
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df.to_string(index=False))

valid_skills = metrics_df[metrics_df['Skill_%'].notna()]['Skill_%'].values
median_skill = np.median(valid_skills)
n_positive = (valid_skills > 0).sum()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Median Skill:    {median_skill:+.1f}%")
print(f"Positive Skill:  {n_positive}/{len(valid_skills)} parameters")
print(f"Training Time:   {total_time:.0f}s")

pred_df.to_csv("itransformer_predictions.csv")
metrics_df.to_csv("itransformer_metrics.csv", index=False)

print("\n[DONE] Results saved")
print(f"\nCompare to MTGNN baseline: +85.0%")
print(f"iTransformer result:      {median_skill:+.1f}%")
print(f"Gap:                      {median_skill - 85.0:+.1f}%")

print("\n" + "="*80)
