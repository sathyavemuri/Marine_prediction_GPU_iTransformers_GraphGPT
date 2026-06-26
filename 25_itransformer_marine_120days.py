#!/usr/bin/env python
"""iTransformer for Marine Forecasting: 120 days, 18 parameters, 10-day forecast.

iTransformer (Inverted Transformers for Time Series Forecasting)
- Each parameter becomes a token (18 tokens for marine data)
- Self-attention learns which parameters inform which directly
- More aligned with MTGNN's parameter-coupling philosophy

Expected: Should match or exceed TimeXer performance.
"""

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
print("ITRANSFORMER FOR MARINE FORECASTING: 120 DAYS, 18 PARAMETERS")
print("="*80)

# ===== CONFIG =====
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOOKBACK = 288
FORECAST = 1440
D_MODEL = 64
N_HEADS = 8
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
PATIENCE = 10

print(f"[CONFIG] Device: {device} | Lookback: {LOOKBACK} | Forecast: {FORECAST}")

# ===== [1/5] LOAD DATA =====
print("\n[1/5] Loading dataset...")
df = pd.read_csv("120day_timestamp_18parameters.csv", index_col=0, parse_dates=True)
all_params = df.columns.tolist()

# Circular encoding
CIRCULAR_PARAMS = ['wind_direction_deg', 'current_direction_deg', 'compass_deg']
df_processed = df.copy()

for param in CIRCULAR_PARAMS:
    rad = np.deg2rad(df_processed[param])
    df_processed[f'{param}_sin'] = np.sin(rad)
    df_processed[f'{param}_cos'] = np.cos(rad)

df_processed = df_processed.drop(columns=CIRCULAR_PARAMS)
params_for_model = df_processed.columns.tolist()

print(f"[OK] Loaded {len(df)} rows | {len(params_for_model)} features")

# ===== [2/5] STANDARDIZE =====
print("\n[2/5] Standardizing...")
scaler = StandardScaler()
df_scaled = df_processed.copy()
df_scaled[:] = scaler.fit_transform(df_processed)

TEST_STEPS = 10 * 144
test_start = len(df_scaled) - TEST_STEPS
train_df = df_scaled.iloc[:test_start].copy()
test_df_orig = df_processed.iloc[test_start:].copy()

print(f"[OK] Train: {len(train_df)} steps | Test: {len(test_df_orig)} steps")

# ===== [3/5] BUILD WINDOWED DATASET =====
print("\n[3/5] Building windowed dataset...")

def create_windows(df, lookback, forecast):
    arr = df.values.astype(np.float32)
    X, Y = [], []
    for origin in range(lookback, len(arr) - forecast):
        X.append(arr[origin - lookback:origin])
        Y.append(arr[origin:origin + forecast])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X_train, Y_train = create_windows(train_df, LOOKBACK, FORECAST)
n_val = max(1, int(0.1 * len(X_train)))
X_tr, Y_tr = X_train[:-n_val], Y_train[:-n_val]
X_val, Y_val = X_train[-n_val:], Y_train[-n_val:]

X_tr_t = torch.from_numpy(X_tr).to(device)
Y_tr_t = torch.from_numpy(Y_tr).to(device)
X_val_t = torch.from_numpy(X_val).to(device)
Y_val_t = torch.from_numpy(Y_val).to(device)

print(f"[OK] Train: {X_tr_t.shape} | Val: {X_val_t.shape}")

last_window = torch.from_numpy(train_df.values[-LOOKBACK:].astype(np.float32)).unsqueeze(0).to(device)

# ===== [4/5] BUILD ITRANSFORMER =====
print("\n[4/5] Building iTransformer...")

class iTransformer(nn.Module):
    """Inverted Transformer: each parameter is a token."""

    def __init__(self, lookback, n_vars, forecast_len, d_model=64, n_heads=8,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.n_vars = n_vars
        self.forecast_len = forecast_len
        self.d_model = d_model

        # Embed each variable's time series to d_model dimension
        self.var_embed = nn.Linear(lookback, d_model)

        # Variable identity tokens (learnable)
        self.var_id = nn.Parameter(torch.randn(n_vars, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder head: project d_model back to forecast_len
        self.decoder = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        # x: (batch, lookback, n_vars)
        batch_size = x.shape[0]

        # Transpose to (batch, n_vars, lookback)
        x_t = x.transpose(1, 2)

        # Embed each variable's history to d_model
        var_embed = self.var_embed(x_t)  # (batch, n_vars, d_model)

        # Add variable identity tokens
        var_embed = var_embed + self.var_id.unsqueeze(0)

        # Apply transformer encoder (self-attention over variables)
        enc_out = self.encoder(var_embed)  # (batch, n_vars, d_model)

        # Decode each variable to forecast horizon
        out = self.decoder(enc_out)  # (batch, n_vars, forecast_len)

        # Transpose back to (batch, forecast_len, n_vars)
        out = out.transpose(1, 2)

        return out

model = iTransformer(
    lookback=LOOKBACK, n_vars=len(params_for_model), forecast_len=FORECAST,
    d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] iTransformer created | Parameters: {total_params:,}")

# ===== [5/5] TRAINING =====
print("\n[5/5] Training...")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=6)
loss_fn = nn.MSELoss()

best_val_loss = float('inf')
best_state = None
wait = 0
t0 = time.time()

for epoch in range(EPOCHS):
    epoch_t0 = time.time()

    # Training
    model.train()
    perm = torch.randperm(len(X_tr_t))
    train_loss = 0.0
    n_batches = 0

    for i in range(0, len(X_tr_t), BATCH_SIZE):
        batch_idx = perm[i:i+BATCH_SIZE]
        x_batch = X_tr_t[batch_idx]
        y_batch = Y_tr_t[batch_idx]

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_batches += 1

    train_loss /= n_batches

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t)
        val_loss = loss_fn(y_val_pred, Y_val_t).item()

    scheduler.step(val_loss)

    elapsed = time.time() - t0
    epoch_time = time.time() - epoch_t0

    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | time={epoch_time:.0f}s")

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stopping")
            break

if best_state is not None:
    model.load_state_dict(best_state)

print(f"\n[OK] Training complete | Time: {time.time()-t0:.0f}s")

# ===== EVALUATION =====
print("\n[EVALUATION] Computing metrics...")

model.eval()
with torch.no_grad():
    y_pred_scaled = model(last_window).cpu().numpy()[0]

# Inverse transform
y_pred = y_pred_scaled * scaler.scale_[:len(params_for_model)] + scaler.mean_[:len(params_for_model)]

# Reconstruct circular
pred_df = pd.DataFrame(y_pred, columns=params_for_model, index=test_df_orig.index)

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

for param in all_params:
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

# Summary
valid_skills = metrics_df[metrics_df['Skill_%'].notna()]['Skill_%'].values
median_skill = np.median(valid_skills)
n_positive = (valid_skills > 0).sum()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Median Skill:    {median_skill:+.1f}%")
print(f"Positive Skill:  {n_positive}/{len(valid_skills)} parameters")
print(f"Model Size:      {total_params:,} params")
print(f"Training Time:   {time.time()-t0:.0f}s")

# Save
pred_df.to_csv("itransformer_predictions.csv")
metrics_df.to_csv("itransformer_metrics.csv", index=False)

print("\n[DONE] Results saved: itransformer_predictions.csv, itransformer_metrics.csv")

print("\n" + "="*80)
print("COMPARISON TO BASELINE (Correlated Input MTGNN: +85.0%)")
print("="*80)
print(f"iTransformer Median Skill: {median_skill:+.1f}%")
print(f"Baseline:                  +85.0%")
print(f"Gap:                       {median_skill - 85.0:+.1f}%")

if median_skill >= 80:
    print("\n[RESULT] EXCELLENT: iTransformer competitive with MTGNN!")
elif median_skill >= 70:
    print("\n[RESULT] GOOD: iTransformer within striking distance")
else:
    print(f"\n[RESULT] Underperforming vs MTGNN baseline")

print("\n" + "="*80)
