#!/usr/bin/env python
"""TimeXer for Marine Forecasting — Efficient Version (generator-based training).

This version addresses memory constraints by using a data generator instead of
loading all 152k windows into RAM at once.
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
print("TIMEXER FOR MARINE FORECASTING (EFFICIENT): 120 DAYS, 18 PARAMETERS")
print("="*80)

# ===== CONFIGURATION =====
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)

# Data parameters
LOOKBACK = 288
FORECAST = 1440
TRAIN_DAYS = 110
TEST_DAYS = 10

# Model hyperparameters (reduced for stability)
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 1
PATCH_LEN = 48              # Larger patches = fewer tokens
N_PATCHES = LOOKBACK // PATCH_LEN
DROPOUT = 0.1
BATCH_SIZE = 8              # Smaller batches
LR = 1e-3
EPOCHS = 25
PATIENCE = 10

print(f"\n[CONFIG] Device: {device} | Lookback: {LOOKBACK} | Forecast: {FORECAST}")
print(f"[CONFIG] Patch: {PATCH_LEN} | n_patches: {N_PATCHES} | d_model: {D_MODEL}")

# ===== [1/4] LOAD AND PREPARE DATA =====
print("\n[1/4] Loading and preparing dataset...")
df = pd.read_csv("120day_timestamp_18parameters.csv", index_col=0, parse_dates=True)
all_params = df.columns.tolist()
print(f"      Loaded {len(df)} rows × {len(all_params)} parameters")

# Circular encoding
CIRCULAR_PARAMS = ['wind_direction_deg', 'current_direction_deg', 'compass_deg']
df_processed = df.copy()

for param in CIRCULAR_PARAMS:
    rad = np.deg2rad(df_processed[param])
    df_processed[f'{param}_sin'] = np.sin(rad)
    df_processed[f'{param}_cos'] = np.cos(rad)

df_processed = df_processed.drop(columns=CIRCULAR_PARAMS)
params_for_model = df_processed.columns.tolist()

# Standardization
scaler = StandardScaler()
df_scaled = df_processed.copy()
df_scaled[:] = scaler.fit_transform(df_processed)

# Split
TEST_STEPS = TEST_DAYS * 144
test_start = len(df_scaled) - TEST_STEPS
train_df = df_scaled.iloc[:test_start].copy()
test_df_orig = df_processed.iloc[test_start:].copy()
test_df_scaled = df_scaled.iloc[test_start:].copy()

print(f"[OK] Train: {len(train_df)} steps | Test: {len(test_df_scaled)} steps")
print(f"[OK] Features: {len(params_for_model)} (after circular encoding)")

# ===== [2/4] BUILD MODEL =====
print("\n[2/4] Building TimeXer model...")


class DataEmbedding(nn.Module):
    def __init__(self, patch_len, n_features, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.linear_proj = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x):
        batch_size, lookback, n_features = x.shape
        n_patches = lookback // self.patch_len
        patches = x.reshape(batch_size, n_patches, self.patch_len * n_features)
        embedded = self.linear_proj(patches)
        return embedded


class TimeXer(nn.Module):
    def __init__(self, lookback, n_features, forecast_len, d_model=32, n_heads=4,
                 n_layers=1, patch_len=48, dropout=0.1):
        super().__init__()

        self.lookback = lookback
        self.n_features = n_features
        self.forecast_len = forecast_len
        self.d_model = d_model
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len

        # Embeddings
        self.endo_embed = DataEmbedding(patch_len, n_features, d_model)
        self.endo_pos_enc = nn.Parameter(torch.randn(self.n_patches, d_model) * 0.02)
        self.exo_embed = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder (bottleneck approach)
        intermediate_dim = min(256, forecast_len * n_features // 8)
        self.decoder_compress = nn.Linear(self.n_patches * d_model, intermediate_dim)
        self.decoder_expand = nn.Linear(intermediate_dim, forecast_len * n_features)

    def forward(self, x):
        batch_size = x.shape[0]

        endo_tok = self.endo_embed(x) + self.endo_pos_enc.unsqueeze(0)
        exo_tok = self.exo_embed.unsqueeze(0).expand(batch_size, 1, -1)

        tok = torch.cat([endo_tok, exo_tok], dim=1)
        enc_out = self.encoder(tok)
        enc_out_endo = enc_out[:, :self.n_patches, :]

        flat = enc_out_endo.reshape(batch_size, -1)
        compressed = torch.relu(self.decoder_compress(flat))
        out = self.decoder_expand(compressed)
        out = out.reshape(batch_size, self.forecast_len, self.n_features)

        return out


model = TimeXer(
    lookback=LOOKBACK, n_features=len(params_for_model), forecast_len=FORECAST,
    d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, patch_len=PATCH_LEN, dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] TimeXer model created | Parameters: {total_params:,}")

# ===== [3/4] TRAINING (with data generator) =====
print("\n[3/4] Training with data generator...")


def data_generator(arr, lookback, forecast, batch_size, shuffle=True):
    """Memory-efficient data generator for windowed time series."""
    n_windows = len(arr) - lookback - forecast + 1

    if shuffle:
        indices = np.random.permutation(n_windows)
    else:
        indices = np.arange(n_windows)

    for i in range(0, n_windows, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = []
        Y_batch = []

        for idx in batch_indices:
            X_batch.append(arr[idx:idx+lookback])
            Y_batch.append(arr[idx+lookback:idx+lookback+forecast])

        yield np.array(X_batch, dtype=np.float32), np.array(Y_batch, dtype=np.float32)


train_arr = train_df.values
val_size = max(1, int(0.1 * len(train_arr) - LOOKBACK - FORECAST))
val_start = len(train_arr) - val_size

val_arr = train_arr[val_start:]

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
    train_loss_total = 0.0
    n_batches = 0

    gen = data_generator(train_arr[:val_start], LOOKBACK, FORECAST, BATCH_SIZE, shuffle=True)
    for x_batch, y_batch in gen:
        x_t = torch.from_numpy(x_batch).to(device)
        y_t = torch.from_numpy(y_batch).to(device)

        optimizer.zero_grad()
        y_pred = model(x_t)
        loss = loss_fn(y_pred, y_t)
        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()
        n_batches += 1

    train_loss = train_loss_total / n_batches

    # Validation
    model.eval()
    val_loss_total = 0.0
    val_batches = 0

    with torch.no_grad():
        gen_val = data_generator(val_arr, LOOKBACK, FORECAST, BATCH_SIZE, shuffle=False)
        for x_batch, y_batch in gen_val:
            x_t = torch.from_numpy(x_batch).to(device)
            y_t = torch.from_numpy(y_batch).to(device)
            y_pred = model(x_t)
            loss = loss_fn(y_pred, y_t)
            val_loss_total += loss.item()
            val_batches += 1

    val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
    scheduler.step(val_loss)

    elapsed = time.time() - t0
    epoch_time = time.time() - epoch_t0

    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | train_loss={train_loss:.6f} | "
          f"val_loss={val_loss:.6f} | time={epoch_time:.0f}s")

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"       Early stopping")
            break

if best_state is not None:
    model.load_state_dict(best_state)

print(f"\n[OK] Training complete | Time: {time.time()-t0:.0f}s")

# ===== [4/4] EVALUATION =====
print("\n[4/4] Evaluation...")

model.eval()
with torch.no_grad():
    last_window = torch.from_numpy(train_df.values[-LOOKBACK:].astype(np.float32)).unsqueeze(0).to(device)
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
        'TimeXer_MAE': round(mae_model, 4),
        'TimeXer_RMSE': round(rmse_model, 4) if not np.isnan(rmse_model) else 'N/A',
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
print(f"Training Time:   {time.time()-t0:.0f}s")

# Save
pred_df.to_csv("timexer_predictions.csv")
metrics_df.to_csv("timexer_metrics.csv", index=False)

print("\n[DONE] Results saved: timexer_predictions.csv, timexer_metrics.csv")

if median_skill >= 80:
    print("\n[RESULT] EXCELLENT: TimeXer competitive with MTGNN baseline!")
elif median_skill >= 70:
    print("\n[RESULT] GOOD: TimeXer within striking distance of MTGNN")
else:
    print(f"\n[RESULT] Underperforming: {median_skill:+.1f}% vs MTGNN +85%")

print("\n" + "="*80)
