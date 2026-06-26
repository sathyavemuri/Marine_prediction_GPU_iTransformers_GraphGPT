#!/usr/bin/env python
"""TimeXer for Marine Forecasting: 120-day data, 18 parameters, 10-day (1440-step) forecast.

TimeXer (Empowering Transformers for Time Series Forecasting with Exogenous Variables)
leverages exogenous variables via cross-attention between patch embeddings and exogenous
tokens. This implementation adapts the architecture for marine prediction.

Expected results: Should compare favorably to Correlated Input MTGNN (+85% skill baseline).
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
print("TIMEXER FOR MARINE FORECASTING: 120 DAYS, 18 PARAMETERS, 10-DAY FORECAST")
print("="*80)

# ===== CONFIGURATION =====
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

# Data parameters
LOOKBACK = 288           # 2 days at 10-min resolution (2 * 24 * 60 / 10)
FORECAST = 1440          # 10 days at 10-min resolution (10 * 24 * 60 / 10)
TRAIN_DAYS = 110
TEST_DAYS = 10
TRAIN_STEPS = TRAIN_DAYS * 144  # 144 steps per day (24 * 60 / 10)
TEST_STEPS = TEST_DAYS * 144

# Model hyperparameters
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 1
PATCH_LEN = 24              # Patch length for temporal tokenization
N_PATCHES = LOOKBACK // PATCH_LEN
DROPOUT = 0.1
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 30
PATIENCE = 12

print(f"\n[CONFIG] Device: {device} | Lookback: {LOOKBACK} | Forecast: {FORECAST}")
print(f"[CONFIG] Patch: {PATCH_LEN} | n_patches: {N_PATCHES} | d_model: {D_MODEL}")

# ===== [1/6] LOAD DATA =====
print("\n[1/6] Loading dataset...")
df = pd.read_csv("120day_timestamp_18parameters.csv", index_col=0, parse_dates=True)
all_params = df.columns.tolist()
print(f"      Loaded {len(df)} rows × {len(all_params)} parameters")
print(f"      Parameters: {', '.join(all_params)}")

# Circular parameters need sin/cos encoding
CIRCULAR_PARAMS = ['wind_direction_deg', 'current_direction_deg', 'compass_deg']
df_processed = df.copy()

for param in CIRCULAR_PARAMS:
    rad = np.deg2rad(df_processed[param])
    df_processed[f'{param}_sin'] = np.sin(rad)
    df_processed[f'{param}_cos'] = np.cos(rad)

# Remove original circular params, keep sin/cos
df_processed = df_processed.drop(columns=CIRCULAR_PARAMS)
params_for_model = df_processed.columns.tolist()
print(f"[OK] After circular encoding: {len(params_for_model)} features")

# ===== [2/6] STANDARDIZATION =====
print("\n[2/6] Standardizing...")
scaler = StandardScaler()
df_scaled = df_processed.copy()
df_scaled[:] = scaler.fit_transform(df_processed)

# Train/test split
test_start = len(df_scaled) - TEST_STEPS
train_df = df_scaled.iloc[:test_start].copy()
test_df_orig = df_processed.iloc[test_start:].copy()
test_df_scaled = df_scaled.iloc[test_start:].copy()

print(f"[OK] Train: {len(train_df)} steps ({len(train_df)/144:.1f} days)")
print(f"[OK] Test: {len(test_df_scaled)} steps ({len(test_df_scaled)/144:.1f} days)")

# ===== [3/6] BUILD WINDOWED DATASET =====
print("\n[3/6] Building windowed dataset...")

def create_windows(df, lookback, forecast, target_idx=None):
    """Create sliding windows for direct multi-step forecasting."""
    arr = df.values.astype(np.float32)
    X, Y = [], []

    for origin in range(lookback, len(arr) - forecast):
        X.append(arr[origin - lookback:origin])
        Y.append(arr[origin:origin + forecast])

    X = np.array(X, dtype=np.float32)  # (n_windows, lookback, n_features)
    Y = np.array(Y, dtype=np.float32)  # (n_windows, forecast, n_features)

    return X, Y

X_train, Y_train = create_windows(train_df, LOOKBACK, FORECAST)
print(f"[OK] Train windows: X={X_train.shape}, Y={Y_train.shape}")

# Validation split (last 10% of training windows)
n_val = max(1, int(0.1 * len(X_train)))
X_tr, Y_tr = X_train[:-n_val], Y_train[:-n_val]
X_val, Y_val = X_train[-n_val:], Y_train[-n_val:]

X_tr_t = torch.from_numpy(X_tr)
Y_tr_t = torch.from_numpy(Y_tr)
X_val_t = torch.from_numpy(X_val)
Y_val_t = torch.from_numpy(Y_val)

print(f"[OK] Train windows: {X_tr_t.shape} | Val windows: {X_val_t.shape}")

# Last window for final forecast
last_window_scaled = torch.from_numpy(train_df.values[-LOOKBACK:].astype(np.float32)).unsqueeze(0)

# ===== [4/6] BUILD TIMEXER MODEL =====
print("\n[4/6] Building TimeXer model...")

class DataEmbedding(nn.Module):
    """Patch-based embedding for time series (endogenous)."""
    def __init__(self, patch_len, n_features, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.linear_proj = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x):
        # x: (batch, lookback, n_features)
        batch_size, lookback, n_features = x.shape
        n_patches = lookback // self.patch_len

        # Reshape into patches: (batch, n_patches, patch_len * n_features)
        patches = x.reshape(batch_size, n_patches, self.patch_len * n_features)

        # Project to d_model: (batch, n_patches, d_model)
        embedded = self.linear_proj(patches)
        return embedded


class TimeXer(nn.Module):
    """TimeXer: Transformers for Time Series with Exogenous Variables.

    Architecture:
    - Endogenous embedding: patch-based tokenization of input series
    - Exogenous embedding: learned token for exogenous variables
    - Encoder: stacked transformer layers with self-attention
    - Decoder: output projection to forecast horizon
    """

    def __init__(self, lookback, n_features, forecast_len, d_model=64, n_heads=4,
                 n_layers=2, patch_len=24, dropout=0.1):
        super().__init__()

        self.lookback = lookback
        self.n_features = n_features
        self.forecast_len = forecast_len
        self.d_model = d_model
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len

        # Endogenous path: patch embedding
        self.endo_embed = DataEmbedding(patch_len, n_features, d_model)
        self.endo_pos_enc = nn.Parameter(torch.randn(self.n_patches, d_model) * 0.02)

        # Exogenous path: global token for exogenous variables
        # (In full TimeXer, this would be a sequence of exogenous covariates;
        #  here we use a learnable global exogenous token as a baseline)
        self.exo_embed = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder: hierarchical projection to forecast
        # Instead of directly projecting to forecast_len*n_features (too large),
        # use intermediate layer to compress, then expand
        intermediate_dim = min(256, forecast_len * n_features // 4)
        self.decoder_compress = nn.Linear(self.n_patches * d_model, intermediate_dim)
        self.decoder_expand = nn.Linear(intermediate_dim, forecast_len * n_features)

    def forward(self, x):
        """
        Args:
            x: (batch, lookback, n_features)
        Returns:
            out: (batch, forecast_len, n_features)
        """
        batch_size = x.shape[0]

        # Endogenous embedding with positional encoding
        endo_tok = self.endo_embed(x)  # (batch, n_patches, d_model)
        endo_tok = endo_tok + self.endo_pos_enc.unsqueeze(0)  # add positional encoding

        # Exogenous embedding (global token)
        exo_tok = self.exo_embed.unsqueeze(0).expand(batch_size, 1, -1)  # (batch, 1, d_model)

        # Concatenate endogenous and exogenous tokens
        # (In full TimeXer with sequence exogenous vars, these would be interleaved or fused)
        tok = torch.cat([endo_tok, exo_tok], dim=1)  # (batch, n_patches+1, d_model)

        # Self-attention encoder
        enc_out = self.encoder(tok)  # (batch, n_patches+1, d_model)

        # Take only endogenous patches (exclude the exogenous token for decoder)
        enc_out_endo = enc_out[:, :self.n_patches, :]  # (batch, n_patches, d_model)

        # Flatten and project to forecast (through bottleneck)
        flat = enc_out_endo.reshape(batch_size, -1)  # (batch, n_patches*d_model)
        compressed = self.decoder_compress(flat)  # (batch, intermediate_dim)
        compressed = torch.relu(compressed)
        out = self.decoder_expand(compressed)  # (batch, forecast_len*n_features)

        # Reshape to (batch, forecast_len, n_features)
        out = out.reshape(batch_size, self.forecast_len, self.n_features)

        return out


model = TimeXer(
    lookback=LOOKBACK,
    n_features=len(params_for_model),
    forecast_len=FORECAST,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    patch_len=PATCH_LEN,
    dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] TimeXer model created | Total parameters: {total_params:,}")

# ===== [5/6] TRAINING =====
print("\n[5/6] Training...")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=6
)
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
        x_batch = X_tr_t[batch_idx].to(device)
        y_batch = Y_tr_t[batch_idx].to(device)

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
        y_val_pred = model(X_val_t.to(device))
        val_loss = loss_fn(y_val_pred, Y_val_t.to(device)).item()

    scheduler.step(val_loss)

    # Logging
    elapsed = time.time() - t0
    epoch_time = time.time() - epoch_t0

    print(f"  Epoch {epoch+1:3d}/{EPOCHS} | train_loss={train_loss:.6f} | "
          f"val_loss={val_loss:.6f} | epoch_time={epoch_time:.1f}s | elapsed={elapsed:.0f}s")

    # Early stopping
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"       [BEST] val_loss: {best_val_loss:.6f}")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"       Early stopping (patience={PATIENCE})")
            break

if best_state is not None:
    model.load_state_dict(best_state)

print(f"\n[OK] Training complete | Best val_loss: {best_val_loss:.6f} | "
      f"Total time: {time.time()-t0:.1f}s")

# ===== [6/6] EVALUATION & METRICS =====
print("\n[6/6] Evaluation on test set...")

model.eval()
with torch.no_grad():
    y_test_pred_scaled = model(last_window_scaled.to(device)).cpu().numpy()[0]

# Inverse transform
y_test_pred = y_test_pred_scaled * scaler.scale_[:len(params_for_model)] + scaler.mean_[:len(params_for_model)]

# Reconstruct circular parameters
def inverse_circular(df_pred, df_orig, param_name):
    """Reconstruct original circular parameter from sin/cos."""
    if f'{param_name}_sin' in df_pred.columns and f'{param_name}_cos' in df_pred.columns:
        sin_col = df_pred[f'{param_name}_sin'].values
        cos_col = df_pred[f'{param_name}_cos'].values
        angle = np.rad2deg(np.arctan2(sin_col, cos_col)) % 360
        return angle
    return None

# Create prediction dataframe
pred_df = pd.DataFrame(y_test_pred, columns=params_for_model, index=test_df_orig.index)

# Reconstruct circular parameters
for param in CIRCULAR_PARAMS:
    angle = inverse_circular(pred_df, test_df_orig, param)
    if angle is not None:
        pred_df[param] = angle
        pred_df = pred_df.drop(columns=[f'{param}_sin', f'{param}_cos'])

# Get actual test data
actual_df = test_df_orig.copy()

# Calculate metrics
print("\n" + "="*80)
print("SKILLSCORE METRICS (vs Persistence)")
print("="*80)

def circular_mae(y_true, y_pred):
    """Circular mean absolute error for angles."""
    return np.abs((y_true - y_pred + 180) % 360 - 180).mean()

metrics_list = []
last_obs = df_processed.iloc[test_start - 1]

for param in all_params:
    if param not in pred_df.columns:
        continue

    y_true = actual_df[param].values
    y_pred = pred_df[param].values

    # Persistence baseline
    y_persist = np.repeat(last_obs[param], len(y_true))

    # MAE computation
    if param in CIRCULAR_PARAMS:
        mae_model = circular_mae(y_true, y_pred)
        mae_persist = circular_mae(y_true, y_persist)
    else:
        mae_model = mean_absolute_error(y_true, y_pred)
        mae_persist = mean_absolute_error(y_true, y_persist)

    # RMSE
    if param in CIRCULAR_PARAMS:
        rmse_model = np.nan
    else:
        rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))

    # Skill (%)
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

# Summary statistics
valid_skills = metrics_df[metrics_df['Skill_%'].notna()]['Skill_%'].values
median_skill = np.median(valid_skills)
mean_skill = np.mean(valid_skills)
n_positive = (valid_skills > 0).sum()
n_total = len(valid_skills)

print("\n" + "="*80)
print("OVERALL PERFORMANCE")
print("="*80)
print(f"Median Skill:        {median_skill:+.1f}%")
print(f"Mean Skill:          {mean_skill:+.1f}%")
print(f"Positive Skill:      {n_positive}/{n_total} parameters ({100*n_positive/n_total:.0f}%)")
print(f"Model Parameters:    {total_params:,}")
print(f"Training Time:       {time.time()-t0:.1f}s")

# Save results
pred_df.to_csv("timexer_predictions.csv")
metrics_df.to_csv("timexer_metrics.csv", index=False)

print("\n[DONE] Results saved:")
print("       - timexer_predictions.csv: 10-day forecast for all 18 parameters")
print("       - timexer_metrics.csv: MAE, RMSE, Skill% metrics")

print("\n" + "="*80)
print(f"COMPARISON TO BASELINE (Correlated Input MTGNN: +85.0%)")
print("="*80)
print(f"TimeXer Median Skill: {median_skill:+.1f}%")
print(f"Baseline:            +85.0%")
print(f"Gap:                 {median_skill - 85.0:+.1f}%")

if median_skill >= 80.0:
    print("\n[EXCELLENT] TimeXer comparable to or exceeds MTGNN baseline")
elif median_skill >= 70.0:
    print("\n[GOOD] TimeXer within 15% of MTGNN baseline")
elif median_skill >= 50.0:
    print("\n[FAIR] TimeXer underperforms MTGNN; may need architecture tuning")
else:
    print("\n[POOR] TimeXer significantly underperforms; unlikely to beat MTGNN")

print("\n" + "="*80)
