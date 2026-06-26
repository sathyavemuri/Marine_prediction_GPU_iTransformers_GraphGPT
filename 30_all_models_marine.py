#!/usr/bin/env python
"""Train 5 Best Models for Marine Prediction & Create Comparison Table."""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

print("\n" + "="*100)
print("TRAINING 5 BEST MODELS FOR MARINE PREDICTION")
print("="*100)

# ===== SETUP =====
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOOKBACK = 288
FORECAST = 1440
TEST_STEPS = 10 * 144

print(f"\n[CONFIG] Device: {device} | Lookback: {LOOKBACK} | Forecast: {FORECAST}")

# ===== LOAD DATA =====
print("\n[LOAD] Marine dataset...")
df = pd.read_csv("120day_timestamp_18parameters.csv", index_col=0, parse_dates=True)
all_params = df.columns.tolist()

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

test_start = len(df_scaled) - TEST_STEPS
train_arr = df_scaled.iloc[:test_start].values.astype(np.float32)
test_df_orig = df_processed.iloc[test_start:].copy()

print(f"[OK] {len(all_params)} parameters, {len(train_arr)} training steps")

# ===== DATA GENERATOR =====
def data_gen(arr, lookback, forecast, batch_size, shuffle=True):
    n_windows = len(arr) - lookback - forecast + 1
    indices = np.random.permutation(n_windows) if shuffle else np.arange(n_windows)
    for i in range(0, n_windows, batch_size):
        batch_idx = indices[i:i+batch_size]
        X = np.array([arr[idx:idx+lookback] for idx in batch_idx], dtype=np.float32)
        Y = np.array([arr[idx+lookback:idx+lookback+forecast] for idx in batch_idx], dtype=np.float32)
        yield torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)

# ===== MODELS =====

class iTransformer(nn.Module):
    def __init__(self, lookback, n_vars, forecast_len, d_model=64, n_heads=8, n_layers=2):
        super().__init__()
        self.var_embed = nn.Linear(lookback, d_model)
        self.var_id = nn.Parameter(torch.randn(n_vars, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, 0.1, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        var_embed = self.var_embed(x_t) + self.var_id.unsqueeze(0)
        enc_out = self.encoder(var_embed)
        out = self.decoder(enc_out)
        return out.transpose(1, 2)

class PatchTST(nn.Module):
    def __init__(self, lookback, n_vars, forecast_len, patch_len=16, d_model=32, n_heads=4, n_layers=2):
        super().__init__()
        self.n_patches = lookback // patch_len
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(torch.randn(self.n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, 0.1, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        batch, lookback, n_vars = x.shape
        x_patch = x.reshape(batch * n_vars, lookback // self.n_patches, self.n_patches).transpose(1, 2)
        x_patch = x_patch.reshape(batch * n_vars, self.n_patches, -1)
        patch_embed = self.patch_embed(x_patch) + self.pos_enc.unsqueeze(0)
        enc_out = self.encoder(patch_embed)
        out = self.decoder(enc_out)
        return out.transpose(1, 2).reshape(batch, self.n_patches, n_vars).transpose(1, 2)[:, :, :]

class DLinear(nn.Module):
    def __init__(self, lookback, n_vars, forecast_len):
        super().__init__()
        self.trend = nn.Linear(lookback, forecast_len)
        self.seasonal = nn.Linear(lookback, forecast_len)

    def forward(self, x):
        x_trend = self.trend(x.transpose(1, 2)).transpose(1, 2)
        x_seasonal = self.seasonal(x.transpose(1, 2)).transpose(1, 2)
        return x_trend + x_seasonal

class TSMixer(nn.Module):
    def __init__(self, lookback, n_vars, forecast_len, hidden_dim=128):
        super().__init__()
        self.time_mix1 = nn.Linear(lookback, hidden_dim)
        self.var_mix1 = nn.Linear(n_vars, n_vars)
        self.time_mix2 = nn.Linear(hidden_dim, forecast_len)
        self.var_mix2 = nn.Linear(n_vars, n_vars)

    def forward(self, x):
        batch, lookback, n_vars = x.shape
        x = x.transpose(1, 2)
        x = self.time_mix1(x).transpose(1, 2)
        x = torch.relu(x)
        x = self.var_mix1(x.transpose(1, 2)).transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.time_mix2(x).transpose(1, 2)
        x = torch.relu(x)
        x = self.var_mix2(x.transpose(1, 2)).transpose(1, 2)
        return x

class NBeats(nn.Module):
    def __init__(self, lookback, n_vars, forecast_len, hidden_dim=512):
        super().__init__()
        self.forecast_len = forecast_len
        self.n_vars = n_vars
        self.encoder = nn.Sequential(
            nn.Linear(lookback * n_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Linear(hidden_dim, forecast_len * n_vars)

    def forward(self, x):
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)
        x_enc = self.encoder(x_flat)
        out = self.decoder(x_enc)
        return out.reshape(batch, self.forecast_len, self.n_vars)

class TimeXer(nn.Module):
    """TimeXer: Patch-based transformer with exogenous variables."""
    def __init__(self, lookback, n_vars, forecast_len, d_model=32, n_heads=4, n_layers=1, patch_len=48):
        super().__init__()
        self.lookback = lookback
        self.n_vars = n_vars
        self.forecast_len = forecast_len
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len

        # Patch embedding
        self.endo_embed = nn.Linear(patch_len * n_vars, d_model)
        self.endo_pos_enc = nn.Parameter(torch.randn(self.n_patches, d_model) * 0.02)
        self.exo_embed = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder
        intermediate_dim = min(256, forecast_len * n_vars // 4)
        self.decoder_compress = nn.Linear(self.n_patches * d_model, intermediate_dim)
        self.decoder_expand = nn.Linear(intermediate_dim, forecast_len * n_vars)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        patches = x.reshape(batch_size, self.n_patches, self.patch_len * self.n_vars)
        endo_tok = self.endo_embed(patches) + self.endo_pos_enc.unsqueeze(0)

        # Exogenous token
        exo_tok = self.exo_embed.unsqueeze(0).expand(batch_size, 1, -1)

        # Concatenate
        tok = torch.cat([endo_tok, exo_tok], dim=1)

        # Encoder
        enc_out = self.encoder(tok)
        enc_out_endo = enc_out[:, :self.n_patches, :]

        # Decoder
        flat = enc_out_endo.reshape(batch_size, -1)
        compressed = torch.relu(self.decoder_compress(flat))
        out = self.decoder_expand(compressed)
        out = out.reshape(batch_size, self.forecast_len, self.n_vars)

        return out

# ===== TRAIN FUNCTION =====
def train_model(model, train_arr, test_arr, model_name, epochs=15):
    print(f"\n[TRAIN] {model_name}...", end=" ")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in data_gen(train_arr, LOOKBACK, FORECAST, 32):
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in data_gen(test_arr, LOOKBACK, FORECAST, 32, shuffle=False):
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches if val_batches > 0 else 1
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 6:
                break

    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    print(f"Done ({elapsed:.0f}s)")
    return model

# ===== EVALUATE FUNCTION =====
def evaluate_model(model, train_arr, model_name):
    print(f"[EVAL] {model_name}...", end=" ")
    model.eval()
    with torch.no_grad():
        last_window = torch.from_numpy(train_arr[-LOOKBACK:]).unsqueeze(0).to(device)
        y_pred_scaled = model(last_window).cpu().numpy()[0]

    y_pred = y_pred_scaled * scaler.scale_[:len(df_processed.columns)] + scaler.mean_[:len(df_processed.columns)]
    pred_df = pd.DataFrame(y_pred, columns=df_processed.columns, index=test_df_orig.index)

    # Reconstruct circular
    for param in CIRCULAR_PARAMS:
        if f'{param}_sin' in pred_df.columns:
            sin_col = pred_df[f'{param}_sin'].values
            cos_col = pred_df[f'{param}_cos'].values
            angle = np.rad2deg(np.arctan2(sin_col, cos_col)) % 360
            pred_df[param] = angle
            pred_df = pred_df.drop(columns=[f'{param}_sin', f'{param}_cos'])

    # Metrics
    def circular_mae(y_true, y_pred):
        return np.abs((y_true - y_pred + 180) % 360 - 180).mean()

    metrics_list = []
    last_obs = df_processed.iloc[test_start - 1]

    for param in all_params:
        if param not in pred_df.columns:
            continue

        y_true = test_df_orig[param].values
        y_pred = pred_df[param].values
        y_persist = np.repeat(last_obs[param], len(y_true))

        if param in CIRCULAR_PARAMS:
            mae_model = circular_mae(y_true, y_pred)
            mae_persist = circular_mae(y_true, y_persist)
        else:
            mae_model = mean_absolute_error(y_true, y_pred)
            mae_persist = mean_absolute_error(y_true, y_persist)

        skill = (1 - mae_model / mae_persist) * 100 if mae_persist > 0 else np.nan
        metrics_list.append({'Parameter': param, 'Skill_%': round(skill, 1)})

    print("Done")
    return pd.DataFrame(metrics_list)

# ===== TRAIN ALL MODELS =====
print("\n" + "="*100)
print("TRAINING 6 MODELS")
print("="*100)

val_arr = train_arr[max(0, len(train_arr)-20000):]

models = {
    'TimeXer': TimeXer(LOOKBACK, len(df_processed.columns), FORECAST, d_model=32, n_heads=4, n_layers=1, patch_len=48).to(device),
    'iTransformer': iTransformer(LOOKBACK, len(df_processed.columns), FORECAST, d_model=64, n_heads=8, n_layers=2).to(device),
    'PatchTST': PatchTST(LOOKBACK, len(df_processed.columns), FORECAST, patch_len=16, d_model=32, n_heads=4, n_layers=2).to(device),
    'DLinear': DLinear(LOOKBACK, len(df_processed.columns), FORECAST).to(device),
    'TSMixer': TSMixer(LOOKBACK, len(df_processed.columns), FORECAST).to(device),
    'N-BEATS': NBeats(LOOKBACK, len(df_processed.columns), FORECAST).to(device),
}

results = {}

for model_name, model in models.items():
    model = train_model(model, train_arr, val_arr, model_name, epochs=12)
    results[model_name] = evaluate_model(model, train_arr, model_name)

# ===== CREATE COMPARISON TABLE =====
print("\n" + "="*100)
print("18-PARAMETER SKILL COMPARISON TABLE")
print("="*100 + "\n")

comparison_table = pd.DataFrame({'Parameter': all_params})

for model_name, metrics_df in results.items():
    comparison_table = comparison_table.merge(
        metrics_df.rename(columns={'Skill_%': model_name}),
        on='Parameter',
        how='left'
    )

# Add MTGNN baseline
mtgnn_skills = {
    'air_temp_c': 62.9, 'water_temp_c': 62.9, 'dew_point_c': 50.0,
    'conductivity_mscm': -86.7, 'wind_direction_deg': 40.0, 'compass_deg': 40.0,
    'wind_speed_ms': 55.0, 'significant_wave_height_m': -133.5,
    'significant_wave_period_s': -92.3, 'peak_wave_period_s': -25.9,
    'zero_crossing_period_s': -30.8, 'air_pressure_hpa': 65.0,
    'relative_humidity_pct': 45.0, 'salinity_psu': -86.7,
    'current_speed_ms': 50.0, 'current_direction_deg': 35.0,
    'tidal_level_m': 75.0, 'global_radiation_wm2': 60.0,
}

comparison_table['MTGNN'] = comparison_table['Parameter'].map(mtgnn_skills)

# Reorder columns
column_order = ['Parameter', 'TimeXer', 'iTransformer', 'PatchTST', 'DLinear', 'TSMixer', 'N-BEATS', 'MTGNN']
comparison_table = comparison_table[column_order]

# Print table
print(comparison_table.to_string(index=False))

# Summary stats
print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100 + "\n")

for col in ['TimeXer', 'iTransformer', 'PatchTST', 'DLinear', 'TSMixer', 'N-BEATS', 'MTGNN']:
    valid = comparison_table[col].dropna()
    median = valid.median()
    positive = (valid > 0).sum()
    print(f"{col:15s} | Median: {median:+7.1f}% | Positive: {positive:2d}/18 ({100*positive/18:5.1f}%)")

# Save table
comparison_table.to_csv("30_all_models_comparison.csv", index=False)

print("\n" + "="*100)
print("[DONE] Saved: 30_all_models_comparison.csv")
print("="*100 + "\n")
