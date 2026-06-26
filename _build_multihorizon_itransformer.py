import nbformat as nbf
import time

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h+ Multi-Horizon Forecast — iTransformer across 2–7 days

Test iTransformer's skill degradation as horizon extends from 2 days (28-step lookback × 14 = 4,032 training rows)
to 7 days (98-step lookback × 14 = 14,112 training rows). Validates the 14× ratio scaling rule across all 24 parameters
(18 good + 6 reconstructed duplicates).

**Data:** 120 days at 1-minute resolution, resampled to 10-minute (17,280 rows).
**Training ratios:** 14× horizon for each tested length (2d→28d train, 3d→42d train, ..., 7d→98d train).
**Test windows:** Non-overlapping final N days per horizon.
**Architecture:** iTransformer (same config across all horizons, only horizon changes).""")

md("## 0. Setup & load data")
code(r"""import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")
torch.set_num_threads(8)

print("PyTorch:", torch.__version__, "| torch threads:", torch.get_num_threads())

# Load 120-day data at 1-min resolution
df_1min = pd.read_csv("marine_data_120days_1min.csv", index_col=0, parse_dates=True)
print(f"Loaded: {df_1min.shape[0]} rows at 1-min resolution ({df_1min.shape[0] / (24*60):.1f} days)")

# Drop categorical column (precip_type) before resampling
df_1min = df_1min.drop(columns=['precip_type'], errors='ignore')

# Resample to 10-min
df_10min = df_1min.resample("10min").mean()
df_10min = df_10min.dropna()
print(f"Resampled to 10-min: {df_10min.shape[0]} rows ({df_10min.shape[0] / (24*6):.1f} days)")

# Map CSV columns (snake_case) to parameter names
CSV_COL_MAP = {
    "air_temp_c": "airTemperature",
    "air_pressure_hpa": "airPressure",
    "relative_humidity_pct": "relativeHumidity",
    "dew_point_c": "dewPointTemperature",
    "wind_chill_c": "windChillTemperature",
    "wind_speed_ms": "windSpeed",
    "wind_direction_deg": "windDirection",
    "compass_deg": "compass",
    "global_radiation_wm2": "globalRadiation",
    "current_speed_ms": "currentSpeed",
    "current_direction_deg": "currentDirection",
    "water_pressure_dbar": "waterPressure",
    "tide_pressure_dbar": "tidePressure",
    "tidal_level_m": "tideLevel",
    "water_temp_c": "waterTemperature",
    "conductivity_mscm": "conductivity",
    "salinity_psu": "salinity",
    "water_temp_quality_c": "waterTemperature_WQ",
    "significant_wave_height_m": "significantWaveHeight",
    "max_wave_height_m": "maxWaveHeight",
    "water_level_m": "waterLevel",
    "significant_wave_period_s": "significantWavePeriod",
    "peak_wave_period_s": "peakWaveEnergyPeriod",
    "zero_crossing_period_s": "zeroCrossingPeriod",
}

# Rename columns
df_10min = df_10min.rename(columns=CSV_COL_MAP)

# Parameters (18 good + 6 duplicates)
GOOD_PARAMS = [
    "airTemperature", "airPressure", "relativeHumidity", "dewPointTemperature",
    "windSpeed", "windDirection", "globalRadiation", "currentSpeed", "currentDirection",
    "tideLevel", "waterTemperature", "conductivity", "salinity", "significantWaveHeight",
    "significantWavePeriod", "peakWaveEnergyPeriod", "zeroCrossingPeriod", "compass",
]
DUPLICATES = [
    ("airTemperature", "windChillTemperature"),
    ("tideLevel", "tidePressure"),
    ("tideLevel", "waterPressure"),
    ("tideLevel", "waterLevel"),
    ("waterTemperature", "waterTemperature_WQ"),
    ("significantWaveHeight", "maxWaveHeight"),
]
DUP_PARAMS = [d[1] for d in DUPLICATES]
ALL_PARAMS = GOOD_PARAMS + DUP_PARAMS

# Add calendar features
idx = df_10min.index
df_10min["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
df_10min["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
df_10min["dom_sin"] = np.sin(2 * np.pi * idx.day / 30)
df_10min["dom_cos"] = np.cos(2 * np.pi * idx.day / 30)
calendar_cols = ["hour_sin", "hour_cos", "dom_sin", "dom_cos"]

print(f"All params: {len(ALL_PARAMS)} (18 good + 6 duplicates)")""")

md("## 1. Multi-horizon training configuration")
code(r"""# Horizons and corresponding training data (14x ratio)
HORIZON_CONFIGS = [
    {"horizon_days": 2, "horizon_steps": 288, "train_days": 28, "train_steps": 4032},
    {"horizon_days": 3, "horizon_steps": 432, "train_days": 42, "train_steps": 6048},
    {"horizon_days": 4, "horizon_steps": 576, "train_days": 56, "train_steps": 8064},
    {"horizon_days": 5, "horizon_steps": 720, "train_days": 70, "train_steps": 10080},
    {"horizon_days": 6, "horizon_steps": 864, "train_days": 84, "train_steps": 12096},
    {"horizon_days": 7, "horizon_steps": 1008, "train_days": 98, "train_steps": 14112},
]

print("Horizon configs (14x ratio rule):")
for cfg in HORIZON_CONFIGS:
    print(f"  {cfg['horizon_days']:1d}d: {cfg['train_steps']:5d} train steps ({cfg['train_days']:2d}d) → {cfg['horizon_steps']:4d} step forecast")

results_by_horizon = {}""")

md("## 2. iTransformer architecture (same for all horizons)")
code(r"""class iTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=2, horizon=288, d_ff=512, dropout=0.1, n_params=24):
        super().__init__()
        self.d_model = d_model
        self.n_params = n_params
        self.horizon = horizon

        # Project each parameter's lookback window to d_model
        self.param_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_params, d_model))

        # Transformer: attention ACROSS parameters (inverted)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decode to forecast
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (batch, lookback, n_params)
        batch_size = x.shape[0]
        # Compress each parameter's lookback to one vector
        x_compressed = self.param_proj(x.mean(dim=1, keepdim=True).transpose(1, 2))  # (B, n_params, d_model)
        x_compressed = x_compressed + self.pos_emb
        # Attention across parameters
        x_attn = self.transformer(x_compressed)  # (B, n_params, d_model)
        # Forecast all parameters
        out = self.head(x_attn)  # (B, n_params, horizon)
        return out

print("iTransformer model defined.")""")

md("## 3. Training loop (run for each horizon)")
code(r"""def train_horizon(df, config, all_params, calendar_cols):
    horizon_days = config["horizon_days"]
    horizon_steps = config["horizon_steps"]
    train_steps = config["train_steps"]

    print(f"\n{'='*70}")
    print(f"Training for {horizon_days}-day horizon ({horizon_steps} steps forward)")
    print(f"{'='*70}")

    # Split: last train_steps for training, next horizon_steps for testing
    test_start_idx = len(df) - horizon_steps
    train_end_idx = test_start_idx
    train_start_idx = train_end_idx - train_steps

    train_df = df.iloc[train_start_idx:train_end_idx].copy()
    test_df = df.iloc[test_start_idx:].copy()

    print(f"Train: {train_df.shape[0]} rows | Test: {test_df.shape[0]} rows")

    # Standardize (per parameter, on training data only)
    param_stats = {}
    for p in all_params:
        param_stats[p] = {
            "mean": train_df[p].mean(),
            "std": train_df[p].std()
        }
        train_df[p] = (train_df[p] - param_stats[p]["mean"]) / param_stats[p]["std"]

    # Build training windows: sliding window of lookback=horizon_steps
    lookback = horizon_steps
    X_train, Y_train = [], []
    for i in range(lookback, len(train_df) - horizon_steps, 2):  # stride=2 to reduce data
        x = train_df[all_params].iloc[i - lookback:i].values.astype(np.float32)
        y = train_df[all_params].iloc[i:i + horizon_steps].values.astype(np.float32)
        X_train.append(x)
        Y_train.append(y)

    X_train = np.array(X_train)  # (n_windows, lookback, n_params)
    Y_train = np.array(Y_train)  # (n_windows, horizon, n_params)
    print(f"Training windows: {X_train.shape[0]}")

    # Validation split
    n_val = max(1, int(0.1 * len(X_train)))
    perm = np.random.permutation(len(X_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, Y_tr = X_train[tr_idx], Y_train[tr_idx]
    X_val, Y_val = X_train[val_idx], Y_train[val_idx]

    # Model
    model = iTransformer(d_model=128, n_heads=8, n_layers=2, horizon=horizon_steps, n_params=len(all_params))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Training
    t_start = time.time()
    X_tr_t = torch.from_numpy(X_tr).to(device)
    Y_tr_t = torch.from_numpy(Y_tr).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience, wait = 20, 0

    for ep in range(100):
        model.train()
        perm_b = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 32):
            b = perm_b[i:i+32]
            opt.zero_grad()
            pred = model(X_tr_t[b])
            loss = criterion(pred, Y_tr_t[b])
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
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    t_train = time.time() - t_start
    print(f"Training time: {t_train:.1f}s ({t_train/60:.1f}m)")

    # Forecast on test window
    model.eval()
    t_infer = time.time()

    # Use last lookback steps from train data as context, forecast test
    last_context = train_df[all_params].iloc[-lookback:].values.astype(np.float32)
    X_test = torch.from_numpy(last_context).unsqueeze(0).to(device)

    with torch.no_grad():
        Y_pred_norm = model(X_test)[0].cpu().numpy()  # (horizon, n_params)

    t_infer = time.time() - t_infer
    print(f"Inference time: {t_infer*1000:.1f}ms")

    # Inverse normalization
    Y_pred = np.zeros_like(Y_pred_norm)
    for j, p in enumerate(all_params):
        Y_pred[:, j] = Y_pred_norm[:, j] * param_stats[p]["std"] + param_stats[p]["mean"]

    # Evaluate
    Y_true = test_df[all_params].iloc[:horizon_steps].values
    last_obs = df[all_params].iloc[-horizon_steps - 1].values
    Y_persist = np.tile(last_obs, (horizon_steps, 1))

    metrics = []
    for j, p in enumerate(all_params):
        y_true = Y_true[:, j]
        y_pred = Y_pred[:, j]
        y_persist = Y_persist[:, j]

        mae_p = mean_absolute_error(y_true, y_persist)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        skill = (1 - mae / mae_p) * 100 if mae_p > 0 else np.nan

        metrics.append({
            "parameter": p,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "skill_%": round(skill, 1),
            "persistence_MAE": round(mae_p, 4),
        })

    metrics_df = pd.DataFrame(metrics)
    mean_skill = metrics_df["skill_%"].mean()
    print(f"Mean skill: {mean_skill:+.1f}%")

    return {
        "config": config,
        "metrics": metrics_df,
        "forecast": Y_pred,
        "actual": Y_true,
        "train_time": t_train,
        "infer_time": t_infer,
        "mean_skill": mean_skill,
    }

# Train all horizons
for cfg in HORIZON_CONFIGS:
    h = cfg["horizon_days"]
    result = train_horizon(df_10min, cfg, ALL_PARAMS, calendar_cols)
    results_by_horizon[h] = result
    result["metrics"].to_csv(f"metrics_horizon_{h}d.csv", index=False)
    np.save(f"forecast_{h}d.npy", result["forecast"])
    np.save(f"actual_{h}d.npy", result["actual"])

print("\n" + "="*70)
print("All horizons trained and saved.")
print("="*70)""")

md("## 4. CPU & Timing summary table")
code(r"""timing_data = []
for h in [2, 3, 4, 5, 6, 7]:
    r = results_by_horizon[h]
    cfg = r["config"]
    timing_data.append({
        "Horizon": f"{h}d",
        "Train steps": cfg["train_steps"],
        "Test steps": cfg["horizon_steps"],
        "Training time (s)": round(r["train_time"], 1),
        "Training time (min)": round(r["train_time"] / 60, 2),
        "Inference time (ms)": round(r["infer_time"] * 1000, 2),
        "Mean skill (%)": round(r["mean_skill"], 1),
    })

timing_df = pd.DataFrame(timing_data)
timing_df.to_csv("timing_multihorizon.csv", index=False)
print(timing_df.to_string(index=False))""")

md("## 5. Conclusion")
code(r"""total_train_time = sum(results_by_horizon[h]['train_time'] for h in [2,3,4,5,6,7])
avg_skill = np.mean([results_by_horizon[h]['mean_skill'] for h in [2,3,4,5,6,7]])
print(f"Total training time for all 6 horizons: {total_train_time:.1f}s")
print(f"Average skill across horizons: {avg_skill:.1f}%")
print("\nAll metrics and forecasts saved. Ready for dashboard.")""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_MultiHorizon_iTransformer.ipynb")
print("Notebook written: Marine_Forecast_MultiHorizon_iTransformer.ipynb")
