"""
=============================================================================
MARINE 5-DAY FORECAST: LSTM + XGBoost with VALIDATION
=============================================================================
Trains two models to forecast all 16 marine parameters 5 days (120 hours)
ahead, then validates against held-out ground truth.

STRATEGY
--------
- Split: last 5 days (120 hours) = TEST (held-out validation)
         everything before    = TRAIN
- Direct multi-horizon recursive forecasting.
- LSTM: multivariate sequence-to-one, rolled forward 120 steps (recursive).
- XGBoost: one model per parameter, lag + calendar features, recursive roll.
- Wind direction handled with sin/cos encoding (circular).
- Metrics per parameter: MAE, RMSE, plus skill vs persistence baseline.

OUTPUTS
-------
- forecast_vs_actual.csv      : hourly predictions vs ground truth (both models)
- metrics_summary.csv         : MAE/RMSE/skill per parameter per model
- forecast_plots.png          : visual validation panels
=============================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_PATH   = "/home/claude/marine_data_75days.csv"
HORIZON     = 120        # forecast 5 days = 120 hours
LOOKBACK    = 72         # use last 72 hours (3 days) as input window
LSTM_EPOCHS = 40

# ----------------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.set_index("timestamp")

# Encode wind direction as sin/cos (circular) -> two columns
wd_rad = np.deg2rad(df["wind_direction_deg"])
df["wind_dir_sin"] = np.sin(wd_rad)
df["wind_dir_cos"] = np.cos(wd_rad)

# Feature/target columns (model on these; reconstruct direction at the end)
target_cols = [
    "significant_wave_height_m", "wave_period_s", "wind_speed_ms",
    "wind_dir_sin", "wind_dir_cos", "tidal_level_m", "current_speed_ms",
    "sea_surface_temp_c", "salinity_psu", "conductivity_mscm",
    "air_pressure_hpa", "air_temp_c", "relative_humidity_pct",
    "dew_point_c", "precipitation_mmh", "solar_radiation_wm2", "visibility_km",
]
data = df[target_cols].copy()

# Calendar features (help both models with diurnal/tidal structure)
idx = data.index
data["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
data["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
data["doy_sin"]  = np.sin(2 * np.pi * idx.dayofyear / 365)
data["doy_cos"]  = np.cos(2 * np.pi * idx.dayofyear / 365)

feature_cols = list(data.columns)            # targets + calendar
calendar_cols = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

# ----------------------------------------------------------------------------
# TRAIN / TEST SPLIT  (last HORIZON hours held out)
# ----------------------------------------------------------------------------
train_df = data.iloc[:-HORIZON].copy()
test_df  = data.iloc[-HORIZON:].copy()

print(f"Train: {train_df.shape[0]} hours ({train_df.shape[0]/24:.0f} days)")
print(f"Test : {test_df.shape[0]} hours ({test_df.shape[0]/24:.0f} days)  <- 5-day validation")

# Scale using TRAIN stats only (no leakage)
scaler = StandardScaler()
train_scaled = pd.DataFrame(
    scaler.fit_transform(train_df), columns=feature_cols, index=train_df.index
)
# Full series scaled (for building windows / recursion) using train scaler
full_scaled = pd.DataFrame(
    scaler.transform(data), columns=feature_cols, index=data.index
)

target_idx = [feature_cols.index(c) for c in target_cols]
calendar_idx = [feature_cols.index(c) for c in calendar_cols]

# Precompute the true future calendar values (known ahead of time)
future_calendar_scaled = full_scaled[calendar_cols].iloc[-HORIZON:].values

# =============================================================================
# MODEL 1: LSTM  (multivariate seq -> next-step, recursive roll for 120h)
# =============================================================================
print("\n" + "="*78)
print("TRAINING LSTM")
print("="*78)

def make_sequences(arr, lookback):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i])
        y.append(arr[i])          # predict all features next step
    return np.array(X), np.array(y)

train_arr = train_scaled.values
X_tr, y_tr = make_sequences(train_arr, LOOKBACK)
print(f"LSTM training samples: {X_tr.shape[0]}, window={LOOKBACK}, features={X_tr.shape[2]}")

lstm = Sequential([
    Input(shape=(LOOKBACK, len(feature_cols))),
    LSTM(96, return_sequences=True),
    Dropout(0.2),
    LSTM(48),
    Dropout(0.2),
    Dense(len(feature_cols)),
])
lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])
es = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")
hist = lstm.fit(
    X_tr, y_tr, validation_split=0.1, epochs=LSTM_EPOCHS,
    batch_size=32, callbacks=[es], verbose=0
)
print(f"LSTM trained. Best val_loss={min(hist.history['val_loss']):.4f}, "
      f"epochs run={len(hist.history['loss'])}")

# Recursive forecast: seed with last LOOKBACK hours of TRAIN, roll 120 steps.
# At each step we OVERRIDE the calendar channels with their known true values.
window = train_arr[-LOOKBACK:].copy()
lstm_preds_scaled = []
for h in range(HORIZON):
    x_in = window.reshape(1, LOOKBACK, -1)
    nxt = lstm.predict(x_in, verbose=0)[0]
    # inject known future calendar values
    nxt[calendar_idx] = future_calendar_scaled[h]
    lstm_preds_scaled.append(nxt)
    window = np.vstack([window[1:], nxt])
lstm_preds_scaled = np.array(lstm_preds_scaled)
lstm_preds = scaler.inverse_transform(lstm_preds_scaled)
lstm_pred_df = pd.DataFrame(lstm_preds, columns=feature_cols, index=test_df.index)

# =============================================================================
# MODEL 2: XGBoost  (DIRECT multi-horizon -- no error compounding)
# -----------------------------------------------------------------------------
# For each parameter we train a SINGLE model with the forecast step `h` as a
# feature, plus lag features anchored at the forecast origin and the known
# future calendar values at the target hour. This learns the mapping
# (recent history, lead time h, target-hour calendar) -> value at origin+h.
# Direct forecasting avoids the recursive drift seen with rolled predictions.
# =============================================================================
print("\n" + "="*78)
print("TRAINING XGBoost (direct multi-horizon, one model per parameter)")
print("="*78)

ORIGIN_LAGS = [1, 2, 3, 6, 12, 24, 48, 72]   # lags measured back from the origin

def make_direct_training(scaled_frame, target_cols, calendar_cols, lags, horizon, origin_step=4):
    """
    Build a supervised set where each row corresponds to one (origin, lead h).
    Features: <param>_lag{L} at the origin for all params, lead h, and the
              calendar columns AT the target hour (origin + h).
    Target:   value of the parameter at origin + h.
    origin_step subsamples origins to keep the training set tractable.
    """
    arr = scaled_frame
    n = len(arr)
    max_lag = max(lags)
    rows = {c: [] for c in target_cols}
    feats = []
    for origin in range(max_lag, n - horizon, origin_step):
        base = {}
        for c in target_cols:
            for L in lags:
                base[f"{c}_lag{L}"] = arr[c].iloc[origin - L]
        for h in range(1, horizon + 1):
            row = dict(base)
            row["lead_h"] = h
            for cc in calendar_cols:
                row[cc] = arr[cc].iloc[origin + h]   # known future calendar
            feats.append(row)
            for c in target_cols:
                rows[c].append(arr[c].iloc[origin + h])
    X = pd.DataFrame(feats)
    Y = {c: np.array(rows[c]) for c in target_cols}
    return X, Y

# Subsample origins (every 4 hours) -> ~12k rows, fast and accurate enough.
X_direct, Y_direct = make_direct_training(
    train_scaled, target_cols, calendar_cols, ORIGIN_LAGS, HORIZON, origin_step=4
)
print(f"Direct training rows: {X_direct.shape[0]:,}  features: {X_direct.shape[1]}")

feat_order = list(X_direct.columns)
xgb_models = {}
for c in target_cols:
    model = xgb.XGBRegressor(
        n_estimators=180, max_depth=5, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=4, objective="reg:squarederror", tree_method="hist",
    )
    model.fit(X_direct, Y_direct[c])
    xgb_models[c] = model
print(f"Trained {len(xgb_models)} direct XGBoost models.")

# Forecast: single origin = last training hour. Build base lag row once,
# then predict all 120 leads in one batch per parameter.
origin_idx = len(train_scaled) - 1
base_row = {}
for c in target_cols:
    for L in ORIGIN_LAGS:
        base_row[f"{c}_lag{L}"] = train_scaled[c].iloc[origin_idx - (L - 1)]

pred_rows = []
for h in range(1, HORIZON + 1):
    ts = test_df.index[h - 1]
    row = dict(base_row)
    row["lead_h"] = h
    for cc in calendar_cols:
        row[cc] = full_scaled.loc[ts, cc]
    pred_rows.append(row)
X_fore = pd.DataFrame(pred_rows)[feat_order]

xgb_scaled_pred = {c: xgb_models[c].predict(X_fore) for c in target_cols}
xgb_scaled_full = pd.DataFrame(index=test_df.index, columns=feature_cols, dtype=float)
for c in target_cols:
    xgb_scaled_full[c] = xgb_scaled_pred[c]
for cc in calendar_cols:
    xgb_scaled_full[cc] = full_scaled[cc].iloc[-HORIZON:].values
xgb_preds = scaler.inverse_transform(xgb_scaled_full.values)
xgb_pred_df = pd.DataFrame(xgb_preds, columns=feature_cols, index=test_df.index)

# =============================================================================
# RECONSTRUCT physical parameters (incl. wind direction from sin/cos)
# =============================================================================
def reconstruct(pred_df):
    out = pred_df.copy()
    wd = (np.rad2deg(np.arctan2(out["wind_dir_sin"], out["wind_dir_cos"])) % 360)
    out["wind_direction_deg"] = wd
    return out

lstm_final = reconstruct(lstm_pred_df)
xgb_final  = reconstruct(xgb_pred_df)

# Ground truth for the test window (original units)
truth = df.iloc[-HORIZON:].copy()

# Physical parameter list for reporting (exclude sin/cos helpers)
report_params = [
    "significant_wave_height_m", "wave_period_s", "wind_speed_ms",
    "wind_direction_deg", "tidal_level_m", "current_speed_ms",
    "sea_surface_temp_c", "salinity_psu", "conductivity_mscm",
    "air_pressure_hpa", "air_temp_c", "relative_humidity_pct",
    "dew_point_c", "precipitation_mmh", "solar_radiation_wm2", "visibility_km",
]

# =============================================================================
# METRICS  (MAE, RMSE, skill vs persistence)
# =============================================================================
def circ_mae(true, pred):
    """Angular MAE in degrees."""
    d = np.abs((true - pred + 180) % 360 - 180)
    return d.mean()

# persistence baseline = last observed value held constant over horizon
last_obs = df.iloc[-HORIZON-1]
metrics = []
for p in report_params:
    yt = truth[p].values
    yl = lstm_final[p].values
    yx = xgb_final[p].values
    yp = np.repeat(last_obs[p], HORIZON)   # persistence

    if p == "wind_direction_deg":
        mae_l, mae_x, mae_p = circ_mae(yt, yl), circ_mae(yt, yx), circ_mae(yt, yp)
        rmse_l = rmse_x = np.nan  # rmse not meaningful for circular
    else:
        mae_l = mean_absolute_error(yt, yl)
        mae_x = mean_absolute_error(yt, yx)
        mae_p = mean_absolute_error(yt, yp)
        rmse_l = np.sqrt(mean_squared_error(yt, yl))
        rmse_x = np.sqrt(mean_squared_error(yt, yx))

    skill_l = (1 - mae_l/mae_p)*100 if mae_p > 0 else np.nan
    skill_x = (1 - mae_x/mae_p)*100 if mae_p > 0 else np.nan
    metrics.append({
        "parameter": p,
        "LSTM_MAE": round(mae_l, 4), "LSTM_RMSE": round(rmse_l, 4) if rmse_l==rmse_l else np.nan,
        "XGB_MAE": round(mae_x, 4),  "XGB_RMSE": round(rmse_x, 4) if rmse_x==rmse_x else np.nan,
        "Persistence_MAE": round(mae_p, 4),
        "LSTM_skill_%": round(skill_l, 1), "XGB_skill_%": round(skill_x, 1),
        "best_model": "LSTM" if mae_l < mae_x else "XGBoost",
    })

metrics_df = pd.DataFrame(metrics)

print("\n" + "="*78)
print("VALIDATION METRICS (5-day / 120-hour held-out forecast)")
print("="*78)
print(metrics_df.to_string(index=False))

# =============================================================================
# SAVE OUTPUTS
# =============================================================================
# Forecast vs actual (long format, both models)
fva = pd.DataFrame({"timestamp": truth.index})
for p in report_params:
    fva[f"{p}__actual"] = truth[p].values
    fva[f"{p}__lstm"]   = lstm_final[p].values
    fva[f"{p}__xgb"]    = xgb_final[p].values
fva.to_csv("/home/claude/forecast_vs_actual.csv", index=False)
metrics_df.to_csv("/home/claude/metrics_summary.csv", index=False)

print("\nSaved: forecast_vs_actual.csv, metrics_summary.csv")

# =============================================================================
# PLOTS
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Show last 5 days of history + forecast window for context
hist_tail = df.iloc[-HORIZON-72:-HORIZON]

key_plots = [
    "significant_wave_height_m", "wind_speed_ms", "tidal_level_m",
    "current_speed_ms", "air_pressure_hpa", "sea_surface_temp_c",
    "relative_humidity_pct", "wave_period_s", "visibility_km",
]
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
for ax, p in zip(axes.ravel(), key_plots):
    ax.plot(hist_tail.index, hist_tail[p], color="0.5", lw=1, label="history")
    ax.plot(truth.index, truth[p], color="black", lw=2, label="actual")
    ax.plot(truth.index, lstm_final[p], color="tab:blue", lw=1.5, ls="--", label="LSTM")
    ax.plot(truth.index, xgb_final[p], color="tab:red", lw=1.5, ls=":", label="XGBoost")
    ax.axvline(truth.index[0], color="green", lw=1, alpha=0.5)
    ax.set_title(p, fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(alpha=0.3)
axes.ravel()[0].legend(fontsize=9, loc="upper left")
fig.suptitle("Marine 5-Day Forecast Validation: LSTM vs XGBoost vs Actual",
             fontsize=15, y=1.0)
fig.tight_layout()
fig.savefig("/home/claude/forecast_plots.png", dpi=120, bbox_inches="tight")
print("Saved: forecast_plots.png")

# Summary line
better = metrics_df["best_model"].value_counts()
print("\n" + "="*78)
print("SUMMARY")
print("="*78)
print(f"Parameters where LSTM wins:   {better.get('LSTM', 0)}")
print(f"Parameters where XGBoost wins: {better.get('XGBoost', 0)}")
avg_lstm_skill = metrics_df["LSTM_skill_%"].mean()
avg_xgb_skill  = metrics_df["XGB_skill_%"].mean()
print(f"Mean skill vs persistence -> LSTM: {avg_lstm_skill:.1f}%, XGBoost: {avg_xgb_skill:.1f}%")
