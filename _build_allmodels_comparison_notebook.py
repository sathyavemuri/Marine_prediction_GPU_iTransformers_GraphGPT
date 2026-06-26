import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h Forecast — All-Models Comparison (18 Good Parameters + 6 Reconstructed Duplicates)

A presentation-ready consolidation notebook. **No models are retrained here** — every forecast curve
used below was already produced and saved by an earlier notebook in this project; this notebook only
loads those saved forecasts, rescoring them with one consistent methodology, and reconstructs the 6
duplicate parameters from each model's own good-18 forecast (same `slope×twin + intercept` formula
used throughout this project).

**Models included with full detail (mean skill > 70% on the 18 good parameters):**
- iTransformer (the project baseline)
- PatchTST (runner-up in the original 11-model bake-off)
- RevIN-iTransformer (normalization ablation)
- Dual-Channel iTransformer (architecture ablation)
- SOFTS / STAR module (architecture ablation)
- Chronos-2 zero-shot (pretrained foundation model, no training at all)

**Models tried but excluded from full detail** (mean skill ≤ 70% on the 18 good parameters; not
re-run here — their already-measured values are cited as-is in the final verdict): LSTM, XGBoost,
N-BEATS, N-HiTS, DLinear, TiDE, TSMixer, Harmonic-Residual, DeepAR. These were all part of the
original 11-model bake-off (`Marine_Forecast_RealEMS_31Param.ipynb`) and are not retrained or
re-scored here — retraining any of them would add 5-20+ minutes each for no new information, since
their results are already on record.

Standalone — does not modify any other notebook, dashboard, or CSV in this project.""")

md("## 0. Setup")
code(r"""import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("No training in this notebook -- loading previously saved forecasts only.")""")

md("## 1. Load raw data, recompute duplicate-reconstruction coefficients\n"
   "(a train-only linear fit, identical to every prior notebook — not a retrain, just a polyfit)")
code(r"""df_10min = pd.read_csv("ems_10min_resampled.csv", index_col=0, parse_dates=True)
DUPLICATES = [
    ("airTemperature", "windChillTemperature"),
    ("tideLevel", "tidePressure"),
    ("tideLevel", "waterPressure"),
    ("tideLevel", "waterLevel"),
    ("waterTemperature", "waterTemperature_WQ"),
    ("significantWaveHeight", "maxWaveHeight"),
]
HORIZON = 288
DUP_UNITS = {
    "windChillTemperature": "°C", "tidePressure": "hPa", "waterPressure": "hPa",
    "waterLevel": "m", "waterTemperature_WQ": "°C", "maxWaveHeight": "m",
}

recon_coef = {}
for keep, drop in DUPLICATES:
    x = df_10min[keep].iloc[:-HORIZON].values
    y = df_10min[drop].iloc[:-HORIZON].values
    slope, intercept = np.polyfit(x, y, 1)
    pred_train = slope * x + intercept
    r2 = 1 - np.sum((y - pred_train) ** 2) / np.sum((y - y.mean()) ** 2)
    recon_coef[drop] = (keep, float(slope), float(intercept), float(r2))
print("Duplicate reconstruction coefficients ready (re-derived, not retrained):")
for drop, (keep, slope, intercept, r2) in recon_coef.items():
    print(f"  {drop:24s} = {slope:.4f} x {keep} + {intercept:.4f}   (train R2={r2:.5f})")""")

md("## 2. Persistence baseline (shared scoring reference for every model)")
code(r"""GOOD_PARAMS = [
    "airTemperature", "airPressure", "relativeHumidity", "dewPointTemperature",
    "windSpeed", "windDirection", "globalRadiation", "currentSpeed", "currentDirection",
    "tideLevel", "waterTemperature", "conductivity", "salinity", "significantWaveHeight",
    "significantWavePeriod", "peakWaveEnergyPeriod", "zeroCrossingPeriod", "compass",
]
CIRCULAR_PARAMS = {"windDirection", "currentDirection", "compass"}
truth = df_10min.iloc[-HORIZON:]
last_obs = df_10min.iloc[-HORIZON - 1]

def circ_mae(true, pred):
    return np.abs((true - pred + 180) % 360 - 180).mean()

persistence_mae = {}
for p in GOOD_PARAMS:
    yt = truth[p].values
    yp = np.repeat(last_obs[p], HORIZON)
    persistence_mae[p] = circ_mae(yt, yp) if p in CIRCULAR_PARAMS else mean_absolute_error(yt, yp)
for keep, drop in DUPLICATES:
    yt = truth[drop].values
    yp = np.repeat(last_obs[drop], HORIZON)
    persistence_mae[drop] = mean_absolute_error(yt, yp)
print(f"Persistence baseline computed for {len(persistence_mae)} parameters (18 good + 6 duplicates).")""")

md("## 3. Load each qualifying model's already-saved forecast, score it, reconstruct its duplicates\n"
   "Each block below reads a CSV produced by an earlier notebook — **no model is being trained here**.")
code(r"""MODEL_SOURCES = {
    "iTransformer": ("forecast_vs_actual_dualchannel.csv", "baseline"),
    "PatchTST": ("forecast_vs_actual_realdata.csv", "patchtst"),
    "RevIN-iTransformer": ("forecast_vs_actual_revin.csv", "RevIN_iTransformer"),
    "Dual-Channel iTransformer": ("forecast_vs_actual_dualchannel.csv", "dualchannel"),
    "SOFTS": ("forecast_vs_actual_softs.csv", "SOFTS"),
    "Chronos-2 (zero-shot)": ("forecast_vs_actual_chronos2.csv", "chronos2"),
}

def score(yt, yhat, is_circular):
    mae = circ_mae(yt, yhat) if is_circular else mean_absolute_error(yt, yhat)
    rmse = np.nan if is_circular else np.sqrt(mean_squared_error(yt, yhat))
    return mae, rmse

all_rows = []
model_forecasts = {}
for model_name, (csv_path, suffix) in MODEL_SOURCES.items():
    src = pd.read_csv(csv_path, parse_dates=["timestamp"])
    fc = {}
    for p in GOOD_PARAMS:
        col = f"{p}__{suffix}"
        if col not in src.columns:
            continue
        yhat = src[col].values
        yt = truth[p].values
        is_circular = p in CIRCULAR_PARAMS
        mae, rmse = score(yt, yhat, is_circular)
        skill = (1 - mae / persistence_mae[p]) * 100
        all_rows.append({"model": model_name, "parameter": p, "type": "good", "MAE": round(mae, 4),
                          "RMSE": round(rmse, 4) if rmse == rmse else np.nan, "skill_%": round(skill, 1)})
        fc[p] = yhat

    # reconstruct this model's own 6 duplicates from its own good-18 forecast
    for keep, drop in DUPLICATES:
        if keep not in fc:
            continue
        _, slope, intercept, r2 = recon_coef[drop]
        dup_pred = slope * fc[keep] + intercept
        yt = truth[drop].values
        mae, rmse = score(yt, dup_pred, False)
        skill = (1 - mae / persistence_mae[drop]) * 100
        all_rows.append({"model": model_name, "parameter": drop, "type": "duplicate", "MAE": round(mae, 4),
                          "RMSE": round(rmse, 4), "skill_%": round(skill, 1)})
        fc[drop] = dup_pred

    model_forecasts[model_name] = fc
    n_params = len([r for r in all_rows if r["model"] == model_name])
    print(f"  loaded {model_name:28s} -- scored {n_params} parameters (18 good + up to 6 duplicates)")

long_df = pd.DataFrame(all_rows)
long_df.to_csv("all_models_long_metrics.csv", index=False)
print(f"\nTotal rows: {len(long_df)}  |  Models: {long_df['model'].nunique()}")""")

md("## 4. Wide comparison table (one row per parameter, one column per model)")
code(r"""wide_df = long_df.pivot_table(index=["parameter", "type"], columns="model", values="skill_%").reset_index()
wide_df = wide_df.sort_values(["type", "parameter"])
wide_df.to_csv("all_models_wide_skill.csv", index=False)
wide_df""")

md("## 5. Per-model summary statistics")
code(r"""summary_rows = []
for model_name in MODEL_SOURCES:
    sub = long_df[long_df["model"] == model_name]
    good_sub = sub[sub["type"] == "good"]
    dup_sub = sub[sub["type"] == "duplicate"]
    summary_rows.append({
        "model": model_name,
        "mean_skill_good18_%": round(good_sub["skill_%"].mean(), 1),
        "median_skill_good18_%": round(good_sub["skill_%"].median(), 1),
        "n_good_above_70": int((good_sub["skill_%"] > 70).sum()),
        "n_good_above_80": int((good_sub["skill_%"] > 80).sum()),
        "n_good_above_persistence": int((good_sub["skill_%"] > 0).sum()),
        "mean_skill_duplicates_%": round(dup_sub["skill_%"].mean(), 1) if len(dup_sub) else np.nan,
        "n_duplicates_above_70": int((dup_sub["skill_%"] > 70).sum()),
        "n_duplicates_above_80": int((dup_sub["skill_%"] > 80).sum()),
    })
summary_df = pd.DataFrame(summary_rows).sort_values("mean_skill_good18_%", ascending=False).reset_index(drop=True)
summary_df.insert(0, "rank", summary_df.index + 1)
summary_df.to_csv("all_models_summary.csv", index=False)
summary_df""")

md("## 6. Excluded models — cited from earlier results, not retrained here\n"
   "These models scored mean skill ≤ 70% on the 18 good parameters in the original 11-model bake-off "
   "(`Marine_Forecast_RealEMS_31Param.ipynb`). Retraining any of them here would cost 5-20+ minutes "
   "each for results already on record.")
code(r"""EXCLUDED_MODELS_KNOWN_VALUES = {
    "TiDE": 65.6, "TSMixer": 51.2, "Harmonic-Residual": 46.1, "N-BEATS": 40.8,
    "N-HiTS": 40.7, "XGBoost": 40.2, "DLinear": 32.6, "LSTM": 12.6, "DeepAR": 5.8,
}
excluded_df = pd.DataFrame([
    {"model": m, "mean_skill_good18_%": v, "retrained_in_this_notebook": False,
     "source": "Marine_Forecast_RealEMS_31Param.ipynb (original 11-model bake-off)"}
    for m, v in EXCLUDED_MODELS_KNOWN_VALUES.items()
]).sort_values("mean_skill_good18_%", ascending=False).reset_index(drop=True)
excluded_df.to_csv("all_models_excluded_cited.csv", index=False)
excluded_df""")

md("## 7. The verdict")
code(r"""best_model = summary_df.iloc[0]
print(f"Best on the 18 good parameters: {best_model['model']} "
      f"(mean skill {best_model['mean_skill_good18_%']:+.1f}%, "
      f"{best_model['n_good_above_70']}/18 parameters above 70% skill, "
      f"{best_model['n_good_above_persistence']}/18 beating persistence)")
print()
print("Full ranking (qualifying models, retrained nowhere in this notebook, freshly rescored):")
print(summary_df[["rank", "model", "mean_skill_good18_%", "n_good_above_70", "mean_skill_duplicates_%"]].to_string(index=False))
print()
print("For reference, models that did not clear the 70% bar (cited, not retrained):")
print(excluded_df[["model", "mean_skill_good18_%"]].to_string(index=False))
print()
print("Note on the hard 6 (visibility x4, precipitation x2): none of the models above were tested on "
      "those parameters in this notebook -- see Marine_Forecast_RealEMS_Hybrid_iTransformer_DeepAR.ipynb "
      "(DeepAR-hybrid, +2.2% mean skill, best from-scratch result) and "
      "Marine_Forecast_RealEMS_Chronos2_ZeroShot.ipynb (Chronos-2 zero-shot, +1.5%, best zero-training result).")""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_RealEMS_AllModels_Comparison.ipynb")
print("Notebook written: Marine_Forecast_RealEMS_AllModels_Comparison.ipynb")
