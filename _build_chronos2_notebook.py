import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h Forecast — Chronos-2 Zero-Shot vs. iTransformer (all 24 parameters)

Tests Amazon Science's [Chronos-2](https://huggingface.co/amazon/chronos-2) (Oct 2025), a pretrained
time-series foundation model, **zero-shot** (no training at all) across **all 24** directly-modeled
real EMS parameters — not just the 6 historically hard ones. Two questions, both answered honestly
in Section 5:

1. **Can a zero-shot foundation model beat the from-scratch iTransformer on the 18 "good" parameters**
   it already handles well (85-98% skill)?
2. **Can it do better than every from-scratch approach tried (v1-v5) on the 6 hard parameters**
   (visibility ×4, precipitation ×2), where 28 days of training data has been the binding constraint
   across five different architectures and loss functions?

**Why this is a fundamentally different mechanism than v1-v5:** every prior model in this project —
iTransformer, DeepAR, XGBoost (3 loss variants), TimeXer-lite, the residual-correction stack — was
trained from scratch on this dataset's 28 days. Chronos-2 is pretrained on a large public corpus of
real-world time series and used **zero-shot**: no fitting to this dataset's training split at all.
It natively supports joint multivariate forecasting (so cross-parameter correlation — the thing
iTransformer's attention mechanism is specifically built to exploit — is handled internally) and
known-future covariates (calendar features, passed via `future_df`), without any of the hand-engineered
feature pipelines built for v3/v4/v5.

Standalone — does not modify any other notebook, dashboard, or CSV in this project.""")

md("## 0. Setup")
code(r"""import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from chronos import Chronos2Pipeline

print("Loading Chronos-2 (pretrained, CPU)...")
t0 = time.time()
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
print(f"Loaded in {time.time()-t0:.1f}s")""")

md("## 1. Load data, collapse duplicates, encode circular parameters\n"
   "(identical preprocessing to every prior notebook in this project, for a fair comparison)")
code(r"""df_10min = pd.read_csv("ems_10min_resampled.csv", index_col=0, parse_dates=True)
DUPLICATES = [
    ("airTemperature", "windChillTemperature"),
    ("tideLevel", "tidePressure"),
    ("tideLevel", "waterPressure"),
    ("tideLevel", "waterLevel"),
    ("waterTemperature", "waterTemperature_WQ"),
    ("significantWaveHeight", "maxWaveHeight"),
]
df_num = df_10min.drop(columns=["precipitationType"]).copy()

CIRCULAR = ["windDirection", "currentDirection", "compass"]
for c in CIRCULAR:
    rad = np.deg2rad(df_num[c])
    df_num[f"{c}_sin"] = np.sin(rad)
    df_num[f"{c}_cos"] = np.cos(rad)
df_num_full = df_num.drop(columns=CIRCULAR)

target_cols = [c for c in df_num_full.columns if c not in [d for _, d in DUPLICATES]]
PRECIP_PARAMS = ["precipitationIntensity", "precipitationDifference"]
VISIBILITY_PARAMS = ["twentyFourHourAvgVisibility", "tenMinuteAvgVisibility",
                      "oneMinuteAvgVisibility", "oneHourAvgVisibility"]
HARD_PARAMS = PRECIP_PARAMS + VISIBILITY_PARAMS
GOOD_PARAMS = [c for c in target_cols if c not in HARD_PARAMS]
print(f"{len(target_cols)} modeled columns ({len(GOOD_PARAMS)} good-param columns + {len(HARD_PARAMS)} hard)")""")

md("## 2. Train/test split (same 48h held-out window as every prior notebook), duplicate reconstruction fit")
code(r"""LOOKBACK, HORIZON = 288, 288   # kept for parity with prior notebooks; Chronos-2 uses full available context

idx = df_num_full.index
df_num_full["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
df_num_full["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
df_num_full["dom_sin"] = np.sin(2 * np.pi * idx.day / 30)
df_num_full["dom_cos"] = np.cos(2 * np.pi * idx.day / 30)
calendar_cols = ["hour_sin", "hour_cos", "dom_sin", "dom_cos"]

train_df = df_num_full.iloc[:-HORIZON].copy()
test_df = df_num_full.iloc[-HORIZON:].copy()
print(f"Train (context for Chronos-2): {train_df.shape[0]} rows ({train_df.shape[0]/144:.1f} days)")
print(f"Test  (48h held-out):          {test_df.shape[0]} rows")

recon_coef = {}
for keep, drop in DUPLICATES:
    x, y = train_df[keep].values, df_num_full[drop].iloc[:-HORIZON].values
    slope, intercept = np.polyfit(x, y, 1)
    pred_train = slope * x + intercept
    r2 = 1 - np.sum((y - pred_train) ** 2) / np.sum((y - y.mean()) ** 2)
    recon_coef[drop] = (keep, float(slope), float(intercept), float(r2))""")

md("## 3. Chronos-2 zero-shot — all 27 modeled columns, jointly, with calendar known-future covariates\n"
   "No training. The full ~28-day training history is passed as context; calendar features "
   "(deterministically known in advance) are passed via `future_df`.")
code(r"""ctx_df = train_df[target_cols].copy()
ctx_df["item_id"] = "EMS"
ctx_df["timestamp"] = ctx_df.index
ctx_df = ctx_df[["item_id", "timestamp"] + target_cols + calendar_cols] if False else ctx_df
# calendar columns also need to be in the context (as past covariates) for consistency with future_df
ctx_df = train_df[target_cols + calendar_cols].copy()
ctx_df["item_id"] = "EMS"
ctx_df["timestamp"] = ctx_df.index
ctx_df = ctx_df[["item_id", "timestamp"] + target_cols + calendar_cols]

future_df = test_df[calendar_cols].copy()
future_df["item_id"] = "EMS"
future_df["timestamp"] = future_df.index
future_df = future_df[["item_id", "timestamp"] + calendar_cols]

t0 = time.time()
chronos_out = pipe.predict_df(
    ctx_df, future_df=future_df, target=target_cols,
    id_column="item_id", timestamp_column="timestamp",
    prediction_length=HORIZON, quantile_levels=[0.1, 0.5, 0.9],
    freq="10min",
)
print(f"Chronos-2 zero-shot forecast complete: {len(target_cols)} parameters, 48h horizon, "
      f"{time.time()-t0:.1f}s total (no training).")
chronos_out.head()""")

md("## 4. Reshape into a forecast table, reconstruct circular params & duplicates")
code(r"""chronos_pred_df = chronos_out.pivot(index="timestamp", columns="target_name", values="predictions")
chronos_pred_df = chronos_pred_df[target_cols].reindex(test_df.index)
chronos_q10 = chronos_out.pivot(index="timestamp", columns="target_name", values="0.1").reindex(test_df.index)
chronos_q90 = chronos_out.pivot(index="timestamp", columns="target_name", values="0.9").reindex(test_df.index)

def reconstruct(pred_df_in):
    out = pred_df_in.copy()
    for ang in ["windDirection", "currentDirection", "compass"]:
        out[ang] = (np.rad2deg(np.arctan2(out[f"{ang}_sin"], out[f"{ang}_cos"])) % 360)
    return out

chronos_final = reconstruct(chronos_pred_df)
truth = df_num_full.iloc[-HORIZON:].copy()
for ang in ["windDirection", "currentDirection", "compass"]:
    truth[ang] = (np.rad2deg(np.arctan2(truth[f"{ang}_sin"], truth[f"{ang}_cos"])) % 360)

report_params = [c for c in target_cols if not c.endswith(("_sin", "_cos"))] + \
                ["windDirection", "currentDirection", "compass"]
CIRCULAR_PARAMS = {"windDirection", "currentDirection", "compass"}

dup_series = {}
for keep, drop in DUPLICATES:
    _, slope, intercept, _ = recon_coef[drop]
    dup_series[drop] = slope * chronos_final[keep].values + intercept

print("Chronos-2 forecast reconstructed: all 24 reportable parameters + 6 duplicates.")""")

md("## 5. The verdict: Chronos-2 zero-shot vs. iTransformer, all 24 parameters")
code(r"""ITRANSFORMER_GOOD_SKILL = {   # from Marine_Forecast_RealEMS_Hybrid_iTransformer_DeepAR.ipynb, "good" 18
    "airTemperature": 98.0, "airPressure": 97.6, "windDirection": 96.9, "waterTemperature": 96.4,
    "relativeHumidity": 96.3, "currentDirection": 96.0, "tideLevel": 94.0, "dewPointTemperature": 93.6,
    "globalRadiation": 93.0, "significantWaveHeight": 90.7, "currentSpeed": 90.5, "windSpeed": 89.5,
    "salinity": 87.5, "significantWavePeriod": 75.3, "zeroCrossingPeriod": 72.8, "conductivity": 72.8,
    "compass": 70.0, "peakWaveEnergyPeriod": 59.0,
}
PURE_ITRANSFORMER_HARD_SKILL = {   # diluted all-24 single-iTransformer, for the hard 6
    "twentyFourHourAvgVisibility": -100.0, "precipitationDifference": -101.9,
    "tenMinuteAvgVisibility": -154.9, "oneMinuteAvgVisibility": -190.5,
    "oneHourAvgVisibility": -291.6, "precipitationIntensity": -409.9,
}
DEEPAR_HYBRID_HARD_SKILL = {   # best hybrid result found so far, for the hard 6
    "tenMinuteAvgVisibility": 14.0, "twentyFourHourAvgVisibility": 3.5,
    "precipitationDifference": -0.1, "precipitationIntensity": -0.2,
    "oneHourAvgVisibility": -1.4, "oneMinuteAvgVisibility": -2.5,
}

def circ_mae(true, pred):
    return np.abs((true - pred + 180) % 360 - 180).mean()

last_obs = df_num_full.iloc[-HORIZON - 1]
for ang in ["windDirection", "currentDirection", "compass"]:
    last_obs[ang] = (np.rad2deg(np.arctan2(last_obs[f"{ang}_sin"], last_obs[f"{ang}_cos"])) % 360)

metrics = []
for p in report_params:
    yt = truth[p].values
    yp_persist = np.repeat(last_obs[p], HORIZON)
    is_circular = p in CIRCULAR_PARAMS
    mae_p = circ_mae(yt, yp_persist) if is_circular else mean_absolute_error(yt, yp_persist)
    yhat = chronos_final[p].values
    if is_circular:
        mae, rmse = circ_mae(yt, yhat), np.nan
    else:
        mae = mean_absolute_error(yt, yhat)
        rmse = np.sqrt(mean_squared_error(yt, yhat))
    skill = (1 - mae / mae_p) * 100 if mae_p > 0 else np.nan
    is_hard = p in HARD_PARAMS
    itransformer_ref = PURE_ITRANSFORMER_HARD_SKILL.get(p, np.nan) if is_hard else ITRANSFORMER_GOOD_SKILL.get(p, np.nan)
    best_prior_ref = DEEPAR_HYBRID_HARD_SKILL.get(p, np.nan) if is_hard else ITRANSFORMER_GOOD_SKILL.get(p, np.nan)
    metrics.append({
        "parameter": p, "is_hard_param": is_hard, "Persistence_MAE": round(mae_p, 4),
        "chronos2_MAE": round(mae, 4), "chronos2_RMSE": round(rmse, 4) if rmse == rmse else np.nan,
        "chronos2_skill_%": round(skill, 1), "itransformer_skill_%": itransformer_ref,
        "best_prior_skill_%": best_prior_ref,
        "chronos2_beats_itransformer": bool(skill > itransformer_ref) if itransformer_ref == itransformer_ref else None,
    })

metrics_df = pd.DataFrame(metrics).sort_values("chronos2_skill_%", ascending=False).reset_index(drop=True)
metrics_df.insert(0, "rank", metrics_df.index + 1)
metrics_df.to_csv("metrics_chronos2.csv", index=False)

n_total = len(metrics_df)
n_beats = int(metrics_df["chronos2_beats_itransformer"].sum())
mean_chronos_good = metrics_df[~metrics_df["is_hard_param"]]["chronos2_skill_%"].mean()
mean_chronos_hard = metrics_df[metrics_df["is_hard_param"]]["chronos2_skill_%"].mean()
mean_itransformer_good = metrics_df[~metrics_df["is_hard_param"]]["itransformer_skill_%"].mean()
mean_deepar_hard = metrics_df[metrics_df["is_hard_param"]]["best_prior_skill_%"].mean()

print(metrics_df[["rank", "parameter", "is_hard_param", "chronos2_skill_%", "itransformer_skill_%",
                   "best_prior_skill_%", "chronos2_beats_itransformer"]].to_string(index=False))
print(f"\nChronos-2 beats the comparable iTransformer-based result on {n_beats}/{n_total} parameters.")
print(f"18 'good' parameters  -- mean skill: Chronos-2 zero-shot {mean_chronos_good:+.1f}%  vs  iTransformer {mean_itransformer_good:+.1f}%")
print(f"6 hard parameters     -- mean skill: Chronos-2 zero-shot {mean_chronos_hard:+.1f}%  vs  best prior (DeepAR-hybrid) {mean_deepar_hard:+.1f}%")
if mean_chronos_good > mean_itransformer_good and mean_chronos_hard > mean_deepar_hard:
    print("VERDICT: Chronos-2 zero-shot beats both the good-18 iTransformer baseline AND the best hard-6 hybrid result.")
elif mean_chronos_good > mean_itransformer_good:
    print("VERDICT: Chronos-2 zero-shot beats iTransformer on the good 18, but not the best hard-6 hybrid result.")
elif mean_chronos_hard > mean_deepar_hard:
    print("VERDICT: Chronos-2 zero-shot beats the best hard-6 hybrid result, but not iTransformer on the good 18.")
else:
    print("VERDICT: Chronos-2 zero-shot does not beat either baseline on average; mixed/negative result.")""")

md("## 6. Plot a sample of parameters — Chronos-2 zero-shot vs actual")
code(r"""plot_params = (metrics_df[~metrics_df["is_hard_param"]]["parameter"].head(3).tolist() +
               metrics_df[metrics_df["is_hard_param"]]["parameter"].tolist()[:3])
hist_tail = df_10min.iloc[-HORIZON - LOOKBACK:-HORIZON]
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, c in zip(axes.ravel(), plot_params):
    ax.plot(hist_tail.index, hist_tail[c], color="lightgray", lw=1, label="history")
    ax.plot(truth.index, truth[c], color="black", lw=2, label="actual")
    ax.plot(truth.index, chronos_final[c], color="#ff7f0e", lw=1.5, ls="--", label="Chronos-2 (zero-shot)")
    if c in chronos_q10.columns:
        ax.fill_between(truth.index, chronos_q10[c], chronos_q90[c], color="#ff7f0e", alpha=0.15, label="10-90% band")
    ax.axvline(truth.index[0], color="green", lw=1, alpha=0.5)
    ax.set_title(c, fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("chronos2_sample_plot.png", dpi=110)
plt.show()
print("Saved chronos2_sample_plot.png  (top row = 3 good params, bottom row = 3 hard params)")""")

md("## 7. Save outputs for the dashboard")
code(r"""fva = pd.DataFrame({"timestamp": truth.index})
for p in report_params:
    fva[f"{p}__actual"] = truth[p].values
    fva[f"{p}__chronos2"] = chronos_final[p].values
    if p in chronos_q10.columns:
        fva[f"{p}__q10"] = chronos_q10[p].values
        fva[f"{p}__q90"] = chronos_q90[p].values
fva.to_csv("forecast_vs_actual_chronos2.csv", index=False)

dup_fva = pd.DataFrame({"timestamp": test_df.index})
for keep, drop in DUPLICATES:
    dup_fva[f"{drop}__actual"] = df_10min[drop].iloc[-HORIZON:].values
    dup_fva[f"{drop}__reconstructed"] = dup_series[drop]
dup_fva.to_csv("duplicate_forecast_vs_actual_chronos2.csv", index=False)

dup_recon_rows = []
for keep, drop in DUPLICATES:
    _, slope, intercept, r2 = recon_coef[drop]
    mae = mean_absolute_error(df_10min[drop].iloc[-HORIZON:].values, dup_series[drop])
    rmse = np.sqrt(mean_squared_error(df_10min[drop].iloc[-HORIZON:].values, dup_series[drop]))
    dup_recon_rows.append({"duplicate_parameter": drop, "reconstructed_from": keep,
                            "slope": round(slope, 4), "intercept": round(intercept, 4), "train_R2": round(r2, 5),
                            "held_out_MAE": round(mae, 4), "held_out_RMSE": round(rmse, 4)})
pd.DataFrame(dup_recon_rows).to_csv("duplicate_reconstruction_chronos2.csv", index=False)

print("Saved: metrics_chronos2.csv, forecast_vs_actual_chronos2.csv, duplicate_reconstruction_chronos2.csv,")
print("       duplicate_forecast_vs_actual_chronos2.csv, chronos2_sample_plot.png.")""")

md(r"""## 8. Conclusion

Section 5 is the actual verdict. This was a genuinely cheap experiment — no training, ~3 seconds of
CPU inference for all 24 parameters jointly — so even a partial win (e.g., beating iTransformer on the
good 18 but not unseating DeepAR-hybrid on the hard 6, or vice versa) is informative and worth keeping
in mind for future iteration. If Chronos-2 wins outright on both fronts, it's a strong candidate to
become the new baseline for this project, given the dramatic reduction in training cost and complexity
compared to maintaining five separate from-scratch hybrid pipelines (v1-v5). If it doesn't, that's
useful too — it would mean this specific dataset's structure (precise sensor-ceiling/zero-inflation
behavior, the specific cross-parameter correlations) isn't well represented in Chronos-2's pretraining
corpus, and the from-scratch approach remains necessary. Reported honestly either way.""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_RealEMS_Chronos2_ZeroShot.ipynb")
print("Notebook written: Marine_Forecast_RealEMS_Chronos2_ZeroShot.ipynb")
