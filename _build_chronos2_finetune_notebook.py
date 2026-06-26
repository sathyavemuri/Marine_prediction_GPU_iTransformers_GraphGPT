import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h Forecast — Chronos-2 LoRA Fine-Tuned (all 24 parameters)

`Marine_Forecast_RealEMS_Chronos2_ZeroShot.ipynb` showed Chronos-2 used **zero-shot** (no training)
nearly matches iTransformer on the 18 good parameters (83.4% vs 87.2% mean skill) and **ties** the best
hybrid result on the hard 6 (+1.5% vs DeepAR-hybrid's +2.2%) — all in ~3 seconds of CPU inference.

This notebook tests whether **light fine-tuning** closes the remaining gap, using Chronos-2's native
`pipeline.fit()` method with **LoRA** (Low-Rank Adaptation — only the attention projections and output
head are updated, not the full 120M-parameter backbone). LoRA is specifically chosen over full
fine-tuning because of this project's repeated finding (v3, v4, v5, the dual-channel ablation): adding
unconstrained capacity on a 28-day dataset tends to overfit and regress results. LoRA constrains the
update to a low-rank subspace, which is a much smaller bet than retraining the entire model.

**Feasibility check done first:** a timing probe (10 LoRA steps, all 27 modeled columns) measured
~7.5s/step on this CPU — full fine-tuning runs (500+ steps) would take over an hour, so this notebook
uses **150 steps (~19 minutes)**, the practical middle ground between "meaningful update" and
"reasonable wait," at `learning_rate=1e-5` (the library's recommended rate for LoRA mode).

Standalone — does not modify any other notebook, dashboard, or CSV in this project, and does not
modify the original pretrained Chronos-2 weights (`fit()` returns a new pipeline).""")

md("## 0. Setup")
code(r"""import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_data_frame

print("Loading Chronos-2 (pretrained, CPU)...")
t0 = time.time()
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
print(f"Loaded in {time.time()-t0:.1f}s")""")

md("## 1. Load data, collapse duplicates, encode circular parameters\n"
   "(identical preprocessing to every prior notebook, for a fair comparison)")
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
code(r"""HORIZON = 288
LOOKBACK = 288

idx = df_num_full.index
df_num_full["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
df_num_full["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
df_num_full["dom_sin"] = np.sin(2 * np.pi * idx.day / 30)
df_num_full["dom_cos"] = np.cos(2 * np.pi * idx.day / 30)
calendar_cols = ["hour_sin", "hour_cos", "dom_sin", "dom_cos"]

train_df = df_num_full.iloc[:-HORIZON].copy()
test_df = df_num_full.iloc[-HORIZON:].copy()
print(f"Train (fine-tuning data): {train_df.shape[0]} rows ({train_df.shape[0]/144:.1f} days)")
print(f"Test  (48h held-out, never seen during fine-tuning): {test_df.shape[0]} rows")

recon_coef = {}
for keep, drop in DUPLICATES:
    x, y = train_df[keep].values, df_num_full[drop].iloc[:-HORIZON].values
    slope, intercept = np.polyfit(x, y, 1)
    pred_train = slope * x + intercept
    r2 = 1 - np.sum((y - pred_train) ** 2) / np.sum((y - y.mean()) ** 2)
    recon_coef[drop] = (keep, float(slope), float(intercept), float(r2))""")

md("## 3. LoRA fine-tune on the 28-day training history\n"
   "150 steps, `learning_rate=1e-5`, batch_size=8 — calibrated from a timing probe (~7.5s/step on "
   "this CPU) to fit a ~19-minute budget. The held-out 48h test window is never shown to `fit()`.")
code(r"""ctx_df = train_df[target_cols + calendar_cols].copy()
ctx_df["item_id"] = "EMS"
ctx_df["timestamp"] = ctx_df.index
ctx_df = ctx_df[["item_id", "timestamp"] + target_cols + calendar_cols]

finetune_inputs = from_data_frame(
    ctx_df, target_columns=target_cols, prediction_length=HORIZON,
    id_column="item_id", timestamp_column="timestamp",
)

print("Starting LoRA fine-tuning (150 steps, ~19 min expected)...")
t0 = time.time()
pipe_finetuned = pipe.fit(
    finetune_inputs, prediction_length=HORIZON,
    finetune_mode="lora", learning_rate=1e-5, num_steps=150, batch_size=8,
    remove_printer_callback=True, disable_data_parallel=True,
)
print(f"LoRA fine-tuning complete in {time.time()-t0:.1f}s ({(time.time()-t0)/150:.2f}s/step).")""")

md("## 4. Forecast the held-out 48h window with the fine-tuned model")
code(r"""future_df = test_df[calendar_cols].copy()
future_df["item_id"] = "EMS"
future_df["timestamp"] = future_df.index
future_df = future_df[["item_id", "timestamp"] + calendar_cols]

t0 = time.time()
chronos_ft_out = pipe_finetuned.predict_df(
    ctx_df, future_df=future_df, target=target_cols,
    id_column="item_id", timestamp_column="timestamp",
    prediction_length=HORIZON, quantile_levels=[0.1, 0.5, 0.9],
    freq="10min",
)
print(f"Fine-tuned forecast complete: {time.time()-t0:.1f}s.")

chronos_ft_pred_df = chronos_ft_out.pivot(index="timestamp", columns="target_name", values="predictions")
chronos_ft_pred_df = chronos_ft_pred_df[target_cols].reindex(test_df.index)
chronos_ft_q10 = chronos_ft_out.pivot(index="timestamp", columns="target_name", values="0.1").reindex(test_df.index)
chronos_ft_q90 = chronos_ft_out.pivot(index="timestamp", columns="target_name", values="0.9").reindex(test_df.index)

def reconstruct(pred_df_in):
    out = pred_df_in.copy()
    for ang in ["windDirection", "currentDirection", "compass"]:
        out[ang] = (np.rad2deg(np.arctan2(out[f"{ang}_sin"], out[f"{ang}_cos"])) % 360)
    return out

chronos_ft_final = reconstruct(chronos_ft_pred_df)
truth = df_num_full.iloc[-HORIZON:].copy()
for ang in ["windDirection", "currentDirection", "compass"]:
    truth[ang] = (np.rad2deg(np.arctan2(truth[f"{ang}_sin"], truth[f"{ang}_cos"])) % 360)

report_params = [c for c in target_cols if not c.endswith(("_sin", "_cos"))] + \
                ["windDirection", "currentDirection", "compass"]
CIRCULAR_PARAMS = {"windDirection", "currentDirection", "compass"}

dup_series = {}
for keep, drop in DUPLICATES:
    _, slope, intercept, _ = recon_coef[drop]
    dup_series[drop] = slope * chronos_ft_final[keep].values + intercept
print("Fine-tuned forecast reconstructed.")""")

md("## 5. The verdict: fine-tuned vs zero-shot Chronos-2 vs iTransformer vs DeepAR-hybrid")
code(r"""zeroshot_metrics = pd.read_csv("metrics_chronos2.csv").set_index("parameter")

ITRANSFORMER_GOOD_SKILL = {
    "airTemperature": 98.0, "airPressure": 97.6, "windDirection": 96.9, "waterTemperature": 96.4,
    "relativeHumidity": 96.3, "currentDirection": 96.0, "tideLevel": 94.0, "dewPointTemperature": 93.6,
    "globalRadiation": 93.0, "significantWaveHeight": 90.7, "currentSpeed": 90.5, "windSpeed": 89.5,
    "salinity": 87.5, "significantWavePeriod": 75.3, "zeroCrossingPeriod": 72.8, "conductivity": 72.8,
    "compass": 70.0, "peakWaveEnergyPeriod": 59.0,
}
DEEPAR_HYBRID_HARD_SKILL = {
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
    yhat = chronos_ft_final[p].values
    if is_circular:
        mae, rmse = circ_mae(yt, yhat), np.nan
    else:
        mae = mean_absolute_error(yt, yhat)
        rmse = np.sqrt(mean_squared_error(yt, yhat))
    skill = (1 - mae / mae_p) * 100 if mae_p > 0 else np.nan
    is_hard = p in HARD_PARAMS
    itransformer_ref = np.nan if is_hard else ITRANSFORMER_GOOD_SKILL.get(p, np.nan)
    deepar_ref = DEEPAR_HYBRID_HARD_SKILL.get(p, np.nan) if is_hard else np.nan
    zeroshot_skill = zeroshot_metrics.loc[p, "chronos2_skill_%"] if p in zeroshot_metrics.index else np.nan
    metrics.append({
        "parameter": p, "is_hard_param": is_hard, "Persistence_MAE": round(mae_p, 4),
        "chronos2_finetuned_MAE": round(mae, 4), "chronos2_finetuned_skill_%": round(skill, 1),
        "chronos2_zeroshot_skill_%": zeroshot_skill,
        "itransformer_skill_%": itransformer_ref, "deepar_hybrid_skill_%": deepar_ref,
        "finetune_improved_vs_zeroshot": bool(skill > zeroshot_skill) if zeroshot_skill == zeroshot_skill else None,
    })

metrics_df = pd.DataFrame(metrics).sort_values("chronos2_finetuned_skill_%", ascending=False).reset_index(drop=True)
metrics_df.insert(0, "rank", metrics_df.index + 1)
metrics_df.to_csv("metrics_chronos2_finetuned.csv", index=False)

n_improved = int(metrics_df["finetune_improved_vs_zeroshot"].sum())
mean_ft_good = metrics_df[~metrics_df["is_hard_param"]]["chronos2_finetuned_skill_%"].mean()
mean_zs_good = metrics_df[~metrics_df["is_hard_param"]]["chronos2_zeroshot_skill_%"].mean()
mean_ft_hard = metrics_df[metrics_df["is_hard_param"]]["chronos2_finetuned_skill_%"].mean()
mean_zs_hard = metrics_df[metrics_df["is_hard_param"]]["chronos2_zeroshot_skill_%"].mean()
mean_itransformer_good = metrics_df[~metrics_df["is_hard_param"]]["itransformer_skill_%"].mean()
mean_deepar_hard = metrics_df[metrics_df["is_hard_param"]]["deepar_hybrid_skill_%"].mean()

print(metrics_df[["rank", "parameter", "is_hard_param", "chronos2_finetuned_skill_%",
                   "chronos2_zeroshot_skill_%", "finetune_improved_vs_zeroshot"]].to_string(index=False))
print(f"\nFine-tuning improved {n_improved}/{len(metrics_df)} parameters vs. zero-shot.")
print(f"Good 18 -- fine-tuned: {mean_ft_good:+.1f}%  zero-shot: {mean_zs_good:+.1f}%  iTransformer: {mean_itransformer_good:+.1f}%")
print(f"Hard 6  -- fine-tuned: {mean_ft_hard:+.1f}%  zero-shot: {mean_zs_hard:+.1f}%  DeepAR-hybrid: {mean_deepar_hard:+.1f}%")
if mean_ft_good > mean_zs_good and mean_ft_hard > mean_zs_hard:
    print("VERDICT: fine-tuning improved both fronts vs zero-shot Chronos-2.")
elif mean_ft_good > mean_zs_good or mean_ft_hard > mean_zs_hard:
    print("VERDICT: fine-tuning improved one front but not the other -- mixed result.")
else:
    print("VERDICT: fine-tuning did not improve on zero-shot Chronos-2 -- the 150-step LoRA update "
          "did not help (or slightly hurt) on this 28-day dataset.")""")

md("## 6. Plot a sample of parameters — fine-tuned vs zero-shot vs actual")
code(r"""zs_fva = pd.read_csv("forecast_vs_actual_chronos2.csv", parse_dates=["timestamp"])
plot_params = (metrics_df[~metrics_df["is_hard_param"]]["parameter"].head(3).tolist() +
               metrics_df[metrics_df["is_hard_param"]]["parameter"].tolist()[:3])
hist_tail = df_10min.iloc[-HORIZON - LOOKBACK:-HORIZON]
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, c in zip(axes.ravel(), plot_params):
    ax.plot(hist_tail.index, hist_tail[c], color="lightgray", lw=1, label="history")
    ax.plot(truth.index, truth[c], color="black", lw=2, label="actual")
    if f"{c}__chronos2" in zs_fva.columns:
        ax.plot(zs_fva["timestamp"], zs_fva[f"{c}__chronos2"], color="#ff7f0e", lw=1.2, ls=":", label="zero-shot")
    ax.plot(truth.index, chronos_ft_final[c], color="#2ca02c", lw=1.5, ls="--", label="LoRA fine-tuned")
    ax.axvline(truth.index[0], color="green", lw=1, alpha=0.5)
    ax.set_title(c, fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("chronos2_finetuned_sample_plot.png", dpi=110)
plt.show()
print("Saved chronos2_finetuned_sample_plot.png")""")

md("## 7. Save outputs for the dashboard")
code(r"""fva = pd.DataFrame({"timestamp": truth.index})
for p in report_params:
    fva[f"{p}__actual"] = truth[p].values
    fva[f"{p}__chronos2_finetuned"] = chronos_ft_final[p].values
    if f"{p}__chronos2" in zs_fva.columns:
        fva[f"{p}__chronos2_zeroshot"] = zs_fva[f"{p}__chronos2"].values
    if f"{p}" in chronos_ft_q10.columns:
        fva[f"{p}__q10"] = chronos_ft_q10[p].values
        fva[f"{p}__q90"] = chronos_ft_q90[p].values
fva.to_csv("forecast_vs_actual_chronos2_finetuned.csv", index=False)

dup_fva = pd.DataFrame({"timestamp": test_df.index})
for keep, drop in DUPLICATES:
    dup_fva[f"{drop}__actual"] = df_10min[drop].iloc[-HORIZON:].values
    dup_fva[f"{drop}__reconstructed"] = dup_series[drop]
dup_fva.to_csv("duplicate_forecast_vs_actual_chronos2_finetuned.csv", index=False)

dup_recon_rows = []
for keep, drop in DUPLICATES:
    _, slope, intercept, r2 = recon_coef[drop]
    mae = mean_absolute_error(df_10min[drop].iloc[-HORIZON:].values, dup_series[drop])
    rmse = np.sqrt(mean_squared_error(df_10min[drop].iloc[-HORIZON:].values, dup_series[drop]))
    dup_recon_rows.append({"duplicate_parameter": drop, "reconstructed_from": keep,
                            "slope": round(slope, 4), "intercept": round(intercept, 4), "train_R2": round(r2, 5),
                            "held_out_MAE": round(mae, 4), "held_out_RMSE": round(rmse, 4)})
pd.DataFrame(dup_recon_rows).to_csv("duplicate_reconstruction_chronos2_finetuned.csv", index=False)

print("Saved: metrics_chronos2_finetuned.csv, forecast_vs_actual_chronos2_finetuned.csv,")
print("       duplicate_reconstruction_chronos2_finetuned.csv, duplicate_forecast_vs_actual_chronos2_finetuned.csv,")
print("       chronos2_finetuned_sample_plot.png.")""")

md(r"""## 8. Conclusion

Section 5 is the actual verdict. A win here would mean light, constrained (LoRA) fine-tuning is worth
the ~19-minute training cost on top of Chronos-2's already-fast zero-shot result. A null or negative
result would be consistent with this project's repeated finding across v3, v4, v5, and the dual-channel
ablation: with only 28 days of training data, additional adaptation capacity doesn't reliably help and
can introduce overfitting risk, even when constrained via LoRA's low-rank update. Either way, reported
honestly against both the zero-shot baseline and the from-scratch references (iTransformer, DeepAR-hybrid).""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_RealEMS_Chronos2_Finetuned.ipynb")
print("Notebook written: Marine_Forecast_RealEMS_Chronos2_Finetuned.ipynb")
