import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h Forecast — v7: Physics-Informed Pressure-Tendency Model for Precipitation

A seventh attempt, precipitation only (the 2 visibility-style physics model was checked first and
**dropped** — see note below). No neural network, no gradient boosting: a classical, fully
interpretable model built from a real, verified physical relationship in this data.

**Why visibility's physics model (Koschmieder/VIS-RH) isn't included here:** before building anything,
the actual relationship was checked directly in this dataset. Visibility shows ~0 correlation with
humidity/dew-point depression (-0.005 to -0.04), and the test window's fog dip is a perfectly smooth,
symmetric ~4-hour ramp down and back up that is uncorrelated with every other measured variable
(strongest correlation found: windSpeed at -0.09). That's consistent with an independently-injected
synthetic event in the EMS simulator, not real fog physics tied to humidity — building a humidity-based
visibility model here would rely on a relationship the data itself doesn't have.

**Precipitation is different — the physical relationship actually checks out.** 3-hour pressure
tendency (`airPressure.diff(18)`) correlates with rain occurrence at -0.208 for
`precipitationDifference` and -0.07 for `precipitationIntensity` — falling pressure really is
associated with more rain in this data, consistent with real meteorology (and literally why ships have
carried barometers since the 1800s).

**The model:** a hurdle decomposition with each half handled by an interpretable, classical method:
1. **Occurrence**: logistic regression, `P(rain) = sigmoid(a + b * pressure_tendency)` — one
   coefficient, fully inspectable.
2. **Magnitude given occurrence**: Croston's method (1972) — simple exponential smoothing applied
   separately to the historical non-zero magnitudes and to the gaps between rain events, the classical
   approach for intermittent/zero-inflated series (originally developed for spare-parts demand
   forecasting, structurally identical to rainfall: mostly zero, occasional bursts).

Final forecast = `P(rain | tendency) × E[magnitude | rain]` (Croston-smoothed).

Reuses the already-saved iTransformer forecast of `airPressure` for the pressure trajectory needed at
each future step (no retraining — same known-future-exogenous pattern used in v4/v5).
Standalone — does not modify any other notebook, dashboard, or CSV in this project.""")

md("## 0. Setup")
code(r"""import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

PRECIP_PARAMS = ["precipitationIntensity", "precipitationDifference"]
HORIZON = 288
TENDENCY_WINDOW = 18   # 3 hours at 10-min resolution

df_10min = pd.read_csv("ems_10min_resampled.csv", index_col=0, parse_dates=True)
train_df = df_10min.iloc[:-HORIZON]
test_df = df_10min.iloc[-HORIZON:]
print(f"Train: {train_df.shape[0]} rows ({train_df.shape[0]/144:.1f} days)  |  Test: {test_df.shape[0]} rows")""")

md("## 1. Verify the physical relationship before fitting anything")
code(r"""tendency_train = train_df["airPressure"].diff(TENDENCY_WINDOW)
valid = tendency_train.notna()
for p in PRECIP_PARAMS:
    occurred = (train_df[p] > 0).astype(int)
    corr = np.corrcoef(tendency_train[valid], occurred[valid])[0, 1]
    print(f"{p:25s} rain rate={occurred.mean():.1%}  corr(pressure_tendency, occurrence)={corr:+.3f}")""")

md("## 2. Fit the occurrence model (logistic regression on pressure tendency) and the Croston magnitude model")
code(r"""def fit_croston(values, alpha=0.3):
    # values: full historical series (zeros included). Returns smoothed (magnitude, interval) estimates.
    nonzero_idx = np.where(values > 0)[0]
    if len(nonzero_idx) < 2:
        return values[values > 0].mean() if (values > 0).any() else 0.0, len(values)
    magnitudes = values[nonzero_idx]
    intervals = np.diff(nonzero_idx, prepend=nonzero_idx[0])
    intervals[0] = nonzero_idx[0] + 1

    z_hat = magnitudes[0]
    for z in magnitudes[1:]:
        z_hat = alpha * z + (1 - alpha) * z_hat
    q_hat = intervals[0]
    for q in intervals[1:]:
        q_hat = alpha * q + (1 - alpha) * q_hat
    return float(z_hat), float(q_hat)


models = {}
for p in PRECIP_PARAMS:
    occurred = (train_df[p] > 0).astype(int).values
    X = tendency_train[valid].values.reshape(-1, 1)
    y = occurred[valid.values]
    clf = LogisticRegression()
    clf.fit(X, y)

    z_hat, q_hat = fit_croston(train_df[p].values)
    base_rate = 1.0 / q_hat if q_hat > 0 else occurred.mean()

    models[p] = {"clf": clf, "z_hat": z_hat, "q_hat": q_hat, "base_rate": base_rate}
    print(f"{p:25s} logistic coef={clf.coef_[0][0]:+.4f}  intercept={clf.intercept_[0]:+.4f}  "
          f"Croston magnitude={z_hat:.4f}  Croston interval={q_hat:.1f} steps  "
          f"(unconditional base rate={base_rate:.1%})")""")

md("## 3. Build the future pressure-tendency trajectory from the already-saved iTransformer forecast")
code(r"""itransformer_fva = pd.read_csv("forecast_vs_actual_dualchannel.csv", parse_dates=["timestamp"])
pressure_forecast = itransformer_fva["airPressure__baseline"].values   # 288 future values, no retraining

pressure_combined = np.concatenate([train_df["airPressure"].values[-TENDENCY_WINDOW:], pressure_forecast])
tendency_future = pressure_combined[TENDENCY_WINDOW:] - pressure_combined[:-TENDENCY_WINDOW]
print(f"Future pressure tendency trajectory built: {len(tendency_future)} steps "
      f"(mean={tendency_future.mean():+.3f}, std={tendency_future.std():.3f} hPa/3h)")""")

md("## 4. Forecast: P(rain | tendency) x Croston-smoothed magnitude")
code(r"""precip_pred = {}
for p in PRECIP_PARAMS:
    m = models[p]
    p_rain = m["clf"].predict_proba(tendency_future.reshape(-1, 1))[:, 1]
    precip_pred[p] = p_rain * m["z_hat"]

precip_pred_df = pd.DataFrame(precip_pred, index=test_df.index)
print(precip_pred_df.describe().T[["mean", "min", "max"]])""")

md("## 5. Score against persistence and all six prior precipitation attempts")
code(r"""PURE_ITRANSFORMER_SKILL = {"precipitationDifference": -101.9, "precipitationIntensity": -409.9}
DEEPAR_HYBRID_SKILL = {"precipitationDifference": -0.1, "precipitationIntensity": -0.2}
XGB_V2_SKILL = {"precipitationIntensity": -0.1, "precipitationDifference": -0.6}
XGB_V3_SKILL = {"precipitationDifference": -34.2, "precipitationIntensity": -74.9}
TIMEXER_V4_SKILL = {"precipitationIntensity": -25.0, "precipitationDifference": -29.6}
RESIDUAL_V5_SKILL = {"precipitationIntensity": -88.0, "precipitationDifference": -103.0}
DET_V6_SKILL = {"precipitationDifference": -3.1, "precipitationIntensity": -335.6}

truth = df_10min.iloc[-HORIZON:]
last_obs = df_10min.iloc[-HORIZON - 1]

metrics = []
for p in PRECIP_PARAMS:
    yt = truth[p].values
    yp_persist = np.repeat(last_obs[p], HORIZON)
    mae_p = mean_absolute_error(yt, yp_persist)
    mae = mean_absolute_error(yt, precip_pred_df[p].values)
    skill = (1 - mae / mae_p) * 100 if mae_p > 0 else np.nan
    metrics.append({
        "parameter": p, "Persistence_MAE": round(mae_p, 4),
        "physics_v7_MAE": round(mae, 4), "physics_v7_skill_%": round(skill, 1),
        "pure_iTransformer_skill_%": PURE_ITRANSFORMER_SKILL[p],
        "deepar_hybrid_skill_%": DEEPAR_HYBRID_SKILL[p], "xgb_v2_skill_%": XGB_V2_SKILL[p],
        "xgb_v3_skill_%": XGB_V3_SKILL[p], "timexer_v4_skill_%": TIMEXER_V4_SKILL[p],
        "residual_v5_skill_%": RESIDUAL_V5_SKILL[p], "det_v6_skill_%": DET_V6_SKILL[p],
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("metrics_physics_v7.csv", index=False)
print(metrics_df.to_string(index=False))

best_prior = max(metrics_df["deepar_hybrid_skill_%"].mean(), metrics_df["xgb_v2_skill_%"].mean())
mean_v7 = metrics_df["physics_v7_skill_%"].mean()
print(f"\nMean skill -- v7 (physics-informed): {mean_v7:+.1f}%  |  best prior (DeepAR-hybrid/XGBoost v2): {best_prior:+.1f}%")
if mean_v7 > best_prior:
    print("VERDICT: v7 (physics-informed) is the new best result for precipitation.")
else:
    print("VERDICT: v7 does not beat the prior best -- DeepAR-hybrid/XGBoost v2 remain the best results found.")""")

md("## 6. Plot")
code(r"""hist_tail = df_10min.iloc[-HORIZON - 288:-HORIZON]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, p in zip(axes, PRECIP_PARAMS):
    ax.plot(hist_tail.index, hist_tail[p], color="lightgray", lw=1, label="history")
    ax.plot(truth.index, truth[p], color="black", lw=2, label="actual")
    ax.plot(truth.index, precip_pred_df[p], color="#2ca02c", lw=1.5, ls="--", label="physics-informed v7")
    ax.axvline(truth.index[0], color="green", lw=1, alpha=0.5)
    ax.set_title(p, fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("physics_v7_precip_plot.png", dpi=110)
plt.show()
print("Saved physics_v7_precip_plot.png")""")

md("## 7. Save outputs for the dashboard")
code(r"""fva = pd.DataFrame({"timestamp": truth.index})
for p in PRECIP_PARAMS:
    fva[f"{p}__actual"] = truth[p].values
    fva[f"{p}__physics_v7"] = precip_pred_df[p].values
fva.to_csv("forecast_vs_actual_physics_v7.csv", index=False)
print("Saved: metrics_physics_v7.csv, forecast_vs_actual_physics_v7.csv, physics_v7_precip_plot.png")""")

md(r"""## 8. Conclusion

Section 5 is the actual verdict. Whether or not this beats the prior best, it's a genuinely different
kind of model than everything else tried — fully interpretable (one logistic coefficient, two Croston
smoothing values), near-zero training cost, and built from a relationship that was verified in the
data first rather than assumed. The visibility half of this idea was correctly abandoned before any
code was written, once the data showed the fog dip isn't physically coupled to humidity in this
dataset — a useful negative finding in its own right, not just a skipped step.""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_RealEMS_Physics_PressureTendency.ipynb")
print("Notebook written: Marine_Forecast_RealEMS_Physics_PressureTendency.ipynb")
