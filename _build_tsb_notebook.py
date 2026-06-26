import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h Forecast — v8: TSB (Teunter-Syntetos-Babai) for Precipitation

An eighth attempt at precipitation, using [Nixtla's `statsforecast`](https://github.com/Nixtla/statsforecast)
(mature, production-grade, pip-installable — not a reimplementation from an abstract). **TSB** is the
established refinement of Croston's method (used in v7's hand-rolled version): instead of only
updating its probability-of-rain estimate when rain actually occurs, TSB updates it **every period**,
including the long stretches of exact zeros. That's the specific mechanism v7 lacked — v7's logistic
regression and Croston combination produced a nonzero forecast at 100% of test steps; TSB's per-period
probability decay is built to avoid exactly that.

Univariate, no exogenous features at all (TSB doesn't use covariates) — the simplest possible model in
this entire project, by design. Standalone — does not modify any other notebook, dashboard, or CSV.""")

md("## 0. Setup")
code(r"""import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from statsforecast import StatsForecast
from statsforecast.models import TSB
from sklearn.metrics import mean_absolute_error

PRECIP_PARAMS = ["precipitationIntensity", "precipitationDifference"]
HORIZON = 288

df_10min = pd.read_csv("ems_10min_resampled.csv", index_col=0, parse_dates=True)
train_df = df_10min.iloc[:-HORIZON]
test_df = df_10min.iloc[-HORIZON:]
print(f"Train: {train_df.shape[0]} rows  |  Test: {test_df.shape[0]} rows")""")

md("## 1. Fit TSB per parameter and forecast")
code(r"""precip_pred = {}
for p in PRECIP_PARAMS:
    sf_df = pd.DataFrame({
        "unique_id": "EMS", "ds": train_df.index, "y": train_df[p].values,
    })
    model = StatsForecast(models=[TSB(alpha_d=0.2, alpha_p=0.2)], freq="10min", n_jobs=1)
    model.fit(sf_df)
    fcst = model.predict(h=HORIZON)
    precip_pred[p] = fcst["TSB"].values
    print(f"{p:25s} forecast mean={fcst['TSB'].mean():.4f}  min={fcst['TSB'].min():.4f}  max={fcst['TSB'].max():.4f}")

precip_pred_df = pd.DataFrame(precip_pred, index=test_df.index)""")

md("## 2. Score against persistence and all seven prior precipitation attempts")
code(r"""PURE_ITRANSFORMER_SKILL = {"precipitationDifference": -101.9, "precipitationIntensity": -409.9}
DEEPAR_HYBRID_SKILL = {"precipitationDifference": -0.1, "precipitationIntensity": -0.2}
XGB_V2_SKILL = {"precipitationIntensity": -0.1, "precipitationDifference": -0.6}
XGB_V3_SKILL = {"precipitationDifference": -34.2, "precipitationIntensity": -74.9}
TIMEXER_V4_SKILL = {"precipitationIntensity": -25.0, "precipitationDifference": -29.6}
RESIDUAL_V5_SKILL = {"precipitationIntensity": -88.0, "precipitationDifference": -103.0}
DET_V6_SKILL = {"precipitationDifference": -3.1, "precipitationIntensity": -335.6}
PHYSICS_V7_SKILL = {"precipitationIntensity": -35.0, "precipitationDifference": -81.7}

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
        "tsb_v8_MAE": round(mae, 4), "tsb_v8_skill_%": round(skill, 1),
        "pure_iTransformer_skill_%": PURE_ITRANSFORMER_SKILL[p],
        "deepar_hybrid_skill_%": DEEPAR_HYBRID_SKILL[p], "xgb_v2_skill_%": XGB_V2_SKILL[p],
        "xgb_v3_skill_%": XGB_V3_SKILL[p], "timexer_v4_skill_%": TIMEXER_V4_SKILL[p],
        "residual_v5_skill_%": RESIDUAL_V5_SKILL[p], "det_v6_skill_%": DET_V6_SKILL[p],
        "physics_v7_skill_%": PHYSICS_V7_SKILL[p],
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("metrics_tsb_v8.csv", index=False)
print(metrics_df.to_string(index=False))

mean_v8 = metrics_df["tsb_v8_skill_%"].mean()
best_prior = max(metrics_df["deepar_hybrid_skill_%"].mean(), metrics_df["xgb_v2_skill_%"].mean())
print(f"\nMean skill -- v8 (TSB): {mean_v8:+.1f}%  |  best prior: {best_prior:+.1f}%")
if mean_v8 > best_prior:
    print("VERDICT: v8 (TSB) is the new best result for precipitation.")
else:
    print("VERDICT: v8 does not beat the prior best -- DeepAR-hybrid/XGBoost v2 remain the best results found.")

n_nonzero_pred = {p: (precip_pred_df[p] > 0.01).mean() for p in PRECIP_PARAMS}
n_nonzero_actual = {p: (truth[p] > 0).mean() for p in PRECIP_PARAMS}
print(f"\nFraction of forecast steps predicting >0.01: {n_nonzero_pred}")
print(f"Fraction of actual steps with real rain:        {n_nonzero_actual}")""")

md("## 3. Save outputs")
code(r"""fva = pd.DataFrame({"timestamp": truth.index})
for p in PRECIP_PARAMS:
    fva[f"{p}__actual"] = truth[p].values
    fva[f"{p}__tsb_v8"] = precip_pred_df[p].values
fva.to_csv("forecast_vs_actual_tsb_v8.csv", index=False)
print("Saved: metrics_tsb_v8.csv, forecast_vs_actual_tsb_v8.csv")""")

md(r"""## 4. Conclusion

Section 2 is the actual verdict, including the nonzero-fraction diagnostic — the direct test of
whether TSB's per-period probability update actually solves the problem v7 had (predicting nonzero
100% of the time). If TSB's nonzero fraction is much closer to the actual rain rate, the core
hypothesis behind trying it is confirmed even if the skill score itself doesn't beat the prior best.""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_RealEMS_TSB_Precip.ipynb")
print("Notebook written: Marine_Forecast_RealEMS_TSB_Precip.ipynb")
