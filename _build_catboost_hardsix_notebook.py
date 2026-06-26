import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h Forecast — v12: CatBoost for the Hard 6 (per-parameter)

A twelfth attempt, testing CatBoost with the **exact same feature recipe** as v11's LightGBM
(same lags, same calendar features, same known-future-exogenous covariates reused from iTransformer's
already-saved forecast) — isolating the comparison to "does the choice of gradient-boosting library
matter" rather than confounding it with a different feature set.

**Why per-parameter only, no pooled variant this time:** v11 already tested pooling rows across the
visibility group and across the precipitation group with LightGBM, and it made every single parameter
worse, not better. Repeating that test with a different GBM library is unlikely to change that
conclusion, so this notebook focuses on the per-parameter comparison only.

**What v11 found, that this notebook checks for the second time:** LightGBM scored +65.7% on
`precipitationDifference` — a genuine new best, verified against the actual data, not a leak. The
question here is whether that's a real, recoverable pattern in this specific 6-row lag/exogenous
feature set (in which case CatBoost should find something similar), or an XGBoost/LightGBM-family-
specific result that doesn't generalize to a third, differently-built gradient-boosting library
(CatBoost's ordered boosting and symmetric trees are a genuinely different algorithm, not just a
re-skin of the same leaf-wise/depth-wise tree-growing idea).

No retraining of iTransformer. Standalone — does not modify any other notebook, dashboard, or CSV.""")

md("## 0. Setup")
code(r"""import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
np.random.seed(SEED)
print("CatBoost ready.")""")

md("## 1. Load data, calendar + known-future-exogenous features (same recipe as v11)")
code(r"""df_10min = pd.read_csv("ems_10min_resampled.csv", index_col=0, parse_dates=True)
PRECIP_PARAMS = ["precipitationIntensity", "precipitationDifference"]
VISIBILITY_PARAMS = ["twentyFourHourAvgVisibility", "tenMinuteAvgVisibility",
                      "oneMinuteAvgVisibility", "oneHourAvgVisibility"]
HARD_PARAMS = PRECIP_PARAMS + VISIBILITY_PARAMS
EXOGENOUS_FUTURE_PARAMS = ["relativeHumidity", "dewPointTemperature", "airPressure", "windSpeed"]

HORIZON = 288
idx = df_10min.index
df_10min["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
df_10min["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
df_10min["dom_sin"] = np.sin(2 * np.pi * idx.day / 30)
df_10min["dom_cos"] = np.cos(2 * np.pi * idx.day / 30)
calendar_cols = ["hour_sin", "hour_cos", "dom_sin", "dom_cos"]

train_df = df_10min.iloc[:-HORIZON].copy()
test_df = df_10min.iloc[-HORIZON:].copy()

itransformer_fva = pd.read_csv("forecast_vs_actual_dualchannel.csv", parse_dates=["timestamp"])
exo_future_real = itransformer_fva[[f"{p}__baseline" for p in EXOGENOUS_FUTURE_PARAMS]].values

ORIGIN_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 144, 288]
print(f"Train: {train_df.shape[0]} rows  |  Test: {test_df.shape[0]} rows")""")

md("## 2. Feature builder (identical to v11)")
code(r"""def make_training_table(df, target_col, lags, calendar_cols, exo_cols, horizon, origin_step=16):
    n, max_lag = len(df), max(lags)
    feats, targets = [], []
    for origin in range(max_lag, n - horizon, origin_step):
        base = {f"lag{L}": df[target_col].iloc[origin - L] for L in lags}
        for h in range(1, horizon + 1, 2):
            row = dict(base); row["lead_h"] = h
            for cc in calendar_cols:
                row[cc] = df[cc].iloc[origin + h]
            for ec in exo_cols:
                row[f"{ec}_future"] = df[ec].iloc[origin + h]
            feats.append(row)
            targets.append(df[target_col].iloc[origin + h])
    return pd.DataFrame(feats), np.array(targets)


def make_forecast_table(train_df, target_col, lags, calendar_cols, exo_future_real, exo_cols,
                         calendar_test_df, horizon):
    origin_idx = len(train_df) - 1
    base_row = {f"lag{L}": train_df[target_col].iloc[origin_idx - (L - 1)] for L in lags}
    rows = []
    for h in range(1, horizon + 1):
        ts = calendar_test_df.index[h - 1]
        row = dict(base_row); row["lead_h"] = h
        for cc in calendar_cols:
            row[cc] = calendar_test_df.loc[ts, cc]
        for k, ec in enumerate(exo_cols):
            row[f"{ec}_future"] = exo_future_real[h - 1, k]
        rows.append(row)
    return pd.DataFrame(rows)

print("Feature builders ready.")""")

md("## 3. Train CatBoost per parameter (regression for visibility, two-stage hurdle for precipitation)")
code(r"""per_param_pred = {}
for c in VISIBILITY_PARAMS:
    X, Y = make_training_table(train_df, c, ORIGIN_LAGS, calendar_cols, EXOGENOUS_FUTURE_PARAMS, HORIZON)
    model = CatBoostRegressor(iterations=200, depth=5, learning_rate=0.05, subsample=0.8,
                               random_state=SEED, verbose=False)
    model.fit(X, Y)
    X_fore = make_forecast_table(train_df, c, ORIGIN_LAGS, calendar_cols, exo_future_real,
                                  EXOGENOUS_FUTURE_PARAMS, test_df, HORIZON)[X.columns]
    per_param_pred[c] = model.predict(X_fore)
    print(f"  {c:28s} rows={len(Y):,}")

for c in PRECIP_PARAMS:
    X, Y = make_training_table(train_df, c, ORIGIN_LAGS, calendar_cols, EXOGENOUS_FUTURE_PARAMS, HORIZON)
    occurred = (Y > 0).astype(int)
    clf = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.08, subsample=0.8,
                              random_state=SEED, verbose=False)
    clf.fit(X, occurred)
    mask = occurred.astype(bool)
    reg = CatBoostRegressor(iterations=150, depth=4, learning_rate=0.08, subsample=0.8,
                             random_state=SEED, verbose=False)
    reg.fit(X[mask], np.log1p(Y[mask]))

    X_fore = make_forecast_table(train_df, c, ORIGIN_LAGS, calendar_cols, exo_future_real,
                                  EXOGENOUS_FUTURE_PARAMS, test_df, HORIZON)[X.columns]
    p_rain = clf.predict_proba(X_fore)[:, 1]
    amount = np.expm1(reg.predict(X_fore))
    per_param_pred[c] = np.clip(p_rain * amount, 0, None)
    print(f"  {c:28s} rows={len(Y):,}  rain_rate={occurred.mean():.1%}")

print("CatBoost per-parameter training complete.")""")

md("## 4. Score against persistence and all eleven prior hard-6 attempts")
code(r"""PURE_ITRANSFORMER_SKILL = {
    "twentyFourHourAvgVisibility": -100.0, "precipitationDifference": -101.9,
    "tenMinuteAvgVisibility": -154.9, "oneMinuteAvgVisibility": -190.5,
    "oneHourAvgVisibility": -291.6, "precipitationIntensity": -409.9,
}
DEEPAR_HYBRID_SKILL = {
    "tenMinuteAvgVisibility": 14.0, "twentyFourHourAvgVisibility": 3.5,
    "precipitationDifference": -0.1, "precipitationIntensity": -0.2,
    "oneHourAvgVisibility": -1.4, "oneMinuteAvgVisibility": -2.5,
}
DET_V6_SKILL = {
    "tenMinuteAvgVisibility": 16.5, "oneMinuteAvgVisibility": 0.8, "oneHourAvgVisibility": -2.7,
    "precipitationDifference": -3.1, "twentyFourHourAvgVisibility": -12.3, "precipitationIntensity": -335.6,
}
TSB_V8_SKILL = {"precipitationIntensity": 0.0, "precipitationDifference": 0.0}

lgbm_metrics = pd.read_csv("metrics_lightgbm_v11.csv").set_index("parameter")["lgbm_perparam_skill_%"]

truth = df_10min.iloc[-HORIZON:]
last_obs = df_10min.iloc[-HORIZON - 1]

metrics = []
for c in HARD_PARAMS:
    yt = truth[c].values
    yp_persist = np.repeat(last_obs[c], HORIZON)
    mae_p = mean_absolute_error(yt, yp_persist)
    mae = mean_absolute_error(yt, per_param_pred[c])
    skill = (1 - mae / mae_p) * 100 if mae_p > 0 else np.nan
    metrics.append({
        "parameter": c, "Persistence_MAE": round(mae_p, 4),
        "catboost_v12_skill_%": round(skill, 1),
        "lightgbm_v11_skill_%": lgbm_metrics[c],
        "pure_iTransformer_skill_%": PURE_ITRANSFORMER_SKILL[c],
        "deepar_hybrid_skill_%": DEEPAR_HYBRID_SKILL[c], "det_v6_skill_%": DET_V6_SKILL[c],
        "tsb_v8_skill_%": TSB_V8_SKILL.get(c, np.nan),
    })

metrics_df = pd.DataFrame(metrics).sort_values("catboost_v12_skill_%", ascending=False).reset_index(drop=True)
metrics_df.to_csv("metrics_catboost_v12.csv", index=False)
print(metrics_df.to_string(index=False))

mean_v12 = metrics_df["catboost_v12_skill_%"].mean()
mean_v11 = metrics_df["lightgbm_v11_skill_%"].mean()
mean_deepar = metrics_df["deepar_hybrid_skill_%"].mean()
print(f"\nMean skill -- CatBoost: {mean_v12:+.1f}%  |  LightGBM v11: {mean_v11:+.1f}%  |  DeepAR-hybrid (best prior): {mean_deepar:+.1f}%")

precipdiff_cb = metrics_df.set_index("parameter").loc["precipitationDifference", "catboost_v12_skill_%"]
print(f"\nprecipitationDifference -- CatBoost: {precipdiff_cb:+.1f}%  vs  LightGBM's +65.7% (does the pattern generalize across GBM libraries?)")""")

md("## 5. Save outputs for the dashboard")
code(r"""fva = pd.DataFrame({"timestamp": truth.index})
for c in HARD_PARAMS:
    fva[f"{c}__actual"] = truth[c].values
    fva[f"{c}__catboost_v12"] = per_param_pred[c]
fva.to_csv("forecast_vs_actual_catboost_v12.csv", index=False)
print("Saved: metrics_catboost_v12.csv, forecast_vs_actual_catboost_v12.csv")""")

md(r"""## 6. Conclusion

Section 4's `precipitationDifference` comparison is the key check. If CatBoost also finds strong skill
there, that confirms the lag/exogenous feature set genuinely contains a recoverable signal for this
parameter, and the choice of which gradient-boosting library exploits it is secondary. If CatBoost
doesn't, that would suggest LightGBM's specific leaf-wise tree growth found something algorithm-
specific rather than a robust, library-independent pattern — worth knowing either way before relying
on the v11 result operationally.""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_RealEMS_CatBoost_HardSix.ipynb")
print("Notebook written: Marine_Forecast_RealEMS_CatBoost_HardSix.ipynb")
