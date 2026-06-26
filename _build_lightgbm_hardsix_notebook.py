import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h Forecast — v11: LightGBM for the Hard 6 (per-parameter vs pooled-group)

An eleventh attempt, testing a user-supplied production strategy document recommending LightGBM/
CatBoost as a standard gradient-boosting baseline for all 6 hard parameters, with a two-stage
occurrence+intensity model for `precipitationIntensity`. Two variants are tested side by side:

1. **Per-parameter (6 independent models)** — the structure this project has used throughout
   (DET v6, TSB v8, ZIDF v9), and what the literature/our own evidence favors for these 6 specifically.
2. **Pooled-by-group (2 shared models)** — one shared LightGBM model trained on **pooled rows across
   all 4 visibility parameters** (same timestamps, but each now contributes 4 training rows instead of
   1, with a `target_id` feature distinguishing which visibility window), and one shared model pooled
   across the 2 precipitation parameters. This is a genuinely different idea from anything tried in
   this project so far — it directly targets the diagnosed core problem (data volume) by increasing
   effective training rows, rather than changing architecture or loss function.

**Why this is a fair test, not a repeat of v2/v3/v4/v5:** those used XGBoost with similar lag/
cross-feature recipes and lost (v3 was the worst result of any approach). LightGBM is a close sibling
algorithm (leaf-wise vs. depth-wise tree growth), so a similar result is the reasonable prior — but
it's being tested directly rather than assumed. The pooled-group variant is the one part of this
notebook that hasn't been tried in any form before.

No retraining of iTransformer — known-future exogenous covariates reuse its already-saved forecasts
(same pattern as v4/v5/v7). Standalone — does not modify any other notebook, dashboard, or CSV.""")

md("## 0. Setup")
code(r"""import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
np.random.seed(SEED)
print("LightGBM version:", lgb.__version__)""")

md("## 1. Load data, build calendar + known-future-exogenous features (reusing iTransformer's saved forecast)")
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
exo_future_real = itransformer_fva[[f"{p}__baseline" for p in EXOGENOUS_FUTURE_PARAMS]].values  # (HORIZON, 4)

ORIGIN_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 144, 288]
print(f"Train: {train_df.shape[0]} rows  |  Test: {test_df.shape[0]} rows")""")

md("## 2. Feature builder — same lag/calendar/exogenous recipe for every variant")
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

md("## 3. Variant A — per-parameter (6 independent LightGBM models)")
code(r"""per_param_pred = {}
for c in VISIBILITY_PARAMS:
    X, Y = make_training_table(train_df, c, ORIGIN_LAGS, calendar_cols, EXOGENOUS_FUTURE_PARAMS, HORIZON)
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, random_state=SEED, verbosity=-1)
    model.fit(X, Y)
    X_fore = make_forecast_table(train_df, c, ORIGIN_LAGS, calendar_cols, exo_future_real,
                                  EXOGENOUS_FUTURE_PARAMS, test_df, HORIZON)[X.columns]
    per_param_pred[c] = model.predict(X_fore)
    print(f"  [per-param] {c:28s} rows={len(Y):,}")

for c in PRECIP_PARAMS:
    X, Y = make_training_table(train_df, c, ORIGIN_LAGS, calendar_cols, EXOGENOUS_FUTURE_PARAMS, HORIZON)
    occurred = (Y > 0).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED, verbosity=-1)
    clf.fit(X, occurred)
    mask = occurred.astype(bool)
    reg = lgb.LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8,
                             colsample_bytree=0.8, random_state=SEED, verbosity=-1)
    reg.fit(X[mask], np.log1p(Y[mask]))

    X_fore = make_forecast_table(train_df, c, ORIGIN_LAGS, calendar_cols, exo_future_real,
                                  EXOGENOUS_FUTURE_PARAMS, test_df, HORIZON)[X.columns]
    p_rain = clf.predict_proba(X_fore)[:, 1]
    amount = np.expm1(reg.predict(X_fore))
    per_param_pred[c] = np.clip(p_rain * amount, 0, None)
    print(f"  [per-param] {c:28s} rows={len(Y):,}  rain_rate={occurred.mean():.1%}")

print("Variant A (per-parameter) complete.")""")

md("## 4. Variant B — pooled-by-group (1 shared visibility model + 1 shared precipitation model)")
code(r"""def make_pooled_table(df, target_cols, lags, calendar_cols, exo_cols, horizon, origin_step=16):
    frames, targets = [], []
    for tid, c in enumerate(target_cols):
        X_c, Y_c = make_training_table(df, c, lags, calendar_cols, exo_cols, horizon, origin_step)
        X_c["target_id"] = tid
        frames.append(X_c); targets.append(Y_c)
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)

# --- visibility pooled (4x the rows of any single per-parameter visibility model) ---
X_vis_pool, Y_vis_pool = make_pooled_table(train_df, VISIBILITY_PARAMS, ORIGIN_LAGS, calendar_cols,
                                            EXOGENOUS_FUTURE_PARAMS, HORIZON)
vis_pooled_model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, random_state=SEED, verbosity=-1)
vis_pooled_model.fit(X_vis_pool, Y_vis_pool)
print(f"Visibility pooled model trained on {len(Y_vis_pool):,} rows (vs ~{len(Y_vis_pool)//4:,} for a single per-parameter model).")

pooled_pred = {}
for tid, c in enumerate(VISIBILITY_PARAMS):
    X_fore = make_forecast_table(train_df, c, ORIGIN_LAGS, calendar_cols, exo_future_real,
                                  EXOGENOUS_FUTURE_PARAMS, test_df, HORIZON)
    X_fore["target_id"] = tid
    X_fore = X_fore[X_vis_pool.columns]
    pooled_pred[c] = vis_pooled_model.predict(X_fore)

# --- precipitation pooled (shared classifier + shared regressor across both precip params) ---
X_precip_pool, Y_precip_pool = make_pooled_table(train_df, PRECIP_PARAMS, ORIGIN_LAGS, calendar_cols,
                                                  EXOGENOUS_FUTURE_PARAMS, HORIZON)
occurred_pool = (Y_precip_pool > 0).astype(int)
clf_pool = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8,
                               colsample_bytree=0.8, random_state=SEED, verbosity=-1)
clf_pool.fit(X_precip_pool, occurred_pool)
mask_pool = occurred_pool.astype(bool)
reg_pool = lgb.LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED, verbosity=-1)
reg_pool.fit(X_precip_pool[mask_pool], np.log1p(Y_precip_pool[mask_pool]))
print(f"Precipitation pooled model trained on {len(Y_precip_pool):,} rows (vs ~{len(Y_precip_pool)//2:,} for a single per-parameter model).")

for tid, c in enumerate(PRECIP_PARAMS):
    X_fore = make_forecast_table(train_df, c, ORIGIN_LAGS, calendar_cols, exo_future_real,
                                  EXOGENOUS_FUTURE_PARAMS, test_df, HORIZON)
    X_fore["target_id"] = tid
    X_fore = X_fore[X_precip_pool.columns]
    p_rain = clf_pool.predict_proba(X_fore)[:, 1]
    amount = np.expm1(reg_pool.predict(X_fore))
    pooled_pred[c] = np.clip(p_rain * amount, 0, None)

print("Variant B (pooled-by-group) complete.")""")

md("## 5. Score both variants against persistence and all ten prior hard-6 attempts")
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

truth = df_10min.iloc[-HORIZON:]
last_obs = df_10min.iloc[-HORIZON - 1]

metrics = []
for c in HARD_PARAMS:
    yt = truth[c].values
    yp_persist = np.repeat(last_obs[c], HORIZON)
    mae_p = mean_absolute_error(yt, yp_persist)

    row = {"parameter": c, "Persistence_MAE": round(mae_p, 4)}
    for label, preds in [("perparam", per_param_pred), ("pooled", pooled_pred)]:
        mae = mean_absolute_error(yt, preds[c])
        skill = (1 - mae / mae_p) * 100 if mae_p > 0 else np.nan
        row[f"lgbm_{label}_skill_%"] = round(skill, 1)
    row["pure_iTransformer_skill_%"] = PURE_ITRANSFORMER_SKILL[c]
    row["deepar_hybrid_skill_%"] = DEEPAR_HYBRID_SKILL[c]
    row["det_v6_skill_%"] = DET_V6_SKILL[c]
    row["tsb_v8_skill_%"] = TSB_V8_SKILL.get(c, np.nan)
    metrics.append(row)

metrics_df = pd.DataFrame(metrics).sort_values("lgbm_perparam_skill_%", ascending=False).reset_index(drop=True)
metrics_df.to_csv("metrics_lightgbm_v11.csv", index=False)
print(metrics_df.to_string(index=False))

mean_perparam = metrics_df["lgbm_perparam_skill_%"].mean()
mean_pooled = metrics_df["lgbm_pooled_skill_%"].mean()
mean_deepar = metrics_df["deepar_hybrid_skill_%"].mean()
print(f"\nMean skill -- LightGBM per-param: {mean_perparam:+.1f}%  |  LightGBM pooled: {mean_pooled:+.1f}%  |  DeepAR-hybrid (best prior): {mean_deepar:+.1f}%")
best_lgbm = max(mean_perparam, mean_pooled)
if best_lgbm > mean_deepar:
    winner = "per-parameter" if mean_perparam > mean_pooled else "pooled-by-group"
    print(f"VERDICT: LightGBM ({winner}) is the new best result for the hard 6.")
else:
    print(f"VERDICT: neither LightGBM variant beats DeepAR-hybrid -- it remains the best result found.")
if mean_pooled > mean_perparam:
    print("Pooling rows across the group HELPED relative to per-parameter LightGBM.")
else:
    print("Pooling rows across the group did NOT help relative to per-parameter LightGBM.")""")

md("## 6. Save outputs for the dashboard")
code(r"""fva = pd.DataFrame({"timestamp": truth.index})
for c in HARD_PARAMS:
    fva[f"{c}__actual"] = truth[c].values
    fva[f"{c}__lgbm_perparam"] = per_param_pred[c]
    fva[f"{c}__lgbm_pooled"] = pooled_pred[c]
fva.to_csv("forecast_vs_actual_lightgbm_v11.csv", index=False)
print("Saved: metrics_lightgbm_v11.csv, forecast_vs_actual_lightgbm_v11.csv")""")

md(r"""## 7. Conclusion

Section 5 is the actual verdict, including the specific per-param-vs-pooled comparison. A win for the
pooled variant would be a genuinely new finding for this project — confirmation that the binding
constraint really is row count, and that pooling related targets is a viable, simple way to get more
of it without needing more calendar days of history. A loss for both, consistent with v2/v3/v4/v5's
XGBoost results, would close the gradient-boosting line of inquiry for the hard 6 with reasonable
confidence rather than lingering uncertainty about whether a different GBM library would have worked.""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_RealEMS_LightGBM_HardSix.ipynb")
print("Notebook written: Marine_Forecast_RealEMS_LightGBM_HardSix.ipynb")
