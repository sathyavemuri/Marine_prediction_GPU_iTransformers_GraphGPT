import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h Forecast — iTransformer Seed-Robustness Check (18 "good" parameters)

Tests a methodological critique raised by Miño Calero, Rasheed & Lekkas
(["Unveiling the limitations of transformer models in time series forecasting"](https://doi.org/10.1007/s13748-026-00450-y),
*Progress in Artificial Intelligence*, 2026): Transformers show *"significantly higher variability of
results, hence lacking robustness"* across different initializations and data splits than simpler
models — and point-metric comparisons (a single MSE/MAE run) aren't sufficient evidence of which model
actually wins.

**Why this matters here specifically:** every comparison in this project so far — iTransformer vs.
DeepAR, vs. XGBoost, vs. Dual-Channel, vs. SOFTS, vs. RevIN, vs. Chronos-2 — used a **single fixed seed
(42)**, one run per model. iTransformer has now survived three separate architecture/normalization
challenges (Dual-Channel, SOFTS, RevIN), all on the strength of one run each. This notebook checks
whether that result is robust: train the *exact same* baseline iTransformer architecture **5 times**,
with 5 different random seeds, same data/split/training procedure, and measure how much the resulting
skill scores actually move around.

**What this notebook is not:** it isn't testing a new architecture or challenger. It's a robustness
audit of the baseline itself, to put a confidence interval around the headline ~87% mean-skill figure
that's been used as the reference point for every ablation so far.

Standalone — does not modify any other notebook, dashboard, or CSV in this project.""")

md("## 0. Setup")
code(r"""import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device("cpu")
torch.set_num_threads(8)

SEEDS = [42, 7, 123, 2024, 31416]
print("PyTorch:", torch.__version__, "| torch threads:", torch.get_num_threads())
print("Seeds to test:", SEEDS)""")

md("## 1. Load data, collapse duplicates, encode circular parameters")
code(r"""df_10min = pd.read_csv("ems_10min_resampled.csv", index_col=0, parse_dates=True)
DUPLICATES = [
    ("airTemperature", "windChillTemperature"),
    ("tideLevel", "tidePressure"),
    ("tideLevel", "waterPressure"),
    ("tideLevel", "waterLevel"),
    ("waterTemperature", "waterTemperature_WQ"),
    ("significantWaveHeight", "maxWaveHeight"),
]
df_cat = df_10min[["precipitationType"]].copy()
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
print(f"This robustness check covers the {len(GOOD_PARAMS)} 'good' parameters only.")""")

md("## 2. Train/test split, scaling (identical to every prior good-18 ablation)")
code(r"""LOOKBACK, HORIZON = 288, 288

idx = df_num_full.index
df_num_full["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
df_num_full["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
df_num_full["dom_sin"] = np.sin(2 * np.pi * idx.day / 30)
df_num_full["dom_cos"] = np.cos(2 * np.pi * idx.day / 30)
calendar_cols = ["hour_sin", "hour_cos", "dom_sin", "dom_cos"]

feature_cols = target_cols + calendar_cols
model_data = df_num_full[feature_cols].copy()
n_features = len(feature_cols)
good_idx = [feature_cols.index(c) for c in GOOD_PARAMS]

train_df = model_data.iloc[:-HORIZON].copy()
test_df = model_data.iloc[-HORIZON:].copy()
mean, std = train_df.mean(), train_df.std().replace(0, 1)
train_scaled = (train_df - mean) / std

print(f"Train: {train_df.shape[0]} rows ({train_df.shape[0]/144:.1f} days)")
print(f"Test : {test_df.shape[0]} rows ({test_df.shape[0]/144:.1f} days)")

truth = df_num_full.iloc[-HORIZON:].copy()
for ang in ["windDirection", "currentDirection", "compass"]:
    truth[ang] = (np.rad2deg(np.arctan2(truth[f"{ang}_sin"], truth[f"{ang}_cos"])) % 360)

last_obs = df_num_full.iloc[-HORIZON - 1]
for ang in ["windDirection", "currentDirection", "compass"]:
    last_obs[ang] = (np.rad2deg(np.arctan2(last_obs[f"{ang}_sin"], last_obs[f"{ang}_cos"])) % 360)

report_params = [c for c in GOOD_PARAMS if not c.endswith(("_sin", "_cos"))] + \
                [a for a in ["windDirection", "currentDirection", "compass"] if f"{a}_sin" in GOOD_PARAMS]
CIRCULAR_PARAMS = {"windDirection", "currentDirection", "compass"}

def circ_mae(true, pred):
    return np.abs((true - pred + 180) % 360 - 180).mean()""")

md("## 3. Shared windowing and training loop")
code(r"""def make_direct_windows(scaled_df, lookback, horizon, out_idx):
    arr = scaled_df.values.astype(np.float32)
    X, Y = [], []
    for origin in range(lookback, len(arr) - horizon):
        X.append(arr[origin - lookback:origin])
        Y.append(arr[origin:origin + horizon][:, out_idx])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X_direct, Y_good = make_direct_windows(train_scaled, LOOKBACK, HORIZON, good_idx)
last_window = torch.from_numpy(train_scaled.values[-LOOKBACK:].astype(np.float32)).unsqueeze(0)


class ITransformer(nn.Module):
    def __init__(self, lookback, n_features, horizon, out_idx, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.out_idx = out_idx
        self.embed = nn.Linear(lookback, d_model)
        self.var_id = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 2,
                                            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        tok = self.embed(x.transpose(1, 2)) + self.var_id.unsqueeze(0)
        tok = self.encoder(tok)
        out = self.head(tok)
        return out.transpose(1, 2)[:, :, self.out_idx]


def train_model(model, X_tr, Y_tr, X_val, Y_val, epochs=150, batch_size=64, lr=1e-3,
                 patience=20, name=""):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
    loss_fn = nn.MSELoss()
    best_val, best_state, wait = float("inf"), None, 0
    n = len(X_tr); t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            xb, yb = X_tr[b].to(device), Y_tr[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val.to(device)), Y_val.to(device)).item()
        sched.step(val_loss)
        if val_loss < best_val - 1e-5:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    print(f"{name:18s} best_val_loss={best_val:.4f}  epochs_run={ep+1:3d}  time={time.time()-t0:5.1f}s")
    return model""")

md("## 4. Train the baseline iTransformer 5 times, one per seed")
code(r"""all_runs = {}
for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_t, Y_good_t = torch.from_numpy(X_direct), torch.from_numpy(Y_good)
    n = len(X_t)
    perm = np.random.permutation(n)   # seed-dependent train/val split too, not just init
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, Y_tr = X_t[tr_idx], Y_good_t[tr_idx]
    X_val, Y_val = X_t[val_idx], Y_good_t[val_idx]

    model = ITransformer(LOOKBACK, n_features, HORIZON, good_idx, d_model=64, n_heads=4, n_layers=2)
    model = train_model(model, X_tr, Y_tr, X_val, Y_val, epochs=150, patience=20, name=f"seed={seed}")

    with torch.no_grad():
        pred_scaled = model(last_window.to(device))[0].cpu().numpy()
    preds_real = pred_scaled * std[GOOD_PARAMS].values + mean[GOOD_PARAMS].values
    pred_df = pd.DataFrame(preds_real, columns=GOOD_PARAMS, index=test_df.index)

    pred_final = pred_df.copy()
    for ang in ["windDirection", "currentDirection", "compass"]:
        if f"{ang}_sin" in pred_final.columns:
            pred_final[ang] = (np.rad2deg(np.arctan2(pred_final[f"{ang}_sin"], pred_final[f"{ang}_cos"])) % 360)
    all_runs[seed] = pred_final

print(f"\nCompleted {len(SEEDS)} independent training runs.")""")

md("## 5. The verdict: how much does skill actually move across seeds?")
code(r"""rows = []
for p in report_params:
    yt = truth[p].values
    yp_persist = np.repeat(last_obs[p], HORIZON)
    is_circular = p in CIRCULAR_PARAMS
    mae_p = circ_mae(yt, yp_persist) if is_circular else mean_absolute_error(yt, yp_persist)

    skills = []
    for seed in SEEDS:
        yhat = all_runs[seed][p].values
        mae = circ_mae(yt, yhat) if is_circular else mean_absolute_error(yt, yhat)
        skill = (1 - mae / mae_p) * 100 if mae_p > 0 else np.nan
        skills.append(skill)

    row = {"parameter": p, "mean_skill_%": round(float(np.mean(skills)), 1),
           "std_skill_%": round(float(np.std(skills)), 2),
           "min_skill_%": round(float(np.min(skills)), 1),
           "max_skill_%": round(float(np.max(skills)), 1),
           "range_pp": round(float(np.max(skills) - np.min(skills)), 1)}
    for seed, sk in zip(SEEDS, skills):
        row[f"seed_{seed}_skill_%"] = round(sk, 1)
    rows.append(row)

robustness_df = pd.DataFrame(rows).sort_values("range_pp", ascending=False).reset_index(drop=True)
robustness_df.to_csv("metrics_seed_robustness.csv", index=False)

overall_mean_per_seed = {seed: np.mean([all_runs[seed][p] is not None for p in report_params]) for seed in SEEDS}
mean_skill_per_seed = []
for seed in SEEDS:
    skills_this_seed = []
    for p in report_params:
        yt = truth[p].values
        yp_persist = np.repeat(last_obs[p], HORIZON)
        is_circular = p in CIRCULAR_PARAMS
        mae_p = circ_mae(yt, yp_persist) if is_circular else mean_absolute_error(yt, yp_persist)
        yhat = all_runs[seed][p].values
        mae = circ_mae(yt, yhat) if is_circular else mean_absolute_error(yt, yhat)
        skills_this_seed.append((1 - mae / mae_p) * 100 if mae_p > 0 else np.nan)
    mean_skill_per_seed.append(np.mean(skills_this_seed))

print(robustness_df[["parameter", "mean_skill_%", "std_skill_%", "min_skill_%", "max_skill_%", "range_pp"]].to_string(index=False))
print(f"\nOverall mean skill (across all 18 params), per seed:")
for seed, m in zip(SEEDS, mean_skill_per_seed):
    print(f"  seed={seed:6d}: {m:+.2f}%")
print(f"\nOverall mean skill across seeds: {np.mean(mean_skill_per_seed):+.2f}%  "
      f"(std across seeds: {np.std(mean_skill_per_seed):.2f}pp, "
      f"range: {np.max(mean_skill_per_seed)-np.min(mean_skill_per_seed):.2f}pp)")

n_volatile = int((robustness_df["range_pp"] > 5).sum())
print(f"\n{n_volatile}/{len(robustness_df)} parameters show >5pp range across seeds.")
if np.std(mean_skill_per_seed) < 1.0:
    print("VERDICT: the headline ~87% mean-skill result is robust to seed choice (sub-1pp spread).")
elif np.std(mean_skill_per_seed) < 3.0:
    print("VERDICT: moderate seed sensitivity (1-3pp spread on the mean) -- prior single-run "
          "comparisons should be read with that margin of uncertainty in mind.")
else:
    print("VERDICT: substantial seed sensitivity (>3pp spread on the mean) -- single-run comparisons "
          "throughout this project (including the Dual-Channel/SOFTS/RevIN ablation verdicts) carry "
          "real risk of being decided by random initialization rather than a genuine architecture difference.")""")

md("## 6. Plot: per-parameter skill spread across the 5 seeds")
code(r"""fig, ax = plt.subplots(figsize=(12, 7))
plot_df = robustness_df.sort_values("mean_skill_%")
for i, (_, row) in enumerate(plot_df.iterrows()):
    seed_vals = [row[f"seed_{s}_skill_%"] for s in SEEDS]
    ax.scatter(seed_vals, [i] * len(seed_vals), color="#1f77b4", alpha=0.7, s=40)
    ax.plot([row["min_skill_%"], row["max_skill_%"]], [i, i], color="lightgray", lw=1, zorder=0)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df["parameter"])
ax.set_xlabel("Skill vs persistence (%) across 5 seeds")
ax.set_title("Per-parameter skill spread across 5 random seeds (same architecture, same data)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("seed_robustness_plot.png", dpi=110)
plt.show()
print("Saved seed_robustness_plot.png")""")

md("## 7. Save outputs")
code(r"""print("Saved: metrics_seed_robustness.csv, seed_robustness_plot.png")""")

md(r"""## 8. Conclusion

Section 5 is the actual finding. This notebook doesn't crown a new winner — it puts an honest
uncertainty band around the baseline iTransformer's own headline result, which every ablation in this
project (Dual-Channel, SOFTS, RevIN) was compared against using only a single seed=42 run each. If the
spread here is small, those verdicts stand on solid ground. If it's large, the honest conclusion is
that those three "iTransformer wins" verdicts should be treated as suggestive, not conclusive — exactly
the caution the Miño Calero et al. paper raises about evaluating transformer-based forecasters.""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_RealEMS_iTransformer_SeedRobustness.ipynb")
print("Notebook written: Marine_Forecast_RealEMS_iTransformer_SeedRobustness.ipynb")
