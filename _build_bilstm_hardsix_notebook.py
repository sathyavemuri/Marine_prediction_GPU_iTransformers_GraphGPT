import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Marine 48h Forecast — v10: BiLSTM for the Hard 6

A tenth attempt at the 6 historically hard parameters (visibility ×4, precipitation ×2), testing
**BiLSTM** (bidirectional LSTM) chosen over plain unidirectional LSTM for a specific reason: in this
project's direct multi-horizon setup, the recurrent network is used purely as an **encoder** of an
already-fully-observed 2-day lookback window (not as an autoregressive generator) — there's no
"future leakage" concern with bidirectionality here, since both directions only ever see already-
observed history. Richer bidirectional context should help characterize the symmetric, rare-event
window shapes (like the fog dip's smooth ramp-down-then-up) the way DET v6's patch attention did.

**Speed was checked first, not assumed:** plain LSTM already ran at this exact setup in the original
11-model bake-off at 188.8s/47 epochs — comparable to iTransformer, faster per-epoch than PatchTST.
BiLSTM costs roughly 1.5-2x that. Not a concern at this scale.

**Univariate per parameter** (6 independent small models), consistent with this project's repeated
finding (DET v6, and visibility-forecasting literature) that cross-channel mixing hurts these 6
specifically. **Tweedie head for the 2 precipitation parameters** (plain regression on zero-inflated
data is already proven to fail, twice, in this project); **direct regression head for the 4 visibility
parameters**.

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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")
torch.set_num_threads(8)

print("PyTorch:", torch.__version__, "| torch threads:", torch.get_num_threads())""")

md("## 1. Load data")
code(r"""df_10min = pd.read_csv("ems_10min_resampled.csv", index_col=0, parse_dates=True)
PRECIP_PARAMS = ["precipitationIntensity", "precipitationDifference"]
VISIBILITY_PARAMS = ["twentyFourHourAvgVisibility", "tenMinuteAvgVisibility",
                      "oneMinuteAvgVisibility", "oneHourAvgVisibility"]
HARD_PARAMS = PRECIP_PARAMS + VISIBILITY_PARAMS

LOOKBACK, HORIZON = 288, 288
idx = df_10min.index
calendar_df = pd.DataFrame({
    "hour_sin": np.sin(2 * np.pi * idx.hour / 24), "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
    "dom_sin": np.sin(2 * np.pi * idx.day / 30), "dom_cos": np.cos(2 * np.pi * idx.day / 30),
}, index=idx)
calendar_cols = list(calendar_df.columns)

train_df = df_10min.iloc[:-HORIZON]
test_df = df_10min.iloc[-HORIZON:]
train_calendar = calendar_df.iloc[:-HORIZON]
print(f"Train: {train_df.shape[0]} rows  |  Test: {test_df.shape[0]} rows")""")

md("## 2. BiLSTM encoder + task-appropriate head")
code(r"""def tweedie_deviance_loss(mu, y, p=1.5, eps=1e-6):
    mu = mu.clamp_min(eps)
    a = y * torch.pow(mu, 1 - p) / (1 - p)
    b = torch.pow(mu, 2 - p) / (2 - p)
    return (-a + b).mean()


class BiLSTMForecaster(nn.Module):
    def __init__(self, n_calendar, horizon, hidden=64, n_layers=2, dropout=0.1, positive_head=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1 + n_calendar, hidden_size=hidden, num_layers=n_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden * 2, horizon)
        self.positive_head = positive_head

    def forward(self, x_target, x_calendar):
        x = torch.cat([x_target, x_calendar], dim=-1)         # (B, lookback, 1+n_calendar)
        _, (h_n, _) = self.lstm(x)
        context = torch.cat([h_n[-2], h_n[-1]], dim=-1)         # last layer: forward + backward
        out = self.head(context)
        if self.positive_head:
            out = torch.nn.functional.softplus(out)
        return out


def build_windows(series, calendar_arr, lookback, horizon):
    n = len(series)
    X_t, X_c, Y = [], [], []
    for origin in range(lookback, n - horizon):
        X_t.append(series[origin - lookback:origin])
        X_c.append(calendar_arr[origin - lookback:origin])
        Y.append(series[origin:origin + horizon])
    return np.array(X_t, dtype=np.float32), np.array(X_c, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_model(model, Xt, Xc, Y, loss_fn, epochs=100, batch_size=64, lr=1e-3, patience=15, name=""):
    Xt_t, Xc_t, Y_t = torch.from_numpy(Xt), torch.from_numpy(Xc), torch.from_numpy(Y)
    n = len(Xt_t); n_val = max(1, int(0.1 * n))
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xt_tr, Xc_tr, Y_tr = Xt_t[tr_idx], Xc_t[tr_idx], Y_t[tr_idx]
    Xt_val, Xc_val, Y_val = Xt_t[val_idx], Xc_t[val_idx], Y_t[val_idx]

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
    best_val, best_state, wait = float("inf"), None, 0
    t0 = time.time()
    n_tr = len(Xt_tr)
    for ep in range(epochs):
        model.train()
        perm_b = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            b = perm_b[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt_tr[b].unsqueeze(-1), Xc_tr[b])
            loss = loss_fn(pred, Y_tr[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xt_val.unsqueeze(-1), Xc_val), Y_val).item()
        sched.step(val_loss)
        if val_loss < best_val - 1e-6:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    print(f"{name:28s} best_val_loss={best_val:.4f}  epochs_run={ep+1:3d}  time={time.time()-t0:5.1f}s")
    return model""")

md("## 3. Train the precipitation models (BiLSTM + Tweedie head)")
code(r"""precip_pred = {}
for c in PRECIP_PARAMS:
    series_train = train_df[c].values.astype(np.float32)
    cal_train = train_calendar.values.astype(np.float32)
    Xt, Xc, Y = build_windows(series_train, cal_train, LOOKBACK, HORIZON)

    model = BiLSTMForecaster(len(calendar_cols), HORIZON, hidden=64, n_layers=2, positive_head=True)
    model = train_model(model, Xt, Xc, Y, loss_fn=lambda mu, y: tweedie_deviance_loss(mu, y, p=1.5),
                         epochs=100, patience=15, name=f"BiLSTM-precip:{c}")

    last_t = torch.from_numpy(series_train[-LOOKBACK:]).unsqueeze(0).unsqueeze(-1)
    last_c = torch.from_numpy(cal_train[-LOOKBACK:]).unsqueeze(0)
    with torch.no_grad():
        pred = model(last_t, last_c)[0].numpy()
    precip_pred[c] = np.clip(pred, 0, None)
print("Precipitation BiLSTM models trained.")""")

md("## 4. Train the visibility models (BiLSTM + regression head)")
code(r"""visibility_pred = {}
huber = nn.SmoothL1Loss()
for c in VISIBILITY_PARAMS:
    series_train = train_df[c].values.astype(np.float32)
    cal_train = train_calendar.values.astype(np.float32)
    mean_c, std_c = series_train.mean(), series_train.std()
    series_scaled = (series_train - mean_c) / std_c

    Xt, Xc, Y = build_windows(series_scaled, cal_train, LOOKBACK, HORIZON)
    model = BiLSTMForecaster(len(calendar_cols), HORIZON, hidden=64, n_layers=2, positive_head=False)
    model = train_model(model, Xt, Xc, Y, loss_fn=huber, epochs=100, patience=15, name=f"BiLSTM-visibility:{c}")

    last_t = torch.from_numpy(series_scaled[-LOOKBACK:]).unsqueeze(0).unsqueeze(-1)
    last_c = torch.from_numpy(cal_train[-LOOKBACK:]).unsqueeze(0)
    with torch.no_grad():
        pred_s = model(last_t, last_c)[0].numpy()
    visibility_pred[c] = pred_s * std_c + mean_c
print("Visibility BiLSTM models trained.")""")

md("## 5. Score against persistence and all nine prior hard-6 attempts")
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
ZIDF_V9_SKILL = {
    "twentyFourHourAvgVisibility": 11.0, "precipitationDifference": -9.9, "tenMinuteAvgVisibility": -26.5,
    "oneMinuteAvgVisibility": -33.2, "precipitationIntensity": -70.8, "oneHourAvgVisibility": -80.9,
}

all_pred = {**precip_pred, **visibility_pred}
truth = df_10min.iloc[-HORIZON:]
last_obs = df_10min.iloc[-HORIZON - 1]

metrics = []
for c in HARD_PARAMS:
    yt = truth[c].values
    yp_persist = np.repeat(last_obs[c], HORIZON)
    mae_p = mean_absolute_error(yt, yp_persist)
    mae = mean_absolute_error(yt, all_pred[c])
    rmse = np.sqrt(mean_squared_error(yt, all_pred[c]))
    skill = (1 - mae / mae_p) * 100 if mae_p > 0 else np.nan
    metrics.append({
        "parameter": c, "Persistence_MAE": round(mae_p, 4),
        "bilstm_v10_MAE": round(mae, 4), "bilstm_v10_RMSE": round(rmse, 4), "bilstm_v10_skill_%": round(skill, 1),
        "pure_iTransformer_skill_%": PURE_ITRANSFORMER_SKILL[c],
        "deepar_hybrid_skill_%": DEEPAR_HYBRID_SKILL[c], "det_v6_skill_%": DET_V6_SKILL[c],
        "tsb_v8_skill_%": TSB_V8_SKILL.get(c, np.nan), "zidf_v9_skill_%": ZIDF_V9_SKILL[c],
    })

metrics_df = pd.DataFrame(metrics).sort_values("bilstm_v10_skill_%", ascending=False).reset_index(drop=True)
metrics_df.to_csv("metrics_bilstm_v10.csv", index=False)
print(metrics_df.to_string(index=False))

mean_v10 = metrics_df["bilstm_v10_skill_%"].mean()
mean_deepar = metrics_df["deepar_hybrid_skill_%"].mean()
print(f"\nMean skill -- v10 (BiLSTM): {mean_v10:+.1f}%  |  DeepAR-hybrid (best prior): {mean_deepar:+.1f}%")
if mean_v10 > mean_deepar:
    print("VERDICT: v10 is the new best result for the hard 6.")
else:
    print("VERDICT: v10 does not beat DeepAR-hybrid on average.")

n_beats = int((metrics_df["bilstm_v10_skill_%"] > metrics_df["deepar_hybrid_skill_%"]).sum())
print(f"v10 beats DeepAR-hybrid on {n_beats}/6 parameters")""")

md("## 6. Plot all 6")
code(r"""hist_tail = df_10min.iloc[-HORIZON - LOOKBACK:-HORIZON]
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, c in zip(axes.ravel(), HARD_PARAMS):
    ax.plot(hist_tail.index, hist_tail[c], color="lightgray", lw=1, label="history")
    ax.plot(truth.index, truth[c], color="black", lw=2, label="actual")
    ax.plot(truth.index, all_pred[c], color="#17becf", lw=1.5, ls="--", label="BiLSTM v10")
    ax.axvline(truth.index[0], color="green", lw=1, alpha=0.5)
    ax.set_title(c, fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("bilstm_v10_hard6_plot.png", dpi=110)
plt.show()
print("Saved bilstm_v10_hard6_plot.png")""")

md("## 7. Save outputs for the dashboard")
code(r"""fva = pd.DataFrame({"timestamp": truth.index})
for c in HARD_PARAMS:
    fva[f"{c}__actual"] = truth[c].values
    fva[f"{c}__bilstm_v10"] = all_pred[c]
fva.to_csv("forecast_vs_actual_bilstm_v10.csv", index=False)
print("Saved: metrics_bilstm_v10.csv, forecast_vs_actual_bilstm_v10.csv, bilstm_v10_hard6_plot.png")""")

md(r"""## 8. Conclusion

Section 5 is the actual verdict. This tests whether BiLSTM's bidirectional encoding of the lookback
window — a legitimate, leakage-free technique in this direct-multi-horizon setup — extracts a
meaningfully richer representation than DET v6's patch+attention encoder, given the same
task-appropriate output heads (Tweedie for precipitation, direct regression for visibility). If it
doesn't beat DeepAR-hybrid, that's the tenth architecturally distinct approach to land near the same
ceiling, continuing to point at data volume as the binding constraint rather than encoder choice.""")

nb["cells"] = cells
nbf.write(nb, "Marine_Forecast_RealEMS_BiLSTM_HardSix.ipynb")
print("Notebook written: Marine_Forecast_RealEMS_BiLSTM_HardSix.ipynb")
