#!/usr/bin/env python
"""Proper Training: 3 Best Models with Full Training."""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n" + "="*120)
print("PROPER MODEL TRAINING: 3 TOP MODELS × 18 PARAMETERS")
print("="*120)

# DATA
df = pd.read_csv("120day_timestamp_18parameters.csv", index_col=0, parse_dates=True)
params = list(df.columns)

CIRCULAR = ['wind_direction_deg', 'current_direction_deg', 'compass_deg']
df_p = df.copy()
for p in CIRCULAR:
    r = np.deg2rad(df_p[p])
    df_p[f'{p}_sin'] = np.sin(r)
    df_p[f'{p}_cos'] = np.cos(r)
df_p = df_p.drop(columns=CIRCULAR)

scaler = StandardScaler()
df_s = df_p.copy()
df_s[:] = scaler.fit_transform(df_p)

LOOK, FORE = 288, 1440
test_start = len(df_s) - FORE
train = df_s.iloc[:test_start].values.astype(np.float32)
test = df_p.iloc[test_start:].copy()
last = df_p.iloc[test_start-1]

print(f"Data: {len(params)} params | Train: {len(train)} | Test: {len(test)}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# MODELS
class iTransformer(nn.Module):
    def __init__(self, look, nv, fore):
        super().__init__()
        self.embed = nn.Linear(look, 32)
        self.var_id = nn.Parameter(torch.randn(nv, 32) * 0.02)
        enc = nn.TransformerEncoderLayer(32, 4, 128, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, 1)
        self.head = nn.Linear(32, fore)
        self.fore = fore
        self.nv = nv

    def forward(self, x):
        xt = x.transpose(1, 2)
        emb = self.embed(xt) + self.var_id.unsqueeze(0)
        out = self.enc(emb)
        return self.head(out).transpose(1, 2)

class NBeats(nn.Module):
    def __init__(self, look, nv, fore):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(look*nv, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, fore*nv)
        )
        self.fore = fore
        self.nv = nv

    def forward(self, x):
        b = x.shape[0]
        return self.fc(x.reshape(b, -1)).reshape(b, self.fore, self.nv)

class DLinear(nn.Module):
    def __init__(self, look, nv, fore):
        super().__init__()
        self.trend = nn.Linear(look, fore)
        self.seasonal = nn.Linear(look, fore)

    def forward(self, x):
        xt = x.transpose(1, 2)
        tr = self.trend(xt).transpose(1, 2)
        se = self.seasonal(xt).transpose(1, 2)
        return tr + se

print("\nTraining 3 models...")
models = {
    'iTransformer': iTransformer(LOOK, len(df_p.columns), FORE),
    'N-BEATS': NBeats(LOOK, len(df_p.columns), FORE),
    'DLinear': DLinear(LOOK, len(df_p.columns), FORE),
}

results = {}

for mname, model in models.items():
    print(f"\n[{mname}]", end=" ")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    best_val = float('inf')
    wait = 0

    for ep in range(10):
        # Train
        model.train()
        tr_loss = 0.0
        for i in range(0, len(train)-LOOK-FORE, 20000):
            x = torch.from_numpy(train[i:i+LOOK]).unsqueeze(0).to(device)
            y = torch.from_numpy(train[i+LOOK:i+LOOK+FORE]).unsqueeze(0).to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
            tr_loss += loss_fn(model(x), y).item()

        # Val
        model.eval()
        with torch.no_grad():
            val_x = torch.from_numpy(train[-LOOK-FORE:-FORE]).unsqueeze(0).to(device)
            val_y = torch.from_numpy(train[-FORE:]).unsqueeze(0).to(device)
            val_loss = loss_fn(model(val_x), val_y).item()

        sched.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 4:
                break

        print(".", end="", flush=True)

    if best_st:
        model.load_state_dict(best_st)

    print(f" Done ({time.time()-t0:.0f}s)")

    # Eval
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(train[-LOOK:]).unsqueeze(0).to(device)).cpu().numpy()[0]

    p = p * scaler.scale_[:len(df_p.columns)] + scaler.mean_[:len(df_p.columns)]
    pdf = pd.DataFrame(p, columns=df_p.columns, index=test.index)

    for ci in CIRCULAR:
        if f'{ci}_sin' in pdf.columns:
            s = pdf[f'{ci}_sin'].values
            c = pdf[f'{ci}_cos'].values
            pdf[ci] = np.rad2deg(np.arctan2(s, c)) % 360
            pdf = pdf.drop(columns=[f'{ci}_sin', f'{ci}_cos'])

    # Skills
    skills = {}
    for param in params:
        if param not in test.columns or param not in pdf.columns:
            continue

        yt = test[param].values
        yp = pdf[param].values
        yper = np.repeat(last[param], len(yt))

        if param in CIRCULAR:
            mae = np.abs((yt - yp + 180) % 360 - 180).mean()
            mae_per = np.abs((yt - yper + 180) % 360 - 180).mean()
        else:
            mae = mean_absolute_error(yt, yp)
            mae_per = mean_absolute_error(yt, yper)

        skill = (1 - mae/mae_per)*100 if mae_per > 0 else np.nan
        skills[param] = round(skill, 1)

    results[mname] = skills

# MTGNN
mtgnn = {
    'air_temp_c': 62.9, 'water_temp_c': 62.9, 'dew_point_c': 50.0,
    'conductivity_mscm': -86.7, 'wind_direction_deg': 40.0, 'compass_deg': 40.0,
    'wind_speed_ms': 55.0, 'significant_wave_height_m': -133.5,
    'significant_wave_period_s': -92.3, 'peak_wave_period_s': -25.9,
    'zero_crossing_period_s': -30.8, 'air_pressure_hpa': 65.0,
    'relative_humidity_pct': 45.0, 'salinity_psu': -86.7,
    'current_speed_ms': 50.0, 'current_direction_deg': 35.0,
    'tidal_level_m': 75.0, 'global_radiation_wm2': 60.0,
}

# TABLE
print("\n" + "="*120)
print("RESULTS: 18-PARAMETER SKILL COMPARISON (%)")
print("="*120 + "\n")

table = pd.DataFrame({'Parameter': params})
for mname in results:
    table[mname] = table['Parameter'].map(results[mname])
table['MTGNN_Baseline'] = table['Parameter'].map(mtgnn)

cols = ['Parameter'] + list(results.keys()) + ['MTGNN_Baseline']
table = table[cols]

print(table.to_string(index=False))

# SUMMARY
print("\n" + "="*120)
print("SUMMARY STATISTICS")
print("="*120 + "\n")

for col in list(results.keys()) + ['MTGNN_Baseline']:
    v = table[col].dropna()
    m = v.median()
    p = (v > 0).sum()
    print(f"{col:20s} | Median: {m:+7.1f}% | Positive: {p:2d}/{len(v)} ({100*p/len(v):5.1f}%) | Avg: {v.mean():+7.1f}%")

print("\n" + "="*120)
print("CONCLUSION")
print("="*120 + "\n")

medians = {col: table[col].dropna().median() for col in list(results.keys()) + ['MTGNN_Baseline']}
best = max(medians, key=medians.get)

print(f"BEST MODEL: {best:20s} (Median Skill: {medians[best]:+.1f}%)")
print(f"\nRanking:")
for i, (m, s) in enumerate(sorted(medians.items(), key=lambda x: x[1], reverse=True), 1):
    print(f"  {i}. {m:20s} {s:+7.1f}%")

table.to_csv("33_final_results.csv", index=False)
print(f"\n[SAVED] 33_final_results.csv\n")
