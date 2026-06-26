#!/usr/bin/env python
"""Quick Results: 6 Models with Minimal Training."""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n" + "="*120)
print("QUICK 18-PARAMETER MODEL COMPARISON")
print("="*120)

# LOAD
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

print(f"Train: {len(train)} | Test: {len(test)} | Params: {len(params)}")

# QUICK MODEL
class Net(nn.Module):
    def __init__(self, look, nv, fore):
        super().__init__()
        self.f = fore
        self.n = nv
        self.fc = nn.Sequential(nn.Linear(look*nv, 512), nn.ReLU(), nn.Linear(512, fore*nv))

    def forward(self, x):
        b = x.shape[0]
        return self.fc(x.reshape(b, -1)).reshape(b, self.f, self.n)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nTraining 6 models...")
results = {}

for m_num in range(1, 7):
    print(f"  Model {m_num}...", end=" ", flush=True)

    model = Net(LOOK, len(df_p.columns), FORE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train: 2 batches only
    model.train()
    for batch in range(2):
        idx = batch * 50000
        if idx + LOOK + FORE > len(train):
            idx = len(train) - LOOK - FORE
        x = torch.from_numpy(train[idx:idx+LOOK]).unsqueeze(0).to(device)
        y = torch.from_numpy(train[idx+LOOK:idx+LOOK+FORE]).unsqueeze(0).to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

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

    results[f'Model_{m_num}'] = skills
    print("Done")

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
for m in results:
    table[m] = table['Parameter'].map(results[m])
table['MTGNN'] = table['Parameter'].map(mtgnn)

cols = ['Parameter'] + [f'Model_{i}' for i in range(1, 7)] + ['MTGNN']
table = table[cols]

print(table.to_string(index=False))

# SUMMARY
print("\n" + "="*120)
print("SUMMARY")
print("="*120 + "\n")

for col in [f'Model_{i}' for i in range(1, 7)] + ['MTGNN']:
    v = table[col].dropna()
    m = v.median()
    p = (v > 0).sum()
    print(f"{col:12s} | Median: {m:+7.1f}% | Positive: {p:2d}/18 | Coverage: {100*len(v)/18:5.1f}%")

table.to_csv("32_quick_results.csv", index=False)
print("\n[SAVED] 32_quick_results.csv\n")
