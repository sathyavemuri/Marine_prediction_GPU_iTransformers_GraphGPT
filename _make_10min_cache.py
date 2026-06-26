"""One-time: downsample the raw 1-minute, 43200-row Excel file to a 10-minute,
4320-row CSV cache. Re-parsing the 7.7MB xlsx (openpyxl) on every run/debug attempt
is slow and unnecessary -- the notebook and all debugging should load this instead."""
import time
import numpy as np
import pandas as pd

t0 = time.time()
raw = pd.read_excel("Simulation_30days_Data_31parameters.xlsx", sheet_name="AllParameters")
print(f"Read raw xlsx (1-min, {raw.shape[0]} rows): {time.time()-t0:.1f}s")

raw.columns = [c.split(" (")[0] for c in raw.columns]
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
raw = raw.set_index("timestamp")

t0 = time.time()
numeric_cols = raw.columns.drop("precipitationType")
df_num = raw[numeric_cols].resample("10min").mean()
df_cat = raw[["precipitationType"]].resample("10min").agg(lambda s: s.mode().iat[0])
df_10min = df_num.join(df_cat)
print(f"Resampled to 10-min ({df_10min.shape[0]} rows): {time.time()-t0:.1f}s")

df_10min.to_csv("ems_10min_resampled.csv")
print("Saved: ems_10min_resampled.csv")

t0 = time.time()
check = pd.read_csv("ems_10min_resampled.csv", index_col=0, parse_dates=True)
print(f"Reload from CSV cache: {time.time()-t0:.2f}s  shape={check.shape}")
