#!/usr/bin/env python
"""Prepare 120-day dataset: load, resample to 10-min, save."""
import pandas as pd
import numpy as np

print("\n" + "="*80)
print("STEP 1: DATA PREPARATION")
print("="*80)

# Load original 1-min data
print("\n[Loading] 120day_timestamp_18parameters.csv...")
df_1min = pd.read_csv("120day_timestamp_18parameters.csv", index_col=0, parse_dates=True)
print(f"  Loaded: {df_1min.shape[0]} rows, {df_1min.shape[1]} columns")
print(f"  Date range: {df_1min.index[0]} to {df_1min.index[-1]}")
print(f"  Columns: {list(df_1min.columns)}")

# Resample to 10-minute resolution
print("\n[Resampling] to 10-minute intervals...")
df_10min = df_1min.resample("10min").mean().dropna()
print(f"  Result: {df_10min.shape[0]} rows (120 days at 10-min = {df_10min.shape[0]/(24*6):.1f} days)")
print(f"  Date range: {df_10min.index[0]} to {df_10min.index[-1]}")

# Save with descriptive name
output_file = "marine_120day_18params_10min.csv"
df_10min.to_csv(output_file)
print(f"\n[Saved] {output_file}")
print(f"  Shape: {df_10min.shape}")
print(f"  Null values: {df_10min.isnull().sum().sum()}")

print("\n" + "="*80)
print(f"SUCCESS: Data ready for training")
print(f"  File: {output_file}")
print(f"  Rows: {df_10min.shape[0]}")
print(f"  Columns: {df_10min.shape[1]}")
print("="*80 + "\n")
