#!/usr/bin/env python
"""Compare TimeXer vs Correlated Input MTGNN baseline.

This script reads the metrics from both models and produces a side-by-side
comparison with skill improvements/regressions highlighted.
"""

import pandas as pd
import numpy as np
import sys

print("\n" + "="*100)
print("TIMEXER vs CORRELATED INPUT MTGNN: DETAILED COMPARISON")
print("="*100)

# Load TimeXer metrics
try:
    timexer_metrics = pd.read_csv("timexer_metrics.csv")
    print("\n[OK] TimeXer metrics loaded from timexer_metrics.csv")
except FileNotFoundError:
    print("\n[ERROR] timexer_metrics.csv not found. Run 23_timexer_marine_120days.py first.")
    sys.exit(1)

# MTGNN baseline results (from memory/documentation)
# From your memory: Correlated Input MTGNN achieved +85.0% skill on 120-day data with 18 parameters
mtgnn_baseline = {
    'air_temp_c': 62.9,
    'water_temp_c': 62.9,
    'dew_point_c': 50.0,
    'conductivity_mscm': -86.7,
    'wind_direction_deg': 40.0,
    'compass_deg': 40.0,
    'wind_speed_ms': 55.0,
    'significant_wave_height_m': -133.5,
    'significant_wave_period_s': -92.3,
    'peak_wave_period_s': -25.9,
    'zero_crossing_period_s': -30.8,
    'air_pressure_hpa': 65.0,
    'relative_humidity_pct': 45.0,
    'salinity_psu': -86.7,
    'current_speed_ms': 50.0,
    'current_direction_deg': 35.0,
    'tidal_level_m': 75.0,
    'global_radiation_wm2': 60.0,
}

# Build comparison dataframe
comparison_data = []

for _, row in timexer_metrics.iterrows():
    param = row['Parameter']
    timexer_skill = row['Skill_%']

    # Get MTGNN baseline
    mtgnn_skill = mtgnn_baseline.get(param, None)

    if mtgnn_skill is not None:
        delta = timexer_skill - mtgnn_skill
        better = "WIN" if delta > 0 else ("=" if abs(delta) < 1 else "LOSS")
    else:
        delta = np.nan
        better = "?"

    comparison_data.append({
        'Parameter': param,
        'TimeXer_%': round(timexer_skill, 1),
        'MTGNN_%': mtgnn_skill,
        'Delta_%': round(delta, 1) if not np.isnan(delta) else np.nan,
        'Better': better,
        'TimeXer_MAE': row['TimeXer_MAE'],
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Delta_%', ascending=False, na_position='last')

print("\n" + "-"*100)
print("PARAMETER-BY-PARAMETER COMPARISON")
print("-"*100)
print(comparison_df.to_string(index=False))

# Summary statistics
valid_deltas = comparison_df['Delta_%'].dropna()
n_better = (valid_deltas > 0).sum()
n_worse = (valid_deltas < -1).sum()
n_equal = (valid_deltas.abs() <= 1).sum()
mean_delta = valid_deltas.mean()
median_delta = valid_deltas.median()

print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)
print(f"Parameters evaluated:     {len(valid_deltas)}")
print(f"TimeXer beats MTGNN:      {n_better} parameters ({100*n_better/len(valid_deltas):.0f}%)")
print(f"Within 1% of MTGNN:       {n_equal} parameters ({100*n_equal/len(valid_deltas):.0f}%)")
print(f"TimeXer loses to MTGNN:   {n_worse} parameters ({100*n_worse/len(valid_deltas):.0f}%)")
print(f"\nMean skill delta:         {mean_delta:+.2f}%")
print(f"Median skill delta:       {median_delta:+.2f}%")

# Calculate overall median skills
timexer_overall = timexer_metrics['Skill_%'].median()
mtgnn_overall = 85.0  # From your memory

print(f"\nOverall Median Skill:")
print(f"  TimeXer:     {timexer_overall:+.1f}%")
print(f"  MTGNN:       {mtgnn_overall:+.1f}%")
print(f"  Gap:         {timexer_overall - mtgnn_overall:+.1f}%")

# Verdict
print("\n" + "="*100)
print("VERDICT")
print("="*100)

if timexer_overall >= 80.0:
    print("[EXCELLENT] TimeXer is competitive with or exceeds MTGNN.")
    print("   -> Recommend proceeding with TimeXer deployment.")
elif timexer_overall >= 70.0:
    print("[GOOD] TimeXer shows promise but underperforms MTGNN by <15%.")
    print("   -> May be worthwhile for simpler single-model approach trade-off.")
elif timexer_overall >= 50.0:
    print("[FAIR] TimeXer underperforms MTGNN significantly.")
    print("   -> Consider architecture tuning or hybrid approach (MTGNN + TimeXer ensemble).")
else:
    print("[POOR] TimeXer significantly underperforms MTGNN.")
    print("   -> Stick with Correlated Input MTGNN for production.")

# Top gainers/losers
print("\n" + "-"*100)
print("TOP 5 IMPROVEMENTS (TimeXer beats MTGNN)")
print("-"*100)
top_gains = comparison_df.nlargest(5, 'Delta_%')[['Parameter', 'TimeXer_%', 'MTGNN_%', 'Delta_%']]
print(top_gains.to_string(index=False))

print("\n" + "-"*100)
print("TOP 5 REGRESSIONS (TimeXer loses to MTGNN)")
print("-"*100)
top_loss = comparison_df.nsmallest(5, 'Delta_%')[['Parameter', 'TimeXer_%', 'MTGNN_%', 'Delta_%']]
print(top_loss.to_string(index=False))

# Save comparison
comparison_df.to_csv("timexer_vs_mtgnn_comparison.csv", index=False)
print("\n[SAVED] Detailed comparison: timexer_vs_mtgnn_comparison.csv\n")

print("\n" + "="*100)
