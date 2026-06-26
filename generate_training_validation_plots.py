"""Generate training and validation period plots."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# Load data
print("Loading CSV data...")
df = pd.read_csv('marine_data_120days_1min.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df.columns = df.columns.str.replace('hutimestampmidity', 'humidity')

# Training period: Feb 23 - May 13, 2026
train_start = pd.Timestamp('2026-02-23')
train_end = pd.Timestamp('2026-05-13 23:59:59')
train_data = df[(df.index >= train_start) & (df.index <= train_end)].copy()

# Validation period: May 14 - Jun 22, 2026
val_start = pd.Timestamp('2026-05-14')
val_end = pd.Timestamp('2026-06-22 23:59:59')
val_data = df[(df.index >= val_start) & (df.index <= val_end)].copy()

print(f"Training data: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data) / (24*60):.0f} days)")
print(f"Validation data: {val_data.index[0]} to {val_data.index[-1]} ({len(val_data) / (24*60):.0f} days)")

# Create outputs directory
os.makedirs('static_plots/training_period', exist_ok=True)
os.makedirs('static_plots/validation_period', exist_ok=True)

# Define parameters with their units
categories = {
    'Atmosphere': {
        'air_temp_c': '(degree C)',
        'air_pressure_hpa': '(hPa)',
        'relative_humidity_pct': '(%)',
        'wind_speed_ms': '(m/s)'
    },
    'Marine_Water': {
        'tidal_level_m': '(m)',
        'water_temp_c': '(degree C)',
        'salinity_psu': '(psu)',
        'water_level_m': '(m)'
    },
    'Marine_Waves': {
        'significant_wave_height_m': '(m)',
        'significant_wave_period_s': '(s)',
        'zero_crossing_period_s': '(s)',
        'max_wave_height_m': '(m)'
    },
    'Marine_Current': {
        'current_speed_ms': '(m/s)',
        'current_direction_deg': '(degrees)'
    }
}

print("\nGenerating TRAINING PERIOD plots...")

for category_name, params in categories.items():
    existing_params = [p for p in params.keys() if p in train_data.columns]
    if not existing_params:
        continue

    fig, axes = plt.subplots(len(existing_params), 1, figsize=(14, 4*len(existing_params)))
    if len(existing_params) == 1:
        axes = [axes]

    for idx, param_name in enumerate(existing_params):
        ax = axes[idx]
        actual_values = train_data[param_name]

        ax.plot(actual_values.index, actual_values.values,
               linewidth=1.0, color='darkblue', alpha=0.8, label='Training Data', zorder=2)

        ax.set_ylabel(f'{param_name}\n{params[param_name]}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=8)

        mean_val = actual_values.mean()
        std_val = actual_values.std()
        ax.text(0.98, 0.95, f'Mean: {mean_val:.2f} | Std: {std_val:.2f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    fig.suptitle(f'{category_name.replace("_", " ")} - TRAINING PERIOD (Feb 23 - May 13, 110 days)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f'static_plots/training_period/{category_name}_training.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"  Created: {filename}")
    plt.close()

print("\nGenerating VALIDATION PERIOD plots...")

for category_name, params in categories.items():
    existing_params = [p for p in params.keys() if p in val_data.columns]
    if not existing_params:
        continue

    fig, axes = plt.subplots(len(existing_params), 1, figsize=(14, 4*len(existing_params)))
    if len(existing_params) == 1:
        axes = [axes]

    for idx, param_name in enumerate(existing_params):
        ax = axes[idx]
        actual_values = val_data[param_name]

        ax.plot(actual_values.index, actual_values.values,
               linewidth=1.0, color='darkgreen', alpha=0.8, label='Validation Data', zorder=2)

        ax.set_ylabel(f'{param_name}\n{params[param_name]}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=8)

        mean_val = actual_values.mean()
        std_val = actual_values.std()
        ax.text(0.98, 0.95, f'Mean: {mean_val:.2f} | Std: {std_val:.2f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    fig.suptitle(f'{category_name.replace("_", " ")} - VALIDATION PERIOD (May 14 - Jun 22, 40 days)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f'static_plots/validation_period/{category_name}_validation.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"  Created: {filename}")
    plt.close()

print("\n" + "="*90)
print("TRAINING & VALIDATION PLOTS GENERATED")
print("="*90)
print(f"Training Period: 2026-02-23 to 2026-05-13 (110 days, 158,400 records)")
print(f"Validation Period: 2026-05-14 to 2026-06-22 (40 days, 57,600 records)")
print(f"\nFiles saved to:")
print(f"  - static_plots/training_period/")
print(f"  - static_plots/validation_period/")
print("="*90)
