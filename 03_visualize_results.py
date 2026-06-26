#!/usr/bin/env python
"""Visualize iTransformer 2-day forecast results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

print("\n[Plotting] iTransformer 2-Day Forecast Results...")

# Load results
metrics_df = pd.read_csv("itransformer_2day_metrics.csv")
results_df = pd.read_csv("itransformer_2day_results.csv")

# Load predictions (reconstruct from data)
df = pd.read_csv("marine_120day_18params_10min.csv", index_col=0, parse_dates=True)
df_10min = df

horizon = 288
test_start = len(df_10min) - horizon
test_df = df_10min.iloc[test_start:].copy()

params = df_10min.columns.tolist()

# ===== PLOT 1: SKILL DISTRIBUTION =====
fig, ax = plt.subplots(figsize=(14, 6))
colors = ['green' if x > 0 else 'red' for x in metrics_df['Skill_%']]
bars = ax.barh(metrics_df['Parameter'], metrics_df['Skill_%'], color=colors, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_xlabel('Skill (%)', fontsize=12, fontweight='bold')
ax.set_title('iTransformer 2-Day Forecast: Per-Parameter Skill', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(metrics_df.iterrows()):
    ax.text(row['Skill_%'] + 2, i, f"{row['Skill_%']:+.1f}%", va='center', fontsize=9)
plt.tight_layout()
plt.savefig('plot_01_skill_distribution.png', dpi=150, bbox_inches='tight')
print("  [Saved] plot_01_skill_distribution.png")
plt.close()

# ===== PLOT 2: TOP 10 PERFORMERS =====
fig, ax = plt.subplots(figsize=(12, 6))
top_10 = metrics_df.nlargest(10, 'Skill_%')
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_10)))
bars = ax.bar(range(len(top_10)), top_10['Skill_%'], color=colors, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(top_10)))
ax.set_xticklabels(top_10['Parameter'], rotation=45, ha='right')
ax.set_ylabel('Skill (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Parameters: Highest Skill', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
for i, (idx, row) in enumerate(top_10.iterrows()):
    ax.text(i, row['Skill_%'] + 2, f"{row['Skill_%']:+.1f}%", ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_02_top_10_parameters.png', dpi=150, bbox_inches='tight')
print("  [Saved] plot_02_top_10_parameters.png")
plt.close()

# ===== PLOT 3: BOTTOM 10 PERFORMERS =====
fig, ax = plt.subplots(figsize=(12, 6))
bottom_10 = metrics_df.nsmallest(10, 'Skill_%')
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(bottom_10)))
bars = ax.bar(range(len(bottom_10)), bottom_10['Skill_%'], color=colors, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(bottom_10)))
ax.set_xticklabels(bottom_10['Parameter'], rotation=45, ha='right')
ax.set_ylabel('Skill (%)', fontsize=12, fontweight='bold')
ax.set_title('Bottom 10 Parameters: Lowest Skill', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='green', linestyle='--', linewidth=2)
for i, (idx, row) in enumerate(bottom_10.iterrows()):
    ax.text(i, row['Skill_%'] - 10, f"{row['Skill_%']:+.1f}%", ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_03_bottom_10_parameters.png', dpi=150, bbox_inches='tight')
print("  [Saved] plot_03_bottom_10_parameters.png")
plt.close()

# ===== PLOT 4: MAE COMPARISON =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Top performers by skill
top_5 = metrics_df.nlargest(5, 'Skill_%')
ax1.bar(range(len(top_5)), top_5['MAE'], color='green', alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(top_5)))
ax1.set_xticklabels(top_5['Parameter'], rotation=45, ha='right')
ax1.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax1.set_title('Top 5: Mean Absolute Error', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Bottom performers by skill
bottom_5 = metrics_df.nsmallest(5, 'Skill_%')
ax2.bar(range(len(bottom_5)), bottom_5['MAE'], color='red', alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(bottom_5)))
ax2.set_xticklabels(bottom_5['Parameter'], rotation=45, ha='right')
ax2.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax2.set_title('Bottom 5: Mean Absolute Error', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plot_04_mae_comparison.png', dpi=150, bbox_inches='tight')
print("  [Saved] plot_04_mae_comparison.png")
plt.close()

# ===== PLOT 5: SKILL vs MAE SCATTER =====
fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(metrics_df['MAE'], metrics_df['Skill_%'],
                     s=200, alpha=0.6, c=metrics_df['Skill_%'],
                     cmap='RdYlGn', edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Neutral Skill')
ax.set_xlabel('MAE', fontsize=12, fontweight='bold')
ax.set_ylabel('Skill (%)', fontsize=12, fontweight='bold')
ax.set_title('Parameter Performance: Skill vs MAE', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Annotate points
for idx, row in metrics_df.iterrows():
    ax.annotate(row['Parameter'][:10],
                (row['MAE'], row['Skill_%']),
                fontsize=8,
                xytext=(5, 5),
                textcoords='offset points',
                alpha=0.7)

cbar = plt.colorbar(scatter, ax=ax, label='Skill (%)')
plt.tight_layout()
plt.savefig('plot_05_skill_vs_mae.png', dpi=150, bbox_inches='tight')
print("  [Saved] plot_05_skill_vs_mae.png")
plt.close()

# ===== PLOT 6: SUMMARY STATISTICS =====
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Skill distribution histogram
axes[0, 0].hist(metrics_df['Skill_%'], bins=10, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(metrics_df['Skill_%'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {metrics_df['Skill_%'].mean():.1f}%")
axes[0, 0].set_xlabel('Skill (%)', fontweight='bold')
axes[0, 0].set_ylabel('Count', fontweight='bold')
axes[0, 0].set_title('Skill Distribution', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MAE distribution
axes[0, 1].hist(metrics_df['MAE'], bins=10, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(metrics_df['MAE'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {metrics_df['MAE'].mean():.2f}")
axes[0, 1].set_xlabel('MAE', fontweight='bold')
axes[0, 1].set_ylabel('Count', fontweight='bold')
axes[0, 1].set_title('MAE Distribution', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# RMSE distribution
axes[1, 0].hist(metrics_df['RMSE'], bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(metrics_df['RMSE'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {metrics_df['RMSE'].mean():.2f}")
axes[1, 0].set_xlabel('RMSE', fontweight='bold')
axes[1, 0].set_ylabel('Count', fontweight='bold')
axes[1, 0].set_title('RMSE Distribution', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Summary stats text
axes[1, 1].axis('off')
summary_text = f"""
SUMMARY STATISTICS

Skill (%):
  Mean: {metrics_df['Skill_%'].mean():.2f}%
  Median: {metrics_df['Skill_%'].median():.2f}%
  Std: {metrics_df['Skill_%'].std():.2f}%
  Min: {metrics_df['Skill_%'].min():.2f}%
  Max: {metrics_df['Skill_%'].max():.2f}%

MAE:
  Mean: {metrics_df['MAE'].mean():.4f}
  Median: {metrics_df['MAE'].median():.4f}
  Std: {metrics_df['MAE'].std():.4f}

RMSE:
  Mean: {metrics_df['RMSE'].mean():.4f}
  Median: {metrics_df['RMSE'].median():.4f}
  Std: {metrics_df['RMSE'].std():.4f}

Parameters: {len(metrics_df)}
Overall Skill: {results_df['Skill_%'].values[0]:.2f}%
Training Time: {results_df['Training_Time_s'].values[0]:.1f}s
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plot_06_summary_statistics.png', dpi=150, bbox_inches='tight')
print("  [Saved] plot_06_summary_statistics.png")
plt.close()

print("\n[OK] All plots saved successfully!")
print("\nGenerated files:")
print("  - plot_01_skill_distribution.png")
print("  - plot_02_top_10_parameters.png")
print("  - plot_03_bottom_10_parameters.png")
print("  - plot_04_mae_comparison.png")
print("  - plot_05_skill_vs_mae.png")
print("  - plot_06_summary_statistics.png")
