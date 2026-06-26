#!/usr/bin/env python
"""Compare iTransformer vs HPMixer on 118-day training."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print("\n" + "="*80)
print("MODEL COMPARISON: iTransformer vs HPMixer")
print("="*80)

try:
    # Load results
    it_results = pd.read_csv("itransformer_118day_results.csv")
    hpm_results = pd.read_csv("hpmixer_118day_results.csv")

    it_metrics = pd.read_csv("itransformer_118day_metrics.csv")
    hpm_metrics = pd.read_csv("hpmixer_118day_metrics.csv")

    print("\n[OK] Loaded results for both models")

except FileNotFoundError as e:
    print(f"\n[ERROR] Results not ready yet: {e}")
    print("Waiting for training to complete...")
    exit(1)

# ===== SUMMARY COMPARISON =====
print("\n" + "="*80)
print("SUMMARY METRICS")
print("="*80)

summary = pd.DataFrame([
    {
        "Model": "iTransformer",
        "Skill_%": it_results["Skill_%"].values[0],
        "MAE": it_results["MAE"].values[0],
        "RMSE": it_results["RMSE"].values[0],
        "Training_Time_s": it_results["Training_Time_s"].values[0],
        "Inference_Time_ms": it_results["Inference_Time_ms"].values[0],
    },
    {
        "Model": "HPMixer",
        "Skill_%": hpm_results["Skill_%"].values[0],
        "MAE": hpm_results["MAE"].values[0],
        "RMSE": hpm_results["RMSE"].values[0],
        "Training_Time_s": hpm_results["Training_Time_s"].values[0],
        "Inference_Time_ms": hpm_results["Inference_Time_ms"].values[0],
    }
])

print("\n" + summary.to_string(index=False))

# Determine winner for each metric
print("\n" + "="*80)
print("WINNER BY METRIC")
print("="*80)

metrics_to_compare = {
    "Skill_%": "Higher is better",
    "MAE": "Lower is better",
    "RMSE": "Lower is better",
    "Training_Time_s": "Lower is better",
    "Inference_Time_ms": "Lower is better",
}

for metric, description in metrics_to_compare.items():
    it_val = it_results[metric].values[0]
    hpm_val = hpm_results[metric].values[0]

    if "Lower" in description:
        winner = "HPMixer" if hpm_val < it_val else "iTransformer"
        diff = abs(it_val - hpm_val)
    else:
        winner = "iTransformer" if it_val > hpm_val else "HPMixer"
        diff = abs(it_val - hpm_val)

    print(f"\n{metric}:")
    print(f"  iTransformer: {it_val:.4f}")
    print(f"  HPMixer:      {hpm_val:.4f}")
    print(f"  Winner: {winner} (diff: {diff:.4f})")

# ===== PARAMETER-LEVEL COMPARISON =====
print("\n" + "="*80)
print("PARAMETER-LEVEL PERFORMANCE")
print("="*80)

# Merge metrics
comparison_df = it_metrics[["Parameter", "Skill_%"]].copy()
comparison_df.rename(columns={"Skill_%": "iTransformer_Skill"}, inplace=True)
comparison_df["HPMixer_Skill"] = hpm_metrics["Skill_%"].values
comparison_df["Skill_Difference"] = comparison_df["HPMixer_Skill"] - comparison_df["iTransformer_Skill"]

print("\nParameters where iTransformer is better:")
better_it = comparison_df[comparison_df["Skill_Difference"] < 0].sort_values("Skill_Difference").head(5)
for _, row in better_it.iterrows():
    print(f"  {row['Parameter']:30s} iT: {row['iTransformer_Skill']:+6.1f}% | HPM: {row['HPMixer_Skill']:+6.1f}% | Diff: {row['Skill_Difference']:+6.1f}%")

print("\nParameters where HPMixer is better:")
better_hpm = comparison_df[comparison_df["Skill_Difference"] > 0].sort_values("Skill_Difference", ascending=False).head(5)
for _, row in better_hpm.iterrows():
    print(f"  {row['Parameter']:30s} iT: {row['iTransformer_Skill']:+6.1f}% | HPM: {row['HPMixer_Skill']:+6.1f}% | Diff: {row['Skill_Difference']:+6.1f}%")

# ===== GENERATE PLOTS =====
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: Overall Metrics Comparison
fig, axes = plt.subplots(1, 5, figsize=(18, 5))

metrics_list = ["Skill_%", "MAE", "RMSE", "Training_Time_s", "Inference_Time_ms"]
titles = ["Skill (%)", "MAE", "RMSE", "Training Time (s)", "Inference Time (ms)"]

for idx, (metric, title) in enumerate(zip(metrics_list, titles)):
    ax = axes[idx]
    values = [it_results[metric].values[0], hpm_results[metric].values[0]]
    colors = ['#1f77b4', '#ff7f0e']

    bars = ax.bar(['iTransformer', 'HPMixer'], values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('plot_10_model_comparison.png', dpi=150, bbox_inches='tight')
print("\n[Saved] plot_10_model_comparison.png")
plt.close()

# Plot 2: Parameter Skill Comparison
fig, ax = plt.subplots(figsize=(14, 10))

x = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.barh(x - width/2, comparison_df['iTransformer_Skill'], width, label='iTransformer', alpha=0.8, color='#1f77b4', edgecolor='black')
bars2 = ax.barh(x + width/2, comparison_df['HPMixer_Skill'], width, label='HPMixer', alpha=0.8, color='#ff7f0e', edgecolor='black')

ax.set_yticks(x)
ax.set_yticklabels(comparison_df['Parameter'], fontsize=9)
ax.set_xlabel('Skill (%)', fontweight='bold')
ax.set_title('Parameter-Level Skill Comparison: iTransformer vs HPMixer', fontweight='bold', fontsize=13)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('plot_11_parameter_comparison.png', dpi=150, bbox_inches='tight')
print("[Saved] plot_11_parameter_comparison.png")
plt.close()

# Plot 3: Skill Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.hist(it_metrics['Skill_%'], bins=10, alpha=0.7, color='#1f77b4', edgecolor='black', label='iTransformer')
ax1.axvline(it_results['Skill_%'].values[0], color='#1f77b4', linestyle='--', linewidth=2, label=f"Mean: {it_results['Skill_%'].values[0]:.1f}%")
ax1.set_xlabel('Skill (%)', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('iTransformer: Skill Distribution', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.hist(hpm_metrics['Skill_%'], bins=10, alpha=0.7, color='#ff7f0e', edgecolor='black', label='HPMixer')
ax2.axvline(hpm_results['Skill_%'].values[0], color='#ff7f0e', linestyle='--', linewidth=2, label=f"Mean: {hpm_results['Skill_%'].values[0]:.1f}%")
ax2.set_xlabel('Skill (%)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('HPMixer: Skill Distribution', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_12_skill_distribution.png', dpi=150, bbox_inches='tight')
print("[Saved] plot_12_skill_distribution.png")
plt.close()

# Save comparison table
comparison_df.to_csv("model_comparison_118day.csv", index=False)
summary.to_csv("model_comparison_summary.csv", index=False)

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print("Files:")
print("  - model_comparison_118day.csv (parameter-level)")
print("  - model_comparison_summary.csv (overall)")
print("  - plot_10_model_comparison.png")
print("  - plot_11_parameter_comparison.png")
print("  - plot_12_skill_distribution.png")
print("="*80 + "\n")

# Print final recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

it_skill = it_results['Skill_%'].values[0]
hpm_skill = hpm_results['Skill_%'].values[0]
it_time = it_results['Training_Time_s'].values[0]
hpm_time = hpm_results['Training_Time_s'].values[0]

print(f"\n[Accuracy]")
if abs(it_skill - hpm_skill) < 2:
    print(f"  Skills are very close (iT: {it_skill:+.1f}%, HPM: {hpm_skill:+.1f}%)")
    print(f"  -> Consider using faster model")
elif it_skill > hpm_skill:
    print(f"  iTransformer better (iT: {it_skill:+.1f}% > HPM: {hpm_skill:+.1f}%)")
    print(f"  -> Use iTransformer for best accuracy")
else:
    print(f"  HPMixer better (HPM: {hpm_skill:+.1f}% > iT: {it_skill:+.1f}%)")
    print(f"  -> Use HPMixer for best accuracy")

print(f"\n[Speed]")
speedup = it_time / hpm_time
if speedup > 1.2:
    print(f"  HPMixer {speedup:.1f}x faster (iT: {it_time:.0f}s, HPM: {hpm_time:.0f}s)")
    print(f"  -> Use HPMixer for deployment")
elif speedup < 0.8:
    print(f"  iTransformer {1/speedup:.1f}x faster")
    print(f"  -> Use iTransformer for deployment")
else:
    print(f"  Training times are similar")

print("\n" + "="*80 + "\n")
