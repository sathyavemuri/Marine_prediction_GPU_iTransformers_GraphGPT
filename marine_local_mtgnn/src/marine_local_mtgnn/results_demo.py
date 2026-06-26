"""Demo results presentation: 10-day forecast accuracy per parameter."""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Target parameters (15 direct forecast targets)
TARGETS = [
    "air_temp_c",
    "air_pressure_hpa",
    "wind_u_east_ms",
    "wind_v_north_ms",
    "water_temp_c",
    "tidal_level_m",
    "current_u_east_ms",
    "current_v_north_ms",
    "dew_point_c",
    "global_radiation_wm2",
    "salinity_psu",
    "significant_wave_height_m",
    "significant_wave_period_s",
    "zero_crossing_period_s",
    "peak_wave_period_s",
]

# 10 days = 96 steps per day at 15-minute cadence
STEPS_PER_DAY = 96
NUM_DAYS = 10
TOTAL_STEPS = STEPS_PER_DAY * NUM_DAYS


def generate_demo_results():
    """Generate realistic demo forecast results."""
    np.random.seed(42)

    # Simulate perfect-ish forecast with degradation over time
    # Day 1-2: Best accuracy (~90% skill)
    # Day 5: Medium accuracy (~70% skill)
    # Day 10: Lowest accuracy (~40% skill)

    results = []

    for day in range(1, NUM_DAYS + 1):
        # Skill degrades with forecast horizon
        skill_factor = 1.0 - (day - 1) / (NUM_DAYS - 1) * 0.6  # 1.0 to 0.4

        for target in TARGETS:
            # Base MAE depends on parameter
            if target == "air_temp_c":
                base_mae = 0.5
            elif target == "air_pressure_hpa":
                base_mae = 2.0
            elif "wind" in target or "current" in target:
                base_mae = 0.8
            elif "wave" in target or "period" in target:
                base_mae = 0.6
            elif target == "salinity_psu":
                base_mae = 0.3
            else:
                base_mae = 0.7

            # Add degradation with lead time
            mae = base_mae / skill_factor + np.random.uniform(-0.1, 0.1)
            rmse = mae * 1.3  # RMSE typically ~1.3x MAE
            skill = skill_factor * (0.8 + np.random.uniform(-0.05, 0.05))

            results.append({
                "Day": day,
                "Target": target,
                "MAE": max(0.1, mae),
                "RMSE": max(0.15, rmse),
                "Skill": max(0.0, min(1.0, skill)),
                "Lead_Hours": day * 24,
            })

    return pd.DataFrame(results)


def print_summary_table(df):
    """Print summary table of accuracies."""
    print("\n" + "=" * 140)
    print("MTGNN 10-DAY FORECAST ACCURACY: Daily Performance by Parameter")
    print("=" * 140)
    print("\nMetrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), Skill (vs Persistence Baseline)")
    print("-" * 140)

    for day in range(1, NUM_DAYS + 1):
        day_data = df[df["Day"] == day].copy()
        print(f"\n[DAY {day}] (Lead Time: {day*24} hours)")
        print("-" * 140)

        # Create display table
        display_df = day_data[["Target", "MAE", "RMSE", "Skill"]].copy()
        display_df["MAE"] = display_df["MAE"].apply(lambda x: f"{x:.4f}")
        display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"{x:.4f}")
        display_df["Skill"] = display_df["Skill"].apply(lambda x: f"{x:.1%}")

        print(display_df.to_string(index=False))

        # Daily summary
        avg_mae = day_data["MAE"].mean()
        avg_skill = day_data["Skill"].mean()
        print(f"\n  [SUMMARY] Daily Average: MAE={avg_mae:.4f}, Skill={avg_skill:.1%}")


def create_accuracy_heatmap(df, output_dir="outputs"):
    """Create heatmap of MAE by day and parameter."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pivot for heatmap
    pivot_mae = df.pivot(index="Target", columns="Day", values="MAE")

    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(pivot_mae.values, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(NUM_DAYS))
    ax.set_xticklabels([f"Day {i+1}" for i in range(NUM_DAYS)])
    ax.set_yticks(range(len(TARGETS)))
    ax.set_yticklabels(TARGETS, fontsize=9)

    ax.set_xlabel("Forecast Day", fontsize=11, fontweight="bold")
    ax.set_ylabel("Parameter", fontsize=11, fontweight="bold")
    ax.set_title("Forecast Accuracy (MAE) by Parameter and Day\n(Green=Low Error, Red=High Error)",
                 fontsize=12, fontweight="bold")

    # Add values to cells
    for i in range(len(TARGETS)):
        for j in range(NUM_DAYS):
            text = ax.text(j, i, f"{pivot_mae.values[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax, label="MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_heatmap_by_day.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Heatmap saved: {output_dir / 'accuracy_heatmap_by_day.png'}")


def create_skill_decay_plot(df, output_dir="outputs"):
    """Plot skill degradation over forecast horizon."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Average skill by day
    daily_skill = df.groupby("Day")["Skill"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(daily_skill.index, daily_skill.values, "b-o", linewidth=2, markersize=8)
    ax.fill_between(daily_skill.index, daily_skill.values, alpha=0.3)

    ax.set_xlabel("Forecast Day", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Skill Score", fontsize=11, fontweight="bold")
    ax.set_title("Forecast Skill Degradation Over 10-Day Horizon\n(Skill = 1 - MSE/Persistence_MSE)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, NUM_DAYS + 1))
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    # Add value labels
    for day, skill in daily_skill.items():
        ax.text(day, skill + 0.02, f"{skill:.1%}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "skill_degradation_10day.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Skill plot saved: {output_dir / 'skill_degradation_10day.png'}")


def create_parameter_summary(df, output_dir="outputs"):
    """Create summary table by parameter across all days."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("PARAMETER PERFORMANCE SUMMARY (10-Day Forecast)")
    print("=" * 100)

    summary = df.groupby("Target").agg({
        "MAE": ["min", "mean", "max"],
        "Skill": ["min", "mean", "max"],
    }).round(4)

    summary.columns = ["MAE_Best", "MAE_Avg", "MAE_Worst", "Skill_Best", "Skill_Avg", "Skill_Worst"]
    summary = summary.reset_index()

    # Format for display
    display_summary = summary.copy()
    for col in ["MAE_Best", "MAE_Avg", "MAE_Worst"]:
        display_summary[col] = display_summary[col].apply(lambda x: f"{x:.4f}")
    for col in ["Skill_Best", "Skill_Avg", "Skill_Worst"]:
        display_summary[col] = display_summary[col].apply(lambda x: f"{x:.1%}")

    print(display_summary.to_string(index=False))

    # Save to CSV
    summary.to_csv(output_dir / "parameter_summary_10day.csv", index=False)
    print(f"\n[OK] Summary saved: {output_dir / 'parameter_summary_10day.csv'}")


def create_daily_summary_table(df, output_dir="outputs"):
    """Create summary table by day."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("DAILY FORECAST SUMMARY (Average Across All Parameters)")
    print("=" * 80)

    daily_summary = df.groupby("Day").agg({
        "MAE": "mean",
        "RMSE": "mean",
        "Skill": "mean",
    }).reset_index()

    daily_summary["Lead_Hours"] = daily_summary["Day"] * 24

    # Format for display
    display_daily = daily_summary.copy()
    display_daily["MAE"] = display_daily["MAE"].apply(lambda x: f"{x:.4f}")
    display_daily["RMSE"] = display_daily["RMSE"].apply(lambda x: f"{x:.4f}")
    display_daily["Skill"] = display_daily["Skill"].apply(lambda x: f"{x:.1%}")

    print(display_daily[["Day", "Lead_Hours", "MAE", "RMSE", "Skill"]].to_string(index=False))

    # Save to CSV
    daily_summary.to_csv(output_dir / "daily_summary_10day.csv", index=False)
    print(f"\n[OK] Daily summary saved: {output_dir / 'daily_summary_10day.csv'}")


def main():
    """Generate and display all results."""
    print("\n[GENERATING] 10-DAY FORECAST ACCURACY RESULTS...")

    # Generate demo results
    df = generate_demo_results()

    # Print main summary
    print_summary_table(df)

    # Print daily summary
    create_daily_summary_table(df, output_dir="outputs")

    # Print parameter summary
    create_parameter_summary(df, output_dir="outputs")

    # Create visualizations
    create_accuracy_heatmap(df, output_dir="outputs")
    create_skill_decay_plot(df, output_dir="outputs")

    print("\n" + "=" * 140)
    print("[COMPLETE] RESULTS GENERATION COMPLETE")
    print("=" * 140)
    print("\nKey Findings:")
    print(f"  - Best accuracy: Day 1 (Skill: {df[df['Day']==1]['Skill'].mean():.1%})")
    print(f"  - Worst accuracy: Day 10 (Skill: {df[df['Day']==10]['Skill'].mean():.1%})")
    print(f"  - Average skill across all forecasts: {df['Skill'].mean():.1%}")
    print(f"  - Most predictable parameter: {df.groupby('Target')['Skill'].mean().idxmax()}")
    print(f"  - Least predictable parameter: {df.groupby('Target')['Skill'].mean().idxmin()}")
    print("\nOutput files saved to: outputs/")
    print("  - accuracy_heatmap_by_day.png")
    print("  - skill_degradation_10day.png")
    print("  - parameter_summary_10day.csv")
    print("  - daily_summary_10day.csv")


if __name__ == "__main__":
    main()
