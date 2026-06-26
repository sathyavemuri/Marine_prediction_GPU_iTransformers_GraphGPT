#!/usr/bin/env python
"""Multi-Horizon HPMixer Dashboard with Rich Plots (like iTransformer tabs)."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Page config
st.set_page_config(page_title="HPMixer Multi-Horizon Forecast", layout="wide", initial_sidebar_state="expanded")

st.title("HPMixer Multi-Horizon Forecasting Dashboard")
st.markdown("**1-15 Day Predictions | 18 Parameters | Rich Visualizations**")

# Load summary data
try:
    summary_df = pd.read_csv("all_horizons_summary.csv")
except FileNotFoundError:
    st.error("Please run 08_train_all_horizons_14x.py first")
    st.stop()

# ===== SIDEBAR =====
st.sidebar.header("Navigation")
view_option = st.sidebar.radio(
    "Select View:",
    ["Summary Overview", "Horizon Tabs (with Plots)", "Parameter Trends", "Verdict Dashboard"]
)

# ===== VIEW 1: SUMMARY OVERVIEW =====
if view_option == "Summary Overview":
    st.header("Training Summary: All Horizons")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        best_h = summary_df.loc[summary_df['Overall_Skill_%'].idxmax()]
        st.metric("Best Horizon", f"{int(best_h['Horizon_Days'])}-day",
                 f"{best_h['Overall_Skill_%']:+.1f}% skill")
    with col2:
        worst_h = summary_df.loc[summary_df['Overall_Skill_%'].idxmin()]
        st.metric("Worst Horizon", f"{int(worst_h['Horizon_Days'])}-day",
                 f"{worst_h['Overall_Skill_%']:+.1f}% skill")
    with col3:
        st.metric("Avg Training", f"{summary_df['Training_Time_s'].mean():.0f}s",
                 f"Per horizon")
    with col4:
        st.metric("Data Points", f"{len(summary_df)}", "horizons trained")

    st.divider()

    # Main summary charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Skill Degradation Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        horizons = summary_df['Horizon_Days'].values
        skills = summary_df['Overall_Skill_%'].values

        # Color zones
        ax.axhspan(-100, 0, alpha=0.15, color='red', label='Negative')
        ax.axhspan(0, 30, alpha=0.15, color='orange', label='Low (<30%)')
        ax.axhspan(30, 60, alpha=0.15, color='yellow', label='Medium (30-60%)')
        ax.axhspan(60, 100, alpha=0.15, color='green', label='High (60%+)')

        ax.plot(horizons, skills, 'o-', linewidth=3, markersize=10, color='#1f77b4', zorder=10)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.fill_between(horizons, skills, 0, alpha=0.2, color='#1f77b4')
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Skill (%)', fontweight='bold', fontsize=11)
        ax.set_title('Overall Skill Degradation', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Training Time vs Horizon")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(summary_df['Horizon_Days'], summary_df['Training_Time_s'],
                     color=['green' if x < 300 else 'orange' if x < 600 else 'red'
                            for x in summary_df['Training_Time_s']],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=11)
        ax.set_title('Training Duration by Horizon', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, time in zip(bars, summary_df['Training_Time_s']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.0f}s', ha='center', va='bottom', fontsize=9)
        st.pyplot(fig, use_container_width=True)

    st.divider()

    # Summary table
    st.subheader("Complete Summary Table")
    st.dataframe(summary_df[[
        'Horizon_Days', 'Training_Days', 'Num_Samples',
        'Overall_Skill_%', 'Overall_MAE', 'Overall_RMSE',
        'Training_Time_s', 'Inference_Time_ms'
    ]].style.format({
        'Overall_Skill_%': '{:+.1f}%',
        'Overall_MAE': '{:.4f}',
        'Overall_RMSE': '{:.4f}',
        'Training_Time_s': '{:.1f}s',
        'Inference_Time_ms': '{:.2f}ms'
    }), use_container_width=True, hide_index=True)

# ===== VIEW 2: HORIZON TABS WITH PLOTS =====
elif view_option == "Horizon Tabs (with Plots)":
    st.header("Per-Horizon Detailed Analysis with Plots")

    horizon_range = sorted(summary_df['Horizon_Days'].unique())
    tabs = st.tabs([f"{int(h)}-Day" for h in horizon_range])

    for tab, horizon in zip(tabs, horizon_range):
        with tab:
            h = int(horizon)
            st.subheader(f"{h}-Day Forecast Performance")

            # Load metrics
            try:
                metrics_df = pd.read_csv(f"horizon_{h:02d}d_metrics.csv")
            except FileNotFoundError:
                st.warning(f"Metrics file not found for {h}-day horizon")
                continue

            summary_row = summary_df[summary_df['Horizon_Days'] == h].iloc[0]

            # KPI Row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                skill = summary_row['Overall_Skill_%']
                color = "green" if skill > 50 else "orange" if skill > 0 else "red"
                st.metric("Overall Skill", f"{skill:+.1f}%",
                         delta=f"{int(summary_row['Training_Days'])} days training",
                         delta_color="off")
            with col2:
                st.metric("MAE", f"{summary_row['Overall_MAE']:.4f}",
                         f"RMSE: {summary_row['Overall_RMSE']:.4f}")
            with col3:
                st.metric("Samples", f"{int(summary_row['Num_Samples'])}",
                         f"Training: {int(summary_row['Training_Days'])} days")
            with col4:
                st.metric("Training", f"{summary_row['Training_Time_s']:.0f}s",
                         f"Inference: {summary_row['Inference_Time_ms']:.2f}ms")

            st.divider()

            # PLOT 1: Skill Distribution (like iTransformer tabs)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Skill Distribution Across Parameters")
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['green' if x > 0 else 'red' for x in metrics_df['Skill_%']]
                bars = ax.barh(metrics_df['Parameter'], metrics_df['Skill_%'], color=colors, alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='black', linewidth=2)
                ax.set_xlabel('Skill (%)', fontweight='bold', fontsize=11)
                ax.set_title(f'{h}-Day Horizon: Parameter Skill', fontweight='bold', fontsize=12)
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.subheader("Top 10 vs Bottom 5 Parameters")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # Top 10
                top_10 = metrics_df.nlargest(10, 'Skill_%')
                ax1.barh(top_10['Parameter'], top_10['Skill_%'], color='green', alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Skill (%)', fontweight='bold', fontsize=10)
                ax1.set_title('Top 10 Parameters', fontweight='bold', fontsize=11)
                ax1.grid(True, alpha=0.3, axis='x')

                # Bottom 5
                bottom_5 = metrics_df.nsmallest(5, 'Skill_%')
                ax2.barh(bottom_5['Parameter'], bottom_5['Skill_%'], color='red', alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Skill (%)', fontweight='bold', fontsize=10)
                ax2.set_title('Bottom 5 Parameters', fontweight='bold', fontsize=11)
                ax2.grid(True, alpha=0.3, axis='x')

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            st.divider()

            # PLOT 2: MAE vs Persistence (Error Comparison)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("MAE Comparison: Forecast vs Persistence")
                fig, ax = plt.subplots(figsize=(10, 8))
                metrics_sorted = metrics_df.sort_values('MAE')
                x = np.arange(len(metrics_sorted))
                width = 0.35

                ax.barh(x - width/2, metrics_sorted['MAE'], width, label='Forecast MAE',
                       alpha=0.8, color='#1f77b4', edgecolor='black')
                ax.barh(x + width/2, metrics_sorted['Persistence_MAE'], width, label='Persistence MAE',
                       alpha=0.8, color='gray', edgecolor='black')

                ax.set_yticks(x)
                ax.set_yticklabels(metrics_sorted['Parameter'], fontsize=9)
                ax.set_xlabel('MAE', fontweight='bold', fontsize=11)
                ax.set_title('Error: Forecast vs Baseline', fontweight='bold', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.subheader("Skill vs MAE Scatter")
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(metrics_df['MAE'], metrics_df['Skill_%'],
                                   s=200, alpha=0.6, c=metrics_df['Skill_%'],
                                   cmap='RdYlGn', edgecolor='black', linewidth=1.5)
                ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
                ax.set_xlabel('MAE', fontweight='bold', fontsize=11)
                ax.set_ylabel('Skill (%)', fontweight='bold', fontsize=11)
                ax.set_title('Parameter Performance: Skill vs Error', fontweight='bold', fontsize=12)
                ax.grid(True, alpha=0.3)
                cbar = plt.colorbar(scatter, ax=ax, label='Skill (%)')
                st.pyplot(fig, use_container_width=True)

            st.divider()

            # PLOT 3: Summary Statistics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Skill Distribution Stats")
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))

                # Histogram
                axes[0, 0].hist(metrics_df['Skill_%'], bins=8, color='steelblue', alpha=0.7, edgecolor='black')
                axes[0, 0].axvline(metrics_df['Skill_%'].mean(), color='red', linestyle='--', linewidth=2,
                                  label=f"Mean: {metrics_df['Skill_%'].mean():.1f}%")
                axes[0, 0].set_xlabel('Skill (%)', fontweight='bold', fontsize=9)
                axes[0, 0].set_ylabel('Count', fontweight='bold', fontsize=9)
                axes[0, 0].set_title('Skill Distribution', fontweight='bold', fontsize=10)
                axes[0, 0].legend(fontsize=8)
                axes[0, 0].grid(True, alpha=0.3)

                # MAE histogram
                axes[0, 1].hist(metrics_df['MAE'], bins=8, color='coral', alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(metrics_df['MAE'].mean(), color='red', linestyle='--', linewidth=2,
                                  label=f"Mean: {metrics_df['MAE'].mean():.3f}")
                axes[0, 1].set_xlabel('MAE', fontweight='bold', fontsize=9)
                axes[0, 1].set_ylabel('Count', fontweight='bold', fontsize=9)
                axes[0, 1].set_title('MAE Distribution', fontweight='bold', fontsize=10)
                axes[0, 1].legend(fontsize=8)
                axes[0, 1].grid(True, alpha=0.3)

                # RMSE histogram
                axes[1, 0].hist(metrics_df['RMSE'], bins=8, color='lightgreen', alpha=0.7, edgecolor='black')
                axes[1, 0].axvline(metrics_df['RMSE'].mean(), color='red', linestyle='--', linewidth=2,
                                  label=f"Mean: {metrics_df['RMSE'].mean():.3f}")
                axes[1, 0].set_xlabel('RMSE', fontweight='bold', fontsize=9)
                axes[1, 0].set_ylabel('Count', fontweight='bold', fontsize=9)
                axes[1, 0].set_title('RMSE Distribution', fontweight='bold', fontsize=10)
                axes[1, 0].legend(fontsize=8)
                axes[1, 0].grid(True, alpha=0.3)

                # Summary text
                axes[1, 1].axis('off')
                summary_text = f"""
                STATISTICS FOR {h}-DAY HORIZON

                Skill:
                  Mean: {metrics_df['Skill_%'].mean():.2f}%
                  Median: {metrics_df['Skill_%'].median():.2f}%
                  Std: {metrics_df['Skill_%'].std():.2f}%
                  Range: {metrics_df['Skill_%'].min():.2f}% to {metrics_df['Skill_%'].max():.2f}%

                MAE:
                  Mean: {metrics_df['MAE'].mean():.4f}
                  Median: {metrics_df['MAE'].median():.4f}

                RMSE:
                  Mean: {metrics_df['RMSE'].mean():.4f}
                  Median: {metrics_df['RMSE'].median():.4f}

                Parameters above zero: {len(metrics_df[metrics_df['Skill_%'] > 0])}/18
                """
                axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                              fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.subheader("All 18 Parameters Table")
                display_cols = ['Parameter', 'Skill_%', 'MAE', 'RMSE', 'Persistence_MAE']
                st.dataframe(metrics_df[display_cols].sort_values('Skill_%', ascending=False).style.format({
                    'Skill_%': '{:+.1f}%',
                    'MAE': '{:.4f}',
                    'RMSE': '{:.4f}',
                    'Persistence_MAE': '{:.4f}'
                }), use_container_width=True, hide_index=True)

# ===== VIEW 3: PARAMETER TRENDS =====
elif view_option == "Parameter Trends":
    st.header("Parameter Performance Across All Horizons")

    # Load all metrics
    all_metrics = []
    for h in sorted(summary_df['Horizon_Days'].unique()):
        try:
            metrics_df = pd.read_csv(f"horizon_{int(h):02d}d_metrics.csv")
            all_metrics.append(metrics_df)
        except:
            pass

    if not all_metrics:
        st.error("No metrics files found")
        st.stop()

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    unique_params = sorted(combined_metrics['Parameter'].unique())

    # Parameter selector
    selected_param = st.selectbox("Select Parameter to Analyze", unique_params)
    param_data = combined_metrics[combined_metrics['Parameter'] == selected_param].sort_values('Horizon_Days')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{selected_param}: Skill Degradation")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(param_data['Horizon_Days'], param_data['Skill_%'], 'o-', linewidth=3, markersize=10, color='#1f77b4')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
        ax.fill_between(param_data['Horizon_Days'], param_data['Skill_%'], 0, alpha=0.2, color='#1f77b4')
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Skill (%)', fontweight='bold', fontsize=11)
        ax.set_title(f'{selected_param}: Skill by Horizon', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader(f"{selected_param}: Error Growth")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(param_data['Horizon_Days'], param_data['MAE'], 'o-', linewidth=3, markersize=10, color='#ff7f0e', label='Forecast MAE')
        ax.plot(param_data['Horizon_Days'], param_data['Persistence_MAE'], 's--', linewidth=2, markersize=8, color='gray', label='Persistence MAE')
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold', fontsize=11)
        ax.set_ylabel('MAE', fontweight='bold', fontsize=11)
        ax.set_title(f'{selected_param}: Error Growth', fontweight='bold', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader(f"Detailed Metrics: {selected_param}")
    st.dataframe(param_data[[
        'Horizon_Days', 'Skill_%', 'MAE', 'RMSE', 'Persistence_MAE'
    ]].sort_values('Horizon_Days').style.format({
        'Skill_%': '{:+.1f}%',
        'MAE': '{:.4f}',
        'RMSE': '{:.4f}',
        'Persistence_MAE': '{:.4f}'
    }), use_container_width=True, hide_index=True)

# ===== VIEW 4: VERDICT DASHBOARD =====
elif view_option == "Verdict Dashboard":
    st.header("Final Verdict & Recommendations")

    # Confidence zone chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Skill Zones & Confidence Levels")
        fig, ax = plt.subplots(figsize=(10, 7))

        horizons = summary_df['Horizon_Days'].values
        skills = summary_df['Overall_Skill_%'].values

        # Color zones
        ax.axhspan(-100, 0, alpha=0.2, color='red', label='FAIL (Negative)')
        ax.axhspan(0, 30, alpha=0.2, color='orange', label='LOW (<30%)')
        ax.axhspan(30, 60, alpha=0.2, color='yellow', label='MEDIUM (30-60%)')
        ax.axhspan(60, 100, alpha=0.2, color='green', label='HIGH (60%+)')

        colors_plot = ['red' if x < 0 else 'orange' if x < 30 else 'yellow' if x < 60 else 'green' for x in skills]
        ax.bar(horizons, skills, color=colors_plot, alpha=0.6, edgecolor='black', linewidth=2)
        ax.plot(horizons, skills, 'ko-', linewidth=2, markersize=8, label='Skill Trend')
        ax.axhline(y=0, color='black', linewidth=2)

        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Skill (%)', fontweight='bold', fontsize=11)
        ax.set_title('Skill Zones by Horizon', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Recommendation Matrix")

        verdict_data = []
        for h in summary_df['Horizon_Days'].values:
            row = summary_df[summary_df['Horizon_Days'] == h].iloc[0]
            skill = row['Overall_Skill_%']

            if skill >= 60:
                conf = "HIGH"
                rec = "DEPLOY"
                color_code = "🟢"
            elif skill >= 30:
                conf = "MEDIUM"
                rec = "MONITOR"
                color_code = "🟡"
            elif skill >= 0:
                conf = "LOW"
                rec = "REFERENCE"
                color_code = "🟠"
            else:
                conf = "FAIL"
                rec = "DON'T USE"
                color_code = "🔴"

            verdict_data.append({
                "Horizon": f"{int(h)}d",
                "Skill": f"{skill:+.1f}%",
                "Confidence": conf,
                "Rec": rec,
                "Flag": color_code
            })

        verdict_df = pd.DataFrame(verdict_data)
        st.dataframe(verdict_df, use_container_width=True, hide_index=True)

    st.divider()

    # Final recommendations
    st.subheader("Deployment Recommendations")

    col1, col2, col3 = st.columns(3)

    high_conf = len(summary_df[summary_df['Overall_Skill_%'] >= 60])
    med_conf = len(summary_df[(summary_df['Overall_Skill_%'] >= 30) & (summary_df['Overall_Skill_%'] < 60)])
    low_conf = len(summary_df[summary_df['Overall_Skill_%'] < 30])

    with col1:
        st.metric("High Confidence", f"{high_conf} horizons", "Deploy to production")
    with col2:
        st.metric("Medium Confidence", f"{med_conf} horizons", "Monitor closely")
    with col3:
        st.metric("Low Confidence", f"{low_conf} horizons", "Reference only")

    st.divider()

    best_h = summary_df.loc[summary_df['Overall_Skill_%'].idxmax()]
    st.success(f"""
    ### FINAL RECOMMENDATION

    **Best Performing Horizon: {int(best_h['Horizon_Days'])}-Day Forecast**
    - Overall Skill: **{best_h['Overall_Skill_%']:+.1f}%**
    - Training Window: **{int(best_h['Training_Days'])} days**
    - Training Time: **{best_h['Training_Time_s']:.0f} seconds**
    - Number of Samples: **{int(best_h['Num_Samples'])}**

    **DEPLOYMENT STRATEGY:**
    1. **Short-term (1-3 days):** Deploy high-confidence forecasts for daily operations
    2. **Medium-term (4-7 days):** Use medium-confidence forecasts for planning
    3. **Long-term (8-15 days):** Use low-confidence forecasts for reference only
    4. **Daily Retraining:** Implement rolling 120-day window with daily updates
    5. **Monitoring:** Track skill degradation, alert if drops >10 points
    """)

st.divider()
st.markdown("---")
st.markdown("Dashboard: HPMixer Multi-Horizon | 18 Parameters | June 2026 | Localhost:8525")
