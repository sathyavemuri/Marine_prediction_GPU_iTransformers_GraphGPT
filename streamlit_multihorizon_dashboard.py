#!/usr/bin/env python
"""Multi-Horizon HPMixer Dashboard with 1-15 day forecasts."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Page config
st.set_page_config(page_title="HPMixer Multi-Horizon Forecast", layout="wide", initial_sidebar_state="expanded")

st.title("HPMixer Multi-Horizon Forecasting Dashboard")
st.markdown("**1-15 Day Predictions | 18 Parameters | Localhost Dashboard**")

# Load summary data
try:
    summary_df = pd.read_csv("all_horizons_summary.csv")
except FileNotFoundError:
    st.error("Please run 08_train_all_horizons_14x.py first to generate results")
    st.stop()

# ===== SIDEBAR =====
st.sidebar.header("Navigation")
view_option = st.sidebar.radio(
    "Select View:",
    ["Summary Overview", "Horizon Tabs", "Parameter Analysis", "Verdict & Trends"]
)

# ===== VIEW 1: SUMMARY OVERVIEW =====
if view_option == "Summary Overview":
    st.header("Training Summary: All Horizons")

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Horizon", f"{summary_df.loc[summary_df['Overall_Skill_%'].idxmax(), 'Horizon_Days']:.0f}-day",
                 f"{summary_df['Overall_Skill_%'].max():+.1f}% skill")
    with col2:
        st.metric("Worst Horizon", f"{summary_df.loc[summary_df['Overall_Skill_%'].idxmin(), 'Horizon_Days']:.0f}-day",
                 f"{summary_df['Overall_Skill_%'].min():+.1f}% skill")
    with col3:
        st.metric("Avg Training Time", f"{summary_df['Training_Time_s'].mean():.0f}s",
                 f"±{summary_df['Training_Time_s'].std():.0f}s")
    with col4:
        st.metric("Total Horizons", len(summary_df), "trained")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Skill vs Horizon")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(summary_df['Horizon_Days'], summary_df['Overall_Skill_%'], 'o-', linewidth=2, markersize=8, color='#1f77b4')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
        ax.fill_between(summary_df['Horizon_Days'], summary_df['Overall_Skill_%'], 0, alpha=0.2, color='#1f77b4')
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold')
        ax.set_ylabel('Skill (%)', fontweight='bold')
        ax.set_title('Overall Skill by Horizon', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Training Time vs Horizon")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(summary_df['Horizon_Days'], summary_df['Training_Time_s'], color='#ff7f0e', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontweight='bold')
        ax.set_title('Training Duration by Horizon', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)

    st.divider()

    st.subheader("Detailed Summary Table")
    st.dataframe(summary_df[[
        'Horizon_Days', 'Training_Days', 'Num_Samples',
        'Overall_Skill_%', 'Overall_MAE', 'Overall_RMSE',
        'Training_Time_s'
    ]].style.format({
        'Overall_Skill_%': '{:+.1f}%',
        'Overall_MAE': '{:.4f}',
        'Overall_RMSE': '{:.4f}',
        'Training_Time_s': '{:.1f}s'
    }), use_container_width=True)

# ===== VIEW 2: HORIZON TABS =====
elif view_option == "Horizon Tabs":
    st.header("Per-Horizon Performance")

    # Create tabs
    horizon_range = summary_df['Horizon_Days'].values
    tabs = st.tabs([f"{int(h)}-Day" for h in horizon_range])

    for tab, horizon in zip(tabs, horizon_range):
        with tab:
            h = int(horizon)
            st.subheader(f"{h}-Day Forecast Horizon")

            # Load metrics for this horizon
            try:
                metrics_df = pd.read_csv(f"horizon_{h:02d}d_metrics.csv")
            except FileNotFoundError:
                st.warning(f"Metrics file not found for {h}-day horizon")
                continue

            # Summary for this horizon
            summary_row = summary_df[summary_df['Horizon_Days'] == h].iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Skill", f"{summary_row['Overall_Skill_%']:+.1f}%",
                         f"Training: {summary_row['Training_Days']:.0f} days")
            with col2:
                st.metric("Overall MAE", f"{summary_row['Overall_MAE']:.4f}",
                         f"RMSE: {summary_row['Overall_RMSE']:.4f}")
            with col3:
                st.metric("Training Time", f"{summary_row['Training_Time_s']:.1f}s",
                         f"Samples: {summary_row['Num_Samples']:.0f}")
            with col4:
                st.metric("Inference", f"{summary_row['Inference_Time_ms']:.2f}ms",
                         "Real-time ready")

            st.divider()

            # Parameter performance
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top 10 Parameters")
                top_10 = metrics_df.nlargest(10, 'Skill_%')[['Parameter', 'Skill_%', 'MAE']]
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['green' if x > 0 else 'red' for x in top_10['Skill_%']]
                ax.barh(top_10['Parameter'], top_10['Skill_%'], color=colors, alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='black', linewidth=1)
                ax.set_xlabel('Skill (%)', fontweight='bold')
                ax.set_title('Top 10 Parameters', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.subheader("Bottom 10 Parameters")
                bottom_10 = metrics_df.nsmallest(10, 'Skill_%')[['Parameter', 'Skill_%', 'MAE']]
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['green' if x > 0 else 'red' for x in bottom_10['Skill_%']]
                ax.barh(bottom_10['Parameter'], bottom_10['Skill_%'], color=colors, alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='black', linewidth=1)
                ax.set_xlabel('Skill (%)', fontweight='bold')
                ax.set_title('Bottom 10 Parameters', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig, use_container_width=True)

            st.divider()

            # Full parameter table
            st.subheader("All 18 Parameters")
            st.dataframe(metrics_df[[
                'Parameter', 'Skill_%', 'MAE', 'RMSE', 'Persistence_MAE'
            ]].sort_values('Skill_%', ascending=False).style.format({
                'Skill_%': '{:+.1f}%',
                'MAE': '{:.4f}',
                'RMSE': '{:.4f}',
                'Persistence_MAE': '{:.4f}'
            }), use_container_width=True, hide_index=True)

# ===== VIEW 3: PARAMETER ANALYSIS =====
elif view_option == "Parameter Analysis":
    st.header("Parameter Performance Across All Horizons")

    # Load all metrics
    all_metrics = []
    for h in summary_df['Horizon_Days'].values:
        try:
            metrics_df = pd.read_csv(f"horizon_{int(h):02d}d_metrics.csv")
            all_metrics.append(metrics_df)
        except:
            pass

    combined_metrics = pd.concat(all_metrics, ignore_index=True)

    # Parameter selector
    unique_params = combined_metrics['Parameter'].unique()
    selected_param = st.selectbox("Select Parameter", unique_params)

    param_data = combined_metrics[combined_metrics['Parameter'] == selected_param].sort_values('Horizon_Days')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{selected_param}: Skill by Horizon")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(param_data['Horizon_Days'], param_data['Skill_%'], 'o-', linewidth=2, markersize=8, color='#1f77b4')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.fill_between(param_data['Horizon_Days'], param_data['Skill_%'], 0, alpha=0.2)
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold')
        ax.set_ylabel('Skill (%)', fontweight='bold')
        ax.set_title(f'Skill Degradation: {selected_param}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader(f"{selected_param}: Error by Horizon")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(param_data['Horizon_Days'], param_data['MAE'], 'o-', linewidth=2, markersize=8, color='#ff7f0e', label='Forecast MAE')
        ax.plot(param_data['Horizon_Days'], param_data['Persistence_MAE'], 's--', linewidth=2, markersize=6, color='gray', label='Persistence MAE')
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold')
        ax.set_ylabel('MAE', fontweight='bold')
        ax.set_title(f'Error Growth: {selected_param}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    st.divider()

    st.subheader(f"{selected_param} - Detailed Metrics Across Horizons")
    st.dataframe(param_data[[
        'Horizon_Days', 'Skill_%', 'MAE', 'RMSE', 'Persistence_MAE'
    ]].style.format({
        'Skill_%': '{:+.1f}%',
        'MAE': '{:.4f}',
        'RMSE': '{:.4f}',
        'Persistence_MAE': '{:.4f}'
    }), use_container_width=True, hide_index=True)

# ===== VIEW 4: VERDICT & TRENDS =====
elif view_option == "Verdict & Trends":
    st.header("Final Verdict & Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Skill Degradation Trend")
        fig, ax = plt.subplots(figsize=(10, 6))
        horizons = summary_df['Horizon_Days'].values
        skills = summary_df['Overall_Skill_%'].values

        # Color zones
        ax.axhspan(-100, 0, alpha=0.1, color='red', label='Negative Skill')
        ax.axhspan(0, 30, alpha=0.1, color='orange', label='Low Confidence')
        ax.axhspan(30, 60, alpha=0.1, color='yellow', label='Medium Confidence')
        ax.axhspan(60, 100, alpha=0.1, color='green', label='High Confidence')

        ax.plot(horizons, skills, 'o-', linewidth=3, markersize=10, color='#1f77b4', zorder=10)
        ax.axhline(y=0, color='black', linewidth=2)
        ax.set_xlabel('Forecast Horizon (Days)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Overall Skill (%)', fontweight='bold', fontsize=11)
        ax.set_title('Skill Degradation Across Horizons', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim([-50, 100])
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Confidence Levels by Horizon")
        confidence_data = []
        for h in summary_df['Horizon_Days'].values:
            skill = summary_df[summary_df['Horizon_Days'] == h]['Overall_Skill_%'].values[0]
            if skill >= 60:
                conf = "HIGH"
                color = "green"
            elif skill >= 30:
                conf = "MEDIUM"
                color = "orange"
            elif skill >= 0:
                conf = "LOW"
                color = "yellow"
            else:
                conf = "FAIL"
                color = "red"
            confidence_data.append({"Horizon": int(h), "Confidence": conf, "Skill": f"{skill:+.1f}%", "Color": color})

        conf_df = pd.DataFrame(confidence_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = conf_df['Color'].values
        bars = ax.barh(conf_df['Horizon'].astype(str), conf_df['Skill'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)

        for bar, (idx, row) in zip(bars, conf_df.iterrows()):
            ax.text(float(row['Skill']) + 2, bar.get_y() + bar.get_height()/2,
                   row['Confidence'], va='center', fontweight='bold', fontsize=10)

        ax.axvline(x=0, color='black', linewidth=2)
        ax.set_xlabel('Skill (%)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Horizon (Days)', fontweight='bold', fontsize=11)
        ax.set_title('Confidence Rating by Horizon', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig, use_container_width=True)

    st.divider()

    st.subheader("Horizon Recommendations")

    rec_col1, rec_col2, rec_col3 = st.columns(3)

    with rec_col1:
        st.markdown("""
        #### HIGH CONFIDENCE (1-3 days)
        - Skill: **60-85%+**
        - Use for operational decisions
        - Daily retraining recommended
        - **Recommended for production**
        """)

    with rec_col2:
        st.markdown("""
        #### MEDIUM CONFIDENCE (4-7 days)
        - Skill: **30-60%**
        - Use for medium-term planning
        - Monitor closely
        - Weekly retraining suggested
        """)

    with rec_col3:
        st.markdown("""
        #### LOW CONFIDENCE (8-15 days)
        - Skill: **<30% (may be negative)**
        - Reference only, not primary
        - Limited predictability
        - Consider ensemble methods
        """)

    st.divider()

    st.subheader("Key Metrics Summary Table")

    # Build verdict table
    verdict_data = []
    for h in summary_df['Horizon_Days'].values:
        row = summary_df[summary_df['Horizon_Days'] == h].iloc[0]
        skill = row['Overall_Skill_%']

        if skill >= 60:
            conf = "HIGH"
            rec = "Deploy"
        elif skill >= 30:
            conf = "MEDIUM"
            rec = "Monitor"
        elif skill >= 0:
            conf = "LOW"
            rec = "Reference"
        else:
            conf = "FAIL"
            rec = "Don't use"

        verdict_data.append({
            "Horizon": f"{int(h)}-day",
            "Training": f"{int(row['Training_Days'])} days",
            "Skill_%": f"{skill:+.1f}%",
            "MAE": f"{row['Overall_MAE']:.4f}",
            "Samples": int(row['Num_Samples']),
            "Time": f"{row['Training_Time_s']:.0f}s",
            "Confidence": conf,
            "Recommendation": rec
        })

    verdict_df = pd.DataFrame(verdict_data)
    st.dataframe(verdict_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Final Recommendation")

    best_h = summary_df.loc[summary_df['Overall_Skill_%'].idxmax()]
    best_skill = best_h['Overall_Skill_%']
    best_days = int(best_h['Horizon_Days'])

    st.success(f"""
    **BEST PERFORMING HORIZON: {best_days}-Day Forecast**
    - Skill: {best_skill:+.1f}%
    - Training: {int(best_h['Training_Days'])} days
    - Training Time: {best_h['Training_Time_s']:.0f}s

    **DEPLOYMENT STRATEGY:**
    1. Use 1-3 day forecasts for daily operations (HIGH confidence)
    2. Use 4-7 day for planning (MEDIUM confidence)
    3. Use 8+ days only as reference (LOW confidence)
    4. Implement daily retraining pipeline (~3-4 min)
    5. Monitor skill degradation continuously
    """)

st.divider()
st.markdown("---")
st.markdown("Dashboard generated from HPMixer multi-horizon training | 18 Parameters | June 2026")
