#!/usr/bin/env python
"""Dashboard for 110-day training -> 10-day continuous forecast."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

st.set_page_config(page_title="10-Day Continuous Forecast", layout="wide")

st.title("110-Day Training → 10-Day Continuous Forecast")
st.markdown("**Single model, one training, one forecast, day-by-day parameter analysis**")

# Load summary
try:
    summary_df = pd.read_csv("forecast_10days_summary.csv")
except FileNotFoundError:
    st.error("Run 09_train_110days_forecast_10days.py first")
    st.stop()

# ===== SIDEBAR =====
st.sidebar.header("Navigation")
view = st.sidebar.radio("Select View:", [
    "Overview", "Day-by-Day Analysis", "Parameter Trends",
    "Verdict & Recommendation"
])

# ===== VIEW 1: OVERVIEW =====
if view == "Overview":
    st.header("Overall Performance Summary")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        best_day = summary_df.loc[summary_df['Overall_Skill_%'].idxmax()]
        st.metric("Best Day", f"{int(best_day['Day'])}",
                 f"{best_day['Overall_Skill_%']:+.1f}% skill")
    with col2:
        worst_day = summary_df.loc[summary_df['Overall_Skill_%'].idxmin()]
        st.metric("Worst Day", f"{int(worst_day['Day'])}",
                 f"{worst_day['Overall_Skill_%']:+.1f}% skill")
    with col3:
        st.metric("Avg Skill", f"{summary_df['Overall_Skill_%'].mean():+.1f}%",
                 f"Range: {summary_df['Overall_Skill_%'].min():.1f}% to {summary_df['Overall_Skill_%'].max():.1f}%")
    with col4:
        st.metric("Training Data", "110 days", "Forecast: 10 days")

    st.divider()

    # Main chart: Skill degradation
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Skill Degradation Curve")
        fig, ax = plt.subplots(figsize=(10, 6))

        days = summary_df['Day'].values
        skills = summary_df['Overall_Skill_%'].values

        # Color zones
        ax.axhspan(-100, 0, alpha=0.15, color='red', label='Negative')
        ax.axhspan(0, 30, alpha=0.15, color='orange', label='Low')
        ax.axhspan(30, 60, alpha=0.15, color='yellow', label='Medium')
        ax.axhspan(60, 100, alpha=0.15, color='green', label='High')

        ax.plot(days, skills, 'o-', linewidth=3, markersize=12, color='#1f77b4', zorder=10)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.fill_between(days, skills, 0, alpha=0.2, color='#1f77b4')

        # Annotate each point
        for day, skill in zip(days, skills):
            ax.text(day, skill + 3, f'{skill:.1f}%', ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Forecast Day', fontweight='bold', fontsize=12)
        ax.set_ylabel('Skill (%)', fontweight='bold', fontsize=12)
        ax.set_title('10-Day Skill Degradation', fontweight='bold', fontsize=13)
        ax.set_xticks(days)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Error Growth (MAE & RMSE)")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(days, summary_df['Overall_MAE'], 'o-', linewidth=2.5, markersize=10,
               color='#ff7f0e', label='MAE')
        ax.plot(days, summary_df['Overall_RMSE'], 's--', linewidth=2.5, markersize=8,
               color='#2ca02c', label='RMSE')

        ax.set_xlabel('Forecast Day', fontweight='bold', fontsize=12)
        ax.set_ylabel('Error', fontweight='bold', fontsize=12)
        ax.set_title('Error Growth Over 10 Days', fontweight='bold', fontsize=13)
        ax.set_xticks(days)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    st.divider()

    st.subheader("Summary Table")
    st.dataframe(summary_df[[
        'Day', 'Overall_Skill_%', 'Overall_MAE', 'Overall_RMSE'
    ]].style.format({
        'Overall_Skill_%': '{:+.1f}%',
        'Overall_MAE': '{:.4f}',
        'Overall_RMSE': '{:.4f}'
    }), use_container_width=True, hide_index=True)

# ===== VIEW 2: DAY-BY-DAY ANALYSIS =====
elif view == "Day-by-Day Analysis":
    st.header("Per-Day Parameter Performance")

    # Day selector
    day_selected = st.selectbox("Select Day", options=summary_df['Day'].values.astype(int))

    # Load day metrics
    try:
        day_metrics = pd.read_csv(f"day_{day_selected:02d}_metrics.csv")
    except FileNotFoundError:
        st.error(f"Metrics file not found for day {day_selected}")
        st.stop()

    # Day summary
    day_summary = summary_df[summary_df['Day'] == day_selected].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"Day {day_selected} Skill", f"{day_summary['Overall_Skill_%']:+.1f}%")
    with col2:
        st.metric("MAE", f"{day_summary['Overall_MAE']:.4f}")
    with col3:
        st.metric("RMSE", f"{day_summary['Overall_RMSE']:.4f}")
    with col4:
        # Confidence level
        skill = day_summary['Overall_Skill_%']
        if skill >= 60:
            conf = "HIGH"
            color = "green"
        elif skill >= 30:
            conf = "MEDIUM"
            color = "orange"
        elif skill >= 0:
            conf = "LOW"
            color = "red"
        else:
            conf = "FAIL"
            color = "darkred"
        st.metric("Confidence", conf, f"({skill:+.1f}%)")

    st.divider()

    # Plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Day {day_selected}: Skill Distribution")
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['green' if x > 0 else 'red' for x in day_metrics['Skill_%']]
        bars = ax.barh(day_metrics['Parameter'], day_metrics['Skill_%'], color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linewidth=2)
        ax.set_xlabel('Skill (%)', fontweight='bold', fontsize=11)
        ax.set_title(f'Day {day_selected}: Parameter Skill', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, day_metrics['Skill_%'])):
            ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                   va='center', fontsize=8)

        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader(f"Day {day_selected}: Top vs Bottom")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        top_5 = day_metrics.nlargest(5, 'Skill_%')
        ax1.barh(top_5['Parameter'], top_5['Skill_%'], color='green', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Skill (%)', fontweight='bold')
        ax1.set_title('Top 5', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        bottom_5 = day_metrics.nsmallest(5, 'Skill_%')
        ax2.barh(bottom_5['Parameter'], bottom_5['Skill_%'], color='red', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Skill (%)', fontweight='bold')
        ax2.set_title('Bottom 5', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.divider()

    st.subheader(f"Day {day_selected}: All 18 Parameters")
    st.dataframe(day_metrics[['Parameter', 'Skill_%', 'MAE', 'RMSE']].sort_values('Skill_%', ascending=False).style.format({
        'Skill_%': '{:+.1f}%',
        'MAE': '{:.4f}',
        'RMSE': '{:.4f}'
    }), use_container_width=True, hide_index=True)

# ===== VIEW 3: PARAMETER TRENDS =====
elif view == "Parameter Trends":
    st.header("How Each Parameter Degrades Over 10 Days")

    # Load all day metrics
    all_params = None
    for day in range(1, 11):
        try:
            day_df = pd.read_csv(f"day_{day:02d}_metrics.csv")
            if all_params is None:
                all_params = {p: [] for p in day_df['Parameter'].unique()}

            for _, row in day_df.iterrows():
                all_params[row['Parameter']].append({
                    'Day': day,
                    'Skill': row['Skill_%'],
                    'MAE': row['MAE'],
                    'RMSE': row['RMSE']
                })
        except FileNotFoundError:
            pass

    if not all_params:
        st.error("No daily metrics found")
        st.stop()

    # Parameter selector
    param_list = sorted(all_params.keys())
    selected_param = st.selectbox("Select Parameter", param_list)

    param_data = all_params[selected_param]
    param_df = pd.DataFrame(param_data).sort_values('Day')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{selected_param}: Skill Degradation")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(param_df['Day'], param_df['Skill'], 'o-', linewidth=3, markersize=10, color='#1f77b4')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.fill_between(param_df['Day'], param_df['Skill'], 0, alpha=0.2, color='#1f77b4')

        ax.set_xlabel('Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Skill (%)', fontweight='bold', fontsize=11)
        ax.set_title(f'{selected_param}: Skill Over 10 Days', fontweight='bold', fontsize=12)
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader(f"{selected_param}: Error Growth")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(param_df['Day'], param_df['MAE'], 'o-', linewidth=2.5, markersize=10, color='#ff7f0e', label='MAE')
        ax.plot(param_df['Day'], param_df['RMSE'], 's--', linewidth=2.5, markersize=8, color='#2ca02c', label='RMSE')

        ax.set_xlabel('Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Error', fontweight='bold', fontsize=11)
        ax.set_title(f'{selected_param}: Error Over 10 Days', fontweight='bold', fontsize=12)
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    st.divider()

    st.subheader(f"Detailed Metrics: {selected_param}")
    st.dataframe(param_df[['Day', 'Skill', 'MAE', 'RMSE']].style.format({
        'Skill': '{:+.1f}%',
        'MAE': '{:.4f}',
        'RMSE': '{:.4f}'
    }), use_container_width=True, hide_index=True)

# ===== VIEW 4: VERDICT =====
elif view == "Verdict & Recommendation":
    st.header("Final Verdict: 110-Day Training -> 10-Day Forecast")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Skill Zone Classification")
        fig, ax = plt.subplots(figsize=(10, 7))

        days = summary_df['Day'].values
        skills = summary_df['Overall_Skill_%'].values

        # Zone coloring
        colors = []
        for skill in skills:
            if skill >= 60:
                colors.append('#00cc00')  # green
            elif skill >= 30:
                colors.append('#ffaa00')  # orange
            elif skill >= 0:
                colors.append('#ff6666')  # light red
            else:
                colors.append('#cc0000')  # dark red

        bars = ax.bar(days, skills, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

        # Zone bands
        ax.axhspan(-100, 0, alpha=0.1, color='red', label='Negative')
        ax.axhspan(0, 30, alpha=0.1, color='orange', label='Low (<30%)')
        ax.axhspan(30, 60, alpha=0.1, color='yellow', label='Medium (30-60%)')
        ax.axhspan(60, 100, alpha=0.1, color='green', label='High (60%+)')

        ax.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax.set_ylabel('Skill (%)', fontweight='bold', fontsize=12)
        ax.set_title('Forecast Confidence by Day', fontweight='bold', fontsize=13)
        ax.set_xticks(days)
        ax.axhline(y=0, color='black', linewidth=2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)

        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Deployment Recommendations")

        # Create verdict table
        verdict_rows = []
        for _, row in summary_df.iterrows():
            day = int(row['Day'])
            skill = row['Overall_Skill_%']

            if skill >= 60:
                conf = "HIGH"
                rec = "Deploy"
                emoji = "🟢"
            elif skill >= 30:
                conf = "MEDIUM"
                rec = "Monitor"
                emoji = "🟡"
            elif skill >= 0:
                conf = "LOW"
                rec = "Reference"
                emoji = "🟠"
            else:
                conf = "FAIL"
                rec = "Don't use"
                emoji = "🔴"

            verdict_rows.append({
                "Day": day,
                "Skill": f"{skill:+.1f}%",
                "Confidence": conf,
                "Action": rec,
                "Flag": emoji
            })

        verdict_df = pd.DataFrame(verdict_rows)
        st.dataframe(verdict_df, use_container_width=True, hide_index=True)

    st.divider()

    # Key insights
    st.subheader("Key Insights")

    best_day = summary_df.loc[summary_df['Overall_Skill_%'].idxmax()]
    worst_day = summary_df.loc[summary_df['Overall_Skill_%'].idxmin()]
    positive_days = len(summary_df[summary_df['Overall_Skill_%'] > 0])
    avg_skill = summary_df['Overall_Skill_%'].mean()

    insight_cols = st.columns(4)
    with insight_cols[0]:
        st.metric("Best Day", f"Day {int(best_day['Day'])}", f"{best_day['Overall_Skill_%']:+.1f}%")
    with insight_cols[1]:
        st.metric("Worst Day", f"Day {int(worst_day['Day'])}", f"{worst_day['Overall_Skill_%']:+.1f}%")
    with insight_cols[2]:
        st.metric("Positive Days", f"{positive_days}/10", f"({positive_days*10}%)")
    with insight_cols[3]:
        st.metric("Average Skill", f"{avg_skill:+.1f}%", "All 10 days")

    st.divider()

    st.success(f"""
    ### FINAL RECOMMENDATION

    **Forecast Quality: {"HIGH ✓" if avg_skill > 30 else "MEDIUM ⚠" if avg_skill > 0 else "LOW ✗"}**

    - **Days 1-{positive_days}:** Deploy forecasts for operational use
    - **Days {positive_days+1}-10:** Use for reference only
    - **Average Skill:** {avg_skill:+.1f}% across 10 days

    **Best Performing Day:** Day {int(best_day['Day'])} with {best_day['Overall_Skill_%']:+.1f}% skill

    **Training Configuration:**
    - Training data: 110 days
    - Forecast horizon: 10 days
    - Parameters modeled: 18 (good parameters)
    - Duplicates: 6 (reconstructed from twins)
    """)

st.divider()
st.markdown("---")
st.markdown("Dashboard: 110-Day Training → 10-Day Continuous Forecast | 18 Parameters | June 2026")
