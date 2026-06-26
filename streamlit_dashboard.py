#!/usr/bin/env python
"""Interactive Streamlit dashboard for iTransformer results."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Page config
st.set_page_config(page_title="iTransformer 2-Day Forecast", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("🌊 Marine Sensor iTransformer Forecasting")
st.markdown("**2-Day Ahead Prediction with 28-Day Training Window**")

# Load data
metrics_df = pd.read_csv("itransformer_2day_metrics.csv")
results_df = pd.read_csv("itransformer_2day_results.csv")

# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")
    view_option = st.radio(
        "Select View:",
        ["Summary", "All Plots", "Skill Analysis", "Parameter Details", "Raw Data"]
    )

# ===== SUMMARY VIEW =====
if view_option == "Summary":
    st.header("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Skill", f"{results_df['Skill_%'].values[0]:.1f}%", delta=None)
    with col2:
        st.metric("MAE", f"{results_df['MAE'].values[0]:.2f}")
    with col3:
        st.metric("RMSE", f"{results_df['RMSE'].values[0]:.2f}")
    with col4:
        st.metric("Training Time", f"{results_df['Training_Time_s'].values[0]:.0f}s")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 5 Performers")
        top_5 = metrics_df.nlargest(5, "Skill_%")[["Parameter", "Skill_%", "MAE"]]
        st.dataframe(top_5, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Bottom 5 Performers")
        bottom_5 = metrics_df.nsmallest(5, "Skill_%")[["Parameter", "Skill_%", "MAE"]]
        st.dataframe(bottom_5, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Key Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        best_param = metrics_df.loc[metrics_df["Skill_%"].idxmax()]
        st.write(f"**Best Parameter:** {best_param['Parameter']}")
        st.write(f"Skill: {best_param['Skill_%']:+.1f}%")

    with col2:
        worst_param = metrics_df.loc[metrics_df["Skill_%"].idxmin()]
        st.write(f"**Worst Parameter:** {worst_param['Parameter']}")
        st.write(f"Skill: {worst_param['Skill_%']:+.1f}%")

    with col3:
        positive = len(metrics_df[metrics_df["Skill_%"] > 0])
        st.write(f"**Parameters with +ve Skill:** {positive}/{len(metrics_df)}")

# ===== ALL PLOTS VIEW =====
elif view_option == "All Plots":
    st.header("Complete Visualization Suite")

    plot_files = [
        "plot_01_skill_distribution.png",
        "plot_02_top_10_parameters.png",
        "plot_03_bottom_10_parameters.png",
        "plot_04_mae_comparison.png",
        "plot_05_skill_vs_mae.png",
        "plot_06_summary_statistics.png",
    ]

    for plot_file in plot_files:
        if os.path.exists(plot_file):
            image = Image.open(plot_file)
            st.image(image, use_column_width=True)
            st.divider()

# ===== SKILL ANALYSIS VIEW =====
elif view_option == "Skill Analysis":
    st.header("Skill Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Skill Distribution")
        if os.path.exists("plot_01_skill_distribution.png"):
            image = Image.open("plot_01_skill_distribution.png")
            st.image(image, use_column_width=True)

    with col2:
        st.subheader("Statistics")
        st.write(f"**Mean Skill:** {metrics_df['Skill_%'].mean():.2f}%")
        st.write(f"**Median Skill:** {metrics_df['Skill_%'].median():.2f}%")
        st.write(f"**Std Dev:** {metrics_df['Skill_%'].std():.2f}%")
        st.write(f"**Min:** {metrics_df['Skill_%'].min():.2f}%")
        st.write(f"**Max:** {metrics_df['Skill_%'].max():.2f}%")
        st.write(f"**+ve Skill:** {len(metrics_df[metrics_df['Skill_%'] > 0])}/{len(metrics_df)}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists("plot_02_top_10_parameters.png"):
            image = Image.open("plot_02_top_10_parameters.png")
            st.image(image, use_column_width=True)

    with col2:
        if os.path.exists("plot_03_bottom_10_parameters.png"):
            image = Image.open("plot_03_bottom_10_parameters.png")
            st.image(image, use_column_width=True)

# ===== PARAMETER DETAILS VIEW =====
elif view_option == "Parameter Details":
    st.header("Individual Parameter Performance")

    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        sort_by = st.selectbox("Sort by:", ["Skill_%", "MAE", "RMSE"])
    with col2:
        ascending = st.checkbox("Ascending", value=False)
    with col3:
        limit = st.number_input("Show top N:", min_value=1, max_value=len(metrics_df), value=len(metrics_df))

    # Display table
    display_df = metrics_df.sort_values(sort_by, ascending=ascending).head(limit)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # Search functionality
    st.subheader("Search Parameter")
    search_term = st.text_input("Enter parameter name:")
    if search_term:
        results = metrics_df[metrics_df["Parameter"].str.contains(search_term, case=False)]
        if len(results) > 0:
            st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.warning("No parameters found matching search term.")

# ===== RAW DATA VIEW =====
elif view_option == "Raw Data":
    st.header("Raw Results Data")

    st.subheader("Summary Results")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.subheader("Per-Parameter Metrics")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Download options
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics CSV",
            data=csv,
            file_name="itransformer_2day_metrics.csv",
            mime="text/csv"
        )

    with col2:
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="itransformer_2day_results.csv",
            mime="text/csv"
        )

# Footer
st.divider()
st.markdown("""
---
**Dataset:** 120-day marine sensor data (18 parameters)
**Training:** 28 days (May 24 - Jun 20) | **Test:** 2 days (Jun 21-22)
**Model:** iTransformer with 18 parameters
**Generated:** 2026-06-24
""")
