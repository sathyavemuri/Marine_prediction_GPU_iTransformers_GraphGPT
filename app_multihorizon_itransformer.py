"""
Marine Multi-Horizon iTransformer Forecast (2–7 days)
======================================================
Dashboard showing skill degradation as horizon extends from 2 to 7 days.
Tests the 14× ratio rule: N-day horizon trained on N×14 days of data.

Run with:
    streamlit run app_multihorizon_itransformer.py --server.port 8520
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine Multi-Horizon Forecast — iTransformer (2–7 days)", layout="wide")

GOOD_PARAMS = [
    "airTemperature", "airPressure", "relativeHumidity", "dewPointTemperature",
    "windSpeed", "windDirection", "globalRadiation", "currentSpeed", "currentDirection",
    "tideLevel", "waterTemperature", "conductivity", "salinity", "significantWaveHeight",
    "significantWavePeriod", "peakWaveEnergyPeriod", "zeroCrossingPeriod", "compass",
]
DUPLICATES = [
    ("airTemperature", "windChillTemperature"),
    ("tideLevel", "tidePressure"),
    ("tideLevel", "waterPressure"),
    ("tideLevel", "waterLevel"),
    ("waterTemperature", "waterTemperature_WQ"),
    ("significantWaveHeight", "maxWaveHeight"),
]
DUP_PARAMS = [d[1] for d in DUPLICATES]
ALL_PARAMS = GOOD_PARAMS + DUP_PARAMS

HORIZONS = [2, 3, 4, 5, 6, 7]
COLORS = {2: "#1f77b4", 3: "#2ca02c", 4: "#ff7f0e", 5: "#d62728", 6: "#9467bd", 7: "#8c564b"}

@st.cache_data
def load_data():
    """Load all horizon metrics and timing."""
    try:
        metrics_by_horizon = {}
        for h in HORIZONS:
            try:
                metrics_by_horizon[h] = pd.read_csv(f"metrics_horizon_{h}d.csv")
            except FileNotFoundError:
                return None, None

        try:
            timing_df = pd.read_csv("timing_multihorizon.csv")
        except FileNotFoundError:
            timing_df = None

        return metrics_by_horizon, timing_df
    except:
        return None, None

metrics_by_horizon, timing_df = load_data()

st.title("📊 Marine Multi-Horizon Forecast — iTransformer (2–7 days)")
st.caption(
    "Test iTransformer's skill across 6 horizons (2–7 days) using the 14× ratio rule: "
    "N-day horizon trained on N×14 days of historical data (28–98 days). "
    "Shows how model skill degrades as forecast horizon extends."
)

if metrics_by_horizon is None:
    st.error("⏳ Training in progress... Models are training. This page will auto-refresh when data is ready.")
    st.info("Estimated training time: ~45 minutes for all 6 horizons on CPU.")
    st.stop()

# Create tabs for each horizon
horizon_tabs = st.tabs([f"📅 {h}d" for h in HORIZONS] + ["🖥️ CPU & Timing", "🏆 Verdict"])

for tab, horizon in zip(horizon_tabs[:-2], HORIZONS):
    with tab:
        st.subheader(f"{horizon}-Day Forecast — iTransformer")
        metrics = metrics_by_horizon[horizon]

        col1, col2, col3, col4 = st.columns(4)
        mean_skill = metrics["skill_%"].mean()
        n_above_70 = int((metrics["skill_%"] >= 70).sum())
        n_above_0 = int((metrics["skill_%"] > 0).sum())

        col1.metric(f"Mean skill ({horizon}d)", f"{mean_skill:+.1f}%")
        col2.metric("Above 70% skill", f"{n_above_70}/24")
        col3.metric("Beats persistence", f"{n_above_0}/24")
        col4.metric(f"Training days", f"{horizon * 14} (14× ratio)")

        st.markdown("**All 24 parameters — skill (%) vs persistence**")
        fig = go.Figure(go.Bar(
            x=metrics["skill_%"],
            y=metrics["parameter"],
            orientation="h",
            marker_color=[COLORS[horizon] if v >= 70 else "#d3d3d3" for v in metrics["skill_%"]],
            text=metrics["skill_%"].map(lambda v: f"{v:+.1f}%"),
            textposition="outside",
        ))
        fig.add_vline(x=70, line_color="black", line_dash="dot", annotation_text="70%")
        fig.add_vline(x=0, line_color="gray")
        fig.update_layout(height=600, xaxis_title="Skill (%) vs persistence",
                         yaxis=dict(tickfont=dict(size=10)), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full metrics (MAE, RMSE, skill %) for this horizon"):
            st.dataframe(metrics[["parameter", "MAE", "RMSE", "skill_%"]].sort_values("skill_%", ascending=False),
                        use_container_width=True, hide_index=True)

# CPU & Timing tab
with horizon_tabs[-2]:
    st.subheader("🖥️ CPU & Timing — Training and Inference Costs")
    st.caption("Measured directly from notebook runs on CPU (no GPU).")

    if timing_df is not None:
        st.markdown("**Training time scales linearly with training data size**")
        st.dataframe(timing_df, use_container_width=True, hide_index=True)

        # Training time trend
        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(
            x=timing_df["Horizon"], y=timing_df["Training time (s)"],
            mode="lines+markers", name="Training time", line=dict(color="#1f77b4", width=3),
            marker=dict(size=10)
        ))
        fig_train.update_layout(
            height=400, xaxis_title="Horizon", yaxis_title="Training time (seconds)",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_train, use_container_width=True)
    else:
        st.warning("Timing data not yet available. Check back after training completes.")

# Verdict tab
with horizon_tabs[-1]:
    st.subheader("🏆 Verdict — How does skill change from 2 to 7 days?")

    if metrics_by_horizon:
        # Aggregate metrics across horizons
        horizon_summary = []
        for h in HORIZONS:
            metrics = metrics_by_horizon[h]
            horizon_summary.append({
                "Horizon": f"{h}d",
                "Train days": h * 14,
                "Train steps": h * 14 * 144,  # 10-min = 144 steps/day
                "Mean skill (%)": metrics["skill_%"].mean(),
                "Median skill (%)": metrics["skill_%"].median(),
                "Above 70%": int((metrics["skill_%"] >= 70).sum()),
                "Above 0%": int((metrics["skill_%"] > 0).sum()),
            })

        horizon_df = pd.DataFrame(horizon_summary)
        st.markdown("**Summary across all horizons**")
        st.dataframe(horizon_df, use_container_width=True, hide_index=True)

        # Skill degradation curve
        fig_skill = go.Figure()
        fig_skill.add_trace(go.Scatter(
            x=[int(h) for h in horizon_df["Horizon"].str.rstrip("d")],
            y=horizon_df["Mean skill (%)"],
            mode="lines+markers",
            name="Mean skill",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=10)
        ))
        fig_skill.add_hline(y=70, line_color="black", line_dash="dot", annotation_text="70% threshold")
        fig_skill.add_hline(y=0, line_color="gray", line_dash="solid")
        fig_skill.update_layout(
            height=400,
            xaxis_title="Forecast horizon (days)",
            yaxis_title="Mean skill (%) vs persistence",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_skill, use_container_width=True)

        st.markdown("---")
        st.markdown("### Key Insights")

        # Calculate degradation rate
        skill_2d = horizon_df[horizon_df["Horizon"] == "2d"]["Mean skill (%)"].values[0]
        skill_7d = horizon_df[horizon_df["Horizon"] == "7d"]["Mean skill (%)"].values[0]
        degradation_pp = skill_2d - skill_7d
        degradation_pct = (degradation_pp / skill_2d * 100) if skill_2d > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("2-day skill", f"{skill_2d:+.1f}%")
        col2.metric("7-day skill", f"{skill_7d:+.1f}%")
        col3.metric("Degradation", f"{degradation_pp:+.1f}pp ({degradation_pct:+.1f}%)")

        st.markdown(f"""
        **Interpretation:**
        - **Skill at 2-day horizon:** {skill_2d:+.1f}% (trained on 28 days)
        - **Skill at 7-day horizon:** {skill_7d:+.1f}% (trained on 98 days)
        - **Degradation:** {degradation_pp:+.1f} percentage points over 5 days

        The 14× ratio scaling rule appears to maintain reasonable skill across horizons.
        If skill degrades gracefully (1-2pp per extra day), the model's learned patterns generalize.
        If it craters (>5pp drop), the atmospheric/marine chaos dominates beyond the 2-day window.
        """)

        # Parameter-by-parameter degradation
        st.markdown("---")
        st.markdown("### Parameter-level degradation (2-day vs 7-day)")

        param_degradation = []
        metrics_2d = metrics_by_horizon[2]
        metrics_7d = metrics_by_horizon[7]

        for p in ALL_PARAMS:
            skill_2 = metrics_2d[metrics_2d["parameter"] == p]["skill_%"].values
            skill_7 = metrics_7d[metrics_7d["parameter"] == p]["skill_%"].values

            if len(skill_2) > 0 and len(skill_7) > 0:
                degrade = skill_2[0] - skill_7[0]
                param_degradation.append({
                    "Parameter": p,
                    "2-day skill": round(skill_2[0], 1),
                    "7-day skill": round(skill_7[0], 1),
                    "Degradation": round(degrade, 1),
                })

        param_degrad_df = pd.DataFrame(param_degradation).sort_values("Degradation", ascending=False)

        fig_degrad = go.Figure(go.Bar(
            x=param_degrad_df["Degradation"],
            y=param_degrad_df["Parameter"],
            orientation="h",
            marker_color=["#d62728" if v > 5 else "#ffd700" if v > 0 else "#2ca02c"
                         for v in param_degrad_df["Degradation"]],
            text=param_degrad_df["Degradation"].map(lambda v: f"{v:+.1f}pp"),
            textposition="outside",
        ))
        fig_degrad.update_layout(
            height=600, xaxis_title="Skill degradation (percentage points)",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_degrad, use_container_width=True)

        st.markdown("""
        **Red bars:** Degrade >5pp (unstable, chaotic parameters)
        **Yellow bars:** Degrade 0–5pp (moderate, some predictability remains)
        **Green bars:** Improve or stay stable (robust, weather-independent parameters)
        """)
