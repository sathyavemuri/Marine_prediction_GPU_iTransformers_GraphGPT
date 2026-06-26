"""
Marine 48h Forecast — Hybrid v2 (iTransformer + Tweedie/Huber XGBoost) Dashboard
==================================================================================
Streamlit viewer for Marine_Forecast_RealEMS_Hybrid_iTransformer_XGBoost.ipynb. Leads
with the honest three-way comparison: pure iTransformer vs. the DeepAR-hybrid vs. this
distribution-correct XGBoost hybrid (Tweedie for zero-inflated precipitation,
Pseudo-Huber for ceiling-saturated visibility) on the 6 historically hard parameters.

Run with:
    streamlit run app_hybrid_v2.py --server.port 8505
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine Forecast — Hybrid v2 (Tweedie/Huber)", layout="wide")

HISTORY_PATH = "ems_10min_resampled.csv"
FORECAST_PATH = "forecast_vs_actual_hybrid_v2.csv"
METRICS_PATH = "metrics_hybrid_v2.csv"
DUP_RECON_PATH = "duplicate_reconstruction_hybrid_v2.csv"
DUP_FVA_PATH = "duplicate_forecast_vs_actual_hybrid_v2.csv"
ATTENTION_PATH = "attention_weights_hybrid_v2.csv"

HORIZON_STEPS = 288
LOOKBACK_STEPS = 288

PRECIP_PARAMS = ["precipitationIntensity", "precipitationDifference"]
VISIBILITY_PARAMS = ["twentyFourHourAvgVisibility", "tenMinuteAvgVisibility",
                      "oneMinuteAvgVisibility", "oneHourAvgVisibility"]
HARD_PARAMS = PRECIP_PARAMS + VISIBILITY_PARAMS
CIRCULAR_PARAMS = {"windDirection", "currentDirection", "compass"}

REPORT_PARAMS = [
    "airTemperature", "airPressure", "relativeHumidity", "dewPointTemperature",
    "windSpeed", "windDirection", "globalRadiation", "precipitationDifference",
    "precipitationIntensity", "currentSpeed", "currentDirection", "tideLevel",
    "waterTemperature", "conductivity", "salinity", "significantWaveHeight",
    "significantWavePeriod", "peakWaveEnergyPeriod", "zeroCrossingPeriod",
    "oneMinuteAvgVisibility", "tenMinuteAvgVisibility", "oneHourAvgVisibility",
    "twentyFourHourAvgVisibility", "compass",
]
UNITS = {
    "airTemperature": "°C", "airPressure": "hPa", "relativeHumidity": "%",
    "dewPointTemperature": "K", "windSpeed": "m/s", "windDirection": "deg",
    "globalRadiation": "W/m²", "precipitationDifference": "mm",
    "precipitationIntensity": "mm/h", "currentSpeed": "m/s", "currentDirection": "deg",
    "tideLevel": "m", "waterTemperature": "°C", "conductivity": "mS/cm", "salinity": "PSU",
    "significantWaveHeight": "m", "significantWavePeriod": "s", "peakWaveEnergyPeriod": "s",
    "zeroCrossingPeriod": "s", "oneMinuteAvgVisibility": "m", "tenMinuteAvgVisibility": "m",
    "oneHourAvgVisibility": "m", "twentyFourHourAvgVisibility": "m", "compass": "deg",
}
DUP_UNITS = {
    "windChillTemperature": "°C", "tidePressure": "hPa", "waterPressure": "hPa",
    "waterLevel": "m", "waterTemperature_WQ": "°C", "maxWaveHeight": "m",
}
ENGINE_COLOR = {"iTransformer": "#bcbd22", "XGBoost": "#17becf"}


@st.cache_data
def load_history():
    return pd.read_csv(HISTORY_PATH, index_col=0, parse_dates=True)


@st.cache_data
def load_forecast():
    return pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_duplicates():
    return pd.read_csv(DUP_RECON_PATH), pd.read_csv(DUP_FVA_PATH, parse_dates=["timestamp"])


@st.cache_data
def load_attention():
    return pd.read_csv(ATTENTION_PATH, index_col=0)


history = load_history()
fva = load_forecast()
metrics = load_metrics()
dup_recon, dup_fva = load_duplicates()
attn = load_attention()

hard_metrics = metrics[metrics["parameter"].isin(HARD_PARAMS)]
mean_v2 = hard_metrics["hybrid_v2_skill_%"].mean()
mean_deepar = hard_metrics["deepar_hybrid_skill_%"].mean()
mean_pure = hard_metrics["pure_iTransformer_skill_%"].mean()
v2_wins = int((hard_metrics["improvement_vs_deepar_pp"] > 0).sum())

st.title("🔀 Marine 48-Hour Forecast — Hybrid v2: iTransformer + Tweedie/Huber XGBoost")
st.caption(
    "After checking each hard parameter's actual distribution — precipitation is "
    "zero-inflated (82-96.5% exact zeros), visibility is ceiling-saturated (mean sits "
    "at the sensor max) — this swaps DeepAR for XGBoost with a Tweedie loss "
    "(precipitation) and Pseudo-Huber loss (visibility), the literature-standard fit "
    "for each shape."
)

if mean_v2 > mean_deepar:
    st.success(f"**v2 wins on average**: mean skill {mean_v2:+.1f}% vs DeepAR-hybrid's {mean_deepar:+.1f}% "
               f"(both far ahead of pure iTransformer's {mean_pure:+.1f}%).")
else:
    st.warning(
        f"**Honest result: v2 does NOT beat the DeepAR-hybrid on average.** Mean skill on "
        f"the 6 hard parameters — v2: {mean_v2:+.1f}%, DeepAR-hybrid: {mean_deepar:+.1f}%, "
        f"pure iTransformer: {mean_pure:+.1f}%. v2 only wins outright on {v2_wins}/6 "
        f"parameters. Both v2 and DeepAR are far better than pure iTransformer, and both "
        f"are converging near the persistence floor — consistent with this project's own "
        f"literature review, which flags precipitation/visibility beyond a few hours as "
        f"having a genuine forecastability ceiling, not a model-choice problem."
    )

tab_compare, tab_forecast, tab_dup, tab_attn, tab_metrics, tab_about = st.tabs(
    ["⚖️ Three-Way Comparison", "📈 Forecast", "🔂 Duplicate Reconstruction",
     "🧠 Attention Map", "📊 Metrics", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ Pure iTransformer vs. DeepAR-hybrid vs. XGBoost-hybrid (v2)")
    st.markdown("On the 6 historically hard parameters specifically:")
    disp = hard_metrics[["parameter", "hybrid_v2_skill_%", "deepar_hybrid_skill_%",
                          "pure_iTransformer_skill_%", "improvement_vs_pure_pp",
                          "improvement_vs_deepar_pp"]].sort_values("hybrid_v2_skill_%", ascending=False)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Pure iTransformer", x=disp["parameter"], y=disp["pure_iTransformer_skill_%"],
                          marker_color="#d62728"))
    fig.add_trace(go.Bar(name="DeepAR-hybrid", x=disp["parameter"], y=disp["deepar_hybrid_skill_%"],
                          marker_color="#ffd700"))
    fig.add_trace(go.Bar(name="XGBoost-hybrid (v2)", x=disp["parameter"], y=disp["hybrid_v2_skill_%"],
                          marker_color="#17becf"))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(barmode="group", height=500, yaxis_title="Skill vs persistence (%)",
                       title="Lower bars (pure iTransformer) clipped for readability — see table for exact values",
                       yaxis=dict(range=[-50, 30]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Pure iTransformer's bars (−100% to −410%) are off-scale above and clipped to "
        "keep the DeepAR-vs-v2 comparison readable — see the table for exact values."
    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean skill — Pure iTransformer", f"{mean_pure:+.1f}%")
    c2.metric("Mean skill — DeepAR-hybrid", f"{mean_deepar:+.1f}%")
    c3.metric("Mean skill — XGBoost-hybrid (v2)", f"{mean_v2:+.1f}%",
              delta=f"{mean_v2 - mean_deepar:+.1f}pp vs DeepAR")
    st.metric("Parameters where v2 beats DeepAR-hybrid", f"{v2_wins} / 6")

# ---------------------------------------------------------------------------
with tab_forecast:
    col1, col2 = st.columns([3, 1])
    with col1:
        parameter = st.selectbox(
            "Parameter", REPORT_PARAMS,
            format_func=lambda p: f"{p} ({UNITS.get(p, '')}) — {'XGBoost' if p in HARD_PARAMS else 'iTransformer'}",
        )
        history_hours = st.slider("History context (hours)", 12, 240, 96, step=12)
        history_steps = history_hours * 6
        hist_tail = history[parameter].iloc[-(HORIZON_STEPS + history_steps):-HORIZON_STEPS]
        engine = "XGBoost" if parameter in HARD_PARAMS else "iTransformer"

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                                   line=dict(color="lightgray", width=1.5)))
        fig2.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__actual"],
                                   name="actual", line=dict(color="black", width=3)))
        fig2.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__hybrid_v2"],
                                   name=f"hybrid v2 ({engine})",
                                   line=dict(color=ENGINE_COLOR[engine], width=2, dash="dash")))
        fig2.add_vline(x=fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
        fig2.update_layout(
            title=f"{parameter} — served by {engine}",
            xaxis_title="Time", yaxis_title=f"{parameter} ({UNITS.get(parameter, '')})",
            height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=80),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Scorecard")
        row = metrics[metrics["parameter"] == parameter].iloc[0]
        st.metric("Engine", engine)
        st.metric("Rank (of 24)", int(row["rank"]))
        st.metric("Persistence MAE", f"{row['Persistence_MAE']:.4f} {UNITS.get(parameter, '')}")
        st.metric("Hybrid v2 MAE", f"{row['hybrid_v2_MAE']:.4f} {UNITS.get(parameter, '')}",
                   delta=f"{row['hybrid_v2_skill_%']:+.1f}% vs persistence")
        if parameter in HARD_PARAMS:
            st.info(f"DeepAR-hybrid scored {row['deepar_hybrid_skill_%']:+.1f}% here "
                    f"({'v2 wins' if row['improvement_vs_deepar_pp']>0 else 'DeepAR wins'} "
                    f"by {abs(row['improvement_vs_deepar_pp']):.1f}pp).")

# ---------------------------------------------------------------------------
with tab_dup:
    st.subheader("🔂 The 6 duplicate parameters, reconstructed")
    st.markdown("All kept twins are iTransformer-served, so reconstructions inherit its forecast.")
    st.dataframe(dup_recon, use_container_width=True, hide_index=True)

    dup_param = st.selectbox(
        "Inspect a duplicate parameter", dup_recon["duplicate_parameter"].tolist(),
        format_func=lambda p: f"{p} ({DUP_UNITS.get(p, '')})",
    )
    row = dup_recon[dup_recon["duplicate_parameter"] == dup_param].iloc[0]
    hist_tail = history[dup_param].iloc[-(HORIZON_STEPS + LOOKBACK_STEPS):-HORIZON_STEPS]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                               line=dict(color="lightgray", width=1.5)))
    fig3.add_trace(go.Scatter(x=dup_fva["timestamp"], y=dup_fva[f"{dup_param}__actual"],
                               name="actual", line=dict(color="black", width=3)))
    fig3.add_trace(go.Scatter(x=dup_fva["timestamp"], y=dup_fva[f"{dup_param}__reconstructed"],
                               name=f"reconstructed (from {row['reconstructed_from']})",
                               line=dict(color="#d62728", width=2, dash="dash")))
    fig3.add_vline(x=dup_fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
    fig3.update_layout(
        title=f"{dup_param} — train R²={row['train_R2']:.5f}",
        xaxis_title="Time", yaxis_title=f"{dup_param} ({DUP_UNITS.get(dup_param, '')})",
        height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=80),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"Held-out MAE = {row['held_out_MAE']:.4f}, RMSE = {row['held_out_RMSE']:.4f} {DUP_UNITS.get(dup_param, '')}")

# ---------------------------------------------------------------------------
with tab_attn:
    st.subheader("🧠 iTransformer cross-variate attention (18-parameter model, unchanged from v1)")
    fig4 = go.Figure(data=go.Heatmap(
        z=attn.values, x=attn.columns.tolist(), y=attn.index.tolist(),
        colorscale="Viridis", colorbar=dict(title="attention"),
    ))
    fig4.update_layout(
        height=800, xaxis_title="attends TO (key)", yaxis_title="query parameter",
        xaxis=dict(tickfont=dict(size=8)), yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    query_param = st.selectbox("Query parameter", attn.index.tolist(), key="attn_query")
    top5 = attn.loc[query_param].drop(query_param, errors="ignore").sort_values(ascending=False).head(5)
    st.bar_chart(top5)

# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Full metrics — all 24 parameters")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.metric("Median hybrid v2 skill", f"{metrics['hybrid_v2_skill_%'].median():.1f}%")
    st.metric("Parameters beating persistence", f"{int((metrics['hybrid_v2_skill_%']>0).sum())} / 24")

# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
        ### What this is
        A second hybrid attempt: same iTransformer for the 18 "good" parameters, but the
        6 "hard" ones now use **XGBoost with a distribution-appropriate loss** instead of
        DeepAR — Tweedie for the zero-inflated precipitation parameters, Pseudo-Huber
        (on standardized values) for the ceiling-saturated visibility parameters.

        ### Why this swap was proposed
        DeepAR assumes every channel is approximately Gaussian. Checking the actual data:
        precipitation is 82-96.5% exact zeros (zero-inflated, compound Poisson-Gamma
        shape — Tweedie's literature use case), and visibility sits almost always at a
        sensor ceiling with rare large drops (not Gaussian, and an RNN's recursive
        rollout can drift past the physical envelope — which is exactly what caused
        DeepAR's catastrophic 2 misses and what caused the *first* attempt at this fix,
        using `reg:pseudohubererror` on visibility's raw scale, to blow up to billions of
        percent skill from numerical instability — fixed by standardizing first).

        ### The honest result
        **v2 does not clearly beat the DeepAR-hybrid.** Mean skill on the 6 hard
        parameters: pure iTransformer {mean_pure:+.1f}%, DeepAR-hybrid {mean_deepar:+.1f}%,
        XGBoost-hybrid v2 {mean_v2:+.1f}%. v2 wins outright on only {v2_wins} of 6. Both
        hybrid approaches are dramatically better than pure iTransformer and both
        converge near the persistence floor — which matches this project's own
        literature review (`MARINE_FORECASTING_IMPLEMENTATION_GUIDE.md`): precipitation
        and visibility beyond a few hours have a genuine **forecastability ceiling** at
        only 28 days of training history. Getting the distributional assumption right
        fixed the *catastrophic* failure mode (no more billion-percent blowups, no more
        −400% skill) but didn't meaningfully beat a reasonably-chosen alternative
        architecture (DeepAR) on a dataset this size — both are bumping against a data
        ceiling, not a model-choice gap.

        ### Practical recommendation
        For these 6 parameters specifically, either approach is defensible; persistence
        itself is competitive. If more training history becomes available, revisit —
        Tweedie/Huber's theoretical edge should become more visible with more rare-event
        examples to learn from.

        ### Source notebook
        `Marine_Forecast_RealEMS_Hybrid_iTransformer_XGBoost.ipynb`
        """
    )
