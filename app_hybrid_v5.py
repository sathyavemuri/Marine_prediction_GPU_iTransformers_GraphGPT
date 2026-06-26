"""
Marine 48h Forecast — Hybrid v5 (iTransformer Direct + XGBoost Residual Correction) Dashboard
================================================================================================
Streamlit viewer for Marine_Forecast_RealEMS_Hybrid_iTransformer_ResidualXGB.ipynb.
Leads with the honest six-way comparison — this version is the WORST of the five hybrid
approaches tried for the 6 hard parameters. Root cause: the residual-correction literature
pattern (NWP bias correction with XGBoost) assumes the base forecast has decent skill with a
learnable systematic bias. Here the base forecast (a dedicated iTransformer trained directly
on the hard 6) doesn't clear persistence on 5 of 6 parameters — there's no good systematic
bias to learn, only noise, so XGBoost's correction step actively overfits on 3 of 6.

Run with:
    streamlit run app_hybrid_v5.py --server.port 8508
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine Forecast — Hybrid v5 (Residual-Correction)", layout="wide")

HISTORY_PATH = "ems_10min_resampled.csv"
FORECAST_PATH = "forecast_vs_actual_hybrid_v5.csv"
METRICS_PATH = "metrics_hybrid_v5.csv"
DUP_RECON_PATH = "duplicate_reconstruction_hybrid_v5.csv"
DUP_FVA_PATH = "duplicate_forecast_vs_actual_hybrid_v5.csv"
BASE_VS_CORRECTED_PATH = "hybrid_v5_base_vs_corrected.csv"

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
ENGINE_COLOR = {"iTransformer": "#bcbd22", "iTransformer+XGBoost-Residual": "#17becf"}


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
def load_base_vs_corrected():
    return pd.read_csv(BASE_VS_CORRECTED_PATH)


history = load_history()
fva = load_forecast()
metrics = load_metrics()
dup_recon, dup_fva = load_duplicates()
base_vs_corrected = load_base_vs_corrected()

hard_metrics = metrics[metrics["parameter"].isin(HARD_PARAMS)]
mean_v5 = hard_metrics["hybrid_v5_skill_%"].mean()
mean_v4 = hard_metrics["timexer_v4_skill_%"].mean()
mean_v3 = hard_metrics["xgb_v3_skill_%"].mean()
mean_v2 = hard_metrics["xgb_v2_skill_%"].mean()
mean_deepar = hard_metrics["deepar_hybrid_skill_%"].mean()
mean_pure = hard_metrics["pure_iTransformer_skill_%"].mean()

st.title("🔀 Marine 48-Hour Forecast — Hybrid v5: iTransformer Direct + XGBoost Residual Correction")
st.caption(
    "Tests a literature pattern not tried in v1-v4: residual/bias-correction stacking (the approach "
    "used for NWP precipitation bias-correction with XGBoost). A dedicated iTransformer is trained "
    "directly on the hard 6 (own loss, not diluted by the 18 good params); XGBoost then learns to "
    "correct its residual using lags, the base forecast itself, and known-future exogenous drivers."
)

st.error(
    f"**Negative result: v5 is the worst of the five hybrid approaches tried.** Mean skill on the 6 "
    f"hard parameters — pure iTransformer: {mean_pure:+.1f}%, DeepAR-hybrid (best, port 8504): "
    f"{mean_deepar:+.1f}%, XGBoost v2: {mean_v2:+.1f}%, XGBoost v3: {mean_v3:+.1f}%, TimeXer-lite v4: "
    f"{mean_v4:+.1f}%, **iTransformer+XGBoost-Residual (this): {mean_v5:+.1f}%**. See the Root Cause "
    f"tab — the literature pattern's own precondition (a base forecast with learnable systematic bias) "
    f"doesn't hold on this dataset."
)

tab_compare, tab_root, tab_forecast, tab_dup, tab_metrics, tab_about = st.tabs(
    ["⚖️ Six-Way Comparison", "🔍 Root Cause", "📈 Forecast",
     "🔂 Duplicate Reconstruction", "📊 Metrics", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ All six approaches on the 6 historically hard parameters")
    disp = hard_metrics[["parameter", "hybrid_v5_skill_%", "timexer_v4_skill_%", "xgb_v3_skill_%",
                          "xgb_v2_skill_%", "deepar_hybrid_skill_%", "pure_iTransformer_skill_%"]].sort_values(
        "hybrid_v5_skill_%", ascending=False)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="DeepAR-hybrid (best, v1)", x=disp["parameter"], y=disp["deepar_hybrid_skill_%"],
                          marker_color="#ffd700"))
    fig.add_trace(go.Bar(name="XGBoost v2", x=disp["parameter"], y=disp["xgb_v2_skill_%"], marker_color="#17becf"))
    fig.add_trace(go.Bar(name="XGBoost v3", x=disp["parameter"], y=disp["xgb_v3_skill_%"], marker_color="#9467bd"))
    fig.add_trace(go.Bar(name="TimeXer-lite v4", x=disp["parameter"], y=disp["timexer_v4_skill_%"],
                          marker_color="#e377c2"))
    fig.add_trace(go.Bar(name="Residual-Correction v5 (this)", x=disp["parameter"], y=disp["hybrid_v5_skill_%"],
                          marker_color="#d62728"))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(barmode="group", height=520, yaxis_title="Skill vs persistence (%)",
                       title="Pure iTransformer (-100% to -410%) omitted from this chart for readability")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pure iTransformer", f"{mean_pure:+.1f}%")
    c2.metric("DeepAR-hybrid", f"{mean_deepar:+.1f}%", help="Best so far")
    c3.metric("XGBoost v2", f"{mean_v2:+.1f}%")
    c4.metric("XGBoost v3", f"{mean_v3:+.1f}%")
    c5.metric("TimeXer-lite v4", f"{mean_v4:+.1f}%")
    c6.metric("v5 (this)", f"{mean_v5:+.1f}%", delta=f"{mean_v5-mean_v4:+.1f}pp vs v4")

# ---------------------------------------------------------------------------
with tab_root:
    st.subheader("🔍 Why v5 regressed: the base forecast itself doesn't clear persistence")
    st.markdown(
        "Residual-correction stacking (used in NWP-bias-correction literature for precipitation) "
        "assumes the base model has **decent skill with a learnable systematic bias** — XGBoost then "
        "just nudges that bias away. Here's the dedicated iTransformer-hard-base's own skill, "
        "*before* any correction, vs. persistence:"
    )
    base_skill_rows = []
    for p in HARD_PARAMS:
        actual = fva[f"{p}__actual"]
        base = fva[f"{p}__base_uncorrected"]
        hist_val = history[p].iloc[-(HORIZON_STEPS + 1)]
        mae_persist = (actual - hist_val).abs().mean()
        mae_base = (actual - base).abs().mean()
        skill_base = (1 - mae_base / mae_persist) * 100
        base_skill_rows.append({"parameter": p, "persistence_MAE": round(mae_persist, 4),
                                 "base_MAE_uncorrected": round(mae_base, 4),
                                 "base_skill_vs_persistence_%": round(skill_base, 1)})
    base_skill_df = pd.DataFrame(base_skill_rows)
    st.dataframe(base_skill_df, use_container_width=True, hide_index=True)
    st.warning(
        "**5 of 6 parameters: the uncorrected base is already worse than persistence** (only "
        "`twentyFourHourAvgVisibility` clears it, at +8.0%). This is actually *better* than the "
        "original diluted all-24-parameter single-iTransformer on every parameter — dedicating the "
        "loss to just the hard 6 helps the base model somewhat — but 'better than catastrophic' "
        "is not the same as 'good enough to correct.' There is no clean systematic bias here for "
        "XGBoost to learn; the errors are large and largely unsystematic (the model misses real "
        "excursions rather than being off by a consistent offset)."
    )

    st.subheader("Did the correction step help or hurt, relative to the (already poor) base?")
    st.dataframe(base_vs_corrected, use_container_width=True, hide_index=True)
    n_helped = int(base_vs_corrected["correction_helped"].sum())
    st.markdown(
        f"Correction reduced MAE vs. the uncorrected base on **{n_helped}/6** parameters (the 3 "
        f"visibility-window parameters), but **made it worse on the other 3** — notably both "
        f"precipitation parameters, where trying to correct a noisy, near-zero base pushed the "
        f"forecast further from actual."
    )
    st.info(
        "**What this confirms:** the literature pattern itself isn't wrong — it works well when "
        "applied to physical NWP models that already have meaningful skill with a known bias "
        "(that's the actual context the cited papers operate in). Applying it to a from-scratch "
        "deep-learning base trained on 28 days of data, where the base itself doesn't clear "
        "persistence, removes the precondition the pattern depends on. **DeepAR-hybrid (port 8504) "
        "remains the best result found for these 6 parameters.**"
    )

# ---------------------------------------------------------------------------
with tab_forecast:
    col1, col2 = st.columns([3, 1])
    with col1:
        parameter = st.selectbox(
            "Parameter", REPORT_PARAMS,
            format_func=lambda p: f"{p} ({UNITS.get(p, '')})",
        )
        history_hours = st.slider("History context (hours)", 12, 240, 96, step=12)
        history_steps = history_hours * 6
        hist_tail = history[parameter].iloc[-(HORIZON_STEPS + history_steps):-HORIZON_STEPS]
        engine_row = fva[f"{parameter}__engine"].iloc[0] if f"{parameter}__engine" in fva.columns else "iTransformer"

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                                   line=dict(color="lightgray", width=1.5)))
        fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__actual"],
                                   name="actual", line=dict(color="black", width=3)))
        if f"{parameter}__base_uncorrected" in fva.columns:
            fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__base_uncorrected"],
                                       name="iTransformer-hard-base (uncorrected)",
                                       line=dict(color="#1f77b4", width=1.5, dash="dot")))
        fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__hybrid_v5"],
                                   name=f"hybrid v5 ({engine_row})",
                                   line=dict(color=ENGINE_COLOR.get(engine_row, "#d62728"), width=2, dash="dash")))
        fig3.add_vline(x=fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
        fig3.update_layout(
            title=f"{parameter} — served by {engine_row}",
            xaxis_title="Time", yaxis_title=f"{parameter} ({UNITS.get(parameter, '')})",
            height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=80),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Scorecard")
        row = metrics[metrics["parameter"] == parameter].iloc[0]
        st.metric("Engine", engine_row)
        st.metric("Persistence MAE", f"{row['Persistence_MAE']:.4f} {UNITS.get(parameter, '')}")
        st.metric("Hybrid v5 MAE", f"{row['hybrid_v5_MAE']:.4f} {UNITS.get(parameter, '')}",
                   delta=f"{row['hybrid_v5_skill_%']:+.1f}% vs persistence")
        if parameter in HARD_PARAMS:
            st.info(f"v4: {row['timexer_v4_skill_%']:+.1f}% | v3: {row['xgb_v3_skill_%']:+.1f}% | "
                    f"v2: {row['xgb_v2_skill_%']:+.1f}% | DeepAR-hybrid: {row['deepar_hybrid_skill_%']:+.1f}%")

# ---------------------------------------------------------------------------
with tab_dup:
    st.subheader("🔂 The 6 duplicate parameters, reconstructed")
    st.dataframe(dup_recon, use_container_width=True, hide_index=True)
    dup_param = st.selectbox(
        "Inspect a duplicate parameter", dup_recon["duplicate_parameter"].tolist(),
        format_func=lambda p: f"{p} ({DUP_UNITS.get(p, '')})",
    )
    row = dup_recon[dup_recon["duplicate_parameter"] == dup_param].iloc[0]
    hist_tail = history[dup_param].iloc[-(HORIZON_STEPS + LOOKBACK_STEPS):-HORIZON_STEPS]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                               line=dict(color="lightgray", width=1.5)))
    fig4.add_trace(go.Scatter(x=dup_fva["timestamp"], y=dup_fva[f"{dup_param}__actual"],
                               name="actual", line=dict(color="black", width=3)))
    fig4.add_trace(go.Scatter(x=dup_fva["timestamp"], y=dup_fva[f"{dup_param}__reconstructed"],
                               name=f"reconstructed (from {row['reconstructed_from']})",
                               line=dict(color="#d62728", width=2, dash="dash")))
    fig4.add_vline(x=dup_fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
    fig4.update_layout(
        title=f"{dup_param} — train R²={row['train_R2']:.5f}",
        xaxis_title="Time", yaxis_title=f"{dup_param} ({DUP_UNITS.get(dup_param, '')})",
        height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=80),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(f"Held-out MAE = {row['held_out_MAE']:.4f}, RMSE = {row['held_out_RMSE']:.4f} {DUP_UNITS.get(dup_param, '')}")

# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Full metrics — all 24 parameters")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
        ### What this is
        A fifth hybrid attempt for the 6 historically hard parameters, testing a literature pattern
        not tried in v1-v4: **residual/bias-correction stacking**, the approach used in NWP
        precipitation bias-correction with XGBoost. Unlike v1-v4 (which all *replace* iTransformer
        outright for the hard 6), this trains a dedicated iTransformer directly on the hard 6, then
        trains XGBoost to predict the *residual* (`actual - base_forecast`) rather than the raw target.

        ### The negative result
        **v5 underperforms every prior hybrid** ({mean_v5:+.1f}% vs DeepAR-hybrid's {mean_deepar:+.1f}%,
        XGBoost v2's {mean_v2:+.1f}%, XGBoost v3's {mean_v3:+.1f}%, TimeXer-lite v4's {mean_v4:+.1f}%).

        ### Root cause (traceable, not just "it didn't work")
        The residual-correction pattern's precondition — a base forecast with decent skill and a
        learnable *systematic* bias — doesn't hold here. The dedicated iTransformer-hard-base (better
        than the original diluted all-24 single-iTransformer, but still trained on only ~26
        independent 28-day windows) doesn't clear persistence on 5 of 6 parameters. With no clean
        systematic bias to learn, XGBoost's correction step helps marginally on the 3 visibility
        parameters and actively worsens both precipitation parameters.

        ### What this confirms, again
        Five genuinely different modeling philosophies — Gaussian RNN, Tweedie/Huber GBM, quantile
        GBM with cross-features, patch+exogenous-attention with a hurdle model, and now
        literature-grounded residual-correction stacking — have all been tried on these exact 6
        parameters. The recurring failure mode is the same: **28 days of single-site sensor data is
        not enough to learn either the raw target or a stable base-model bias for visibility and
        precipitation at a 48h horizon.** **DeepAR-hybrid (port 8504) remains the best result found.**

        ### Source notebook
        `Marine_Forecast_RealEMS_Hybrid_iTransformer_ResidualXGB.ipynb`
        """
    )
