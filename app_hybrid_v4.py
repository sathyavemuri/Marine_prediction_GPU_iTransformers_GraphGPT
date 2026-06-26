"""
Marine 48h Forecast — Hybrid v4 (TimeXer-lite + Hurdle XGBoost) Dashboard
================================================================================
Streamlit viewer for Marine_Forecast_RealEMS_Hybrid_iTransformer_TimeXer.ipynb.
Leads with the honest five-way comparison — this version is the WORST of the four
hard-6 approaches tried so far. Root cause: the visibility model collapsed toward
the unconditional ceiling and missed the one fog dip in the test window entirely
(predicted std ~390 vs actual std ~1800-1815); the precipitation hurdle model is an
improvement over v3 but still well behind DeepAR-hybrid.

Run with:
    streamlit run app_hybrid_v4.py --server.port 8507
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine Forecast — Hybrid v4 (TimeXer-lite+Hurdle)", layout="wide")

HISTORY_PATH = "ems_10min_resampled.csv"
FORECAST_PATH = "forecast_vs_actual_hybrid_v4.csv"
METRICS_PATH = "metrics_hybrid_v4.csv"
DUP_RECON_PATH = "duplicate_reconstruction_hybrid_v4.csv"
DUP_FVA_PATH = "duplicate_forecast_vs_actual_hybrid_v4.csv"

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
ENGINE_COLOR = {"iTransformer": "#bcbd22", "TimeXer-lite": "#e377c2", "Hurdle-XGBoost": "#8c564b"}


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


history = load_history()
fva = load_forecast()
metrics = load_metrics()
dup_recon, dup_fva = load_duplicates()

hard_metrics = metrics[metrics["parameter"].isin(HARD_PARAMS)]
mean_v4 = hard_metrics["hybrid_v4_skill_%"].mean()
mean_v3 = hard_metrics["xgb_v3_skill_%"].mean()
mean_v2 = hard_metrics["xgb_v2_skill_%"].mean()
mean_deepar = hard_metrics["deepar_hybrid_skill_%"].mean()
mean_pure = hard_metrics["pure_iTransformer_skill_%"].mean()

st.title("🔀 Marine 48-Hour Forecast — Hybrid v4: TimeXer-lite + Hurdle XGBoost")
st.caption(
    "Built from a design brief asking for TimeXer/InvDec-PatchTST-style architecture with "
    "known-future exogenous covariates, a two-stage rain/no-rain precipitation model, and a "
    "weighted loss for low-visibility/fog cases. The top-priority option in that brief — "
    "TimeXer with NWP/radar inputs — isn't implementable: this dataset has no external "
    "atmospheric forecast or radar feed, only single-point EMS sensor data. Implemented "
    "instead: iTransformer's own 48h forecasts of humidity/dew-point/pressure/wind (90-98% "
    "skill) used as the best available stand-in for known-future exogenous drivers."
)

st.error(
    f"**Negative result: v4 is the worst of the four hybrid approaches tried.** Mean skill "
    f"on the 6 hard parameters — pure iTransformer: {mean_pure:+.1f}%, DeepAR-hybrid (best, "
    f"port 8504): {mean_deepar:+.1f}%, XGBoost v2: {mean_v2:+.1f}%, XGBoost v3: {mean_v3:+.1f}%, "
    f"**TimeXer-lite + Hurdle-XGBoost (this): {mean_v4:+.1f}%**. The hurdle precipitation "
    f"model improved over v3's destructively-weighted quantile model, but the TimeXer-lite "
    f"visibility model is worse than every prior attempt at all 4 visibility parameters — "
    f"see the Root Cause tab."
)

tab_compare, tab_root, tab_forecast, tab_dup, tab_metrics, tab_about = st.tabs(
    ["⚖️ Five-Way Comparison", "🔍 Root Cause", "📈 Forecast",
     "🔂 Duplicate Reconstruction", "📊 Metrics", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ All five approaches on the 6 historically hard parameters")
    disp = hard_metrics[["parameter", "hybrid_v4_skill_%", "xgb_v3_skill_%", "xgb_v2_skill_%",
                          "deepar_hybrid_skill_%", "pure_iTransformer_skill_%"]].sort_values(
        "hybrid_v4_skill_%", ascending=False)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="DeepAR-hybrid (best, v1)", x=disp["parameter"], y=disp["deepar_hybrid_skill_%"],
                          marker_color="#ffd700"))
    fig.add_trace(go.Bar(name="XGBoost v2 (Tweedie/Huber)", x=disp["parameter"], y=disp["xgb_v2_skill_%"],
                          marker_color="#17becf"))
    fig.add_trace(go.Bar(name="XGBoost v3 (Quantile+CrossFeat)", x=disp["parameter"],
                          y=disp["xgb_v3_skill_%"], marker_color="#9467bd"))
    fig.add_trace(go.Bar(name="TimeXer-lite + Hurdle (v4, this)", x=disp["parameter"],
                          y=disp["hybrid_v4_skill_%"], marker_color="#e377c2"))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(barmode="group", height=520, yaxis_title="Skill vs persistence (%)",
                       title="Pure iTransformer (-100% to -410%) omitted from this chart for readability")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pure iTransformer", f"{mean_pure:+.1f}%")
    c2.metric("DeepAR-hybrid", f"{mean_deepar:+.1f}%", help="Best so far")
    c3.metric("XGBoost v2", f"{mean_v2:+.1f}%")
    c4.metric("XGBoost v3", f"{mean_v3:+.1f}%")
    c5.metric("v4 (this)", f"{mean_v4:+.1f}%", delta=f"{mean_v4-mean_v3:+.1f}pp vs v3")

    st.divider()
    st.subheader("Split by sub-model: did the two v4 design changes help independently?")
    precip_disp = disp[disp["parameter"].isin(PRECIP_PARAMS)]
    vis_disp = disp[disp["parameter"].isin(VISIBILITY_PARAMS)]
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Precipitation (hurdle model) — improved over v3, still behind v1/v2**")
        st.dataframe(precip_disp[["parameter", "hybrid_v4_skill_%", "xgb_v3_skill_%", "deepar_hybrid_skill_%"]],
                     use_container_width=True, hide_index=True)
    with cc2:
        st.markdown("**Visibility (TimeXer-lite) — worse than every prior attempt**")
        st.dataframe(vis_disp[["parameter", "hybrid_v4_skill_%", "xgb_v3_skill_%", "deepar_hybrid_skill_%"]],
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
with tab_root:
    st.subheader("🔍 Why v4 regressed: the visibility model missed the only fog event in the test window")
    st.markdown(
        "The 48h test window contains exactly one low-visibility (fog) dip. The actual series "
        "drops to a minimum far below the sensor ceiling; TimeXer-lite's predicted series barely "
        "moves off the ceiling at all, despite the 3x weighted-loss term specifically targeting "
        "this tail. Predicted variance collapses to roughly a fifth of actual variance:"
    )
    diag_rows = []
    for p in VISIBILITY_PARAMS:
        a = fva[f"{p}__actual"]
        b = fva[f"{p}__hybrid_v4"]
        diag_rows.append({"parameter": p, "actual_std": round(a.std(), 1), "predicted_std": round(b.std(), 1),
                           "actual_min": round(a.min(), 1), "predicted_min": round(b.min(), 1),
                           "variance_captured_%": round(100 * (b.std() / a.std()) ** 2, 1)})
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
    st.warning(
        "This is a small-data deep-learning failure mode, not a bug: with ~26 independent training "
        "windows (28 days, 288-step lookback/horizon, heavily overlapping), a model with extra "
        "capacity (patch embeddings + a second exogenous-attention branch on top of iTransformer's "
        "own architecture) has more freedom to fit the *common case* (the long flat ceiling) and "
        "less ability to generalize to the *rare case* (the one fog dip) than a smaller or "
        "differently-biased model. DeepAR's recurrent, probabilistic structure apparently makes a "
        "better implicit bias for this specific failure mode than patch+cross-attention does."
    )
    st.markdown(
        "**Precipitation did improve.** Replacing v3's destructive 8x-sample-weight-on-quantile-loss "
        "with a clean hurdle decomposition (binary rain/no-rain classifier, then a regressor trained "
        "only on rain-occurring rows) raised `precipitationIntensity` from -74.9% (v3) to -25.0% (v4) "
        "and `precipitationDifference` from -34.2% to -29.6% — real progress, just not enough to beat "
        "DeepAR-hybrid's -0.2%/-0.1%."
    )
    st.info(
        "**What this confirms, again:** four genuinely different modeling philosophies — Gaussian "
        "RNN (DeepAR), Tweedie/Huber GBM, quantile GBM with cross-features, and now patch+exogenous-"
        "attention with a hurdle model — have all been tried on these exact 6 parameters. None comes "
        "close to a consistently positive skill score. The caveat raised before building this notebook "
        "holds: at 28 days of history and a 48h horizon, the binding constraint is data volume and the "
        "absence of true external atmospheric predictors (NWP/radar), not which architecture is used. "
        "**DeepAR-hybrid (port 8504) remains the best result found for these 6 parameters.**"
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
        fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__hybrid_v4"],
                                   name=f"hybrid v4 ({engine_row})",
                                   line=dict(color=ENGINE_COLOR.get(engine_row, "#e377c2"), width=2, dash="dash")))
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
        st.metric("Hybrid v4 MAE", f"{row['hybrid_v4_MAE']:.4f} {UNITS.get(parameter, '')}",
                   delta=f"{row['hybrid_v4_skill_%']:+.1f}% vs persistence")
        if parameter in HARD_PARAMS:
            st.info(f"v3: {row['xgb_v3_skill_%']:+.1f}% | v2: {row['xgb_v2_skill_%']:+.1f}% | "
                    f"DeepAR-hybrid: {row['deepar_hybrid_skill_%']:+.1f}%")

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
        A fourth hybrid attempt for the 6 historically hard parameters, built from a design
        brief specifying TimeXer/InvDec-PatchTST-style architecture with known-future exogenous
        covariates, a two-stage precipitation model, and weighted visibility loss.

        ### What couldn't be implemented as specified
        The brief's top priority — **TimeXer with NWP/radar inputs** — requires external
        atmospheric forecast or radar data this single-point EMS sensor dataset doesn't have.
        Implemented instead: iTransformer's own 48h forecasts of `relativeHumidity`,
        `dewPointTemperature`, `airPressure`, `windSpeed` (90-98% skill) as the best available
        substitute for known-future exogenous drivers — ground truth at train time (hindsight),
        the upstream model's own forecast at inference time, exactly as a real NWP-input pipeline
        would be trained and deployed.

        ### What was implemented
        - **TimeXer-lite** for the 4 visibility parameters jointly: PatchTST-style endogenous
          patch embedding + iTransformer-style exogenous variable tokens (the known-future
          humidity/dew-point/pressure/wind forecast), fused by a small shared self-attention
          encoder — the brief's "small iTransformer-style variable-attention branch." Trained
          with a gentle 3x weighted Huber loss on the low-visibility tail.
        - **Two-stage hurdle model** for the 2 precipitation parameters: an XGBoost rain/no-rain
          classifier, then an XGBoost regressor trained only on rain-occurring rows
          (log1p target), combined as P(rain) x E[amount | rain] — the standard zero-inflated
          decomposition, replacing v3's destructive sample-weighting trick.

        ### The negative result
        **v4 underperforms every prior hybrid on mean hard-6 skill** ({mean_v4:+.1f}% vs
        DeepAR-hybrid's {mean_deepar:+.1f}%, XGBoost v2's {mean_v2:+.1f}%, XGBoost v3's
        {mean_v3:+.1f}%). The hurdle precipitation model is a real improvement over v3's
        approach; the TimeXer-lite visibility model is worse than every prior attempt, having
        missed the one fog dip in the 48h test window almost entirely (see Root Cause tab).

        ### What this confirms
        Four different modeling philosophies (Gaussian RNN, Tweedie/Huber GBM, quantile GBM
        with cross-features, patch+exogenous-attention with a hurdle model) now converge near
        the same floor on these 6 parameters. This is strong, repeated evidence the limit is
        **data volume and the absence of true external atmospheric predictors**, not which
        architecture is used. **DeepAR-hybrid (port 8504) remains the best result found.**

        ### Source notebook
        `Marine_Forecast_RealEMS_Hybrid_iTransformer_TimeXer.ipynb`
        """
    )
