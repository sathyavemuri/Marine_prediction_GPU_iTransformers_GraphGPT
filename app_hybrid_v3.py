"""
Marine 48h Forecast — Hybrid v3 (Quantile XGBoost + Cross-Features) Dashboard
================================================================================
Streamlit viewer for Marine_Forecast_RealEMS_Hybrid_iTransformer_QuantileXGB.ipynb.
Leads with the honest four-way comparison — this version is the WORST of the four
approaches tried for the 6 hard parameters, and the dashboard explains exactly why
(traced to feature importance + the mechanism by which rare-event sample weighting
hurts a median-regression objective).

Run with:
    streamlit run app_hybrid_v3.py --server.port 8506
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine Forecast — Hybrid v3 (Quantile+CrossFeatures)", layout="wide")

HISTORY_PATH = "ems_10min_resampled.csv"
FORECAST_PATH = "forecast_vs_actual_hybrid_v3.csv"
METRICS_PATH = "metrics_hybrid_v3.csv"
DUP_RECON_PATH = "duplicate_reconstruction_hybrid_v3.csv"
DUP_FVA_PATH = "duplicate_forecast_vs_actual_hybrid_v3.csv"
FEAT_IMP_PATH = "feature_importance_hybrid_v3.csv"

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
ENGINE_COLOR = {"iTransformer": "#bcbd22", "XGBoost-Quantile": "#9467bd"}


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
def load_feature_importance():
    return pd.read_csv(FEAT_IMP_PATH)


history = load_history()
fva = load_forecast()
metrics = load_metrics()
dup_recon, dup_fva = load_duplicates()
feat_imp = load_feature_importance()

hard_metrics = metrics[metrics["parameter"].isin(HARD_PARAMS)]
mean_v3 = hard_metrics["hybrid_v3_skill_%"].mean()
mean_v2 = hard_metrics["xgb_v2_skill_%"].mean()
mean_deepar = hard_metrics["deepar_hybrid_skill_%"].mean()
mean_pure = hard_metrics["pure_iTransformer_skill_%"].mean()

st.title("🔀 Marine 48-Hour Forecast — Hybrid v3: Quantile XGBoost + Cross-Features")
st.caption(
    "A literature comparison table for these parameter categories suggested 3 concrete "
    "fixes for the hard 6 (precipitation ×2, visibility ×4): cross-parameter "
    "meteorological features, quantile/median loss (directly optimizes MAE), and "
    "rare-event sample weighting. This notebook tried all three together."
)

st.error(
    f"**Negative result: v3 is the worst of the four approaches tried.** Mean skill on "
    f"the 6 hard parameters — pure iTransformer: {mean_pure:+.1f}%, DeepAR-hybrid: "
    f"{mean_deepar:+.1f}%, XGBoost-hybrid v2 (Tweedie/Huber): {mean_v2:+.1f}%, "
    f"**XGBoost-hybrid v3 (this): {mean_v3:+.1f}%**. v3 underperforms v2 and DeepAR on "
    f"every single one of the 6 parameters. See the Feature Importance tab for the "
    f"diagnosed root cause."
)

tab_compare, tab_feat, tab_forecast, tab_dup, tab_metrics, tab_about = st.tabs(
    ["⚖️ Four-Way Comparison", "🔍 Feature Importance (root cause)", "📈 Forecast",
     "🔂 Duplicate Reconstruction", "📊 Metrics", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ All four approaches on the 6 historically hard parameters")
    disp = hard_metrics[["parameter", "hybrid_v3_skill_%", "xgb_v2_skill_%",
                          "deepar_hybrid_skill_%", "pure_iTransformer_skill_%"]].sort_values(
        "hybrid_v3_skill_%", ascending=False)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="DeepAR-hybrid (best)", x=disp["parameter"], y=disp["deepar_hybrid_skill_%"],
                          marker_color="#ffd700"))
    fig.add_trace(go.Bar(name="XGBoost v2 (Tweedie/Huber)", x=disp["parameter"], y=disp["xgb_v2_skill_%"],
                          marker_color="#17becf"))
    fig.add_trace(go.Bar(name="XGBoost v3 (Quantile+CrossFeat, this)", x=disp["parameter"],
                          y=disp["hybrid_v3_skill_%"], marker_color="#9467bd"))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(barmode="group", height=500, yaxis_title="Skill vs persistence (%)",
                       title="Pure iTransformer (-100% to -410%) omitted from this chart for readability")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pure iTransformer", f"{mean_pure:+.1f}%")
    c2.metric("DeepAR-hybrid", f"{mean_deepar:+.1f}%", help="Best so far")
    c3.metric("XGBoost v2", f"{mean_v2:+.1f}%")
    c4.metric("XGBoost v3 (this)", f"{mean_v3:+.1f}%", delta=f"{mean_v3-mean_v2:+.1f}pp vs v2")

# ---------------------------------------------------------------------------
with tab_feat:
    st.subheader("🔍 Why v3 regressed: feature importance reveals the cross-features weren't used")
    st.markdown(
        "The literature table flagged `relativeHumidity`, `dewPointTemperature`, and "
        "`windSpeed` as key inputs. Here's what the trained models actually relied on — "
        "**red = cross-feature, blue = calendar/own-lag**:"
    )
    param_choice = st.selectbox("Parameter", HARD_PARAMS)
    sub = feat_imp[feat_imp["parameter"] == param_choice].sort_values("importance", ascending=True)
    fig2 = go.Figure(go.Bar(
        x=sub["importance"], y=sub["feature"], orientation="h",
        marker_color=["#d62728" if c else "#1f77b4" for c in sub["is_cross_feature"]],
    ))
    fig2.update_layout(height=400, xaxis_title="Feature importance", title=f"Top-10 features for {param_choice}")
    st.plotly_chart(fig2, use_container_width=True)

    cross_feat_share = feat_imp.groupby("parameter")["is_cross_feature"].apply(
        lambda s: feat_imp.loc[s.index][s]["importance"].sum() / feat_imp.loc[s.index]["importance"].sum()
    )
    st.markdown("**Cross-feature share of total top-10 importance, per parameter:**")
    st.bar_chart(cross_feat_share)
    st.caption(
        "Mostly under 20-30% even where present (and dominated by `airPressure`, not the "
        "humidity/dew-point/wind signals the literature specifically named) — calendar "
        "features (`hour_sin/cos`, `dom_sin/cos`) dominate instead. With only 28 days of "
        "training data, the model doesn't have enough examples to learn genuine "
        "meteorological coupling; calendar position is a stronger (if less physically "
        "meaningful) statistical signal at this sample size."
    )
    st.warning(
        "**The second, larger culprit is the 8× rare-event sample weighting.** Quantile "
        "(median) regression is optimal *for the true, unweighted empirical "
        "distribution*. Artificially upweighting rare events shifts the fitted median "
        "away from that optimum — directly increasing MAE on the majority of normal "
        "periods to (unsuccessfully) chase the rare ones. This is a traceable causal "
        "mechanism, not random variation."
    )

# ---------------------------------------------------------------------------
with tab_forecast:
    col1, col2 = st.columns([3, 1])
    with col1:
        parameter = st.selectbox(
            "Parameter", REPORT_PARAMS,
            format_func=lambda p: f"{p} ({UNITS.get(p, '')}) — {'XGBoost-Quantile' if p in HARD_PARAMS else 'iTransformer'}",
        )
        history_hours = st.slider("History context (hours)", 12, 240, 96, step=12)
        history_steps = history_hours * 6
        hist_tail = history[parameter].iloc[-(HORIZON_STEPS + history_steps):-HORIZON_STEPS]
        engine = "XGBoost-Quantile" if parameter in HARD_PARAMS else "iTransformer"

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                                   line=dict(color="lightgray", width=1.5)))
        fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__actual"],
                                   name="actual", line=dict(color="black", width=3)))
        fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__hybrid_v3"],
                                   name=f"hybrid v3 ({engine})",
                                   line=dict(color=ENGINE_COLOR[engine], width=2, dash="dash")))
        fig3.add_vline(x=fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
        fig3.update_layout(
            title=f"{parameter} — served by {engine}",
            xaxis_title="Time", yaxis_title=f"{parameter} ({UNITS.get(parameter, '')})",
            height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=80),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Scorecard")
        row = metrics[metrics["parameter"] == parameter].iloc[0]
        st.metric("Engine", engine)
        st.metric("Persistence MAE", f"{row['Persistence_MAE']:.4f} {UNITS.get(parameter, '')}")
        st.metric("Hybrid v3 MAE", f"{row['hybrid_v3_MAE']:.4f} {UNITS.get(parameter, '')}",
                   delta=f"{row['hybrid_v3_skill_%']:+.1f}% vs persistence")
        if parameter in HARD_PARAMS:
            st.info(f"v2 scored {row['xgb_v2_skill_%']:+.1f}%, DeepAR-hybrid scored "
                    f"{row['deepar_hybrid_skill_%']:+.1f}% here.")

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
        A third hybrid attempt for the 6 historically hard parameters, this time guided
        by a literature comparison table for these exact parameter categories
        (directional/wave-period/precipitation/visibility). The table suggested 3
        concrete, specific fixes neither v1 (DeepAR) nor v2 (Tweedie/Huber XGBoost) had
        tried: cross-parameter meteorological features, quantile (median) loss to
        directly target MAE, and rare-event sample weighting.

        ### The negative result
        **v3 underperforms both prior hybrids on every one of the 6 parameters.** Mean
        skill: pure iTransformer {mean_pure:+.1f}%, DeepAR-hybrid {mean_deepar:+.1f}%,
        XGBoost v2 {mean_v2:+.1f}%, **XGBoost v3 {mean_v3:+.1f}%**.

        ### Root cause (traceable, not just "it didn't work")
        1. **Cross-features were barely used.** Feature importance shows calendar
           features (`hour_sin/cos`, `dom_sin/cos`) dominate; the named meteorological
           inputs rank low. With only 28 days of training data, there isn't enough
           signal for genuine humidity/dew-point/wind coupling to outrank simpler (if
           less physically meaningful) calendar patterns.
        2. **Rare-event sample weighting actively hurt.** Quantile/median regression's
           value is being optimal *for the true, unweighted distribution* — directly
           equal to minimizing MAE. Artificially upweighting rare events pulls the
           fitted median away from that optimum on purpose, trading overall-MAE
           performance for (unsuccessful) attempts to chase rare extremes.

        ### What this confirms
        This project's literature review already flagged precipitation and visibility
        beyond a few hours as having a genuine forecastability ceiling at this data
        volume. Three different modeling philosophies (Gaussian RNN, Tweedie/Huber GBM,
        quantile GBM with richer features) all converge near the same floor — strong
        evidence the limit is **data quantity**, not model choice. **DeepAR-hybrid
        (port 8504) remains the best approach found** for these 6 parameters.

        ### Source notebook
        `Marine_Forecast_RealEMS_Hybrid_iTransformer_QuantileXGB.ipynb`
        """
    )
