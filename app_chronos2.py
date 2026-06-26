"""
Marine 48h Forecast — Chronos-2 Zero-Shot Dashboard
========================================================================
Streamlit viewer for Marine_Forecast_RealEMS_Chronos2_ZeroShot.ipynb.
A pretrained foundation model (Amazon Chronos-2), used zero-shot — no training at all on this
dataset — forecasting all 24 parameters jointly in ~3 seconds of CPU inference. Nearly matches
iTransformer on the 18 "good" parameters and ties the best from-scratch result (DeepAR-hybrid,
5 architectures deep) on the 6 historically hard ones.

Run with:
    streamlit run app_chronos2.py --server.port 8509
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine Forecast — Chronos-2 Zero-Shot", layout="wide")

HISTORY_PATH = "ems_10min_resampled.csv"
FORECAST_PATH = "forecast_vs_actual_chronos2.csv"
METRICS_PATH = "metrics_chronos2.csv"
DUP_RECON_PATH = "duplicate_reconstruction_chronos2.csv"
DUP_FVA_PATH = "duplicate_forecast_vs_actual_chronos2.csv"

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

good_metrics = metrics[~metrics["is_hard_param"]]
hard_metrics = metrics[metrics["is_hard_param"]]
mean_chronos_good = good_metrics["chronos2_skill_%"].mean()
mean_itransformer_good = good_metrics["itransformer_skill_%"].mean()
mean_chronos_hard = hard_metrics["chronos2_skill_%"].mean()
mean_deepar_hard = hard_metrics["best_prior_skill_%"].mean()
n_beats_good = int(good_metrics["chronos2_beats_itransformer"].sum())
n_beats_hard = int((hard_metrics["chronos2_skill_%"] > hard_metrics["best_prior_skill_%"]).sum())

st.title("⚡ Marine 48-Hour Forecast — Chronos-2 Zero-Shot")
st.caption(
    "Amazon Science's Chronos-2 pretrained foundation model, used **zero-shot** — no training on "
    "this dataset at all. All 24 parameters forecast jointly (native multivariate support handles "
    "cross-parameter correlation internally) with calendar features as known-future covariates, in "
    "~3 seconds of CPU inference total. Compared honestly against the from-scratch iTransformer "
    "baseline (good 18) and the best hybrid result found across 5 architectures (DeepAR-hybrid, hard 6)."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Good-18 mean skill (Chronos-2)", f"{mean_chronos_good:+.1f}%",
          delta=f"{mean_chronos_good - mean_itransformer_good:+.1f}pp vs iTransformer")
c2.metric("Good-18 wins (of 18)", f"{n_beats_good}/18")
c3.metric("Hard-6 mean skill (Chronos-2)", f"{mean_chronos_hard:+.1f}%",
          delta=f"{mean_chronos_hard - mean_deepar_hard:+.1f}pp vs DeepAR-hybrid")
c4.metric("Hard-6 wins (of 6)", f"{n_beats_hard}/6")

st.success(
    f"**A remarkably close, zero-training result.** On the 18 good parameters, Chronos-2 zero-shot "
    f"trails iTransformer by only {mean_itransformer_good - mean_chronos_good:.1f}pp on average "
    f"({mean_chronos_good:+.1f}% vs {mean_itransformer_good:+.1f}%) — beating it outright on 3/18. "
    f"On the hard 6, it **ties the best result found across five different from-scratch architectures** "
    f"({mean_chronos_hard:+.1f}% vs DeepAR-hybrid's {mean_deepar_hard:+.1f}%), including the single best "
    f"visibility skill score recorded anywhere in this project (`tenMinuteAvgVisibility` +18.1%). All of "
    f"this from one unified model, with **no training step at all** — versus 5 separate from-scratch "
    f"hybrid pipelines (v1-v5) that took many minutes each to reach the prior best result."
)

tab_compare, tab_forecast, tab_dup, tab_metrics, tab_about = st.tabs(
    ["⚖️ Comparison", "📈 Forecast", "🔂 Duplicate Reconstruction", "📊 Metrics", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ Chronos-2 zero-shot vs. iTransformer / best-prior-hybrid, all 24 parameters")
    disp = metrics[["parameter", "is_hard_param", "chronos2_skill_%", "itransformer_skill_%",
                     "best_prior_skill_%", "chronos2_beats_itransformer"]].copy()
    disp = disp.rename(columns={"itransformer_skill_%": "reference_skill_%",
                                 "chronos2_beats_itransformer": "chronos2_wins"})
    disp["reference"] = np.where(disp["is_hard_param"], "DeepAR-hybrid (best prior, hard-6)",
                                  "iTransformer (good-18 baseline)")
    st.dataframe(disp[["parameter", "is_hard_param", "chronos2_skill_%", "reference_skill_%",
                        "reference", "chronos2_wins"]].sort_values("chronos2_skill_%", ascending=False),
                 use_container_width=True, hide_index=True)

    st.divider()
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**18 good parameters — Chronos-2 vs iTransformer**")
        g = good_metrics.sort_values("chronos2_skill_%", ascending=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name="iTransformer", y=g["parameter"], x=g["itransformer_skill_%"],
                               orientation="h", marker_color="#bcbd22"))
        fig1.add_trace(go.Bar(name="Chronos-2 (zero-shot)", y=g["parameter"], x=g["chronos2_skill_%"],
                               orientation="h", marker_color="#ff7f0e"))
        fig1.update_layout(barmode="group", height=550, xaxis_title="Skill vs persistence (%)")
        st.plotly_chart(fig1, use_container_width=True)
    with cc2:
        st.markdown("**6 hard parameters — Chronos-2 vs DeepAR-hybrid (best prior)**")
        h = hard_metrics.sort_values("chronos2_skill_%", ascending=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="DeepAR-hybrid (best prior)", y=h["parameter"], x=h["best_prior_skill_%"],
                               orientation="h", marker_color="#ffd700"))
        fig2.add_trace(go.Bar(name="Chronos-2 (zero-shot)", y=h["parameter"], x=h["chronos2_skill_%"],
                               orientation="h", marker_color="#ff7f0e"))
        fig2.add_vline(x=0, line_color="black")
        fig2.update_layout(barmode="group", height=550, xaxis_title="Skill vs persistence (%)")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_forecast:
    col1, col2 = st.columns([3, 1])
    with col1:
        parameter = st.selectbox(
            "Parameter", REPORT_PARAMS,
            format_func=lambda p: f"{p} ({UNITS.get(p, '')}) — {'hard' if p in HARD_PARAMS else 'good'}",
        )
        history_hours = st.slider("History context (hours)", 12, 240, 96, step=12)
        history_steps = history_hours * 6
        hist_tail = history[parameter].iloc[-(HORIZON_STEPS + history_steps):-HORIZON_STEPS]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                                   line=dict(color="lightgray", width=1.5)))
        if f"{parameter}__q10" in fva.columns:
            fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__q90"],
                                       line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__q10"], name="10-90% band",
                                       fill="tonexty", line=dict(width=0), fillcolor="rgba(255,127,14,0.15)"))
        fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__actual"],
                                   name="actual", line=dict(color="black", width=3)))
        fig3.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__chronos2"],
                                   name="Chronos-2 (zero-shot)", line=dict(color="#ff7f0e", width=2, dash="dash")))
        fig3.add_vline(x=fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
        fig3.update_layout(
            title=f"{parameter} — 48h zero-shot forecast",
            xaxis_title="Time", yaxis_title=f"{parameter} ({UNITS.get(parameter, '')})",
            height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=80),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Scorecard")
        row = metrics[metrics["parameter"] == parameter].iloc[0]
        st.metric("Persistence MAE", f"{row['Persistence_MAE']:.4f} {UNITS.get(parameter, '')}")
        st.metric("Chronos-2 MAE", f"{row['chronos2_MAE']:.4f} {UNITS.get(parameter, '')}",
                   delta=f"{row['chronos2_skill_%']:+.1f}% vs persistence")
        ref_label = "DeepAR-hybrid (best prior)" if row["is_hard_param"] else "iTransformer"
        ref_value = row["best_prior_skill_%"] if row["is_hard_param"] else row["itransformer_skill_%"]
        st.info(f"{ref_label}: {ref_value:+.1f}%")

# ---------------------------------------------------------------------------
with tab_dup:
    st.subheader("🔂 The 6 duplicate parameters, reconstructed from Chronos-2's forecast")
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
        A test of Amazon Science's **Chronos-2** (Oct 2025) pretrained time-series foundation model,
        used **zero-shot** — no fitting to this dataset's training split at all — across all 24
        directly-modeled real EMS parameters simultaneously. Native multivariate support handles
        cross-parameter correlation internally; calendar features are passed as known-future
        covariates via the model's `future_df` API. Total cost: ~3 seconds of CPU inference, no
        training step.

        ### Why this is a different kind of result than v1-v5
        Every prior model in this project (iTransformer, DeepAR, XGBoost ×3 loss variants,
        TimeXer-lite, residual-correction stacking) was trained from scratch on this dataset's 28
        days. Chronos-2 brings in external knowledge from pretraining on a large public time-series
        corpus — a fundamentally different way to address the data-volume ceiling every from-scratch
        approach has hit on the hard 6.

        ### The result
        - **18 good parameters:** Chronos-2 zero-shot mean skill {mean_chronos_good:+.1f}% vs
          iTransformer's {mean_itransformer_good:+.1f}% — trails by {mean_itransformer_good-mean_chronos_good:.1f}pp
          on average, winning outright on {n_beats_good}/18.
        - **6 hard parameters:** Chronos-2 zero-shot mean skill {mean_chronos_hard:+.1f}%, **tying**
          DeepAR-hybrid's {mean_deepar_hard:+.1f}% (the best result found across 5 from-scratch
          architectures) — winning outright on {n_beats_hard}/6, including the best individual
          visibility score recorded anywhere in this project.

        ### What this means
        Not a clean outright win on either front, but a genuinely strong showing given the radical
        difference in cost: one model, zero training, seconds of inference, versus five separate
        from-scratch hybrid pipelines that each took meaningful build/train time to reach the prior
        best results. Worth treating as a serious baseline going forward, especially for rapid
        iteration or scenarios where training data is even more limited than the 28 days available here.

        ### Source notebook
        `Marine_Forecast_RealEMS_Chronos2_ZeroShot.ipynb`
        """
    )
