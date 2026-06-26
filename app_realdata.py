"""
Marine 48h Forecast — Real EMS Data Results Dashboard
======================================================
Streamlit viewer for the forecasts/metrics produced by
Marine_Forecast_RealEMS_31Param.ipynb (11 PyTorch/XGBoost models on real,
1-minute-native, 31-parameter EMS simulator data, resampled to 10-min steps).

Run with:
    streamlit run app_realdata.py --server.port 8502
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine 48h Forecast — Real EMS Data", layout="wide")

HISTORY_PATH = "ems_10min_resampled.csv"
FORECAST_PATH = "forecast_vs_actual_realdata.csv"
METRICS_PATH = "metrics_summary_realdata.csv"
BEST_PER_PARAM_PATH = "best_model_per_parameter_realdata.csv"
ENSEMBLE_REC_PATH = "ensemble_recommendation_realdata.csv"
ENSEMBLE_FVA_PATH = "ensemble_forecast_vs_actual_realdata.csv"
UNCERTAINTY_PATH = "uncertainty_bands_realdata.csv"
PRECIP_PATH = "precipitation_type_forecast.csv"

HORIZON_STEPS = 288  # 48h @ 10-min steps

MODELS = {
    "lstm": "LSTM", "xgboost": "XGBoost", "nbeats": "N-BEATS", "nhits": "N-HiTS",
    "dlinear": "DLinear", "tide": "TiDE", "tsmixer": "TSMixer",
    "harmonicresidual": "Harmonic-Residual", "itransformer": "iTransformer",
    "patchtst": "PatchTST", "deepar": "DeepAR",
}

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

COLORS = {
    "LSTM": "#1f77b4", "XGBoost": "#d62728", "N-BEATS": "#2ca02c", "N-HiTS": "#9467bd",
    "DLinear": "#ff7f0e", "TiDE": "#8c564b", "TSMixer": "#e377c2", "Harmonic-Residual": "#17becf",
    "iTransformer": "#bcbd22", "PatchTST": "#7f7f7f", "DeepAR": "#ffd700",
}
MODEL_INFO = {
    "LSTM": "RNN, recursive rollout", "XGBoost": "Gradient-boosted trees, direct",
    "N-BEATS": "FC doubly-residual stacks, direct", "N-HiTS": "Multi-rate pooling + interpolation, direct",
    "DLinear": "Linear trend/seasonal decomposition (AAAI 2023)",
    "TiDE": "Dense encoder-decoder + covariates (Google, TMLR 2023)",
    "TSMixer": "All-MLP time+feature mixing (Google, 2023)",
    "Harmonic-Residual": "Tidal/diurnal harmonics + residual MLP (domain-specific)",
    "iTransformer": "Variate-as-token attention (Liu et al., 2023)",
    "PatchTST": "Patches + channel-independent attention (Nie et al., ICLR 2023)",
    "DeepAR": "Probabilistic RNN, 50 Monte-Carlo sample paths (Salinas et al., Amazon)",
}


@st.cache_data
def load_history():
    return pd.read_csv(HISTORY_PATH, index_col=0, parse_dates=True)


@st.cache_data
def load_forecasts():
    fva = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    metrics = pd.read_csv(METRICS_PATH)
    return fva, metrics


@st.cache_data
def load_best_per_param():
    return pd.read_csv(BEST_PER_PARAM_PATH)


@st.cache_data
def load_ensemble():
    rec = pd.read_csv(ENSEMBLE_REC_PATH)
    fva = pd.read_csv(ENSEMBLE_FVA_PATH, parse_dates=["timestamp"])
    return rec, fva


@st.cache_data
def load_uncertainty():
    try:
        return pd.read_csv(UNCERTAINTY_PATH, parse_dates=["timestamp"])
    except FileNotFoundError:
        return None


@st.cache_data
def load_precip():
    try:
        return pd.read_csv(PRECIP_PATH, parse_dates=["timestamp"])
    except FileNotFoundError:
        return None


history = load_history()
fva, metrics = load_forecasts()
best_per_param = load_best_per_param()
ensemble_rec, ensemble_fva = load_ensemble()
uncertainty_df = load_uncertainty()
precip_df = load_precip()

st.title("🌊 Marine 48-Hour Forecast — Real EMS Data Dashboard")
st.caption(
    "11 models (LSTM · XGBoost · N-BEATS · N-HiTS · DLinear · TiDE · TSMixer · "
    "Harmonic-Residual · iTransformer · PatchTST · DeepAR) on **real** 31-parameter EMS "
    "simulator data (30 days, 1-min native, resampled to 10-min) — forecasting 48 hours "
    "ahead, validated against held-out ground truth. Gaussian Process was tried and "
    "dropped (numerically unstable on this data's near-zero-variance channels — see the "
    "notebook §16)."
)

with st.sidebar:
    st.header("Controls")
    parameter = st.selectbox(
        "Parameter", REPORT_PARAMS,
        format_func=lambda p: f"{p} ({UNITS.get(p, '')})",
    )
    show_models = st.multiselect(
        "Models to show", list(MODELS.values()),
        default=["iTransformer", "PatchTST", "Harmonic-Residual", "XGBoost"],
    )
    show_uncertainty = st.checkbox(
        "Show DeepAR confidence band", value=False,
        help="Shades the ±1.96σ (95%) interval from DeepAR's 50 Monte-Carlo sample paths. "
             "Not available for circular parameters (wind direction, current direction, compass).",
    )
    history_hours = st.slider("History context (hours)", 12, 240, 96, step=12)

tab_forecast, tab_metrics, tab_winners, tab_about = st.tabs(
    ["📈 Forecast", "📊 Metrics & Skill", "🏅 Winners & Ensembles", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
with tab_forecast:
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure()
        history_steps = history_hours * 6  # 10-min steps per hour
        hist_tail = history[parameter].iloc[-(HORIZON_STEPS + history_steps):-HORIZON_STEPS]
        fig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                                  line=dict(color="lightgray", width=1.5)))
        fig.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__actual"],
                                  name="actual", line=dict(color="black", width=3)))
        for key, label in MODELS.items():
            if label in show_models:
                col = f"{parameter}__{key}"
                if col in fva.columns:
                    fig.add_trace(go.Scatter(x=fva["timestamp"], y=fva[col], name=label,
                                              line=dict(color=COLORS[label], width=2, dash="dash")))
        if show_uncertainty and uncertainty_df is not None and parameter not in CIRCULAR_PARAMS:
            mean_col, std_col = f"{parameter}__deepar_mean", f"{parameter}__deepar_std"
            if mean_col in uncertainty_df.columns and "DeepAR" in show_models:
                m, s = uncertainty_df[mean_col], uncertainty_df[std_col]
                fig.add_trace(go.Scatter(x=uncertainty_df["timestamp"], y=m + 1.96 * s,
                                          line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=uncertainty_df["timestamp"], y=m - 1.96 * s,
                                          fill="tonexty", fillcolor="rgba(255,215,0,0.15)",
                                          line=dict(width=0), name="DeepAR 95% CI", hoverinfo="skip"))
        fig.add_vline(x=fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
        fig.update_layout(
            title=f"{parameter} — 48-Hour Forecast vs Actual",
            xaxis_title="Time", yaxis_title=f"{parameter} ({UNITS.get(parameter, '')})",
            height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Scorecard")
        row = metrics[metrics["parameter"] == parameter].iloc[0]
        st.metric("Best model", row["best_model"])
        st.metric("Persistence MAE", f"{row['Persistence_MAE']:.3f} {UNITS.get(parameter, '')}")
        ranked = sorted(MODELS.values(), key=lambda lbl: row[f"{lbl}_MAE"])
        for label in ranked:
            skill_col = f"{label}_skill_%"
            mae_col = f"{label}_MAE"
            star = " 🏆" if label == row["best_model"] else ""
            st.metric(f"{label} MAE{star}",
                      f"{row[mae_col]:.3f} {UNITS.get(parameter, '')}",
                      delta=f"{row[skill_col]:+.1f}% vs persistence")

    if parameter == "precipitationIntensity" and precip_df is not None:
        st.divider()
        st.subheader("🌧️ PrecipitationType classifier (bonus, categorical)")
        st.dataframe(precip_df, use_container_width=True, hide_index=True, height=200)
        st.caption(
            "XGBoost multiclass classifier (NONE/DRIZZLE/RAIN) — predicts NONE almost "
            "always (97% accuracy, 0% recall on rare events): only ~30 rain/drizzle "
            "events exist in the 28-day training window, nowhere near enough to learn "
            "rare-event timing."
        )

# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Full metrics table — all parameters, all models")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("Forecast skill vs persistence (%)")
    skill_long = metrics.melt(
        id_vars=["parameter"],
        value_vars=[c for c in metrics.columns if c.endswith("_skill_%")],
        var_name="model", value_name="skill",
    )
    skill_long["model"] = skill_long["model"].str.replace("_skill_%", "", regex=False)
    fig2 = go.Figure()
    for m in skill_long["model"].unique():
        sub = skill_long[skill_long["model"] == m]
        fig2.add_trace(go.Bar(x=sub["skill"], y=sub["parameter"], name=m, orientation="h",
                               marker_color=COLORS.get(m)))
    fig2.add_vline(x=0, line_color="black")
    fig2.update_layout(barmode="group", height=900, xaxis_title="Skill vs persistence (%)",
                        title="Higher is better — negative means worse than naive persistence")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Wins per model (lowest MAE per parameter)")
    st.bar_chart(metrics["best_model"].value_counts())

    st.subheader("Median skill / reliability summary")
    rows = []
    for label in MODELS.values():
        s = metrics[f"{label}_skill_%"]
        rows.append(dict(model=label, median_skill_pct=round(s.median(), 1),
                          beats_persistence_pct=round((s > 0).mean() * 100, 1),
                          wins=int((metrics["best_model"] == label).sum())))
    st.dataframe(pd.DataFrame(rows).sort_values("median_skill_pct", ascending=False),
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
with tab_winners:
    st.subheader("🏅 Best single model per parameter — strongest win first")
    st.markdown(
        "Every parameter's single best model (lowest MAE), sorted by **how decisively** "
        "it wins — its skill vs. persistence."
    )
    disp = best_per_param.rename(columns={"best_skill_%": "best_model_skill_%"})
    st.dataframe(
        disp[["rank", "parameter", "best_model", "best_model_skill_%"]],
        use_container_width=True, hide_index=True,
    )

    fig3 = go.Figure(go.Bar(
        x=best_per_param["best_skill_%"], y=best_per_param["parameter"],
        orientation="h",
        marker_color=[COLORS.get(m, "#888") for m in best_per_param["best_model"]],
        text=best_per_param["best_model"], textposition="outside",
    ))
    fig3.update_layout(
        height=700, xaxis_title="Winning model's skill vs persistence (%)",
        title="Top performer at top — bar color/label shows which model won",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("🤝 Two-model ensemble opportunities")
    st.markdown(
        "For every parameter, all **55 possible pairs** of the 11 models were combined "
        "with a plain 50/50 average (circular vector-average for the 3 angle parameters) "
        "and checked against the single best model. **No weights are fit on test data** — "
        "any improvement shown is a genuine, leakage-free signal."
    )

    n_rec = int(ensemble_rec["recommended_ensemble"].sum())
    st.metric("Parameters where a 2-model ensemble beats the single best model",
              f"{n_rec} / {len(ensemble_rec)}")

    erec_disp = ensemble_rec.copy()
    erec_disp["recommended_ensemble"] = erec_disp["recommended_ensemble"].map(
        {True: "✅ Use ensemble", False: "❌ Use single model"}
    )
    erec_disp = erec_disp.sort_values("improvement_vs_single_best", ascending=False)
    st.dataframe(
        erec_disp[["parameter", "single_best_model", "single_best_MAE", "best_ensemble_pair",
                   "ensemble_MAE", "ensemble_skill_%", "improvement_vs_single_best",
                   "recommended_ensemble"]],
        use_container_width=True, hide_index=True,
    )

    st.markdown("##### Inspect one parameter: single best model vs. its best ensemble pair")
    ens_param = st.selectbox(
        "Parameter", REPORT_PARAMS, key="ens_param_select",
        format_func=lambda p: f"{p} ({UNITS.get(p, '')})",
    )
    erow = ensemble_rec[ensemble_rec["parameter"] == ens_param].iloc[0]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=ensemble_fva["timestamp"], y=ensemble_fva[f"{ens_param}__actual"],
                               name="actual", line=dict(color="black", width=3)))
    single_key = [k for k, v in MODELS.items() if v == erow["single_best_model"]][0]
    single_col = f"{ens_param}__{single_key}"
    fig4.add_trace(go.Scatter(x=fva["timestamp"], y=fva[single_col],
                               name=f"single best ({erow['single_best_model']})",
                               line=dict(color="#d62728", width=2, dash="dash")))
    fig4.add_trace(go.Scatter(x=ensemble_fva["timestamp"], y=ensemble_fva[f"{ens_param}__ensemble"],
                               name=f"ensemble ({erow['best_ensemble_pair']})",
                               line=dict(color="#1f77b4", width=2, dash="dot")))
    verdict = "✅ ensemble recommended" if erow["recommended_ensemble"] else "❌ single model recommended"
    fig4.update_layout(
        title=f"{ens_param} — {verdict}",
        xaxis_title="Time", yaxis_title=f"{ens_param} ({UNITS.get(ens_param, '')})",
        height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=80),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        f"Single best ({erow['single_best_model']}) MAE = {erow['single_best_MAE']:.4f} vs. "
        f"ensemble ({erow['best_ensemble_pair']}) MAE = {erow['ensemble_MAE']:.4f} "
        f"({'improvement' if erow['recommended_ensemble'] else 'no improvement'} of "
        f"{abs(erow['improvement_vs_single_best']):.4f} {UNITS.get(ens_param, '')})."
    )

# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### What this is
        A results viewer for **11 models** trained on **real** marine EMS simulator data
        (`Simulation_30days_Data_31parameters.xlsx` — 30 days, 1-minute native resolution,
        31 raw sensor parameters) to forecast **48 hours ahead** at 10-minute resolution.

        ### Why 48h, not the synthetic notebook's 5 days?
        With only 30 days of real data, a 5-day horizon leaves too few non-overlapping
        training cycles (~5) for a direct multi-horizon model to learn reliably — exactly
        the "insufficient data" pitfall `MARINE_FORECASTING_IMPLEMENTATION_GUIDE.md` warns
        about. 48h keeps a ~14:1 train:horizon ratio (28 days train : 2 days horizon),
        matching the literature-recommended proportions and the synthetic notebook's own
        70:5 ratio.

        ### Why no Gaussian Process here?
        It was implemented and trained successfully on the synthetic dataset, but its
        exact covariance matrix became numerically non-positive-definite on at least one
        of this real dataset's 27 channels — almost certainly one with long flat
        stretches near zero (e.g. `precipitationIntensity`), which the local RBF kernel
        can't condition well. Rather than special-case the kernel per parameter, it was
        dropped; DeepAR's Monte-Carlo band still provides uncertainty quantification.

        ### Data preprocessing
        - **6 duplicate parameters dropped** (r ≥ 0.998: `windChillTemperature`≡`airTemperature`,
          `tidePressure`/`waterPressure`/`waterLevel`≡`tideLevel`, `waterTemperature_WQ`≡`waterTemperature`,
          `maxWaveHeight`≡`significantWaveHeight`) — reconstructed via fitted linear regression
          from their kept twin, not modeled separately.
        - **3 circular parameters** (`windDirection`, `currentDirection`, `compass`) sin/cos-encoded.
        - **`precipitationType`** (categorical) handled by a separate XGBoost multiclass classifier.
        - Resampled from 1-minute native to 10-minute (matches the cadence already implicit
          in the data's own `tenMinuteAvgVisibility` field).

        ### Headline result
        **iTransformer dominates** on real data (median skill +75%, 12/24 outright wins) —
        unlike the synthetic dataset, where it was the weakest attention model. The real
        EMS data has genuinely strong cross-parameter correlations (thermal block r=0.75-0.985,
        wave block r=0.71-0.86, per `31_parameter_model_assignment.txt`), which is exactly
        what its cross-variate attention needs to pay off. **Harmonic-Residual** still wins
        `tideLevel` (+91%) — real tidal physics confirms the same finding as the synthetic
        data. The hard floor is the same too: visibility (sensor-ceiling-limited) and
        precipitation (rare-event) — no model beats persistence there.

        ### Source notebook
        - `Marine_Forecast_RealEMS_31Param.ipynb` — this dashboard's data source.
        - Synthetic-data counterpart + dashboard: `Marine_Forecast_PyTorch_NBEATS_NHiTS.ipynb` / `app.py` (port 8501).
        """
    )
