"""
Marine 5-Day Forecast — Results Dashboard
==========================================
Streamlit viewer for the forecasts/metrics produced by
Marine_Forecast_PyTorch_NBEATS_NHiTS.ipynb (PyTorch LSTM, XGBoost, N-BEATS, N-HiTS)
and, if present, the original Marine_Forecast_LSTM_XGBoost.ipynb (TensorFlow LSTM, XGBoost).

Run with:
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine 5-Day Forecast Dashboard", layout="wide")

DATA_PATH = "marine_data_75days.csv"
PT_FORECAST_PATH = "forecast_vs_actual_pytorch.csv"
PT_METRICS_PATH = "metrics_summary_pytorch.csv"
TF_FORECAST_PATH = "forecast_vs_actual.csv"
TF_METRICS_PATH = "metrics_summary.csv"
BENCHMARK_PATH = "compute_benchmark.csv"
BEST_PER_PARAM_PATH = "best_model_per_parameter.csv"
ENSEMBLE_REC_PATH = "ensemble_recommendation.csv"
ENSEMBLE_FVA_PATH = "ensemble_forecast_vs_actual.csv"
UNCERTAINTY_PATH = "uncertainty_bands.csv"

PT_MODELS = {
    "lstm": "LSTM", "xgboost": "XGBoost", "nbeats": "N-BEATS", "nhits": "N-HiTS",
    "dlinear": "DLinear", "tide": "TiDE", "tsmixer": "TSMixer",
    "harmonicresidual": "Harmonic-Residual", "itransformer": "iTransformer", "patchtst": "PatchTST",
    "deepar": "DeepAR", "gaussianprocess": "Gaussian Process",
}
UQ_MODELS = {"deepar": "DeepAR", "gp": "Gaussian Process"}  # have confidence-interval bands
TF_MODELS = {"lstm": "LSTM (TensorFlow)", "xgb": "XGBoost (TF run)"}

REPORT_PARAMS = [
    "significant_wave_height_m", "wave_period_s", "wind_speed_ms",
    "wind_direction_deg", "tidal_level_m", "current_speed_ms",
    "sea_surface_temp_c", "salinity_psu", "conductivity_mscm",
    "air_pressure_hpa", "air_temp_c", "relative_humidity_pct",
    "dew_point_c", "precipitation_mmh", "solar_radiation_wm2", "visibility_km",
]
UNITS = {
    "significant_wave_height_m": "m", "wave_period_s": "s", "wind_speed_ms": "m/s",
    "wind_direction_deg": "deg", "tidal_level_m": "m", "current_speed_ms": "m/s",
    "sea_surface_temp_c": "°C", "salinity_psu": "PSU", "conductivity_mscm": "mS/cm",
    "air_pressure_hpa": "hPa", "air_temp_c": "°C", "relative_humidity_pct": "%",
    "dew_point_c": "°C", "precipitation_mmh": "mm/h", "solar_radiation_wm2": "W/m²",
    "visibility_km": "km",
}

COLORS = {
    "LSTM": "#1f77b4", "XGBoost": "#d62728", "N-BEATS": "#2ca02c", "N-HiTS": "#9467bd",
    "DLinear": "#ff7f0e", "TiDE": "#8c564b", "TSMixer": "#e377c2", "Harmonic-Residual": "#17becf",
    "iTransformer": "#bcbd22", "PatchTST": "#7f7f7f", "DeepAR": "#ffd700", "Gaussian Process": "#8b0000",
    "LSTM (TensorFlow)": "#aec7e8", "XGBoost (TF run)": "#ffbb78",
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
    "DeepAR": "Probabilistic RNN, 100 Monte-Carlo sample paths (Salinas et al., Amazon)",
    "Gaussian Process": "Bayesian, frozen-period periodic + RBF kernels (GPyTorch)",
}

MODEL_VERDICT = {
    "PatchTST": (
        "**#1 overall.** Median skill **+48%** vs persistence, beats persistence on 69% "
        "of parameters, wins outright on 2 of 16 (wave period, current speed) and is "
        "runner-up on several more. Channel-independent patch attention effectively "
        "trains on 21× more (channel, patch) examples than a model that fits one set of "
        "weights per channel, which matters a lot on this size of dataset. Only 1.2 "
        "ms/forecast (114K params) — the default real-time choice."
    ),
    "DLinear": (
        "Simplest model that's still excellent: 0.11 ms/forecast on a 17.5K-parameter "
        "linear model. Median skill **+40%**, beats persistence on **75%** of parameters "
        "(tied for the best reliability rate), 1 outright win (wave height). If PatchTST's "
        "accuracy edge isn't worth the extra architecture, DLinear remains the simplest "
        "competent choice."
    ),
    "TSMixer": (
        "Median skill **+37%**, beats persistence on **75%** of parameters, wins 3 "
        "outright (wind speed, relative humidity, visibility) — the most outright wins of "
        "any model — via feature-mixing that explicitly learns cross-parameter coupling "
        "(wind→wave, pressure→wind) that channel-independent models (DLinear, PatchTST) "
        "cannot. Still ~1ms inference on a 45K-parameter model."
    ),
    "TiDE": (
        "Median skill **+36%**, 75% beat persistence, at 0.41 ms inference — no outright "
        "wins this run but consistently near the top (2nd-3rd on several parameters). "
        "Justified mainly if you plan to feed it more known-future covariates later (e.g. "
        "NWP model output), since its architecture is built for that."
    ),
    "XGBoost": (
        "The classical baseline holds up: median skill **+14%**, wins solar radiation and "
        "wind direction, battle-tested for production. The real-time catch: it's **17 "
        "independent models**, one per parameter, so total inference (~15 ms) and training "
        "(~53s) are an order of magnitude more deployment overhead than any single global "
        "PyTorch model above, for lower median accuracy."
    ),
    "N-HiTS": (
        "Median skill **+8%**, just over half of parameters beat persistence, **zero** "
        "outright wins — despite a non-trivial 1.05M-parameter footprint. Its multi-rate "
        "pooling (24h/12h/1h) is theoretically well-matched to this domain but "
        "underdelivers at this 120h horizon; worth revisiting at longer horizons."
    ),
    "N-BEATS": (
        "Median skill **+8%**, wins sea surface temp & conductivity outright — the two "
        "hardest parameters in the whole comparison, where every model struggles — but by "
        "far the **largest model here at 4.2M parameters** for that. Still real-time-fast "
        "in absolute terms (0.7 ms, one forward pass)."
    ),
    "Gaussian Process": (
        "Median skill **+16%**, beats persistence on 62% of parameters, no outright wins "
        "but consistently a strong #2-3 (tidal level, wind speed, air temp). The one model "
        "here with **native, calibrated uncertainty** — no sampling needed, just a closed-"
        "form posterior variance. Prediction is cheap once fit (~90ms for all 17 "
        "parameters) but each parameter's GP needs periodic refitting (~32s total, "
        "offline) as new data arrives — budget that into a retraining schedule, not "
        "per-forecast latency."
    ),
    "Harmonic-Residual": (
        "Fastest point-forecast inference of all (0.13 ms) and by far the most accurate "
        "model **for one parameter**: tidal level, at **+94% skill** — clear of every "
        "other model including PatchTST and the GP. But its harmonic basis is tidal-"
        "specific, so median skill across all 16 parameters is only **+2%**. Deploy it "
        "only for `tidal_level_m`, as one arm of a per-parameter ensemble."
    ),
    "iTransformer": (
        "Median skill only **+1%** overall, but don't undersell it: it **wins air "
        "pressure and air temp outright** (+77% and +89% skill respectively) — its cross-"
        "variate attention found real structure there that nothing else did. Zero free "
        "lunch elsewhere though: with only 21 tokens (one per parameter) to attend over, "
        "it's less data-efficient than PatchTST's channel-independent patches on most "
        "other parameters. Fast (1.4 ms, 81K params)."
    ),
    "LSTM": (
        "Median skill **+2%**, just over half of parameters beat persistence. Forecasting "
        "120 hours means **120 sequential forward passes** (recursive rollout) — ~66 ms "
        "per forecast, 50-600x slower than the direct-horizon models — yet its 2 wins "
        "(salinity, dew point) show recursive rollout isn't always fatal, just usually "
        "suboptimal. Keep as the literature-standard reference baseline."
    ),
    "DeepAR": (
        "Wins **precipitation outright** — the single hardest, burstiest parameter in the "
        "dataset — because training on Gaussian likelihood instead of raw MSE produces a "
        "smoother, more robust mean estimate where point-forecast models overfit noise. "
        "But its real-time cost has two faces: a **point-only** forecast (1 sample path, "
        "no uncertainty) costs ~86 ms, similar to plain LSTM; the **full uncertainty-"
        "quantified** forecast (100 Monte-Carlo sample paths, true ancestral sampling) "
        "costs **~9.4 seconds** — by far the most expensive in this comparison. Use the "
        "cheap point-only mode unless you specifically need the growing confidence band, "
        "which is otherwise unavailable from any other model here except the GP."
    ),
}


@st.cache_data
def load_history():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).set_index("timestamp")
    return df


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
def load_forecasts():
    pt = pd.read_csv(PT_FORECAST_PATH, parse_dates=["timestamp"])
    pt_metrics = pd.read_csv(PT_METRICS_PATH)
    tf, tf_metrics = None, None
    try:
        tf = pd.read_csv(TF_FORECAST_PATH, parse_dates=["timestamp"])
        tf_metrics = pd.read_csv(TF_METRICS_PATH)
    except FileNotFoundError:
        pass
    return pt, pt_metrics, tf, tf_metrics


@st.cache_data
def load_benchmark():
    return pd.read_csv(BENCHMARK_PATH)


@st.cache_data
def build_recommendation(pt_metrics, bench_df):
    """Rank models for real-time deployment: accuracy (robust to outlier skills) vs inference latency."""
    rows = []
    for model in PT_MODELS.values():
        skill = pt_metrics[f"{model}_skill_%"]
        median_skill = skill.median()
        pct_beats = (skill > 0).mean() * 100
        wins = int((pt_metrics["best_model"] == model).sum())
        b = bench_df[bench_df["model"] == model].iloc[0]
        composite = median_skill - b["inference_ms"] / 10.0
        rows.append(dict(
            model=model, median_skill=median_skill, pct_beats_persistence=pct_beats, wins=wins,
            inference_ms=b["inference_ms"], params=int(b["params"]), train_time_s=b["train_time_s"],
            forward_passes=int(b["forward_passes_per_forecast"]), composite=composite,
        ))
    rec = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    rec.insert(0, "rank", rec.index + 1)
    return rec


history = load_history()
pt_fva, pt_metrics, tf_fva, tf_metrics = load_forecasts()
bench_df = load_benchmark()
recommendation = build_recommendation(pt_metrics, bench_df)
best_per_param = load_best_per_param()
ensemble_rec, ensemble_fva = load_ensemble()
uncertainty_df = load_uncertainty()

st.title("🌊 Marine 5-Day Forecast — Model Comparison Dashboard")
st.caption(
    "12 models (LSTM · XGBoost · N-BEATS · N-HiTS · DLinear · TiDE · TSMixer · "
    "Harmonic-Residual Hybrid · iTransformer · PatchTST · DeepAR · Gaussian Process) — "
    "forecasting 16 marine/ship-mooring parameters 120 hours (5 days) ahead, validated "
    "against held-out ground truth."
)

with st.sidebar:
    st.header("Controls")
    parameter = st.selectbox(
        "Parameter", REPORT_PARAMS,
        format_func=lambda p: f"{p.replace('_', ' ').title()} ({UNITS.get(p, '')})",
    )
    show_models = st.multiselect(
        "Models to show", list(PT_MODELS.values()),
        default=["XGBoost", "PatchTST", "DLinear", "Harmonic-Residual"],
    )
    show_uncertainty = st.checkbox(
        "Show DeepAR / GP confidence band", value=False,
        help="Shades the ±1.96σ (95%) interval from the Monte-Carlo DeepAR forecast and "
             "the closed-form Gaussian Process posterior. Not available for wind direction "
             "(circular).",
    )
    show_tf = st.checkbox(
        "Overlay original TensorFlow run (LSTM/XGBoost)", value=False,
        disabled=tf_fva is None,
        help="Requires forecast_vs_actual.csv from Marine_Forecast_LSTM_XGBoost.ipynb",
    )
    history_hours = st.slider("History context (hours)", 24, 240, 72, step=24)

tab_forecast, tab_metrics, tab_realtime, tab_winners, tab_about = st.tabs(
    ["📈 Forecast", "📊 Metrics & Skill", "🚀 Best for Real-Time", "🏅 Winners & Ensembles", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
with tab_forecast:
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure()
        hist_tail = history[parameter].iloc[-(120 + history_hours):-120]
        fig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                                  line=dict(color="lightgray", width=1.5)))
        fig.add_trace(go.Scatter(x=pt_fva["timestamp"], y=pt_fva[f"{parameter}__actual"],
                                  name="actual", line=dict(color="black", width=3)))
        for key, label in PT_MODELS.items():
            if label in show_models:
                col = f"{parameter}__{key}"
                if col in pt_fva.columns:
                    fig.add_trace(go.Scatter(x=pt_fva["timestamp"], y=pt_fva[col], name=label,
                                              line=dict(color=COLORS[label], width=2, dash="dash")))
        if show_tf and tf_fva is not None:
            for key, label in TF_MODELS.items():
                col = f"{parameter}__{key}"
                if col in tf_fva.columns:
                    fig.add_trace(go.Scatter(x=tf_fva["timestamp"], y=tf_fva[col], name=label,
                                              line=dict(color=COLORS[label], width=1.5, dash="dot")))
        if show_uncertainty and uncertainty_df is not None and parameter != "wind_direction_deg":
            for prefix, label, fill_rgba in [("deepar", "DeepAR", "255,215,0"), ("gp", "Gaussian Process", "139,0,0")]:
                mean_col, std_col = f"{parameter}__{prefix}_mean", f"{parameter}__{prefix}_std"
                if mean_col in uncertainty_df.columns and label in show_models:
                    m, s = uncertainty_df[mean_col], uncertainty_df[std_col]
                    fig.add_trace(go.Scatter(x=uncertainty_df["timestamp"], y=m + 1.96 * s,
                                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=uncertainty_df["timestamp"], y=m - 1.96 * s,
                                              fill="tonexty", fillcolor=f"rgba({fill_rgba},0.15)",
                                              line=dict(width=0), name=f"{label} 95% CI", hoverinfo="skip"))
        fig.add_vline(x=pt_fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
        fig.update_layout(
            title=f"{parameter.replace('_', ' ').title()} — 5-Day Forecast vs Actual",
            xaxis_title="Time", yaxis_title=f"{parameter} ({UNITS.get(parameter, '')})",
            height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Scorecard")
        row = pt_metrics[pt_metrics["parameter"] == parameter].iloc[0]
        st.metric("Best model", row["best_model"])
        st.metric("Persistence MAE", f"{row['Persistence_MAE']:.3f} {UNITS.get(parameter, '')}")
        ranked = sorted(PT_MODELS.values(), key=lambda lbl: row[f"{lbl}_MAE"])
        for label in ranked:
            skill_col = f"{label}_skill_%"
            mae_col = f"{label}_MAE"
            star = " 🏆" if label == row["best_model"] else ""
            st.metric(f"{label} MAE{star}",
                      f"{row[mae_col]:.3f} {UNITS.get(parameter, '')}",
                      delta=f"{row[skill_col]:+.1f}% vs persistence")

# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Full metrics table — all parameters, all models")
    st.dataframe(pt_metrics, use_container_width=True, hide_index=True)

    st.subheader("Forecast skill vs persistence (%)")
    skill_long = pt_metrics.melt(
        id_vars=["parameter"],
        value_vars=[c for c in pt_metrics.columns if c.endswith("_skill_%")],
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
    st.bar_chart(pt_metrics["best_model"].value_counts())

# ---------------------------------------------------------------------------
with tab_realtime:
    st.subheader("Which model should run in production, in real time?")
    st.markdown(
        "Ranked by a **composite real-time suitability score** = "
        "`median forecast skill vs persistence (%)` − `inference latency (ms) / 10`. "
        "Median (not mean) skill is used because a few parameters have extreme outlier "
        "skill values that would otherwise dominate the average — median reflects "
        "*typical* performance across all 16 parameters. The latency penalty is small for "
        "every model except the LSTM, where it dominates because forecasting 120 hours "
        "needs 120 sequential forward passes (recursive rollout) instead of one."
    )

    for _, r in recommendation.iterrows():
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(int(r["rank"]), f"#{int(r['rank'])}")
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([0.6, 2, 1.4, 1.4, 1.4])
            with c1:
                st.markdown(f"## {medal}")
            with c2:
                st.markdown(f"### {r['model']}")
                st.caption(MODEL_INFO.get(r["model"], ""))
                st.metric("Composite score", f"{r['composite']:+.1f}")
            with c3:
                st.metric("Median skill vs persistence", f"{r['median_skill']:+.1f}%")
                st.metric("Beats persistence on", f"{r['pct_beats_persistence']:.0f}% of params")
            with c4:
                st.metric("Inference latency / forecast", f"{r['inference_ms']:.2f} ms")
                st.metric("Forward passes / forecast", int(r["forward_passes"]))
            with c5:
                st.metric("Outright parameter wins", f"{int(r['wins'])} / 16")
                st.metric("Params · train time", f"{r['params']:,} · {r['train_time_s']:.1f}s")
            st.markdown(MODEL_VERDICT.get(r["model"], ""))

    st.divider()
    st.subheader("Full comparison table")
    show_cols = ["rank", "model", "composite", "median_skill", "pct_beats_persistence",
                 "wins", "inference_ms", "forward_passes", "params", "train_time_s"]
    st.dataframe(
        recommendation[show_cols].rename(columns={
            "median_skill": "median_skill_%", "pct_beats_persistence": "beats_persistence_%",
            "inference_ms": "inference_ms_per_forecast", "forward_passes": "forward_passes_per_forecast",
            "train_time_s": "train_time_s_total",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Inference latency and parameter counts measured on CPU (same hardware/environment "
        "as training). XGBoost trains/infers one model per parameter (17 models); DeepAR's "
        "latency reflects the full 100-Monte-Carlo-sample uncertainty band (a point-only "
        "single sample path costs ~86ms, comparable to LSTM); Gaussian Process latency "
        "reflects prediction only — its periodic refit (~32s for all 17 parameters) is a "
        "separate, offline cost. All other models are single global multivariate models. "
        "Training time is an offline, one-time (or periodic-retrain) cost and does not "
        "affect real-time inference suitability — shown for completeness."
    )
    st.info(
        "💡 **Practical recommendation**: use **PatchTST** (or DLinear if you want the "
        "simplest possible model) as the default real-time forecaster for most "
        "parameters, **TSMixer** where cross-parameter coupling matters, **iTransformer** "
        "specifically for air pressure/air temp, swap in the **Harmonic-Residual Hybrid** "
        "for `tidal_level_m`, and add the **Gaussian Process** (cheap to query once fit) "
        "wherever a calibrated confidence interval matters more than the last few points "
        "of point-accuracy. Reserve **DeepAR** for precipitation specifically, and run it "
        "in point-only mode (skip the 100-sample MC rollout) unless its uncertainty band "
        "is actually needed downstream."
    )

# ---------------------------------------------------------------------------
with tab_winners:
    st.subheader("🏅 Best single model per parameter — strongest win first")
    st.markdown(
        "Every parameter's single best model (lowest MAE), sorted by **how decisively** "
        "it wins — its skill vs. persistence. The top rows are where one model is clearly "
        "the right specialist; the bottom rows are where every model struggles "
        "(no model beats the naive baseline by much, if at all)."
    )
    disp = best_per_param.copy()
    disp["parameter"] = disp["parameter"].str.replace("_", " ").str.title()
    disp = disp.rename(columns={"best_skill_%": "best_model_skill_%"})
    st.dataframe(
        disp[["rank", "parameter", "best_model", "best_model_skill_%"]],
        use_container_width=True, hide_index=True,
    )

    fig3 = go.Figure(go.Bar(
        x=best_per_param["best_skill_%"], y=best_per_param["parameter"].str.replace("_", " ").str.title(),
        orientation="h",
        marker_color=[COLORS.get(m, "#888") for m in best_per_param["best_model"]],
        text=best_per_param["best_model"], textposition="outside",
    ))
    fig3.update_layout(
        height=600, xaxis_title="Winning model's skill vs persistence (%)",
        title="Top performer at top — bar color/label shows which model won",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("🤝 Two-model ensemble opportunities")
    st.markdown(
        "For every parameter, all **66 possible pairs** of the 12 models were combined with "
        "a plain 50/50 average (circular average for wind direction) and checked against "
        "the single best model. **No weights are fit on the test data** — a 50/50 blend has "
        "zero free parameters, so any improvement shown here is a genuine, leakage-free "
        "signal that the two models' errors partially cancel, not test-set overfitting. "
        "Ensembling is only *recommended* where it actually wins — adding a second model's "
        "inference cost for no accuracy gain fails the practicality test."
    )

    n_rec = int(ensemble_rec["recommended_ensemble"].sum())
    st.metric("Parameters where a 2-model ensemble beats the single best model",
              f"{n_rec} / {len(ensemble_rec)}")

    erec_disp = ensemble_rec.copy()
    erec_disp["parameter"] = erec_disp["parameter"].str.replace("_", " ").str.title()
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
        format_func=lambda p: f"{p.replace('_', ' ').title()} ({UNITS.get(p, '')})",
    )
    erow = ensemble_rec[ensemble_rec["parameter"] == ens_param].iloc[0]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=ensemble_fva["timestamp"], y=ensemble_fva[f"{ens_param}__actual"],
                               name="actual", line=dict(color="black", width=3)))
    single_col = f"{ens_param}__{[k for k, v in PT_MODELS.items() if v == erow['single_best_model']][0]}"
    fig4.add_trace(go.Scatter(x=pt_fva["timestamp"], y=pt_fva[single_col],
                               name=f"single best ({erow['single_best_model']})",
                               line=dict(color="#d62728", width=2, dash="dash")))
    fig4.add_trace(go.Scatter(x=ensemble_fva["timestamp"], y=ensemble_fva[f"{ens_param}__ensemble"],
                               name=f"ensemble ({erow['best_ensemble_pair']})",
                               line=dict(color="#1f77b4", width=2, dash="dot")))
    verdict = "✅ ensemble recommended" if erow["recommended_ensemble"] else "❌ single model recommended"
    fig4.update_layout(
        title=f"{ens_param.replace('_', ' ').title()} — {verdict}",
        xaxis_title="Time", yaxis_title=f"{ens_param} ({UNITS.get(ens_param, '')})",
        height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=80),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        f"Single best ({erow['single_best_model']}) MAE = {erow['single_best_MAE']:.3f} vs. "
        f"ensemble ({erow['best_ensemble_pair']}) MAE = {erow['ensemble_MAE']:.3f} "
        f"({'improvement' if erow['recommended_ensemble'] else 'no improvement'} of "
        f"{abs(erow['improvement_vs_single_best']):.3f} {UNITS.get(ens_param, '')})."
    )

# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### What this is
        A results viewer for a **PyTorch port** of the marine ship-mooring 5-day
        forecasting pipeline, comparing **12 models** trained to predict 16 marine
        parameters 120 hours ahead:

        | Model | Type | Horizon strategy |
        |---|---|---|
        | **LSTM** | PyTorch RNN | Recursive rollout (1-step model rolled forward 120×) |
        | **XGBoost** | Gradient-boosted trees | Direct multi-horizon (one model/parameter) |
        | **N-BEATS** | FC, doubly-residual stacks (Oreshkin et al. 2020) | Direct multi-horizon |
        | **N-HiTS** | Multi-rate pooling + hierarchical interpolation (Challu et al. 2023) | Direct multi-horizon |
        | **DLinear** | Single linear layer, trend/seasonal decomposition (Zeng et al., AAAI 2023) | Direct multi-horizon |
        | **TiDE** | Dense encoder-decoder + known covariates (Das et al., Google, TMLR 2023) | Direct multi-horizon |
        | **TSMixer** | All-MLP time-mixing + feature-mixing (Chen et al., Google, 2023) | Direct multi-horizon |
        | **Harmonic-Residual** | Tidal/diurnal harmonic regression + residual MLP (domain-specific) | Direct multi-horizon |
        | **iTransformer** | Variate-as-token attention (Liu et al., 2023) | Direct multi-horizon |
        | **PatchTST** | Patches + channel-independent attention (Nie et al., ICLR 2023) | Direct multi-horizon |
        | **DeepAR** | Probabilistic RNN, Gaussian likelihood (Salinas et al., Amazon) | 100 Monte-Carlo sample paths |
        | **Gaussian Process** | Bayesian, frozen-period periodic + RBF kernels (GPyTorch) | Closed-form posterior, per-parameter |

        Everything except the LSTM and DeepAR forecasts the entire 120-hour horizon in a
        single forward pass, so only those two can suffer recursive error compounding.

        **Why these particular additions?** DLinear is the field's standard sanity-check
        baseline (a single linear layer that famously beats many Transformers on
        long-horizon benchmarks). TiDE and TSMixer are 2023 all-MLP architectures that
        are currently near the top of long-horizon forecasting leaderboards without
        Transformer-level compute cost. The Harmonic-Residual hybrid directly implements
        the technique `MARINE_FORECASTING_IMPLEMENTATION_GUIDE.md` recommends specifically
        for **tidal level** — and it wins that parameter by a wide margin (≈94% skill vs
        persistence), validating the literature review against this dataset.

        **iTransformer and PatchTST** were added after reviewing `latest_research.txt`
        (a 2023-2026 survey of multivariate time-series forecasting research) — they fill
        the attention-based gap the other 8 models left. **PatchTST is the best model
        overall** (median skill +48%, channel-independent patches effectively train on
        21× more examples than any per-channel model). iTransformer's cross-variate
        attention has only 21 tokens to learn from per window — weaker on median skill,
        but it **wins air pressure and air temp outright**, finding cross-parameter
        structure nothing else captured.

        **DeepAR and Gaussian Process** were added after reviewing `ML models_few_more.txt`
        (a survey covering foundation models, weather/climate-specific architectures, GPs,
        Neural ODEs, PINNs, and spatio-temporal GNNs). Most of that survey doesn't fit a
        single-buoy tabular problem (GraphCast/Pangu/FourCastNet/Neural Operators need a
        spatial grid this data doesn't have; Chronos/Moirai/Lag-Llama are pretrained
        foundation models out of scope for from-scratch training) — DeepAR and GP are the
        two pieces that *do* fit, and both add **uncertainty quantification**, which none
        of models 1-10 provide. DeepAR wins precipitation outright (Gaussian-likelihood
        training is more robust to bursty noise than MSE); its full 100-sample uncertainty
        band costs ~9.4s, vastly more than a point-only forecast (~86ms) — see the
        🚀 Best for Real-Time tab. The GP gives calibrated intervals natively and is
        consistently a strong #2-3 on periodic parameters (tide, wind speed, air temp),
        cheap to query once fit.

        ### Winners & Ensembles tab
        Ranks every parameter by how decisively its single best model wins, then tests
        all 66 possible 2-model averages per parameter to see whether combining models
        beats the single best one — using a plain unweighted average (no weights fit on
        test data, so no leakage) and only recommending the ensemble where it genuinely
        wins. See `README.md` §7 for the full numbers and reasoning.

        ### A note on the compute environment
        An earlier bug caused notebook execution to silently run under base Anaconda's
        Python rather than the `marinepred` conda env this project specifies (a Jupyter
        kernel/launcher resolution issue, now fixed). The results shown here are from the
        first run confirmed to genuinely execute inside `marinepred` (verified via printed
        `torch`/`xgboost` versions in the notebook) — small numeric differences from any
        earlier session are expected and are an environment artifact, not a code change to
        models 1-10.

        ### Data
        75 days of physically-realistic synthetic hourly marine data (tidal harmonics,
        diurnal cycles, storm fronts, cross-parameter coupling) — see
        `generate_marine_data.py`. Last 5 days held out for validation.

        ### Source notebooks
        - `Marine_Forecast_PyTorch_NBEATS_NHiTS.ipynb` — this dashboard's data source.
        - `Marine_Forecast_LSTM_XGBoost.ipynb` — original TensorFlow baseline (optional overlay).
        """
    )
