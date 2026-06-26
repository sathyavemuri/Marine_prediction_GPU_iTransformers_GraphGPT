"""
Marine 48h Forecast — Single iTransformer Deep-Dive Dashboard
================================================================
Streamlit viewer for Marine_Forecast_RealEMS_iTransformer_Only.ipynb — one iTransformer
model, all 24 directly-modeled real parameters in AND out, plus duplicate-parameter
reconstruction and cross-variate attention-map interpretability.

Run with:
    streamlit run app_itransformer.py --server.port 8503
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine Forecast — iTransformer Deep-Dive", layout="wide")

HISTORY_PATH = "ems_10min_resampled.csv"
FORECAST_PATH = "forecast_vs_actual_itransformer_only.csv"
METRICS_PATH = "metrics_itransformer_only.csv"
DUP_RECON_PATH = "duplicate_reconstruction_itransformer_only.csv"
DUP_FVA_PATH = "duplicate_forecast_vs_actual_itransformer_only.csv"
ATTENTION_PATH = "attention_weights_itransformer_only.csv"

HORIZON_STEPS = 288  # 48h @ 10-min steps
LOOKBACK_STEPS = 288

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
DUPLICATE_PARAMS = [
    "windChillTemperature", "tidePressure", "waterPressure",
    "waterLevel", "waterTemperature_WQ", "maxWaveHeight",
]
ALL_FORECAST_PARAMS = REPORT_PARAMS + DUPLICATE_PARAMS
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

DUPLICATES = [
    ("airTemperature", "windChillTemperature"),
    ("tideLevel", "tidePressure"),
    ("tideLevel", "waterPressure"),
    ("tideLevel", "waterLevel"),
    ("waterTemperature", "waterTemperature_WQ"),
    ("significantWaveHeight", "maxWaveHeight"),
]
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
    recon = pd.read_csv(DUP_RECON_PATH)
    fva = pd.read_csv(DUP_FVA_PATH, parse_dates=["timestamp"])
    return recon, fva


@st.cache_data
def load_attention():
    return pd.read_csv(ATTENTION_PATH, index_col=0)


history = load_history()
fva = load_forecast()
metrics = load_metrics()
dup_recon, dup_fva = load_duplicates()
attn = load_attention()

st.title("🧠 Marine 48-Hour Forecast — Single iTransformer Deep-Dive")
st.caption(
    "One iTransformer model takes all 24 directly-modeled real parameters as input "
    "tokens and predicts all 24 as output (the same model that won 12/24 parameters in "
    "the 11-model comparison). This view adds two things the bake-off doesn't: "
    "**duplicate-parameter reconstruction plots** and a **cross-variate attention map** "
    "showing which parameters the model relies on when forecasting each other."
)

tab_forecast, tab_dup, tab_attn, tab_metrics, tab_about = st.tabs(
    ["📈 Forecast", "🔁 Duplicate Reconstruction", "🧠 Attention Map", "📊 Metrics", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
with tab_forecast:
    col1, col2 = st.columns([3, 1])
    with col1:
        parameter = st.selectbox(
            "Parameter", ALL_FORECAST_PARAMS,
            format_func=lambda p: (f"{p} ({DUP_UNITS.get(p, '')}) — reconstructed"
                                    if p in DUPLICATE_PARAMS else f"{p} ({UNITS.get(p, '')})"),
        )
        is_duplicate = parameter in DUPLICATE_PARAMS
        history_hours = st.slider("History context (hours)", 12, 240, 96, step=12)
        history_steps = history_hours * 6
        hist_tail = history[parameter].iloc[-(HORIZON_STEPS + history_steps):-HORIZON_STEPS]
        units = DUP_UNITS.get(parameter, "") if is_duplicate else UNITS.get(parameter, "")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                                  line=dict(color="lightgray", width=1.5)))
        if is_duplicate:
            fig.add_trace(go.Scatter(x=dup_fva["timestamp"], y=dup_fva[f"{parameter}__actual"],
                                      name="actual", line=dict(color="black", width=3)))
            fig.add_trace(go.Scatter(x=dup_fva["timestamp"], y=dup_fva[f"{parameter}__reconstructed"],
                                      name="reconstructed", line=dict(color="#d62728", width=2, dash="dash")))
            vline_x = dup_fva["timestamp"].iloc[0]
        else:
            fig.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__actual"],
                                      name="actual", line=dict(color="black", width=3)))
            fig.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__itransformer"],
                                      name="iTransformer", line=dict(color="#bcbd22", width=2, dash="dash")))
            vline_x = fva["timestamp"].iloc[0]
        fig.add_vline(x=vline_x, line_color="green", line_width=1, opacity=0.5)
        fig.update_layout(
            title=f"{parameter} — 48-Hour Forecast vs Actual" + (" (reconstructed)" if is_duplicate else ""),
            xaxis_title="Time", yaxis_title=f"{parameter} ({units})",
            height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Scorecard")
        if is_duplicate:
            row = dup_recon[dup_recon["duplicate_parameter"] == parameter].iloc[0]
            st.metric("Reconstructed from", row["reconstructed_from"])
            st.metric("Train R²", f"{row['train_R2']:.5f}")
            st.metric("Held-out MAE", f"{row['held_out_MAE']:.4f} {units}")
            st.metric("Held-out RMSE", f"{row['held_out_RMSE']:.4f} {units}")
            st.info(
                f"Not modeled directly — reconstructed as "
                f"`{row['slope']}×{row['reconstructed_from']} + {row['intercept']}` from "
                f"iTransformer's forecast of **{row['reconstructed_from']}**. Accuracy is "
                f"entirely downstream of that twin's own forecast error, scaled by the slope."
            )
        else:
            row = metrics[metrics["parameter"] == parameter].iloc[0]
            st.metric("Rank (of 24)", int(row["rank"]))
            st.metric("Persistence MAE", f"{row['Persistence_MAE']:.4f} {units}")
            st.metric("iTransformer MAE", f"{row['iTransformer_MAE']:.4f} {units}",
                       delta=f"{row['iTransformer_skill_%']:+.1f}% vs persistence")
            if row["iTransformer_skill_%"] < 0:
                st.warning("Negative skill — this model doesn't beat naive persistence here. "
                           "Same hard-floor parameters (visibility, precipitation) as every "
                           "other model tested in this project.")
            elif row["iTransformer_skill_%"] > 80:
                st.success("Strong result — among the best skill scores in this entire project.")

# ---------------------------------------------------------------------------
with tab_dup:
    st.subheader("🔁 The 6 duplicate parameters, reconstructed")
    st.markdown(
        "These parameters weren't modeled directly — they're near-perfect linear "
        "functions of another parameter already being forecast (r ≥ 0.998), so they're "
        "**reconstructed** as `slope × kept_twin's_forecast + intercept` instead. "
        "Reconstruction error here is bounded by two things: how tight the linear fit is "
        "(`train_R2`) *and* how good the kept twin's own 48h forecast is — any error in "
        "forecasting the kept parameter passes straight through to the reconstruction."
    )
    st.dataframe(dup_recon, use_container_width=True, hide_index=True)

    dup_param = st.selectbox(
        "Inspect a duplicate parameter", [d for _, d in DUPLICATES],
        format_func=lambda p: f"{p} ({DUP_UNITS.get(p, '')})",
    )
    row = dup_recon[dup_recon["duplicate_parameter"] == dup_param].iloc[0]

    hist_tail = history[dup_param].iloc[-(HORIZON_STEPS + 288):-HORIZON_STEPS]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                               line=dict(color="lightgray", width=1.5)))
    fig2.add_trace(go.Scatter(x=dup_fva["timestamp"], y=dup_fva[f"{dup_param}__actual"],
                               name="actual", line=dict(color="black", width=3)))
    fig2.add_trace(go.Scatter(x=dup_fva["timestamp"], y=dup_fva[f"{dup_param}__reconstructed"],
                               name=f"reconstructed (from {row['reconstructed_from']})",
                               line=dict(color="#d62728", width=2, dash="dash")))
    fig2.add_vline(x=dup_fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
    fig2.update_layout(
        title=f"{dup_param} — Actual vs. Reconstructed "
              f"(train R²={row['train_R2']:.5f}, formula: {row['slope']}×{row['reconstructed_from']}+{row['intercept']})",
        xaxis_title="Time", yaxis_title=f"{dup_param} ({DUP_UNITS.get(dup_param, '')})",
        height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=100),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        f"Held-out reconstruction MAE = {row['held_out_MAE']:.4f} {DUP_UNITS.get(dup_param, '')}, "
        f"RMSE = {row['held_out_RMSE']:.4f} {DUP_UNITS.get(dup_param, '')}."
    )

# ---------------------------------------------------------------------------
with tab_attn:
    st.subheader("🧠 Cross-variate attention map")
    st.markdown(
        "iTransformer's defining trick: each **parameter** (not each timestep) becomes a "
        "token, and self-attention runs **across parameters**. This heatmap is the "
        "trained model's actual layer-1 attention weights — row = the parameter being "
        "predicted (query), column = the parameter it's attending to (key). Brighter = "
        "more influence. The diagonal (self-attention) is usually strong since a "
        "parameter's own history is normally its best predictor; the interesting signal "
        "is the **off-diagonal** brightness — real learned cross-parameter dependencies."
    )

    fig3 = go.Figure(data=go.Heatmap(
        z=attn.values, x=attn.columns.tolist(), y=attn.index.tolist(),
        colorscale="Viridis", colorbar=dict(title="attention"),
    ))
    fig3.update_layout(
        height=800, xaxis_title="attends TO (key)", yaxis_title="query parameter (being predicted)",
        xaxis=dict(tickfont=dict(size=8)), yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Top-3 attended-to parameters, per query parameter")
    query_param = st.selectbox("Query parameter", attn.index.tolist(), key="attn_query")
    top3 = attn.loc[query_param].drop(query_param, errors="ignore").sort_values(ascending=False).head(5)
    st.bar_chart(top3)
    st.caption(
        f"What **{query_param}**'s forecast relies on most, besides its own history "
        f"(self-attention weight = {attn.loc[query_param, query_param]:.3f})."
    )

# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Full metrics — all 24 parameters, ranked")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    fig4 = go.Figure(go.Bar(
        x=metrics["iTransformer_skill_%"], y=metrics["parameter"], orientation="h",
        marker_color=["#2ca02c" if v > 0 else "#d62728" for v in metrics["iTransformer_skill_%"]],
    ))
    fig4.add_vline(x=0, line_color="black")
    fig4.update_layout(
        height=700, xaxis_title="Skill vs persistence (%)",
        title="Higher is better — red bars mean this model loses to naive persistence",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    n_positive = int((metrics["iTransformer_skill_%"] > 0).sum())
    st.metric("Parameters beating persistence", f"{n_positive} / {len(metrics)}")
    st.metric("Median skill", f"{metrics['iTransformer_skill_%'].median():.1f}%")

# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### What this is
        A standalone, single-model deep-dive: **one iTransformer** (Liu et al., 2023)
        takes all 24 directly-modeled real EMS parameters as input and predicts all 24
        as output, forecasting **48 hours ahead** at 10-minute resolution. This is the
        same model and architecture from the 11-model comparison
        (`Marine_Forecast_RealEMS_31Param.ipynb`), isolated here for closer inspection —
        it does **not** modify or depend on that notebook, `app.py`, or `app_realdata.py`.

        ### Why iTransformer specifically?
        In the 11-model comparison it won 12 of 24 parameters outright and posted the
        highest median skill (+75.3%) of any model — by a wide margin over the runner-up
        (PatchTST, +68.0%). Real EMS data has genuinely strong cross-parameter
        correlations (thermal block r=0.75-0.985, wave block r=0.71-0.86, per
        `31_parameter_model_assignment.txt`), which is exactly the structure its
        cross-variate attention is built to exploit.

        ### What's new in this notebook vs. the 11-model comparison
        - **Duplicate reconstruction plots**: the 6 collapsed parameters
          (`windChillTemperature`, `tidePressure`, `waterPressure`, `waterLevel`,
          `waterTemperature_WQ`, `maxWaveHeight`) get actual-vs-reconstructed time series
          plots, not just a one-line MAE summary.
        - **Attention map**: the trained model's actual cross-variate attention weights,
          extracted via a parallel forward pass through the first encoder layer's
          `self_attn` with `need_weights=True` (the fast path used during normal
          training/inference skips weight computation for speed).

        ### Why these 24 parameters, not all 31?
        See the 11-model dashboard's About tab / README for the full explanation: 6 are
        collapsed duplicates (near-perfectly correlated with another modeled parameter,
        r ≥ 0.998) and 1 (`precipitationType`) is categorical, handled by a separate
        XGBoost classifier in the 11-model notebook, not relevant to a single-regression-model deep-dive.

        ### Source notebook
        `Marine_Forecast_RealEMS_iTransformer_Only.ipynb`
        """
    )
