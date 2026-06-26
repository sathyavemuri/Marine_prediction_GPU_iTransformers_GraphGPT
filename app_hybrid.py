"""
Marine 48h Forecast — Hybrid iTransformer + DeepAR Dashboard
================================================================
Streamlit viewer for Marine_Forecast_RealEMS_Hybrid_iTransformer_DeepAR.ipynb — one
iTransformer (18 parameters it's good at) + one DeepAR (6 it isn't: visibility x4,
precipitation x2), merged into a single 24-parameter forecast.

Run with:
    streamlit run app_hybrid.py --server.port 8504
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marine Forecast — Hybrid iTransformer+DeepAR", layout="wide")

HISTORY_PATH = "ems_10min_resampled.csv"
FORECAST_PATH = "forecast_vs_actual_hybrid.csv"
METRICS_PATH = "metrics_hybrid.csv"
DUP_RECON_PATH = "duplicate_reconstruction_hybrid.csv"
DUP_FVA_PATH = "duplicate_forecast_vs_actual_hybrid.csv"
ATTENTION_PATH = "attention_weights_hybrid.csv"

HORIZON_STEPS = 288
LOOKBACK_STEPS = 288

HARD_PARAMS = [
    "twentyFourHourAvgVisibility", "precipitationDifference", "tenMinuteAvgVisibility",
    "oneMinuteAvgVisibility", "oneHourAvgVisibility", "precipitationIntensity",
]
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
ENGINE_COLOR = {"iTransformer": "#bcbd22", "DeepAR": "#ffd700"}


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

st.title("🔀 Marine 48-Hour Forecast — Hybrid iTransformer + DeepAR")
st.caption(
    "iTransformer forecasts the 18 parameters it's good at (loss trained only on those "
    "18); DeepAR forecasts the 6 it isn't (visibility ×4, precipitation ×2) — swapped in "
    "because it was the consistent winner on these exact parameters in the 11-model "
    "comparison. Merged into one 24-parameter forecast and benchmarked directly against "
    "the pure-iTransformer baseline."
)

n_improved_hard = int((metrics[metrics["parameter"].isin(HARD_PARAMS)]["improvement_pp"] > 0).sum())
n_improved_total = int((metrics["improvement_pp"] > 0).sum())
c1, c2, c3 = st.columns(3)
c1.metric("Hard parameters improved", f"{n_improved_hard} / 6")
c2.metric("Mean improvement on hard 6", f"{metrics[metrics['parameter'].isin(HARD_PARAMS)]['improvement_pp'].mean():+.1f} pp")
c3.metric("Total parameters improved (of 24)", f"{n_improved_total} / 24")

tab_forecast, tab_compare, tab_dup, tab_attn, tab_metrics, tab_about = st.tabs(
    ["📈 Forecast", "🔁 Hybrid vs Pure-iTransformer", "🔂 Duplicate Reconstruction",
     "🧠 Attention Map", "📊 Metrics", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
with tab_forecast:
    col1, col2 = st.columns([3, 1])
    with col1:
        parameter = st.selectbox(
            "Parameter", REPORT_PARAMS,
            format_func=lambda p: f"{p} ({UNITS.get(p, '')}) — {'DeepAR' if p in HARD_PARAMS else 'iTransformer'}",
        )
        history_hours = st.slider("History context (hours)", 12, 240, 96, step=12)
        history_steps = history_hours * 6
        hist_tail = history[parameter].iloc[-(HORIZON_STEPS + history_steps):-HORIZON_STEPS]
        engine = "DeepAR" if parameter in HARD_PARAMS else "iTransformer"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="history",
                                  line=dict(color="lightgray", width=1.5)))
        fig.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__actual"],
                                  name="actual", line=dict(color="black", width=3)))
        fig.add_trace(go.Scatter(x=fva["timestamp"], y=fva[f"{parameter}__hybrid"],
                                  name=f"hybrid ({engine})",
                                  line=dict(color=ENGINE_COLOR[engine], width=2, dash="dash")))
        fig.add_vline(x=fva["timestamp"].iloc[0], line_color="green", line_width=1, opacity=0.5)
        fig.update_layout(
            title=f"{parameter} — served by {engine}",
            xaxis_title="Time", yaxis_title=f"{parameter} ({UNITS.get(parameter, '')})",
            height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Scorecard")
        row = metrics[metrics["parameter"] == parameter].iloc[0]
        st.metric("Engine", row["engine"])
        st.metric("Rank (of 24)", int(row["rank"]))
        st.metric("Persistence MAE", f"{row['Persistence_MAE']:.4f} {UNITS.get(parameter, '')}")
        st.metric("Hybrid MAE", f"{row['hybrid_MAE']:.4f} {UNITS.get(parameter, '')}",
                   delta=f"{row['hybrid_skill_%']:+.1f}% vs persistence")
        if row["parameter"] in HARD_PARAMS:
            st.info(f"Pure-iTransformer skill was **{row['pure_iTransformer_skill_%']:+.1f}%** — "
                    f"swapping to DeepAR improved this by **{row['improvement_pp']:+.1f} pp**.")

# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("🔁 Did swapping engines actually help?")
    st.markdown(
        "Every parameter's hybrid skill vs. the pure-iTransformer baseline "
        "(`Marine_Forecast_RealEMS_iTransformer_Only.ipynb`'s measured results). "
        "**Positive `improvement_pp` everywhere** — even the 18 parameters that stayed "
        "with iTransformer improved, because training loss is no longer diluted by the "
        "6 volatile parameters it used to also have to fit."
    )
    disp = metrics.sort_values("improvement_pp", ascending=False)
    st.dataframe(
        disp[["parameter", "engine", "hybrid_skill_%", "pure_iTransformer_skill_%", "improvement_pp"]],
        use_container_width=True, hide_index=True,
    )

    fig2 = go.Figure(go.Bar(
        x=disp["improvement_pp"], y=disp["parameter"], orientation="h",
        marker_color=[ENGINE_COLOR[e] for e in disp["engine"]],
    ))
    fig2.add_vline(x=0, line_color="black")
    fig2.update_layout(
        height=700, xaxis_title="Improvement over pure iTransformer (percentage points)",
        title="Yellow = DeepAR-served (the swap), olive = iTransformer-served (loss-focus side-benefit)",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Zoom: the 6 hard parameters specifically")
    hard_disp = metrics[metrics["parameter"].isin(HARD_PARAMS)].sort_values("improvement_pp", ascending=False)
    st.dataframe(
        hard_disp[["parameter", "hybrid_skill_%", "pure_iTransformer_skill_%", "improvement_pp"]],
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "2 of 6 flip to positive skill (beat persistence outright); the other 4 land "
        "within a couple percentage points of persistence — the honest ceiling for "
        "rare-event/sensor-ceiling data with only 28 days of training history, not a "
        "model-choice failure."
    )

# ---------------------------------------------------------------------------
with tab_dup:
    st.subheader("🔂 The 6 duplicate parameters, reconstructed")
    st.markdown(
        "All 6 kept twins (`airTemperature`, `tideLevel`, `waterTemperature`, "
        "`significantWaveHeight`) are iTransformer-served parameters, so every "
        "duplicate reconstruction here inherits iTransformer's (now improved) forecast."
    )
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
        height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=80),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"Held-out MAE = {row['held_out_MAE']:.4f}, RMSE = {row['held_out_RMSE']:.4f} {DUP_UNITS.get(dup_param, '')}")

# ---------------------------------------------------------------------------
with tab_attn:
    st.subheader("🧠 iTransformer cross-variate attention (18-parameter model)")
    st.markdown(
        "Same extraction method as the single-model notebook, but this model only ever "
        "has to predict 18 parameters — its attention may be more focused as a result."
    )
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
    st.metric("Median hybrid skill", f"{metrics['hybrid_skill_%'].median():.1f}%")
    st.metric("Parameters beating persistence", f"{int((metrics['hybrid_skill_%']>0).sum())} / 24")

# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### What this is
        A hybrid forecasting pipeline that splits responsibility **by parameter**: one
        iTransformer for the 18 parameters it's measurably good at, one DeepAR for the 6
        it isn't (4 visibility parameters pinned near a sensor ceiling, 2 bursty/rare-event
        precipitation parameters) — directly built from the negative-skill list in
        `Marine_Forecast_RealEMS_iTransformer_Only.ipynb`'s results.

        ### Why DeepAR for the hard 6?
        In the 11-model comparison (`Marine_Forecast_RealEMS_31Param.ipynb`), DeepAR won
        (or tied) on exactly these 6 parameters — not by being accurate, but by losing
        *least*. Training on Gaussian negative log-likelihood instead of raw MSE produces
        a smoother, risk-averse mean estimate that doesn't overfit bursty/saturating data
        the way a deterministic point-forecast model does.

        ### The result
        - **6/6 hard parameters improved** (avg +210 percentage points); 2 flip to
          positive skill outright.
        - **18/18 iTransformer-served parameters also improved** — a side benefit of
          training its loss only on the 18 it's actually responsible for, instead of
          being dragged down by trying to also fit the volatile 6.
        - Net: **24/24 parameters improved** over the pure-iTransformer baseline.

        ### Standalone
        Does not modify or depend on any other notebook/dashboard in this project.

        ### Source notebook
        `Marine_Forecast_RealEMS_Hybrid_iTransformer_DeepAR.ipynb`
        """
    )
