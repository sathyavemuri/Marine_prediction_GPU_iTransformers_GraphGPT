# Marine 5-Day Forecast — PyTorch Models & Real-Time Dashboard

This repo forecasts 16 marine/ship-mooring parameters **120 hours (5 days) ahead** using
**12 models** (1 classical, 11 deep-learning/statistical/Bayesian), validated against
held-out ground truth, and exposes the results through an interactive Streamlit
dashboard. Models 9-10 (iTransformer, PatchTST) were added after reviewing
`latest_research.txt`, a 2023-2026 survey of multivariate time-series forecasting
research, to cover the attention-based architecture family the original 8 models didn't
represent. Models 11-12 (DeepAR, Gaussian Process) were added after reviewing
`ML models_few_more.txt`, a survey covering foundation models, weather/climate-specific
architectures, and Bayesian methods — they add **uncertainty quantification**, which none
of models 1-10 provide (see §2 for why most of that second survey *doesn't* fit this
problem and was deliberately left out).

This README explains **what every number on the dashboard actually means**, and breaks
down **how well each model predicts each marine parameter**, with the numbers that
justify it.

> **Note on the compute environment**: an earlier bug caused notebook execution to
> silently run under base Anaconda's Python instead of the `marinepred` conda env this
> project specifies (a Jupyter kernel/launcher resolution issue — `jupyter nbconvert`
> was dispatching to base Python's site-packages regardless of which env was activated).
> This is now fixed (verified via `torch.__version__`/`xgboost.__version__` printed
> inside the notebook). All numbers below are from the first run confirmed to genuinely
> execute inside `marinepred`. If you compare against an older copy of this README, small
> differences in models 1-10's numbers are this environment fix, not a code change.

- Notebook: `Marine_Forecast_PyTorch_NBEATS_NHiTS.ipynb` (run it to regenerate every CSV/PNG below)
- Dashboard: `app.py` → `streamlit run app.py` → http://localhost:8501
- Data: `marine_data_75days.csv` (75 days hourly, synthetic but physically realistic — see `generate_marine_data.py`)

---

## 1. The 16 marine parameters

| Parameter | Unit | What it is | Why it matters for ship mooring |
|---|---|---|---|
| `significant_wave_height_m` | m | Average height of the highest third of waves | Drives mooring line tension, surge motion; safety threshold for cargo ops |
| `wave_period_s` | s | Time between successive wave crests | Resonance risk — moored ships can resonate with certain wave periods |
| `wind_speed_ms` | m/s | Sustained wind speed | Direct load on hull/superstructure; gust risk during berthing |
| `wind_direction_deg` | ° (circular) | Wind heading (0–360°) | Determines whether wind pushes vessel onto or off the berth |
| `tidal_level_m` | m | Water surface height relative to datum | Determines under-keel clearance, line slack/tension over the tide cycle |
| `current_speed_ms` | m/s | Water current speed | Lateral force on hull; affects approach/departure maneuvering |
| `sea_surface_temp_c` | °C | Sea surface temperature | Affects density/draft, biofouling, and is an operational/environmental indicator |
| `salinity_psu` | PSU | Dissolved salt concentration | Affects water density (draft calculations), corrosion rate |
| `conductivity_mscm` | mS/cm | Electrical conductivity of seawater | Proxy/cross-check for salinity; sensor-derived |
| `air_pressure_hpa` | hPa | Atmospheric pressure | Leading indicator of approaching storms/fronts |
| `air_temp_c` | °C | Air temperature | Crew/equipment safety, icing risk |
| `relative_humidity_pct` | % | Relative humidity | Fog risk, corrosion, cargo sensitivity (e.g. grain, electronics) |
| `dew_point_c` | °C | Temperature at which air saturates | Fog/condensation forecasting, closely tied to humidity & temperature |
| `precipitation_mmh` | mm/h | Rainfall intensity | Visibility, deck safety, cargo exposure risk |
| `solar_radiation_wm2` | W/m² | Incoming solar energy flux | Diurnal-cycle indicator, deck heating, thermal expansion of structures |
| `visibility_km` | km | Horizontal visibility | Direct go/no-go factor for berthing and pilotage |

`wind_direction_deg` is **circular** (0° = 360°), so it's modeled via `sin`/`cos`
encoding internally and scored with circular MAE (degrees), not RMSE.

---

## 2. The 12 models — what they are, in plain language

| Model | One-line idea | Forecasts the 120h horizon by |
|---|---|---|
| **LSTM** | Recurrent network: reads history, remembers state, predicts next hour | Predicting 1 hour, feeding that prediction back in, repeating **120 times** (recursive rollout) |
| **XGBoost** | Gradient-boosted decision trees, one independent model **per parameter** | Each tree ensemble directly predicts the value at lead time *h*, for every *h* = 1..120 (direct) |
| **N-BEATS** | Stack of fully-connected blocks, each refining what the previous block got wrong (residual stacking) | One forward pass outputs all 120 hours × 16 parameters at once (direct) |
| **N-HiTS** | Like N-BEATS, but each stack first **downsamples** the input at a different rate (24h/12h/1h) before predicting, then smoothly interpolates the result back up | One forward pass (direct) |
| **DLinear** | Splits each series into a slow trend + a seasonal wiggle, fits one linear layer to each, adds them | One forward pass through 2 linear layers (direct) |
| **TiDE** | Dense (MLP-only) encoder/decoder that also reads the **known future calendar** (hour-of-day, day-of-year) as input, plus a linear shortcut | One forward pass (direct) |
| **TSMixer** | Alternates "mix across time" and "mix across parameters" MLP layers, so wind can inform the wave forecast and vice versa | One forward pass (direct) |
| **Harmonic-Residual Hybrid** | Fits exact sine/cosine waves at the known tidal/diurnal periods (M2, K1, O1, 24h, etc.) by simple least-squares, then trains a small network only on what's left over | Harmonic part is a closed-form formula (instant, never "wrong" in shape); residual part is one small forward pass |
| **iTransformer** | "Inverts" the standard Transformer: each *parameter* (not each timestep) becomes a token, so self-attention learns which parameters inform which directly | One forward pass through a Transformer encoder over 21 variate-tokens (direct) |
| **PatchTST** | Splits each parameter's history into time patches, runs one **shared-weight** Transformer over all 21 parameters' patches independently (no cross-parameter attention) | One forward pass; same weights process every channel, multiplying effective training data 21× (direct) |
| **DeepAR** | RNN that predicts a Gaussian `(μ, σ)` at every step instead of a single number, trained by maximizing likelihood | **100 Monte-Carlo sample paths**, each sampling and feeding the sample back in (ancestral sampling); point forecast = mean across paths |
| **Gaussian Process** | Bayesian model, fit independently per parameter, with kernel = periodic(24h) + periodic(12.42h) + local RBF | Closed-form posterior mean **and** variance at the 120 future timestamps — no sampling needed |

**Why "direct" vs "recursive" matters**: a recursive model (LSTM, DeepAR) makes a small
error at hour 1, then uses its own slightly-wrong prediction as input for hour 2, and
so on — errors compound over 120 steps. A "direct" model predicts all 120 hours from
the *real* last-known data in one shot, so there's no compounding. This is the single
biggest reason the LSTM underperforms in this comparison.

**Why iTransformer and PatchTST, specifically?** They're the one architectural family
missing from models 1-8: none of those use attention. `latest_research.txt` highlights
attention-based models (Crossformer, iTransformer, PatchTST, Pathformer) as a leading
2023-2025 research direction alongside the MLP/linear family already covered here. The
two differ in *what* the attention looks across: iTransformer attends **across
parameters** (so it can in principle learn wind→wave coupling, like TSMixer but via
learned attention instead of a fixed MLP); PatchTST attends **across time patches within
one parameter**, sharing weights across all 21 parameters instead. On this dataset
PatchTST has the better median accuracy overall, but iTransformer's cross-parameter
attention finds real structure on specific parameters (air pressure, air temp) nothing
else captures — see §4.

**Why DeepAR and Gaussian Process, and what didn't make the cut.** `ML models_few_more.txt`
surveys a much broader field — foundation models (Chronos, Moirai, Lag-Llama),
weather/climate models (GraphCast, Pangu-Weather, FourCastNet), Neural Operators, PINNs,
spatio-temporal GNNs, Neural ODEs. Almost none of it fits a single-buoy, regularly-sampled,
multivariate tabular problem:
- **GraphCast / Pangu-Weather / FourCastNet / Neural Operators (FNO)** all forecast whole
  spatial fields over a map/mesh — this dataset has no spatial dimension (one location),
  so there's nothing for these architectures to operate on.
- **Chronos / Moirai / Lag-Llama** are pretrained foundation models meant for zero-shot
  transfer from massive external corpora — using them well means downloading large
  pretrained checkpoints, not training from scratch on 70 days of one site's data, which
  is out of scope for this notebook's fully-reproducible, from-scratch approach.
- **PINNs / Neural ODEs** need known governing equations or irregular/continuous sampling
  to earn their complexity; this data is regularly sampled hourly, and the one parameter
  with a clean known equation (tides) already has the Harmonic-Residual model.
- **Spatio-temporal GNNs** need a graph; the one place a "graph" means something here is
  *across parameters*, which iTransformer and TSMixer already cover.

What **does** fit, and is genuinely new versus models 1-10: **DeepAR** and **Gaussian
Processes**, both explicitly recommended in that survey for small, single-site,
multivariate settings, and both provide **uncertainty quantification** — a calibrated
sense of "how confident is this forecast" — that no other model here offers.

---

## 3. Real-Time tab metrics — what each number means

The dashboard's **🚀 Best for Real-Time** tab ranks models for production deployment.
Here's exactly what each column means and how it's computed:

| Metric | Definition | How it's computed | Why it's used |
|---|---|---|---|
| **MAE** | Mean Absolute Error, in the parameter's own unit (m, °C, hPa, …) | `mean(\|actual − predicted\|)` over the 120 held-out hours | The most directly interpretable accuracy number — "on average, off by this much" |
| **RMSE** | Root Mean Squared Error, same unit | `sqrt(mean((actual − predicted)²))` | Penalizes large errors more than MAE; flags models with occasional big misses |
| **Skill vs persistence (%)** | How much better than the naive "tomorrow = now" baseline | `(1 − MAE_model / MAE_persistence) × 100` | Normalizes across parameters with wildly different scales/units. **Positive = beats the dumbest possible forecast. Negative = loses to it.** |
| **Median skill** | The middle skill value across all 16 parameters for one model | `median()` of that model's 16 skill-% values | Used **instead of the mean** because a few parameters (e.g. conductivity, SST) produce extreme negative skill outliers for some models — a mean would be dominated by those outliers and not reflect *typical* performance |
| **% beats persistence** | Fraction of the 16 parameters where skill > 0 | `count(skill > 0) / 16 × 100` | A simple "how often is this model actually useful" reliability measure |
| **Outright wins** | Number of parameters where this model has the lowest MAE of all 8 | Count of rows in `best_model` column | Shows where a model is the single best choice, not just "good on average" |
| **Inference latency (ms)** | Time to produce one full 120-hour forecast, on CPU | Wall-clock time of the model's forward pass(es), averaged over repeated calls | The metric that actually matters for "real-time" — how fast can you get a fresh forecast when new buoy data lands |
| **Forward passes / forecast** | How many times the network has to run to produce one forecast | 1 for all direct models; **120** for the LSTM (recursive); **17** for XGBoost (one model per parameter) | Explains *why* latency differs — it's a structural property of the model, not a tuning choice |
| **Parameters (params)** | Number of trainable weights in the network | `sum of numel() over all weights` (trees×count for XGBoost, as a rough size proxy — not directly comparable to NN weights) | A proxy for memory footprint / deployment size / overfitting risk |
| **Train time (s)** | Wall-clock time to train the model from scratch, on CPU | Measured during the actual notebook run | An **offline** cost (you retrain periodically, not per-forecast) — shown for completeness, doesn't affect real-time suitability directly |
| **Composite score** | The single number used to rank models | `median_skill_% − inference_ms / 10` | Combines typical accuracy with a latency penalty; the divide-by-10 keeps latency from swamping the score for fast models, while still being severe enough that genuinely slow models (LSTM's 66ms recursive rollout, DeepAR's 9.4s full Monte-Carlo uncertainty band) are penalized in proportion to how unsuitable they are for per-forecast real-time use |

---

## 4. How well does each model predict each parameter?

Full numbers are in `metrics_summary_pytorch.csv`. Below: each model's **3 best** and
**3 worst** parameters by skill, with MAE for context, ordered by overall ranking
(median skill — see §5).

### 🥇 PatchTST — best overall
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| Air temp | 0.808 °C | **+78.7%** | | Conductivity | 1.146 mS/cm | **−165.2%** |
| Current speed | 0.087 m/s | +77.9% | | Sea surface temp | 0.755 °C | −113.7% |
| Dew point | 0.743 °C | +76.6% | | Precipitation | 0.807 mm/h | −110.0% |

Wins **2 of 16 parameters outright** (wave period, current speed) and posts the highest
median skill of any model (+48%, see §5). Its channel-independent patches mean the
*same* shared weights see all 21 channels' worth of training examples — effectively 21×
the data of a model that fits separate weights per channel — which matters a lot on a
dataset this size. Still weak on conductivity/SST, same blind spot every model shares.

### 🥈 DLinear — simplest model that's still genuinely competitive
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| Wind speed | 1.955 m/s | **+67.4%** | | Conductivity | 2.714 mS/cm | **−528.2%** |
| Air temp | 1.397 °C | +63.2% | | Sea surface temp | 1.926 °C | −445.5% |
| Dew point | 1.187 °C | +62.6% | | Precipitation | 0.608 mm/h | −58.1% |

Wins wave height outright. A single linear layer per channel can't model conductivity's
nonlinear dependence on salinity *and* temperature jointly — but it's excellent wherever
a series is mostly linear trend + seasonal cycle, at 6.5× fewer parameters than PatchTST.

### 🥉 TSMixer / TiDE — cross-parameter coupling specialists
**TSMixer** (wins 3 outright: wind speed, relative humidity, visibility — the most wins
of any model)
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| Air temp | 0.648 °C | **+82.9%** | | Precipitation | 0.744 mm/h | **−93.5%** |
| Wind speed | 1.781 m/s | +70.3% | | Conductivity | 1.345 mS/cm | −211.4% |
| Tidal level | 0.39 m | +65.3% | | Sea surface temp | 1.22 °C | −245.5% |

**TiDE** (no outright wins this run, but consistently 2nd-3rd on several parameters)
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| Dew point | 0.755 °C | **+76.2%** | | Precipitation | 1.136 mm/h | **−195.5%** |
| Air temp | 0.941 °C | +75.2% | | Salinity | 0.176 PSU | −108.3% |
| Tidal level | 0.283 m | +74.8% | | Sea surface temp | 0.689 °C | −95.0% |

Both benefit from explicit structure — TSMixer's feature-mixing, TiDE's known-covariate
encoder — on parameters with strong coupling or calendar signal.

### XGBoost — the reliable classical baseline
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| Air temp | 0.475 °C | **+87.5%** | | Conductivity | 1.168 mS/cm | **−170.4%** |
| Solar radiation | 28.69 W/m² | +86.6% | | Salinity | 0.219 PSU | −160.4% |
| Dew point | 0.907 °C | +71.4% | | Sea surface temp | 0.691 °C | −95.6% |

Wins **solar radiation** and **wind direction** outright — tree ensembles handle solar's
hard diurnal cutoffs (zero at night, sharp ramps) and circular wind direction very well.

### N-HiTS / N-BEATS — large models, modest accuracy
**N-HiTS** (0 outright wins, despite a 1.05M-parameter footprint)
| Best at | Skill | | Worst at | Skill |
|---|---|---|---|---|
| Dew point | **+68.8%** | | Precipitation | **−347.0%** |
| Air temp | +67.7% | | Salinity | −135.8% |

**N-BEATS** (wins sea surface temp & conductivity — the two *hardest* parameters here,
where every other model is deeply negative — at the cost of a 4.2M-parameter model)
| Best at | Skill | | Worst at | Skill |
|---|---|---|---|---|
| Air temp | **+76.9%** | | Precipitation | **−349.6%** |
| Tidal level | +68.6% | | Salinity | −82.2% |

N-HiTS's multi-rate pooling (24h/12h/1h) should shine more on longer horizons than 120h.

### Gaussian Process — the calibrated-uncertainty specialist
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| Tidal level | 0.14 m | **+87.5%** | | Precipitation | 0.492 mm/h | **−27.9%** |
| Air temp | 0.526 °C | +86.1% | | Sea surface temp | 0.637 °C | −80.4% |
| Solar radiation | 38.82 W/m² | +81.9% | | Conductivity | 3.169 mS/cm | −633.3% |

No outright wins, but consistently a strong #2-3 on the periodic parameters (tide, air
temp, solar) where its frozen-period kernels are exactly the right inductive bias — and
it's the only model besides DeepAR that gives a calibrated confidence interval natively.

### Harmonic-Residual Hybrid — tidal specialist
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| **Tidal level** | 0.068 m | **+93.9%** | | Conductivity | 2.612 mS/cm | **−504.5%** |
| Solar radiation | 48.81 W/m² | +77.2% | | Sea surface temp | 1.776 °C | −403.1% |
| Air temp | 1.372 °C | +63.9% | | Precipitation | 1.066 mm/h | −177.5% |

Tidal level skill (+93.9%) is **6+ points clear of the next-best model** (the GP, at
+87.5%) and far clear of everything else — exactly what the literature review
(`MARINE_FORECASTING_IMPLEMENTATION_GUIDE.md`, §1.1) predicts for a harmonic-decomposition
approach. Outside periodic parameters it has nothing extra to offer over a plain
residual model.

### iTransformer — wins where it counts, even with a low median
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| **Air temp** | 0.435 °C | **+88.5%** | | Conductivity | 1.011 mS/cm | **−134.0%** |
| **Tidal level** | 0.186 m | +83.5% | | Sea surface temp | 0.833 °C | −135.8% |
| Solar radiation | 47.84 W/m² | +77.7% | | Precipitation | 2.106 mm/h | −448.1% |

Lowest median skill of the 12 (+1%), but **wins air pressure and air temp outright**
(+76.7% and +88.5%) — its cross-variate attention, with only 21 tokens (one per
parameter) to learn from per window, found real structure on these two specifically that
no other model matched. A genuinely mixed result: weak typical performance, strong peak
performance on a couple of parameters.

### LSTM — reference baseline, not recommended for real-time
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| Air temp | 0.7 °C | **+81.5%** | | Conductivity | 1.655 mS/cm | **−283.1%** |
| Dew point | 0.632 °C | +80.1% | | Sea surface temp | 1.147 °C | −224.8% |
| Solar radiation | 46.33 W/m² | +78.4% | | Visibility | 2.897 km | −50.2% |

Its 2 outright wins (salinity, dew point) show recursive rollout isn't always fatal —
just usually a 66ms-per-forecast liability for no accuracy benefit over direct models.

### DeepAR — wins the hardest parameter, at a real cost
| Best at | MAE | Skill | | Worst at | MAE | Skill |
|---|---|---|---|---|---|---|
| Solar radiation | 51.1 W/m² | **+76.2%** | | Sea surface temp | 2.07 °C | **−486.2%** |
| Dew point | 1.304 °C | +58.9% | | Conductivity | 2.464 mS/cm | −470.3% |
| Wind speed | 2.73 m/s | +54.5% | | Salinity | 0.188 PSU | −123.6% |

Wins **precipitation outright** (MAE 0.396, skill −3.1% — the *least negative* skill any
model achieves on this parameter; nothing beats persistence here, but DeepAR loses the
least). Training on Gaussian negative log-likelihood instead of raw MSE produces a
smoother, more robust mean estimate on bursty, heavy-tailed data than point-forecast
models get from minimizing squared error directly. The catch: that's its **point**
forecast; getting its actual selling point — a real uncertainty band — costs 100
Monte-Carlo sample paths and ~9.4 seconds (see §5).

---

## 5. Real-time suitability ranking (recap)

| Rank | Model | Median skill | Beats persistence | Wins | Latency | Params |
|---|---|---|---|---|---|---|
| 🥇 1 | **PatchTST** | **+48.0%** | 69% | 2/16 | 1.20 ms | 114K |
| 🥈 2 | DLinear | +39.5% | 75% | 1/16 | 0.11 ms | 17.5K |
| 🥉 3 | TSMixer | +36.7% | 75% | 3/16 | 1.07 ms | 45K |
| 4 | TiDE | +36.3% | 75% | 0/16 | 0.41 ms | 792K |
| 5 | XGBoost | +14.3% | 56% | 2/16 | 15.3 ms (×17 models) | — |
| 6 | N-HiTS | +8.2% | 56% | 0/16 | 0.37 ms | 1.05M |
| 7 | N-BEATS | +7.7% | 56% | 2/16 | 0.70 ms | 4.2M |
| 8 | Gaussian Process | +16.4%* | 62% | 0/16 | 89.9 ms (predict only) | — |
| 9 | Harmonic-Residual | +2.2%** | 56% | 1/16 | 0.13 ms | 473K |
| 10 | iTransformer | +0.6% | 50% | 2/16 | 1.43 ms | 81K |
| 11 | LSTM | +1.9% | 56% | 2/16 | 65.7 ms (×120 passes) | 74.8K |
| 12 | DeepAR | +2.6%*** | 56% | 1/16 | 9359 ms (×12,000 passes) | 75.8K |

\* GP's prediction is cheap (90ms for all 17 parameters combined) but each parameter's
kernel needs periodic refitting (~32s total, offline) — budget that separately from
per-forecast latency. \*\* Harmonic-Residual's median is dragged down by non-periodic
parameters; its **tidal-level skill alone is +93.9%**, the best single result in the
whole comparison. \*\*\* DeepAR's listed latency is for the **full** 100-sample
Monte-Carlo uncertainty band; a point-only forecast (1 sample path) costs ~86ms instead,
comparable to LSTM.

**PatchTST remains #1** after adding DeepAR/GP — its composite score (+47.9) is still
clear of DLinear (+39.5), TSMixer (+36.6), and TiDE (+36.3). The Gaussian Process lands
mid-table on the composite score (its accuracy is solid but it has no outright wins, and
its prediction-only latency, while cheap, is non-trivial compared to the sub-2ms PyTorch
direct models). **DeepAR ranks last** purely because of its full-uncertainty-band cost —
its point-forecast accuracy and even its point-only latency are unremarkable but not
bad; the 9.4-second number is the price of the one thing none of the other 11 models can
do at all (a Monte-Carlo-sampled, horizon-growing confidence band).

**Bottom line / recommended production setup**: run **PatchTST** (or DLinear if
simplicity matters more than the last few points of accuracy) as the default forecaster
for most parameters, **TSMixer** where cross-parameter coupling matters, **iTransformer**
specifically for air pressure/air temp, swap in the **Harmonic-Residual Hybrid** for
`tidal_level_m`, and layer in the **Gaussian Process** (or DeepAR in point-only mode, if
the downstream consumer specifically needs growing-uncertainty Monte-Carlo samples
rather than a closed-form interval) wherever a confidence interval is operationally
required. This per-parameter ensemble strategy is what both the dashboard and the
original literature review (`MARINE_FORECASTING_IMPLEMENTATION_GUIDE.md`) converge on.

---

## 6. Winners & Ensembles — best model per parameter, and do 2-model blends help?

The dashboard's **🏅 Winners & Ensembles** tab (and notebook §20–23) answers two follow-on
questions once you already know the single best model per parameter. Both are
recomputed automatically as models are added — the ranking and ensemble search are
generic over however many models exist, no logic changes needed when DeepAR/GP joined.

### 6.1 Best single model per parameter, ranked by how decisively it wins

Sorted by the winning model's skill vs persistence — strongest win at the top, weakest
(or negative — every model struggles) at the bottom:

| Rank | Parameter | Best model | Skill |
|---|---|---|---|
| 1 | Tidal level | Harmonic-Residual | **+93.9%** |
| 2 | Air temp | **iTransformer** | **+88.5%** |
| 3 | Solar radiation | XGBoost | +86.6% |
| 4 | Dew point | LSTM | +80.1% |
| 5 | Current speed | PatchTST | +77.9% |
| 6 | Air pressure | **iTransformer** | **+76.7%** |
| 7 | Wind speed | TSMixer | +70.3% |
| 8 | Wave height | DLinear | +55.0% |
| 9 | Visibility | TSMixer | +47.9% |
| 10 | Wave period | PatchTST | +47.8% |
| 11 | Wind direction | XGBoost | +47.1% |
| 12 | Relative humidity | TSMixer | +25.5% |
| 13 | Salinity | LSTM | +1.2% |
| 14 | Precipitation | **DeepAR** | −3.1% |
| 15 | Sea surface temp | N-BEATS | −32.4% |
| 16 | Conductivity | N-BEATS | −42.5% |

**Adding DeepAR and the Gaussian Process diversified the leaderboard further**: 9
different models now win at least one parameter outright (vs. 4-5 before), with
iTransformer's pair of wins (air pressure, air temp) and DeepAR's precipitation win the
standout new entries — both are cases where the model's *design*, not just raw capacity,
matched something specific about that parameter (cross-variate attention; robust
likelihood-based training on bursty data). The bottom 2 (sea surface temp, conductivity)
still have **no model** beating persistence at all.

### 6.2 Can a 2-model ensemble beat the single best model?

For every parameter, all **66 possible pairs** of the 12 models were combined with a
plain **50/50 average** of their predictions (circular vector-average for wind
direction) and compared against the single best individual model.

**Why an unweighted average, not a fitted/weighted blend?** Fitting blend weights (e.g.
by inverse error) needs a validation window *separate* from the 120-hour test window
used to score everything else — there's only one such window here, so fitting weights
against it would leak test information into the result and overstate the apparent
benefit. A 50/50 average has **zero free parameters**, so any improvement it shows is a
real, leakage-free signal that the two models' mistakes partially cancel.

**Result: ensembling helps on 10 of 16 parameters**, including one striking case the
extra models surfaced — sea surface temp goes from −32.4% (N-BEATS alone) to **+58.7%**
(TiDE+iTransformer averaged), because the two specific weaknesses partially cancel:

| Parameter | Single best | Best ensemble pair | Single best MAE | Ensemble MAE | Recommendation |
|---|---|---|---|---|---|
| Wind direction | XGBoost | XGBoost + PatchTST | 19.94° | **19.12°** | ✅ Use ensemble |
| Conductivity | N-BEATS | TiDE + iTransformer | 0.616 | **0.289** | ✅ Use ensemble |
| Sea surface temp | N-BEATS | TiDE + iTransformer | 0.468 | **0.146** | ✅ Use ensemble |
| Air temp | iTransformer | iTransformer + Gaussian Process | 0.435 | **0.318** | ✅ Use ensemble |
| Dew point | LSTM | XGBoost + Gaussian Process | 0.632 | **0.536** | ✅ Use ensemble |
| Relative humidity | TSMixer | N-BEATS + TSMixer | 3.332 | **3.243** | ✅ Use ensemble |
| Wave height | DLinear | TSMixer + PatchTST | 0.238 | **0.165** | ✅ Use ensemble |
| Wave period | PatchTST | TiDE + PatchTST | 0.282 | **0.242** | ✅ Use ensemble |
| Salinity | LSTM | LSTM + iTransformer | 0.083 | **0.059** | ✅ Use ensemble |
| Wind speed | TSMixer | TSMixer + Gaussian Process | 1.781 | 1.780 | ✅ Use ensemble (marginal) |
| Current speed | PatchTST | iTransformer + PatchTST | 0.087 | 0.092 | ❌ Use single model |
| **Tidal level** | **Harmonic-Residual** | Harmonic-Residual + Gaussian Process | **0.068** | 0.088 | ❌ Use single model |
| Precipitation | DeepAR | LSTM + DeepAR | 0.396 | 0.430 | ❌ Use single model |
| Visibility | TSMixer | DLinear + iTransformer | 1.005 | 1.116 | ❌ Use single model |
| Air pressure | iTransformer | TiDE + TSMixer | 1.570 | 1.706 | ❌ Use single model |
| Solar radiation | XGBoost | XGBoost + Gaussian Process | 28.692 | 32.913 | ❌ Use single model |

The 6 "use single model" rows are precisely the parameters with the **most dominant
specialist** — Harmonic-Residual on tide, DeepAR on precipitation, TSMixer on visibility,
iTransformer on air pressure, XGBoost on solar radiation, and PatchTST on current speed.
This is the expected, healthy outcome of a leakage-free ensemble search: it finds real
wins where two models are complementary (the SST/conductivity cases are dramatic
examples), and correctly declines to recommend an ensemble where one model — old or new
— is already dominant.

**Implementation**: this is computed directly in the notebook (§20–22) from the already-trained
models' saved predictions — no retraining, ~instant. Outputs:
- `best_model_per_parameter.csv` — the §6.1 ranking
- `ensemble_recommendation.csv` — the §6.2 table (all 16 parameters)
- `ensemble_forecast_vs_actual.csv` — the actual best-pair ensemble forecast values, for plotting
- `uncertainty_bands.csv` — DeepAR Monte-Carlo std + GP posterior std per parameter (for confidence-interval plotting)

**Practical takeaway for production**: serve the single best model from §6.1 for
tidal level, precipitation, visibility, air pressure, solar radiation, and current
speed; serve the 50/50-averaged pair from §6.2 for everything else.

---

## 7. Repo guide

| File | What it is |
|---|---|
| `generate_marine_data.py` | Synthesizes 75 days of physically-realistic hourly marine data (tidal harmonics, storms, diurnal cycles, cross-parameter coupling) |
| `Marine_Forecast_LSTM_XGBoost.ipynb` | Original TensorFlow/Keras LSTM + XGBoost baseline |
| `Marine_Forecast_PyTorch_NBEATS_NHiTS.ipynb` | This project's main notebook — all **12** PyTorch/XGBoost/GPyTorch models, trained and scored, plus the §6 winner/ensemble analysis |
| `app.py` | Streamlit dashboard (`streamlit run app.py`) — Forecast (with confidence-band toggle), Metrics & Skill, Best for Real-Time, Winners & Ensembles, About tabs |
| `metrics_summary_pytorch.csv` | MAE/RMSE/skill per parameter per model (source of truth for §4 above) |
| `forecast_vs_actual_pytorch.csv` | Hourly actual vs. predicted values for every parameter × every model |
| `compute_benchmark.csv` | Parameter counts, inference latency, training time per model (source of truth for §3/§5 above) |
| `best_model_per_parameter.csv` | Best model per parameter, ranked by win margin (source of truth for §6.1) |
| `ensemble_recommendation.csv` | Best 2-model ensemble per parameter vs. single best (source of truth for §6.2) |
| `ensemble_forecast_vs_actual.csv` | Hourly actual vs. best-pair-ensemble forecast, for plotting |
| `uncertainty_bands.csv` | DeepAR Monte-Carlo std + Gaussian Process posterior std per parameter, for confidence-interval plotting |
| `MARINE_FORECASTING_IMPLEMENTATION_GUIDE.md` | Literature review this project's domain-specific modeling choices (LSTM, XGBoost, Harmonic-Residual) are grounded in |
| `latest_research.txt` | 2023-2026 survey of multivariate time-series forecasting research — motivated adding iTransformer and PatchTST (§2) |
| `ML models_few_more.txt` | Survey of foundation models, weather/climate architectures, GPs, Neural ODEs, PINNs, GNNs — motivated adding DeepAR and Gaussian Process, and explains what was deliberately left out (§2) |
