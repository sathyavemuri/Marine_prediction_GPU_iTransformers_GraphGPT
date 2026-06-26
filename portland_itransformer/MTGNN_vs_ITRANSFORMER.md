# MTGNN vs iTransformer: Side-by-Side Comparison

## Why We're Building iTransformer

You tested MTGNN on your real 120-day marine buoy data and found:
- ✅ **Good**: Thermodynamic variables (+20% skill)
- ❌ **Bad**: Wind/current/pressure (-30% skill, worse than persistence)

This new iTransformer system addresses that and provides an alternative architecture for comparison.

---

## Architecture Comparison

### MTGNN (Your Current System)

```
Architecture
├─ Graph Neural Network
│  └─ Sparse adjacency matrix (top-k correlation)
├─ Temporal Dilated Convolutions
│  └─ 9 blocks, exponential dilation
├─ Residual Learning
│  ├─ Persistence baseline
│  ├─ Seasonal (daily/weekly)
│  ├─ UTide + pvlib
│  └─ Model learns residuals
├─ Skip Connections
└─ Horizon-Conditioned Decoder

Input/Output
├─ Input: [batch, 672, 19] - 7-day history
├─ Output: [batch, 672, 15] - 7-day forecast
└─ Direct multi-step (no rollout)

Parameters
└─ ~60K

Training
├─ Loss: Weighted MSE per-target
├─ Optimizer: Adam
├─ Device: CPU/CUDA
└─ Training Time: ~30 min (120 days)
```

### iTransformer (New System)

```
Architecture
├─ Inverted Transformer
│  └─ Each variable = token (not time step)
├─ Token Embedding
│  └─ 1344 timesteps → 128-dim embedding per channel
├─ Transformer Encoder
│  ├─ 3 layers, 4 heads
│  ├─ Attention over 19 features (not 1344 steps)
│  └─ Operates on variable relationships
├─ Forecast Head
│  └─ 128-dim → 672-step prediction
└─ Future Covariate Head
   ├─ Small MLP (64 hidden)
   ├─ Learns to add tide/solar/calendar contribution
   └─ Prevents large decoder overhead

Input/Output
├─ Input: [batch, 1344, 19] - 14-day history
├─ Output: [batch, 672, 13] - 7-day forecast
└─ Direct multi-step (no rollout)

Parameters
└─ ~180K (larger due to longer history)

Training
├─ Loss: Huber (SmoothL1) with masks
├─ Optimizer: AdamW
├─ Scheduler: ReduceLROnPlateau
├─ Device: CPU/CUDA with mixed precision
└─ Training Time: ~2-3 hours (synthetic 365-day data)
```

---

## Feature Handling

### MTGNN

```
Raw Input (18 columns)
├─ Validates raw degrees
├─ Computes u/v components (wind, current)
├─ Applies log transforms (waves)
├─ Applies log1p transform (radiation)
├─ Computes derived outputs (RH, conductivity)
└─ Model targets: 15 parameters

Baseline Subtraction
├─ Persistence
├─ Seasonal (daily + weekly)
├─ UTide tide
├─ pvlib radiation
└─ Model learns residuals (baseline + residual = final)
```

### iTransformer

```
Raw Input (18 columns)
├─ Validates raw degrees
├─ Computes u/v components (wind, current)
├─ Applies log transforms (waves + radiation clearness)
├─ Creates cyclical features (hour, day-of-year sin/cos)
├─ Computes deterministic baselines
└─ Model targets: 13 parameters (residuals only, not derived)

Baseline Integration
├─ Deterministic features in KNOWN_FEATURES
│  ├─ tide_baseline_m (UTide)
│  ├─ clear_sky_radiation_wm2 (pvlib)
│  ├─ hour_sin, hour_cos
│  ├─ dayofyear_sin, dayofyear_cos
└─ Future covariate head learns to condition on these
```

---

## Data Pipeline

### MTGNN

```
1. Load raw 120-day CSV
2. Direction → u/v (wind, current)
3. Log transforms
4. Resample 10-min → 15-min
5. Split chronologically (90/20/10)
6. Fit baselines on train split
7. Compute residuals = actual - baseline
8. StandardScaler on residuals
9. Create windows (672 lookback, 672 horizon)
10. Train model on residuals
```

### iTransformer

```
1. Load raw 365-day synthetic CSV
2. Validate schema, cadence, ranges
3. Direction → u/v (wind, current)
4. Fit UTide on first 60 days ONLY
5. Compute tide_baseline_m for all timestamps
6. pvlib clear-sky radiation
7. Log transforms (waves, clearness)
8. Cyclical features (hour, dayofyear)
9. Chronological split (60-day baseline, 186/60/60 train/val/test)
10. StandardScaler on targets (training data only)
11. StandardScaler on known features (training data only)
12. FIT calibrators on training data ONLY:
    ├─ conductivity = f(salinity, water_temp)
    ├─ sig_wave_period = f(log_Tz, log_Hs)
    └─ peak_wave_period = f(log_Tz, log_Hs)
13. Create windows (1344 lookback, 672 horizon)
14. Train model on 13 direct targets
15. Reconstruct all 18 columns using calibrators
```

---

## Forecast Targets

### MTGNN (15 targets)

```python
1. air_temp_c                 → residuals
2. air_pressure_hpa           → residuals
3. wind_u_east_ms             → residuals (u/v components)
4. wind_v_north_ms            → residuals
5. water_temp_c               → residuals
6. tidal_level_m              → residuals
7. current_u_east_ms          → residuals
8. current_v_north_ms         → residuals
9. dew_point_c                → residuals
10. global_radiation_wm2       → residuals (log1p)
11. salinity_psu               → residuals
12. significant_wave_height_m  → residuals (log)
13. significant_wave_period_s  → residuals (log)
14. zero_crossing_period_s     → residuals (log)
15. peak_wave_period_s         → residuals (log)

Final = baseline + MTGNN_prediction
```

### iTransformer (13 targets + 6 known)

```python
# Direct targets (model learns these directly)
1. air_temp_c
2. air_pressure_hpa
3. water_temp_c
4. dew_point_c
5. salinity_psu
6. wind_u_east_ms
7. wind_v_north_ms
8. current_u_east_ms
9. current_v_north_ms
10. tidal_residual_m           (observed - UTide baseline)
11. log_significant_wave_height_m
12. log_zero_crossing_period_s
13. log_clearness_index        (log(radiation / clear_sky))

# Known covariates (not model targets, but features)
1. tide_baseline_m             (deterministic: UTide)
2. clear_sky_radiation_wm2     (deterministic: pvlib)
3. hour_sin
4. hour_cos
5. dayofyear_sin
6. dayofyear_cos
```

---

## Performance on Your Data

### Your MTGNN Results (120-day real marine buoy data)

| Parameter | Skill | Status |
|-----------|-------|--------|
| **Global Radiation** | +21.1% | ✅ Excellent |
| **Water Temperature** | +21.0% | ✅ Excellent |
| **Wave Height** | +20.2% | ✅ Excellent |
| **Air Temperature** | +1.5% | ⚠️ Marginal |
| **Wind U/V** | -29.9% | ❌ Worse than persistence |
| **Current U/V** | -60.5% | ❌ Much worse than persistence |
| **Salinity** | -683.0% | ❌ Terrible |
| **Pressure** | -415.8% | ❌ Terrible |
| **Tide** | -368.6% | ❌ Terrible |

### iTransformer Expected Results (365-day synthetic data)

Since this is synthetic data optimized for the Portland location, expected results differ:

| Parameter | Expected | Notes |
|-----------|----------|-------|
| **Global Radiation** | +22-25% | Slightly better (clearness_index log) |
| **Water Temperature** | +20-23% | Similar or slightly better |
| **Wave Height** | +20-25% | Similar or slightly better |
| **Air Temperature** | +2-5% | Slightly better (longer history) |
| **Wind U/V** | -20% to -30% | Similar (still chaotic) |
| **Current U/V** | -50% to -70% | Similar (still chaotic) |
| **Pressure/Salinity** | Likely negative | Without physics constraints |

**Key difference**: iTransformer has 14-day history vs 7-day in MTGNN, which may help on longer-range patterns.

---

## Evaluation Metrics

### MTGNN

```
Per-parameter MAE/RMSE/Skill
├─ Overall average
├─ By day (1-7)
└─ No specific horizon buckets

Baseline comparison
├─ Persistence
└─ Seasonal
```

### iTransformer

```
Per-parameter MAE/RMSE/Skill
├─ Overall average
├─ By horizon bucket:
│  ├─ 0-6 hours (fastest degradation)
│  ├─ 6-24 hours (synoptic timescale)
│  ├─ 24-72 hours (medium range)
│  └─ 72-168 hours (week-long)
├─ By parameter (13 targets)
└─ Circular MAE for directions

Baseline comparison
├─ Persistence (last value)
├─ Daily seasonal (same time 1 day ago)
├─ UTide tide baseline alone
└─ pvlib clear-sky radiation alone
```

---

## Advantages of Each

### MTGNN Advantages

✅ **Proven on real data** - Tested on your 120-day marine buoy dataset
✅ **Graph structure** - Explicitly learns variable relationships
✅ **Residual learning** - Leverages physics baselines effectively
✅ **Shorter horizon** - Works well for 7-day with 7-day history
✅ **Faster training** - 30 minutes on 120 days
✅ **Interpretable** - Graph adjacency shows variable importance
✅ **Physics-aware** - Builds on UTide, pvlib, seasonal patterns

### iTransformer Advantages

✅ **Longer history** - 14 days input vs 7 days, captures longer patterns
✅ **Modern architecture** - State-of-the-art attention mechanism
✅ **Channel-independent embedding** - Avoids spurious cross-variable coupling
✅ **Scalable** - Transformer efficiency for longer sequences
✅ **Known covariates** - Explicit conditioning on deterministic features
✅ **Hourly decomposition** - Hour/day-of-year cyclical features
✅ **Better on clean data** - Synthetic data may play to its strengths

---

## When to Use Each

### Use MTGNN When

- ✅ You have <6 months of real data (good for data-efficient learning)
- ✅ You trust physics baselines (UTide, seasonal, radiation)
- ✅ You want interpretable variable relationships (graph structure)
- ✅ You need fast training (<1 hour)
- ✅ You've validated it works on your data type

### Use iTransformer When

- ✅ You have 1+ year of data (reduces overfitting on large models)
- ✅ You want to test if longer history helps (14 vs 7 days)
- ✅ You're doing research/comparison studies
- ✅ You want state-of-the-art attention mechanisms
- ✅ Training time is not a constraint

---

## Direct Comparison: Same 120-Day Data

What would happen if you trained **iTransformer on your 120-day real data** vs MTGNN?

| Aspect | MTGNN | iTransformer |
|--------|-------|-----------|
| **Thermodynamic skill** | +21% (validated) | +20-22% (likely) |
| **Wave skill** | +20% (validated) | +20-22% (likely) |
| **Wind skill** | -30% (validated) | -25% to -35% (estimate) |
| **Training time** | 30 min | 2-3 hours |
| **Overfitting risk** | Low (60K params) | Medium-High (180K params) |
| **Data efficiency** | Better | Worse (needs more data) |
| **Physical interpretability** | High (graph) | Low (black box) |
| **Attention to longer patterns** | No | Yes (14-day history) |
| **Overall advantage** | **Better for real 120-day** | **Better for 1+ year synthetic** |

**Verdict**: On your 120-day real data, **MTGNN is likely better** due to:
1. Lower parameter count (less overfitting)
2. Physics baselines (UTide, seasonal)
3. Graph structure (interpretability)
4. Proven results on your data type

---

## Why Build iTransformer Then?

1. **Comparison/Research**: Show MTGNN isn't just "lucky" on your data
2. **Future data**: If you collect 1+ year of data, iTransformer may become competitive
3. **Architecture validation**: Test if "each variable as token" works for marine data
4. **Hybrid ensemble**: Could blend MTGNN + iTransformer predictions

---

## My Recommendation

### For Production (Your 120-Day Data)

**Use MTGNN + Hybrid Fallback**:
```python
forecast = {
    'water_temp': MTGNN (use when skill > 0%),
    'radiation': MTGNN (use when skill > 0%),
    'wave_height': MTGNN (use when skill > 0%),
    'wind': persistence (accept that chaotic),
    'current': persistence (accept that chaotic),
    'pressure': persistence (not predictable),
}
```

**Result**: No negative skill, honest about what's predictable.

### For Research/Comparison

**Build and test iTransformer**:
- Train on synthetic 365-day data
- Compare metrics vs MTGNN results
- Analyze if 14-day history helps
- Consider ensemble (0.6*MTGNN + 0.4*iTransformer)

---

## Summary Table

| Criterion | MTGNN | iTransformer | Winner |
|-----------|-------|-----------|--------|
| Real data validation | ✅ Yes (120 days) | ❌ No | MTGNN |
| Physics integration | ✅ Explicit | ⚠️ Via features | MTGNN |
| Interpretability | ✅ High (graph) | ❌ Low (transformer) | MTGNN |
| Data efficiency | ✅ Good (<6mo) | ⚠️ Poor (>1yr) | MTGNN |
| Training speed | ✅ 30 min | ❌ 2-3 hours | MTGNN |
| Long-range patterns | ⚠️ 7 days | ✅ 14 days | iTransformer |
| Modern architecture | ⚠️ GNN | ✅ Transformer | iTransformer |
| Parameter efficiency | ✅ 60K | ❌ 180K | MTGNN |
| **Overall** | **MTGNN wins** | **Better for research** | **MTGNN** |

---

**Conclusion**: Build iTransformer for **comparison and research**, but deploy MTGNN for **production forecasting** on your real 120-day marine buoy data.

---

**Last Updated**: 2026-06-25
