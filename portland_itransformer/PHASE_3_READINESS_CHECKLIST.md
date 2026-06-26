# Phase 3 Readiness: Remediation Checklist

**Current Status:** Single iTransformer trained, identified critical architectural issues  
**Decision Point:** Split into two models OR continue with fixes to single model  
**Timeline:** 6–8 hours for full remediation + retraining

---

## Critical Issues Found

### Issue 1: Chaotic Variables Catastrophic Failure
```
air_pressure_hpa:  −6974% skill
dew_point_c:       −6887% skill
air_temp_c:        −5761% skill
wind_u_east_ms:    −2654% skill
```

**Root Cause:** Single model trained on both deterministic (tides +96%) and chaotic (pressure −6974%) variables simultaneously. Conflicting gradients cause overfitting to synthetic noise patterns.

**Solution:** Train separate Atmospheric iTransformer on **anomalies** (e.g., `air_temp_anomaly = raw − harmonic_baseline`) instead of raw values.

---

### Issue 2: Physics-Derived Variables Trained Directly
```
Current implementation:
  Input: [wind_u, wind_v, ...] → Model → [wind_speed, wind_direction, ...]
  
Better approach:
  Input: [wind_u, wind_v, ...] → Model → [wind_u, wind_v]
  Post-process: sqrt(u² + v²) → wind_speed
                atan2(u, v) → wind_direction
```

**Impact:** Removes 3–5 indirect targets from model's responsibility, reducing confusion.

---

### Issue 3: Single-Model Overloaded
- 13 direct targets
- 18 total outputs
- Mix of physics-constrained and chaotic variables
- One loss function can't balance tides (want tight control) vs atmosphere (want anomaly prediction)

**Solution:** Two specialized models:
1. **Marine iTransformer** (8 targets) — Ocean/tide/radiation — Direct targets
2. **Atmospheric iTransformer** (5 targets) — Weather/wind/pressure — Anomaly targets

---

## Recommended Path Forward

### Option A: Full Remediation (Recommended)
Implement two-model architecture with anomaly-based training.

**Steps:**
1. ✅ Create `HarmonicAnomalyBaseline` class (DONE)
2. [ ] Create separate preprocessing for Marine and Atmosphere
3. [ ] Create `configs/marine_24h.yaml` and `configs/atmosphere_24h.yaml`
4. [ ] Train Marine iTransformer (expect +70–85% skill)
5. [ ] Train Atmospheric iTransformer (expect +40–60% skill)
6. [ ] Add validation pipeline (timestamp, ordering, scaling)
7. [ ] Test Phase 3 integration (merge 18 outputs)

**Effort:** 6–8 hours  
**Risk:** Low (architectural fix is sound)  
**Expected improvement:** Atmospheric +40–60% (vs −6974%)

---

### Option B: Quick Fix (Faster but Incomplete)
Apply anomaly baseline to atmosphere without full two-model refactoring.

**Steps:**
1. Add anomaly subtraction to preprocessing
2. Retrain single model with atmosphere anomalies
3. Keep Marine targets as-is
4. Add post-processing validation

**Effort:** 2–3 hours  
**Risk:** Medium (single model still suboptimal)  
**Expected improvement:** Atmospheric +20–40% (vs −6974%)

---

### Option C: Accept Current Results + Phase 3
Proceed with current single model, document limitations.

**Effort:** 0 hours  
**Risk:** High (atmospheric outputs unreliable)  
**Expected improvement:** None

---

## Decision Framework

**Choose Option A if:**
- You have real NOAA data for retraining
- You want production-ready forecasts
- You have 6–8 hours available
- Atmospheric variables are important to your use case

**Choose Option B if:**
- You want quicker improvements
- You're willing to accept moderate atmospheric skill
- You have 2–3 hours available
- You'll deploy to real data soon

**Choose Option C if:**
- Marine/tide/current forecasts are your primary use case
- You'll immediately retrain on real data
- You want to move to Phase 3 immediately with the understanding that atmospheric variables need work

---

## Implementation Details for Option A

### 1. Marine iTransformer Targets (8)
```yaml
# configs/marine_24h.yaml
targets:
  - tidal_residual_m           # residual from UTide
  - current_u_east_ms          # direct
  - current_v_north_ms         # direct
  - salinity_psu               # direct
  - water_temp_c               # direct (or residual from daily baseline)
  - log1p_global_radiation_wm2 # log transform
  - log_significant_wave_height_m
  - log_zero_crossing_period_s

seq_len: 672      # 7 days
pred_len: 96      # 24 hours
```

**Expected performance:**
- tidal_residual: +96% (no change)
- currents: +90% (no change)
- salinity: +88% (no change)
- water_temp: +50% → +70% (improvement with daily baseline)
- radiation: +95% (no change)
- waves: +70% (no change)

---

### 2. Atmospheric iTransformer Targets (5)
```yaml
# configs/atmosphere_24h.yaml
targets:
  - air_temp_anomaly_c              # air_temp - daily_harmonic
  - log1p_dewpoint_depression_c     # log(max(air_temp - dew_point, 0.001))
  - air_pressure_anomaly_hpa        # pressure - daily_harmonic
  - wind_u_anomaly_ms               # wind_u - climatology_u
  - wind_v_anomaly_ms               # wind_v - climatology_v

seq_len: 672      # 7 days
pred_len: 96      # 24 hours
```

**Expected performance (24-hour):**
- air_temp_anomaly: −5761% → +50% (massive improvement)
- pressure_anomaly: −6974% → +35% (massive improvement)
- wind_u_anomaly: −2654% → +25% (improvement)
- dew_point: −6887% → +40% (improvement via log1p)

---

### 3. Preprocessing Changes

Create separate pipelines:

**`preprocessing_marine.py`:**
```python
def preprocess_marine(df_raw, config):
    """8-target marine pipeline."""
    
    # 1. Validate
    df_raw = validate(df_raw)
    
    # 2. Direction transforms
    df['wind_u'], df['wind_v'] = speed_dir_to_uv(...)
    df['current_u'], df['current_v'] = speed_dir_to_uv(...)
    
    # 3. UTide baseline
    utide_baseline.fit(df_train['tidal_level_m'])
    df['tidal_residual_m'] = df['tidal_level_m'] - utide_baseline.predict(df['timestamp'])
    
    # 4. Clear-sky radiation baseline
    df['global_radiation_wm2'] -= clearsky_baseline.get_clear_sky(df['timestamp'])
    
    # 5. Log transforms
    df['log_wave_height'] = log(df['significant_wave_height_m'] + 1e-4)
    df['log_zero_crossing'] = log(df['zero_crossing_period_s'])
    df['log_radiation'] = log1p(df['global_radiation_wm2'])
    
    # 6. Split chronologically
    train, val, test = split(df, [0.5, 0.25, 0.25])
    
    # 7. Fit scalers on TRAIN ONLY
    scaler_targets = StandardScaler().fit(train[TARGET_MARINE])
    scaler_inputs = StandardScaler().fit(train[[...]])
    
    # 8. Scale all splits
    df[TARGET_MARINE] = scaler_targets.transform(df[TARGET_MARINE])
    
    return df, scaler_targets, scaler_inputs
```

**`preprocessing_atmosphere.py`:**
```python
def preprocess_atmosphere(df_raw, config):
    """5-target atmospheric pipeline."""
    
    # 1. Validate
    df_raw = validate(df_raw)
    
    # 2. Direction transforms
    df['wind_u'], df['wind_v'] = speed_dir_to_uv(...)
    
    # 3. Harmonic baselines (TRAIN ONLY)
    harmonic_temp = HarmonicAnomalyBaseline()
    harmonic_temp.fit(df['timestamp'], df['air_temp_c'], train_mask)
    
    harmonic_pressure = HarmonicAnomalyBaseline()
    harmonic_pressure.fit(df['timestamp'], df['air_pressure_hpa'], train_mask)
    
    # 4. Compute anomalies
    df['air_temp_anomaly_c'] = df['air_temp_c'] - harmonic_temp.predict(df['timestamp'])
    df['air_pressure_anomaly_hpa'] = df['air_pressure_hpa'] - harmonic_pressure.predict(df['timestamp'])
    
    # 5. Dew point depression
    depression = np.maximum(df['air_temp_c'] - df['dew_point_c'], 0.001)
    df['log1p_dewpoint_depression_c'] = np.log1p(depression)
    
    # 6. Wind climatology (TRAIN ONLY)
    wind_u_clim = df_train.groupby(df_train['timestamp'].dt.hour)['wind_u'].mean()
    wind_v_clim = df_train.groupby(df_train['timestamp'].dt.hour)['wind_v'].mean()
    df['wind_u_anomaly_ms'] = df['wind_u'] - df['timestamp'].dt.hour.map(wind_u_clim)
    df['wind_v_anomaly_ms'] = df['wind_v'] - df['timestamp'].dt.hour.map(wind_v_clim)
    
    # 7. Split, scale
    # (same as marine)
    
    return df, scaler_targets, scaler_inputs
```

---

### 4. Training

```bash
# Train Marine model
python src/train_marine.py --config configs/marine_24h.yaml

# Train Atmospheric model
python src/train_atmosphere.py --config configs/atmosphere_24h.yaml

# Expected runtime: ~10 minutes each on CPU
```

---

### 5. Evaluation & Validation

```python
# src/validate_pipeline.py (NEW)

def test_target_ordering():
    """Ensure canonical ordering."""
    assert json.load('artifacts/target_columns_marine.json') == TARGET_MARINE
    assert json.load('artifacts/target_columns_atmosphere.json') == TARGET_ATMOSPHERE

def test_scaler_fit_on_train_only():
    """Verify no data leakage."""
    scaler = load_scaler('artifacts/scaler_targets_marine.joblib')
    # Scaler must be fit on rows 0:17855 only
    
def test_transform_inverse_roundtrip():
    """Transform then inverse-transform must have error < 1e-8."""
    for param in TARGET_MARINE + TARGET_ATMOSPHERE:
        x_original = np.random.randn(100)
        x_transformed = scaler.transform(x_original)
        x_recovered = scaler.inverse_transform(x_transformed)
        assert np.allclose(x_original, x_recovered, atol=1e-7)

def test_timestamp_alignment():
    """For every window: last(X) + 15min == first(Y)."""
    for dataset in [train, val, test]:
        for batch in dataset:
            X_ts, Y_ts = batch['X_timestamp'], batch['Y_timestamp']
            assert (X_ts[-1] + pd.Timedelta(minutes=15) == Y_ts[0]).all()

def test_reconstruction_valid_ranges():
    """Reconstructed values must be physical."""
    # RH in [0, 100]
    assert 0 <= rh_pred <= 100
    # Pressure in [950, 1050] hPa
    assert 950 <= pressure_pred <= 1050
    # Dew point <= air temperature
    assert dew_point_pred <= air_temp_pred
```

---

## Success Criteria

### Marine iTransformer
- ✓ All 8 targets achieve positive skill
- ✓ Tidal skill remains +96%
- ✓ No skill regression on currents/radiation
- ✓ Water temperature improves to +60%+

### Atmospheric iTransformer
- ✓ Temperature anomaly skill +40%+
- ✓ Pressure anomaly skill +30%+
- ✓ Wind anomaly skill +20%+ at 24h
- ✓ All tests pass (ordering, scaling, timestamps)
- ✓ RH ∈ [0, 100]
- ✓ Dew point ≤ air temperature
- ✓ No NaN or inf in outputs

### Integration (Phase 3)
- ✓ Merge 18 outputs in canonical order
- ✓ Inference < 100ms per forecast
- ✓ All tests pass

---

## Effort Estimation

| Task | Time | Priority |
|------|------|----------|
| Create preprocessing_marine.py | 1.0h | CRITICAL |
| Create preprocessing_atmosphere.py | 1.0h | CRITICAL |
| Train Marine model | 0.5h | CRITICAL |
| Train Atmospheric model | 0.5h | CRITICAL |
| Validation test suite | 1.0h | HIGH |
| Evaluation & metrics | 0.5h | HIGH |
| Phase 3 integration | 1.0h | MEDIUM |
| Documentation | 0.5h | LOW |
| **TOTAL** | **6.0h** | — |

---

## Recommendation

**Proceed with Option A (Full Remediation)** because:

1. Architectural fix is proven (anomaly-based forecasting is standard in meteorology)
2. Expected improvement is massive: −6974% → +35–50% on atmospheric variables
3. Marine performance unaffected or improved
4. Foundation for real NOAA data retraining
5. Effort is reasonable (6 hours)

**Next steps:**
1. Confirm you want to proceed with Option A
2. I'll implement marine + atmosphere preprocessing
3. Train both models
4. Validate all success criteria
5. Deliver Phase 3-ready system

Would you like me to implement Option A remediation?

