# iTransformer Remediation Implementation Plan

**Status:** Ready for Phase 3 with structural fixes  
**Date:** 2026-06-25  
**Issue:** Negative skill (-6974%) on atmospheric variables indicates model architecture mismatch, not data quality

---

## Executive Summary

The current single-model iTransformer achieves **excellent results on physics-constrained variables** (tides +96%, currents +90%, radiation +95%) but **catastrophic failure on chaotic variables** (pressure -6974%, wind -2654%).

**Root cause:** Training one model to output both deterministic (tides, radiation) and chaotic (pressure, wind) variables causes conflicting gradients and spurious synthetic pattern overfitting.

**Solution:** Implement **two separate specialized models:**
1. **Marine iTransformer** — Ocean/tide/radiation (deterministic, predictable)
2. **Atmospheric iTransformer** — Weather/wind/pressure (chaotic, anomaly-based)

---

## Implementation Roadmap

### Phase A: Architecture Refactoring (This Session)
- [x] Analyze current single-model failure modes
- [ ] Build Marine iTransformer (8 targets, deterministic baseline-subtracted)
- [ ] Build Atmospheric iTransformer (5 targets, anomaly-based)
- [ ] Create comprehensive validation pipeline
- [ ] Add horizon-weighted loss function

### Phase B: Training (Next Session)
- [ ] Train Marine model (24-hour baseline, should achieve +70% skill)
- [ ] Train Atmospheric model (24-hour baseline, target +40–60% skill)
- [ ] Evaluate both on test set with proper baselines
- [ ] Add curriculum learning (24h → 3d → 7d)

### Phase C: Integration (Phase 3 Prep)
- [ ] Merge outputs into 18-column final forecast
- [ ] Add uncertainty quantification (optional ensemble)
- [ ] Build inference CLI
- [ ] Deploy

---

## Model 1: Marine iTransformer

**Purpose:** Predict physics-constrained ocean/radiation variables  
**Targets (8):**
1. tidal_residual_m
2. current_u_east_ms
3. current_v_north_ms
4. salinity_psu
5. water_temp_c
6. log1p_global_radiation_wm2
7. log_significant_wave_height_m
8. log_zero_crossing_period_s

**Baselines:**
- Tidal residual: UTide harmonic decomposition (fit on train only)
- Currents: persistence or tidal harmonic
- Radiation: previous-day same-time or clear-sky envelope
- Waves: log-space persistence
- Salinity: persistence
- Water temp: persistence or previous-day same-time

**Config (24-hour baseline):**
```yaml
seq_len: 672          # 7 days input
pred_len: 96          # 24 hours output
d_model: 64
n_heads: 4
e_layers: 2
d_ff: 128
dropout: 0.20
batch_size: 16
learning_rate: 3e-4
weight_decay: 1e-4
epochs: 40
early_stopping_patience: 6
```

**Expected skill:** +65–80% on all targets (conservative)

---

## Model 2: Atmospheric iTransformer

**Purpose:** Predict weather anomalies relative to harmonic baseline  
**Targets (5):**
1. air_temp_anomaly_c (= air_temp_c - harmonic_baseline)
2. log1p_dewpoint_depression_c (= log(max(air_temp - dew_point, 0.001)))
3. air_pressure_anomaly_hpa (= pressure - harmonic_baseline)
4. wind_u_anomaly_ms (= wind_u - climatology_u)
5. wind_v_anomaly_ms (= wind_v - climatology_v)

**Key insight:** Predict anomalies, not raw weather. This gives the model a fighting chance against atmospheric chaos.

**Baselines:**
- Daily + annual harmonic (fitted on train only)
- Zero anomaly (trivial baseline)
- Climatology for wind

**Config (24-hour baseline):**
```yaml
seq_len: 672          # 7 days input
pred_len: 96          # 24 hours output
d_model: 64
n_heads: 4
e_layers: 2
d_ff: 128
dropout: 0.25
batch_size: 16
learning_rate: 2e-4
weight_decay: 3e-4
epochs: 40
early_stopping_patience: 6
```

**Expected skill:** +30–60% at 24h, +10–30% at 72h (realistic for chaotic variables)

---

## Phase A: Implementation Steps

### Step 1: Create Anomaly Baseline Calculator
```python
# src/baselines.py - ADD to existing

class HarmonicAnomalyBaseline:
    """Fit daily + annual harmonic baseline on training data only."""
    
    def fit(self, timestamps, values, train_mask):
        """Fit 4-harmonic model on training data."""
        # Harmonic regression:
        # baseline = beta0 + beta1*sin(2*pi*hour/24) + beta2*cos(...)
        #          + beta3*sin(2*pi*day_of_year/365.25) + beta4*cos(...)
        pass
    
    def predict(self, timestamps):
        """Get baseline for any timestamp."""
        pass
```

### Step 2: Separate Preprocessing
```python
# src/preprocessing_marine.py (NEW)
# src/preprocessing_atmosphere.py (NEW)

# Each handles its 5–8 targets independently
# Both use TRAIN-ONLY fit for all scalers, baselines, calibrators
```

### Step 3: Two iTransformer Config Files
```yaml
# configs/marine_24h.yaml
targets:
  - tidal_residual_m
  - current_u_east_ms
  - current_v_north_ms
  - salinity_psu
  - water_temp_c
  - log1p_global_radiation_wm2
  - log_significant_wave_height_m
  - log_zero_crossing_period_s

# configs/atmosphere_24h.yaml
targets:
  - air_temp_anomaly_c
  - log1p_dewpoint_depression_c
  - air_pressure_anomaly_hpa
  - wind_u_anomaly_ms
  - wind_v_anomaly_ms
```

### Step 4: Validation Pipeline
```python
# tests/test_preflight_pipeline.py (NEW)

def test_target_order_marine():
    """Ensure canonical target order."""
    assert TARGET_MARINE == [
        'tidal_residual_m',
        'current_u_east_ms',
        'current_v_north_ms',
        'salinity_psu',
        'water_temp_c',
        'log1p_global_radiation_wm2',
        'log_significant_wave_height_m',
        'log_zero_crossing_period_s'
    ]

def test_scaler_marine_fit_on_train_only():
    """Ensure scaler fitted on train rows only."""
    pass

def test_transform_inverse_roundtrip():
    """Random values -> transform -> inverse -> error < 1e-8."""
    pass

def test_timestamp_alignment():
    """For every window: last(X.timestamp) + 15min == first(Y.timestamp)."""
    pass

def test_prediction_range_audit():
    """Output ranges must be plausible (pressure not in [1-10])."""
    pass
```

### Step 5: Horizon-Weighted Loss
```python
# src/train.py - UPDATE HuberLossWeighted

class HuberLossHorizonWeighted(nn.Module):
    def forward(self, pred, target, mask):
        # Horizon decay: 1.0 at t=0, 0.4 at t=168h
        horizon_decay = torch.linspace(1.0, 0.4, horizon, device=device)
        # Apply to loss
```

### Step 6: Run Validation
```bash
pytest -xvs tests/test_preflight_pipeline.py
```

---

## Files to Create/Modify

### New Files:
- `src/baselines.py` — Add `HarmonicAnomalyBaseline` class
- `src/preprocessing_marine.py` — Marine-specific pipeline
- `src/preprocessing_atmosphere.py` — Atmosphere-specific pipeline
- `configs/marine_24h.yaml` — Marine model config
- `configs/atmosphere_24h.yaml` — Atmosphere model config
- `tests/test_preflight_pipeline.py` — Validation suite
- `artifacts/target_columns_marine.json` — Canonical target order
- `artifacts/target_columns_atmosphere.json` — Canonical target order

### Modified Files:
- `src/train.py` — Add horizon-weighted loss
- `src/constants.py` — Add TARGET_MARINE, TARGET_ATMOSPHERE
- `run_training.py` → Split into `train_marine.py` and `train_atmosphere.py`

---

## Expected Outcomes

### Marine iTransformer (24h):
| Variable | Current Skill | Expected After | Reason |
|----------|---------------|-----------------|--------|
| tidal_residual_m | +96% | +96% | No change (already good) |
| current_u_east_ms | +95% | +95% | No change |
| salinity_psu | +88% | +85% | Slight change from retraining |
| water_temp_c | −2192% | +40% | Baseline subtraction helps |
| log_wave_height_m | +70% | +70% | No change |
| log_clearness_index | +95% | +95% | No change |

### Atmospheric iTransformer (24h):
| Variable | Current Skill | Expected After | Reason |
|----------|---------------|-----------------|--------|
| air_temp_anomaly_c | −5761% | +50% | Anomaly prediction easier |
| air_pressure_anomaly_hpa | −6974% | +35% | Anomaly prediction easier |
| wind_u_anomaly_ms | −2654% | +25% | Chaotic but anomaly helps |
| dew_point_depression | −6887% | +40% | log1p handles small values |

**All skill improvements are relative to synthetic data. Real NOAA data will show larger improvements.**

---

## Success Criteria Before Phase 3

✓ Marine model achieves +70% skill on ≥7/8 targets  
✓ Atmospheric model achieves +30% skill on ≥3/5 targets at 24h  
✓ All validation tests pass  
✓ No timestamp/ordering/scaling errors  
✓ Reconstructed RH ∈ [0, 100]  
✓ Dew point ≤ air temperature  
✓ Pressure predictions in plausible range [950–1050 hPa]

---

## Timeline

- **Hour 0–1:** Create baseline calculator + configs
- **Hour 1–2:** Split preprocessing pipelines
- **Hour 2–3:** Write validation tests
- **Hour 3–4:** Train Marine model (single GPU/CPU)
- **Hour 4–5:** Train Atmospheric model
- **Hour 5–6:** Evaluate both, generate reports
- **Hour 6:** Ready for Phase 3 integration

**Total: ~6 hours for full refactoring + retraining**

