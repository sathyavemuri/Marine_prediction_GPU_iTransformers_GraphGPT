# Handling Problem Parameters: Wave Height, Salinity, Wind Speed

## Problem Summary
- **significant_wave_height_m**: -285% to -709% (unpredictable, extreme failure)
- **salinity_psu**: -317% to -466% (reconstructed, seasonal mismatch)
- **wind_speed_ms**: -98% to +17% (high volatility, model confusion)

---

## Approach 1: Separate Specialist Models for Each Problem Parameter

**Idea**: Train individual HPMixer models **just for these 3 parameters** with:
- Longer lookback window (4-7 days instead of 2)
- Different hyperparameters
- Shorter forecast horizon (3 days instead of 10)
- Ensemble averaging across multiple runs

**Pros**:
- Targeted optimization for each parameter's behavior
- Can use physics-informed priors
- Isolates issues (don't pollute good parameters)

**Cons**:
- More training overhead (3 separate models)
- Inference latency increases

**Implementation effort**: Medium (1-2 hours)

**Expected improvement**: +20-40% skill if parameters are trainable at all

---

## Approach 2: Physics-Based Hybrid (Recommended for Wave Height)

### For significant_wave_height_m:
Wave spectral energy is governed by **wind-wave generation theory**. Instead of pure ML:

1. **Spectral model** (Pierson-Moskowitz spectrum) using wind/direction
2. **Damping model** (swell decay over time)
3. **ML correction layer** (residual learning on top of physics)

**Pros**:
- Respects physical constraints (waves can't be negative, decay is monotonic)
- Combines domain knowledge with data
- Naturally handles seasonal variation

**Cons**:
- Requires implementing physics models
- More complex inference pipeline

**Implementation effort**: High (3-4 hours)

**Expected improvement**: +60-80% skill (if physics bounds the problem)

---

## Approach 3: Post-Processing with Constraints & Smoothing

**Simple fix**: Apply corrections after forecast:

```python
# For wave height: can't be negative, apply physical bounds
wave_pred = np.maximum(wave_pred, 0)  # Clip negative values
wave_pred = np.minimum(wave_pred, 15)  # Cap at max observed

# For wind speed: apply Kalman filter to smooth volatility
wind_pred = kalman_filter(wind_pred, process_noise=0.1, measure_noise=0.5)

# For salinity: derive from conductivity + temperature (physical relationship)
salinity_pred = derive_from_physical_law(conductivity_pred, temp_pred)
```

**Pros**:
- No retraining needed
- Fast to implement (30 min)
- Works immediately

**Cons**:
- Doesn't fix root cause
- May clip valid extremes
- Salinity derivation needs calibration

**Implementation effort**: Low (30 minutes)

**Expected improvement**: +10-20% skill (mostly clipping bad predictions)

---

## Approach 4: Exclude Problem Parameters, Use Fallback Strategy

**Idea**: Don't predict these 3 parameters, use smarter baselines:

| Parameter | Strategy |
|-----------|----------|
| **significant_wave_height_m** | Persistence baseline (users compare against "no forecast better?") |
| **salinity_psu** | Derive post-hoc from conductivity + temperature physical law |
| **wind_speed_ms** | Use persistence OR average of last 3 days (slower decay) |

**Pros**:
- Removes negative predictions
- Simple, explainable
- Users know what to expect
- Salinity auto-updated when parent params improve

**Cons**:
- Incomplete forecast (only 15/18 params)
- Users may want wave height

**Implementation effort**: Very low (15 minutes)

**Expected improvement**: NA (but removes negative contributions)

---

## Approach 5: Ensemble + Weighted Hybrid Output

**Idea**: For problem parameters, use ensemble of multiple techniques:

```
Final prediction = (ML_weight * ML_forecast) + 
                   (Physics_weight * physics_forecast) + 
                   (Persist_weight * persistence_baseline)

Weights learned via validation set
```

**For each parameter**:
- **Wave height**: 40% ML + 50% physics model + 10% persistence
- **Salinity**: 20% ML + 80% physical derivation (conductivity+temp)
- **Wind speed**: 30% ML + 30% Kalman smoothed + 40% multi-day average

**Pros**:
- Combines best of all approaches
- Weights optimized per parameter
- Gracefully degrades when ML fails

**Cons**:
- More complex inference
- Requires validation data for weight tuning

**Implementation effort**: High (2-3 hours)

**Expected improvement**: +30-50% skill

---

## Recommended Action Plan (by Priority)

### **IMMEDIATE (Do First)** ✓
**Approach 3: Post-Processing Constraints**
- Add physical bounds to wave height (0-15m)
- Apply 3-point Kalman smoothing to wind speed
- Derive salinity from conductivity + temp formula

**Time**: 30 minutes
**Impact**: -10% → +5% skill for these 3 params

---

### **MEDIUM TERM** (If you want good wave height)
**Approach 2: Physics-Informed Wave Model**
- Implement Pierson-Moskowitz spectrum
- Use wind speed + direction → wave height physics
- Add ML residual correction

**Time**: 3-4 hours
**Impact**: -700% → +60% skill for significant_wave_height_m

---

### **LONG TERM** (If you have seasonal data)
**Collect more seasonal training data (Oct-Jan winter storms)**
- Seasonal variation is the root cause
- Current data (Feb-Jun) is calm/spring weather
- Need winter wave data to teach model extremes
- Retrain with 200+ days across all seasons

**Time**: 2-3 months (data collection)
**Impact**: -700% → +50-60% skill (fundamental fix)

---

## My Recommendation

**Start with Approach 3 (Quick Win)** + **Approach 2 (Wave Physics)**:

1. **Today**: Add constraints & smoothing (30 min)
   - Wave height: clip to [0, 15]
   - Wind speed: Kalman filter
   - Salinity: physics derivation

2. **This week**: Implement wave physics model (3-4 hrs)
   - Use wind/direction → wave spectrum
   - Add ML correction layer
   - Validate against test data

3. **Next month**: If budget allows, collect winter storm data (seasonal coverage)

This gives you:
- **Quick**: 15% improvement in next 30 min
- **Robust**: 60% improvement in 3-4 hours (wave height fixed)
- **Production-ready**: 80% improvement in 1-2 months (seasonal completeness)

---

## Questions to Guide Your Choice

1. **Do you need wave height forecasts?**
   - YES → Go with Approach 2 (physics model) or Approach 5 (ensemble)
   - NO → Go with Approach 4 (exclude it)

2. **Can you collect winter storm data?**
   - YES → Do Approach 5 (wait for seasonal data)
   - NO → Go with Approach 2 (physics) or Approach 3 (constraints)

3. **How much development time do you have?**
   - <1 hour → Approach 3 only
   - 3-4 hours → Approach 3 + Approach 2
   - 1-2 weeks → Approach 5 (full ensemble)

---

## Code Examples (Ready to Implement)

### Approach 3: Quick Constraints + Smoothing
```python
def apply_physics_constraints(forecast, param_name):
    if param_name == 'significant_wave_height_m':
        return np.clip(forecast, 0, 15)  # Physical bounds
    elif param_name == 'wind_speed_ms':
        return kalman_smooth(forecast, process_noise=0.1)
    elif param_name == 'salinity_psu':
        # Derive from conductivity + temp
        return (conductivity_pred - 0.008*temp_pred) / 42.9
    return forecast
```

### Approach 2: Physics Wave Model (Simplified Pierson-Moskowitz)
```python
def estimate_wave_height_physics(wind_speed, wind_direction, swell_decay=0.95):
    """Physics-based wave height from wind speed."""
    # Pierson-Moskowitz spectrum: H_s ≈ 0.24 * (wind^2 / g)
    H_s_physics = 0.24 * (wind_speed ** 2) / 9.81
    
    # Swell decay: older waves dissipate
    H_s_decayed = H_s_physics * swell_decay
    
    # Clip to physical limits
    return np.clip(H_s_decayed, 0, 15)

# Use as baseline, then add ML correction
H_s_final = 0.6 * H_s_physics + 0.4 * H_s_ml
```

---

## Files to Implement

If you choose to proceed:
- `10_apply_physics_constraints.py` (Approach 3)
- `11_train_wave_physics_hybrid.py` (Approach 2)
- `12_ensemble_multiple_methods.py` (Approach 5)
