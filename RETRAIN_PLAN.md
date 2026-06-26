# Plan to Add Missing Parameters and Retrain iTransformer

## Current Status

**iTransformer Current Configuration:**
- Input features (enc_in): 6 marine parameters
- Trained on: 110-80 days (old split)
- Not including:
  - conductivity_mscm (water quality)
  - global_radiation_wm2 (atmospheric)
  - peak_wave_period_s (wave energy)

---

## Step 1: Add conductivity_mscm to iTransformer

**Why:**
- Water quality indicator
- Correlates with salinity
- Could improve marine forecasting by 2-5%

**Current Status:**
- Present in CSV: YES (column 19)
- In iTransformer training: NO

**Action:**
```
Add to iTransformer input features:
  Old: 6 features (current, water_temp, salinity, tidal, wave_height, wave_period)
  New: 7 features + conductivity_mscm
  
Then retrain with new 3-way split:
  Training: 80 days (Feb 23 - May 13)
  Validation: 20 days (May 14 - Jun 2)
  Test: 7 days (Jun 3 - Jun 9)
```

---

## Step 2: global_radiation_wm2 (Keep with GraphCast)

**Why:**
- Solar radiation (atmospheric)
- Better handled by GraphCast (weather model)
- iTransformer is for marine, not solar

**Current Status:**
- Present in CSV: YES (column 9)
- In iTransformer: NO (correct - not marine)
- In GraphCast: Maybe (depends on pre-training)

**Action:**
```
KEEP with GraphCast or add solar forecast model
Do NOT add to iTransformer (architectural mismatch)
```

---

## Step 3: peak_wave_period_s (Add to iTransformer)

**Why:**
- Wave energy parameter
- Related to significant wave period
- Useful for ocean energy, extreme event detection

**Current Status:**
- Present in CSV: YES (column 26)
- In iTransformer: NO

**Action:**
```
Add to iTransformer output targets:
  Old: 2 targets (tidal_level, current)
  New: 3 targets + peak_wave_period_s
  
Retrain with extended features
```

---

## Summary: What to Add

| Parameter | Type | Add to iTransformer? | Effort | Impact |
|-----------|------|---------------------|--------|--------|
| **conductivity_mscm** | Water Quality | YES | 30 min | +2-5% skill |
| **global_radiation_wm2** | Atmospheric | NO (use GraphCast) | None | Existing |
| **peak_wave_period_s** | Wave Energy | YES | 30 min | +1-3% skill |

---

## Retraining Timeline

```
Step 1: Prepare data with new split
  Time: 5 minutes
  Output: Train/Val/Test datasets

Step 2: Modify iTransformer config
  Time: 10 minutes
  Change: Add conductivity input, peak_wave period output

Step 3: Retrain model
  Time: 45-60 minutes (CPU)
  Output: New best_model.pt

Step 4: Evaluate results
  Time: 10 minutes
  Output: New metrics (skill %, loss, etc.)

Step 5: Update dashboard tabs
  Time: 15 minutes
  Output: Updated Methodology, Training Results, Data Plots

TOTAL: ~2 hours (mostly retraining time)
```

---

## Expected Results After Retraining

**With conductivity_mscm added:**
- Salinity forecasting: 95.2% → 97%+ (better)
- Overall marine skill: 84.9% → 86-87%

**With peak_wave_period_s added:**
- Wave energy forecasting: NEW capability
- Wave modeling: +1-3% improvement

**Updated 3-Way Split:**
- Training: 80 days (67%) - Feb 23 to May 13
- Validation: 20 days (17%) - May 14 to Jun 2
- Testing: 7 days (6%) - Jun 3 to Jun 9
- Unused: 13 days - Jun 10 to Jun 22

---

## What to Update in Dashboard

After retraining:
1. **Training Results tab:**
   - New model parameters count
   - New training configuration (7 inputs instead of 6)
   - New validation metrics (higher skill %)
   - New per-parameter skills (conductivity, peak_wave)

2. **Data Plots tabs:**
   - Training period plots (same data, 80 days)
   - Validation period plots (reduced to 20 days)
   - Test period plots (7 days, completely new)

3. **Methodology tab:**
   - Updated configuration details
   - New 3-way split explanation
   - New training times
   - New expected results

4. **Skill Matrix tab:**
   - Add conductivity_mscm (row 19)
   - Add peak_wave_period_s (row 26)
   - Update Day 1-7 skills
   - Update 7-day averages

---

## Decision

**Recommendation:** GO AHEAD

YES, add:
- conductivity_mscm to iTransformer
- peak_wave_period_s as additional target

NO, don't add:
- global_radiation_wm2 (use GraphCast or separate solar model)

Retrain with:
- New 3-way split (80/20/7 days)
- Extended features (7 inputs)
- Extended targets (3 outputs)
- Same architecture (iTransformer)

This will:
- Improve model accuracy (+2-5%)
- Add missing capabilities
- Follow proper ML methodology
- Make dashboard fully comprehensive
