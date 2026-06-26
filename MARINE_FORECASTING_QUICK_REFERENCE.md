# MARINE FORECASTING 5-DAY PREDICTION: QUICK REFERENCE & VISUAL SUMMARY

## 🎯 EXECUTIVE SNAPSHOT

**Question**: What is the BEST technique to predict each marine parameter 5 days ahead?  
**Answer**: See the matrix below. **Single Best Approach**: LSTM backbone + parameter-specific variants

---

## 1. TECHNIQUE SELECTION MATRIX (5-Day Forecast)

```
┌────────────────────────────────────────────────────────────────────┐
│                    PARAMETER GROUP                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  🌊 WAVE PARAMETERS                                               │
│  ├─ Significant Wave Height (SWH)    → LSTM + VMD               │
│  ├─ Wave Period (Peak/Sig)            → LSTM / CNN-LSTM          │
│  └─ Forecast Horizon: 5–7 days       (hourly @ 30–60 min)       │
│                                                                    │
│  💨 WIND PARAMETERS                                               │
│  ├─ Wind Speed                        → Temporal Fusion Transformer  │
│  ├─ Wind Direction                    → Circular-aware LSTM      │
│  └─ Forecast Horizon: 5 days          (hourly @ 10 min)          │
│                                                                    │
│  🌊 WATER PARAMETERS                                              │
│  ├─ Tidal Level / Water Level         → VMD-LSTM + Harmonic     │
│  ├─ Tidal Current Speed               → Attention-ResNet (AR-ANN)│
│  ├─ Sea Surface Temperature           → CNN-LSTM / UNet-LSTM     │
│  ├─ Salinity / Conductivity           → CNN-LSTM                │
│  └─ Forecast Horizon: 3–5 days        (varies by parameter)     │
│                                                                    │
│  🌤️  ATMOSPHERIC PARAMETERS                                       │
│  ├─ Air Pressure                      → LSTM / XGBoost           │
│  ├─ Relative Humidity                 → LSTM / XGBoost           │
│  ├─ Dew Point Temperature             → Linear Regression + LSTM │
│  ├─ Precipitation Intensity           → ConvLSTM (nowcast best)  │
│  ├─ Global Solar Radiation            → LSTM / Ensemble          │
│  ├─ Visibility                        → Logistic Reg + LSTM      │
│  └─ Forecast Horizon: 1–5 days        (varies by parameter)     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. TECHNIQUES RANKED BY FORECAST HORIZON

### **1-3 Day Horizon** (Best Predictability)
1. **LSTM** (all parameters) – Stable, proven
2. **XGBoost** (wind, pressure, temperature)
3. **ARIMA** (tidal alone, pressure)
4. **Harmonic Analysis** (tidal deterministic component)

### **5-Day Horizon** (Recommended Balance)
1. **Temporal Fusion Transformer** (wind speed, multi-variable)
2. **LSTM + VMD preprocessing** (wave height, current speed)
3. **CNN-LSTM** (spatial-temporal: temperature, salinity)
4. **XGBoost ensemble** (wind speed with decomposition)
5. **VMD-LSTM + Harmonic** (tidal level + surge)

### **10+ Day Horizon** (Research Stage)
1. **Transformer** (self-attention handles long dependencies)
2. **Physics-Informed Neural Networks** (SST, seasonal)
3. **Hybrid Physics + ML** (improving for climate timescales)
4. ⚠️ **Warning**: Accuracy degrades rapidly; use conservative confidence intervals

---

## 3. DATA REQUIREMENTS CHEAT SHEET

```
                Minimum     Optimal     Frequency    Notes
Parameter       (days)      (days)      Required
═══════════════════════════════════════════════════════════════
Wave Height      30          90–180      Hourly      VMD preprocessing helps
Wind Speed       14          60          Hourly      Decomposition useful
Water Temp       30          90–180      Hourly      Seasonal: need 6–12 mo
Tidal Level      60          90          Hourly      Lunar cycle critical!
Current Speed    30          90          Hourly      Couples with tide
Air Pressure     14          30          Hourly      Proxy for weather
Salinity         30          90–180      6-hourly    Seasonal variation
Precipitation    30          90          Hourly      Nowcasting better
Visibility       30          60          Hourly      Limited predictability
```

**KEY**: Red cell (Tidal Level 60 days) is critical—lunar cycle ~28 days, so need ≥2 cycles.

---

## 4. MODEL ACCURACY BENCHMARKS (5-Day)

```
Parameter             Unit    Best Technique           Achievable Error
═══════════════════════════════════════════════════════════════════════════
Wave Height           m       LSTM+VMD                 MAE: 0.05–0.16 m
Wind Speed            m/s     Transformer              RMSE: 0.55–0.70 m/s
Water Temperature     °C      CNN-LSTM                 RMSE: <0.75°C
Tidal Level           m       VMD-LSTM+Harmonic        RMSE: 0.05–0.15 m
Current Speed         m/s     Attention-ResNet         RMSE: 0.1–0.3 m/s
Air Pressure          hPa     LSTM / XGBoost           RMSE: 1–3 hPa
Salinity              PSU     CNN-LSTM                 RMSE: 0.5–2 PSU
Wave Period           s       LSTM                     MAE: 0.27–0.49 s
Visibility            km      Logistic Reg             Categorical: 65–85%
Precipitation         mm/h    ConvLSTM (nowcast)       Limited >3 days
```

**Confidence**: ±1 standard deviation from these benchmarks based on 2024–2025 literature.

---

## 5. ONE-PAGE IMPLEMENTATION ROADMAP

```
WEEK 1: DATA & SETUP
├─ Download 90 days buoy data (NOAA, SmartAtlantic ERDDAP)
├─ Check completeness (<5% missing OK)
├─ Normalize features (MinMaxScaler per parameter)
└─ Create sequences (lookback=60, horizon=120 timesteps)

WEEK 2: MODEL TRAINING
├─ Train LSTM baseline
│  ├─ Architecture: 2–3 layers, 50–150 units, dropout 0.2
│  ├─ Loss: MSE; Optimizer: Adam (lr=0.001)
│  └─ Early stopping: patience=15 epochs
├─ Alternative models: CNN-LSTM, XGBoost for wind
└─ Validation: Time-based split (70/15/15)

WEEK 3: EVALUATION & TUNING
├─ Metrics: RMSE, MAE, MAPE per day (1–5)
├─ Hyperparameter search: Grid or Bayesian optimization
├─ Ensemble: Average 3–5 models with learned weights
└─ Target: Wave ±0.15m, Wind ±0.6 m/s, Tide ±0.10m

WEEK 4: DEPLOYMENT
├─ Save model, scalers, feature names
├─ Set up inference pipeline (batch hourly or real-time)
├─ Monitor drift: Compare recent vs. historical accuracy
└─ Retrain: Monthly or when RMSE degrades >10%
```

---

## 6. COMMON QUESTIONS ANSWERED

**Q: Why LSTM instead of simpler methods like ARIMA?**
A: LSTM handles non-linear, non-stationary marine data better. ARIMA effective only for tidal component alone.

**Q: Which parameter is hardest to predict?**
A: Precipitation & Visibility (random cloud/fog), then Current Speed. Easiest: Wave Height (driven by wind), Tidal Level (deterministic).

**Q: Can I predict 10+ days ahead accurately?**
A: Degradation rapid after 5 days (typical oceanography limitation). Use Transformer or Physics-Informed ML; expect 30–50% accuracy loss by day 10.

**Q: How much data do I need?**
A: Minimum 30–60 days; optimal 90–180 days. **Tidal Level exception**: Must have 60+ days (lunar cycle ~28 days).

**Q: Should I predict all parameters together or separately?**
A: **Together (multivariate)** is better—captures correlations (wind↔wave, temp↔pressure). Use single LSTM with multivariate input/output.

**Q: Which cloud/GPU do I need?**
A: LSTM training: CPU (12+ cores) fine for 90–180 days data. Transformer: GPU recommended (CUDA/NVIDIA). Inference: CPU sufficient.

**Q: How do I handle missing data?**
A: Forward-fill (tidal level, temperature—slow change) or linear interpolation (wind, waves). Discard >10% consecutive gaps.

**Q: Can I use other people's trained models?**
A: **NO**—each location/season different. Train on your own buoy data. Transfer learning emerging but still experimental.

---

## 7. TOOL SELECTION QUICK GUIDE

### **Python Frameworks**

| Task | Framework | License | Use Case |
|---|---|---|---|
| **LSTM Training** | TensorFlow/Keras | Apache 2.0 | Fast, batteries included |
| | PyTorch | BSD | Research, more flexible |
| **Preprocessing** | scikit-learn | BSD | Scaling, ensemble, XGBoost |
| **Transformer** | Hugging Face | Apache 2.0 | Pre-built TFT models |
| **VMD** | PyVMD | MIT | Decomposition preprocessing |
| **Deployment** | FastAPI / Flask | BSD | Production inference |

### **Data Sources**

| Source | Data Type | Format | Cost | URL |
|---|---|---|---|---|
| **NOAA NDBC** | Buoys, stations | CSV, NetCDF | Free | ndbc.noaa.gov |
| **SmartAtlantic** | Atlantic buoys | ERDDAP | Free | smartatlantic.ca/erddap |
| **Copernicus** | Satellite SST | NetCDF | Free | cmems.copernicus.eu |
| **ECMWF ERA5** | Reanalysis | NetCDF, GRIB | Free | cds.climate.copernicus.eu |

---

## 8. KEY PAPERS BY PARAMETER

### **Wave Height (SWH)**
- Bekiryazici et al. (2025) "Forecasting Significant Wave Height using RNN-LSTM Models" – GRU performs well across waters
- Shen et al. (2024) "VMD-CNN-BiLSTM for monsoon regions" – 15–25% accuracy improvement via decomposition

### **Wind Speed**
- ScienceDirect (Dec 2025) – Transformer consistently lowest MSE/RMSE across locations; self-attention key
- Liu et al. (2024) "Offshore wind speed forecast by seasonal ARIMA vs GRU/LSTM" – GRU competitive, lighter

### **Water Temperature / SST**
- Ham et al. (2022) "Deep learning model for global SST anomalies" – UNet-LSTM, 24-month forecast, RMSE <0.75°C
- SSTFormer (2026) – Physics-guided Transformer; emerging leader for seasonal/interannual

### **Tidal Level**
- Zhang et al. (2024) "VMD-LSTM for tidal level prediction" – Modal components via VMD reduce non-stationarity, improve forecast
- Nature Sci Reports (2025) "Coastal flood forecasting" – Harmonic + LSTM residual approach validated

### **Current Speed**
- Qian et al. (2022) "ELM-LSTM hybrid for tidal current" – Periodic (H-ELM) + random (LSTM) decomposition effective
- Liao et al. (2024) "Attention-ResNet for tidal currents with ROMS" – 42% RMSE reduction vs physics baseline

---

## 9. ARCHITECTURE TEMPLATES

### **Template A: Simple LSTM** (Recommended for quick start)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(100, activation='relu', return_sequences=True, 
         input_shape=(lookback, n_features)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(120)  # Predict 5 days (120 hours)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### **Template B: CNN-LSTM** (For spatial-temporal)

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Input, LSTM
from tensorflow.keras.models import Model

inputs = Input(shape=(lookback, n_features))
x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = LSTM(100, activation='relu')(x)
x = Dense(50, activation='relu')(x)
outputs = Dense(120)(x)
model = Model(inputs, outputs)
```

### **Template C: VMD Preprocessing + LSTM**

```python
from PyVMD import VMD

# Decompose signal
u, u_hat, omega = VMD(signal, alpha=2000, tau=0.01, K=4, DC=0, init=1, tol=1e-6)

# Train LSTM on each IMF separately, then ensemble
for imf in u:
    model_imf = Sequential([...])  # LSTM architecture
    model_imf.fit(X_train[imf], y_train, ...)
    
# Combine predictions
final_pred = np.mean([m.predict(...) for m in models])
```

---

## 10. MONITORING & MAINTENANCE

```
WEEKLY CHECKS
├─ Plot predictions vs actuals (rolling 7-day window)
├─ Check RMSE trends (should stay ±10% of baseline)
└─ Alert if RMSE spikes >20%

MONTHLY ACTIONS
├─ Retrain model with latest 90 days data
├─ Evaluate on hold-out test set
├─ Update feature importance / attention weights
└─ Log all model versions (Git LFS or MLflow)

QUARTERLY REVIEWS
├─ Compare different techniques (LSTM vs Transformer vs XGBoost)
├─ Assess data quality (outliers, drift)
├─ Adjust hyperparameters if performance plateaus
└─ Document lessons learned
```

---

## 11. COMMON ERRORS & FIXES

| Error | Cause | Fix |
|---|---|---|
| **RMSE → ∞ on test set** | Lookahead bias (shuffled data) | Use time-based split ONLY |
| **Perfect training, poor test** | Overfitting | Add dropout, regularization, early stopping |
| **Model fails in new season** | No seasonal indicator | Add month/quarter feature |
| **Predictions flat (no variance)** | Data scaled wrong | Scale features separately; inverse transform output |
| **NaN loss during training** | Learning rate too high | Reduce lr or clip gradients |
| **Slow inference (>1s/sample)** | Model too large | Quantize weights; use TensorFlow Lite |

---

## 12. EXPECTED OUTCOMES (5-Day Forecast)

| Metric | Target | Achievable | Notes |
|---|---|---|---|
| **Wave Height** | ±0.10 m | ±0.15 m | 95% CI assuming normal distribution |
| **Wind Speed** | ±0.5 m/s | ±0.6–0.7 m/s | Transformer achieves 0.55 RMSE |
| **Tidal Level** | ±0.08 m | ±0.12 m | Harmonic component ~0.02 m; surge adds error |
| **Water Temp** | ±0.5°C | ±0.75°C | Improves with spatial/climate data |
| **Current Speed** | ±0.15 m/s | ±0.2–0.3 m/s | Short-term coupled with tide |
| **Air Pressure** | ±1 hPa | ±2–3 hPa | Correlates with wind; synoptic scales |

**Interpretation**: "±0.15 m for wave height" = 68% of forecasts within 0.15 m (±1σ).

---

## GLOSSARY

- **LSTM**: Long Short-Term Memory; RNN for sequence learning
- **CNN-LSTM**: Convolutional + LSTM hybrid; spatial-temporal
- **Transformer**: Attention-based architecture; best for 5–10 day horizons
- **VMD**: Variational Mode Decomposition; decomposes signal into modes
- **RMSE**: Root Mean Squared Error; penalizes large errors
- **MAE**: Mean Absolute Error; interpretable in original units
- **Harmonic Analysis**: Deterministic tidal prediction via Fourier series
- **PINNs**: Physics-Informed Neural Networks; ML + differential equations
- **Nowcasting**: 0–6 hour forecasting (precipitation, visibility)
- **Ensemble**: Combine multiple models; often better than single model

---

## FINAL CHECKLIST: "AM I READY TO BUILD?"

- [ ] I have 90+ days of hourly buoy data
- [ ] Data is <5% missing, quality-checked
- [ ] I understand LSTM basics (look at Colah's blog)
- [ ] I've set up Python environment (TensorFlow/PyTorch)
- [ ] I can clone a GitHub repo and modify it
- [ ] I have computational resources (CPU for LSTM; GPU for Transformer)
- [ ] I know my target accuracy (±0.15 m for waves, ±0.6 m/s for wind, etc.)
- [ ] I have a plan to retrain monthly
- [ ] I have monitoring/alerting for model drift

**If YES to all → You're ready to start!**  
**If NO → Review sections 1–4 again; don't skip data & setup.**

---

**Version**: 1.0 | **Last Updated**: June 21, 2026  
**Prepared by**: Comprehensive Marine Forecasting Research Synthesis  
**For**: Ship Mooring & Docking Operations

