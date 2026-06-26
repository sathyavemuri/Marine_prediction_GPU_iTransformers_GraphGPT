# Marine Ship-Mooring Parameter Forecasting: Comprehensive 5-Day Prediction Guide

**Document Date**: June 21, 2026  
**Scope**: Ship mooring & docking, marine operational safety  
**Data Source**: NOAA buoys, EMS sensors, SmartAtlantic datasets  
**Target Horizon**: 5-day forecast (primary), with guidance for 10+ days  

---

## EXECUTIVE SUMMARY

This guide synthesizes **latest 2024-2025 research** on forecasting marine parameters critical to ship mooring and docking operations. For **5-day predictions across all 14 parameters**, a **unified LSTM backbone with hybrid technique selection per parameter** is recommended:

| Parameter Group | Best Single Technique | Alternative |
|---|---|---|
| **Wave/Current** | LSTM + VMD preprocessing | CNN-LSTM, Transformer (10+ days) |
| **Wind Parameters** | Temporal Fusion Transformer | XGBoost, LSTM+ensemble |
| **Temperature/Pressure** | LSTM / CNN-LSTM | Transformer (longer horizons) |
| **Tidal/Level** | VMD-LSTM + Harmonic Analysis | ARIMA (tidal component alone) |

**Key Finding**: Transformer models with self-attention achieve the lowest MSE (0.3–0.5) and RMSE (0.55–0.70), outperforming LSTM and BiLSTM for 5-10 day wind speed forecasts, but **LSTM remains most stable and practical** for real-time operational deployment.

---

## 1. PARAMETER SELECTION & FORECASTING STRATEGIES

### 1.1 HIGH-PRIORITY PARAMETERS (Ship Safety Critical)

#### **SIGNIFICANT WAVE HEIGHT (SWH)** – 0.05–0.16 m MAE Achievable
- **Best Technique**: LSTM and GRU models show good results with window sizes of 4–7 days, with MAE results within 0.161 m to 0.051 m for wave heights
- **Why**: Nonlinear, non-stationary; LSTM handles long-term dependencies
- **Data Required**: 30–60 days minimum (better: 90–180 days)
- **Practical Implementation**:
  - Use **VMD (Variational Mode Decomposition)** as preprocessing
  - Train separate LSTM per IMF (Intrinsic Mode Function)
  - Ensemble predictions from 3–5 VMD modes
  - Rolling decomposition method avoids information leakage while retaining preprocessing benefits

**GitHub Templates**:
- `anandlo/Ocean-Wave-Height-Prediction-with-LSTM` (60-timestep sequences)
- `SmartAtlantic/ERDDAP` (real buoy data with wave heights)

---

#### **WIND SPEED** – RMSE 0.55–0.70 m/s (5-day)
- **Best Technique**: Transformer consistently produced the lowest MSE (0.3–0.5) and RMSE (0.55–0.70), showing its ability to capture short-term temporal dependencies, with XGBoost following closely
- **Why**: Transformer self-attention captures multi-scale wind patterns; XGBoost handles non-linearities
- **Data Required**: 14–30 days minimum (better: 60+ days)
- **Practical Implementation**:
  - **Approach A (Transformer-based)**: Use Temporal Fusion Transformer (TFT) for 5–10 day horizons
  - **Approach B (XGBoost ensemble)**: Lagged features (wind speed, pressure, humidity, temperature, altitude) → XGBoost → RMSE ~0.60 m/s
  - **Preprocessing**: Normalize features separately; include seasonal indicators

**GitHub Templates**:
- `fengjiqiang/LSTM-Wind-Speed-Forecasting` (basic LSTM)
- `blackeye735/Wind-Speed-Prediction` (TensorFlow implementation)
- `ArielDrabkin/Wind-Speed-Predictor` (CNN-LSTM 24h forecast)

---

#### **TIDAL LEVEL / WATER LEVEL** – RMSE 0.05–0.15 m
- **Best Technique**: VMD decomposes tidal level into multiple modal components reflecting different timescales; LSTM predicts each component independently, enhancing accuracy by mitigating prediction errors from level fluctuations
- **Why**: Tides = 80–90% periodic (lunar/solar) + 10–20% residual (surge, wind)
- **Data Required**: 60–90 days minimum (lunar cycle critical: ~28 days)
- **Practical Implementation**:
  1. **Harmonic Analysis** (classical): Predict tidal component deterministically 5–14 days ahead (achieves ~0.02 m RMSE)
  2. **Residual Surge** (ML): LSTM on residuals (tide - harmonic) → captures storm surge, wind effect
  3. **Combine**: Harmonic + LSTM residual = total water level

**Key Papers**:
- Zhang et al. (2024) VMD-LSTM, Ocean Engineering
- Nature Scientific Reports (2025) Coastal flood forecasting

---

### 1.2 SECONDARY PARAMETERS (Operational Efficiency)

#### **WATER TEMPERATURE (SEA SURFACE TEMP)** – RMSE < 0.75°C
- **Best Technique**: Unet-LSTM models learn underlying physics of 2D global SST; accurately predict over 24 months with RMSE remaining below 0.75°C
- **Why**: Spatial structure (CNN U-Net) + temporal (LSTM)
- **Data Required**: 30–90 days (diurnal); 6–12 months (seasonal)
- **Practical Implementation**:
  - CNN U-Net for spatial patterns (grid data)
  - LSTM for temporal evolution
  - Include auxiliary: air temp, wind stress, ENSO index
  - Physics-informed ML combining ML with physical constraints (ODEs/PDEs) emerges as rapidly evolving research enabling mesh-less regression

---

#### **CURRENT SPEED** – RMSE 0.1–0.3 m/s
- **Best Technique**: Attention-ResNet (AR-ANN) shows excellent performance in both meridional and zonal components; reduces RMSE by 42% vs ROMS baseline
- **Why**: Quasi-periodic (lunar tidal) + random (wind, turbulence)
- **Data Required**: 30–90 days
- **Practical Implementation**:
  - Decompose: Periodic (H-ELM) + Random (LSTM)
  - H-ELM (Hierarchical Extreme Learning Machine): Fast, captures periodic trends
  - LSTM: Captures random/turbulent fluctuations
  - Baseline: Use ROMS (Regional Ocean Modeling System) hydrodynamic model

**Key Papers**:
- Qian et al. (2022) ELM-LSTM hybrid, Ocean Engineering
- Liao et al. (2024) AR-ANN with ROMS, J. Marine Science & Engineering

---

### 1.3 AUXILIARY PARAMETERS (Context/Feature Enhancement)

| Parameter | Best Technique | Notes |
|---|---|---|
| **Wind Direction** | Circular-aware LSTM | Requires circular preprocessing (sin/cos encoding) |
| **Air Pressure** | LSTM / XGBoost | Proxy for weather patterns; correlates with wind |
| **Salinity/Conductivity** | CNN-LSTM | Responds to freshwater input, tides, mixing |
| **Precipitation** | ConvLSTM / 3D-ConvLSTM | Limited skill >3 days; nowcasting 0–6h more practical |
| **Wave Period** | LSTM / CNN-LSTM | Couples with SWH and wind; spectral methods useful |
| **Visibility** | Logistic Regression + LSTM | Often categorical; fog/haze prediction difficult |
| **Solar Radiation** | LSTM / Ensemble | Diurnal cycle regular; cloud cover main challenge |
| **Relative Humidity** | LSTM / XGBoost | Derives from temperature & dew point |
| **Dew Point** | Linear regression + LSTM | Highly correlated with temperature; physics-based regression effective |

---

## 2. UNIFIED IMPLEMENTATION FRAMEWORK

### 2.1 Data Pipeline

```
┌─────────────────────────────────────────┐
│ 1. DATA COLLECTION (90–180 days)        │
│    ├─ NOAA buoys (NDBC)                │
│    ├─ SmartAtlantic ERDDAP server      │
│    └─ EMS sensor network               │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 2. PREPROCESSING                        │
│    ├─ Handle missing values (forward-fill, interpolation) │
│    ├─ Outlier detection (IQR, isolation forest) │
│    ├─ Normalize: MinMaxScaler per feature │
│    ├─ Detrend if needed                │
│    └─ Check stationarity (ADF test)    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING                  │
│    ├─ Lagged features (t-1, t-2, ... t-k) │
│    ├─ Seasonal indicators (month, day)  │
│    ├─ Cyclical encoding (sin/cos for wind direction) │
│    ├─ Rolling statistics (mean, std)    │
│    └─ Fourier features (spectral)       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 4. SEQUENCE CREATION                    │
│    ├─ Look-back: 30–60 timesteps       │
│    ├─ Forecast horizon: 120 timesteps (5 days @ hourly) │
│    └─ Train/val/test split: 70/15/15  │
└─────────────────────────────────────────┘
```

### 2.2 Model Architecture Recommendation

#### **OPTION A: Unified LSTM Pipeline (RECOMMENDED for 5-day)**

```python
# Pseudocode
class MarineForecaster:
    def __init__(self, lookback=60, horizon=120):
        self.lookback = lookback
        self.horizon = horizon
    
    def build_lstm(self, n_features):
        model = Sequential([
            LSTM(100, activation='relu', return_sequences=True, 
                 input_shape=(self.lookback, n_features)),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(self.horizon)  # Multivariate output (all 14 parameters)
        ])
        return model
    
    def build_cnn_lstm(self, n_features):
        # CNN layer for spatial patterns
        inputs = Input(shape=(self.lookback, n_features))
        x = Conv1D(64, 3, activation='relu')(inputs)
        x = MaxPooling1D()(x)
        x = LSTM(100, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        outputs = Dense(self.horizon)(x)
        model = Model(inputs, outputs)
        return model
    
    def build_transformer(self, n_features):
        # Temporal Fusion Transformer for longer horizons
        # Use Hugging Face library or PyTorch implementation
        pass
```

#### **OPTION B: Parameter-Specific Ensemble**

```
┌─────────────────────────────────────────┐
│ MULTIVARIATE INPUT (all 14 params)      │
└─────────────────────────────────────────┘
         ↓         ↓         ↓
┌────────┴───┬─────┴────┬────┴─────┐
│  LSTM      │ XGBoost  │ Transformer
│ (Wave)     │ (Wind)   │ (Tides)
└────────┬───┴─────┬────┴────┬─────┘
         ↓         ↓         ↓
┌─────────────────────────────────────────┐
│ WEIGHTED ENSEMBLE (Weights tuned by    │
│ validation set performance)            │
│ Prediction = w1*LSTM + w2*XGB + w3*TFT │
└─────────────────────────────────────────┘
```

---

## 3. TRAINING & HYPERPARAMETER TUNING

### 3.1 Data Requirements Summary

| Parameter | Min. Historical | Optimal | Forecast Days | Hourly Resolution |
|---|---|---|---|---|
| Wave Height | 30 days | 90–180 days | 5–7 | 30–60 min |
| Wind Speed | 14 days | 60 days | 5 | 10 min |
| Water Temp | 30 days | 90 days (seasonal: 180–365) | 5 | Hourly |
| Tidal Level | **60 days** | 90 days (lunar cycle ~28 days) | 5–14 | Hourly |
| Current Speed | 30 days | 90 days | 1–3 | Hourly |
| Salinity | 30 days | 60 days (seasonal: 180 days) | 3–5 | 6-hourly |

### 3.2 LSTM Hyperparameter Ranges

```python
# Empirically validated ranges for marine parameters
hyperparameters = {
    'lstm_units': [50, 100, 150],          # Larger for complex multi-scale features
    'lstm_layers': [1, 2, 3],              # 2–3 optimal; >3 diminishing returns
    'dropout': [0.1, 0.2, 0.3],            # Prevent overfitting
    'learning_rate': [0.001, 0.01],        # Start low, increase if needed
    'batch_size': [16, 32, 64],            # Smaller for noisy marine data
    'epochs': [50, 100, 200],              # Early stopping at validation plateau
    'optimizer': ['adam', 'rmsprop'],      # Adam typically best
}

# Validation approach
train_size = 0.7
val_size = 0.15
test_size = 0.15
cv_folds = 5  # k-fold cross-validation
```

### 3.3 Training Strategy

1. **Data Splitting**: Time-based (80/10/10) to avoid lookahead bias
2. **Normalization**: Per-feature MinMaxScaler; inverse-transform for metrics
3. **Loss Function**: MSE for regression; MAE for robustness to outliers
4. **Monitoring**: Early stopping on validation MAE; reduce LR on plateau
5. **Evaluation**:
   - **Primary**: RMSE, MAE (interpretable in original units)
   - **Secondary**: MAPE, Correlation, Directional Accuracy (for angles)
   - **Horizon-specific**: Separate metrics for each day (1–5)

---

## 4. LATEST RESEARCH HIGHLIGHTS (2024–2025)

### 4.1 Key Breakthrough Techniques

#### **1. Variational Mode Decomposition (VMD) + LSTM**

Recent work using rolling VMD decomposition for wind speed prediction shows that combining periodic and random component decomposition with GRU enhances predictive accuracy by avoiding information leakage while retaining preprocessing benefits

**Why It Works**: Separates multi-scale oscillations → LSTM trains on cleaner sub-series

**Applications**: Wave height, wind speed, tidal level, salinity

**Reference**: Shen et al. (2024), Frontiers in Marine Science

---

#### **2. Temporal Fusion Transformer (TFT)**

Among wind speed forecasting models tested, Transformer consistently produced the lowest MSE (0.3–0.5) and RMSE (0.55–0.70) compared to LSTM, BiLSTM, RNN, and ARIMA, with the Transformer leveraging self-attention for capturing temporal dependencies across multiple timescales

**Why It Works**: Self-attention learns which historical timesteps matter most

**Best For**: 5–10 day forecasts of wind, pressure, SST

**Reference**: ScienceDirect (2025), Multiple authors

---

#### **3. Physics-Informed Neural Networks (PINNs)**

Physics-informed ML combines ML with physical constraints from ODEs/PDEs, enabling mesh-less regression using available observations without interpolation to grids; this approach shows promise for correcting model biases and capturing diverse SST variability

**Why It Works**: Incorporates oceanographic equations → more robust extrapolation

**Best For**: Long-term SST (30+ days), seasonal tidal predictions

**Reference**: Karniadakis et al. (2021), Ham et al. (2022)

---

#### **4. Conformer & Hybrid Attention Models**

A Hybrid Conformer-LSTM model with adaptive feature fusion weight network outperformed traditional LSTM, CNN, and CNN-LSTM models at multiple NOAA buoy stations for wave height prediction

**Why It Works**: Combines convolutional feature extraction with transformer attention

**Best For**: Multi-scale wave patterns, complex interactions

**Reference**: MDPI (2024)

---

### 4.2 Top Recent Papers & Journals

**MUST-READ Papers** (2024–2025):

1. **Wave Height**:
   - Shen et al. (2024) "VMD-CNN-BiLSTM for monsoon regions" – *Frontiers Marine Science*
   - Bekiryazici et al. (2025) "GRU performance in different waters" – *Ocean Engineering* (Feb 2025)

2. **Wind Speed**:
   - Shen et al. (2025) "VMD-TCN-BiLSTM for offshore wind" – *Applied Energy* (June 2025)
   - Liu et al. (2025) "Transformer review for wind forecasting" – *ScienceDirect* (Dec 2025)

3. **Sea Surface Temperature**:
   - SSTFormer (2026) "Physics-guided Transformer for global SST" – *ScienceDirect*
   - DUNE (2024) "Deep UNet++ for seasonal/annual forecasting" – *arXiv*

4. **Tides & Currents**:
   - Zhang et al. (2024) "VMD-LSTM for tidal level" – *Water* (MDPI)
   - Liao et al. (2024) "Attention-ResNet for tidal currents with ROMS" – *J. Marine Sci. Eng.*

**Top Journals** (Q1–Q2):

| Journal | IF | Focus |
|---|---|---|
| Ocean Engineering | 4.0–4.5 | Waves, forecasting, applied engineering |
| Applied Energy | 8.5–10.0 | Wind, solar, hybrid models |
| Renewable Energy | 6.0–7.5 | Wind, marine energy |
| J. Marine Science & Engineering | 2.7–3.1 | All marine parameters; open access |
| Frontiers in Marine Science | 3.2–3.8 | Interdisciplinary; rapid publication |
| Nature Climate Change | 18+ | Climate, SST, ENSO |
| Weather & Forecasting | 3.0–3.5 | Short-term forecasting methods |

---

## 5. COMMON PITFALLS & SOLUTIONS

| Pitfall | Consequence | Solution |
|---|---|---|
| **Insufficient data** (< 30 days) | High variance, poor generalization | Collect minimum 60–90 days; use data augmentation if needed |
| **Not handling non-stationarity** | Model assumes constant mean/variance | Use differencing, detrending, or VMD preprocessing |
| **Lookahead bias** | Overly optimistic results | Time-based train/val/test split (no shuffling) |
| **Single model** | Model-specific failures on unseen patterns | Ensemble 3–5 models with weighted averaging |
| **Ignoring seasonality** | Poor performance in off-season | Include seasonal indicators or train separate models per season |
| **Raw scaling** | Features with large magnitude dominate | MinMaxScaler (0–1) or StandardScaler per feature |
| **Overfitting** (few epochs, no dropout) | Perfect training, poor test performance | Early stopping, dropout (0.2–0.3), regularization (L1/L2) |
| **Multicollinearity** | Unstable coefficients, overfitting | PCA, feature selection, or accept correlation (RNNs robust) |

---

## 6. RECOMMENDED GITHUB REPOSITORIES & QUICK START

### **High-Priority Repos** (Start Here)

1. **Wave Height** (RECOMMENDED)
   ```
   Repository: anandlo/Ocean-Wave-Height-Prediction-with-LSTM
   Language: Python (TensorFlow/Keras)
   Dataset: SmartAtlantic Halifax Buoy
   Architecture: LSTM (2 layers, 50 units), 60-timestep sequences
   Quick Start: Clone → Update data source → Train
   ```

2. **Wind Speed** (RECOMMENDED)
   ```
   Repository: fengjiqiang/LSTM-Wind-Speed-Forecasting
   Language: Python
   Architecture: Basic LSTM (easy to modify)
   Advantage: Simple, easy to understand and adapt
   ```

3. **Wind Speed + CNN** (Advanced)
   ```
   Repository: ArielDrabkin/Wind-Speed-Predictor
   Language: Python (PyTorch)
   Architecture: CNN-LSTM encoder/decoder
   Features: 24-hour forecast, CNN feature extraction
   Best For: Building production models
   ```

4. **Data Source** (CRITICAL)
   ```
   SmartAtlantic ERDDAP Server
   URL: https://www.smartatlantic.ca/erddap/
   Data: Wave height, wind, temperature, current, pressure
   Format: CSV, NetCDF
   Advantage: Real buoy data, quality-controlled
   ```

### **Specialized Repos**

| Need | Repository | Language |
|---|---|---|
| CNN-LSTM architecture | ozancanozdemir/CNN-LSTM | PyTorch |
| Transformer baseline | huggingface/transformers | PyTorch |
| Hydrodynamic modeling | ROMS (open-source) | FORTRAN |
| Time series templates | PyTorch/fastai | PyTorch |
| VMD preprocessing | PyVMD | Python |

---

## 7. QUICK IMPLEMENTATION CHECKLIST

```
DATA & SETUP
□ Collect 90–180 days of hourly buoy data (NOAA, SmartAtlantic)
□ Verify data completeness (< 5% missing acceptable)
□ Check temporal resolution (hourly required for wave, wind)
□ Handle missing values (forward-fill or interpolation)

PREPROCESSING
□ Normalize each feature independently (MinMaxScaler)
□ Detrend if needed (differencing, seasonal decomposition)
□ Check stationarity (ADF test)
□ Create sliding window sequences (lookback=60, horizon=120)

MODEL TRAINING
□ Split data: 70 train / 15 val / 15 test (time-based)
□ Build LSTM (2–3 layers, 50–150 units, dropout 0.2–0.3)
□ Alternative: CNN-LSTM or Transformer for comparison
□ Train with early stopping (patience=10–20 epochs)

EVALUATION
□ Compute RMSE, MAE, MAPE on test set
□ Separate metrics per forecast day (1–5)
□ Check for model drift (recent data vs. old)
□ Visualize: predictions vs. actuals, error distribution

DEPLOYMENT
□ Save model, scaler, feature names to production
□ Set up inference pipeline (batch or real-time)
□ Monitor prediction quality weekly (drift detection)
□ Retrain monthly or when RMSE degrades >10%
```

---

## 8. ESTIMATED ACCURACY BENCHMARKS

These ranges reflect **validated literature results** for 5-day forecasts with **60–90 days training data**:

| Parameter | Unit | Best Technique | MAE / RMSE | Notes |
|---|---|---|---|---|
| **Wave Height** | m | LSTM+VMD | 0.05–0.16 m | Excellent; RMSE 0.10–0.25 |
| **Wind Speed** | m/s | Transformer/XGBoost | 0.1–0.2 RMSE 0.55–0.70 | Transformer best; RMSE improves to 0.55 |
| **Water Temp** | °C | CNN-LSTM | RMSE < 0.75 | Accuracy improves with spatial data |
| **Tidal Level** | m | VMD-LSTM+Harmonic | 0.05–0.15 m | Tidal ~0.02 m; surge adds error |
| **Current Speed** | m/s | AR-ANN (Attention) | 0.1–0.3 m/s | Correlates with wind; couples with tide |
| **Salinity** | PSU | CNN-LSTM | RMSE 0.5–2 PSU | Longer memory; seasonal variation key |
| **Wave Period** | s | LSTM | MAE 0.27–0.49 s | Couples with height and wind |
| **Visibility** | km | Logistic Reg + LSTM | Categorical 0.65–0.85 | Limited predictability; nowcasting better |

---

## 9. REFERENCES & CITATION GUIDE

### **Seminal Works**

- **Hochreiter & Schmidhuber (1997)** "Long Short-Term Memory" – Neural Computation, Vol. 9, No. 8
  - Foundation of LSTM architecture; cite for LSTM justification

- **Vaswani et al. (2017)** "Attention Is All You Need" – NeurIPS
  - Transformer architecture; cite for Transformer-based models

- **Dragomiretskiy & Zosso (2014)** "Variational Mode Decomposition" – IEEE TSP
  - VMD theory; cite for preprocessing justification

### **Recent Marine Forecasting** (2024–2025)

- Shen et al. (2024) "VMD-CNN-BiLSTM for significant wave height in monsoon regions" – *Frontiers in Marine Science*
- Bekiryazici et al. (2025) "Forecasting Significant Wave Height using RNN-LSTM Models" – *Ocean Engineering*
- Shen et al. (2025) "Multivariate VMD-TCN-BiLSTM for offshore wind speed" – *Applied Energy*
- Zhang et al. (2024) "Tidal Level Prediction Model Based on VMD-LSTM Neural Network" – *Water* (MDPI)

### **Where to Find Papers**

- **Google Scholar**: scholar.google.com (open-access filter available)
- **ResearchGate**: researchgate.net (authors often share PDFs)
- **arXiv**: arxiv.org (preprints in ML, physics)
- **ScienceDirect**: sciencedirect.com (Elsevier journals)
- **MDPI**: mdpi.com (open-access journals with low fees)
- **Nature Portfolio**: nature.com (high-impact, paywalled)

---

## 10. CONCLUSION & NEXT STEPS

### **For Ship Mooring/Docking Operations:**

1. **Immediate (Week 1–2)**:
   - Collect 90 days of buoy data from SmartAtlantic or NOAA
   - Implement baseline LSTM using `anandlo/Ocean-Wave-Height-Prediction-with-LSTM`
   - Evaluate on validation set; target RMSE for wave height < 0.20 m

2. **Short-term (Month 1–2)**:
   - Build multi-parameter ensemble (LSTM wave + XGBoost wind + ARIMA tide)
   - Achieve 5-day accuracy: wave ±0.15 m, wind ±0.6 m/s, tide ±0.10 m
   - Deploy real-time inference pipeline

3. **Medium-term (Months 3–6)**:
   - Experiment with Transformer for 10-day forecasts
   - Integrate physics-informed corrections (storm surge, coastal effects)
   - Monitor drift; retrain monthly

### **Expected Performance (5-Day Horizon)**:

- **Wave Height**: ±0.10–0.15 m (95% confidence)
- **Wind Speed**: ±0.5–0.7 m/s
- **Tidal Level**: ±0.08–0.12 m
- **Current Speed**: ±0.2–0.3 m/s
- **Water Temp**: ±0.5–0.8°C

**Risk**: Parameters beyond 5 days degrade rapidly; use ensemble & conservative confidence intervals for >7-day forecasts.

---

## Appendix A: Complete Parameter Quick Reference

| # | Parameter | Unit | Technique | Data (days) | Horizon | Accuracy (5d) |
|---|---|---|---|---|---|
| 1 | Wave Height | m | LSTM+VMD | 60–90 | 5–7d | 0.05–0.16m MAE |
| 2 | Wind Speed | m/s | Transformer | 60 | 5d | 0.55–0.70 RMSE |
| 3 | Water Temp | °C | CNN-LSTM | 90 | 5d | <0.75 RMSE |
| 4 | Wind Direction | deg | Circ-LSTM | 14–30 | 5d | 10–25° MAE |
| 5 | Current Speed | m/s | AR-ANN | 30–90 | 1–3d | 0.1–0.3 RMSE |
| 6 | Tidal Level | m | VMD-LSTM | 60–90 | 5–14d | 0.05–0.15 RMSE |
| 7 | Air Pressure | hPa | LSTM | 14–30 | 5–7d | 1–3 hPa RMSE |
| 8 | Conductivity | mS/cm | CNN-LSTM | 30–60 | 3–5d | 0.5–2 RMSE |
| 9 | Precipitation | mm/h | ConvLSTM | 30–90 | 1–3d | Limited (nowcast best) |
| 10 | Wave Period | s | LSTM | 30–60 | 3–5d | 0.27–0.49 MAE |
| 11 | Visibility | km | Logistic Reg | 30–60 | 1–3d | 0.65–0.85 categoric |
| 12 | Solar Radiation | W/m² | LSTM | 30 | 1–5d | 50–150 RMSE |
| 13 | Rel. Humidity | % | LSTM/XGB | 14–30 | 5–7d | 5–10% RMSE |
| 14 | Dew Point | °C/K | Linear Reg | 14–30 | 5–7d | 2–3 RMSE |

---

**Document Version**: 1.0  
**Last Updated**: June 21, 2026  
**Classification**: Technical Guide (Public)  
**Prepared For**: Shankari Analytics & Systems (Marine/Docking Operations)

