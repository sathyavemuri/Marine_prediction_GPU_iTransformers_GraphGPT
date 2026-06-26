# Phase 3: Hybrid Inference Architecture

## Overview
Phase 3 implements a hybrid forecasting system combining:
1. **Marine iTransformer** (+64.7% skill) for deterministic variables (tides, currents, waves, radiation, salinity)
2. **Local statistical models** for chaotic atmospheric variables (wind, pressure, temperature, humidity)

This pragmatic approach separates variables with conflicting training objectives and uses statistically sound methods for each domain.

## Key Components

### 1. Statistical Models (`src/local_models/`)

#### CalendarFeatures (`calendar_features.py`)
- Create cyclical hour and day-of-year features
- Chronological train/val/test split
- HarmonicBaseline: OLS fitting of daily/annual cycles

#### AirTemperatureModel (`atmospheric_state_space.py`)
- **Method**: Harmonic baseline + UnobservedComponents (state-space)
- **Targets**: air_temp_c
- **Logic**: Decompose into predictable cycle + short-term anomaly

#### AirPressureModel (`atmospheric_state_space.py`)
- **Method**: Damped persistence (exponential decay to climatological mean)
- **Formula**: P(h) = P_LTM + exp(-h/τ) * (P₀ - P_LTM)
- **τ**: 48-hour decay time constant
- **Rationale**: Pressure changes are local and short-lived; long-range forecasting requires external data

#### WindVectorModel (`wind_vector_model.py`)
- **Method**: Damped persistence on u/v components + climatology
- **Targets**: wind_u_ms, wind_v_ms → wind_speed_ms, wind_direction_deg
- **τ**: 24-hour decay time constant
- **Derivation**: WindDerivation class for speed↔direction conversions

#### DewPointModel (`wind_vector_model.py`)
- **Method**: Harmonic baseline + UnobservedComponents on dew point depression
- **Formula**: depression = air_temp - dew_point (log-transformed for stability)
- **Constraint**: dew_point ≤ air_temp (enforced during reconstruction)

#### WaterTemperatureModel (`water_temperature_model.py`)
- **Method**: Harmonic baseline + ExponentialSmoothing (Holt-Winters)
- **Targets**: water_temp_c
- **Trend**: Additive
- **Seasonal**: None (handled by harmonic baseline)

### 2. Inference Pipeline (`inference.py`)

**HybridInference** orchestrates:
1. Load Marine iTransformer checkpoint and local statistical models
2. Run Marine iTransformer on 8 deterministic targets
3. Run local models on 7 atmospheric/water targets
4. Derive outputs: humidity (Magnus formula), wind/current direction, radiation, wave periods

### 3. Reconstruction (`reconstruction.py`)

**PhysicalReconstruction** enforces constraints:
- dew_point ≤ air_temp
- RH ∈ [0, 100]
- wind_speed ≥ 0
- direction ∈ [0, 360)
- radiation ∈ [0, 1200] W/m²
- wave_height ∈ [0, 15] m
- salinity ∈ [0, 40] PSU

## Architecture Comparison

### Why Not Single Neural Network?
- **Conflicting gradients**: Marine variables respond to lunar cycles (tidal_residual responds to M2 tide), while atmospheric variables driven by distant weather systems
- **Training instability**: Unified model achieved -4.91% skill vs +64.7% with separation
- **Atmospheric TimeXer attempt**: -76.25% skill, proved local-only neural nets cannot predict chaotic variables

### Why Statistical Models for Atmosphere?
- **Honest**: Makes no false claims about predicting unobserved weather systems
- **Interpretable**: Harmonic baselines capture daily/annual cycles, state-space models capture short-term anomalies
- **Fast**: No GPU needed, inference < 10ms for 7-day forecast
- **Proven**: UnobservedComponents, ExponentialSmoothing are decades-old standard methods

### Why Damped Persistence for Pressure/Wind?
- **No external forcing**: Local buoy observations cannot distinguish between multiple weather regimes
- **Short skill horizon**: Pressure anomalies decay within 48 hours; wind shifts decay within 24 hours
- **Climatology fallback**: When anomaly fully decays, revert to climatological mean

## Implementation Files

```
src/local_models/
├── __init__.py                       # Module exports
├── calendar_features.py              # CalendarFeatures, HarmonicBaseline
├── atmospheric_state_space.py        # AirTemperatureModel, AirPressureModel
├── wind_vector_model.py              # WindVectorModel, DewPointModel, WindDerivation
├── water_temperature_model.py        # WaterTemperatureModel
├── reconstruction.py                 # PhysicalReconstruction
├── inference.py                      # HybridInference
└── train.py                          # Training entry point

Test files:
└── test_phase3_inference.py          # Integration tests (all passing ✓)
```

## Training Local Models

```bash
python -m src.local_models.train \
    --csv data/raw/portland_harbor_2025_15min_synthetic_calibrated.csv \
    --artifacts artifacts/local_models
```

Outputs:
- `artifacts/local_models/air_temp_model.joblib`
- `artifacts/local_models/air_pressure_model.joblib`
- `artifacts/local_models/dew_point_model.joblib`
- `artifacts/local_models/wind_model.joblib`
- `artifacts/local_models/water_temp_model.joblib`

## Inference Usage

```python
from src.local_models import HybridInference
from omegaconf import DictConfig

# Load configuration
config = DictConfig({...})
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize
inference = HybridInference(config, device=device)
inference.load_marine_model(Path('outputs/marine_itransformer/best.pt'))
inference.load_statistical_models(Path('artifacts/local_models'))
inference.load_scalers(Path('artifacts'))

# Forecast
forecast = inference.forecast(
    recent_data={'tidal_residual_m': [...], ...},
    recent_timestamps=timestamps,
    forecast_steps=672  # 7 days
)

# forecast now contains all 18 parameters:
# - Marine iTransformer: tidal_residual_m, current_u_east_ms, current_v_north_ms,
#                        salinity_psu, water_temp_c, log1p_global_radiation_wm2,
#                        log_significant_wave_height_m, log_zero_crossing_period_s
# - Local models: air_temp_c, air_pressure_hpa, dew_point_c,
#                 wind_u_ms, wind_v_ms, water_temp_c_statistical
# - Derived: relative_humidity_pct, wind_speed_ms, wind_direction_deg,
#            current_speed_ms, current_direction_deg, global_radiation_wm2,
#            significant_wave_height_m, zero_crossing_period_s
```

## Test Coverage

All modules pass comprehensive tests:
- ✓ CalendarFeatures: cyclical encoding, chronological split
- ✓ HarmonicBaseline: OLS fitting, prediction
- ✓ AirTemperatureModel: UCM fitting/forecasting with fallback
- ✓ AirPressureModel: exponential decay fitting
- ✓ WindVectorModel: climatology + damped persistence
- ✓ DewPointModel: depression-based UCM with constraints
- ✓ WaterTemperatureModel: exponential smoothing on anomaly
- ✓ WindDerivation: round-trip speed↔direction conversion (0 error)
- ✓ PhysicalReconstruction: constraint enforcement on all outputs

## Future Improvements

1. **Uncertainty quantification**: Return prediction intervals (±2σ from model residuals)
2. **Adaptive decay time constants**: Fit τ empirically per season
3. **Multi-model ensemble**: Average skill across multiple random initializations
4. **Online learning**: Update models incrementally as new observations arrive
5. **External forcing**: If GFS/ECMWF data becomes available, add as exogenous features

## References

- Guide: "Local_Only_No_API_Forecasting_Implementation_for_Claude_Code.txt"
- Marine iTransformer: ThUML Time-Series-Library
- UnobservedComponents: statsmodels.tsa.statespace.structural
- ExponentialSmoothing: statsmodels.tsa.holtwinters
