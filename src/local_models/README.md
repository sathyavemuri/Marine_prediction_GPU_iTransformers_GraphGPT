# Local Statistical Models for Atmospheric Variables

Phase 3 hybrid forecasting: Marine iTransformer + local statistical models

## Quick Start

### 1. Train Local Models

```bash
cd /path/to/Marine_Prediction
python -m src.local_models.train
```

This trains 5 statistical models on the raw CSV data and saves them to `artifacts/local_models/`:
- `air_temp_model.joblib`
- `air_pressure_model.joblib`
- `dew_point_model.joblib`
- `wind_model.joblib`
- `water_temp_model.joblib`

### 2. Use for Inference

```python
from src.local_models import HybridInference
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
inference = HybridInference(config, device=device)

# Load trained models
inference.load_marine_model('outputs/marine_itransformer/best.pt')
inference.load_statistical_models('artifacts/local_models')
inference.load_scalers('artifacts')

# Forecast
forecast = inference.forecast(
    recent_data={...},  # 14 days (1344 steps) of history
    recent_timestamps=timestamps,
    forecast_steps=672  # 7 days
)

# forecast contains all 18 parameters with physical constraints enforced
```

## Module Architecture

### Core Models

#### `CalendarFeatures` (`calendar_features.py`)
Create cyclical hour/day-of-year features for harmonic regression:
```python
from local_models import CalendarFeatures
features = CalendarFeatures.create_features(timestamps)
# Returns: hour_fraction, day_of_year, sin_hour, cos_hour, sin_year, cos_year
```

#### `HarmonicBaseline` (`calendar_features.py`)
Fit OLS to daily/annual cycles:
```python
from local_models import HarmonicBaseline
baseline = HarmonicBaseline()
baseline.fit(timestamps, values, train_mask)
predictions = baseline.predict(timestamps)
```

#### `AirTemperatureModel` (`atmospheric_state_space.py`)
Harmonic baseline + UnobservedComponents:
```python
from local_models import AirTemperatureModel
model = AirTemperatureModel()
model.fit(timestamps, air_temp_values, train_mask)
forecast = model.predict(timestamps, steps=672)
```

#### `AirPressureModel` (`atmospheric_state_space.py`)
Damped persistence with 48-hour decay:
```python
from local_models import AirPressureModel
model = AirPressureModel(decay_time_hours=48.0, cadence_minutes=15.0)
model.fit(pressure_values, train_mask)
forecast = model.predict(latest_value=pressure_values[-1], steps=672)
```

#### `WindVectorModel` (`wind_vector_model.py`)
Damped persistence on u/v components + climatology:
```python
from local_models import WindVectorModel
model = WindVectorModel(decay_time_hours=24.0, cadence_minutes=15.0)
model.fit(u_values, v_values, train_mask)
u_forecast, v_forecast = model.predict(latest_u, latest_v, steps=672)
```

#### `DewPointModel` (`wind_vector_model.py`)
UnobservedComponents on log depression (air_temp - dew_point):
```python
from local_models import DewPointModel
model = DewPointModel()
model.fit(timestamps, air_temp, dew_point, train_mask)
forecast = model.predict(timestamps, air_temp_forecast, steps=672)
# Constraint: dew_point ≤ air_temp enforced automatically
```

#### `WaterTemperatureModel` (`water_temperature_model.py`)
Harmonic baseline + ExponentialSmoothing:
```python
from local_models import WaterTemperatureModel
model = WaterTemperatureModel()
model.fit(timestamps, water_temp_values, train_mask)
forecast = model.predict(timestamps, steps=672)
```

### Utilities

#### `WindDerivation` (`wind_vector_model.py`)
Convert between speed/direction and u/v components:
```python
from local_models import WindDerivation

# Speed & direction → u/v
u, v = WindDerivation.speed_direction_to_uv(speed, direction, convention='from')

# u/v → Speed & direction
speed, direction = WindDerivation.uv_to_speed_direction(u, v, convention='from')
```

#### `PhysicalReconstruction` (`reconstruction.py`)
Enforce physical constraints on predictions:
```python
from local_models import PhysicalReconstruction

reconstruction = PhysicalReconstruction()

# Constrain humidity to [0, 100]
rh = reconstruction.reconstruct_humidity(air_temp, dew_point)

# Ensure wind_speed ≥ 0, direction ∈ [0, 360)
speed, direction = reconstruction.reconstruct_wind(u, v)

# Clamp radiation to [0, 1200]
radiation = reconstruction.reconstruct_radiation(log_radiation)
```

### Orchestration

#### `HybridInference` (`inference.py`)
Combine Marine iTransformer + local statistical models:
```python
from local_models import HybridInference

inference = HybridInference(config, device='cuda')
inference.load_marine_model('marine_best.pt')
inference.load_statistical_models('artifacts/local_models')

# Returns dict with 18 parameters (marine + local + derived)
forecast_dict = inference.forecast(recent_data, timestamps, forecast_steps=672)
```

## Design Rationale

### Why Statistical Models for Atmosphere?
Local buoy observations contain NO information about distant weather systems:
- **Wind**: Cannot predict storm fronts or jet stream position from local wind
- **Pressure**: Local pressure anomalies decay within 48 hours
- **Temperature**: Local temperature anomalies also decay within hours

**Attempted alternatives**:
- Single unified neural network: -4.91% skill (conflicting gradients)
- Separate neural network for atmosphere (TimeXer): -76.25% skill (learning noise)
- **Statistical approach**: Honest boundaries, interpretable, proven methods

### Decay Time Constants (fitted on training data statistics)
- **Pressure**: τ = 48 hours (synoptic-scale variability)
- **Wind**: τ = 24 hours (mesoscale variability)
- **Temperature**: Harmonic baseline captures predictable cycles

### Why No External APIs?
- Project constraint: "local-only, no external data"
- GFS/ECMWF would require ~5GB/year storage and complex data pipeline
- Statistical methods transparent about limitations

## Performance Expectations

- **Marine targets** (tides, currents, waves, radiation, salinity): +64.7% skill
- **Atmospheric targets**: Baseline-like skill (honest about limited predictability)
  - Short horizon (0-2 days): Reasonable skill from persistence
  - Long horizon (5-7 days): Close to climatological mean

## Testing

```bash
python test_phase3_inference.py
```

All 9 modules pass comprehensive tests:
- Feature generation and temporal splitting
- Model fitting and forecasting
- Constraint enforcement (dew_point ≤ air_temp, etc.)
- Round-trip conversions (speed↔direction 0 error)

## Future Improvements

1. **Uncertainty quantification**: Return prediction intervals from model residuals
2. **Adaptive time constants**: Fit τ per season or synoptic regime
3. **Online learning**: Update models with streaming observations
4. **Multi-model ensemble**: Average across multiple initializations
5. **External data**: If GFS/ECMWF becomes available, add as exogenous features

## References

- **User Guide**: "Local_Only_No_API_Forecasting_Implementation_for_Claude_Code.txt"
- **statsmodels**: UnobservedComponents, ExponentialSmoothing
- **Marine iTransformer**: Time-Series-Library (ThUML)
- **Magnus equation**: Relative humidity from temperature + dew point
