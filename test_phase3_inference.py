"""Test Phase 3 Hybrid Inference pipeline."""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'portland_itransformer' / 'src' / 'portland_itransformer'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules to test they can be loaded
logger.info("\n" + "=" * 80)
logger.info("PHASE 3 INFERENCE PIPELINE - IMPORT TEST")
logger.info("=" * 80)

try:
    from local_models import (
        CalendarFeatures,
        HarmonicBaseline,
        AirTemperatureModel,
        AirPressureModel,
        WindVectorModel,
        DewPointModel,
        WindDerivation,
        WaterTemperatureModel,
        PhysicalReconstruction,
        HybridInference,
    )
    logger.info("✓ All local_models modules imported successfully")
except ImportError as e:
    logger.error(f"✗ Import failed: {e}")
    sys.exit(1)

# Test CalendarFeatures
logger.info("\n" + "-" * 80)
logger.info("Testing CalendarFeatures")
logger.info("-" * 80)

timestamps = pd.date_range('2025-06-01', periods=1344, freq='15min', tz='UTC')
try:
    cal_features = CalendarFeatures.create_features(timestamps)
    assert cal_features.shape == (1344, 6), f"Expected shape (1344, 6), got {cal_features.shape}"
    assert list(cal_features.columns) == ['hour_fraction', 'day_of_year', 'sin_hour', 'cos_hour', 'sin_year', 'cos_year']
    logger.info(f"✓ CalendarFeatures created: shape {cal_features.shape}")
except Exception as e:
    logger.error(f"✗ CalendarFeatures test failed: {e}")
    sys.exit(1)

# Test HarmonicBaseline
logger.info("\n" + "-" * 80)
logger.info("Testing HarmonicBaseline")
logger.info("-" * 80)

try:
    # Create synthetic temperature data
    np.random.seed(42)
    temp_data = 15.0 + 5.0 * np.sin(2 * np.pi * np.arange(1344) / 96)  # Daily cycle
    temp_data += np.random.normal(0, 0.5, 1344)  # Noise

    train_mask = np.arange(1344) < 940  # 70% for training

    harmonic = HarmonicBaseline()
    harmonic.fit(timestamps, temp_data, train_mask)

    baseline_pred = harmonic.predict(timestamps)
    assert baseline_pred.shape == (1344,), f"Expected shape (1344,), got {baseline_pred.shape}"
    logger.info(f"✓ HarmonicBaseline fitted and predicted: shape {baseline_pred.shape}, coeff {harmonic.coefficients}")
except Exception as e:
    logger.error(f"✗ HarmonicBaseline test failed: {e}")
    sys.exit(1)

# Test AirTemperatureModel
logger.info("\n" + "-" * 80)
logger.info("Testing AirTemperatureModel")
logger.info("-" * 80)

try:
    air_temp_model = AirTemperatureModel()
    air_temp_model.fit(timestamps, temp_data, train_mask)

    # Predict 10 steps ahead
    forecast = air_temp_model.predict(timestamps, steps=10)
    assert forecast.shape == (10,), f"Expected shape (10,), got {forecast.shape}"
    logger.info(f"✓ AirTemperatureModel fitted and predicted 10 steps: {forecast}")
except Exception as e:
    logger.error(f"✗ AirTemperatureModel test failed: {e}")
    sys.exit(1)

# Test AirPressureModel
logger.info("\n" + "-" * 80)
logger.info("Testing AirPressureModel")
logger.info("-" * 80)

try:
    pressure_data = 1013.0 + 5.0 * np.sin(2 * np.pi * np.arange(1344) / 288)  # 3-day cycle
    pressure_data += np.random.normal(0, 0.5, 1344)

    pressure_model = AirPressureModel(decay_time_hours=48.0, cadence_minutes=15.0)
    pressure_model.fit(pressure_data, train_mask)

    forecast = pressure_model.predict(latest_value=pressure_data[-1], steps=10)
    assert forecast.shape == (10,), f"Expected shape (10,), got {forecast.shape}"
    logger.info(f"✓ AirPressureModel fitted and predicted 10 steps: mean={np.mean(forecast):.2f}")
except Exception as e:
    logger.error(f"✗ AirPressureModel test failed: {e}")
    sys.exit(1)

# Test WindVectorModel
logger.info("\n" + "-" * 80)
logger.info("Testing WindVectorModel")
logger.info("-" * 80)

try:
    wind_speed = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(1344) / 96)
    wind_speed += np.random.normal(0, 0.5, 1344)
    wind_speed = np.maximum(wind_speed, 0)

    wind_dir = 180.0 + 45.0 * np.sin(2 * np.pi * np.arange(1344) / 288)
    wind_dir = np.mod(wind_dir, 360.0)

    # Convert to u/v
    from features import speed_dir_to_uv
    u_wind, v_wind = speed_dir_to_uv(wind_speed, wind_dir, convention='from')

    wind_model = WindVectorModel(decay_time_hours=24.0, cadence_minutes=15.0)
    wind_model.fit(u_wind, v_wind, train_mask)

    u_forecast, v_forecast = wind_model.predict(u_wind[-1], v_wind[-1], steps=10)
    assert u_forecast.shape == (10,), f"Expected shape (10,), got {u_forecast.shape}"
    assert v_forecast.shape == (10,), f"Expected shape (10,), got {v_forecast.shape}"
    logger.info(f"✓ WindVectorModel fitted and predicted 10 steps: u mean={np.mean(u_forecast):.2f}, v mean={np.mean(v_forecast):.2f}")
except Exception as e:
    logger.error(f"✗ WindVectorModel test failed: {e}")
    sys.exit(1)

# Test WindDerivation
logger.info("\n" + "-" * 80)
logger.info("Testing WindDerivation")
logger.info("-" * 80)

try:
    # Round trip: speed/dir -> u/v -> speed/dir
    original_speed = np.array([3.5, 5.2, 7.1])
    original_dir = np.array([45.0, 180.0, 270.0])

    u, v = WindDerivation.speed_direction_to_uv(original_speed, original_dir, convention='from')
    recovered_speed, recovered_dir = WindDerivation.uv_to_speed_direction(u, v, convention='from')

    speed_error = np.max(np.abs(original_speed - recovered_speed))
    # Find minimum circular distance for direction error
    dir_diff1 = np.abs(original_dir - recovered_dir)
    dir_diff2 = np.abs(original_dir - recovered_dir + 360.0)
    dir_diff3 = np.abs(original_dir - recovered_dir - 360.0)
    dir_error = np.minimum(np.minimum(dir_diff1, dir_diff2), dir_diff3)

    assert speed_error < 1e-10, f"Speed round-trip error too large: {speed_error}"
    assert np.all(dir_error < 1e-10), f"Direction round-trip error too large: {dir_error}"
    logger.info(f"✓ WindDerivation round-trip accurate (speed error={speed_error:.2e}, dir error={np.max(dir_error):.2e})")
except Exception as e:
    logger.error(f"✗ WindDerivation test failed: {e}")
    sys.exit(1)

# Test DewPointModel
logger.info("\n" + "-" * 80)
logger.info("Testing DewPointModel")
logger.info("-" * 80)

try:
    dew_point_data = temp_data - 3.0 - np.random.normal(0, 0.5, 1344)
    dew_point_data = np.minimum(dew_point_data, temp_data - 0.1)  # Ensure dew_point <= air_temp

    dew_model = DewPointModel()
    dew_model.fit(timestamps, temp_data, dew_point_data, train_mask)

    # For dew point prediction, we need an air temp forecast
    air_temp_forecast = np.array([16.0 + np.random.normal(0, 1) for _ in range(10)])
    dew_forecast = dew_model.predict(timestamps, air_temp_forecast, steps=10)

    assert dew_forecast.shape == (10,), f"Expected shape (10,), got {dew_forecast.shape}"
    assert np.all(dew_forecast <= air_temp_forecast + 0.1), "Dew point should be <= air temp"
    logger.info(f"✓ DewPointModel fitted and predicted 10 steps with constraint enforcement")
except Exception as e:
    logger.error(f"✗ DewPointModel test failed: {e}")
    sys.exit(1)

# Test WaterTemperatureModel
logger.info("\n" + "-" * 80)
logger.info("Testing WaterTemperatureModel")
logger.info("-" * 80)

try:
    water_temp_data = 12.0 + 2.0 * np.sin(2 * np.pi * np.arange(1344) / 288)  # 3-day cycle
    water_temp_data += np.random.normal(0, 0.3, 1344)

    wt_model = WaterTemperatureModel()
    wt_model.fit(timestamps, water_temp_data, train_mask)

    forecast = wt_model.predict(timestamps, steps=10)
    assert forecast.shape == (10,), f"Expected shape (10,), got {forecast.shape}"
    logger.info(f"✓ WaterTemperatureModel fitted and predicted 10 steps: mean={np.mean(forecast):.2f}")
except Exception as e:
    logger.error(f"✗ WaterTemperatureModel test failed: {e}")
    sys.exit(1)

# Test PhysicalReconstruction
logger.info("\n" + "-" * 80)
logger.info("Testing PhysicalReconstruction")
logger.info("-" * 80)

try:
    reconstruction = PhysicalReconstruction()

    # Test humidity computation
    air_temps = np.array([20.0, 25.0, 15.0])
    dew_points = np.array([10.0, 15.0, 5.0])
    rh = reconstruction.reconstruct_humidity(air_temps, dew_points)

    assert rh.shape == (3,), f"Expected shape (3,), got {rh.shape}"
    assert np.all(rh >= 0) and np.all(rh <= 100), "RH should be in [0, 100]"
    logger.info(f"✓ Humidity reconstruction: {rh}")

    # Test wind reconstruction
    u_vals = np.array([2.0, -3.0, 0.5])
    v_vals = np.array([1.0, 4.0, -2.0])
    speed, direction = reconstruction.reconstruct_wind(u_vals, v_vals)

    assert speed.shape == (3,), f"Expected shape (3,), got {speed.shape}"
    assert direction.shape == (3,), f"Expected shape (3,), got {direction.shape}"
    assert np.all(speed >= 0), "Speed should be >= 0"
    assert np.all(direction >= 0) and np.all(direction < 360), "Direction should be in [0, 360)"
    logger.info(f"✓ Wind reconstruction: speeds={speed}, directions={direction}")

    # Test radiation
    log_rad = np.array([2.0, 3.5, 4.0])
    radiation = reconstruction.reconstruct_radiation(log_rad)
    assert radiation.shape == (3,), f"Expected shape (3,), got {radiation.shape}"
    assert np.all(radiation >= 0), "Radiation should be >= 0"
    logger.info(f"✓ Radiation reconstruction: {radiation}")

except Exception as e:
    logger.error(f"✗ PhysicalReconstruction test failed: {e}")
    sys.exit(1)

# Final summary
logger.info("\n" + "=" * 80)
logger.info("PHASE 3 INFERENCE PIPELINE - ALL TESTS PASSED ✓")
logger.info("=" * 80)
logger.info("""
Ready for Phase 3 implementation:
- CalendarFeatures: ✓
- HarmonicBaseline: ✓
- AirTemperatureModel (UnobservedComponents): ✓
- AirPressureModel (damped persistence): ✓
- WindVectorModel (damped persistence on u/v): ✓
- DewPointModel (UnobservedComponents on depression): ✓
- WaterTemperatureModel (ExponentialSmoothing): ✓
- PhysicalReconstruction (constraint enforcement): ✓
- HybridInference (Marine iTransformer + local models): Ready to load

Next: Run train_local_models.py to fit models on training data.
""")
