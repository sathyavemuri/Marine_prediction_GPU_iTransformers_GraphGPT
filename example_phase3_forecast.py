"""
Example: Phase 3 Hybrid Forecasting (Marine iTransformer + Local Models)

This script demonstrates:
1. Training local statistical models
2. Loading trained models
3. Running hybrid inference
4. Evaluating forecast accuracy
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'portland_itransformer' / 'src' / 'portland_itransformer'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_train_local_models():
    """Example: Train local statistical models."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 1: Train Local Statistical Models")
    logger.info("=" * 80)

    from local_models.train import main as train_local_models

    csv_path = PROJECT_ROOT / 'data' / 'raw' / 'portland_harbor_2025_15min_synthetic_calibrated.csv'
    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'local_models'

    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return False

    try:
        train_local_models(csv_path, artifacts_dir)
        logger.info("✓ Local models trained and saved")
        return True
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        return False


def example_load_and_forecast():
    """Example: Load models and generate forecast."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Load Models and Generate Forecast")
    logger.info("=" * 80)

    from local_models import HybridInference
    import torch

    # Initialize inference engine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create minimal config
    config = type('Config', (), {
        'data': type('Data', (), {
            'seq_len': 1344,
            'pred_len': 672,
            'cadence_minutes': 15,
        })()
    })()

    inference = HybridInference(config, device=device)

    # Load local models
    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'local_models'
    try:
        inference.load_statistical_models(artifacts_dir)
        logger.info("✓ Statistical models loaded")
    except Exception as e:
        logger.warning(f"Could not load statistical models: {e}")
        return False

    # Create synthetic recent data for demonstration
    logger.info("Creating synthetic recent data (14 days = 1344 steps)...")

    timestamps = pd.date_range('2025-06-01', periods=1344, freq='15min', tz='UTC')

    recent_data = {
        # Marine targets (would come from Marine iTransformer in practice)
        'tidal_residual_m': np.sin(2 * np.pi * np.arange(1344) / 96) * 0.5,
        'current_u_east_ms': np.sin(2 * np.pi * np.arange(1344) / 96) * 0.3,
        'current_v_north_ms': np.cos(2 * np.pi * np.arange(1344) / 96) * 0.2,
        'salinity_psu': 32.0 + np.random.normal(0, 0.1, 1344),
        'water_temp_c': 12.0 + 2 * np.sin(2 * np.pi * np.arange(1344) / 288),
        'log1p_global_radiation_wm2': np.log1p(200 * np.maximum(np.sin(2 * np.pi * (np.arange(1344) % 96) / 96 - np.pi/2), 0)),
        'log_significant_wave_height_m': np.log1p(0.5 + 0.1 * np.sin(2 * np.pi * np.arange(1344) / 288)),
        'log_zero_crossing_period_s': np.log1p(5.0 + 1.0 * np.sin(2 * np.pi * np.arange(1344) / 288)),

        # Atmospheric targets (what local models will predict)
        'air_temp_c': 15.0 + 3 * np.sin(2 * np.pi * np.arange(1344) / 96),
        'air_pressure_hpa': 1013.0 + np.random.normal(0, 1, 1344),
        'dew_point_c': 10.0 + 2 * np.sin(2 * np.pi * np.arange(1344) / 96),
        'wind_u_ms': 2.0 + np.sin(2 * np.pi * np.arange(1344) / 96),
        'wind_v_ms': 1.0 + np.cos(2 * np.pi * np.arange(1344) / 96),

        # Calendar features
        'hour_sin': np.sin(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0) / 24.0),
        'hour_cos': np.cos(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0) / 24.0),
        'dayofyear_sin': np.sin(2 * np.pi * timestamps.dayofyear / 365.25),
        'dayofyear_cos': np.cos(2 * np.pi * timestamps.dayofyear / 365.25),
    }

    # Generate forecast (7 days)
    logger.info("Generating 7-day forecast (672 steps)...")
    try:
        forecast = inference.forecast(
            recent_data=recent_data,
            recent_timestamps=timestamps,
            forecast_steps=672
        )

        logger.info(f"✓ Forecast generated for {len(forecast)} parameters")
        logger.info("\nForecast summary (first 5 steps, last 5 steps):")

        for param_name, values in sorted(forecast.items())[:5]:  # Show first 5 params
            if isinstance(values, np.ndarray) and len(values) > 0:
                logger.info(f"  {param_name:30s}: first={values[0]:7.2f}, "
                           f"mean={np.mean(values):7.2f}, last={values[-1]:7.2f}")

        return True

    except Exception as e:
        logger.error(f"✗ Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_evaluate():
    """Example: Evaluate forecast accuracy (demonstration)."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Evaluate Forecast Accuracy")
    logger.info("=" * 80)

    # In practice, you would:
    # 1. Get test set (unseen data)
    # 2. Generate forecasts for each test window
    # 3. Compute skill metrics: MAE, RMSE, R², etc.

    logger.info("""
Evaluation workflow:

1. Load test data (last 10% of dataset)
2. For each test window:
   - Get last 14 days of history
   - Generate 7-day forecast
   - Compare with actual 7-day data
3. Compute per-parameter metrics:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² (skill score)
4. Summary:
   - Best performing parameters
   - Confidence intervals (±1σ)
   - Comparison with baseline (climatology)

Example expected results for Phase 3:
- Marine targets: +40% to +100% skill (inherits from iTransformer)
- Atmospheric targets: 0-20% skill (damped persistence + climatology)
- Water temperature: 20-40% skill (harmonic baseline + smoothing)

Note: Exact numbers depend on:
- Data characteristics (synthetic vs real)
- Model tuning (decay time constants, trend/seasonal flags)
- Evaluation period (seasonal variation affects skill)
    """)

    return True


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3 HYBRID FORECASTING - EXAMPLES")
    logger.info("=" * 80)

    print("\nAvailable examples:")
    print("  1. Train local statistical models")
    print("  2. Load models and generate forecast")
    print("  3. Evaluate forecast accuracy (demonstration)")
    print("  Or run: python example_phase3_forecast.py all")

    # For this demonstration, run example 2 only (doesn't require prior training)
    logger.info("\nRunning Example 2: Load and Forecast")
    success = example_load_and_forecast()

    if success:
        logger.info("\n✓ Example completed successfully")
    else:
        logger.info("\n✗ Example failed - check logs above")

    # Also show example 3 (no actual computation)
    example_evaluate()


if __name__ == '__main__':
    main()
