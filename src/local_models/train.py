"""Train local statistical models for atmospheric variables."""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import joblib
import sys
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate raw CSV data."""
    logger.info(f"Loading raw data from {csv_path}")

    df = pd.read_csv(csv_path)
    if df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"✓ Loaded {len(df)} rows")
    return df


def create_chronological_split(n_samples: int, train_frac: float = 0.7) -> np.ndarray:
    """Create chronological train/val/test split."""
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + 0.15))

    split = np.zeros(n_samples, dtype=int)
    split[train_end:val_end] = 1
    split[val_end:] = 2

    logger.info(f"Split: {train_end} train, {val_end - train_end} val, {n_samples - val_end} test")
    return split


def derive_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wind speed/direction to u/v components."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'portland_itransformer' / 'src' / 'portland_itransformer'))
    from features import speed_dir_to_uv

    df['wind_u_ms'], df['wind_v_ms'] = speed_dir_to_uv(
        df['wind_speed_ms'].values,
        df['wind_direction_deg'].values,
        convention='from'
    )

    return df


def train_air_temperature_model(df: pd.DataFrame, split: np.ndarray, artifacts_dir: Path):
    """Train air temperature model."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: Air Temperature Model")
    logger.info("=" * 80)

    sys.path.insert(0, str(Path(__file__).parent))
    from atmospheric_state_space import AirTemperatureModel

    train_mask = split == 0
    timestamps = pd.DatetimeIndex(df['timestamp'])
    values = df['air_temp_c'].values

    model = AirTemperatureModel()
    model.fit(timestamps, values, train_mask)

    joblib.dump(model, artifacts_dir / 'air_temp_model.joblib')
    logger.info("✓ Air temperature model saved")


def train_air_pressure_model(df: pd.DataFrame, split: np.ndarray, artifacts_dir: Path):
    """Train air pressure model."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: Air Pressure Model")
    logger.info("=" * 80)

    sys.path.insert(0, str(Path(__file__).parent))
    from atmospheric_state_space import AirPressureModel

    train_mask = split == 0
    values = df['air_pressure_hpa'].values

    model = AirPressureModel(decay_time_hours=48.0, cadence_minutes=15.0)
    model.fit(values, train_mask)

    joblib.dump(model, artifacts_dir / 'air_pressure_model.joblib')
    logger.info("✓ Air pressure model saved")


def train_dew_point_model(df: pd.DataFrame, split: np.ndarray, artifacts_dir: Path):
    """Train dew point model."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: Dew Point Model")
    logger.info("=" * 80)

    sys.path.insert(0, str(Path(__file__).parent))
    from wind_vector_model import DewPointModel

    train_mask = split == 0
    timestamps = pd.DatetimeIndex(df['timestamp'])
    air_temp = df['air_temp_c'].values
    dew_point = df['dew_point_c'].values

    model = DewPointModel()
    model.fit(timestamps, air_temp, dew_point, train_mask)

    joblib.dump(model, artifacts_dir / 'dew_point_model.joblib')
    logger.info("✓ Dew point model saved")


def train_wind_model(df: pd.DataFrame, split: np.ndarray, artifacts_dir: Path):
    """Train wind vector model."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: Wind Vector Model")
    logger.info("=" * 80)

    sys.path.insert(0, str(Path(__file__).parent))
    from wind_vector_model import WindVectorModel

    train_mask = split == 0
    u_values = df['wind_u_ms'].values
    v_values = df['wind_v_ms'].values

    model = WindVectorModel(decay_time_hours=24.0, cadence_minutes=15.0)
    model.fit(u_values, v_values, train_mask)

    joblib.dump(model, artifacts_dir / 'wind_model.joblib')
    logger.info("✓ Wind vector model saved")


def train_water_temperature_model(df: pd.DataFrame, split: np.ndarray, artifacts_dir: Path):
    """Train water temperature model."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: Water Temperature Model")
    logger.info("=" * 80)

    sys.path.insert(0, str(Path(__file__).parent))
    from water_temperature_model import WaterTemperatureModel

    train_mask = split == 0
    timestamps = pd.DatetimeIndex(df['timestamp'])
    values = df['water_temp_c'].values

    model = WaterTemperatureModel()
    model.fit(timestamps, values, train_mask)

    joblib.dump(model, artifacts_dir / 'water_temp_model.joblib')
    logger.info("✓ Water temperature model saved")


def main(csv_path: Path, artifacts_dir: Path):
    """Train all local statistical models."""
    logger.info("\n" + "=" * 80)
    logger.info("LOCAL STATISTICAL MODELS TRAINING PIPELINE")
    logger.info("=" * 80)

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_raw_data(csv_path)

    # Create split
    split = create_chronological_split(len(df), train_frac=0.7)

    # Derive features
    df = derive_wind_components(df)

    # Train each model
    train_air_temperature_model(df, split, artifacts_dir)
    train_air_pressure_model(df, split, artifacts_dir)
    train_dew_point_model(df, split, artifacts_dir)
    train_wind_model(df, split, artifacts_dir)
    train_water_temperature_model(df, split, artifacts_dir)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"✓ All models saved to {artifacts_dir}")


if __name__ == '__main__':
    # Try multiple potential locations for the CSV
    possible_paths = [
        Path('data/raw/portland_harbor_2025_15min_synthetic_calibrated.csv'),
        Path('portland_itransformer/data/raw/portland_harbor_2025_15min_synthetic_calibrated.csv'),
        Path(__file__).parent.parent.parent / 'portland_itransformer' / 'data' / 'raw' / 'portland_harbor_2025_15min_synthetic_calibrated.csv',
    ]

    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        logger.error(f"CSV file not found in any of: {possible_paths}")
        sys.exit(1)

    artifacts_dir = Path('artifacts/local_models')

    main(csv_path, artifacts_dir)
