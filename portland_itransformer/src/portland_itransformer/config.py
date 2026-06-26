"""Configuration classes using Pydantic."""

from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import yaml


class SiteConfig(BaseModel):
    """Site metadata."""
    name: str
    latitude: float
    longitude: float
    timezone: str = "UTC"
    wind_direction_convention: str = "from"  # "from" or "to"
    current_direction_convention: str = "to"  # "from" or "to"
    sensor_pressure_dbar: float = 0.0


class DataConfig(BaseModel):
    """Data configuration."""
    cadence_minutes: int = 15
    seq_len: int = 1344  # 14 days
    pred_len: int = 672  # 7 days
    baseline_fit_days: int = 60
    train_target_start: str
    train_target_end: str
    val_target_start: str
    val_target_end: str
    test_target_start: str
    test_target_end: str
    train_stride_steps: int = 4  # 1 hour
    eval_stride_steps: int = 96  # 1 day
    max_short_gap_minutes: int = 60


class PathsConfig(BaseModel):
    """Path configuration."""
    raw_csv: str
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    outputs_dir: str = "outputs"


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.20
    use_instance_norm: bool = True


class TrainingConfig(BaseModel):
    """Training configuration."""
    batch_size: int = 16
    num_workers: int = 2
    epochs: int = 40
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    use_amp_if_cuda: bool = True


class ReconstructionConfig(BaseModel):
    """Reconstruction configuration."""
    eps: float = 0.0001
    solar_daylight_threshold_wm2: float = 20.0
    clearness_index_max: float = 2.0
    rh_clip_min: float = 0.0
    rh_clip_max: float = 100.0


class Config(BaseModel):
    """Master configuration."""
    project_name: str
    seed: int = 2025
    paths: PathsConfig
    site: SiteConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    reconstruction: ReconstructionConfig

    class Config:
        """Pydantic config."""
        extra = "allow"


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config(**data)
