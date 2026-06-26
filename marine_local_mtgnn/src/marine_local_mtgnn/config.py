"""Configuration system with Pydantic validation."""

from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field, validator


class SiteConfig(BaseModel):
    """Site metadata required for formulas and operations."""

    buoy_id: str = Field(description="Unique buoy identifier")
    latitude: Optional[float] = Field(None, description="Required for UTide and pvlib")
    longitude: Optional[float] = Field(None, description="Required for pvlib")
    altitude_m: float = Field(default=0.0, description="Altitude in meters")
    timezone: str = Field(default="UTC", description="Site timezone (must be confirmed)")
    sea_pressure_dbar: float = Field(default=0.0, description="Sea pressure in dbar")
    wind_direction_convention: str = Field(
        default="from",
        description="'from' (meteorological) or 'to'",
    )
    current_direction_convention: str = Field(
        default="to",
        description="'from' or 'to'",
    )
    sensor_depth_m: Optional[float] = Field(None, description="Buoy sensor depth")
    wind_sensor_height_m: Optional[float] = Field(None, description="Wind sensor height")

    @validator("wind_direction_convention", "current_direction_convention")
    def validate_convention(cls, v):
        if v not in ["from", "to"]:
            raise ValueError("Convention must be 'from' or 'to'")
        return v


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""

    raw_csv: Path = Field(description="Path to raw CSV file")
    resample_rule: str = Field(default="15min", description="Resampling frequency")
    resample_label: str = Field(default="right", description="Resampling label")
    resample_closed: str = Field(default="right", description="Resampling closed side")


class SplitConfig(BaseModel):
    """Train/validation/test split configuration."""

    train_start: str = Field(description="Train start timestamp (ISO 8601 UTC)")
    train_end_exclusive: str = Field(description="Train end (exclusive)")
    validation_start: str = Field(description="Validation start timestamp")
    validation_end_exclusive: str = Field(description="Validation end (exclusive)")
    test_start: str = Field(description="Test start timestamp")
    test_end_exclusive: str = Field(description="Test end (exclusive)")


class ForecastConfig(BaseModel):
    """Forecast profile configuration."""

    profile: str = Field(default="long_7day", description="'long_7day' or 'short_24h'")
    lookback_steps: int = Field(default=672, description="Input history steps")
    horizon_steps: int = Field(default=672, description="Forecast horizon steps")
    sample_stride_steps: int = Field(default=4, description="Train sample stride")
    direct_multi_horizon: bool = Field(default=True, description="Direct vs recursive")


class BaselineConfig(BaseModel):
    """Baseline configuration."""

    persistence: bool = Field(default=True)
    daily_seasonal: bool = Field(default=True)
    weekly_seasonal: bool = Field(default=True)
    local_trend: bool = Field(default=True)
    trend_window_steps: int = Field(default=12, description="Local trend window")
    blended: bool = Field(default=True, description="Blended persistence/seasonal")
    tide_enabled: bool = Field(default=True)
    radiation_enabled: bool = Field(default=True)
    selection_metric: str = Field(
        default="mae_physical",
        description="Metric for baseline selection",
    )
    max_lag_graph_steps: int = Field(default=96, description="Max lag for graph prior")


class ModelConfig(BaseModel):
    """MTGNN model configuration."""

    num_input_nodes: int = Field(default=19)
    num_direct_targets: int = Field(default=15)
    hidden_channels: int = Field(default=32)
    skip_channels: int = Field(default=64)
    end_channels: int = Field(default=128)
    kernel_size: int = Field(default=3)
    dilation_exponential: int = Field(default=2)
    graph_top_k: int = Field(default=4)
    graph_conv_depth: int = Field(default=2)
    dropout: float = Field(default=0.15)
    max_trainable_parameters: int = Field(default=2000000)


class DecoderConfig(BaseModel):
    """Decoder configuration."""

    mode: str = Field(default="horizon_conditioned_direct")
    target_embedding_dim: int = Field(default=8)
    lead_time_embedding_dim: int = Field(default=16)
    decoder_hidden: int = Field(default=64)
    use_future_baseline: bool = Field(default=True)
    use_future_calendar: bool = Field(default=True)


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = Field(default=8)
    num_workers: int = Field(default=2)
    max_epochs: int = Field(default=150)
    early_stopping_patience: int = Field(default=20)
    learning_rate: float = Field(default=0.001)
    weight_decay: float = Field(default=0.0001)
    gradient_clip_norm: float = Field(default=1.0)
    monitor: str = Field(default="validation_weighted_physical_mae")


class PostprocessConfig(BaseModel):
    """Postprocessing configuration."""

    humidity_formula_enabled: bool = Field(default=True)
    conductivity_formula_enabled: bool = Field(default=False)
    conductivity_max_validation_mae_mscm: float = Field(default=3.0)


class ServiceConfig(BaseModel):
    """API service configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    minimum_history_steps: int = Field(default=672)
    prediction_horizon_steps: int = Field(default=672)


class Config(BaseModel):
    """Complete configuration."""

    experiment_name: str = Field(default="local_only_mtgnn_15min_7day")
    seed: int = Field(default=42)
    timezone: str = Field(default="UTC")
    output_root: Path = Field(default=Path("outputs"))

    site: SiteConfig
    data: DataConfig
    splits: SplitConfig
    forecast: ForecastConfig
    baselines: BaselineConfig
    model: ModelConfig
    decoder: DecoderConfig
    training: TrainingConfig
    postprocess: PostprocessConfig
    service: ServiceConfig

    class Config:
        arbitrary_types_allowed = True


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)


def save_config(config: Config, output_path: Path) -> None:
    """Save configuration to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
