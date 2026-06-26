"""Local statistical models for atmospheric variables (Phase 3)."""

from .calendar_features import CalendarFeatures, HarmonicBaseline
from .atmospheric_state_space import AirTemperatureModel, AirPressureModel
from .wind_vector_model import WindVectorModel, DewPointModel, WindDerivation
from .water_temperature_model import WaterTemperatureModel
from .reconstruction import PhysicalReconstruction
from .inference import HybridInference
from .graphcast_atmospheric import GraphCastAtmosphericModule, GraphCastWithFallback
from .aurora_atmospheric import AuroraAtmosphericModule, AuroraWithFallback

__all__ = [
    'CalendarFeatures',
    'HarmonicBaseline',
    'AirTemperatureModel',
    'AirPressureModel',
    'WindVectorModel',
    'DewPointModel',
    'WindDerivation',
    'WaterTemperatureModel',
    'PhysicalReconstruction',
    'HybridInference',
    'GraphCastAtmosphericModule',
    'GraphCastWithFallback',
    'AuroraAtmosphericModule',
    'AuroraWithFallback',
]
