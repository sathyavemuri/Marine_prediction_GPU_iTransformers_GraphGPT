"""Baseline forecasters for marine local MTGNN."""

from .persistence import PersistenceBaseline, SeasonalBaseline
from .seasonal import DailySeasonalBaseline, WeeklySeasonalBaseline
from .trend import TrendBaseline
from .selector import BaselineSelector

__all__ = [
    "PersistenceBaseline",
    "SeasonalBaseline",
    "DailySeasonalBaseline",
    "WeeklySeasonalBaseline",
    "TrendBaseline",
    "BaselineSelector",
]
