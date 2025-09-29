"""
Core module - Interfaces et abstractions fondamentales
"""

from .config import FrameworkConfig
from .container import DIContainer
from .interfaces import DataProvider, FeatureProcessor, Portfolio, RiskManager, Strategy

__all__ = [
    "DataProvider",
    "FeatureProcessor",
    "Strategy",
    "RiskManager",
    "Portfolio",
    "DIContainer",
    "FrameworkConfig",
]
