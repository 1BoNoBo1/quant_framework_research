"""
Core module - Interfaces et abstractions fondamentales
"""

from .interfaces import (
    DataProvider,
    FeatureProcessor,
    Strategy,
    RiskManager,
    Portfolio
)
from .container import DIContainer
from .config import FrameworkConfig

__all__ = [
    "DataProvider",
    "FeatureProcessor",
    "Strategy",
    "RiskManager",
    "Portfolio",
    "DIContainer",
    "FrameworkConfig"
]