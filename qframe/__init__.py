"""
QFrame - Framework Quantitatif Professionnel
===========================================

Framework de trading quantitatif moderne avec architecture hexagonale,
combinant recherche avancée et stratégies de production.

Features:
- Architecture clean avec dependency injection
- Stratégies multi-horizons (research + production)
- Pipeline de données robuste multi-exchanges
- Validation institutionnelle rigoureuse
- Tests complets et CI/CD
"""

__version__ = "0.1.0"
__author__ = "Quantitative Research Team"

# Imports principaux pour API publique
from .core.interfaces import (
    DataProvider,
    FeatureProcessor,
    Strategy,
    RiskManager,
    Portfolio
)

from .core.container import DIContainer
from .core.config import FrameworkConfig

__all__ = [
    "DataProvider",
    "FeatureProcessor",
    "Strategy",
    "RiskManager",
    "Portfolio",
    "DIContainer",
    "FrameworkConfig"
]