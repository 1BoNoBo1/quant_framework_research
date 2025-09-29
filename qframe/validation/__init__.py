"""
🛡️ QFrame Institutional Validation Module

Provides institutional-grade validation methods for quantitative strategies:
- Walk-forward analysis
- Overfitting detection suite
- Probabilistic Sharpe Ratio
- Multiple testing corrections
- Bootstrap confidence intervals
"""

from .institutional_validator import InstitutionalValidator
from .overfitting_detection import OverfittingDetector
from .walk_forward_analyzer import WalkForwardAnalyzer
from .probabilistic_metrics import ProbabilisticMetrics

__all__ = [
    'InstitutionalValidator',
    'OverfittingDetector',
    'WalkForwardAnalyzer',
    'ProbabilisticMetrics'
]