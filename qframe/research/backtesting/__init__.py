"""
🚀 QFrame Research Backtesting v2 - Distributed & Enhanced

Extends the existing QFrame BacktestingService with:
- Distributed computing (Dask/Ray)
- Walk-forward analysis
- Monte Carlo simulations
- Advanced performance analytics
"""

from .distributed_engine import DistributedBacktestEngine
from .walk_forward_analyzer import WalkForwardAnalyzer
from .monte_carlo_simulator import MonteCarloSimulator
from .performance_analyzer import AdvancedPerformanceAnalyzer

__all__ = [
    "DistributedBacktestEngine",
    "WalkForwardAnalyzer",
    "MonteCarloSimulator",
    "AdvancedPerformanceAnalyzer",
]