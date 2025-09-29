"""
QFrame Research & Innovation Platform
====================================

Advanced research platform for continuous strategy development and innovation.
"""

# Import only modules that exist
try:
    from .backtesting import (
        DistributedBacktestEngine,
        WalkForwardAnalyzer,
        MonteCarloSimulator,
        AdvancedPerformanceAnalyzer
    )
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

try:
    from .data_lake import (
        DataLakeStorage,
        FeatureStore,
        DataCatalog
    )
    DATA_LAKE_AVAILABLE = True
except ImportError:
    DATA_LAKE_AVAILABLE = False

try:
    from .integration_layer import create_research_integration
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

__all__ = []

if BACKTESTING_AVAILABLE:
    __all__.extend([
        "DistributedBacktestEngine",
        "WalkForwardAnalyzer",
        "MonteCarloSimulator",
        "AdvancedPerformanceAnalyzer"
    ])

if DATA_LAKE_AVAILABLE:
    __all__.extend([
        "DataLakeStorage",
        "FeatureStore",
        "DataCatalog"
    ])

if INTEGRATION_AVAILABLE:
    __all__.append("create_research_integration")