"""
🖥️ QFrame Research UI - Extends QFrame Streamlit UI

Integrates the research platform with the existing QFrame UI components,
providing research-specific dashboards while reusing existing UI code.
"""

from .research_dashboard import ResearchDashboard
from .experiment_tracker import ExperimentTracker
from .data_explorer import DataExplorer
from .strategy_comparison import StrategyComparison

__all__ = [
    "ResearchDashboard",
    "ExperimentTracker",
    "DataExplorer",
    "StrategyComparison",
]