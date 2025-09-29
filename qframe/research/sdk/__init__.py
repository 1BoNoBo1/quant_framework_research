"""
🛠️ QFrame Research SDK - Developer Tools

High-level SDK that makes QFrame research accessible and productive.
Wraps complex functionality in simple, intuitive APIs.
"""

from .research_api import QFrameResearch
from .strategy_builder import StrategyBuilder
from .experiment_manager import ExperimentManager
from .data_manager import DataManager

__all__ = [
    "QFrameResearch",
    "StrategyBuilder",
    "ExperimentManager",
    "DataManager",
]