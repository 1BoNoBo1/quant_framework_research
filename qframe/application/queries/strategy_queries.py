"""
Strategy Queries
================

Query objects for strategy-related read operations in CQRS architecture.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from uuid import UUID


@dataclass
class GetStrategyByIdQuery:
    """Query to get a strategy by its ID"""
    strategy_id: str


@dataclass
class GetActiveStrategiesQuery:
    """Query to get all active strategies"""
    pass


@dataclass
class GetStrategiesByTypeQuery:
    """Query to get strategies by type"""
    strategy_type: str


@dataclass
class GetStrategyPerformanceQuery:
    """Query to get strategy performance metrics"""
    strategy_id: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class SearchStrategiesQuery:
    """Query to search strategies by various criteria"""
    name_pattern: Optional[str] = None
    status: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: Optional[int] = 100
    offset: Optional[int] = 0