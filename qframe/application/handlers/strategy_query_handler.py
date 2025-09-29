"""
Strategy Query Handler
======================

Handler for strategy-related read operations in CQRS architecture.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from qframe.core.container import injectable
from qframe.domain.entities.strategy import Strategy
from qframe.domain.repositories.strategy_repository import StrategyRepository
from qframe.application.queries.strategy_queries import (
    GetStrategyByIdQuery,
    GetActiveStrategiesQuery,
    GetStrategiesByTypeQuery,
    GetStrategyPerformanceQuery,
    SearchStrategiesQuery
)

logger = logging.getLogger(__name__)


@injectable
class StrategyQueryHandler:
    """
    Handler for strategy queries.

    Implements read operations for the CQRS pattern.
    """

    def __init__(self, strategy_repository: StrategyRepository):
        self.strategy_repository = strategy_repository

    async def handle_get_by_id(self, query: GetStrategyByIdQuery) -> Optional[Strategy]:
        """Handle GetStrategyByIdQuery"""
        try:
            logger.debug(f"Handling GetStrategyByIdQuery for strategy_id: {query.strategy_id}")
            return await self.strategy_repository.find_by_id(query.strategy_id)
        except Exception as e:
            logger.error(f"Error handling GetStrategyByIdQuery: {e}")
            raise

    async def handle_get_active_strategies(self, query: GetActiveStrategiesQuery) -> List[Strategy]:
        """Handle GetActiveStrategiesQuery"""
        try:
            logger.debug("Handling GetActiveStrategiesQuery")
            return await self.strategy_repository.find_active_strategies()
        except Exception as e:
            logger.error(f"Error handling GetActiveStrategiesQuery: {e}")
            raise

    async def handle_get_by_type(self, query: GetStrategiesByTypeQuery) -> List[Strategy]:
        """Handle GetStrategiesByTypeQuery"""
        try:
            logger.debug(f"Handling GetStrategiesByTypeQuery for type: {query.strategy_type}")
            return await self.strategy_repository.find_by_type(query.strategy_type)
        except Exception as e:
            logger.error(f"Error handling GetStrategiesByTypeQuery: {e}")
            raise

    async def handle_get_performance(self, query: GetStrategyPerformanceQuery) -> Dict[str, Any]:
        """Handle GetStrategyPerformanceQuery"""
        try:
            logger.debug(f"Handling GetStrategyPerformanceQuery for strategy_id: {query.strategy_id}")

            # Get strategy
            strategy = await self.strategy_repository.find_by_id(query.strategy_id)
            if not strategy:
                return {}

            # Return performance metrics
            performance = {
                "strategy_id": query.strategy_id,
                "strategy_name": strategy.name,
                "total_trades": strategy.total_trades,
                "winning_trades": strategy.winning_trades,
                "total_pnl": float(strategy.total_pnl),
                "max_drawdown": float(strategy.max_drawdown),
                "win_rate": float(strategy.get_win_rate()),
                "sharpe_ratio": float(strategy.sharpe_ratio) if strategy.sharpe_ratio else None,
                "performance_metrics": strategy.performance_metrics,
                "last_updated": strategy.updated_at.isoformat() if strategy.updated_at else None
            }

            return performance

        except Exception as e:
            logger.error(f"Error handling GetStrategyPerformanceQuery: {e}")
            raise

    async def handle_search_strategies(self, query: SearchStrategiesQuery) -> List[Strategy]:
        """Handle SearchStrategiesQuery"""
        try:
            logger.debug(f"Handling SearchStrategiesQuery with criteria: {query}")

            # For now, return all strategies and filter in memory
            # In production, this would be optimized with database queries
            all_strategies = await self.strategy_repository.find_all()

            filtered_strategies = all_strategies

            # Apply filters
            if query.name_pattern:
                filtered_strategies = [
                    s for s in filtered_strategies
                    if query.name_pattern.lower() in s.name.lower()
                ]

            if query.status:
                filtered_strategies = [
                    s for s in filtered_strategies
                    if s.status.value == query.status
                ]

            if query.author:
                filtered_strategies = [
                    s for s in filtered_strategies
                    if s.author == query.author
                ]

            # Apply pagination
            if query.offset:
                filtered_strategies = filtered_strategies[query.offset:]

            if query.limit:
                filtered_strategies = filtered_strategies[:query.limit]

            return filtered_strategies

        except Exception as e:
            logger.error(f"Error handling SearchStrategiesQuery: {e}")
            raise


# Convenience functions for direct usage
async def get_strategy_by_id(handler: StrategyQueryHandler, strategy_id: str) -> Optional[Strategy]:
    """Convenience function to get strategy by ID"""
    query = GetStrategyByIdQuery(strategy_id=strategy_id)
    return await handler.handle_get_by_id(query)


async def get_active_strategies(handler: StrategyQueryHandler) -> List[Strategy]:
    """Convenience function to get active strategies"""
    query = GetActiveStrategiesQuery()
    return await handler.handle_get_active_strategies(query)


async def get_strategy_performance(handler: StrategyQueryHandler, strategy_id: str,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get strategy performance"""
    query = GetStrategyPerformanceQuery(
        strategy_id=strategy_id,
        start_date=start_date,
        end_date=end_date
    )
    return await handler.handle_get_performance(query)