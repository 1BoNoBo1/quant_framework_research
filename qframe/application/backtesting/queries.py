"""
Application Layer: Backtesting Query Handlers
============================================

Handlers pour les requêtes liées au backtesting.
Implémente le pattern CQRS pour les opérations de lecture.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any

from ..base.query import Query, QueryHandler, QueryResult
from ...domain.entities.backtest import BacktestConfiguration, BacktestResult, BacktestStatus, BacktestType
from ...domain.repositories.backtest_repository import BacktestRepository


# Queries

@dataclass
class GetBacktestConfigurationQuery(Query):
    """Requête pour récupérer une configuration de backtest"""
    configuration_id: str


@dataclass
class GetAllBacktestConfigurationsQuery(Query):
    """Requête pour récupérer toutes les configurations"""
    include_deleted: bool = False


@dataclass
class FindBacktestConfigurationsByNameQuery(Query):
    """Requête pour chercher des configurations par nom"""
    name: str
    exact_match: bool = False


@dataclass
class GetBacktestResultQuery(Query):
    """Requête pour récupérer un résultat de backtest"""
    result_id: str
    include_detailed_data: bool = True  # Portfolio values, trades, etc.


@dataclass
class FindBacktestResultsByConfigurationQuery(Query):
    """Requête pour trouver tous les résultats d'une configuration"""
    configuration_id: str
    status_filter: Optional[BacktestStatus] = None


@dataclass
class FindBacktestResultsByStatusQuery(Query):
    """Requête pour trouver tous les résultats par statut"""
    status: BacktestStatus
    limit: Optional[int] = None


@dataclass
class FindBacktestResultsByDateRangeQuery(Query):
    """Requête pour trouver les résultats par période de création"""
    start_date: datetime
    end_date: datetime
    status_filter: Optional[BacktestStatus] = None


@dataclass
class FindBacktestResultsByStrategyQuery(Query):
    """Requête pour trouver tous les résultats utilisant une stratégie"""
    strategy_id: str
    include_archived: bool = False


@dataclass
class GetLatestBacktestResultsQuery(Query):
    """Requête pour récupérer les derniers résultats"""
    limit: int = 10
    status_filter: Optional[BacktestStatus] = None


@dataclass
class FindBestPerformingBacktestsQuery(Query):
    """Requête pour trouver les backtests les plus performants"""
    metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, calmar_ratio, etc.
    limit: int = 10
    min_trades: int = 10
    time_period_days: Optional[int] = None  # Filtrer par période récente


@dataclass
class FindBacktestsByMetricsCriteriaQuery(Query):
    """Requête pour trouver des backtests selon des critères de performance"""
    min_sharpe_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    min_win_rate: Optional[Decimal] = None
    min_return: Optional[Decimal] = None
    min_trades: Optional[int] = None
    max_volatility: Optional[Decimal] = None


@dataclass
class GetBacktestPerformanceComparisonQuery(Query):
    """Requête pour comparer les performances de plusieurs backtests"""
    result_ids: List[str]
    include_charts_data: bool = False


@dataclass
class SearchBacktestResultsQuery(Query):
    """Requête de recherche textuelle dans les backtests"""
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 50


@dataclass
class GetBacktestStatisticsQuery(Query):
    """Requête pour obtenir des statistiques globales"""
    include_performance_distribution: bool = True
    include_strategy_breakdown: bool = True


@dataclass
class GetStrategyPerformanceSummaryQuery(Query):
    """Requête pour obtenir un résumé de performance par stratégie"""
    strategy_id: str
    time_period_days: Optional[int] = None


@dataclass
class GetMonthlyPerformanceSummaryQuery(Query):
    """Requête pour obtenir un résumé mensuel des performances"""
    year: int
    strategy_id: Optional[str] = None


@dataclass
class GetBacktestDashboardQuery(Query):
    """Requête pour obtenir les données du dashboard"""
    include_recent_results: bool = True
    include_performance_metrics: bool = True
    include_running_backtests: bool = True
    recent_days: int = 30


@dataclass
class GetArchivedBacktestResultsQuery(Query):
    """Requête pour récupérer les résultats archivés"""
    limit: Optional[int] = None


@dataclass
class GetBacktestStorageUsageQuery(Query):
    """Requête pour obtenir l'utilisation du stockage"""
    detailed_breakdown: bool = False


# Query Handlers

class GetBacktestConfigurationHandler(QueryHandler[GetBacktestConfigurationQuery]):
    """Handler pour récupérer une configuration de backtest"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, query: GetBacktestConfigurationQuery) -> QueryResult:
        try:
            config = await self.backtest_repository.get_configuration(query.configuration_id)

            if not config:
                return QueryResult(
                    success=False,
                    error_message=f"Configuration {query.configuration_id} not found"
                )

            return QueryResult(
                success=True,
                data={
                    "configuration": {
                        "id": config.id,
                        "name": config.name,
                        "description": config.description,
                        "start_date": config.start_date.isoformat(),
                        "end_date": config.end_date.isoformat(),
                        "initial_capital": float(config.initial_capital),
                        "strategy_ids": config.strategy_ids,
                        "strategy_allocations": {k: float(v) for k, v in config.strategy_allocations.items()},
                        "benchmark_symbol": config.benchmark_symbol,
                        "transaction_cost": float(config.transaction_cost),
                        "slippage": float(config.slippage),
                        "rebalance_frequency": config.rebalance_frequency.value,
                        "max_position_size": float(config.max_position_size),
                        "max_leverage": float(config.max_leverage),
                        "backtest_type": config.backtest_type.value,
                        "tags": config.tags,
                        "created_at": config.created_at.isoformat(),
                        "created_by": config.created_by
                    }
                }
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Failed to retrieve configuration: {str(e)}"
            )


class GetAllBacktestConfigurationsHandler(QueryHandler[GetAllBacktestConfigurationsQuery]):
    """Handler pour récupérer toutes les configurations"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, query: GetAllBacktestConfigurationsQuery) -> QueryResult:
        try:
            configs = await self.backtest_repository.get_all_configurations()

            configurations = []
            for config in configs:
                configurations.append({
                    "id": config.id,
                    "name": config.name,
                    "description": config.description,
                    "start_date": config.start_date.isoformat(),
                    "end_date": config.end_date.isoformat(),
                    "initial_capital": float(config.initial_capital),
                    "strategy_count": len(config.strategy_ids),
                    "backtest_type": config.backtest_type.value,
                    "created_at": config.created_at.isoformat(),
                    "created_by": config.created_by
                })

            return QueryResult(
                success=True,
                data={
                    "configurations": configurations,
                    "total_count": len(configurations)
                }
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Failed to retrieve configurations: {str(e)}"
            )


class FindBacktestConfigurationsByNameHandler(QueryHandler[FindBacktestConfigurationsByNameQuery]):
    """Handler pour chercher des configurations par nom"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, query: FindBacktestConfigurationsByNameQuery) -> QueryResult:
        try:
            configs = await self.backtest_repository.find_configurations_by_name(query.name)

            if query.exact_match:
                configs = [c for c in configs if c.name == query.name]

            configurations = []
            for config in configs:
                configurations.append({
                    "id": config.id,
                    "name": config.name,
                    "description": config.description,
                    "backtest_type": config.backtest_type.value,
                    "created_at": config.created_at.isoformat()
                })

            return QueryResult(
                success=True,
                data={
                    "configurations": configurations,
                    "search_term": query.name,
                    "exact_match": query.exact_match,
                    "total_count": len(configurations)
                }
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Failed to search configurations: {str(e)}"
            )


class GetBacktestResultHandler(QueryHandler[GetBacktestResultQuery]):
    """Handler pour récupérer un résultat de backtest"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, query: GetBacktestResultQuery) -> QueryResult:
        try:
            result = await self.backtest_repository.get_result(query.result_id)

            if not result:
                return QueryResult(
                    success=False,
                    error_message=f"Backtest result {query.result_id} not found"
                )

            result_data = {
                "id": result.id,
                "configuration_id": result.configuration_id,
                "name": result.name,
                "status": result.status.value,
                "start_time": result.start_time.isoformat() if result.start_time else None,
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration_seconds": result.duration.total_seconds() if result.duration else None,
                "error_message": result.error_message,
                "initial_capital": float(result.initial_capital),
                "final_capital": float(result.final_capital),
                "total_trades": result.total_trades,
                "trading_period_days": result.trading_period_days,
                "created_at": result.created_at.isoformat(),
                "tags": result.tags
            }

            # Ajouter les métriques si disponibles
            if result.metrics:
                result_data["metrics"] = {
                    "total_return": float(result.metrics.total_return),
                    "annualized_return": float(result.metrics.annualized_return),
                    "volatility": float(result.metrics.volatility),
                    "sharpe_ratio": float(result.metrics.sharpe_ratio),
                    "max_drawdown": float(result.metrics.max_drawdown),
                    "win_rate": float(result.metrics.win_rate),
                    "profit_factor": float(result.metrics.profit_factor),
                    "sortino_ratio": float(result.metrics.sortino_ratio),
                    "calmar_ratio": float(result.metrics.calmar_ratio),
                    "total_trades": result.metrics.total_trades,
                    "winning_trades": result.metrics.winning_trades,
                    "losing_trades": result.metrics.losing_trades
                }

                # Assessment de performance
                result_data["performance_assessment"] = result.metrics.assess_performance()

            # Ajouter les données détaillées si demandées
            if query.include_detailed_data:
                if result.portfolio_values is not None:
                    result_data["portfolio_values"] = result.portfolio_values.to_dict()
                if result.returns is not None:
                    result_data["returns"] = result.returns.to_dict()
                if result.drawdown_series is not None:
                    result_data["drawdown_series"] = result.drawdown_series.to_dict()

                # Statistiques des trades
                result_data["trade_statistics"] = result.get_trade_statistics()

            return QueryResult(
                success=True,
                data={"result": result_data}
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Failed to retrieve backtest result: {str(e)}"
            )


class FindBestPerformingBacktestsHandler(QueryHandler[FindBestPerformingBacktestsQuery]):
    """Handler pour trouver les backtests les plus performants"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, query: FindBestPerformingBacktestsQuery) -> QueryResult:
        try:
            results = await self.backtest_repository.find_best_performing_results(
                metric=query.metric,
                limit=query.limit,
                min_trades=query.min_trades
            )

            best_results = []
            for result in results:
                if result.metrics:
                    metric_value = getattr(result.metrics, query.metric, 0)
                    best_results.append({
                        "id": result.id,
                        "name": result.name,
                        "configuration_id": result.configuration_id,
                        "metric_value": float(metric_value),
                        "total_return": float(result.metrics.total_return),
                        "sharpe_ratio": float(result.metrics.sharpe_ratio),
                        "max_drawdown": float(result.metrics.max_drawdown),
                        "total_trades": result.metrics.total_trades,
                        "created_at": result.created_at.isoformat()
                    })

            return QueryResult(
                success=True,
                data={
                    "best_results": best_results,
                    "metric": query.metric,
                    "limit": query.limit,
                    "min_trades": query.min_trades,
                    "total_count": len(best_results)
                }
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Failed to find best performing backtests: {str(e)}"
            )


class GetBacktestPerformanceComparisonHandler(QueryHandler[GetBacktestPerformanceComparisonQuery]):
    """Handler pour comparer les performances de plusieurs backtests"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, query: GetBacktestPerformanceComparisonQuery) -> QueryResult:
        try:
            comparison_data = await self.backtest_repository.get_performance_comparison(query.result_ids)

            comparisons = {}
            for result_id, metrics in comparison_data.items():
                if metrics:
                    comparisons[result_id] = {
                        "total_return": float(metrics.total_return),
                        "annualized_return": float(metrics.annualized_return),
                        "volatility": float(metrics.volatility),
                        "sharpe_ratio": float(metrics.sharpe_ratio),
                        "max_drawdown": float(metrics.max_drawdown),
                        "win_rate": float(metrics.win_rate),
                        "profit_factor": float(metrics.profit_factor),
                        "total_trades": metrics.total_trades
                    }

            # Calculer des métriques de comparaison
            comparison_summary = {}
            if comparisons:
                metrics_keys = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
                for metric in metrics_keys:
                    values = [comp[metric] for comp in comparisons.values() if metric in comp]
                    if values:
                        comparison_summary[metric] = {
                            "best": max(values) if metric != 'max_drawdown' else max(values),  # Pour drawdown, moins négatif est mieux
                            "worst": min(values) if metric != 'max_drawdown' else min(values),
                            "average": sum(values) / len(values),
                            "range": max(values) - min(values)
                        }

            return QueryResult(
                success=True,
                data={
                    "comparisons": comparisons,
                    "summary": comparison_summary,
                    "result_count": len(query.result_ids),
                    "valid_comparisons": len(comparisons)
                }
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Failed to compare backtest performance: {str(e)}"
            )


class GetBacktestStatisticsHandler(QueryHandler[GetBacktestStatisticsQuery]):
    """Handler pour obtenir des statistiques globales"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, query: GetBacktestStatisticsQuery) -> QueryResult:
        try:
            stats = await self.backtest_repository.get_backtest_statistics()

            # Enrichir avec des statistiques calculées
            status_counts = await self.backtest_repository.get_results_count_by_status()
            type_counts = await self.backtest_repository.get_results_count_by_type()

            enhanced_stats = {
                **stats,
                "status_distribution": {status.value: count for status, count in status_counts.items()},
                "type_distribution": {btype.value: count for btype, count in type_counts.items()}
            }

            return QueryResult(
                success=True,
                data={"statistics": enhanced_stats}
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Failed to retrieve backtest statistics: {str(e)}"
            )


class GetBacktestDashboardHandler(QueryHandler[GetBacktestDashboardQuery]):
    """Handler pour obtenir les données du dashboard"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, query: GetBacktestDashboardQuery) -> QueryResult:
        try:
            dashboard_data = {}

            # Résultats récents
            if query.include_recent_results:
                recent_results = await self.backtest_repository.get_latest_results(10)
                dashboard_data["recent_results"] = [
                    {
                        "id": r.id,
                        "name": r.name,
                        "status": r.status.value,
                        "created_at": r.created_at.isoformat(),
                        "total_return": float(r.metrics.total_return) if r.metrics else None,
                        "sharpe_ratio": float(r.metrics.sharpe_ratio) if r.metrics else None
                    }
                    for r in recent_results
                ]

            # Backtests en cours
            if query.include_running_backtests:
                running_results = await self.backtest_repository.find_results_by_status(BacktestStatus.RUNNING)
                dashboard_data["running_backtests"] = [
                    {
                        "id": r.id,
                        "name": r.name,
                        "start_time": r.start_time.isoformat() if r.start_time else None,
                        "configuration_id": r.configuration_id
                    }
                    for r in running_results
                ]

            # Métriques de performance
            if query.include_performance_metrics:
                stats = await self.backtest_repository.get_backtest_statistics()
                dashboard_data["performance_metrics"] = stats

            return QueryResult(
                success=True,
                data={"dashboard": dashboard_data}
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Failed to retrieve dashboard data: {str(e)}"
            )