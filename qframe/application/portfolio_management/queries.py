"""
Application Queries: Portfolio Management
========================================

Requêtes pour la consultation des données de gestion des portfolios.
Implémente le pattern CQRS pour la séparation des responsabilités.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

from ..base.query import Query, QueryHandler, QueryResult
from ...domain.entities.portfolio import Portfolio, PortfolioStatus, PortfolioType
from ...domain.repositories.portfolio_repository import (
    PortfolioRepository,
    PortfolioQuery,
    PortfolioAggregateQuery
)
from ...domain.services.portfolio_service import PortfolioService


@dataclass
class GetPortfolioQuery(Query):
    """Requête pour récupérer un portfolio par ID"""
    portfolio_id: str


@dataclass
class GetPortfolioByNameQuery(Query):
    """Requête pour récupérer un portfolio par nom"""
    name: str


@dataclass
class GetAllPortfoliosQuery(Query):
    """Requête pour récupérer tous les portfolios"""
    include_archived: bool = False


@dataclass
class GetPortfoliosByStatusQuery(Query):
    """Requête pour récupérer les portfolios par statut"""
    status: PortfolioStatus


@dataclass
class GetPortfoliosByTypeQuery(Query):
    """Requête pour récupérer les portfolios par type"""
    portfolio_type: PortfolioType


@dataclass
class GetActivePortfoliosQuery(Query):
    """Requête pour récupérer tous les portfolios actifs"""
    pass


@dataclass
class GetPortfoliosByStrategyQuery(Query):
    """Requête pour récupérer les portfolios utilisant une stratégie"""
    strategy_id: str


@dataclass
class GetPortfoliosBySymbolQuery(Query):
    """Requête pour récupérer les portfolios ayant une position sur un symbole"""
    symbol: str


@dataclass
class GetPortfoliosNeedingRebalancingQuery(Query):
    """Requête pour récupérer les portfolios nécessitant un rééquilibrage"""
    threshold: Decimal = Decimal("0.05")


@dataclass
class GetPortfoliosViolatingConstraintsQuery(Query):
    """Requête pour récupérer les portfolios violant leurs contraintes"""
    pass


@dataclass
class GetPortfolioPerformanceQuery(Query):
    """Requête pour récupérer l'analyse de performance d'un portfolio"""
    portfolio_id: str
    analysis_period_days: int = 252
    benchmark_returns: Optional[List[Decimal]] = None


@dataclass
class GetPortfolioStatisticsQuery(Query):
    """Requête pour récupérer les statistiques d'un portfolio"""
    portfolio_id: str


@dataclass
class GetGlobalPortfolioStatisticsQuery(Query):
    """Requête pour récupérer les statistiques globales des portfolios"""
    pass


@dataclass
class SearchPortfoliosQuery(Query):
    """Requête de recherche avancée de portfolios"""
    query_builder: PortfolioQuery


@dataclass
class GetPortfolioComparisonQuery(Query):
    """Requête pour comparer plusieurs portfolios"""
    portfolio_ids: List[str]
    metric: str = "sharpe_ratio"


@dataclass
class GetPortfolioRebalancingPlanQuery(Query):
    """Requête pour obtenir un plan de rééquilibrage"""
    portfolio_id: str
    target_allocations: Optional[Dict[str, Decimal]] = None
    rebalancing_threshold: Decimal = Decimal("0.05")


@dataclass
class GetPortfolioAllocationAnalysisQuery(Query):
    """Requête pour analyser l'allocation d'un portfolio"""
    portfolio_id: str
    optimization_method: str = "equal_weight"
    parameters: Optional[Dict[str, Any]] = None


class GetPortfolioHandler(QueryHandler[GetPortfolioQuery]):
    """Handler pour récupérer un portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: GetPortfolioQuery) -> QueryResult:
        """
        Traite la requête de récupération de portfolio.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec le portfolio ou erreur si introuvable
        """
        try:
            portfolio = await self.repository.find_by_id(query.portfolio_id)

            if not portfolio:
                return QueryResult(
                    success=False,
                    error_message=f"Portfolio not found: {query.portfolio_id}"
                )

            return QueryResult(
                success=True,
                data=portfolio.to_dict(),
                message="Portfolio retrieved successfully"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving portfolio: {str(e)}"
            )


class GetPortfolioByNameHandler(QueryHandler[GetPortfolioByNameQuery]):
    """Handler pour récupérer un portfolio par nom"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: GetPortfolioByNameQuery) -> QueryResult:
        """
        Traite la requête de récupération par nom.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec le portfolio trouvé
        """
        try:
            portfolio = await self.repository.find_by_name(query.name)

            if not portfolio:
                return QueryResult(
                    success=False,
                    error_message=f"Portfolio not found: {query.name}"
                )

            return QueryResult(
                success=True,
                data=portfolio.to_dict(),
                message="Portfolio retrieved successfully"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving portfolio: {str(e)}"
            )


class GetAllPortfoliosHandler(QueryHandler[GetAllPortfoliosQuery]):
    """Handler pour récupérer tous les portfolios"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: GetAllPortfoliosQuery) -> QueryResult:
        """
        Traite la requête de récupération de tous les portfolios.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec la liste des portfolios
        """
        try:
            portfolios = await self.repository.find_all()

            # Filtrer les portfolios archivés si demandé
            if not query.include_archived:
                portfolios = [p for p in portfolios if p.status != PortfolioStatus.ARCHIVED]

            portfolios_data = [portfolio.to_dict() for portfolio in portfolios]

            return QueryResult(
                success=True,
                data={
                    "portfolios": portfolios_data,
                    "count": len(portfolios_data),
                    "include_archived": query.include_archived
                },
                message=f"{len(portfolios_data)} portfolios retrieved"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving portfolios: {str(e)}"
            )


class GetPortfoliosByStatusHandler(QueryHandler[GetPortfoliosByStatusQuery]):
    """Handler pour récupérer les portfolios par statut"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: GetPortfoliosByStatusQuery) -> QueryResult:
        """
        Traite la requête de récupération par statut.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les portfolios du statut spécifié
        """
        try:
            portfolios = await self.repository.find_by_status(query.status)
            portfolios_data = [portfolio.to_dict() for portfolio in portfolios]

            return QueryResult(
                success=True,
                data={
                    "portfolios": portfolios_data,
                    "count": len(portfolios_data),
                    "status": query.status.value
                },
                message=f"{len(portfolios_data)} portfolios with status {query.status.value}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving portfolios: {str(e)}"
            )


class GetActivePortfoliosHandler(QueryHandler[GetActivePortfoliosQuery]):
    """Handler pour récupérer tous les portfolios actifs"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: GetActivePortfoliosQuery) -> QueryResult:
        """
        Traite la requête de récupération des portfolios actifs.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les portfolios actifs
        """
        try:
            portfolios = await self.repository.find_active_portfolios()
            portfolios_data = [portfolio.to_dict() for portfolio in portfolios]

            # Calculer des statistiques supplémentaires
            total_value = sum(portfolio.total_value for portfolio in portfolios)
            avg_positions = sum(len(portfolio.positions) for portfolio in portfolios) / len(portfolios) if portfolios else 0

            return QueryResult(
                success=True,
                data={
                    "portfolios": portfolios_data,
                    "count": len(portfolios_data),
                    "total_value": float(total_value),
                    "average_positions": round(avg_positions, 1)
                },
                message=f"{len(portfolios_data)} active portfolios retrieved"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving active portfolios: {str(e)}"
            )


class GetPortfoliosByStrategyHandler(QueryHandler[GetPortfoliosByStrategyQuery]):
    """Handler pour récupérer les portfolios utilisant une stratégie"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: GetPortfoliosByStrategyQuery) -> QueryResult:
        """
        Traite la requête de récupération par stratégie.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les portfolios utilisant la stratégie
        """
        try:
            portfolios = await self.repository.find_by_strategy(query.strategy_id)
            portfolios_data = [portfolio.to_dict() for portfolio in portfolios]

            return QueryResult(
                success=True,
                data={
                    "portfolios": portfolios_data,
                    "count": len(portfolios_data),
                    "strategy_id": query.strategy_id
                },
                message=f"{len(portfolios_data)} portfolios using strategy {query.strategy_id}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving portfolios: {str(e)}"
            )


class GetPortfoliosNeedingRebalancingHandler(QueryHandler[GetPortfoliosNeedingRebalancingQuery]):
    """Handler pour récupérer les portfolios nécessitant un rééquilibrage"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: GetPortfoliosNeedingRebalancingQuery) -> QueryResult:
        """
        Traite la requête de récupération des portfolios à rééquilibrer.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les portfolios nécessitant un rééquilibrage
        """
        try:
            portfolios = await self.repository.find_portfolios_needing_rebalancing(query.threshold)
            portfolios_data = []

            for portfolio in portfolios:
                portfolio_dict = portfolio.to_dict()
                # Ajouter des informations sur les écarts d'allocation
                drifts = portfolio.get_allocation_drift()
                portfolio_dict["allocation_drifts"] = {symbol: float(drift) for symbol, drift in drifts.items()}
                portfolio_dict["max_drift"] = float(max(abs(drift) for drift in drifts.values())) if drifts else 0
                portfolios_data.append(portfolio_dict)

            return QueryResult(
                success=True,
                data={
                    "portfolios": portfolios_data,
                    "count": len(portfolios_data),
                    "threshold": float(query.threshold)
                },
                message=f"{len(portfolios_data)} portfolios need rebalancing"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving portfolios: {str(e)}"
            )


class GetPortfolioPerformanceHandler(QueryHandler[GetPortfolioPerformanceQuery]):
    """Handler pour récupérer l'analyse de performance d'un portfolio"""

    def __init__(self, repository: PortfolioRepository, portfolio_service: PortfolioService):
        self.repository = repository
        self.portfolio_service = portfolio_service

    async def handle(self, query: GetPortfolioPerformanceQuery) -> QueryResult:
        """
        Traite la requête d'analyse de performance.

        Args:
            query: Requête d'analyse

        Returns:
            Résultat avec l'analyse de performance
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(query.portfolio_id)
            if not portfolio:
                return QueryResult(
                    success=False,
                    error_message=f"Portfolio not found: {query.portfolio_id}"
                )

            # Analyser la performance
            analysis = self.portfolio_service.analyze_portfolio_performance(
                portfolio,
                query.analysis_period_days,
                query.benchmark_returns
            )

            return QueryResult(
                success=True,
                data=analysis.to_dict(),
                message="Performance analysis completed"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error analyzing performance: {str(e)}"
            )


class GetPortfolioStatisticsHandler(QueryHandler[GetPortfolioStatisticsQuery]):
    """Handler pour récupérer les statistiques d'un portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: GetPortfolioStatisticsQuery) -> QueryResult:
        """
        Traite la requête de statistiques de portfolio.

        Args:
            query: Requête de statistiques

        Returns:
            Résultat avec les statistiques
        """
        try:
            stats = await self.repository.get_portfolio_statistics(query.portfolio_id)

            if not stats:
                return QueryResult(
                    success=False,
                    error_message=f"Portfolio not found: {query.portfolio_id}"
                )

            return QueryResult(
                success=True,
                data=stats,
                message="Portfolio statistics retrieved"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving statistics: {str(e)}"
            )


class GetGlobalPortfolioStatisticsHandler(QueryHandler[GetGlobalPortfolioStatisticsQuery]):
    """Handler pour récupérer les statistiques globales des portfolios"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: GetGlobalPortfolioStatisticsQuery) -> QueryResult:
        """
        Traite la requête de statistiques globales.

        Args:
            query: Requête de statistiques

        Returns:
            Résultat avec les statistiques globales
        """
        try:
            # Récupérer les statistiques globales
            global_stats = await self.repository.get_global_statistics()

            # Ajouter des statistiques par statut et type
            status_counts = await self.repository.count_by_status()
            type_counts = await self.repository.count_by_type()
            type_values = await self.repository.get_total_value_by_type()

            enhanced_stats = {
                **global_stats,
                "by_status": {status.value: count for status, count in status_counts.items()},
                "by_type": {ptype.value: count for ptype, count in type_counts.items()},
                "total_value_by_type": {ptype.value: float(value) for ptype, value in type_values.items()}
            }

            return QueryResult(
                success=True,
                data=enhanced_stats,
                message="Global portfolio statistics retrieved"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving global statistics: {str(e)}"
            )


class SearchPortfoliosHandler(QueryHandler[SearchPortfoliosQuery]):
    """Handler pour la recherche avancée de portfolios"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, query: SearchPortfoliosQuery) -> QueryResult:
        """
        Traite la requête de recherche avancée.

        Args:
            query: Requête de recherche

        Returns:
            Résultat avec les portfolios correspondants
        """
        try:
            # Pour l'instant, on simule une recherche basique
            # Dans une vraie implémentation, on utiliserait le query_builder
            # pour construire une requête complexe

            filters = query.query_builder.filters
            portfolios = []

            # Appliquer les filtres un par un (simplification)
            if "status" in filters:
                portfolios = await self.repository.find_by_status(filters["status"])
            elif "type" in filters:
                portfolios = await self.repository.find_by_type(filters["type"])
            elif "strategy_id" in filters:
                portfolios = await self.repository.find_by_strategy(filters["strategy_id"])
            elif "symbol" in filters:
                portfolios = await self.repository.find_by_symbol(filters["symbol"])
            elif "active_only" in filters and filters["active_only"]:
                portfolios = await self.repository.find_active_portfolios()
            else:
                portfolios = await self.repository.find_all()

            # Appliquer la pagination
            if query.query_builder.limit:
                offset = query.query_builder.offset
                limit = query.query_builder.limit
                portfolios = portfolios[offset:offset + limit]

            portfolios_data = [portfolio.to_dict() for portfolio in portfolios]

            return QueryResult(
                success=True,
                data={
                    "portfolios": portfolios_data,
                    "count": len(portfolios_data),
                    "filters_applied": filters,
                    "pagination": {
                        "limit": query.query_builder.limit,
                        "offset": query.query_builder.offset
                    }
                },
                message=f"Search completed: {len(portfolios_data)} results"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error searching portfolios: {str(e)}"
            )


class GetPortfolioComparisonHandler(QueryHandler[GetPortfolioComparisonQuery]):
    """Handler pour comparer plusieurs portfolios"""

    def __init__(self, repository: PortfolioRepository, portfolio_service: PortfolioService):
        self.repository = repository
        self.portfolio_service = portfolio_service

    async def handle(self, query: GetPortfolioComparisonQuery) -> QueryResult:
        """
        Traite la requête de comparaison de portfolios.

        Args:
            query: Requête de comparaison

        Returns:
            Résultat avec la comparaison des portfolios
        """
        try:
            # Récupérer tous les portfolios
            portfolios = []
            for portfolio_id in query.portfolio_ids:
                portfolio = await self.repository.find_by_id(portfolio_id)
                if portfolio:
                    portfolios.append(portfolio)

            if not portfolios:
                return QueryResult(
                    success=False,
                    error_message="No valid portfolios found for comparison"
                )

            # Comparer les portfolios
            comparison_results = self.portfolio_service.compare_portfolios(portfolios, query.metric)

            comparison_data = []
            for portfolio, analysis in comparison_results:
                comparison_data.append({
                    "portfolio": {
                        "id": portfolio.id,
                        "name": portfolio.name,
                        "total_value": float(portfolio.total_value),
                        "positions_count": len(portfolio.positions)
                    },
                    "performance": analysis.to_dict(),
                    "metric_value": float(getattr(analysis, query.metric, 0))
                })

            return QueryResult(
                success=True,
                data={
                    "comparison": comparison_data,
                    "comparison_metric": query.metric,
                    "portfolios_count": len(comparison_data),
                    "best_performer": comparison_data[0] if comparison_data else None
                },
                message=f"Comparison of {len(comparison_data)} portfolios completed"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error comparing portfolios: {str(e)}"
            )


class GetPortfolioRebalancingPlanHandler(QueryHandler[GetPortfolioRebalancingPlanQuery]):
    """Handler pour obtenir un plan de rééquilibrage"""

    def __init__(self, repository: PortfolioRepository, portfolio_service: PortfolioService):
        self.repository = repository
        self.portfolio_service = portfolio_service

    async def handle(self, query: GetPortfolioRebalancingPlanQuery) -> QueryResult:
        """
        Traite la requête de plan de rééquilibrage.

        Args:
            query: Requête de plan

        Returns:
            Résultat avec le plan de rééquilibrage
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(query.portfolio_id)
            if not portfolio:
                return QueryResult(
                    success=False,
                    error_message=f"Portfolio not found: {query.portfolio_id}"
                )

            # Créer le plan de rééquilibrage
            rebalancing_plan = self.portfolio_service.create_rebalancing_plan(
                portfolio,
                query.target_allocations,
                query.rebalancing_threshold
            )

            if not rebalancing_plan:
                return QueryResult(
                    success=True,
                    data={
                        "portfolio_id": query.portfolio_id,
                        "rebalancing_needed": False,
                        "current_allocations": {
                            symbol: float(weight) for symbol, weight in
                            portfolio.get_allocation_drift().items()
                        }
                    },
                    message="Portfolio does not need rebalancing"
                )

            plan_data = {
                "portfolio_id": rebalancing_plan.portfolio_id,
                "timestamp": rebalancing_plan.timestamp.isoformat(),
                "target_allocations": {symbol: float(weight) for symbol, weight in rebalancing_plan.target_allocations.items()},
                "current_allocations": {symbol: float(weight) for symbol, weight in rebalancing_plan.current_allocations.items()},
                "trades_required": {symbol: float(amount) for symbol, amount in rebalancing_plan.trades_required.items()},
                "estimated_cost": float(rebalancing_plan.estimated_cost),
                "reason": rebalancing_plan.reason,
                "trade_value": float(rebalancing_plan.get_trade_value()),
                "symbols_to_buy": rebalancing_plan.get_symbols_to_buy(),
                "symbols_to_sell": rebalancing_plan.get_symbols_to_sell()
            }

            return QueryResult(
                success=True,
                data=plan_data,
                message="Rebalancing plan generated successfully"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error generating rebalancing plan: {str(e)}"
            )


class GetPortfolioAllocationAnalysisHandler(QueryHandler[GetPortfolioAllocationAnalysisQuery]):
    """Handler pour analyser l'allocation d'un portfolio"""

    def __init__(self, repository: PortfolioRepository, portfolio_service: PortfolioService):
        self.repository = repository
        self.portfolio_service = portfolio_service

    async def handle(self, query: GetPortfolioAllocationAnalysisQuery) -> QueryResult:
        """
        Traite la requête d'analyse d'allocation.

        Args:
            query: Requête d'analyse

        Returns:
            Résultat avec l'analyse d'allocation
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(query.portfolio_id)
            if not portfolio:
                return QueryResult(
                    success=False,
                    error_message=f"Portfolio not found: {query.portfolio_id}"
                )

            # Obtenir les symboles du portfolio
            symbols = list(portfolio.positions.keys())
            if not symbols:
                return QueryResult(
                    success=False,
                    error_message="Portfolio has no positions to analyze"
                )

            # Analyser selon la méthode choisie
            optimization = None
            parameters = query.parameters or {}

            if query.optimization_method == "equal_weight":
                optimization = self.portfolio_service.optimize_allocation_equal_weight(symbols)

            elif query.optimization_method == "risk_parity":
                risk_estimates = parameters.get("risk_estimates", {symbol: Decimal("0.15") for symbol in symbols})
                optimization = self.portfolio_service.optimize_allocation_risk_parity(symbols, risk_estimates)

            elif query.optimization_method == "momentum":
                momentum_scores = parameters.get("momentum_scores", {symbol: Decimal("0.1") for symbol in symbols})
                optimization = self.portfolio_service.optimize_allocation_momentum(symbols, momentum_scores)

            else:
                return QueryResult(
                    success=False,
                    error_message=f"Unknown optimization method: {query.optimization_method}"
                )

            # Calculer les allocations actuelles
            current_allocations = {}
            for symbol, position in portfolio.positions.items():
                if portfolio.total_value > 0:
                    current_allocations[symbol] = abs(position.market_value) / portfolio.total_value

            allocation_data = {
                "portfolio_id": query.portfolio_id,
                "optimization_method": query.optimization_method,
                "current_allocations": {symbol: float(weight) for symbol, weight in current_allocations.items()},
                "optimized_allocations": {symbol: float(weight) for symbol, weight in optimization.optimized_allocations.items()},
                "allocation_changes": {symbol: float(change) for symbol, change in optimization.get_allocation_changes().items()},
                "expected_metrics": {
                    "return": float(optimization.expected_return),
                    "risk": float(optimization.expected_risk),
                    "sharpe_ratio": float(optimization.sharpe_ratio)
                },
                "constraints_applied": optimization.constraints_applied
            }

            return QueryResult(
                success=True,
                data=allocation_data,
                message=f"Allocation analysis completed using {query.optimization_method}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error analyzing allocation: {str(e)}"
            )