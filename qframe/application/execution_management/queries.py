"""
Application Queries: Execution Management
=========================================

Requêtes pour la consultation des données de gestion d'exécution.
Implémente le pattern CQRS pour la séparation des responsabilités.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

from ..base.query import Query, QueryHandler, QueryResult
from ...domain.entities.order import Order, OrderStatus, OrderType, OrderSide, OrderPriority
from ...domain.repositories.order_repository import (
    OrderRepository,
    OrderQuery,
    OrderAggregateQuery
)
from ...domain.services.execution_service import ExecutionService


@dataclass
class GetOrderQuery(Query):
    """Requête pour récupérer un ordre par ID"""
    order_id: str


@dataclass
class GetOrderByClientIdQuery(Query):
    """Requête pour récupérer un ordre par ID client"""
    client_order_id: str


@dataclass
class GetOrdersByStatusQuery(Query):
    """Requête pour récupérer les ordres par statut"""
    status: OrderStatus
    limit: Optional[int] = None


@dataclass
class GetActiveOrdersQuery(Query):
    """Requête pour récupérer tous les ordres actifs"""
    symbol: Optional[str] = None
    portfolio_id: Optional[str] = None


@dataclass
class GetOrdersBySymbolQuery(Query):
    """Requête pour récupérer les ordres par symbole"""
    symbol: str
    include_terminal: bool = False


@dataclass
class GetOrdersByPortfolioQuery(Query):
    """Requête pour récupérer les ordres d'un portfolio"""
    portfolio_id: str
    status_filter: Optional[OrderStatus] = None


@dataclass
class GetOrdersByStrategyQuery(Query):
    """Requête pour récupérer les ordres d'une stratégie"""
    strategy_id: str
    date_range: Optional[tuple[datetime, datetime]] = None


@dataclass
class GetParentOrdersQuery(Query):
    """Requête pour récupérer les ordres parents"""
    include_children: bool = True


@dataclass
class GetChildOrdersQuery(Query):
    """Requête pour récupérer les ordres enfants d'un parent"""
    parent_order_id: str


@dataclass
class GetExpiredOrdersQuery(Query):
    """Requête pour récupérer les ordres expirés"""
    auto_mark_expired: bool = False


@dataclass
class GetExecutionReportQuery(Query):
    """Requête pour récupérer le rapport d'exécution d'un ordre"""
    order_id: str
    benchmark_price: Optional[Decimal] = None


@dataclass
class GetOrderStatisticsQuery(Query):
    """Requête pour récupérer les statistiques d'ordres"""
    symbol: Optional[str] = None
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    date_range: Optional[tuple[datetime, datetime]] = None


@dataclass
class GetExecutionStatisticsQuery(Query):
    """Requête pour récupérer les statistiques d'exécution"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    venue_filter: Optional[str] = None


@dataclass
class SearchOrdersQuery(Query):
    """Requête de recherche avancée d'ordres"""
    query_builder: OrderQuery


@dataclass
class GetOrderBookQuery(Query):
    """Requête pour récupérer le book d'ordres"""
    symbol: str
    side: Optional[OrderSide] = None
    active_only: bool = True


@dataclass
class GetExecutionProgressQuery(Query):
    """Requête pour récupérer le progrès d'exécution"""
    parent_order_id: str


class GetOrderHandler(QueryHandler[GetOrderQuery]):
    """Handler pour récupérer un ordre"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetOrderQuery) -> QueryResult:
        """
        Traite la requête de récupération d'ordre.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec l'ordre ou erreur si introuvable
        """
        try:
            order = await self.repository.find_by_id(query.order_id)

            if not order:
                return QueryResult(
                    success=False,
                    error_message=f"Order not found: {query.order_id}"
                )

            return QueryResult(
                success=True,
                data=order.to_dict(),
                message="Order retrieved successfully"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving order: {str(e)}"
            )


class GetOrderByClientIdHandler(QueryHandler[GetOrderByClientIdQuery]):
    """Handler pour récupérer un ordre par ID client"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetOrderByClientIdQuery) -> QueryResult:
        """
        Traite la requête de récupération par ID client.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec l'ordre trouvé
        """
        try:
            order = await self.repository.find_by_client_order_id(query.client_order_id)

            if not order:
                return QueryResult(
                    success=False,
                    error_message=f"Order not found with client ID: {query.client_order_id}"
                )

            return QueryResult(
                success=True,
                data=order.to_dict(),
                message="Order retrieved successfully"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving order: {str(e)}"
            )


class GetOrdersByStatusHandler(QueryHandler[GetOrdersByStatusQuery]):
    """Handler pour récupérer les ordres par statut"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetOrdersByStatusQuery) -> QueryResult:
        """
        Traite la requête de récupération par statut.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les ordres du statut spécifié
        """
        try:
            orders = await self.repository.find_by_status(query.status)

            # Limiter les résultats si demandé
            if query.limit and len(orders) > query.limit:
                orders = orders[:query.limit]

            orders_data = [order.to_dict() for order in orders]

            return QueryResult(
                success=True,
                data={
                    "orders": orders_data,
                    "count": len(orders_data),
                    "status": query.status.value,
                    "limited": bool(query.limit and len(orders_data) == query.limit)
                },
                message=f"{len(orders_data)} orders with status {query.status.value}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving orders: {str(e)}"
            )


class GetActiveOrdersHandler(QueryHandler[GetActiveOrdersQuery]):
    """Handler pour récupérer tous les ordres actifs"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetActiveOrdersQuery) -> QueryResult:
        """
        Traite la requête de récupération des ordres actifs.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les ordres actifs
        """
        try:
            orders = await self.repository.find_active_orders()

            # Filtrer par symbole si spécifié
            if query.symbol:
                orders = [order for order in orders if order.symbol == query.symbol]

            # Filtrer par portfolio si spécifié
            if query.portfolio_id:
                orders = [order for order in orders if order.portfolio_id == query.portfolio_id]

            orders_data = [order.to_dict() for order in orders]

            # Calculer des statistiques
            total_notional = sum(order.notional_value for order in orders)
            symbols = list(set(order.symbol for order in orders))

            return QueryResult(
                success=True,
                data={
                    "orders": orders_data,
                    "count": len(orders_data),
                    "total_notional_value": float(total_notional),
                    "unique_symbols": len(symbols),
                    "filters": {
                        "symbol": query.symbol,
                        "portfolio_id": query.portfolio_id
                    }
                },
                message=f"{len(orders_data)} active orders retrieved"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving active orders: {str(e)}"
            )


class GetOrdersBySymbolHandler(QueryHandler[GetOrdersBySymbolQuery]):
    """Handler pour récupérer les ordres par symbole"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetOrdersBySymbolQuery) -> QueryResult:
        """
        Traite la requête de récupération par symbole.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les ordres du symbole
        """
        try:
            orders = await self.repository.find_by_symbol(query.symbol)

            # Filtrer les ordres terminaux si pas demandés
            if not query.include_terminal:
                orders = [order for order in orders if not order.is_terminal()]

            orders_data = [order.to_dict() for order in orders]

            # Analyser les ordres
            buy_orders = [order for order in orders if order.side == OrderSide.BUY]
            sell_orders = [order for order in orders if order.side == OrderSide.SELL]

            total_buy_quantity = sum(order.quantity for order in buy_orders)
            total_sell_quantity = sum(order.quantity for order in sell_orders)

            return QueryResult(
                success=True,
                data={
                    "orders": orders_data,
                    "count": len(orders_data),
                    "symbol": query.symbol,
                    "buy_orders": len(buy_orders),
                    "sell_orders": len(sell_orders),
                    "total_buy_quantity": float(total_buy_quantity),
                    "total_sell_quantity": float(total_sell_quantity),
                    "include_terminal": query.include_terminal
                },
                message=f"{len(orders_data)} orders for {query.symbol}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving orders: {str(e)}"
            )


class GetOrdersByPortfolioHandler(QueryHandler[GetOrdersByPortfolioQuery]):
    """Handler pour récupérer les ordres d'un portfolio"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetOrdersByPortfolioQuery) -> QueryResult:
        """
        Traite la requête de récupération par portfolio.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les ordres du portfolio
        """
        try:
            orders = await self.repository.find_by_portfolio(query.portfolio_id)

            # Filtrer par statut si spécifié
            if query.status_filter:
                orders = [order for order in orders if order.status == query.status_filter]

            orders_data = [order.to_dict() for order in orders]

            # Calculer des statistiques du portfolio
            total_orders = len(orders)
            active_orders = len([order for order in orders if order.is_active()])
            filled_orders = len([order for order in orders if order.is_filled()])

            return QueryResult(
                success=True,
                data={
                    "orders": orders_data,
                    "count": total_orders,
                    "portfolio_id": query.portfolio_id,
                    "active_orders": active_orders,
                    "filled_orders": filled_orders,
                    "status_filter": query.status_filter.value if query.status_filter else None
                },
                message=f"{total_orders} orders for portfolio {query.portfolio_id}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving orders: {str(e)}"
            )


class GetParentOrdersHandler(QueryHandler[GetParentOrdersQuery]):
    """Handler pour récupérer les ordres parents"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetParentOrdersQuery) -> QueryResult:
        """
        Traite la requête de récupération des ordres parents.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les ordres parents
        """
        try:
            parent_orders = await self.repository.find_parent_orders()
            parent_orders_data = []

            for parent_order in parent_orders:
                parent_data = parent_order.to_dict()

                if query.include_children:
                    # Récupérer les ordres enfants
                    child_orders = await self.repository.find_child_orders(parent_order.id)
                    parent_data["child_orders"] = [child.to_dict() for child in child_orders]
                    parent_data["children_count"] = len(child_orders)

                parent_orders_data.append(parent_data)

            return QueryResult(
                success=True,
                data={
                    "parent_orders": parent_orders_data,
                    "count": len(parent_orders_data),
                    "include_children": query.include_children
                },
                message=f"{len(parent_orders_data)} parent orders retrieved"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving parent orders: {str(e)}"
            )


class GetChildOrdersHandler(QueryHandler[GetChildOrdersQuery]):
    """Handler pour récupérer les ordres enfants"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetChildOrdersQuery) -> QueryResult:
        """
        Traite la requête de récupération des ordres enfants.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les ordres enfants
        """
        try:
            child_orders = await self.repository.find_child_orders(query.parent_order_id)
            child_orders_data = [order.to_dict() for order in child_orders]

            # Analyser le progrès d'exécution
            total_quantity = sum(order.quantity for order in child_orders)
            total_filled = sum(order.filled_quantity for order in child_orders)
            fill_percentage = (total_filled / total_quantity * 100) if total_quantity > 0 else 0

            active_children = len([order for order in child_orders if order.is_active()])
            completed_children = len([order for order in child_orders if order.is_terminal()])

            return QueryResult(
                success=True,
                data={
                    "child_orders": child_orders_data,
                    "count": len(child_orders_data),
                    "parent_order_id": query.parent_order_id,
                    "execution_progress": {
                        "total_quantity": float(total_quantity),
                        "total_filled": float(total_filled),
                        "fill_percentage": float(fill_percentage),
                        "active_children": active_children,
                        "completed_children": completed_children
                    }
                },
                message=f"{len(child_orders_data)} child orders for parent {query.parent_order_id}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving child orders: {str(e)}"
            )


class GetExecutionReportHandler(QueryHandler[GetExecutionReportQuery]):
    """Handler pour récupérer le rapport d'exécution d'un ordre"""

    def __init__(self, repository: OrderRepository, execution_service: ExecutionService):
        self.repository = repository
        self.execution_service = execution_service

    async def handle(self, query: GetExecutionReportQuery) -> QueryResult:
        """
        Traite la requête de rapport d'exécution.

        Args:
            query: Requête de rapport

        Returns:
            Résultat avec le rapport d'exécution
        """
        try:
            # Récupérer l'ordre
            order = await self.repository.find_by_id(query.order_id)
            if not order:
                return QueryResult(
                    success=False,
                    error_message=f"Order not found: {query.order_id}"
                )

            # Générer le rapport d'exécution
            execution_report = self.execution_service.create_execution_report(
                order, query.benchmark_price
            )

            return QueryResult(
                success=True,
                data=execution_report.to_dict(),
                message="Execution report generated successfully"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error generating execution report: {str(e)}"
            )


class GetOrderStatisticsHandler(QueryHandler[GetOrderStatisticsQuery]):
    """Handler pour récupérer les statistiques d'ordres"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetOrderStatisticsQuery) -> QueryResult:
        """
        Traite la requête de statistiques d'ordres.

        Args:
            query: Requête de statistiques

        Returns:
            Résultat avec les statistiques
        """
        try:
            # Récupérer les statistiques de base
            stats = await self.repository.get_order_statistics(
                query.symbol,
                query.portfolio_id,
                query.strategy_id
            )

            # Ajouter des statistiques par statut
            status_counts = await self.repository.count_by_status()

            # Statistiques par symbole (top 10)
            symbol_counts = await self.repository.count_by_symbol(limit=10)

            enhanced_stats = {
                **stats,
                "by_status": {status.value: count for status, count in status_counts.items()},
                "top_symbols": symbol_counts,
                "filters": {
                    "symbol": query.symbol,
                    "portfolio_id": query.portfolio_id,
                    "strategy_id": query.strategy_id,
                    "date_range": [
                        query.date_range[0].isoformat(),
                        query.date_range[1].isoformat()
                    ] if query.date_range else None
                }
            }

            return QueryResult(
                success=True,
                data=enhanced_stats,
                message="Order statistics retrieved successfully"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving order statistics: {str(e)}"
            )


class GetExecutionStatisticsHandler(QueryHandler[GetExecutionStatisticsQuery]):
    """Handler pour récupérer les statistiques d'exécution"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetExecutionStatisticsQuery) -> QueryResult:
        """
        Traite la requête de statistiques d'exécution.

        Args:
            query: Requête de statistiques

        Returns:
            Résultat avec les statistiques d'exécution
        """
        try:
            # Récupérer les statistiques d'exécution
            execution_stats = await self.repository.get_execution_statistics(
                query.start_date,
                query.end_date
            )

            return QueryResult(
                success=True,
                data={
                    **execution_stats,
                    "filters": {
                        "start_date": query.start_date.isoformat() if query.start_date else None,
                        "end_date": query.end_date.isoformat() if query.end_date else None,
                        "venue_filter": query.venue_filter
                    }
                },
                message="Execution statistics retrieved successfully"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving execution statistics: {str(e)}"
            )


class SearchOrdersHandler(QueryHandler[SearchOrdersQuery]):
    """Handler pour la recherche avancée d'ordres"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: SearchOrdersQuery) -> QueryResult:
        """
        Traite la requête de recherche avancée.

        Args:
            query: Requête de recherche

        Returns:
            Résultat avec les ordres correspondants
        """
        try:
            # Pour l'instant, on simule une recherche basique
            # Dans une vraie implémentation, on utiliserait le query_builder
            # pour construire une requête complexe

            filters = query.query_builder.filters
            orders = []

            # Appliquer les filtres un par un (simplification)
            if "status" in filters:
                orders = await self.repository.find_by_status(filters["status"])
            elif "symbol" in filters:
                orders = await self.repository.find_by_symbol(filters["symbol"])
            elif "portfolio_id" in filters:
                orders = await self.repository.find_by_portfolio(filters["portfolio_id"])
            elif "active_only" in filters and filters["active_only"]:
                orders = await self.repository.find_active_orders()
            else:
                # Recherche générale - limiter les résultats
                orders = await self.repository.find_by_status(OrderStatus.FILLED)

            # Appliquer la pagination
            if query.query_builder.limit:
                offset = query.query_builder.offset
                limit = query.query_builder.limit
                orders = orders[offset:offset + limit]

            orders_data = [order.to_dict() for order in orders]

            return QueryResult(
                success=True,
                data={
                    "orders": orders_data,
                    "count": len(orders_data),
                    "filters_applied": filters,
                    "pagination": {
                        "limit": query.query_builder.limit,
                        "offset": query.query_builder.offset
                    }
                },
                message=f"Search completed: {len(orders_data)} results"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error searching orders: {str(e)}"
            )


class GetOrderBookHandler(QueryHandler[GetOrderBookQuery]):
    """Handler pour récupérer le book d'ordres"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, query: GetOrderBookQuery) -> QueryResult:
        """
        Traite la requête de book d'ordres.

        Args:
            query: Requête de book

        Returns:
            Résultat avec le book d'ordres
        """
        try:
            # Récupérer tous les ordres pour le symbole
            all_orders = await self.repository.find_by_symbol(query.symbol)

            # Filtrer les ordres actifs si demandé
            if query.active_only:
                all_orders = [order for order in all_orders if order.is_active()]

            # Séparer par côté
            buy_orders = [order for order in all_orders if order.side == OrderSide.BUY]
            sell_orders = [order for order in all_orders if order.side == OrderSide.SELL]

            # Filtrer par côté si spécifié
            if query.side == OrderSide.BUY:
                relevant_orders = buy_orders
            elif query.side == OrderSide.SELL:
                relevant_orders = sell_orders
            else:
                relevant_orders = all_orders

            # Trier les ordres (meilleurs prix en premier)
            buy_orders.sort(key=lambda x: x.price or Decimal("0"), reverse=True)
            sell_orders.sort(key=lambda x: x.price or Decimal("999999"))

            # Construire le book d'ordres
            order_book = {
                "symbol": query.symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "buy_orders": [
                    {
                        "order_id": order.id,
                        "price": float(order.price) if order.price else None,
                        "quantity": float(order.remaining_quantity),
                        "order_type": order.order_type.value
                    }
                    for order in buy_orders[:50]  # Top 50
                ],
                "sell_orders": [
                    {
                        "order_id": order.id,
                        "price": float(order.price) if order.price else None,
                        "quantity": float(order.remaining_quantity),
                        "order_type": order.order_type.value
                    }
                    for order in sell_orders[:50]  # Top 50
                ],
                "statistics": {
                    "total_buy_orders": len(buy_orders),
                    "total_sell_orders": len(sell_orders),
                    "total_buy_quantity": float(sum(order.remaining_quantity for order in buy_orders)),
                    "total_sell_quantity": float(sum(order.remaining_quantity for order in sell_orders))
                }
            }

            return QueryResult(
                success=True,
                data=order_book,
                message=f"Order book retrieved for {query.symbol}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving order book: {str(e)}"
            )


class GetExecutionProgressHandler(QueryHandler[GetExecutionProgressQuery]):
    """Handler pour récupérer le progrès d'exécution"""

    def __init__(self, repository: OrderRepository, execution_service: ExecutionService):
        self.repository = repository
        self.execution_service = execution_service

    async def handle(self, query: GetExecutionProgressQuery) -> QueryResult:
        """
        Traite la requête de progrès d'exécution.

        Args:
            query: Requête de progrès

        Returns:
            Résultat avec le progrès d'exécution
        """
        try:
            # Récupérer l'ordre parent
            parent_order = await self.repository.find_by_id(query.parent_order_id)
            if not parent_order:
                return QueryResult(
                    success=False,
                    error_message=f"Parent order not found: {query.parent_order_id}"
                )

            # Récupérer les ordres enfants
            child_orders = await self.repository.find_child_orders(query.parent_order_id)

            # Surveiller le progrès
            progress_report = self.execution_service.monitor_execution_progress(
                parent_order, child_orders
            )

            return QueryResult(
                success=True,
                data=progress_report,
                message="Execution progress retrieved successfully"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Error retrieving execution progress: {str(e)}"
            )