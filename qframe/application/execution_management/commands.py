"""
Application Commands: Execution Management
==========================================

Commandes pour les opérations de gestion d'exécution des ordres.
Implémente le pattern CQRS pour la séparation des responsabilités.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

from ..base.command import Command, CommandHandler, CommandResult
from ...domain.entities.order import (
    Order, OrderType, OrderSide, TimeInForce, OrderPriority, OrderExecution,
    create_market_order, create_limit_order, create_stop_order, create_stop_limit_order
)
from ...domain.repositories.order_repository import OrderRepository
from ...domain.services.execution_service import (
    ExecutionService, RoutingStrategy, ExecutionAlgorithm, ExecutionVenue, VenueQuote
)


@dataclass
class CreateOrderCommand(Command):
    """Commande pour créer un nouvel ordre"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    priority: OrderPriority = OrderPriority.NORMAL
    notes: str = ""


@dataclass
class SubmitOrderCommand(Command):
    """Commande pour soumettre un ordre au marché"""
    order_id: str
    routing_strategy: RoutingStrategy = RoutingStrategy.BEST_PRICE
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.IMMEDIATE
    venue_preference: Optional[List[ExecutionVenue]] = None


@dataclass
class ModifyOrderCommand(Command):
    """Commande pour modifier un ordre"""
    order_id: str
    new_quantity: Optional[Decimal] = None
    new_price: Optional[Decimal] = None
    new_priority: Optional[OrderPriority] = None


@dataclass
class CancelOrderCommand(Command):
    """Commande pour annuler un ordre"""
    order_id: str
    reason: str = "User requested"


@dataclass
class ExecuteOrderCommand(Command):
    """Commande pour exécuter un ordre selon un plan"""
    order_id: str
    market_data: Dict[str, Dict[str, Any]]  # venue -> quote data
    force_execution: bool = False


@dataclass
class AddExecutionCommand(Command):
    """Commande pour ajouter une exécution à un ordre"""
    order_id: str
    executed_quantity: Decimal
    execution_price: Decimal
    venue: str
    commission: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    liquidity_flag: str = "unknown"


@dataclass
class CreateExecutionPlanCommand(Command):
    """Commande pour créer un plan d'exécution"""
    order_id: str
    market_data: Dict[str, Dict[str, Any]]
    routing_strategy: RoutingStrategy = RoutingStrategy.BEST_PRICE
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.IMMEDIATE


@dataclass
class CreateChildOrdersCommand(Command):
    """Commande pour créer des ordres enfants"""
    parent_order_id: str
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.TWAP
    num_slices: int = 5


@dataclass
class BulkCancelOrdersCommand(Command):
    """Commande pour annuler plusieurs ordres"""
    order_ids: List[str]
    reason: str = "Bulk cancellation"


@dataclass
class SetOrderPriorityCommand(Command):
    """Commande pour définir la priorité d'un ordre"""
    order_id: str
    priority: OrderPriority


class CreateOrderHandler(CommandHandler[CreateOrderCommand]):
    """Handler pour créer un nouvel ordre"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, command: CreateOrderCommand) -> CommandResult:
        """
        Traite la commande de création d'ordre.

        Args:
            command: Commande de création

        Returns:
            Résultat avec l'ID de l'ordre créé
        """
        try:
            # Créer l'ordre selon le type
            if command.order_type == OrderType.MARKET:
                order = create_market_order(
                    symbol=command.symbol,
                    side=command.side,
                    quantity=command.quantity,
                    portfolio_id=command.portfolio_id,
                    strategy_id=command.strategy_id
                )
            elif command.order_type == OrderType.LIMIT:
                if command.price is None:
                    return CommandResult(
                        success=False,
                        error_message="Limit order requires a price"
                    )
                order = create_limit_order(
                    symbol=command.symbol,
                    side=command.side,
                    quantity=command.quantity,
                    price=command.price,
                    time_in_force=command.time_in_force,
                    portfolio_id=command.portfolio_id,
                    strategy_id=command.strategy_id
                )
            elif command.order_type == OrderType.STOP:
                if command.stop_price is None:
                    return CommandResult(
                        success=False,
                        error_message="Stop order requires a stop price"
                    )
                order = create_stop_order(
                    symbol=command.symbol,
                    side=command.side,
                    quantity=command.quantity,
                    stop_price=command.stop_price,
                    portfolio_id=command.portfolio_id,
                    strategy_id=command.strategy_id
                )
            elif command.order_type == OrderType.STOP_LIMIT:
                if command.stop_price is None or command.price is None:
                    return CommandResult(
                        success=False,
                        error_message="Stop limit order requires both stop price and limit price"
                    )
                order = create_stop_limit_order(
                    symbol=command.symbol,
                    side=command.side,
                    quantity=command.quantity,
                    stop_price=command.stop_price,
                    limit_price=command.price,
                    portfolio_id=command.portfolio_id,
                    strategy_id=command.strategy_id
                )
            else:
                return CommandResult(
                    success=False,
                    error_message=f"Unsupported order type: {command.order_type}"
                )

            # Configurer les propriétés additionnelles
            order.priority = command.priority
            order.notes = command.notes

            # Sauvegarder
            await self.repository.save(order)

            return CommandResult(
                success=True,
                result_data={
                    "order_id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": float(order.quantity),
                    "order_type": order.order_type.value,
                    "status": order.status.value
                },
                message=f"Order created successfully: {order.symbol} {order.side.value} {order.quantity}"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error creating order: {str(e)}"
            )


class SubmitOrderHandler(CommandHandler[SubmitOrderCommand]):
    """Handler pour soumettre un ordre au marché"""

    def __init__(self, repository: OrderRepository, execution_service: ExecutionService):
        self.repository = repository
        self.execution_service = execution_service

    async def handle(self, command: SubmitOrderCommand) -> CommandResult:
        """
        Traite la commande de soumission d'ordre.

        Args:
            command: Commande de soumission

        Returns:
            Résultat de la soumission
        """
        try:
            # Récupérer l'ordre
            order = await self.repository.find_by_id(command.order_id)
            if not order:
                return CommandResult(
                    success=False,
                    error_message=f"Order not found: {command.order_id}"
                )

            # Vérifier que l'ordre peut être soumis
            if not order.status.value == "pending":
                return CommandResult(
                    success=False,
                    error_message=f"Cannot submit order in status: {order.status.value}"
                )

            # Soumettre l'ordre
            broker_order_id = f"BRK_{order.id[:8]}"  # Simuler un ID broker
            order.submit(broker_order_id)

            # Marquer comme accepté immédiatement (simplification)
            order.accept()

            # Sauvegarder
            await self.repository.update(order)

            return CommandResult(
                success=True,
                result_data={
                    "order_id": order.id,
                    "broker_order_id": order.broker_order_id,
                    "status": order.status.value,
                    "submitted_time": order.submitted_time.isoformat() if order.submitted_time else None,
                    "routing_strategy": command.routing_strategy.value,
                    "execution_algorithm": command.execution_algorithm.value
                },
                message=f"Order submitted successfully: {order.id}"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error submitting order: {str(e)}"
            )


class ModifyOrderHandler(CommandHandler[ModifyOrderCommand]):
    """Handler pour modifier un ordre"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, command: ModifyOrderCommand) -> CommandResult:
        """
        Traite la commande de modification d'ordre.

        Args:
            command: Commande de modification

        Returns:
            Résultat de la modification
        """
        try:
            # Récupérer l'ordre
            order = await self.repository.find_by_id(command.order_id)
            if not order:
                return CommandResult(
                    success=False,
                    error_message=f"Order not found: {command.order_id}"
                )

            # Vérifier que l'ordre peut être modifié
            if not order.is_active():
                return CommandResult(
                    success=False,
                    error_message=f"Cannot modify order in status: {order.status.value}"
                )

            # Appliquer les modifications
            modifications = []

            if command.new_quantity is not None:
                order.modify_quantity(command.new_quantity)
                modifications.append(f"quantity: {float(command.new_quantity)}")

            if command.new_price is not None:
                order.modify_price(command.new_price)
                modifications.append(f"price: {float(command.new_price)}")

            if command.new_priority is not None:
                order.priority = command.new_priority
                modifications.append(f"priority: {command.new_priority.value}")

            # Sauvegarder
            await self.repository.update(order)

            return CommandResult(
                success=True,
                result_data={
                    "order_id": order.id,
                    "modifications": modifications,
                    "new_quantity": float(order.quantity),
                    "new_price": float(order.price) if order.price else None,
                    "new_priority": order.priority.value
                },
                message=f"Order modified successfully: {'; '.join(modifications)}"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error modifying order: {str(e)}"
            )


class CancelOrderHandler(CommandHandler[CancelOrderCommand]):
    """Handler pour annuler un ordre"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, command: CancelOrderCommand) -> CommandResult:
        """
        Traite la commande d'annulation d'ordre.

        Args:
            command: Commande d'annulation

        Returns:
            Résultat de l'annulation
        """
        try:
            # Récupérer l'ordre
            order = await self.repository.find_by_id(command.order_id)
            if not order:
                return CommandResult(
                    success=False,
                    error_message=f"Order not found: {command.order_id}"
                )

            # Annuler l'ordre
            order.cancel(command.reason)

            # Sauvegarder
            await self.repository.update(order)

            return CommandResult(
                success=True,
                result_data={
                    "order_id": order.id,
                    "status": order.status.value,
                    "cancel_reason": command.reason,
                    "cancelled_time": order.last_update_time.isoformat()
                },
                message=f"Order cancelled successfully: {command.reason}"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error cancelling order: {str(e)}"
            )


class ExecuteOrderHandler(CommandHandler[ExecuteOrderCommand]):
    """Handler pour exécuter un ordre"""

    def __init__(self, repository: OrderRepository, execution_service: ExecutionService):
        self.repository = repository
        self.execution_service = execution_service

    async def handle(self, command: ExecuteOrderCommand) -> CommandResult:
        """
        Traite la commande d'exécution d'ordre.

        Args:
            command: Commande d'exécution

        Returns:
            Résultat de l'exécution
        """
        try:
            # Récupérer l'ordre
            order = await self.repository.find_by_id(command.order_id)
            if not order:
                return CommandResult(
                    success=False,
                    error_message=f"Order not found: {command.order_id}"
                )

            # Vérifier que l'ordre peut être exécuté
            if not order.is_active() and not command.force_execution:
                return CommandResult(
                    success=False,
                    error_message=f"Cannot execute order in status: {order.status.value}"
                )

            # Convertir les données de marché
            market_data = {}
            for venue_name, quote_data in command.market_data.items():
                try:
                    venue = ExecutionVenue(venue_name)
                    quote = VenueQuote(
                        venue=venue,
                        symbol=order.symbol,
                        bid_price=Decimal(str(quote_data["bid_price"])),
                        ask_price=Decimal(str(quote_data["ask_price"])),
                        bid_size=Decimal(str(quote_data["bid_size"])),
                        ask_size=Decimal(str(quote_data["ask_size"])),
                        timestamp=datetime.fromisoformat(quote_data.get("timestamp", datetime.utcnow().isoformat()))
                    )
                    market_data[venue] = quote
                except (ValueError, KeyError) as e:
                    continue  # Ignorer les venues avec des données invalides

            if not market_data:
                return CommandResult(
                    success=False,
                    error_message="No valid market data provided"
                )

            # Créer le plan d'exécution
            execution_plan = self.execution_service.create_execution_plan(
                order, market_data, RoutingStrategy.BEST_PRICE, ExecutionAlgorithm.IMMEDIATE
            )

            # Exécuter l'ordre
            executions = self.execution_service.execute_order(order, execution_plan, market_data)

            # Sauvegarder
            await self.repository.update(order)

            # Créer le rapport d'exécution
            execution_report = self.execution_service.create_execution_report(order)

            return CommandResult(
                success=True,
                result_data={
                    "order_id": order.id,
                    "executions_count": len(executions),
                    "total_filled": float(order.filled_quantity),
                    "remaining_quantity": float(order.remaining_quantity),
                    "average_price": float(order.average_fill_price),
                    "order_status": order.status.value,
                    "execution_plan": execution_plan.to_dict(),
                    "execution_report": execution_report.to_dict()
                },
                message=f"Order executed: {len(executions)} fills, {float(order.filled_quantity)} filled"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error executing order: {str(e)}"
            )


class AddExecutionHandler(CommandHandler[AddExecutionCommand]):
    """Handler pour ajouter une exécution à un ordre"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, command: AddExecutionCommand) -> CommandResult:
        """
        Traite la commande d'ajout d'exécution.

        Args:
            command: Commande d'ajout d'exécution

        Returns:
            Résultat de l'ajout
        """
        try:
            # Récupérer l'ordre
            order = await self.repository.find_by_id(command.order_id)
            if not order:
                return CommandResult(
                    success=False,
                    error_message=f"Order not found: {command.order_id}"
                )

            # Créer l'exécution
            execution = OrderExecution(
                executed_quantity=command.executed_quantity,
                execution_price=command.execution_price,
                commission=command.commission,
                fees=command.fees,
                venue=command.venue,
                liquidity_flag=command.liquidity_flag
            )

            # Ajouter à l'ordre
            order.add_execution(execution)

            # Sauvegarder
            await self.repository.update(order)

            return CommandResult(
                success=True,
                result_data={
                    "order_id": order.id,
                    "execution_id": execution.execution_id,
                    "executed_quantity": float(execution.executed_quantity),
                    "execution_price": float(execution.execution_price),
                    "execution_value": float(execution.execution_value),
                    "total_filled": float(order.filled_quantity),
                    "order_status": order.status.value
                },
                message=f"Execution added: {float(execution.executed_quantity)} @ {float(execution.execution_price)}"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error adding execution: {str(e)}"
            )


class CreateExecutionPlanHandler(CommandHandler[CreateExecutionPlanCommand]):
    """Handler pour créer un plan d'exécution"""

    def __init__(self, repository: OrderRepository, execution_service: ExecutionService):
        self.repository = repository
        self.execution_service = execution_service

    async def handle(self, command: CreateExecutionPlanCommand) -> CommandResult:
        """
        Traite la commande de création de plan d'exécution.

        Args:
            command: Commande de création de plan

        Returns:
            Résultat avec le plan d'exécution
        """
        try:
            # Récupérer l'ordre
            order = await self.repository.find_by_id(command.order_id)
            if not order:
                return CommandResult(
                    success=False,
                    error_message=f"Order not found: {command.order_id}"
                )

            # Convertir les données de marché
            market_data = {}
            for venue_name, quote_data in command.market_data.items():
                try:
                    venue = ExecutionVenue(venue_name)
                    quote = VenueQuote(
                        venue=venue,
                        symbol=order.symbol,
                        bid_price=Decimal(str(quote_data["bid_price"])),
                        ask_price=Decimal(str(quote_data["ask_price"])),
                        bid_size=Decimal(str(quote_data["bid_size"])),
                        ask_size=Decimal(str(quote_data["ask_size"])),
                        timestamp=datetime.fromisoformat(quote_data.get("timestamp", datetime.utcnow().isoformat()))
                    )
                    market_data[venue] = quote
                except (ValueError, KeyError):
                    continue

            if not market_data:
                return CommandResult(
                    success=False,
                    error_message="No valid market data provided"
                )

            # Créer le plan d'exécution
            execution_plan = self.execution_service.create_execution_plan(
                order, market_data, command.routing_strategy, command.execution_algorithm
            )

            return CommandResult(
                success=True,
                result_data=execution_plan.to_dict(),
                message="Execution plan created successfully"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error creating execution plan: {str(e)}"
            )


class CreateChildOrdersHandler(CommandHandler[CreateChildOrdersCommand]):
    """Handler pour créer des ordres enfants"""

    def __init__(self, repository: OrderRepository, execution_service: ExecutionService):
        self.repository = repository
        self.execution_service = execution_service

    async def handle(self, command: CreateChildOrdersCommand) -> CommandResult:
        """
        Traite la commande de création d'ordres enfants.

        Args:
            command: Commande de création d'ordres enfants

        Returns:
            Résultat avec les ordres enfants créés
        """
        try:
            # Récupérer l'ordre parent
            parent_order = await self.repository.find_by_id(command.parent_order_id)
            if not parent_order:
                return CommandResult(
                    success=False,
                    error_message=f"Parent order not found: {command.parent_order_id}"
                )

            # Créer un plan d'exécution fictif pour les ordres enfants
            market_data = {
                ExecutionVenue.BINANCE: VenueQuote(
                    venue=ExecutionVenue.BINANCE,
                    symbol=parent_order.symbol,
                    bid_price=Decimal("100"),
                    ask_price=Decimal("101"),
                    bid_size=Decimal("1000"),
                    ask_size=Decimal("1000"),
                    timestamp=datetime.utcnow()
                )
            }

            execution_plan = self.execution_service.create_execution_plan(
                parent_order, market_data, RoutingStrategy.SMART_ORDER_ROUTING, command.execution_algorithm
            )

            # Créer les ordres enfants
            child_orders = self.execution_service.create_child_orders(parent_order, execution_plan)

            # Sauvegarder les ordres enfants
            for child_order in child_orders:
                await self.repository.save(child_order)

            child_orders_data = [
                {
                    "order_id": child.id,
                    "client_order_id": child.client_order_id,
                    "quantity": float(child.quantity),
                    "destination": child.destination
                }
                for child in child_orders
            ]

            return CommandResult(
                success=True,
                result_data={
                    "parent_order_id": parent_order.id,
                    "child_orders": child_orders_data,
                    "total_children": len(child_orders),
                    "execution_algorithm": command.execution_algorithm.value
                },
                message=f"Created {len(child_orders)} child orders for parent {parent_order.id}"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error creating child orders: {str(e)}"
            )


class BulkCancelOrdersHandler(CommandHandler[BulkCancelOrdersCommand]):
    """Handler pour annuler plusieurs ordres"""

    def __init__(self, repository: OrderRepository):
        self.repository = repository

    async def handle(self, command: BulkCancelOrdersCommand) -> CommandResult:
        """
        Traite la commande d'annulation en lot.

        Args:
            command: Commande d'annulation en lot

        Returns:
            Résultat de l'annulation en lot
        """
        try:
            cancelled_orders = []
            failed_cancellations = []

            for order_id in command.order_ids:
                try:
                    order = await self.repository.find_by_id(order_id)
                    if not order:
                        failed_cancellations.append({"order_id": order_id, "reason": "Order not found"})
                        continue

                    if not order.is_active():
                        failed_cancellations.append({"order_id": order_id, "reason": f"Order not active: {order.status.value}"})
                        continue

                    order.cancel(command.reason)
                    await self.repository.update(order)
                    cancelled_orders.append({"order_id": order_id, "status": order.status.value})

                except Exception as e:
                    failed_cancellations.append({"order_id": order_id, "reason": str(e)})

            return CommandResult(
                success=True,
                result_data={
                    "total_requested": len(command.order_ids),
                    "successfully_cancelled": len(cancelled_orders),
                    "failed_cancellations": len(failed_cancellations),
                    "cancelled_orders": cancelled_orders,
                    "failed_orders": failed_cancellations,
                    "reason": command.reason
                },
                message=f"Bulk cancellation completed: {len(cancelled_orders)} cancelled, {len(failed_cancellations)} failed"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error in bulk cancellation: {str(e)}"
            )