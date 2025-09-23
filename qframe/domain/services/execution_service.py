"""
Domain Service: Execution Service
=================================

Service de domaine pour la logique métier d'exécution des ordres.
Gère le routing, l'exécution, et la surveillance des ordres.
"""

from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import random

from ..entities.order import (
    Order, OrderStatus, OrderType, OrderSide, TimeInForce, OrderPriority,
    OrderExecution, create_market_order, create_limit_order
)
from ..value_objects.position import Position


class ExecutionVenue(str, Enum):
    """Venues d'exécution disponibles"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    DARK_POOL = "dark_pool"
    INTERNAL = "internal"


class RoutingStrategy(str, Enum):
    """Stratégies de routing d'ordres"""
    BEST_PRICE = "best_price"
    LOWEST_COST = "lowest_cost"
    FASTEST_EXECUTION = "fastest_execution"
    MINIMIZE_IMPACT = "minimize_impact"
    SMART_ORDER_ROUTING = "smart_order_routing"


class ExecutionAlgorithm(str, Enum):
    """Algorithmes d'exécution"""
    IMMEDIATE = "immediate"
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price
    ICEBERG = "iceberg"
    PARTICIPATE = "participate"  # Participation rate
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"


@dataclass
class VenueQuote:
    """Quote d'un venue pour un symbole"""
    venue: ExecutionVenue
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    timestamp: datetime
    spread: Decimal = None

    def __post_init__(self):
        if self.spread is None:
            self.spread = self.ask_price - self.bid_price

    def get_price_for_side(self, side: OrderSide) -> Decimal:
        """Retourne le prix pour un côté donné"""
        return self.bid_price if side == OrderSide.SELL else self.ask_price

    def get_size_for_side(self, side: OrderSide) -> Decimal:
        """Retourne la taille disponible pour un côté donné"""
        return self.bid_size if side == OrderSide.SELL else self.ask_size


@dataclass
class ExecutionPlan:
    """Plan d'exécution pour un ordre"""
    order_id: str
    target_venues: List[ExecutionVenue]
    routing_strategy: RoutingStrategy
    execution_algorithm: ExecutionAlgorithm
    estimated_cost: Decimal
    estimated_duration: timedelta
    slice_instructions: List[Dict[str, any]]  # Instructions pour les slices d'ordre
    risk_checks_passed: bool
    created_time: datetime

    def to_dict(self) -> Dict[str, any]:
        return {
            "order_id": self.order_id,
            "target_venues": [venue.value for venue in self.target_venues],
            "routing_strategy": self.routing_strategy.value,
            "execution_algorithm": self.execution_algorithm.value,
            "estimated_cost": float(self.estimated_cost),
            "estimated_duration_seconds": self.estimated_duration.total_seconds(),
            "slice_instructions": self.slice_instructions,
            "risk_checks_passed": self.risk_checks_passed,
            "created_time": self.created_time.isoformat()
        }


@dataclass
class ExecutionReport:
    """Rapport d'exécution d'un ordre"""
    order_id: str
    total_executed_quantity: Decimal
    average_execution_price: Decimal
    total_commission: Decimal
    total_fees: Decimal
    execution_time_seconds: float
    venues_used: List[str]
    slippage: Decimal
    implementation_shortfall: Decimal
    execution_quality: str  # "excellent", "good", "fair", "poor"

    def to_dict(self) -> Dict[str, any]:
        return {
            "order_id": self.order_id,
            "total_executed_quantity": float(self.total_executed_quantity),
            "average_execution_price": float(self.average_execution_price),
            "total_commission": float(self.total_commission),
            "total_fees": float(self.total_fees),
            "execution_time_seconds": self.execution_time_seconds,
            "venues_used": self.venues_used,
            "slippage": float(self.slippage),
            "implementation_shortfall": float(self.implementation_shortfall),
            "execution_quality": self.execution_quality
        }


class ExecutionService:
    """
    Service de domaine pour l'exécution des ordres.

    Fournit des méthodes pour le routing intelligent, l'exécution algorithmique,
    et la surveillance de la qualité d'exécution.
    """

    def __init__(self):
        self.supported_venues = {
            ExecutionVenue.BINANCE: {"commission_rate": Decimal("0.001"), "min_size": Decimal("0.01")},
            ExecutionVenue.COINBASE: {"commission_rate": Decimal("0.005"), "min_size": Decimal("0.001")},
            ExecutionVenue.KRAKEN: {"commission_rate": Decimal("0.0025"), "min_size": Decimal("0.001")}
        }

    # === Smart Order Routing ===

    def create_execution_plan(
        self,
        order: Order,
        market_data: Dict[ExecutionVenue, VenueQuote],
        routing_strategy: RoutingStrategy = RoutingStrategy.BEST_PRICE,
        execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.IMMEDIATE
    ) -> ExecutionPlan:
        """
        Crée un plan d'exécution optimal pour un ordre.

        Args:
            order: Ordre à exécuter
            market_data: Données de marché par venue
            routing_strategy: Stratégie de routing
            execution_algorithm: Algorithme d'exécution

        Returns:
            Plan d'exécution optimal
        """
        # Sélectionner les venues selon la stratégie
        target_venues = self._select_venues(order, market_data, routing_strategy)

        # Calculer le coût estimé
        estimated_cost = self._estimate_execution_cost(order, market_data, target_venues)

        # Calculer la durée estimée
        estimated_duration = self._estimate_execution_duration(order, execution_algorithm)

        # Créer les instructions de slicing
        slice_instructions = self._create_slice_instructions(order, target_venues, execution_algorithm)

        # Vérifications de risque
        risk_checks_passed = self._perform_pre_execution_risk_checks(order)

        return ExecutionPlan(
            order_id=order.id,
            target_venues=target_venues,
            routing_strategy=routing_strategy,
            execution_algorithm=execution_algorithm,
            estimated_cost=estimated_cost,
            estimated_duration=estimated_duration,
            slice_instructions=slice_instructions,
            risk_checks_passed=risk_checks_passed,
            created_time=datetime.utcnow()
        )

    def _select_venues(
        self,
        order: Order,
        market_data: Dict[ExecutionVenue, VenueQuote],
        strategy: RoutingStrategy
    ) -> List[ExecutionVenue]:
        """Sélectionne les venues optimales selon la stratégie"""
        available_venues = []

        for venue, quote in market_data.items():
            if venue in self.supported_venues:
                # Vérifier la taille minimale
                min_size = self.supported_venues[venue]["min_size"]
                available_size = quote.get_size_for_side(order.side)

                if order.quantity >= min_size and available_size >= min_size:
                    available_venues.append((venue, quote))

        if not available_venues:
            return []

        if strategy == RoutingStrategy.BEST_PRICE:
            # Trier par meilleur prix
            if order.side == OrderSide.BUY:
                available_venues.sort(key=lambda x: x[1].ask_price)
            else:
                available_venues.sort(key=lambda x: x[1].bid_price, reverse=True)

        elif strategy == RoutingStrategy.LOWEST_COST:
            # Trier par coût total (prix + commissions)
            venue_costs = []
            for venue, quote in available_venues:
                price = quote.get_price_for_side(order.side)
                commission_rate = self.supported_venues[venue]["commission_rate"]
                total_cost = price * (1 + commission_rate)
                venue_costs.append((venue, total_cost))

            if order.side == OrderSide.BUY:
                venue_costs.sort(key=lambda x: x[1])
            else:
                venue_costs.sort(key=lambda x: x[1], reverse=True)

            return [venue for venue, _ in venue_costs[:3]]  # Top 3 venues

        elif strategy == RoutingStrategy.FASTEST_EXECUTION:
            # Prioriser les venues avec plus de liquidité
            available_venues.sort(key=lambda x: x[1].get_size_for_side(order.side), reverse=True)

        elif strategy == RoutingStrategy.MINIMIZE_IMPACT:
            # Distribuer sur plusieurs venues pour minimiser l'impact
            return [venue for venue, _ in available_venues[:min(3, len(available_venues))]]

        elif strategy == RoutingStrategy.SMART_ORDER_ROUTING:
            # Combinaison intelligente de tous les facteurs
            scored_venues = []
            for venue, quote in available_venues:
                price = quote.get_price_for_side(order.side)
                size = quote.get_size_for_side(order.side)
                commission_rate = self.supported_venues[venue]["commission_rate"]

                # Score composite (à optimiser selon les besoins)
                price_score = 100 / float(price) if price > 0 else 0
                size_score = float(size) / 1000  # Normaliser
                cost_score = 100 / (100 + float(commission_rate) * 10000)

                total_score = price_score * 0.4 + size_score * 0.3 + cost_score * 0.3
                scored_venues.append((venue, total_score))

            scored_venues.sort(key=lambda x: x[1], reverse=True)
            return [venue for venue, _ in scored_venues[:3]]

        # Retourner les top venues
        return [venue for venue, _ in available_venues[:min(2, len(available_venues))]]

    def _estimate_execution_cost(
        self,
        order: Order,
        market_data: Dict[ExecutionVenue, VenueQuote],
        target_venues: List[ExecutionVenue]
    ) -> Decimal:
        """Estime le coût total d'exécution"""
        if not target_venues:
            return Decimal("0")

        total_cost = Decimal("0")
        remaining_quantity = order.quantity

        for venue in target_venues:
            if venue not in market_data:
                continue

            quote = market_data[venue]
            venue_config = self.supported_venues[venue]

            # Quantité que ce venue peut absorber
            available_size = quote.get_size_for_side(order.side)
            venue_quantity = min(remaining_quantity, available_size)

            if venue_quantity <= 0:
                continue

            # Coût sur ce venue
            price = quote.get_price_for_side(order.side)
            execution_value = venue_quantity * price
            commission = execution_value * venue_config["commission_rate"]

            total_cost += execution_value + commission
            remaining_quantity -= venue_quantity

            if remaining_quantity <= 0:
                break

        return total_cost

    def _estimate_execution_duration(
        self,
        order: Order,
        algorithm: ExecutionAlgorithm
    ) -> timedelta:
        """Estime la durée d'exécution selon l'algorithme"""
        base_duration = timedelta(seconds=5)  # Exécution de base

        if algorithm == ExecutionAlgorithm.IMMEDIATE:
            return base_duration

        elif algorithm == ExecutionAlgorithm.TWAP:
            # TWAP sur 30 minutes par défaut
            return timedelta(minutes=30)

        elif algorithm == ExecutionAlgorithm.VWAP:
            # VWAP sur 1 heure par défaut
            return timedelta(hours=1)

        elif algorithm == ExecutionAlgorithm.ICEBERG:
            # Dépend de la taille de l'iceberg
            slices = max(5, int(float(order.quantity) / 100))
            return timedelta(minutes=slices * 2)

        elif algorithm == ExecutionAlgorithm.PARTICIPATE:
            # Participation rate de 10% du volume
            estimated_volume_hours = 4  # Estimation
            return timedelta(hours=estimated_volume_hours)

        return base_duration

    def _create_slice_instructions(
        self,
        order: Order,
        target_venues: List[ExecutionVenue],
        algorithm: ExecutionAlgorithm
    ) -> List[Dict[str, any]]:
        """Crée les instructions de découpage d'ordre"""
        instructions = []

        if algorithm == ExecutionAlgorithm.IMMEDIATE:
            # Exécution immédiate sur le meilleur venue
            if target_venues:
                instructions.append({
                    "venue": target_venues[0].value,
                    "quantity": float(order.quantity),
                    "timing": "immediate",
                    "order_type": "market"
                })

        elif algorithm == ExecutionAlgorithm.TWAP:
            # Découper en tranches temporelles
            num_slices = 6  # 6 tranches de 5 minutes
            slice_quantity = order.quantity / num_slices

            for i in range(num_slices):
                instructions.append({
                    "venue": target_venues[i % len(target_venues)].value if target_venues else "binance",
                    "quantity": float(slice_quantity),
                    "timing": f"delay_{i * 5}_minutes",
                    "order_type": "limit"
                })

        elif algorithm == ExecutionAlgorithm.ICEBERG:
            # Découper en icebergs
            iceberg_size = order.quantity / 5  # 5 tranches

            for i in range(5):
                instructions.append({
                    "venue": target_venues[i % len(target_venues)].value if target_venues else "binance",
                    "quantity": float(iceberg_size),
                    "timing": f"after_previous_fill",
                    "order_type": "limit",
                    "hidden": True
                })

        elif algorithm == ExecutionAlgorithm.VWAP:
            # Suivre le profil de volume
            volume_profile = [0.15, 0.25, 0.35, 0.25]  # Profil typique

            for i, volume_pct in enumerate(volume_profile):
                slice_quantity = order.quantity * Decimal(str(volume_pct))
                instructions.append({
                    "venue": target_venues[i % len(target_venues)].value if target_venues else "binance",
                    "quantity": float(slice_quantity),
                    "timing": f"volume_window_{i}",
                    "order_type": "limit"
                })

        return instructions

    def _perform_pre_execution_risk_checks(self, order: Order) -> bool:
        """Effectue les vérifications de risque pré-exécution"""
        # Vérifications de base
        if order.quantity <= 0:
            return False

        if order.status not in [OrderStatus.PENDING, OrderStatus.ACCEPTED]:
            return False

        # Vérification de la valeur notionnelle
        if order.notional_value > Decimal("1000000"):  # Limite de 1M
            return False

        # Vérification des heures de marché (simplifiée)
        now = datetime.utcnow()
        if now.weekday() > 4:  # Pas le weekend
            return False

        # Toutes les vérifications passées
        return True

    # === Exécution d'ordres ===

    def execute_order(
        self,
        order: Order,
        execution_plan: ExecutionPlan,
        market_data: Dict[ExecutionVenue, VenueQuote]
    ) -> List[OrderExecution]:
        """
        Exécute un ordre selon son plan d'exécution.

        Args:
            order: Ordre à exécuter
            execution_plan: Plan d'exécution
            market_data: Données de marché actuelles

        Returns:
            Liste des exécutions réalisées
        """
        if not execution_plan.risk_checks_passed:
            raise ValueError("Risk checks failed for order execution")

        executions = []

        # Exécuter selon l'algorithme
        if execution_plan.execution_algorithm == ExecutionAlgorithm.IMMEDIATE:
            executions = self._execute_immediate(order, execution_plan, market_data)

        elif execution_plan.execution_algorithm == ExecutionAlgorithm.TWAP:
            executions = self._execute_twap(order, execution_plan, market_data)

        elif execution_plan.execution_algorithm == ExecutionAlgorithm.ICEBERG:
            executions = self._execute_iceberg(order, execution_plan, market_data)

        else:
            # Fallback vers exécution immédiate
            executions = self._execute_immediate(order, execution_plan, market_data)

        # Ajouter les exécutions à l'ordre
        for execution in executions:
            order.add_execution(execution)

        return executions

    def _execute_immediate(
        self,
        order: Order,
        execution_plan: ExecutionPlan,
        market_data: Dict[ExecutionVenue, VenueQuote]
    ) -> List[OrderExecution]:
        """Exécute immédiatement sur le meilleur venue"""
        executions = []

        if not execution_plan.target_venues:
            return executions

        venue = execution_plan.target_venues[0]
        if venue not in market_data:
            return executions

        quote = market_data[venue]
        venue_config = self.supported_venues[venue]

        # Simuler l'exécution
        execution_price = quote.get_price_for_side(order.side)
        executed_quantity = min(order.quantity, quote.get_size_for_side(order.side))

        if executed_quantity > 0:
            execution_value = executed_quantity * execution_price
            commission = execution_value * venue_config["commission_rate"]

            execution = OrderExecution(
                executed_quantity=executed_quantity,
                execution_price=execution_price,
                commission=commission,
                venue=venue.value,
                liquidity_flag="taker"  # Market order est généralement taker
            )
            executions.append(execution)

        return executions

    def _execute_twap(
        self,
        order: Order,
        execution_plan: ExecutionPlan,
        market_data: Dict[ExecutionVenue, VenueQuote]
    ) -> List[OrderExecution]:
        """Simule l'exécution TWAP (simplifié pour démo)"""
        executions = []
        remaining_quantity = order.quantity
        num_slices = len(execution_plan.slice_instructions)

        for i, instruction in enumerate(execution_plan.slice_instructions):
            if remaining_quantity <= 0:
                break

            venue_name = instruction["venue"]
            venue = ExecutionVenue(venue_name)

            if venue not in market_data or venue not in self.supported_venues:
                continue

            quote = market_data[venue]
            venue_config = self.supported_venues[venue]

            # Quantité pour cette tranche
            slice_quantity = min(
                Decimal(str(instruction["quantity"])),
                remaining_quantity,
                quote.get_size_for_side(order.side)
            )

            if slice_quantity <= 0:
                continue

            # Prix d'exécution avec un peu de variation aléatoire
            base_price = quote.get_price_for_side(order.side)
            price_variation = base_price * Decimal(str(random.uniform(-0.001, 0.001)))
            execution_price = base_price + price_variation

            execution_value = slice_quantity * execution_price
            commission = execution_value * venue_config["commission_rate"]

            execution = OrderExecution(
                executed_quantity=slice_quantity,
                execution_price=execution_price,
                commission=commission,
                venue=venue.value,
                liquidity_flag="maker" if i > 0 else "taker"  # Premier slice taker, autres maker
            )
            executions.append(execution)

            remaining_quantity -= slice_quantity

        return executions

    def _execute_iceberg(
        self,
        order: Order,
        execution_plan: ExecutionPlan,
        market_data: Dict[ExecutionVenue, VenueQuote]
    ) -> List[OrderExecution]:
        """Simule l'exécution iceberg"""
        executions = []
        remaining_quantity = order.quantity

        for instruction in execution_plan.slice_instructions:
            if remaining_quantity <= 0:
                break

            venue_name = instruction["venue"]
            venue = ExecutionVenue(venue_name)

            if venue not in market_data or venue not in self.supported_venues:
                continue

            quote = market_data[venue]
            venue_config = self.supported_venues[venue]

            # Quantité pour cet iceberg
            iceberg_quantity = min(
                Decimal(str(instruction["quantity"])),
                remaining_quantity
            )

            # Exécuter par petites tranches
            iceberg_remaining = iceberg_quantity
            while iceberg_remaining > 0 and iceberg_remaining <= quote.get_size_for_side(order.side):
                slice_size = min(iceberg_remaining, quote.get_size_for_side(order.side) * Decimal("0.1"))

                if slice_size <= 0:
                    break

                execution_price = quote.get_price_for_side(order.side)
                execution_value = slice_size * execution_price
                commission = execution_value * venue_config["commission_rate"]

                execution = OrderExecution(
                    executed_quantity=slice_size,
                    execution_price=execution_price,
                    commission=commission,
                    venue=venue.value,
                    liquidity_flag="maker"  # Iceberg généralement maker
                )
                executions.append(execution)

                iceberg_remaining -= slice_size
                remaining_quantity -= slice_size

        return executions

    # === Surveillance et reporting ===

    def create_execution_report(self, order: Order, benchmark_price: Optional[Decimal] = None) -> ExecutionReport:
        """
        Crée un rapport d'exécution pour un ordre.

        Args:
            order: Ordre exécuté
            benchmark_price: Prix de référence pour calculer le slippage

        Returns:
            Rapport d'exécution détaillé
        """
        if not order.executions:
            return ExecutionReport(
                order_id=order.id,
                total_executed_quantity=Decimal("0"),
                average_execution_price=Decimal("0"),
                total_commission=Decimal("0"),
                total_fees=Decimal("0"),
                execution_time_seconds=0,
                venues_used=[],
                slippage=Decimal("0"),
                implementation_shortfall=Decimal("0"),
                execution_quality="poor"
            )

        # Calculer les métriques
        total_executed_quantity = sum(exec.executed_quantity for exec in order.executions)
        total_commission = sum(exec.commission for exec in order.executions)
        total_fees = sum(exec.fees for exec in order.executions)

        venues_used = list(set(exec.venue for exec in order.executions))

        # Temps d'exécution
        if order.executions:
            start_time = min(exec.execution_time for exec in order.executions)
            end_time = max(exec.execution_time for exec in order.executions)
            execution_time_seconds = (end_time - start_time).total_seconds()
        else:
            execution_time_seconds = 0

        # Slippage et implementation shortfall
        slippage = Decimal("0")
        implementation_shortfall = Decimal("0")

        if benchmark_price and total_executed_quantity > 0:
            avg_price = order.average_fill_price

            if order.side == OrderSide.BUY:
                slippage = (avg_price - benchmark_price) / benchmark_price * 100
            else:
                slippage = (benchmark_price - avg_price) / benchmark_price * 100

            # Implementation shortfall simplifié
            implementation_shortfall = abs(slippage) + (total_commission / (total_executed_quantity * avg_price) * 100)

        # Qualité d'exécution
        execution_quality = self._assess_execution_quality(order, slippage, implementation_shortfall)

        return ExecutionReport(
            order_id=order.id,
            total_executed_quantity=total_executed_quantity,
            average_execution_price=order.average_fill_price,
            total_commission=total_commission,
            total_fees=total_fees,
            execution_time_seconds=execution_time_seconds,
            venues_used=venues_used,
            slippage=slippage,
            implementation_shortfall=implementation_shortfall,
            execution_quality=execution_quality
        )

    def _assess_execution_quality(
        self,
        order: Order,
        slippage: Decimal,
        implementation_shortfall: Decimal
    ) -> str:
        """Évalue la qualité d'exécution"""
        # Critères simplifiés
        abs_slippage = abs(slippage)
        abs_shortfall = abs(implementation_shortfall)

        if abs_slippage < Decimal("0.05") and abs_shortfall < Decimal("0.1"):
            return "excellent"
        elif abs_slippage < Decimal("0.1") and abs_shortfall < Decimal("0.2"):
            return "good"
        elif abs_slippage < Decimal("0.2") and abs_shortfall < Decimal("0.5"):
            return "fair"
        else:
            return "poor"

    # === Gestion des ordres enfants ===

    def create_child_orders(
        self,
        parent_order: Order,
        execution_plan: ExecutionPlan
    ) -> List[Order]:
        """
        Crée des ordres enfants selon le plan d'exécution.

        Args:
            parent_order: Ordre parent
            execution_plan: Plan d'exécution

        Returns:
            Liste des ordres enfants
        """
        child_orders = []

        for i, instruction in enumerate(execution_plan.slice_instructions):
            child_quantity = Decimal(str(instruction["quantity"]))

            if instruction.get("order_type") == "market":
                child_order = create_market_order(
                    symbol=parent_order.symbol,
                    side=parent_order.side,
                    quantity=child_quantity,
                    portfolio_id=parent_order.portfolio_id,
                    strategy_id=parent_order.strategy_id
                )
            else:
                # Prix limite basé sur l'ordre parent ou le marché
                limit_price = parent_order.price or Decimal("100")  # Placeholder
                child_order = create_limit_order(
                    symbol=parent_order.symbol,
                    side=parent_order.side,
                    quantity=child_quantity,
                    price=limit_price,
                    portfolio_id=parent_order.portfolio_id,
                    strategy_id=parent_order.strategy_id
                )

            # Configurer comme ordre enfant
            child_order.parent_order_id = parent_order.id
            child_order.destination = instruction["venue"]
            child_order.client_order_id = f"{parent_order.client_order_id}_CHILD_{i+1}"

            # Tags pour traçabilité
            child_order.add_tag("parent_order_id", parent_order.id)
            child_order.add_tag("slice_number", str(i+1))
            child_order.add_tag("execution_algorithm", execution_plan.execution_algorithm.value)

            child_orders.append(child_order)

        return child_orders

    def monitor_execution_progress(self, parent_order: Order, child_orders: List[Order]) -> Dict[str, any]:
        """
        Surveille le progrès d'exécution d'un ordre parent et ses enfants.

        Args:
            parent_order: Ordre parent
            child_orders: Ordres enfants

        Returns:
            Rapport de progrès
        """
        total_child_filled = sum(child.filled_quantity for child in child_orders)
        total_child_quantity = sum(child.quantity for child in child_orders)

        active_children = [child for child in child_orders if child.is_active()]
        completed_children = [child for child in child_orders if child.is_terminal()]

        progress_percentage = (total_child_filled / parent_order.quantity * 100) if parent_order.quantity > 0 else 0

        return {
            "parent_order_id": parent_order.id,
            "total_children": len(child_orders),
            "active_children": len(active_children),
            "completed_children": len(completed_children),
            "total_filled_quantity": float(total_child_filled),
            "total_target_quantity": float(total_child_quantity),
            "progress_percentage": float(progress_percentage),
            "estimated_completion": self._estimate_completion_time(active_children),
            "execution_status": "in_progress" if active_children else "completed"
        }

    def _estimate_completion_time(self, active_orders: List[Order]) -> Optional[str]:
        """Estime le temps de completion des ordres actifs"""
        if not active_orders:
            return None

        # Estimation simple basée sur l'âge moyen des ordres
        avg_age = sum(order.age_seconds for order in active_orders) / len(active_orders)
        estimated_remaining = max(60, avg_age * 2)  # Au moins 1 minute

        completion_time = datetime.utcnow() + timedelta(seconds=estimated_remaining)
        return completion_time.isoformat()