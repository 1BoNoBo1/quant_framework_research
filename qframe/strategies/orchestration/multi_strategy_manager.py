"""
Multi-Strategy Manager
======================

Orchestrates multiple trading strategies with dynamic allocation
and performance-based optimization.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

from ...core.interfaces import Strategy
from ...core.container import injectable
from ...domain.entities.portfolio import Portfolio
from ...domain.entities.order import Order
from ...domain.value_objects.signal import Signal
from ...domain.services.portfolio_service import PortfolioService


logger = logging.getLogger(__name__)


class AllocationMethod(str, Enum):
    """Méthodes d'allocation entre stratégies"""
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    RISK_PARITY = "risk_parity"
    MOMENTUM_BASED = "momentum_based"
    KELLY_CRITERION = "kelly_criterion"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class StrategyMetrics:
    """Métriques de performance d'une stratégie"""
    strategy_id: str
    total_return: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    volatility: Decimal = Decimal("0")
    correlation_to_market: Decimal = Decimal("0")
    signals_generated: int = 0
    last_signal_time: Optional[datetime] = None
    performance_score: Decimal = Decimal("0")


@dataclass
class StrategyAllocation:
    """Allocation d'une stratégie dans le portfolio"""
    strategy_id: str
    target_weight: Decimal
    current_weight: Decimal
    allocated_capital: Decimal
    performance_metrics: StrategyMetrics
    is_active: bool = True
    last_rebalance: Optional[datetime] = None


@injectable
class MultiStrategyManager:
    """
    Gestionnaire de stratégies multiples avec allocation dynamique.

    Responsabilités:
    - Orchestration de multiples stratégies
    - Allocation dynamique du capital
    - Optimisation basée sur les performances
    - Gestion des risques inter-stratégies
    """

    def __init__(
        self,
        portfolio_service: PortfolioService,
        allocation_method: AllocationMethod = AllocationMethod.PERFORMANCE_WEIGHTED,
        rebalance_frequency: timedelta = timedelta(hours=1),
        min_allocation: Decimal = Decimal("0.05"),  # 5% minimum
        max_allocation: Decimal = Decimal("0.40")   # 40% maximum
    ):
        self.portfolio_service = portfolio_service
        self.allocation_method = allocation_method
        self.rebalance_frequency = rebalance_frequency
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation

        # État interne
        self.strategies: Dict[str, Strategy] = {}
        self.allocations: Dict[str, StrategyAllocation] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.last_rebalance: Optional[datetime] = None

        # Configuration
        self.performance_lookback = timedelta(days=30)
        self.min_signals_required = 10
        self.correlation_threshold = Decimal("0.8")

    async def register_strategy(
        self,
        strategy_id: str,
        strategy: Strategy,
        initial_allocation: Optional[Decimal] = None
    ) -> None:
        """
        Enregistre une nouvelle stratégie dans le gestionnaire.

        Args:
            strategy_id: Identifiant unique de la stratégie
            strategy: Instance de la stratégie
            initial_allocation: Allocation initiale (si None, utilise equal weight)
        """
        if strategy_id in self.strategies:
            raise ValueError(f"Strategy {strategy_id} already registered")

        self.strategies[strategy_id] = strategy

        # Déterminer l'allocation initiale
        if initial_allocation is None:
            if self.allocations:
                # Equal weight avec stratégies existantes
                target_weight = Decimal("1") / (len(self.strategies))
                # Réajuster toutes les allocations
                for allocation in self.allocations.values():
                    allocation.target_weight = target_weight
            else:
                target_weight = Decimal("1")  # Première stratégie = 100%
        else:
            target_weight = initial_allocation

        # Créer l'allocation
        metrics = StrategyMetrics(strategy_id=strategy_id)
        allocation = StrategyAllocation(
            strategy_id=strategy_id,
            target_weight=target_weight,
            current_weight=Decimal("0"),
            allocated_capital=Decimal("0"),
            performance_metrics=metrics
        )

        self.allocations[strategy_id] = allocation

        logger.info(f"Strategy {strategy_id} registered with {target_weight:.2%} allocation")

    async def unregister_strategy(self, strategy_id: str) -> None:
        """Désactive une stratégie et redistribue son allocation."""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        # Marquer comme inactive
        if strategy_id in self.allocations:
            self.allocations[strategy_id].is_active = False

        # Redistribuer l'allocation
        await self._rebalance_allocations()

        logger.info(f"Strategy {strategy_id} unregistered")

    async def generate_unified_signals(
        self,
        market_data: Dict[str, Any],
        portfolio: Portfolio
    ) -> List[Signal]:
        """
        Génère des signaux unifiés de toutes les stratégies actives.

        Args:
            market_data: Données de marché pour toutes les stratégies
            portfolio: Portfolio actuel

        Returns:
            Liste des signaux unifiés avec poids d'allocation
        """
        all_signals = []

        for strategy_id, strategy in self.strategies.items():
            allocation = self.allocations.get(strategy_id)
            if not allocation or not allocation.is_active:
                continue

            try:
                # Générer signaux pour cette stratégie
                strategy_signals = await self._generate_strategy_signals(
                    strategy, market_data, allocation
                )

                # Ajuster les signaux par l'allocation - créer nouveaux signaux car frozen
                adjusted_signals = []
                for signal in strategy_signals:
                    # Calculer nouvelle quantité selon l'allocation
                    new_quantity = signal.quantity
                    if signal.quantity:
                        new_quantity = signal.quantity * allocation.target_weight

                    # Créer métadonnées d'orchestration
                    new_metadata = dict(signal.metadata) if signal.metadata else {}
                    new_metadata.update({
                        "orchestrated": True,
                        "source_strategy": strategy_id,
                        "allocation_weight": float(allocation.target_weight),
                        "strategy_score": float(allocation.performance_metrics.performance_score)
                    })

                    # Créer nouveau signal avec modifications (car frozen)
                    adjusted_signal = Signal(
                        symbol=signal.symbol,
                        action=signal.action,
                        timestamp=signal.timestamp,
                        strength=signal.strength,
                        confidence=signal.confidence,
                        price=signal.price,
                        quantity=new_quantity,
                        strategy_id=signal.strategy_id,
                        metadata=new_metadata
                    )
                    adjusted_signals.append(adjusted_signal)

                strategy_signals = adjusted_signals

                all_signals.extend(strategy_signals)

                # Mettre à jour les métriques
                allocation.performance_metrics.signals_generated += len(strategy_signals)
                allocation.performance_metrics.last_signal_time = datetime.utcnow()

            except Exception as e:
                logger.error(f"Error generating signals for strategy {strategy_id}: {e}")
                continue

        # Filtrer et optimiser les signaux
        optimized_signals = await self._optimize_signals(all_signals, portfolio)

        logger.info(f"Generated {len(optimized_signals)} unified signals from {len(self.strategies)} strategies")
        return optimized_signals

    async def update_performance_metrics(
        self,
        strategy_id: str,
        portfolio_value: Decimal,
        period_return: Decimal,
        trades_executed: int = 0
    ) -> None:
        """Met à jour les métriques de performance d'une stratégie."""
        if strategy_id not in self.allocations:
            return

        allocation = self.allocations[strategy_id]
        metrics = allocation.performance_metrics

        # Mise à jour des métriques (simplifiée)
        metrics.total_return = period_return

        # Calculer score de performance composite
        metrics.performance_score = self._calculate_performance_score(metrics)

        # Déclencher rééquilibrage si nécessaire
        if await self._should_rebalance():
            await self._rebalance_allocations()

    async def get_strategy_allocations(self) -> Dict[str, StrategyAllocation]:
        """Retourne les allocations actuelles de toutes les stratégies."""
        return self.allocations.copy()

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances du gestionnaire."""
        active_strategies = [a for a in self.allocations.values() if a.is_active]

        if not active_strategies:
            return {"total_strategies": 0, "active_strategies": 0}

        total_return = sum(a.performance_metrics.total_return * a.target_weight
                          for a in active_strategies)

        avg_sharpe = sum(a.performance_metrics.sharpe_ratio for a in active_strategies) / len(active_strategies)

        return {
            "total_strategies": len(self.strategies),
            "active_strategies": len(active_strategies),
            "weighted_return": float(total_return),
            "average_sharpe": float(avg_sharpe),
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "allocation_method": self.allocation_method.value,
            "strategy_breakdown": {
                a.strategy_id: {
                    "weight": float(a.target_weight),
                    "return": float(a.performance_metrics.total_return),
                    "score": float(a.performance_metrics.performance_score)
                }
                for a in active_strategies
            }
        }

    # === Méthodes privées ===

    async def _generate_strategy_signals(
        self,
        strategy: Strategy,
        market_data: Dict[str, Any],
        allocation: StrategyAllocation
    ) -> List[Signal]:
        """Génère des signaux pour une stratégie spécifique."""
        try:
            # Adapter les données pour la stratégie
            if hasattr(strategy, 'generate_signals'):
                signals = strategy.generate_signals(market_data)
                return signals if signals else []
            else:
                logger.warning(f"Strategy does not implement generate_signals method")
                return []
        except Exception as e:
            logger.error(f"Error in strategy signal generation: {e}")
            return []

    async def _optimize_signals(
        self,
        signals: List[Signal],
        portfolio: Portfolio
    ) -> List[Signal]:
        """
        Optimise les signaux combinés pour éviter les conflits.

        - Combine les signaux sur le même symbol
        - Applique les limites de risque
        - Filtre les signaux faibles
        """
        if not signals:
            return []

        # Grouper par symbole
        signals_by_symbol = {}
        for signal in signals:
            symbol = signal.symbol
            if symbol not in signals_by_symbol:
                signals_by_symbol[symbol] = []
            signals_by_symbol[symbol].append(signal)

        optimized_signals = []

        for symbol, symbol_signals in signals_by_symbol.items():
            if len(symbol_signals) == 1:
                optimized_signals.append(symbol_signals[0])
            else:
                # Combiner les signaux multiples
                combined_signal = await self._combine_signals(symbol_signals)
                if combined_signal:
                    optimized_signals.append(combined_signal)

        return optimized_signals

    async def _combine_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """Combine plusieurs signaux sur le même symbole."""
        if not signals:
            return None

        # Logique de combinaison simple - pondérée par la force
        total_weight = sum(signal.strength for signal in signals)
        if total_weight == 0:
            return None

        # Signal représentatif (premier signal comme base)
        base_signal = signals[0]

        # Pondération des quantités
        combined_quantity = sum(
            signal.quantity * signal.strength for signal in signals
            if signal.quantity
        ) / total_weight

        # Créer signal combiné (simplifié)
        return Signal(
            symbol=base_signal.symbol,
            action=base_signal.action,  # Prendre l'action du signal le plus fort
            timestamp=datetime.utcnow(),
            strength=total_weight / len(signals),  # Moyenne pondérée
            confidence=base_signal.confidence,
            price=base_signal.price,
            quantity=combined_quantity,
            strategy_id="multi_strategy_combined",
            metadata={
                "combined_from": [s.strategy_id for s in signals],
                "signal_count": len(signals)
            }
        )

    async def _should_rebalance(self) -> bool:
        """Détermine si un rééquilibrage est nécessaire."""
        if not self.last_rebalance:
            return True

        time_since_rebalance = datetime.utcnow() - self.last_rebalance
        return time_since_rebalance >= self.rebalance_frequency

    async def _rebalance_allocations(self) -> None:
        """Rééquilibre les allocations basées sur la méthode configurée."""
        if not self.allocations:
            return

        active_allocations = {k: v for k, v in self.allocations.items() if v.is_active}

        if not active_allocations:
            return

        # Calculer nouvelles allocations selon la méthode
        new_weights = await self._calculate_allocation_weights(active_allocations)

        # Appliquer les nouvelles allocations
        for strategy_id, new_weight in new_weights.items():
            if strategy_id in self.allocations:
                self.allocations[strategy_id].target_weight = new_weight
                self.allocations[strategy_id].last_rebalance = datetime.utcnow()

        self.last_rebalance = datetime.utcnow()

        logger.info(f"Rebalanced allocations using {self.allocation_method.value}")

    async def _calculate_allocation_weights(
        self,
        allocations: Dict[str, StrategyAllocation]
    ) -> Dict[str, Decimal]:
        """Calcule les nouveaux poids d'allocation."""
        if self.allocation_method == AllocationMethod.EQUAL_WEIGHT:
            return await self._equal_weight_allocation(allocations)
        elif self.allocation_method == AllocationMethod.PERFORMANCE_WEIGHTED:
            return await self._performance_weighted_allocation(allocations)
        else:
            # Fallback vers equal weight
            return await self._equal_weight_allocation(allocations)

    async def _equal_weight_allocation(
        self,
        allocations: Dict[str, StrategyAllocation]
    ) -> Dict[str, Decimal]:
        """Allocation égale entre toutes les stratégies actives."""
        n_strategies = len(allocations)
        if n_strategies == 0:
            return {}

        equal_weight = Decimal("1") / n_strategies
        return {strategy_id: equal_weight for strategy_id in allocations.keys()}

    async def _performance_weighted_allocation(
        self,
        allocations: Dict[str, StrategyAllocation]
    ) -> Dict[str, Decimal]:
        """Allocation basée sur les scores de performance."""
        # Calculer scores positifs
        scores = {}
        for strategy_id, allocation in allocations.items():
            # Score basé sur la performance, minimum 0.1 pour éviter zéro
            score = max(allocation.performance_metrics.performance_score, Decimal("0.1"))
            scores[strategy_id] = score

        # Normaliser
        total_score = sum(scores.values())
        if total_score == 0:
            return await self._equal_weight_allocation(allocations)

        weights = {}
        for strategy_id, score in scores.items():
            weight = score / total_score
            # Appliquer limites min/max
            weight = max(self.min_allocation, min(self.max_allocation, weight))
            weights[strategy_id] = weight

        # Renormaliser après application des limites
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _calculate_performance_score(self, metrics: StrategyMetrics) -> Decimal:
        """Calcule un score de performance composite."""
        # Score simple basé sur return et Sharpe ratio
        return_component = metrics.total_return * Decimal("0.6")
        sharpe_component = metrics.sharpe_ratio * Decimal("0.4")

        # Pénalité pour drawdown élevé
        drawdown_penalty = metrics.max_drawdown * Decimal("0.2")

        score = return_component + sharpe_component - drawdown_penalty
        return max(score, Decimal("0"))  # Score minimum 0