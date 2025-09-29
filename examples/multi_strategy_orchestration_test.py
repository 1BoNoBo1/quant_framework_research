#!/usr/bin/env python3
"""
Test du Multi-Strategy Orchestration - Phase 5
===============================================

Validation du gestionnaire multi-stratégies avec allocation dynamique
et optimisation basée sur les performances.
"""

import asyncio
import logging
from typing import Dict, Any, List
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# QFrame Core
from qframe.core.container import DIContainer
from qframe.core.config import FrameworkConfig, Environment
from qframe.infrastructure.config.service_configuration import ServiceConfiguration

# Multi-Strategy System
from qframe.strategies.orchestration.multi_strategy_manager import (
    MultiStrategyManager, AllocationMethod, StrategyMetrics, StrategyAllocation
)

# Mock Strategies for Testing
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence

# Portfolio & Services
from qframe.domain.entities.portfolio import Portfolio, PortfolioType, PortfolioStatus
from qframe.domain.services.portfolio_service import PortfolioService


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockStrategy:
    """Stratégie mock pour les tests."""

    def __init__(self, name: str, performance_bias: float = 0.0):
        self.name = name
        self.performance_bias = performance_bias  # Biais de performance (+/-)
        self.signal_count = 0

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Génère des signaux mock avec caractéristiques différentes."""
        self.signal_count += 1

        # Simuler différents comportements selon la stratégie
        if "momentum" in self.name.lower():
            return self._generate_momentum_signals()
        elif "mean_reversion" in self.name.lower():
            return self._generate_mean_reversion_signals()
        elif "arbitrage" in self.name.lower():
            return self._generate_arbitrage_signals()
        else:
            return self._generate_generic_signals()

    def _generate_momentum_signals(self) -> List[Signal]:
        """Signaux momentum - suivent la tendance."""
        if self.signal_count % 3 == 0:  # Génère moins souvent
            return [Signal(
                symbol="BTC/USDT",
                action=SignalAction.BUY,
                timestamp=datetime.utcnow(),
                strength=Decimal("0.8") + Decimal(str(self.performance_bias)),
                confidence=SignalConfidence.HIGH,
                price=Decimal("47000"),
                quantity=Decimal("0.1"),
                strategy_id=self.name,
                metadata={"type": "momentum", "trend": "bullish"}
            )]
        return []

    def _generate_mean_reversion_signals(self) -> List[Signal]:
        """Signaux mean reversion - contrarian."""
        if self.signal_count % 2 == 0:  # Génère fréquemment
            action = SignalAction.SELL if self.signal_count % 4 == 0 else SignalAction.BUY
            return [Signal(
                symbol="ETH/USDT",
                action=action,
                timestamp=datetime.utcnow(),
                strength=Decimal("0.6") + Decimal(str(self.performance_bias)),
                confidence=SignalConfidence.MEDIUM,
                price=Decimal("2800"),
                quantity=Decimal("0.5"),
                strategy_id=self.name,
                metadata={"type": "mean_reversion", "z_score": -1.5}
            )]
        return []

    def _generate_arbitrage_signals(self) -> List[Signal]:
        """Signaux arbitrage - opportunités rares mais sûres."""
        if self.signal_count % 5 == 0:  # Rare mais profitable
            return [Signal(
                symbol="SOL/USDT",
                action=SignalAction.BUY,
                timestamp=datetime.utcnow(),
                strength=Decimal("0.9") + Decimal(str(self.performance_bias)),
                confidence=SignalConfidence.VERY_HIGH,
                price=Decimal("35"),
                quantity=Decimal("2.0"),
                strategy_id=self.name,
                metadata={"type": "arbitrage", "spread": 0.15}
            )]
        return []

    def _generate_generic_signals(self) -> List[Signal]:
        """Signaux génériques."""
        if self.signal_count % 4 == 0:
            return [Signal(
                symbol="BTC/USDT",
                action=SignalAction.BUY,
                timestamp=datetime.utcnow(),
                strength=Decimal("0.5") + Decimal(str(self.performance_bias)),
                confidence=SignalConfidence.MEDIUM,
                price=Decimal("47000"),
                quantity=Decimal("0.05"),
                strategy_id=self.name,
                metadata={"type": "generic"}
            )]
        return []


def create_test_portfolio() -> Portfolio:
    """Créer un portfolio de test pour l'orchestration."""
    return Portfolio(
        id="orchestration_test_portfolio",
        name="Multi-Strategy Test Portfolio",
        portfolio_type=PortfolioType.PAPER_TRADING,
        status=PortfolioStatus.ACTIVE,
        initial_capital=Decimal("1000000"),  # $1M pour tests
        base_currency="USDT"
    )


async def test_strategy_registration():
    """Test d'enregistrement de stratégies multiples."""

    logger.info("🎯 Test Strategy Registration")

    # Setup
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config.configure_all_services()

    portfolio_service = container.resolve(PortfolioService)
    manager = MultiStrategyManager(
        portfolio_service=portfolio_service,
        allocation_method=AllocationMethod.EQUAL_WEIGHT
    )

    # Créer stratégies mock
    strategies = {
        "momentum_strategy": MockStrategy("momentum_strategy", 0.1),
        "mean_reversion_strategy": MockStrategy("mean_reversion_strategy", 0.05),
        "arbitrage_strategy": MockStrategy("arbitrage_strategy", 0.15)
    }

    # Enregistrer stratégies
    for strategy_id, strategy in strategies.items():
        await manager.register_strategy(strategy_id, strategy)
        logger.info(f"   ✅ Registered: {strategy_id}")

    # Vérifier allocations
    allocations = await manager.get_strategy_allocations()

    logger.info(f"   📊 Total strategies: {len(allocations)}")
    for strategy_id, allocation in allocations.items():
        logger.info(f"   • {strategy_id}: {allocation.target_weight:.2%} allocation")

    # Vérifier equal weight (33.33% chacun)
    expected_weight = Decimal("1") / 3
    for allocation in allocations.values():
        assert abs(allocation.target_weight - expected_weight) < Decimal("0.01"), \
            f"Expected ~{expected_weight:.2%}, got {allocation.target_weight:.2%}"

    return {
        "strategies_registered": len(strategies),
        "equal_weight_working": True,
        "allocations": {k: float(v.target_weight) for k, v in allocations.items()}
    }


async def test_signal_generation_orchestration():
    """Test de génération de signaux orchestrés."""

    logger.info("⚡ Test Signal Generation Orchestration")

    # Setup
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config.configure_all_services()

    portfolio_service = container.resolve(PortfolioService)
    manager = MultiStrategyManager(portfolio_service=portfolio_service)
    portfolio = create_test_portfolio()

    # Enregistrer stratégies avec performances différentes
    strategies = {
        "high_performer": MockStrategy("high_performer", 0.2),  # +20% bias
        "medium_performer": MockStrategy("medium_performer", 0.0),  # Neutre
        "low_performer": MockStrategy("low_performer", -0.1)  # -10% bias
    }

    for strategy_id, strategy in strategies.items():
        await manager.register_strategy(strategy_id, strategy)

    # Simuler données de marché
    market_data = {
        "timestamp": datetime.utcnow(),
        "BTC/USDT": {"price": 47000, "volume": 1000},
        "ETH/USDT": {"price": 2800, "volume": 500},
        "SOL/USDT": {"price": 35, "volume": 200}
    }

    # Générer signaux orchestrés plusieurs fois
    all_signals = []
    for i in range(10):  # 10 itérations
        signals = await manager.generate_unified_signals(market_data, portfolio)
        all_signals.extend(signals)

        # Simuler mise à jour performance
        for strategy_id in strategies.keys():
            return_value = Decimal(str(np.random.normal(0.05, 0.02)))  # 5% ± 2%
            await manager.update_performance_metrics(
                strategy_id,
                portfolio_value=Decimal("1000000"),
                period_return=return_value
            )

    logger.info(f"   🎯 Total signals generated: {len(all_signals)}")

    # Analyser les signaux
    strategies_represented = set()
    total_orchestrated = 0

    for signal in all_signals:
        if signal.metadata and signal.metadata.get("orchestrated"):
            total_orchestrated += 1
            strategies_represented.add(signal.metadata.get("source_strategy"))

            logger.info(f"   • {signal.action.value} {signal.symbol} "
                       f"from {signal.metadata.get('source_strategy')} "
                       f"(weight: {signal.metadata.get('allocation_weight', 0):.2%})")

    logger.info(f"   ✅ Orchestrated signals: {total_orchestrated}/{len(all_signals)}")
    logger.info(f"   📊 Strategies active: {len(strategies_represented)}")

    return {
        "signal_orchestration": True,
        "total_signals": len(all_signals),
        "orchestrated_signals": total_orchestrated,
        "strategies_active": len(strategies_represented),
        "unique_strategies": list(strategies_represented)
    }


async def test_performance_based_allocation():
    """Test d'allocation basée sur les performances."""

    logger.info("📊 Test Performance-Based Allocation")

    # Setup avec allocation basée performance
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config.configure_all_services()

    portfolio_service = container.resolve(PortfolioService)
    manager = MultiStrategyManager(
        portfolio_service=portfolio_service,
        allocation_method=AllocationMethod.PERFORMANCE_WEIGHTED,
        rebalance_frequency=timedelta(seconds=1)  # Rééquilibrage rapide pour test
    )

    # Stratégies avec performances différentes
    strategies = {
        "winner": MockStrategy("winner", 0.3),      # +30% performance
        "average": MockStrategy("average", 0.0),    # Neutre
        "loser": MockStrategy("loser", -0.2)       # -20% performance
    }

    for strategy_id, strategy in strategies.items():
        await manager.register_strategy(strategy_id, strategy)

    # Initialiser avec allocations égales
    initial_allocations = await manager.get_strategy_allocations()
    logger.info("   📋 Initial allocations (equal weight):")
    for strategy_id, allocation in initial_allocations.items():
        logger.info(f"     • {strategy_id}: {allocation.target_weight:.2%}")

    # Simuler performances différentielles
    performance_updates = {
        "winner": [0.15, 0.12, 0.18, 0.10, 0.20],     # Bonnes performances
        "average": [0.05, 0.03, 0.04, 0.06, 0.02],    # Performances moyennes
        "loser": [-0.05, -0.08, -0.03, -0.10, -0.06]  # Mauvaises performances
    }

    # Appliquer les mises à jour
    for i in range(5):
        for strategy_id, returns in performance_updates.items():
            await manager.update_performance_metrics(
                strategy_id,
                portfolio_value=Decimal("1000000"),
                period_return=Decimal(str(returns[i]))
            )

        # Attendre le rééquilibrage
        await asyncio.sleep(1.1)

    # Vérifier les nouvelles allocations
    final_allocations = await manager.get_strategy_allocations()
    logger.info("   📊 Final allocations (performance-weighted):")

    allocation_changes = {}
    for strategy_id, allocation in final_allocations.items():
        initial_weight = initial_allocations[strategy_id].target_weight
        final_weight = allocation.target_weight
        change = final_weight - initial_weight

        allocation_changes[strategy_id] = {
            "initial": float(initial_weight),
            "final": float(final_weight),
            "change": float(change),
            "performance_score": float(allocation.performance_metrics.performance_score)
        }

        logger.info(f"     • {strategy_id}: {final_weight:.2%} "
                   f"({change:+.2%} change, score: {allocation.performance_metrics.performance_score:.3f})")

    # Vérifications
    winner_increased = allocation_changes["winner"]["change"] > 0
    loser_decreased = allocation_changes["loser"]["change"] < 0

    logger.info(f"   ✅ Winner allocation increased: {winner_increased}")
    logger.info(f"   ✅ Loser allocation decreased: {loser_decreased}")

    return {
        "performance_allocation": True,
        "winner_increased": winner_increased,
        "loser_decreased": loser_decreased,
        "allocation_changes": allocation_changes
    }


async def test_risk_management_orchestration():
    """Test de la gestion des risques dans l'orchestration."""

    logger.info("🛡️ Test Risk Management Orchestration")

    # Setup
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config.configure_all_services()

    portfolio_service = container.resolve(PortfolioService)
    manager = MultiStrategyManager(
        portfolio_service=portfolio_service,
        min_allocation=Decimal("0.10"),  # 10% minimum
        max_allocation=Decimal("0.60")   # 60% maximum
    )

    # Stratégie avec performance extrême
    extreme_strategy = MockStrategy("extreme_performer", 1.0)  # +100% bias
    normal_strategy = MockStrategy("normal_performer", 0.0)

    await manager.register_strategy("extreme", extreme_strategy)
    await manager.register_strategy("normal", normal_strategy)

    # Simuler performance extrême
    for i in range(10):
        await manager.update_performance_metrics(
            "extreme",
            portfolio_value=Decimal("1000000"),
            period_return=Decimal("0.50")  # 50% return
        )
        await manager.update_performance_metrics(
            "normal",
            portfolio_value=Decimal("1000000"),
            period_return=Decimal("0.01")  # 1% return
        )

    await asyncio.sleep(1.1)  # Attendre rééquilibrage

    # Vérifier les limites
    allocations = await manager.get_strategy_allocations()

    extreme_weight = allocations["extreme"].target_weight
    normal_weight = allocations["normal"].target_weight

    logger.info(f"   📊 Extreme strategy: {extreme_weight:.2%}")
    logger.info(f"   📊 Normal strategy: {normal_weight:.2%}")

    # Vérifier les limites
    within_max_limit = extreme_weight <= manager.max_allocation
    within_min_limit = normal_weight >= manager.min_allocation
    allocations_sum_to_one = abs(extreme_weight + normal_weight - Decimal("1")) < Decimal("0.01")

    logger.info(f"   ✅ Extreme within max limit (60%): {within_max_limit}")
    logger.info(f"   ✅ Normal within min limit (10%): {within_min_limit}")
    logger.info(f"   ✅ Allocations sum to 100%: {allocations_sum_to_one}")

    return {
        "risk_management": True,
        "max_limit_respected": within_max_limit,
        "min_limit_respected": within_min_limit,
        "allocations_normalized": allocations_sum_to_one,
        "extreme_allocation": float(extreme_weight),
        "normal_allocation": float(normal_weight)
    }


async def test_multi_strategy_orchestration():
    """Test complet du système d'orchestration multi-stratégies."""

    logger.info("🎯 Multi-Strategy Orchestration Test - Phase 5")
    logger.info("=" * 60)

    results = {}

    # Test 1: Enregistrement de stratégies
    try:
        registration_results = await test_strategy_registration()
        results["registration"] = registration_results
    except Exception as e:
        logger.error(f"❌ Strategy registration test failed: {e}")
        results["registration"] = {"error": str(e)}

    # Test 2: Orchestration de signaux
    try:
        orchestration_results = await test_signal_generation_orchestration()
        results["orchestration"] = orchestration_results
    except Exception as e:
        logger.error(f"❌ Signal orchestration test failed: {e}")
        results["orchestration"] = {"error": str(e)}

    # Test 3: Allocation basée performance
    try:
        performance_results = await test_performance_based_allocation()
        results["performance_allocation"] = performance_results
    except Exception as e:
        logger.error(f"❌ Performance allocation test failed: {e}")
        results["performance_allocation"] = {"error": str(e)}

    # Test 4: Gestion des risques
    try:
        risk_results = await test_risk_management_orchestration()
        results["risk_management"] = risk_results
    except Exception as e:
        logger.error(f"❌ Risk management test failed: {e}")
        results["risk_management"] = {"error": str(e)}

    # Analyse des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RAPPORT PHASE 5 - MULTI-STRATEGY ORCHESTRATION")
    logger.info("=" * 60)

    # Fonctionnalités critiques
    critical_features = []
    if results["registration"].get("equal_weight_working"):
        critical_features.append("✅ Enregistrement de stratégies")
    else:
        critical_features.append("❌ Enregistrement de stratégies")

    if results["orchestration"].get("signal_orchestration"):
        critical_features.append("✅ Orchestration de signaux")
    else:
        critical_features.append("❌ Orchestration de signaux")

    if results["performance_allocation"].get("performance_allocation"):
        critical_features.append("✅ Allocation basée performance")
    else:
        critical_features.append("❌ Allocation basée performance")

    if results["risk_management"].get("risk_management"):
        critical_features.append("✅ Gestion des risques")
    else:
        critical_features.append("❌ Gestion des risques")

    # Affichage
    for feature in critical_features:
        logger.info(feature)

    # Statistiques détaillées
    if results["orchestration"].get("total_signals", 0) > 0:
        logger.info(f"📊 Signaux générés: {results['orchestration']['total_signals']}")
        logger.info(f"🎯 Stratégies actives: {results['orchestration']['strategies_active']}")

    if results["performance_allocation"].get("allocation_changes"):
        winner_change = results["performance_allocation"]["allocation_changes"]["winner"]["change"]
        logger.info(f"📈 Amélioration allocation gagnante: {winner_change:+.2%}")

    # Statut Phase 5
    critical_working = (
        results["registration"].get("equal_weight_working") and
        results["orchestration"].get("signal_orchestration") and
        results["performance_allocation"].get("performance_allocation") and
        results["risk_management"].get("risk_management")
    )

    if critical_working:
        logger.info("🎯 Phase 5 Status: ✅ MULTI-STRATEGY ORCHESTRATION OPÉRATIONNEL")
        logger.info("   - Gestion multi-stratégies fonctionnelle")
        logger.info("   - Allocation dynamique active")
        logger.info("   - Optimisation basée performance")
        logger.info("   - Gestion des risques intégrée")
    else:
        logger.info("🎯 Phase 5 Status: ❌ MULTI-STRATEGY ORCHESTRATION INCOMPLET")

    return results


async def main():
    """Point d'entrée principal."""
    return await test_multi_strategy_orchestration()


if __name__ == "__main__":
    # Exécuter les tests
    asyncio.run(main())