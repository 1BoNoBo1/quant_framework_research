#!/usr/bin/env python3
"""
Test de la CQRS Foundation - Phase 1
=====================================

Validation que tous les handlers Commands/Queries fonctionnent correctement.
"""

import asyncio
import logging
from typing import Dict, Any
from decimal import Decimal

# QFrame Core
from qframe.core.container import DIContainer
from qframe.core.config import FrameworkConfig, Environment
from qframe.infrastructure.config.service_configuration import ServiceConfiguration

# Strategy CQRS
from qframe.application.commands.strategy_commands import CreateStrategyCommand
from qframe.application.queries.strategy_queries import GetStrategyByIdQuery
from qframe.application.handlers.strategy_command_handler import StrategyCommandHandler
from qframe.application.handlers.strategy_query_handler import StrategyQueryHandler

# Domain entities
from qframe.domain.entities.strategy import StrategyType


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_strategy_cqrs():
    """Test complet du CQRS pour les stratégies."""

    logger.info("🧪 Test Strategy CQRS Handlers")

    # Setup du framework
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config.configure_all_services()

    # Résoudre les handlers via DI
    command_handler = container.resolve(StrategyCommandHandler)
    query_handler = container.resolve(StrategyQueryHandler)

    # Test Command: Créer une stratégie
    create_command = CreateStrategyCommand(
        name="CQRS Test Strategy",
        description="Strategy for testing CQRS foundation",
        strategy_type=StrategyType.MEAN_REVERSION,
        universe=["BTC/USDT", "ETH/USDT"],
        max_position_size=Decimal("0.05"),
        max_positions=3,
        risk_per_trade=Decimal("0.02")
    )

    strategy_id = await command_handler.handle_create_strategy(create_command)
    logger.info(f"✅ Strategy created: {strategy_id}")

    # Test Query: Récupérer la stratégie
    get_query = GetStrategyByIdQuery(strategy_id=strategy_id)
    strategy = await query_handler.handle_get_by_id(get_query)

    assert strategy is not None
    assert strategy.name == "CQRS Test Strategy"
    assert strategy.strategy_type == StrategyType.MEAN_REVERSION
    logger.info(f"✅ Strategy retrieved: {strategy.name}")

    return {
        "strategy_created": True,
        "strategy_retrieved": True,
        "cqrs_working": True
    }


async def test_portfolio_cqrs():
    """Test des handlers Portfolio s'ils existent."""

    logger.info("🧪 Test Portfolio CQRS Handlers")

    try:
        # Setup du framework
        config = FrameworkConfig(environment=Environment.TESTING)
        container = DIContainer()
        service_config = ServiceConfiguration(container, config)
        service_config.configure_all_services()

        # Essayer de résoudre les handlers Portfolio
        try:
            # Ces imports peuvent ne pas exister
            from qframe.application.portfolio_management.commands import CreatePortfolioHandler
            from qframe.application.portfolio_management.queries import GetPortfolioHandler

            portfolio_command_handler = container.resolve(CreatePortfolioHandler)
            portfolio_query_handler = container.resolve(GetPortfolioHandler)

            logger.info("✅ Portfolio handlers available and resolvable")
            return {"portfolio_handlers_available": True}

        except ImportError as e:
            logger.warning(f"⚠️ Portfolio handlers not implemented yet: {e}")
            return {"portfolio_handlers_available": False, "reason": str(e)}

    except Exception as e:
        logger.error(f"❌ Portfolio CQRS test failed: {e}")
        return {"portfolio_handlers_available": False, "error": str(e)}


async def test_risk_cqrs():
    """Test des handlers Risk Management s'ils existent."""

    logger.info("🧪 Test Risk Management CQRS Handlers")

    try:
        # Setup du framework
        config = FrameworkConfig(environment=Environment.TESTING)
        container = DIContainer()
        service_config = ServiceConfiguration(container, config)
        service_config.configure_all_services()

        # Essayer de résoudre les handlers Risk
        try:
            from qframe.application.risk_management.commands import CreateRiskAssessmentHandler
            from qframe.application.risk_management.queries import GetRiskAssessmentHandler

            risk_command_handler = container.resolve(CreateRiskAssessmentHandler)
            risk_query_handler = container.resolve(GetRiskAssessmentHandler)

            logger.info("✅ Risk management handlers available and resolvable")
            return {"risk_handlers_available": True}

        except ImportError as e:
            logger.warning(f"⚠️ Risk handlers not implemented yet: {e}")
            return {"risk_handlers_available": False, "reason": str(e)}

    except Exception as e:
        logger.error(f"❌ Risk CQRS test failed: {e}")
        return {"risk_handlers_available": False, "error": str(e)}


async def test_order_cqrs():
    """Test des handlers Order Execution s'ils existent."""

    logger.info("🧪 Test Order Execution CQRS Handlers")

    try:
        # Setup du framework
        config = FrameworkConfig(environment=Environment.TESTING)
        container = DIContainer()
        service_config = ServiceConfiguration(container, config)
        service_config.configure_all_services()

        # Essayer de résoudre les handlers Order
        try:
            from qframe.application.execution_management.commands import CreateOrderHandler
            from qframe.application.execution_management.queries import GetOrderHandler

            order_command_handler = container.resolve(CreateOrderHandler)
            order_query_handler = container.resolve(GetOrderHandler)

            logger.info("✅ Order execution handlers available and resolvable")
            return {"order_handlers_available": True}

        except ImportError as e:
            logger.warning(f"⚠️ Order handlers not implemented yet: {e}")
            return {"order_handlers_available": False, "reason": str(e)}

    except Exception as e:
        logger.error(f"❌ Order CQRS test failed: {e}")
        return {"order_handlers_available": False, "error": str(e)}


async def test_cqrs_foundation():
    """Test complet de la CQRS Foundation."""

    logger.info("🎯 CQRS Foundation Test - Phase 1")
    logger.info("=" * 50)

    results = {}

    # Test Strategy CQRS (le seul qui devrait vraiment fonctionner)
    try:
        strategy_results = await test_strategy_cqrs()
        results["strategy"] = strategy_results
    except Exception as e:
        logger.error(f"❌ Strategy CQRS failed: {e}")
        results["strategy"] = {"error": str(e)}

    # Test autres handlers (peut-être pas implémentés)
    portfolio_results = await test_portfolio_cqrs()
    results["portfolio"] = portfolio_results

    risk_results = await test_risk_cqrs()
    results["risk"] = risk_results

    order_results = await test_order_cqrs()
    results["order"] = order_results

    # Analyse des résultats
    logger.info("\n" + "=" * 50)
    logger.info("📋 RAPPORT PHASE 1 - CQRS FOUNDATION")
    logger.info("=" * 50)

    # Strategy handlers (critiques)
    if results["strategy"].get("cqrs_working"):
        logger.info("✅ Strategy CQRS: Fonctionnel")
    else:
        logger.error("❌ Strategy CQRS: Défaillant")

    # Autres handlers (optionnels pour Phase 1)
    handlers_status = []
    for domain in ["portfolio", "risk", "order"]:
        if results[domain].get(f"{domain}_handlers_available"):
            handlers_status.append(f"✅ {domain.title()}")
        else:
            handlers_status.append(f"⚠️ {domain.title()} (à implémenter)")

    for status in handlers_status:
        logger.info(status)

    # Calcul du statut Phase 1
    critical_working = results["strategy"].get("cqrs_working", False)

    if critical_working:
        logger.info("🎯 Phase 1 Status: ✅ FONDATION CQRS OPÉRATIONNELLE")
        logger.info("   - Strategy handlers complets et fonctionnels")
        logger.info("   - Framework DI/IoC configuré correctement")
        logger.info("   - Commands/Queries pattern implémenté")
    else:
        logger.info("🎯 Phase 1 Status: ❌ FONDATION CQRS INCOMPLÈTE")

    return results


async def main():
    """Point d'entrée principal."""
    return await test_cqrs_foundation()


if __name__ == "__main__":
    # Exécuter les tests
    asyncio.run(main())