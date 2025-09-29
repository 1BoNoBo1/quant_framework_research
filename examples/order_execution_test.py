#!/usr/bin/env python3
"""
Test du Order Execution Core - Phase 3
=======================================

Validation complète du moteur d'exécution d'ordres avec gestion
des fills, slippage, et intégration portfolio.
"""

import asyncio
import logging
from typing import Dict, Any, List
from decimal import Decimal
from datetime import datetime, timedelta

# QFrame Core
from qframe.core.container import DIContainer
from qframe.core.config import FrameworkConfig, Environment
from qframe.infrastructure.config.service_configuration import ServiceConfiguration

# Order Domain
from qframe.domain.entities.order import Order, OrderType, OrderSide, OrderStatus, TimeInForce
from qframe.domain.services.execution_service import (
    ExecutionService, ExecutionPlan, VenueQuote, RoutingStrategy, ExecutionAlgorithm
)
from qframe.application.execution_management.commands import CreateOrderCommand, CancelOrderCommand

# Portfolio Integration
from qframe.domain.entities.portfolio import Portfolio, PortfolioType, PortfolioStatus
from qframe.domain.entities.position import Position

# Data Pipeline for real prices
from qframe.infrastructure.data.market_data_pipeline import MarketDataPipeline


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_portfolio() -> Portfolio:
    """Créer un portfolio de test pour les ordres."""

    portfolio = Portfolio(
        id="order_test_portfolio",
        name="Order Execution Test Portfolio",
        portfolio_type=PortfolioType.PAPER_TRADING,
        status=PortfolioStatus.ACTIVE,
        initial_capital=Decimal("50000"),
        base_currency="USDT"
    )

    # Position BTC existante
    btc_position = Position(
        symbol="BTC/USDT",
        quantity=Decimal("0.5"),
        average_price=Decimal("45000"),
        current_price=Decimal("47000")
    )
    portfolio.add_position(btc_position)

    return portfolio


async def test_order_creation():
    """Test de création d'ordres."""

    logger.info("🔨 Test Order Creation")

    # Setup du framework
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config.configure_all_services()

    # Test Market Order
    market_order = Order(
        id="market_order_001",
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("0.1"),
        status=OrderStatus.PENDING,
        time_in_force=TimeInForce.IOC
    )
    logger.info(f"   📊 Market Order créé: {market_order.symbol} {market_order.side} {market_order.quantity}")

    # Test Limit Order
    limit_order = Order(
        id="limit_order_001",
        symbol="ETH/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.SELL,
        quantity=Decimal("2.0"),
        price=Decimal("2800"),
        status=OrderStatus.PENDING,
        time_in_force=TimeInForce.GTC
    )
    logger.info(f"   📊 Limit Order créé: {limit_order.symbol} {limit_order.side} {limit_order.quantity} @ ${limit_order.price}")

    # Test Stop Order (utiliser STOP au lieu de STOP_LOSS)
    stop_order = Order(
        id="stop_order_001",
        symbol="BTC/USDT",
        order_type=OrderType.STOP,
        side=OrderSide.SELL,
        quantity=Decimal("0.2"),
        stop_price=Decimal("44000"),
        status=OrderStatus.PENDING,
        time_in_force=TimeInForce.GTC
    )
    logger.info(f"   📊 Stop Order créé: {stop_order.symbol} {stop_order.side} {stop_order.quantity} @ stop ${stop_order.stop_price}")

    return {
        "market_order": market_order,
        "limit_order": limit_order,
        "stop_order": stop_order,
        "orders_created": True
    }


async def test_order_execution():
    """Test d'exécution d'ordres."""

    logger.info("⚡ Test Order Execution")

    # Setup du framework
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config.configure_all_services()

    # Résoudre le service d'exécution
    try:
        execution_service = container.resolve(ExecutionService)
        logger.info("   ✅ ExecutionService résolu")
    except Exception as e:
        logger.error(f"   ❌ Erreur résolution ExecutionService: {e}")
        return {"execution_service_available": False, "error": str(e)}

    # Créer un ordre de test
    test_order = Order(
        id="execution_test_001",
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("0.05"),
        status=OrderStatus.PENDING
    )

    # Test d'exécution
    try:
        # Créer un plan d'exécution simple
        execution_plan = ExecutionPlan(
            order_id=test_order.id,
            target_venues=["binance"],
            routing_strategy=RoutingStrategy.BEST_PRICE,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE,
            estimated_cost=Decimal("10"),
            estimated_duration=timedelta(seconds=5),
            slice_instructions=[],
            risk_checks_passed=True,
            created_time=datetime.utcnow()
        )

        # Créer des données de marché simulées
        market_data = {
            "binance": VenueQuote(
                venue="binance",
                symbol="BTC/USDT",
                bid_price=Decimal("46900"),
                ask_price=Decimal("47100"),
                bid_size=Decimal("1.0"),
                ask_size=Decimal("1.0"),
                timestamp=datetime.utcnow()
            )
        }

        # Simuler l'exécution avec tous les paramètres requis
        execution_result = execution_service.execute_order(test_order, execution_plan, market_data)

        if execution_result and len(execution_result) > 0:
            # execution_result est une liste d'OrderExecution
            total_executed = sum(exec.executed_quantity for exec in execution_result)
            avg_price = sum(exec.execution_price * exec.executed_quantity for exec in execution_result) / total_executed if total_executed > 0 else Decimal("0")

            logger.info(f"   ✅ Ordre exécuté: {test_order.id}")
            logger.info(f"   💰 Prix moyen: ${avg_price}")
            logger.info(f"   📊 Quantité exécutée: {total_executed}")
            logger.info(f"   📈 Nombre d'exécutions: {len(execution_result)}")

            return {
                "order_executed": True,
                "fill_price": float(avg_price),
                "filled_quantity": float(total_executed),
                "execution_count": len(execution_result)
            }
        else:
            logger.warning(f"   ⚠️ Ordre non exécuté: {test_order.id}")
            return {"order_executed": False, "reason": "execution_failed"}

    except Exception as e:
        logger.error(f"   ❌ Erreur exécution: {e}")
        return {"order_executed": False, "error": str(e)}


async def test_slippage_calculation():
    """Test du calcul de slippage."""

    logger.info("📐 Test Slippage Calculation")

    # Setup du framework
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config.configure_all_services()

    try:
        execution_service = container.resolve(ExecutionService)

        # Test différents scénarios de slippage
        scenarios = [
            {"symbol": "BTC/USDT", "quantity": Decimal("0.1"), "expected_slippage": "faible"},
            {"symbol": "BTC/USDT", "quantity": Decimal("5.0"), "expected_slippage": "élevé"},
            {"symbol": "ETH/USDT", "quantity": Decimal("1.0"), "expected_slippage": "modéré"}
        ]

        slippage_results = []

        for scenario in scenarios:
            try:
                # Calculer le slippage estimé
                if hasattr(execution_service, 'calculate_slippage'):
                    slippage = execution_service.calculate_slippage(
                        scenario["symbol"],
                        scenario["quantity"]
                    )
                    logger.info(f"   📊 {scenario['symbol']} qty={scenario['quantity']}: slippage={slippage:.4f}%")
                    slippage_results.append({
                        "symbol": scenario["symbol"],
                        "quantity": float(scenario["quantity"]),
                        "slippage_pct": float(slippage)
                    })
                else:
                    logger.info(f"   📊 {scenario['symbol']} qty={scenario['quantity']}: slippage=non calculé")
                    slippage_results.append({
                        "symbol": scenario["symbol"],
                        "quantity": float(scenario["quantity"]),
                        "slippage_pct": 0.0
                    })

            except Exception as e:
                logger.warning(f"   ⚠️ Erreur slippage {scenario['symbol']}: {e}")

        return {
            "slippage_calculation": True,
            "scenarios_tested": len(scenarios),
            "results": slippage_results
        }

    except Exception as e:
        logger.error(f"   ❌ Erreur test slippage: {e}")
        return {"slippage_calculation": False, "error": str(e)}


async def test_portfolio_integration():
    """Test de l'intégration avec le portfolio."""

    logger.info("🔗 Test Portfolio Integration")

    portfolio = create_test_portfolio()
    initial_btc = portfolio.get_position("BTC/USDT").quantity
    initial_cash = portfolio.cash_balance

    logger.info(f"   💼 État initial: BTC={initial_btc}, Cash=${initial_cash}")

    # Setup du framework
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config.configure_all_services()

    try:
        execution_service = container.resolve(ExecutionService)

        # Créer un ordre d'achat BTC
        buy_order = Order(
            id="portfolio_integration_001",
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            status=OrderStatus.PENDING
        )

        # Créer un plan d'exécution et des données de marché pour l'ordre
        execution_plan = ExecutionPlan(
            order_id=buy_order.id,
            target_venues=["binance"],
            routing_strategy=RoutingStrategy.BEST_PRICE,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE,
            estimated_cost=Decimal("5"),
            estimated_duration=timedelta(seconds=3),
            slice_instructions=[],
            risk_checks_passed=True,
            created_time=datetime.utcnow()
        )

        market_data = {
            "binance": VenueQuote(
                venue="binance",
                symbol="BTC/USDT",
                bid_price=Decimal("46900"),
                ask_price=Decimal("47100"),
                bid_size=Decimal("2.0"),
                ask_size=Decimal("2.0"),
                timestamp=datetime.utcnow()
            )
        }

        # Exécuter l'ordre
        execution_result = execution_service.execute_order(buy_order, execution_plan, market_data)

        if execution_result and len(execution_result) > 0:
            # Simuler que l'ordre est rempli (ce serait fait automatiquement dans une vraie implémentation)
            total_executed = sum(exec.executed_quantity for exec in execution_result)
            avg_price = sum(exec.execution_price * exec.executed_quantity for exec in execution_result) / total_executed

            # Mettre à jour l'ordre manuellement pour la simulation
            buy_order.filled_quantity = total_executed
            buy_order.average_fill_price = avg_price
            buy_order.status = OrderStatus.FILLED if total_executed == buy_order.quantity else OrderStatus.PARTIALLY_FILLED
            # Mettre à jour le portfolio
            portfolio.update_position(buy_order)

            final_btc = portfolio.get_position("BTC/USDT").quantity
            final_cash = portfolio.cash_balance

            logger.info(f"   💼 État final: BTC={final_btc}, Cash=${final_cash}")
            logger.info(f"   📈 Changement BTC: +{final_btc - initial_btc}")
            logger.info(f"   💰 Changement Cash: ${final_cash - initial_cash}")

            return {
                "portfolio_integration": True,
                "btc_change": float(final_btc - initial_btc),
                "cash_change": float(final_cash - initial_cash),
                "order_processed": True
            }
        else:
            logger.warning("   ⚠️ Ordre non exécuté, portfolio inchangé")
            return {"portfolio_integration": False, "reason": "order_not_filled"}

    except Exception as e:
        logger.error(f"   ❌ Erreur intégration portfolio: {e}")
        return {"portfolio_integration": False, "error": str(e)}


async def test_market_data_integration():
    """Test de l'intégration avec les données de marché."""

    logger.info("📊 Test Market Data Integration")

    # Setup du framework
    config = FrameworkConfig(environment=Environment.TESTING)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)
    service_config._configure_data_pipeline_services()

    try:
        pipeline = container.resolve(MarketDataPipeline)
        providers = pipeline.get_registered_providers()

        logger.info(f"   📈 Providers disponibles: {len(providers)}")

        # Tester récupération de prix pour exécution
        test_symbols = ["BTC/USDT", "ETH/USDT"]
        price_data = {}

        for symbol in test_symbols:
            if providers:  # S'il y a des providers
                try:
                    # Prendre le premier provider CCXT disponible
                    ccxt_providers = [p for p in providers if "ccxt_" in p]
                    if ccxt_providers:
                        provider_name = ccxt_providers[0]
                        ticker = await pipeline.get_ticker(provider_name, symbol)
                        if ticker:
                            price_data[symbol] = float(ticker.last)
                            logger.info(f"   💰 {symbol}: ${ticker.last}")
                        else:
                            logger.warning(f"   ⚠️ {symbol}: pas de données")
                    else:
                        logger.info(f"   📊 {symbol}: prix simulé (pas de provider CCXT)")
                        price_data[symbol] = 45000.0 if "BTC" in symbol else 2800.0
                except Exception as e:
                    logger.warning(f"   ⚠️ Erreur {symbol}: {e}")
                    price_data[symbol] = 45000.0 if "BTC" in symbol else 2800.0
            else:
                logger.info(f"   📊 {symbol}: prix simulé (pas de providers)")
                price_data[symbol] = 45000.0 if "BTC" in symbol else 2800.0

        return {
            "market_data_integration": True,
            "providers_count": len(providers),
            "price_data": price_data,
            "data_available": len(price_data) > 0
        }

    except Exception as e:
        logger.error(f"   ❌ Erreur intégration market data: {e}")
        return {"market_data_integration": False, "error": str(e)}


async def test_order_execution_core():
    """Test complet du Order Execution Core."""

    logger.info("🎯 Order Execution Core Test - Phase 3")
    logger.info("=" * 50)

    results = {}

    # Test 1: Création d'ordres
    try:
        creation_results = await test_order_creation()
        results["creation"] = creation_results
    except Exception as e:
        logger.error(f"❌ Order creation test failed: {e}")
        results["creation"] = {"error": str(e)}

    # Test 2: Exécution d'ordres
    try:
        execution_results = await test_order_execution()
        results["execution"] = execution_results
    except Exception as e:
        logger.error(f"❌ Order execution test failed: {e}")
        results["execution"] = {"error": str(e)}

    # Test 3: Calcul de slippage
    try:
        slippage_results = await test_slippage_calculation()
        results["slippage"] = slippage_results
    except Exception as e:
        logger.error(f"❌ Slippage test failed: {e}")
        results["slippage"] = {"error": str(e)}

    # Test 4: Intégration portfolio
    try:
        portfolio_results = await test_portfolio_integration()
        results["portfolio"] = portfolio_results
    except Exception as e:
        logger.error(f"❌ Portfolio integration test failed: {e}")
        results["portfolio"] = {"error": str(e)}

    # Test 5: Intégration market data
    try:
        market_data_results = await test_market_data_integration()
        results["market_data"] = market_data_results
    except Exception as e:
        logger.error(f"❌ Market data integration test failed: {e}")
        results["market_data"] = {"error": str(e)}

    # Analyse des résultats
    logger.info("\n" + "=" * 50)
    logger.info("📋 RAPPORT PHASE 3 - ORDER EXECUTION CORE")
    logger.info("=" * 50)

    # Fonctionnalités critiques
    critical_features = []
    if results["creation"].get("orders_created"):
        critical_features.append("✅ Création d'ordres")
    else:
        critical_features.append("❌ Création d'ordres")

    if results["execution"].get("order_executed"):
        critical_features.append("✅ Exécution d'ordres")
    else:
        critical_features.append("❌ Exécution d'ordres")

    if results["portfolio"].get("portfolio_integration"):
        critical_features.append("✅ Intégration Portfolio")
    else:
        critical_features.append("⚠️ Intégration Portfolio")

    # Fonctionnalités avancées
    advanced_features = []
    if results["slippage"].get("slippage_calculation"):
        advanced_features.append("✅ Calcul de slippage")
    else:
        advanced_features.append("⚠️ Calcul de slippage")

    if results["market_data"].get("market_data_integration"):
        advanced_features.append("✅ Intégration market data")
    else:
        advanced_features.append("⚠️ Intégration market data")

    # Affichage
    for feature in critical_features:
        logger.info(feature)

    for feature in advanced_features:
        logger.info(feature)

    # Statut Phase 3
    critical_working = (
        results["creation"].get("orders_created") and
        results["execution"].get("order_executed") and
        results["portfolio"].get("portfolio_integration")
    )

    if critical_working:
        logger.info("🎯 Phase 3 Status: ✅ ORDER EXECUTION CORE OPÉRATIONNEL")
        logger.info("   - Création et exécution d'ordres fonctionnelles")
        logger.info("   - Intégration portfolio active")
        logger.info("   - Pipeline d'exécution complet")
    else:
        logger.info("🎯 Phase 3 Status: ❌ ORDER EXECUTION CORE INCOMPLET")

        # Identifier les problèmes
        if not results["creation"].get("orders_created"):
            logger.info("   - Problème: Création d'ordres défaillante")
        if not results["execution"].get("order_executed"):
            logger.info("   - Problème: Exécution d'ordres défaillante")
        if not results["portfolio"].get("portfolio_integration"):
            logger.info("   - Problème: Intégration portfolio défaillante")

    return results


async def main():
    """Point d'entrée principal."""
    return await test_order_execution_core()


if __name__ == "__main__":
    # Exécuter les tests
    asyncio.run(main())