#!/usr/bin/env python
"""
Test du MemoryOrderRepository complet
=====================================

Test pour vérifier que toutes les méthodes abstraites sont implémentées.
"""

import asyncio
from decimal import Decimal
from datetime import datetime

from qframe.infrastructure.persistence.memory_order_repository import MemoryOrderRepository
from qframe.domain.entities.order import Order, OrderSide, OrderType, OrderStatus

async def test_complete_repository():
    """Test complet du repository avec toutes les méthodes"""

    print("🔧 Testing Complete MemoryOrderRepository")
    print("=" * 60)

    repo = MemoryOrderRepository()

    # Créer quelques ordres de test
    order1 = Order(
        id="order-001",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        portfolio_id="portfolio-001",
        strategy_id="strategy-001"
    )

    order2 = Order(
        id="order-002",
        symbol="BTC/USD",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.05"),
        price=Decimal("50000"),
        portfolio_id="portfolio-001",
        strategy_id="strategy-002"
    )

    # Test des méthodes de base
    print("✅ Testing basic operations...")
    await repo.save(order1)
    await repo.save(order2)

    found_order = await repo.find_by_id("order-001")
    assert found_order is not None
    print(f"  Found order: {found_order.id}")

    # Test des nouvelles méthodes
    print("✅ Testing new search methods...")

    # Client order ID
    client_order = await repo.find_by_client_order_id(order1.client_order_id)
    assert client_order is not None
    print(f"  Found by client ID: {client_order.id}")

    # Portfolio orders
    portfolio_orders = await repo.find_by_portfolio("portfolio-001")
    assert len(portfolio_orders) == 2
    print(f"  Portfolio orders: {len(portfolio_orders)}")

    # Symbol and side
    btc_buy_orders = await repo.find_by_symbol_and_side("BTC/USD", OrderSide.BUY)
    assert len(btc_buy_orders) == 1
    print(f"  BTC buy orders: {len(btc_buy_orders)}")

    # Orders by type
    market_orders = await repo.find_orders_by_type(OrderType.MARKET)
    limit_orders = await repo.find_orders_by_type(OrderType.LIMIT)
    print(f"  Market orders: {len(market_orders)}, Limit orders: {len(limit_orders)}")

    # Test des statistiques
    print("✅ Testing statistics methods...")

    stats = await repo.get_order_statistics(symbol="BTC/USD")
    print(f"  Order statistics: {stats['total_orders']} orders, {stats['total_volume']} volume")

    exec_stats = await repo.get_execution_statistics()
    print(f"  Execution statistics: {exec_stats['total_executions']} executions")

    count_by_status = await repo.count_by_status()
    print(f"  Orders by status: {count_by_status}")

    count_by_symbol = await repo.count_by_symbol()
    print(f"  Orders by symbol: {count_by_symbol}")

    # Test des méthodes de maintenance
    print("✅ Testing maintenance methods...")

    expired_orders = await repo.find_expired_orders()
    print(f"  Expired orders: {len(expired_orders)}")

    cleaned = await repo.cleanup_expired_orders()
    print(f"  Cleaned expired orders: {cleaned}")

    # Test update
    order1.status = OrderStatus.FILLED
    await repo.update(order1)
    updated_order = await repo.find_by_id("order-001")
    assert updated_order.status == OrderStatus.FILLED
    print(f"  Updated order status: {updated_order.status}")

    print("\n✅ All MemoryOrderRepository tests passed!")
    print("Repository is now 100% complete with all abstract methods implemented.")

if __name__ == "__main__":
    asyncio.run(test_complete_repository())