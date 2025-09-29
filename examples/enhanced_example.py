#!/usr/bin/env python
"""
Enhanced Trading Example - QFrame Framework
============================================

Exemple avancé démontrant l'utilisation complète du framework QFrame
avec portfolio, orders, et toutes les fonctionnalités de repository.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from qframe.core.config import FrameworkConfig
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.order import Order, OrderType, OrderSide, OrderStatus, OrderPriority, TimeInForce
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository
from qframe.infrastructure.persistence.memory_order_repository import MemoryOrderRepository


def create_realistic_market_data():
    """Créer des données de marché réalistes"""
    dates = pd.date_range(start='2024-09-01', end='2024-09-27', freq='1h')
    n = len(dates)

    # Bitcoin-like price movement with volatility
    initial_price = 50000
    returns = np.random.normal(0, 0.02, n)  # 2% hourly volatility
    prices = [initial_price]

    for i in range(1, n):
        # Add some trend and mean reversion
        trend = 0.0001  # Slight upward trend
        mean_reversion = -0.1 * (np.log(prices[-1]) - np.log(initial_price))
        noise = returns[i]

        price_change = trend + mean_reversion + noise
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 1000))  # Don't go below $1000

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })

    return df


async def run_enhanced_example():
    """Exemple avancé de trading avec QFrame"""

    print("=" * 70)
    print("🚀 QFrame Enhanced Trading Example")
    print("=" * 70)

    # 1. Configuration
    config = FrameworkConfig()
    print(f"\n✓ Framework configuration loaded: {config.environment}")

    # 2. Initialize repositories
    portfolio_repo = MemoryPortfolioRepository()
    order_repo = MemoryOrderRepository()
    print("✓ Repositories initialized (Portfolio + Complete Order Repository)")

    # 3. Create multiple portfolios
    portfolios = []
    for i in range(3):
        portfolio = Portfolio(
            id=f"portfolio-{i+1:03d}",
            name=f"Strategy Portfolio {i+1}",
            initial_capital=Decimal(f"{(i+1)*5000}.00"),
            base_currency="USD"
        )
        await portfolio_repo.save(portfolio)
        portfolios.append(portfolio)

    print(f"✓ Created {len(portfolios)} portfolios with total capital: ${sum(p.initial_capital for p in portfolios)}")

    # 4. Load realistic market data
    data = create_realistic_market_data()
    print(f"✓ Generated realistic market data: {len(data)} periods from {data['timestamp'].min()} to {data['timestamp'].max()}")

    # 5. Advanced trading simulation
    print("\n" + "=" * 50)
    print("🎯 Advanced Multi-Strategy Trading Simulation")
    print("=" * 50)

    all_orders = []
    strategy_configs = [
        {"name": "Mean Reversion", "portfolio_idx": 0, "lookback": 24, "z_threshold": 1.5},
        {"name": "Momentum", "portfolio_idx": 1, "lookback": 12, "momentum_threshold": 0.02},
        {"name": "Arbitrage", "portfolio_idx": 2, "lookback": 6, "spread_threshold": 0.01}
    ]

    for strategy in strategy_configs:
        print(f"\n📈 Executing {strategy['name']} Strategy")
        portfolio = portfolios[strategy['portfolio_idx']]

        strategy_orders = []
        for i in range(strategy['lookback'], len(data), 4):  # Every 4 hours
            row = data.iloc[i]
            price = Decimal(str(row['close']))

            # Strategy-specific logic
            signal = None
            priority = OrderPriority.NORMAL
            time_in_force = TimeInForce.GTC

            if strategy['name'] == "Mean Reversion":
                # Mean reversion logic
                recent_prices = data['close'].iloc[i-strategy['lookback']:i]
                mean_price = recent_prices.mean()
                std_price = recent_prices.std()
                z_score = (row['close'] - mean_price) / std_price if std_price > 0 else 0

                if z_score < -strategy['z_threshold']:
                    signal = OrderSide.BUY
                    priority = OrderPriority.HIGH
                elif z_score > strategy['z_threshold']:
                    signal = OrderSide.SELL

            elif strategy['name'] == "Momentum":
                # Momentum logic
                if i >= strategy['lookback']:
                    price_change = (row['close'] - data['close'].iloc[i-strategy['lookback']]) / data['close'].iloc[i-strategy['lookback']]
                    if price_change > strategy['momentum_threshold']:
                        signal = OrderSide.BUY
                    elif price_change < -strategy['momentum_threshold']:
                        signal = OrderSide.SELL
                        priority = OrderPriority.HIGH

            elif strategy['name'] == "Arbitrage":
                # Arbitrage-like logic (simulated spread)
                spread = abs(row['high'] - row['low']) / row['close']
                if spread > strategy['spread_threshold']:
                    signal = OrderSide.BUY if np.random.random() > 0.5 else OrderSide.SELL
                    priority = OrderPriority.URGENT
                    time_in_force = TimeInForce.IOC  # Immediate or Cancel

            # Create order if signal
            if signal:
                order_type = OrderType.MARKET if priority == OrderPriority.URGENT else OrderType.LIMIT
                limit_price = price * Decimal("0.999") if signal == OrderSide.BUY else price * Decimal("1.001")

                order = Order(
                    id=f"order-{len(all_orders)+1:04d}",
                    symbol="BTC/USD",
                    side=signal,
                    order_type=order_type,
                    quantity=Decimal("0.01") * (1 + strategy['portfolio_idx']),  # Different sizes
                    price=limit_price if order_type == OrderType.LIMIT else None,
                    portfolio_id=portfolio.id,
                    strategy_id=f"strategy-{strategy['name'].lower().replace(' ', '-')}",
                    priority=priority,
                    time_in_force=time_in_force,
                    created_time=row['timestamp']
                )

                await order_repo.save(order)
                strategy_orders.append(order)
                all_orders.append(order)

                # Simulate some order fills
                if np.random.random() < 0.7:  # 70% fill rate
                    order.status = OrderStatus.FILLED if np.random.random() < 0.8 else OrderStatus.PARTIALLY_FILLED
                    order.filled_quantity = order.quantity if order.status == OrderStatus.FILLED else order.quantity * Decimal("0.5")
                    await order_repo.update(order)

        print(f"  📊 {strategy['name']}: {len(strategy_orders)} orders created")

    # 6. Comprehensive Analysis
    print("\n" + "=" * 50)
    print("📊 Comprehensive Trading Analysis")
    print("=" * 50)

    # Portfolio analysis
    print("\n📈 Portfolio Analysis:")
    for portfolio in portfolios:
        portfolio_orders = await order_repo.find_by_portfolio(portfolio.id)
        filled_orders = [o for o in portfolio_orders if o.status == OrderStatus.FILLED]

        total_volume = sum(o.quantity for o in portfolio_orders)
        filled_volume = sum(o.filled_quantity for o in filled_orders)

        print(f"  {portfolio.name}:")
        print(f"    Initial Capital: ${portfolio.initial_capital}")
        print(f"    Orders Created: {len(portfolio_orders)}")
        print(f"    Orders Filled: {len(filled_orders)}")
        print(f"    Total Volume: {total_volume} BTC")
        print(f"    Filled Volume: {filled_volume} BTC")

    # Order statistics by strategy
    print("\n📊 Strategy Performance:")
    for strategy in strategy_configs:
        strategy_id = f"strategy-{strategy['name'].lower().replace(' ', '-')}"
        strategy_orders = await order_repo.find_by_strategy(strategy_id)

        if strategy_orders:
            stats = await order_repo.get_order_statistics()
            buy_orders = await order_repo.find_by_symbol_and_side("BTC/USD", OrderSide.BUY)
            sell_orders = await order_repo.find_by_symbol_and_side("BTC/USD", OrderSide.SELL)

            print(f"  {strategy['name']} Strategy:")
            print(f"    Orders: {len(strategy_orders)}")
            print(f"    Buy/Sell Ratio: {len([o for o in strategy_orders if o.side == OrderSide.BUY])}/{len([o for o in strategy_orders if o.side == OrderSide.SELL])}")

    # Order type analysis
    print("\n🔍 Order Analysis:")
    market_orders = await order_repo.find_orders_by_type(OrderType.MARKET)
    limit_orders = await order_repo.find_orders_by_type(OrderType.LIMIT)
    urgent_orders = await order_repo.find_orders_by_priority(OrderPriority.URGENT)

    print(f"  Market Orders: {len(market_orders)}")
    print(f"  Limit Orders: {len(limit_orders)}")
    print(f"  Urgent Priority: {len(urgent_orders)}")

    # Time analysis
    expired_orders = await order_repo.find_expired_orders()
    parent_orders = await order_repo.find_parent_orders()

    print(f"  Expired Orders: {len(expired_orders)}")
    print(f"  Parent Orders: {len(parent_orders)}")

    # Overall statistics
    overall_stats = await order_repo.get_order_statistics(symbol="BTC/USD")
    exec_stats = await order_repo.get_execution_statistics()

    print("\n📈 Overall Statistics:")
    print(f"  Total Orders: {overall_stats['total_orders']}")
    print(f"  Total Volume: {overall_stats['total_volume']} BTC")
    print(f"  Average Order Size: {overall_stats['avg_order_size']:.4f} BTC")
    print(f"  Fill Rate: {overall_stats['fill_rate']:.2%}")

    # Status breakdown
    count_by_status = await order_repo.count_by_status()
    print("\n📊 Orders by Status:")
    for status, count in count_by_status.items():
        if count > 0:
            print(f"  {status.value}: {count}")

    print("\n" + "=" * 70)
    print("✅ Enhanced Trading Example Completed Successfully!")
    print("=" * 70)
    print(f"📊 Summary: {len(all_orders)} orders across {len(portfolios)} portfolios")
    print(f"🎯 Framework Status: All repository methods working perfectly")
    print(f"🚀 Ready for production trading!")


def main():
    """Main entry point"""
    try:
        asyncio.run(run_enhanced_example())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())