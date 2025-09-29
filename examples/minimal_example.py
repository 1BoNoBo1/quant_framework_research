#!/usr/bin/env python
"""
Minimal Working Example - QFrame Framework
===========================================

Exemple minimal démontrant l'utilisation basique du framework QFrame.
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
from qframe.core.container import get_container
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.order import Order, OrderType, OrderSide
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository


def create_sample_data():
    """Créer des données de test"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
    n = len(dates)

    # Générer des prix avec tendance et bruit
    trend = np.linspace(50000, 52000, n)
    noise = np.random.normal(0, 500, n)
    prices = trend + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.uniform(-100, 100, n),
        'high': prices + np.random.uniform(0, 200, n),
        'low': prices - np.random.uniform(0, 200, n),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })

    return df


async def run_minimal_example():
    """Exemple minimal de trading"""

    print("=" * 60)
    print("QFrame Minimal Example")
    print("=" * 60)

    # 1. Configuration
    config = FrameworkConfig()
    print(f"\n✓ Configuration loaded: {config.environment}")

    # 2. Initialize repositories
    portfolio_repo = MemoryPortfolioRepository()
    print("✓ Portfolio repository initialized")

    # 3. Create portfolio
    portfolio = Portfolio(
        id="portfolio-001",
        name="Test Portfolio",
        initial_capital=Decimal("10000.00"),
        base_currency="USD"
    )

    await portfolio_repo.save(portfolio)
    print(f"✓ Portfolio created: ${portfolio.initial_capital}")

    # 4. Load sample data
    data = create_sample_data()
    print(f"✓ Sample data loaded: {len(data)} periods")

    # 5. Simple trading logic
    print("\n" + "=" * 40)
    print("Executing Simple Trading Strategy")
    print("=" * 40)

    orders_created = []

    for i in range(5, len(data), 10):  # Every 10 periods
        row = data.iloc[i]
        price = Decimal(str(row['close']))

        # Simple logic: buy if price is below moving average
        ma = data['close'].iloc[i-5:i].mean()

        if row['close'] < ma:
            # Buy signal
            order = Order(
                id=f"order-{i:03d}",
                portfolio_id=portfolio.id,
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.01"),
                price=price,
                created_time=row['timestamp']
            )

            orders_created.append(order)

            print(f"  📈 BUY order at ${price:.2f} (MA: ${ma:.2f})")

        elif row['close'] > ma * 1.02:  # Sell if 2% above MA
            # Sell signal
            order = Order(
                id=f"order-{i:03d}",
                portfolio_id=portfolio.id,
                symbol="BTC/USD",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.01"),
                price=price,
                created_time=row['timestamp']
            )

            orders_created.append(order)

            print(f"  📉 SELL order at ${price:.2f} (MA: ${ma:.2f})")

    # 6. Summary
    print("\n" + "=" * 40)
    print("Trading Summary")
    print("=" * 40)

    print(f"✓ Total orders created: {len(orders_created)}")

    buy_orders = [o for o in orders_created if o.side == OrderSide.BUY]
    sell_orders = [o for o in orders_created if o.side == OrderSide.SELL]

    print(f"  - Buy orders: {len(buy_orders)}")
    print(f"  - Sell orders: {len(sell_orders)}")

    # 7. Portfolio final state
    saved_portfolio = await portfolio_repo.find_by_id(portfolio.id)
    print(f"\n✓ Portfolio retrieved from repository")
    print(f"  - ID: {saved_portfolio.id}")
    print(f"  - Name: {saved_portfolio.name}")
    print(f"  - Balance: ${saved_portfolio.initial_capital}")

    print("\n" + "=" * 60)
    print("✅ Minimal Example Completed Successfully!")
    print("=" * 60)


def main():
    """Main entry point"""
    try:
        asyncio.run(run_minimal_example())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())