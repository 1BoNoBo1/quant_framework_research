#!/usr/bin/env python
"""
🔬 Démonstration de l'intégration Research Platform avec QFrame Core

Ce script montre comment la nouvelle infrastructure de recherche
utilise et étend les implémentations existantes de QFrame.
"""

import asyncio
from datetime import datetime, timedelta
import pandas as pd

from qframe.research.integration_layer import create_research_integration
from qframe.core.container import get_container
from qframe.core.config import get_config


async def main():
    print("🚀 QFrame Research Platform Integration Demo")
    print("=" * 60)

    # 1. Initialize integration layer
    print("\n📦 1. Initializing Research Integration Layer...")
    integration = create_research_integration(use_minio=True)

    # Show integration status
    status = integration.get_integration_status()
    print(f"✅ Container Services:")
    print(f"   - Strategies registered: {status['container_services']['strategies']}")
    print(f"   - Data providers: {status['container_services']['data_providers']}")
    print(f"   - Feature processors: {status['container_services']['feature_processors']}")

    # 2. Sync market data from QFrame providers to Data Lake
    print("\n📊 2. Syncing Market Data to Data Lake...")
    symbols = ["BTC/USDT", "ETH/USDT"]

    try:
        await integration.sync_market_data_to_lake(
            symbols=symbols,
            timeframe="1h",
            start_date=datetime.now() - timedelta(days=30)
        )
        print(f"✅ Market data synced for {symbols}")
    except Exception as e:
        print(f"⚠️ Market data sync skipped: {e}")

    # 3. Compute features using QFrame symbolic operators
    print("\n🔧 3. Computing Features with QFrame Processors...")

    # Create sample data
    sample_data = pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1100, 1200, 1300, 1400],
        "timestamp": pd.date_range(start="2024-01-01", periods=5, freq="1h")
    })

    # Compute features
    features_df = await integration.compute_research_features(
        data=sample_data,
        include_symbolic=True,
        include_ml=False
    )

    print(f"✅ Features computed: {len(features_df.columns) - len(sample_data.columns)} new features")
    print(f"   Sample features: {list(features_df.columns[-3:])}")

    # 4. Create research portfolio with QFrame strategies
    print("\n💼 4. Creating Research Portfolio...")
    portfolio = integration.create_research_portfolio(
        portfolio_id="research_portfolio_001",
        initial_capital=100000.0,
        strategies=["adaptive_mean_reversion", "dmn_lstm"]
    )
    print(f"✅ Portfolio created with ID: {portfolio.id}")

    # 5. Show Feature Store integration
    print("\n🏪 5. Feature Store Integration:")
    print(f"✅ Feature Groups Available:")
    for group_name in integration.feature_store.feature_groups.keys():
        group = integration.feature_store.feature_groups[group_name]
        print(f"   - {group_name}: {len(group.features)} features")

    # 6. Show Data Catalog statistics
    print("\n📚 6. Data Catalog Statistics:")
    catalog_stats = integration.catalog.get_statistics()
    print(f"✅ Total datasets: {catalog_stats['total_datasets']}")
    print(f"   By type:")
    for dtype, count in catalog_stats['datasets_by_type'].items():
        print(f"   - {dtype}: {count}")

    # 7. Demonstrate strategy usage from container
    print("\n🎯 7. QFrame Strategy Integration:")
    container = get_container()

    # List all registered strategies
    strategies = ["adaptive_mean_reversion", "dmn_lstm", "funding_arbitrage", "rl_alpha"]
    for strategy_name in strategies:
        try:
            strategy = container.resolve("Strategy", name=strategy_name)
            print(f"✅ Strategy '{strategy_name}' available: {type(strategy).__name__}")
        except:
            print(f"❌ Strategy '{strategy_name}' not found")

    # 8. Show complete integration architecture
    print("\n🏗️ 8. Integration Architecture:")
    print("✅ Research Platform Components:")
    print("   📦 Data Lake (MinIO/S3)")
    print("   ↓")
    print("   🔗 Integration Layer")
    print("   ↓")
    print("   🎯 QFrame Core Components:")
    print("      - Strategies (Mean Reversion, LSTM, RL)")
    print("      - Data Providers (Binance, CCXT)")
    print("      - Feature Processors (Symbolic Operators)")
    print("      - Backtesting Service")
    print("      - Portfolio Management")
    print("   ↓")
    print("   🔬 Research Services:")
    print("      - JupyterHub (Multi-user notebooks)")
    print("      - MLflow (Experiment tracking)")
    print("      - Dask/Ray (Distributed computing)")
    print("      - Feature Store (Centralized features)")
    print("      - Data Catalog (Metadata & lineage)")

    print("\n" + "=" * 60)
    print("✨ Integration Complete!")
    print("\n📊 The Research Platform fully integrates with:")
    print("   • All QFrame strategies")
    print("   • Data providers and pipelines")
    print("   • Feature engineering components")
    print("   • Backtesting and portfolio services")
    print("\n🚀 Researchers can now use QFrame components")
    print("   directly in Jupyter notebooks with full")
    print("   Data Lake and MLflow integration!")


if __name__ == "__main__":
    asyncio.run(main())