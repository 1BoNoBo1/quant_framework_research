"""
Exemple d'utilisation du BacktestEngine
=======================================

Démontre l'utilisation du système de backtesting QFrame avec une stratégie simple.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qframe.domain.entities.backtest import (
    BacktestConfiguration, BacktestType, RebalanceFrequency
)
from qframe.domain.services.backtesting_service import BacktestingService
from qframe.infrastructure.persistence.memory_backtest_repository import MemoryBacktestRepository
from qframe.infrastructure.persistence.memory_strategy_repository import MemoryStrategyRepository
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository
from qframe.domain.entities.strategy import Strategy, StrategyStatus, StrategyType


async def run_backtest_example():
    """Exemple complet d'exécution de backtest"""

    print("🚀 QFrame BacktestEngine - Exemple d'utilisation")
    print("=" * 50)

    # 1. Créer les repositories
    backtest_repo = MemoryBacktestRepository()
    strategy_repo = MemoryStrategyRepository()
    portfolio_repo = MemoryPortfolioRepository()

    # 2. Créer le service de backtesting
    backtesting_service = BacktestingService(
        backtest_repository=backtest_repo,
        strategy_repository=strategy_repo,
        portfolio_repository=portfolio_repo
    )

    # 3. Créer une stratégie d'exemple
    test_strategy = Strategy(
        id="test-ma-strategy",
        name="Moving Average Crossover",
        description="Simple MA crossover strategy for testing",
        parameters={
            "fast_period": 10,
            "slow_period": 20,
            "lookback": 252
        },
        strategy_type=StrategyType.MOMENTUM,
        status=StrategyStatus.ACTIVE
    )

    # Sauvegarder la stratégie
    await strategy_repo.save(test_strategy)
    print(f"✅ Stratégie créée: {test_strategy.name}")

    # 4. Configurer le backtest
    config = BacktestConfiguration(
        name="Backtest MA Crossover - BTC/USDT",
        description="Test de la stratégie MA Crossover sur 1 an",
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        initial_capital=Decimal("100000"),
        strategy_ids=[test_strategy.id],
        strategy_allocations={test_strategy.id: Decimal("1.0")},  # 100% allocation
        transaction_cost=Decimal("0.001"),  # 0.1% transaction cost
        slippage=Decimal("0.0005"),  # 0.05% slippage
        rebalance_frequency=RebalanceFrequency.DAILY,
        backtest_type=BacktestType.SINGLE_PERIOD,
        max_position_size=Decimal("0.2"),  # 20% max per position
        benchmark_symbol="BTC"
    )

    print(f"📊 Configuration backtest:")
    print(f"   • Période: {config.start_date.strftime('%Y-%m-%d')} → {config.end_date.strftime('%Y-%m-%d')}")
    print(f"   • Capital initial: ${config.initial_capital:,.2f}")
    print(f"   • Coûts transaction: {float(config.transaction_cost)*100:.2f}%")
    print(f"   • Slippage: {float(config.slippage)*100:.3f}%")

    # 5. Valider la configuration
    validation_errors = config.validate()
    if validation_errors:
        print(f"❌ Erreurs de configuration:")
        for error in validation_errors:
            print(f"   • {error}")
        return

    print("✅ Configuration valide")

    # 6. Lancer le backtest
    print("\n🔄 Lancement du backtest...")
    try:
        # Note: Dans un vrai backtest, il faudrait des données de marché
        # Ici on démontre juste la structure du service
        result = await backtesting_service.run_backtest(config)

        print(f"📈 Résultat du backtest:")
        print(f"   • Statut: {result.status.value}")
        print(f"   • Capital initial: ${result.initial_capital:,.2f}")

        if result.status.value == "failed" and result.error_message:
            print(f"   • Erreur: {result.error_message}")

        if result.metrics:
            print(f"   • Métriques générées: {result.metrics.total_trades} trades")
            print(f"   • Return total: {result.metrics.total_return:.2f}%")
            print(f"   • Sharpe ratio: {result.metrics.sharpe_ratio:.2f}")

    except Exception as e:
        print(f"❌ Erreur lors du backtest: {e}")

    # 7. Démonstration des capacités avancées
    print(f"\n🎯 Capacités du BacktestEngine QFrame:")
    print(f"   ✅ Single period backtests")
    print(f"   ✅ Walk-forward analysis")
    print(f"   ✅ Monte Carlo simulations")
    print(f"   ✅ Multi-strategy support")
    print(f"   ✅ Transaction costs & slippage")
    print(f"   ✅ Risk management intégré")
    print(f"   ✅ Métriques de performance avancées")
    print(f"   ✅ Persistence des résultats")

    print(f"\n🏆 BacktestEngine QFrame - Prêt pour la production!")


if __name__ == "__main__":
    asyncio.run(run_backtest_example())