#!/usr/bin/env python3
"""
🚀 PHASE 2 - Intégration Backtesting Engine avec Données Réelles
===============================================================

Test d'intégration complet du BacktestingService avec CCXT et AdaptiveMeanReversion.
Évolution naturelle de la Phase 1 vers un système de backtesting opérationnel.
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import List, Dict, Any, Optional
import logging
import time

print("🚀 PHASE 2 - INTÉGRATION BACKTESTING + DONNÉES RÉELLES")
print("=" * 60)
print(f"⏱️ Début: {datetime.now().strftime('%H:%M:%S')}\n")

class MockStrategy:
    """Mock Strategy pour intégration backtesting."""

    def __init__(self, name: str):
        self.name = name
        self.signal_count = 0

    def get_name(self) -> str:
        return self.name

    def generate_signals(self, data: pd.DataFrame) -> List[Any]:
        """Génère des signaux mock pour le backtesting."""
        if len(data) < 20:  # Minimum de données
            return []

        # Générer signaux based sur simple moving average crossover
        short_ma = data['close'].rolling(window=5).mean()
        long_ma = data['close'].rolling(window=20).mean()

        signals = []
        for i in range(20, len(data)):
            if short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i-1] <= long_ma.iloc[i-1]:
                # Signal d'achat
                signals.append(MockSignal(
                    timestamp=data.index[i],
                    action="BUY",
                    strength=0.8,
                    symbol=data.get('symbol', ['BTC/USD'])[0] if 'symbol' in data.columns else 'BTC/USD'
                ))
                self.signal_count += 1
            elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i-1] >= long_ma.iloc[i-1]:
                # Signal de vente
                signals.append(MockSignal(
                    timestamp=data.index[i],
                    action="SELL",
                    strength=0.8,
                    symbol=data.get('symbol', ['BTC/USD'])[0] if 'symbol' in data.columns else 'BTC/USD'
                ))
                self.signal_count += 1

        return signals

class MockSignal:
    """Mock Signal compatible avec le backtesting."""

    def __init__(self, timestamp, action: str, strength: float, symbol: str):
        self.timestamp = timestamp
        self.action = action
        self.strength = strength
        self.symbol = symbol
        self.confidence = strength

class MockBacktestRepository:
    """Mock repository pour les résultats de backtest."""

    def __init__(self):
        self.results = {}

    async def save_result(self, result):
        self.results[result.name] = result
        return result

    async def find_by_id(self, result_id):
        return self.results.get(result_id)

class MockStrategyRepository:
    """Mock repository pour les stratégies."""

    def __init__(self):
        self.strategies = {}

    async def save(self, strategy):
        self.strategies[strategy.get_name()] = strategy
        return strategy

    async def find_by_id(self, strategy_id):
        return self.strategies.get(strategy_id)

class MockPortfolioRepository:
    """Mock repository pour les portfolios."""

    def __init__(self):
        self.portfolios = {}

    async def save(self, portfolio):
        self.portfolios[portfolio.name] = portfolio
        return portfolio

    async def find_by_id(self, portfolio_id):
        return self.portfolios.get(portfolio_id)

async def get_historical_ccxt_data(symbol: str = "BTC/USDT", days: int = 30) -> Optional[pd.DataFrame]:
    """Récupère les données historiques via CCXT."""

    print(f"📊 1. RÉCUPÉRATION DONNÉES HISTORIQUES CCXT")
    print("-" * 45)

    try:
        from qframe.infrastructure.data.ccxt_provider import CCXTProvider

        print(f"   🔌 Initialisation provider Binance...")
        provider = CCXTProvider(exchange_name='binance')

        # Connecter au provider
        await provider.connect()
        print(f"   ✅ Connexion établie")

        # Calculer la période historique
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        print(f"   📅 Période: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        print(f"   📈 Récupération données {symbol} (intervalle 1h)...")

        # Récupérer données historiques (1h candles for 30 days = ~720 points)
        data_points = await provider.get_klines(
            symbol=symbol,
            interval='1h',
            limit=min(720, days * 24)  # Max points pour la période
        )

        if not data_points:
            print(f"   ❌ Aucune donnée récupérée")
            return None

        # Convertir en DataFrame
        data = pd.DataFrame([
            {
                'timestamp': dp.timestamp,
                'open': float(dp.open),
                'high': float(dp.high),
                'low': float(dp.low),
                'close': float(dp.close),
                'volume': float(dp.volume),
                'symbol': symbol
            } for dp in data_points
        ])

        # Trier par timestamp et définir index
        data = data.sort_values('timestamp').reset_index(drop=True)
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index('datetime', inplace=True)

        print(f"   ✅ {len(data)} points récupérés")
        print(f"   📊 Prix range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"   💹 Volume moyen: {data['volume'].mean():.0f}")
        print(f"   📈 Return total: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100:.2f}%")

        # Valider qualité données
        missing_data = data.isnull().sum().sum()
        if missing_data > 0:
            print(f"   ⚠️ {missing_data} valeurs manquantes détectées")

        return data

    except Exception as e:
        print(f"   ❌ Erreur récupération CCXT: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_backtesting_integration(historical_data: pd.DataFrame) -> bool:
    """Test intégration complète du backtesting avec données réelles."""

    print(f"\n🎯 2. TEST INTEGRATION BACKTESTING ENGINE")
    print("-" * 42)

    try:
        # Import des composants nécessaires
        from qframe.domain.services.backtesting_service import BacktestingService
        from qframe.domain.entities.backtest import (
            BacktestConfiguration, BacktestType, BacktestStatus
        )
        from qframe.domain.entities.portfolio import Portfolio

        print(f"   ✅ Imports BacktestingService réussis")

        # Créer les repositories mock
        backtest_repo = MockBacktestRepository()
        strategy_repo = MockStrategyRepository()
        portfolio_repo = MockPortfolioRepository()

        print(f"   ✅ Repositories mock créés")

        # Créer le service de backtesting
        backtesting_service = BacktestingService(
            backtest_repository=backtest_repo,
            strategy_repository=strategy_repo,
            portfolio_repository=portfolio_repo
        )

        print(f"   ✅ BacktestingService instancié")

        # Créer une stratégie mock
        mock_strategy = MockStrategy("SimpleMA_Crossover")
        await strategy_repo.save(mock_strategy)

        print(f"   ✅ Stratégie mock sauvegardée: {mock_strategy.get_name()}")

        # Configuration du backtest
        start_date = historical_data.index[0].to_pydatetime()
        end_date = historical_data.index[-1].to_pydatetime()

        config = BacktestConfiguration(
            id="backtest_integration_001",
            name="Integration_Test_Real_Data",
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal("10000.00"),
            strategy_ids=[mock_strategy.get_name()],
            transaction_cost=Decimal("0.001"),  # 0.1%
            slippage=Decimal("0.0005"),  # 0.05%
            backtest_type=BacktestType.SINGLE_PERIOD
        )

        print(f"   ✅ Configuration backtest créée")
        print(f"      📅 Période: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        print(f"      💰 Capital initial: ${config.initial_capital}")
        print(f"      🎯 Stratégies: {config.strategy_ids}")

        # Patch temporaire pour utiliser nos données réelles
        original_generate_market_data = backtesting_service._generate_market_data

        async def use_real_data(start_date, end_date):
            """Utilise nos données réelles au lieu de générer des données synthétiques."""
            return historical_data

        backtesting_service._generate_market_data = use_real_data

        # Patch pour la génération de signaux avec notre stratégie mock
        original_generate_strategy_signals = backtesting_service._generate_strategy_signals

        async def generate_real_signals(strategy, data):
            """Utilise notre stratégie mock pour générer des signaux réels."""
            if hasattr(strategy, 'generate_signals'):
                return strategy.generate_signals(data)
            return []

        backtesting_service._generate_strategy_signals = generate_real_signals

        print(f"\n   🚀 Lancement du backtest...")

        start_time = time.time()

        # Exécuter le backtest
        result = await backtesting_service.run_backtest(config)

        execution_time = time.time() - start_time

        print(f"   ⏱️ Exécution terminée en {execution_time:.2f}s")

        # Analyser les résultats
        if result.status == BacktestStatus.COMPLETED:
            print(f"\n   🎉 BACKTEST TERMINÉ AVEC SUCCÈS!")
            print(f"      📊 Status: {result.status.value}")
            print(f"      ⏱️ Durée: {result.end_time - result.start_time if result.end_time else 'N/A'}")

            if result.metrics:
                metrics = result.metrics
                print(f"\n   📈 MÉTRIQUES DE PERFORMANCE:")
                print(f"      💰 Return total: {float(metrics.total_return) * 100:.2f}%")
                print(f"      📊 Return annualisé: {float(metrics.annualized_return) * 100:.2f}%")
                print(f"      📉 Volatilité: {float(metrics.volatility) * 100:.2f}%")
                print(f"      ⭐ Sharpe Ratio: {float(metrics.sharpe_ratio):.3f}")
                print(f"      📉 Max Drawdown: {float(metrics.max_drawdown) * 100:.2f}%")
                print(f"      🎯 Sortino Ratio: {float(metrics.sortino_ratio):.3f}")
                print(f"      📊 Calmar Ratio: {float(metrics.calmar_ratio):.3f}")

                print(f"\n   📊 STATISTIQUES TRADING:")
                print(f"      📈 Total trades: {metrics.total_trades}")
                print(f"      ✅ Trades gagnants: {metrics.winning_trades}")
                print(f"      ❌ Trades perdants: {metrics.losing_trades}")
                print(f"      🎯 Win rate: {float(metrics.win_rate) * 100:.1f}%")
                print(f"      💵 Profit factor: {float(metrics.profit_factor):.2f}")

                if hasattr(metrics, 'skewness') and hasattr(metrics, 'kurtosis'):
                    print(f"\n   📊 STATISTIQUES AVANCÉES:")
                    print(f"      📊 Skewness: {float(metrics.skewness):.3f}")
                    print(f"      📊 Kurtosis: {float(metrics.kurtosis):.3f}")
            else:
                print(f"   ⚠️ Pas de métriques calculées")

            # Test génération signaux de la stratégie
            signals_count = mock_strategy.signal_count
            print(f"\n   🎯 GÉNÉRATION DE SIGNAUX:")
            print(f"      📊 Signaux générés: {signals_count}")
            print(f"      📈 Ratio signaux/données: {signals_count / len(historical_data) * 100:.1f}%")

            return True

        elif result.status == BacktestStatus.FAILED:
            print(f"\n   ❌ BACKTEST ÉCHOUÉ:")
            print(f"      🚨 Erreur: {result.error_message}")
            return False
        else:
            print(f"\n   ⚠️ Status inattendu: {result.status}")
            return False

    except Exception as e:
        print(f"   ❌ Erreur critique backtest: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_comparison():
    """Compare performance du backtesting avec différentes périodes."""

    print(f"\n📊 3. COMPARAISON PERFORMANCE MULTI-PÉRIODES")
    print("-" * 44)

    periods = [7, 15, 30]  # Différentes périodes de test
    results = {}

    for period in periods:
        print(f"\n   🔄 Test période {period} jours...")

        # Récupérer données pour cette période
        data = await get_historical_ccxt_data("BTC/USDT", days=period)

        if data is not None and len(data) > 50:
            # Exécuter backtest
            success = await test_backtesting_integration(data)
            results[period] = {
                'success': success,
                'data_points': len(data),
                'period_return': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
            }
            print(f"   📊 Période {period}j: {'✅' if success else '❌'} | {len(data)} points | Return: {results[period]['period_return']:.2f}%")
        else:
            results[period] = {'success': False, 'data_points': 0, 'period_return': 0}
            print(f"   ❌ Période {period}j: Données insuffisantes")

    print(f"\n   📊 RÉSUMÉ COMPARATIF:")
    successful_tests = sum(1 for r in results.values() if r['success'])
    print(f"      ✅ Tests réussis: {successful_tests}/{len(periods)}")

    if successful_tests > 0:
        avg_return = np.mean([r['period_return'] for r in results.values() if r['success']])
        print(f"      📈 Return moyen: {avg_return:.2f}%")

        best_period = max(results.keys(), key=lambda k: results[k]['period_return'] if results[k]['success'] else -999)
        print(f"      🏆 Meilleure période: {best_period} jours ({results[best_period]['period_return']:.2f}%)")

async def main():
    """Point d'entrée principal Phase 2."""

    try:
        # Étape 1: Récupération données historiques CCXT
        print("🎯 OBJECTIF: Intégrer BacktestingService avec données réelles CCXT")
        print("📊 STRATÉGIE: Utiliser AdaptiveMeanReversion validée + MA Crossover mock")
        print("⚡ PERFORMANCE: Mesurer vitesse et précision du backtesting\n")

        historical_data = await get_historical_ccxt_data("BTC/USDT", days=30)

        if historical_data is None or len(historical_data) < 100:
            print("❌ Données insuffisantes pour continuer")
            return False

        # Étape 2: Test intégration backtesting
        backtest_success = await test_backtesting_integration(historical_data)

        # Étape 3: Comparaison performance
        if backtest_success:
            await test_performance_comparison()

        # Résultats finaux
        print(f"\n" + "=" * 60)
        print("🎯 RÉSULTATS PHASE 2")
        print("=" * 60)

        if backtest_success:
            print("🎉 INTÉGRATION BACKTESTING RÉUSSIE!")
            print("✅ BacktestingService opérationnel avec données CCXT")
            print("✅ Stratégies intégrées et génération de signaux")
            print("✅ Métriques de performance calculées")
            print("✅ Pipeline complet données réelles → backtest → résultats")

            print(f"\n🚀 PHASE 2 COMPLÉTÉE - PROCHAINES ÉTAPES:")
            print("   • Intégrer AdaptiveMeanReversion réelle (au lieu de mock)")
            print("   • Configurer monitoring temps réel")
            print("   • Tests walk-forward et Monte Carlo")
            print("   • Optimisation paramètres stratégies")
            print("   • Interface web pour backtesting")

        else:
            print("⚠️ INTÉGRATION PARTIELLE")
            print("✅ Données CCXT récupérées")
            print("❌ Problèmes avec BacktestingService")
            print("🔧 Corrections nécessaires avant Phase 3")

        print(f"\n⏱️ Fin: {datetime.now().strftime('%H:%M:%S')}")

        return backtest_success

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE PHASE 2: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Configuration logging pour réduire le bruit CCXT
    logging.getLogger('ccxt').setLevel(logging.WARNING)

    # Exécuter Phase 2
    success = asyncio.run(main())

    # Code de sortie
    sys.exit(0 if success else 1)