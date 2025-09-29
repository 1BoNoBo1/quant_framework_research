#!/usr/bin/env python3
"""
🚀 PHASE 1 - Test Framework avec Données Réelles CCXT
===================================================

Test complet d'AdaptiveMeanReversion avec vraies données de marché
via CCXT pour validation du framework complet.
"""

import asyncio
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

print("🚀 PHASE 1 - SCALING UP AVEC DONNÉES RÉELLES")
print("=" * 55)
print(f"⏱️ Début: {datetime.now().strftime('%H:%M:%S')}\n")

async def test_ccxt_data_retrieval():
    """Test récupération données réelles via CCXT."""

    print("🔌 1. TEST CCXT DATA PROVIDER")
    print("-" * 30)

    try:
        from qframe.infrastructure.data.ccxt_provider import CCXTProvider

        # Test avec Binance (pas besoin d'API key pour données publiques)
        print("   📊 Initialisation Binance provider...")
        provider = CCXTProvider(exchange_name='binance')

        print("   ✅ Provider créé")
        print(f"   📛 Exchange: {provider.exchange_name}")

        # Test récupération données OHLCV
        print("   🔄 Récupération données BTC/USDT...")

        try:
            # Connecter d'abord
            await provider.connect()

            # Récupérer dernières données (interval 1h, 100 points)
            data_points = await provider.get_klines(
                symbol='BTC/USDT',
                interval='1h',
                limit=100
            )

            # Convertir en DataFrame si nécessaire
            if data_points:
                data = pd.DataFrame([
                    {
                        'timestamp': dp.timestamp,
                        'open': float(dp.open),
                        'high': float(dp.high),
                        'low': float(dp.low),
                        'close': float(dp.close),
                        'volume': float(dp.volume)
                    } for dp in data_points
                ])
            else:
                data = None

            if data is not None and not data.empty:
                print(f"   ✅ {len(data)} points récupérés")
                print(f"   📈 Prix range: {data['close'].min():.2f} - {data['close'].max():.2f}")
                print(f"   📅 Période: {data['timestamp'].min()} à {data['timestamp'].max()}")
                print(f"   💾 Colonnes: {list(data.columns)}")

                # Vérifier qualité données
                missing = data.isnull().sum().sum()
                print(f"   🔍 Données manquantes: {missing}")

                return data
            else:
                print("   ❌ Aucune donnée récupérée")
                return None

        except Exception as e:
            print(f"   ❌ Erreur récupération: {e}")
            return None

    except Exception as e:
        print(f"   ❌ Erreur provider: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_adaptive_mean_reversion_real_data(real_data):
    """Test AdaptiveMeanReversion avec vraies données."""

    print(f"\n🎯 2. TEST ADAPTIVE MEAN REVERSION")
    print("-" * 35)

    if real_data is None:
        print("   ❌ Pas de données disponibles")
        return False

    try:
        from qframe.strategies.research.adaptive_mean_reversion_strategy import (
            AdaptiveMeanReversionStrategy,
            AdaptiveMeanReversionConfig
        )

        # Mock providers nécessaires
        class RealDataProvider:
            async def fetch_ohlcv(self, symbol, timeframe, limit=1000, start_time=None, end_time=None):
                return real_data
            async def fetch_latest_price(self, symbol):
                return float(real_data['close'].iloc[-1])
            async def get_klines(self, symbol, interval, limit=1000, start_time=None, end_time=None):
                return real_data

        class MockRiskManager:
            def calculate_position_size(self, signal, portfolio_value, current_positions):
                return 0.02 * portfolio_value
            def check_risk_limits(self, signal, current_positions):
                return True

        print("   🏗️ Initialisation stratégie...")

        # Configuration par défaut
        config = AdaptiveMeanReversionConfig()
        provider = RealDataProvider()
        risk_manager = MockRiskManager()

        strategy = AdaptiveMeanReversionStrategy(provider, risk_manager, config)
        print(f"   ✅ Stratégie créée: {strategy.get_name()}")

        # Test génération signaux avec vraies données
        print("   🎯 Génération signaux avec données réelles...")

        import time
        start_time = time.time()

        signals = strategy.generate_signals(real_data)

        generation_time = time.time() - start_time

        if signals and len(signals) > 0:
            print(f"   ✅ {len(signals)} signaux générés en {generation_time:.3f}s")

            # Analyse des signaux
            signal_values = [s.signal for s in signals]
            confidences = [s.confidence for s in signals]
            regimes = [s.regime for s in signals]

            # Statistiques
            positive_signals = sum(1 for s in signal_values if s > 0)
            negative_signals = sum(1 for s in signal_values if s < 0)

            print(f"\n   📊 ANALYSE SIGNAUX:")
            print(f"      📈 Signaux BUY:  {positive_signals}")
            print(f"      📉 Signaux SELL: {negative_signals}")
            print(f"      🎯 Confidence moyenne: {np.mean(confidences):.3f}")
            print(f"      🌍 Régimes détectés: {set(regimes)}")

            # Premiers signaux
            print(f"\n   🔍 PREMIERS SIGNAUX:")
            for i, signal in enumerate(signals[:3]):
                action = "BUY" if signal.signal > 0 else "SELL" if signal.signal < 0 else "HOLD"
                print(f"      {i+1}. {action} | Force: {abs(signal.signal):.3f} | Conf: {signal.confidence:.3f}")

            return True
        else:
            print("   ⚠️ Aucun signal généré")
            return False

    except Exception as e:
        print(f"   ❌ Erreur stratégie: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_comparison(real_data):
    """Compare performance données réelles vs synthétiques."""

    print(f"\n📊 3. COMPARAISON PERFORMANCE")
    print("-" * 28)

    if real_data is None:
        print("   ❌ Pas de données pour comparaison")
        return

    try:
        # Générer données synthétiques de même taille
        def generate_synthetic_data(size):
            dates = pd.date_range(start=datetime.now() - timedelta(hours=size),
                                 end=datetime.now(), freq='1h')
            np.random.seed(42)
            returns = np.random.normal(0.0001, 0.02, size)
            prices = 45000 * np.cumprod(1 + returns)

            return pd.DataFrame({
                'timestamp': dates[:size],
                'open': prices * 0.999,
                'high': prices * 1.001,
                'low': prices * 0.998,
                'close': prices,
                'volume': np.random.uniform(100, 1000, size),
                'symbol': 'BTC/USDT'
            })

        synthetic_data = generate_synthetic_data(len(real_data))

        print(f"   📊 Données réelles:   {len(real_data)} points")
        print(f"   🎲 Données synthétiques: {len(synthetic_data)} points")

        # Statistiques comparatives
        real_volatility = real_data['close'].pct_change().std()
        synth_volatility = synthetic_data['close'].pct_change().std()

        real_returns = real_data['close'].pct_change().mean()
        synth_returns = synthetic_data['close'].pct_change().mean()

        print(f"\n   📈 COMPARAISON STATISTIQUE:")
        print(f"      Volatilité réelle:     {real_volatility:.6f}")
        print(f"      Volatilité synthétique: {synth_volatility:.6f}")
        print(f"      Return réel:           {real_returns:.6f}")
        print(f"      Return synthétique:    {synth_returns:.6f}")

        # Test corrélation
        if len(real_data) == len(synthetic_data):
            correlation = np.corrcoef(
                real_data['close'].pct_change().dropna(),
                synthetic_data['close'].pct_change().dropna()
            )[0,1]
            print(f"      Corrélation R/S:       {correlation:.3f}")

    except Exception as e:
        print(f"   ⚠️ Erreur comparaison: {e}")

async def explore_framework_components():
    """Explorer les composants disponibles du framework."""

    print(f"\n🔍 4. EXPLORATION FRAMEWORK")
    print("-" * 27)

    components = {
        "BacktestingService": "qframe.domain.services.backtesting_service",
        "ExecutionService": "qframe.domain.services.execution_service",
        "PortfolioService": "qframe.domain.services.portfolio_service",
        "RiskCalculationService": "qframe.domain.services.risk_calculation_service"
    }

    available_components = []

    for name, module in components.items():
        try:
            exec(f"from {module} import {name}")
            print(f"   ✅ {name} disponible")
            available_components.append(name)
        except Exception as e:
            print(f"   ⚠️ {name}: {e}")

    print(f"\n   📊 {len(available_components)}/{len(components)} services disponibles")

    # Test documentation
    try:
        import mkdocs
        print(f"   📚 MkDocs disponible pour documentation")
    except:
        print(f"   📚 MkDocs non disponible")

async def main():
    """Point d'entrée principal Phase 1."""

    try:
        # Étape 1: Récupération données réelles
        real_data = await test_ccxt_data_retrieval()

        # Étape 2: Test stratégie avec vraies données
        if real_data is not None:
            strategy_success = await test_adaptive_mean_reversion_real_data(real_data)

            # Étape 3: Comparaison performance
            await test_performance_comparison(real_data)
        else:
            strategy_success = False

        # Étape 4: Exploration framework
        await explore_framework_components()

        # Résultats finaux
        print(f"\n" + "=" * 55)
        print("🎯 RÉSULTATS PHASE 1")
        print("=" * 55)

        if real_data is not None and strategy_success:
            print("🎉 SUCCÈS COMPLET!")
            print("✅ Données réelles récupérées via CCXT")
            print("✅ AdaptiveMeanReversion fonctionne avec vraies données")
            print("✅ Framework validé avec données de marché")

            print(f"\n🚀 PROCHAINES ÉTAPES:")
            print("   • Intégrer backtesting avec données historiques")
            print("   • Configurer monitoring temps réel")
            print("   • Optimiser paramètres avec vraies données")
            print("   • Déployer en mode production")

        elif real_data is not None:
            print("⚠️ SUCCÈS PARTIEL")
            print("✅ Données réelles récupérées")
            print("❌ Problème avec stratégie sur vraies données")

        else:
            print("❌ ÉCHEC - Impossible de récupérer données")
            print("🔧 Vérifier connection internet et CCXT")

        print(f"\n⏱️ Fin: {datetime.now().strftime('%H:%M:%S')}")

        return real_data is not None and strategy_success

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE PHASE 1: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Configuration logging pour réduire le bruit
    logging.getLogger('ccxt').setLevel(logging.WARNING)

    # Exécuter Phase 1
    success = asyncio.run(main())

    # Code de sortie
    sys.exit(0 if success else 1)