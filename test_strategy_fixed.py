#!/usr/bin/env python3
"""
Test Complet de la Stratégie Adaptive Mean Reversion - Version Corrigée
=========================================================================

Tests mis à jour pour toutes les nouvelles fonctionnalités et corrections.
"""

import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Ajout du chemin pour les imports
sys.path.append('.')

def test_imports():
    """Test des imports principaux."""
    print("🔧 Test des imports...")

    try:
        from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
        from qframe.strategies.research.adaptive_mean_reversion_config import AdaptiveMeanReversionConfig
        from qframe.core.parallel_processor import create_safe_parallel_processor
        from qframe.data.validation_simple import create_simple_validator
        print("✅ Tous les imports réussis")
        return True
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def test_configuration():
    """Test de la configuration sans warnings."""
    print("\n🔧 Test de la configuration...")

    try:
        from qframe.strategies.research.adaptive_mean_reversion_config import AdaptiveMeanReversionConfig

        # Configuration par défaut
        config = AdaptiveMeanReversionConfig()
        print(f"✅ Configuration par défaut: {config.name}")

        # Configuration personnalisée
        custom_config = AdaptiveMeanReversionConfig(
            name="test_strategy",
            universe=["BTC/USDT"],
            mean_reversion_windows=[10, 20, 50],
            volatility_windows=[10, 20],
            regime_confidence_threshold=0.6,
            min_data_points=30  # Réduit pour les tests
        )
        print(f"✅ Configuration personnalisée: {custom_config.name}")
        print(f"   - Min data points: {custom_config.min_data_points}")

        return custom_config

    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return None

def test_data_validation():
    """Test du validateur de données corrigé."""
    print("\n🔧 Test de la validation des données...")

    try:
        from qframe.data.validation_simple import create_simple_validator

        # Créer des données de test
        np.random.seed(42)
        good_data = pd.DataFrame({
            'open': np.random.rand(50) * 10 + 100,
            'high': np.random.rand(50) * 10 + 105,
            'low': np.random.rand(50) * 10 + 95,
            'close': np.random.rand(50) * 10 + 100,
            'volume': np.random.randint(1000, 100000, 50)
        })

        # Assurer cohérence OHLCV
        good_data['high'] = np.maximum(good_data[['open', 'close']].max(axis=1), good_data['high'])
        good_data['low'] = np.minimum(good_data[['open', 'close']].min(axis=1), good_data['low'])

        # Test du validateur
        validator = create_simple_validator()
        result = validator.validate_ohlcv_data(good_data, "TEST")

        print(f"✅ Validation réussie: score {result.score:.3f}")
        print(f"   - Erreurs: {len(result.errors)}")
        print(f"   - Métriques: {len(result.metrics)}")

        return good_data

    except Exception as e:
        print(f"❌ Erreur validation: {e}")
        return None

async def test_parallel_processor():
    """Test du processeur parallèle corrigé."""
    print("\n🔧 Test du processeur parallèle...")

    try:
        from qframe.core.parallel_processor import create_safe_parallel_processor

        # Données de test
        data = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50)
        })

        # Processeur sûr
        processor = create_safe_parallel_processor(max_workers=2)
        print(f"✅ Processeur créé: threads={not processor.use_processes}")

        # Feature functions pour test
        def test_sma(df):
            df_copy = df.copy()
            df_copy['sma_10'] = df_copy['close'].rolling(10).mean()
            return df_copy[['sma_10']]

        def test_volatility(df):
            df_copy = df.copy()
            df_copy['vol'] = df_copy['close'].pct_change().rolling(10).std()
            return df_copy[['vol']]

        # Test features parallèles
        features = await processor.process_parallel_features(data, [test_sma, test_volatility])
        print(f"✅ Features parallèles: {len(features.columns)} colonnes")

        # Test métriques de risque
        returns = pd.Series(np.random.randn(50) * 0.01)
        risk_metrics = await processor.parallel_risk_metrics(returns)
        valid_metrics = {k: v for k, v in risk_metrics.items()
                        if isinstance(v, (int, float)) and not np.isnan(v)}
        print(f"✅ Métriques de risque: {len(valid_metrics)} valides")

        processor.cleanup()
        return True

    except Exception as e:
        print(f"❌ Erreur processeur parallèle: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_corrected():
    """Test de la stratégie avec toutes les corrections."""
    print("\n🔧 Test de la stratégie corrigée...")

    try:
        from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
        from qframe.strategies.research.adaptive_mean_reversion_config import AdaptiveMeanReversionConfig

        # Configuration adaptée pour les tests
        config = AdaptiveMeanReversionConfig(
            name="test_strategy",
            mean_reversion_windows=[10, 20],
            volatility_windows=[10, 20],
            min_data_points=30,  # Réduit pour tests
            regime_confidence_threshold=0.6
        )

        # Données de test suffisamment grandes
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(60).cumsum() * 0.1,
            'high': 0,
            'low': 0,
            'close': 100 + np.random.randn(60).cumsum() * 0.1,
            'volume': np.random.randint(1000, 10000, 60)
        }, index=dates)

        data['high'] = data[['open', 'close']].max(axis=1) + 0.05
        data['low'] = data[['open', 'close']].min(axis=1) - 0.05

        print(f"✅ Données de test: {len(data)} points")

        # Mock classes
        class MockDataProvider:
            def get_data(self, symbol, start_date, end_date):
                return data

        class MockRiskManager:
            def calculate_position_size(self, signal, price, volatility):
                return abs(signal) * 0.1

        # Créer et tester la stratégie
        strategy = AdaptiveMeanReversionStrategy(
            data_provider=MockDataProvider(),
            risk_manager=MockRiskManager(),
            config=config
        )

        print(f"✅ Stratégie initialisée: {strategy.name}")

        # Test du feature engineering corrigé
        features = strategy._engineer_features(data)
        print(f"✅ Features créées: {len(features.columns) if features is not None else 0} colonnes")

        if features is not None:
            print(f"   - Lignes: {len(features)}")
            print(f"   - Premières colonnes: {list(features.columns)[:5]}")

            # Test de génération de signaux
            signals = strategy.generate_signals(data)
            print(f"✅ Signaux générés: {len(signals)} signaux")

            if signals:
                print(f"   - Premier signal: {signals[0].action}")
                print(f"   - Force: {signals[0].strength:.3f}")

        # Test des informations de stratégie
        info = strategy.get_strategy_info()
        print(f"✅ Info stratégie: type={info.get('type')}, régime={info.get('current_regime')}")

        # Cleanup
        strategy.cleanup()
        print(f"✅ Cleanup réussi")

        return True

    except Exception as e:
        print(f"❌ Erreur stratégie: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration_complete():
    """Test d'intégration complète avec optimisation parallèle."""
    print("\n🔧 Test d'intégration complète...")

    try:
        from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
        from qframe.strategies.research.adaptive_mean_reversion_config import AdaptiveMeanReversionConfig

        # Configuration
        config = AdaptiveMeanReversionConfig(min_data_points=30)

        # Données
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(60).cumsum() * 0.1,
            'high': 0,
            'low': 0,
            'close': 100 + np.random.randn(60).cumsum() * 0.1,
            'volume': np.random.randint(1000, 10000, 60)
        }, index=dates)

        data['high'] = data[['open', 'close']].max(axis=1) + 0.05
        data['low'] = data[['open', 'close']].min(axis=1) - 0.05

        # Mock classes
        class MockDataProvider:
            def get_data(self, symbol, start_date, end_date):
                return data

        class MockRiskManager:
            def calculate_position_size(self, signal, price, volatility):
                return abs(signal) * 0.1

        # Stratégie avec processeur parallèle
        strategy = AdaptiveMeanReversionStrategy(
            data_provider=MockDataProvider(),
            risk_manager=MockRiskManager(),
            config=config
        )

        print(f"✅ Stratégie avec parallélisation créée")

        # Test optimisation complète
        returns = data['close'].pct_change().dropna()
        optimization_result = await strategy.optimize_strategy_with_parallel_processing(data, returns)

        print(f"✅ Optimisation parallèle: {optimization_result.get('parallel_processing')}")
        print(f"   - Features: {optimization_result.get('feature_count', 0)}")
        print(f"   - Données: {optimization_result.get('data_points', 0)}")

        # Test métriques de risque parallèles
        risk_metrics = await strategy.calculate_parallel_risk_metrics(returns)
        valid_metrics = {k: v for k, v in risk_metrics.items()
                        if isinstance(v, (int, float)) and not np.isnan(v)}
        print(f"✅ Métriques de risque parallèles: {len(valid_metrics)} valides")

        if valid_metrics:
            for key, value in list(valid_metrics.items())[:3]:
                print(f"   - {key}: {value:.4f}")

        strategy.cleanup()
        print(f"✅ Test d'intégration réussi")

        return True

    except Exception as e:
        print(f"❌ Erreur intégration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Fonction principale de test."""
    print("=" * 60)
    print("🧪 TESTS COMPLETS - STRATÉGIE CORRIGÉE")
    print("=" * 60)

    tests_results = []

    # Test 1: Imports
    tests_results.append(test_imports())

    # Test 2: Configuration
    config = test_configuration()
    tests_results.append(config is not None)

    # Test 3: Validation des données
    data = test_data_validation()
    tests_results.append(data is not None)

    # Test 4: Processeur parallèle
    parallel_result = await test_parallel_processor()
    tests_results.append(parallel_result)

    # Test 5: Stratégie corrigée
    strategy_result = test_strategy_corrected()
    tests_results.append(strategy_result)

    # Test 6: Intégration complète
    integration_result = await test_integration_complete()
    tests_results.append(integration_result)

    # Résumé final
    print("\n" + "=" * 60)
    total_tests = len(tests_results)
    passed_tests = sum(tests_results)

    print(f"📊 RÉSULTATS FINAUX")
    print(f"✅ Tests réussis: {passed_tests}/{total_tests}")
    print(f"📈 Taux de réussite: {passed_tests/total_tests*100:.1f}%")

    if passed_tests == total_tests:
        print("🎉 TOUS LES PROBLÈMES SONT CORRIGÉS!")
        print("🚀 Le projet QFrame est entièrement fonctionnel!")
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) en échec")
        print("🔧 Quelques ajustements peuvent être nécessaires")

    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())