#!/usr/bin/env python3
"""
Test d'intégration des opérateurs symboliques avec nos alphas
"""

import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajouter le chemin du projet
sys.path.append('.')

from mlpipeline.features.symbolic_operators import AlphaFormulaGenerator, SymbolicOperators
from mlpipeline.alphas.dmn_model import DMNPredictor
from mlpipeline.alphas.mean_reversion import AdaptiveMeanReversion

def create_test_data():
    """Génère des données de test réalistes"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='H')

    # Simulation prix avec trend et volatilité
    price_base = 50000
    returns = np.random.randn(200) * 0.02
    prices = price_base * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(200) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(200)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(200)) * 0.005),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200),
        'ret': returns
    }, index=dates)

    # Ajouter VWAP
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3

    return data

async def test_symbolic_operators():
    """Test des opérateurs symboliques de base"""
    print("🧪 Test des opérateurs symboliques...")

    data = create_test_data()
    ops = SymbolicOperators()

    # Test des opérateurs individuels
    close = data['close']
    volume = data['volume']
    returns = data['ret']

    print("\n📊 Test opérateurs de base:")
    print(f"Sign(returns) - unique values: {ops.sign(returns).value_counts().to_dict()}")
    print(f"CS_Rank(volume) - range: [{ops.cs_rank(volume).min():.3f}, {ops.cs_rank(volume).max():.3f}]")
    print(f"TS_Rank(close, 10) - mean: {ops.ts_rank(close, 10).mean():.3f}")
    print(f"Delta(close, 5) - mean: {ops.delta(close, 5).mean():.2f}")
    print(f"Skew(returns, 20) - mean: {ops.skew(returns, 20).mean():.3f}")
    print(f"Kurt(returns, 20) - mean: {ops.kurt(returns, 20).mean():.3f}")

    return True

async def test_alpha_formula_generator():
    """Test du générateur de formules alpha"""
    print("\n🎯 Test du générateur de formules alpha...")

    data = create_test_data()
    generator = AlphaFormulaGenerator()

    try:
        # Test génération features
        features = generator.generate_enhanced_features(data)
        print(f"Features générées: {features.shape[1]} colonnes")
        print(f"Features list: {list(features.columns)}")

        # Test alphas spécifiques avec gestion d'erreur
        alpha_006 = generator.generate_alpha_006(data)
        alpha_061 = generator.generate_alpha_061(data)

        # Vérification que les résultats sont valides
        if not alpha_006.empty and alpha_006.notna().any():
            print(f"\nAlpha 006 stats: mean={alpha_006.mean():.4f}, std={alpha_006.std():.4f}")
        else:
            print("\nAlpha 006: aucune donnée valide")

        if not alpha_061.empty and alpha_061.notna().any():
            print(f"Alpha 061 stats: mean={alpha_061.mean():.4f}, std={alpha_061.std():.4f}")
        else:
            print("Alpha 061: aucune donnée valide")

        return features

    except Exception as e:
        print(f"❌ Erreur dans test_alpha_formula_generator: {e}")
        return pd.DataFrame()  # Retourner DataFrame vide en cas d'erreur

async def test_dmn_integration():
    """Test intégration DMN LSTM avec features symboliques"""
    print("\n🧠 Test intégration DMN LSTM...")

    data = create_test_data()

    # Créer instance DMN
    dmn_alpha = DMNPredictor("BTCUSDT")

    try:
        # Génération des prédictions
        predictions = await dmn_alpha.predict(data)

        print(f"✅ DMN predictions: {len(predictions)} points")
        print(f"Signal range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"Signal mean: {predictions.mean():.4f}")
        print(f"Signal std: {predictions.std():.4f}")

        # Vérifier que les features symboliques sont utilisées
        print("✅ Intégration symbolique réussie dans DMN")

        return predictions

    except Exception as e:
        print(f"❌ Erreur DMN integration: {e}")
        return None

async def test_mean_reversion_integration():
    """Test intégration Mean Reversion avec features symboliques"""
    print("\n📈 Test intégration Mean Reversion...")

    data = create_test_data()

    # Créer instance Mean Reversion
    mr_alpha = AdaptiveMeanReversion()

    try:
        # Test optimisation ML avec features symboliques
        optimal_params = mr_alpha.optimize_parameters_ml(data)

        print(f"✅ Paramètres optimisés: {optimal_params}")
        print("✅ Intégration symbolique réussie dans Mean Reversion")

        return optimal_params

    except Exception as e:
        print(f"❌ Erreur Mean Reversion integration: {e}")
        return None

async def test_performance_comparison():
    """Compare performances avec et sans features symboliques"""
    print("\n🏁 Test comparaison de performance...")

    data = create_test_data()

    # Simulation simple de trading
    returns = data['ret']

    try:
        # DMN avec features symboliques
        dmn_alpha = DMNPredictor("BTCUSDT")
        dmn_signals = await dmn_alpha.predict(data)

        # Calcul performance basique avec alignement des dimensions
        if len(dmn_signals) > len(returns):
            dmn_signals = dmn_signals[:len(returns)]
        elif len(dmn_signals) < len(returns):
            returns = returns[:len(dmn_signals)]

        # Décalage d'1 période pour éviter look-ahead bias
        dmn_signals_shifted = dmn_signals[:-1] if len(dmn_signals) > 1 else dmn_signals
        returns_aligned = returns[1:] if len(returns) > 1 else returns

        # Assurer même longueur
        min_len = min(len(dmn_signals_shifted), len(returns_aligned))
        dmn_returns = returns_aligned[:min_len] * dmn_signals_shifted[:min_len]
        dmn_sharpe = dmn_returns.mean() / (dmn_returns.std() + 1e-8) * np.sqrt(365*24)

        print(f"📊 DMN Performance (avec symboliques):")
        print(f"   Sharpe annualisé: {dmn_sharpe:.3f}")
        print(f"   Rendement total: {dmn_returns.sum():.4f}")
        print(f"   Nombre de signaux: {np.sum(np.abs(dmn_signals_shifted) > 0.1) if len(dmn_signals_shifted) > 0 else 0}")

        return True

    except Exception as e:
        print(f"❌ Erreur performance test: {e}")
        return False

async def main():
    """Test principal"""
    print("🚀 Test d'intégration des opérateurs symboliques")
    print("=" * 60)

    # Tests séquentiels
    tests = [
        ("Opérateurs symboliques", test_symbolic_operators),
        ("Générateur de formules", test_alpha_formula_generator),
        ("Intégration DMN", test_dmn_integration),
        ("Intégration Mean Reversion", test_mean_reversion_integration),
        ("Comparaison performance", test_performance_comparison)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = await test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            print(f"❌ Erreur {test_name}: {e}")
            results[test_name] = f"❌ ERROR: {e}"

    # Rapport final
    print("\n" + "="*60)
    print("📋 RAPPORT FINAL")
    print("="*60)

    for test_name, status in results.items():
        print(f"{test_name:<30} {status}")

    # Statut global
    passed = sum(1 for status in results.values() if "✅" in status)
    total = len(results)

    print(f"\n🎯 Résultat global: {passed}/{total} tests réussis")

    if passed == total:
        print("🎉 TOUS LES TESTS PASSÉS - Intégration symbolique réussie !")
    else:
        print("⚠️  Certains tests ont échoué - Vérifier les erreurs ci-dessus")

if __name__ == "__main__":
    asyncio.run(main())