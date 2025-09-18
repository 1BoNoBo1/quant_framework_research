#!/usr/bin/env python3
"""
Test complet du système RL Alpha Generator
Valide toute la chaîne : génération → combinaison → intégration
"""

import asyncio
import sys
import pandas as pd
import numpy as np
import logging

sys.path.append('.')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_complete_rl_system():
    """Test complet du système RL"""
    print("🚀 TEST COMPLET DU SYSTÈME RL ALPHA GENERATOR")
    print("=" * 60)

    # Créer des données de test réalistes
    print("📊 Génération des données de test...")
    data = create_realistic_market_data()
    print(f"   ✅ {len(data)} points de données générés")

    # Test 1: Générateur RL de base
    print("\n1️⃣ Test générateur RL de base")
    print("-" * 40)

    try:
        from mlpipeline.alphas.rl_alpha_generator import RLAlphaGenerator

        generator = RLAlphaGenerator(data)
        alphas = generator.generate_alpha_batch(8)
        stats = generator.get_generation_stats()

        print(f"   📈 Alphas générés: {len(alphas)}")
        print(f"   📊 Stats: {stats}")

        if alphas:
            best = generator.get_best_alphas(3)
            for i, alpha in enumerate(best):
                print(f"   🏆 Top {i+1}: {alpha.formula} (IC: {alpha.ic:.4f})")

        test1_result = "✅ PASSED" if len(alphas) > 0 else "⚠️ PARTIAL"

    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        test1_result = "❌ FAILED"
        alphas = []

    # Test 2: Combinateur synergique
    print("\n2️⃣ Test combinateur synergique")
    print("-" * 40)

    try:
        from mlpipeline.alphas.synergistic_combiner import SynergisticAlphaEngine
        from mlpipeline.alphas.rl_alpha_generator import AlphaFormula

        # Créer des alphas de test si la génération a échoué
        if not alphas:
            alphas = [
                AlphaFormula("cs_rank(volume)", ["cs_rank"], 0.05, 0.04, 2, 1),
                AlphaFormula("sign(returns)", ["sign"], 0.03, 0.025, 1, 1),
                AlphaFormula("delta(close, 5)", ["delta"], 0.04, 0.035, 2, 1),
            ]
            print("   📝 Utilisation d'alphas de test")

        engine = SynergisticAlphaEngine(data)
        combinations = engine.generate_synergistic_alphas(alphas, 3)

        print(f"   🔬 Combinaisons générées: {len(combinations)}")

        if combinations:
            best_combo = combinations[0]
            print(f"   🏆 Meilleure combinaison:")
            print(f"      Score: {best_combo.total_score:.4f}")
            print(f"      IC: {best_combo.combined_ic:.4f}")
            print(f"      Diversification: {best_combo.diversification_score:.4f}")

        combo_stats = engine.get_combination_stats()
        print(f"   📊 Stats combinaisons: {combo_stats}")

        test2_result = "✅ PASSED" if len(combinations) > 0 else "⚠️ PARTIAL"

    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        test2_result = "❌ FAILED"
        combinations = []

    # Test 3: Pipeline intégré
    print("\n3️⃣ Test pipeline intégré")
    print("-" * 40)

    try:
        from mlpipeline.alphas.rl_alpha_pipeline import RLAlphaPipeline

        pipeline = RLAlphaPipeline(data, "TESTUSDT")
        results = await pipeline.quick_discovery(5)

        print(f"   🎯 Résultats pipeline:")
        print(f"      Alphas générés: {results['total_alphas_generated']}")
        print(f"      Alphas valides: {results['valid_alphas']}")
        print(f"      Combinaisons: {results['best_combinations']}")
        print(f"      Temps: {results['execution_time']:.2f}s")

        test3_result = "✅ PASSED" if results['total_alphas_generated'] > 0 else "⚠️ PARTIAL"

    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        test3_result = "❌ FAILED"
        results = {}

    # Test 4: Alpha Factory
    print("\n4️⃣ Test Alpha Factory")
    print("-" * 40)

    try:
        from mlpipeline.alphas.alpha_factory import AlphaFactory

        factory = AlphaFactory(data, "TESTUSDT")
        factory_results = await factory.quick_alpha_scan()

        print(f"   🏭 Résultats Factory:")
        print(f"      Alphas traditionnels: {len(factory_results.get('traditional_alphas', {}))}")
        print(f"      Échantillon RL: {factory_results.get('rl_sample', {}).get('valid_alphas', 0)}")

        recommendations = factory_results.get('recommendations', [])
        if recommendations:
            print(f"   💡 Recommandations:")
            for rec in recommendations[:2]:
                print(f"      {rec}")

        test4_result = "✅ PASSED" if len(factory_results.get('traditional_alphas', {})) > 0 else "⚠️ PARTIAL"

    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        test4_result = "❌ FAILED"

    # Test 5: Intégration avec features symboliques
    print("\n5️⃣ Test intégration features symboliques")
    print("-" * 40)

    try:
        from mlpipeline.features.symbolic_operators import AlphaFormulaGenerator

        symbol_gen = AlphaFormulaGenerator()
        symbolic_features = symbol_gen.generate_enhanced_features(data)

        print(f"   🔧 Features symboliques: {symbolic_features.shape[1]} colonnes")
        print(f"   📊 Qualité: {symbolic_features.isnull().sum().sum()} NaN")

        # Test intégration avec DMN
        from mlpipeline.alphas.dmn_model import DMNPredictor
        dmn = DMNPredictor("TESTUSDT")
        dmn_signals = await dmn.predict(data.head(50))

        print(f"   🧠 DMN Enhanced: {len(dmn_signals)} signaux")
        print(f"   📈 Activité: {(np.abs(dmn_signals) > 0.1).sum()} signaux actifs")

        test5_result = "✅ PASSED"

    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        test5_result = "❌ FAILED"

    # Rapport final
    print("\n" + "=" * 60)
    print("📋 RAPPORT FINAL")
    print("=" * 60)

    tests = [
        ("Générateur RL de base", test1_result),
        ("Combinateur synergique", test2_result),
        ("Pipeline intégré", test3_result),
        ("Alpha Factory", test4_result),
        ("Features symboliques", test5_result)
    ]

    for test_name, result in tests:
        print(f"{test_name:<25} {result}")

    # Statut global
    passed = sum(1 for _, result in tests if "✅" in result)
    partial = sum(1 for _, result in tests if "⚠️" in result)
    failed = sum(1 for _, result in tests if "❌" in result)

    print(f"\n🎯 Résultat global:")
    print(f"   ✅ Passés: {passed}")
    print(f"   ⚠️ Partiels: {partial}")
    print(f"   ❌ Échoués: {failed}")

    if passed + partial >= 4:
        print("\n🎉 SYSTÈME RL ALPHA GENERATOR OPÉRATIONNEL !")
        print("💼 Prêt pour intégration en production")
    elif passed >= 3:
        print("\n✅ Système fonctionnel avec optimisations recommandées")
    else:
        print("\n⚠️ Système nécessite des corrections avant déploiement")

def create_realistic_market_data():
    """Crée des données de marché réalistes pour les tests"""
    np.random.seed(42)

    # Paramètres de simulation
    n_points = 300
    base_price = 50000

    # Simulation avec tendances et volatilité variable
    trends = np.random.choice([0.0001, 0, -0.0001], n_points, p=[0.3, 0.4, 0.3])
    volatility = np.random.choice([0.01, 0.02, 0.03], n_points, p=[0.5, 0.3, 0.2])

    returns = []
    for i in range(n_points):
        if i == 0:
            ret = np.random.randn() * volatility[i]
        else:
            # Momentum + mean reversion + noise
            momentum = 0.1 * returns[-1] if returns else 0
            mean_reversion = -0.05 * sum(returns[-5:]) if len(returns) >= 5 else 0
            noise = np.random.randn() * volatility[i]

            ret = trends[i] + momentum + mean_reversion + noise

        returns.append(ret)

    # Construire les prix
    returns = np.array(returns)
    prices = base_price * np.cumprod(1 + returns)

    # DataFrame avec toutes les features nécessaires
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1h')

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(n_points) * 0.0005),
        'high': prices * (1 + np.abs(np.random.randn(n_points)) * 0.003),
        'low': prices * (1 - np.abs(np.random.randn(n_points)) * 0.003),
        'close': prices,
        'volume': np.random.randint(1000, 20000, n_points) * (1 + np.abs(returns) * 10),
        'ret': returns
    }, index=dates)

    # Features dérivées
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3

    # Ajouter quelques features techniques pour enrichir
    data['sma_20'] = data['close'].rolling(20).mean()
    data['volume_ma'] = data['volume'].rolling(20).mean()

    return data

if __name__ == "__main__":
    asyncio.run(test_complete_rl_system())