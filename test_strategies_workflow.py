#!/usr/bin/env python3
"""
🧪 Test des Workflows Complets de Stratégies
==========================================

Script pour tester toutes les stratégies disponibles avec des données
synthétiques et valider l'ensemble des composants du workflow.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
from typing import List, Dict, Any

print("🧪 TEST WORKFLOWS COMPLETS - Stratégies QFrame")
print("=" * 55)

def generate_synthetic_data(symbol: str = "BTC/USD", days: int = 100) -> pd.DataFrame:
    """Génère des données OHLCV synthétiques réalistes."""

    # Dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')

    # Prix avec random walk + tendance + volatilité variable
    np.random.seed(42)  # Reproductible
    returns = np.random.normal(0.0001, 0.02, len(dates))  # 0.01% mean, 2% volatility

    # Ajouter cycle et tendance
    trend = np.linspace(0, 0.1, len(dates))  # Légère tendance haussière
    cycle = 0.05 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7))  # Cycle hebdomadaire
    returns = returns + trend / len(dates) + cycle / len(dates)

    # Prix cumulatifs
    prices = 45000 * np.cumprod(1 + returns)  # Start à 45k

    # OHLCV avec logique réaliste
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        volatility = max(0.005, 0.01 + 0.01 * np.sin(i / 100))  # Volatilité variable

        high_offset = np.random.uniform(0, volatility)
        low_offset = -np.random.uniform(0, volatility)

        open_price = price * (1 + np.random.uniform(-volatility/2, volatility/2))
        high_price = price * (1 + high_offset)
        low_price = price * (1 + low_offset)
        close_price = price

        # Volume réaliste avec pattern
        base_volume = 100 + 50 * np.sin(i / 24)  # Cycle journalier
        volume = max(10, base_volume + np.random.uniform(-20, 50))

        data.append({
            'timestamp': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })

    df = pd.DataFrame(data)
    df['symbol'] = symbol
    return df

async def test_strategy_workflow(strategy_name: str, strategy_class, config_class=None):
    """Test workflow complet pour une stratégie."""

    print(f"\n🎯 Test {strategy_name}")
    print("-" * 40)

    results = {
        "strategy_name": strategy_name,
        "import_ok": False,
        "instantiation_ok": False,
        "signals_generated": 0,
        "errors": [],
        "performance": {}
    }

    try:
        # 1. Test instantiation
        if config_class:
            config = config_class()
            strategy = strategy_class(config=config)
        else:
            strategy = strategy_class()

        results["instantiation_ok"] = True
        print(f"   ✅ Instantiation: OK")

        # 2. Test avec données synthétiques
        print(f"   🔄 Génération données synthétiques...")
        data = generate_synthetic_data("BTC/USD", days=30)

        # 3. Test génération signaux
        print(f"   📊 Test génération signaux...")

        import time
        start_time = time.time()

        try:
            signals = strategy.generate_signals(data)

            signal_time = time.time() - start_time
            results["performance"]["signal_generation_time"] = signal_time

            if signals:
                results["signals_generated"] = len(signals)
                print(f"   ✅ Signaux générés: {len(signals)} en {signal_time:.3f}s")

                # Analyser les signaux
                actions = [s.action for s in signals]
                buy_signals = sum(1 for a in actions if a.value == 'buy')
                sell_signals = sum(1 for a in actions if a.value == 'sell')
                hold_signals = sum(1 for a in actions if a.value == 'hold')

                print(f"   📈 BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")

                # Test premier signal
                first_signal = signals[0]
                print(f"   🎯 Premier signal: {first_signal.action.value} @ {first_signal.strength:.3f}")

            else:
                print(f"   ⚠️ Aucun signal généré")

        except Exception as e:
            results["errors"].append(f"Signal generation: {e}")
            print(f"   ❌ Erreur génération signaux: {e}")

        # 4. Test méthodes de base
        try:
            name = strategy.get_name()
            print(f"   📛 Nom: {name}")

            config_dict = strategy.get_config()
            print(f"   ⚙️ Config: {len(config_dict)} paramètres")

        except Exception as e:
            results["errors"].append(f"Basic methods: {e}")
            print(f"   ⚠️ Méthodes de base: {e}")

        # 5. Test performance (optionnel)
        if hasattr(strategy, 'calculate_performance'):
            try:
                perf = strategy.calculate_performance(data, signals if 'signals' in locals() else [])
                results["performance"]["metrics"] = perf
                print(f"   📊 Performance calculée: {len(perf)} métriques")
            except Exception as e:
                print(f"   ⚠️ Performance non calculable: {e}")

    except Exception as e:
        results["errors"].append(f"General error: {e}")
        print(f"   ❌ Erreur générale: {e}")

    return results

async def test_all_strategies():
    """Test toutes les stratégies disponibles."""

    print("🚀 Démarrage tests workflows stratégies...")
    print(f"⏱️ Début: {datetime.now().strftime('%H:%M:%S')}")

    # Import stratégies
    from qframe.strategies.research.adaptive_mean_reversion_strategy import (
        AdaptiveMeanReversionStrategy,
        AdaptiveMeanReversionConfig
    )
    from qframe.strategies.research.mean_reversion_strategy import (
        MeanReversionStrategy,
        MeanReversionConfig
    )
    from qframe.strategies.research.funding_arbitrage_strategy import (
        FundingArbitrageStrategy,
        FundingArbitrageConfig
    )
    from qframe.strategies.research.dmn_lstm_strategy import (
        DMNLSTMStrategy,
        DMNConfig
    )
    from qframe.strategies.research.rl_alpha_strategy import (
        RLAlphaStrategy
    )

    # Liste des stratégies à tester
    strategies_to_test = [
        ("AdaptiveMeanReversion", AdaptiveMeanReversionStrategy, AdaptiveMeanReversionConfig),
        ("MeanReversion", MeanReversionStrategy, MeanReversionConfig),
        ("FundingArbitrage", FundingArbitrageStrategy, FundingArbitrageConfig),
        ("DMN LSTM", DMNLSTMStrategy, DMNConfig),
        ("RL Alpha", RLAlphaStrategy, None),  # Pas de config spécifique
    ]

    results = []

    for strategy_name, strategy_class, config_class in strategies_to_test:
        try:
            result = await test_strategy_workflow(strategy_name, strategy_class, config_class)
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur critique pour {strategy_name}: {e}")
            results.append({
                "strategy_name": strategy_name,
                "import_ok": False,
                "errors": [f"Critical error: {e}"]
            })

    return results

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse les résultats des tests."""

    total_strategies = len(results)
    working_strategies = sum(1 for r in results if r.get("signals_generated", 0) > 0)
    importable_strategies = sum(1 for r in results if r.get("instantiation_ok", False))

    total_signals = sum(r.get("signals_generated", 0) for r in results)

    analysis = {
        "total_strategies": total_strategies,
        "importable_strategies": importable_strategies,
        "working_strategies": working_strategies,
        "total_signals_generated": total_signals,
        "success_rate": (working_strategies / total_strategies) * 100 if total_strategies > 0 else 0,
        "strategies_status": {}
    }

    for result in results:
        name = result["strategy_name"]
        signals = result.get("signals_generated", 0)
        errors = len(result.get("errors", []))

        if signals > 0:
            status = "✅ FONCTIONNEL"
        elif result.get("instantiation_ok", False):
            status = "⚠️ PARTIEL"
        else:
            status = "❌ PROBLÈME"

        analysis["strategies_status"][name] = {
            "status": status,
            "signals": signals,
            "errors": errors
        }

    return analysis

async def main():
    """Point d'entrée principal."""

    try:
        # Tester toutes les stratégies
        results = await test_all_strategies()

        # Analyser les résultats
        analysis = analyze_results(results)

        print("\n" + "=" * 55)
        print("📊 RÉSULTATS GLOBAUX")
        print("=" * 55)

        print(f"📈 Stratégies testées: {analysis['total_strategies']}")
        print(f"✅ Importables: {analysis['importable_strategies']}")
        print(f"🎯 Fonctionnelles: {analysis['working_strategies']}")
        print(f"📊 Signaux générés total: {analysis['total_signals_generated']}")
        print(f"📈 Taux de succès: {analysis['success_rate']:.1f}%")

        print("\n📋 DÉTAIL PAR STRATÉGIE:")
        for name, status_info in analysis["strategies_status"].items():
            print(f"   {status_info['status']} {name}: {status_info['signals']} signaux")
            if status_info['errors'] > 0:
                print(f"      ⚠️ {status_info['errors']} erreur(s)")

        print("\n🔍 RECOMMANDATIONS:")

        if analysis["working_strategies"] == analysis["total_strategies"]:
            print("🎉 EXCELLENT! Toutes les stratégies fonctionnent parfaitement!")
            print("   → Prêt pour backtesting et déploiement")

        elif analysis["working_strategies"] >= analysis["total_strategies"] * 0.8:
            print("✅ TRÈS BON! La majorité des stratégies sont fonctionnelles")
            print("   → Concentrer sur fiabilisation des stratégies partielles")

        elif analysis["working_strategies"] >= analysis["total_strategies"] * 0.5:
            print("⚠️ MOYEN - Quelques stratégies nécessitent des corrections")
            print("   → Prioriser les stratégies les plus prometteuses")

        else:
            print("🔧 BESOIN TRAVAIL - Plusieurs stratégies nécessitent des corrections")
            print("   → Focus sur correction des erreurs d'import et instantiation")

        print(f"\n⏱️ Fin: {datetime.now().strftime('%H:%M:%S')}")
        print("🚀 Tests workflows terminés")

        return analysis["working_strategies"] == analysis["total_strategies"]

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        return False

if __name__ == "__main__":
    # Exécuter les tests
    success = asyncio.run(main())

    # Code de sortie pour intégration CI/CD
    sys.exit(0 if success else 1)