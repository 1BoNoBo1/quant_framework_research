#!/usr/bin/env python3
"""
🔧 Test Stratégies - Version Corrigée
===================================

Test intelligent qui s'adapte aux différentes signatures des stratégies.
"""

import asyncio
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def generate_synthetic_data(symbol: str = "BTC/USD", days: int = 30) -> pd.DataFrame:
    """Génère des données OHLCV synthétiques réalistes."""

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')  # Corrigé: '1h' au lieu de '1H'

    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, len(dates))

    trend = np.linspace(0, 0.1, len(dates))
    cycle = 0.05 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7))
    returns = returns + trend / len(dates) + cycle / len(dates)

    prices = 45000 * np.cumprod(1 + returns)

    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        volatility = max(0.005, 0.01 + 0.01 * np.sin(i / 100))

        high_offset = np.random.uniform(0, volatility)
        low_offset = -np.random.uniform(0, volatility)

        open_price = price * (1 + np.random.uniform(-volatility/2, volatility/2))
        high_price = price * (1 + high_offset)
        low_price = price * (1 + low_offset)
        close_price = price

        base_volume = 100 + 50 * np.sin(i / 24)
        volume = max(10, base_volume + np.random.uniform(-20, 50))

        data.append({
            'timestamp': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2),
            'symbol': symbol
        })

    return pd.DataFrame(data)

class MockDataProvider:
    """Mock DataProvider pour tests."""

    async def fetch_ohlcv(self, symbol, timeframe, limit=1000, start_time=None, end_time=None):
        return generate_synthetic_data(symbol, days=30)

    async def fetch_latest_price(self, symbol):
        return 45000.0

class MockRiskManager:
    """Mock RiskManager pour tests."""

    def calculate_position_size(self, signal, portfolio_value, current_positions):
        return 0.02 * portfolio_value

    def check_risk_limits(self, signal, current_positions):
        return True

async def test_strategy_fixed(strategy_name: str, strategy_class, config_class=None):
    """Test intelligent d'une stratégie avec gestion des signatures."""

    print(f"\n🎯 Test {strategy_name}")
    print("-" * 40)

    result = {
        "name": strategy_name,
        "instantiation": False,
        "signals_generated": 0,
        "errors": [],
        "warnings": []
    }

    try:
        # Créer l'instance selon la signature
        if strategy_name == "AdaptiveMeanReversion":
            # Signature spéciale: nécessite data_provider et risk_manager
            mock_provider = MockDataProvider()
            mock_risk = MockRiskManager()
            config = config_class() if config_class else None
            strategy = strategy_class(mock_provider, mock_risk, config)
        else:
            # Signature normale: config optionnel
            config = config_class() if config_class else None
            if config:
                strategy = strategy_class(config=config)
            else:
                strategy = strategy_class()

        result["instantiation"] = True
        print(f"   ✅ Instantiation réussie")

        # Test génération de données
        data = generate_synthetic_data("BTC/USD", days=30)
        print(f"   📊 Données générées: {len(data)} points")

        # Test génération de signaux
        try:
            signals = strategy.generate_signals(data)

            if signals and len(signals) > 0:
                result["signals_generated"] = len(signals)
                print(f"   ✅ Signaux générés: {len(signals)}")

                # Analyser premiers signaux
                if len(signals) > 0:
                    first_signal = signals[0]
                    actions = [s.action.value for s in signals[:5]]  # Premiers 5
                    print(f"   🎯 Premiers signaux: {actions}")
                    print(f"   💪 Premier strength: {first_signal.strength:.3f}")
            else:
                print(f"   ⚠️ Aucun signal généré (peut-être normal)")
                result["warnings"].append("No signals generated")

        except Exception as e:
            result["errors"].append(f"Signal generation: {e}")
            print(f"   ❌ Erreur signaux: {e}")

        # Test méthodes de base
        try:
            name = strategy.get_name()
            print(f"   📛 Nom stratégie: {name}")
        except Exception as e:
            result["warnings"].append(f"get_name error: {e}")
            print(f"   ⚠️ get_name: {e}")

        # Test features (si disponible)
        if hasattr(strategy, 'calculate_features'):
            try:
                features = strategy.calculate_features(data)
                print(f"   🔧 Features calculées: {features.shape if hasattr(features, 'shape') else len(features)}")
            except Exception as e:
                result["warnings"].append(f"features error: {e}")

    except Exception as e:
        result["errors"].append(f"Critical: {e}")
        print(f"   ❌ Erreur critique: {e}")

    return result

async def test_all_strategies_fixed():
    """Test toutes les stratégies avec gestion intelligente."""

    print("🔧 TEST STRATÉGIES - VERSION CORRIGÉE")
    print("=" * 45)
    print(f"⏱️ Début: {datetime.now().strftime('%H:%M:%S')}\n")

    # Import stratégies avec gestion d'erreurs
    strategies_info = []

    try:
        from qframe.strategies.research.adaptive_mean_reversion_strategy import (
            AdaptiveMeanReversionStrategy, AdaptiveMeanReversionConfig
        )
        strategies_info.append(("AdaptiveMeanReversion", AdaptiveMeanReversionStrategy, AdaptiveMeanReversionConfig))
    except Exception as e:
        print(f"❌ Import AdaptiveMeanReversion: {e}")

    try:
        from qframe.strategies.research.mean_reversion_strategy import (
            MeanReversionStrategy, MeanReversionConfig
        )
        strategies_info.append(("MeanReversion", MeanReversionStrategy, MeanReversionConfig))
    except Exception as e:
        print(f"❌ Import MeanReversion: {e}")

    try:
        from qframe.strategies.research.funding_arbitrage_strategy import (
            FundingArbitrageStrategy, FundingArbitrageConfig
        )
        strategies_info.append(("FundingArbitrage", FundingArbitrageStrategy, FundingArbitrageConfig))
    except Exception as e:
        print(f"❌ Import FundingArbitrage: {e}")

    try:
        from qframe.strategies.research.dmn_lstm_strategy import (
            DMNLSTMStrategy, DMNConfig
        )
        strategies_info.append(("DMN_LSTM", DMNLSTMStrategy, DMNConfig))
    except Exception as e:
        print(f"❌ Import DMN_LSTM: {e}")

    try:
        from qframe.strategies.research.rl_alpha_strategy import RLAlphaStrategy
        strategies_info.append(("RL_Alpha", RLAlphaStrategy, None))
    except Exception as e:
        print(f"❌ Import RL_Alpha: {e}")

    print(f"📦 {len(strategies_info)} stratégies à tester\n")

    # Test chaque stratégie
    results = []
    for strategy_name, strategy_class, config_class in strategies_info:
        result = await test_strategy_fixed(strategy_name, strategy_class, config_class)
        results.append(result)

    return results

def analyze_results_fixed(results: List[Dict]) -> Dict:
    """Analyse des résultats avec focus sur les stratégies fonctionnelles."""

    total = len(results)
    instantiable = sum(1 for r in results if r["instantiation"])
    with_signals = sum(1 for r in results if r["signals_generated"] > 0)
    total_signals = sum(r["signals_generated"] for r in results)

    analysis = {
        "total_strategies": total,
        "instantiable": instantiable,
        "generating_signals": with_signals,
        "total_signals": total_signals,
        "success_rate": (with_signals / total * 100) if total > 0 else 0,
        "details": {}
    }

    for result in results:
        name = result["name"]
        signals = result["signals_generated"]
        errors = len(result["errors"])
        warnings = len(result["warnings"])

        if signals > 0:
            status = "✅ FONCTIONNEL"
        elif result["instantiation"]:
            status = "⚠️ PARTIEL"
        else:
            status = "❌ PROBLÈME"

        analysis["details"][name] = {
            "status": status,
            "signals": signals,
            "errors": errors,
            "warnings": warnings
        }

    return analysis

async def main():
    """Point d'entrée principal."""

    try:
        results = await test_all_strategies_fixed()
        analysis = analyze_results_fixed(results)

        print("\n" + "=" * 45)
        print("📊 RÉSULTATS FINAUX")
        print("=" * 45)

        print(f"📈 Total: {analysis['total_strategies']}")
        print(f"🏗️ Instantiables: {analysis['instantiable']}")
        print(f"🎯 Générant signaux: {analysis['generating_signals']}")
        print(f"📊 Total signaux: {analysis['total_signals']}")
        print(f"💯 Taux réussite: {analysis['success_rate']:.1f}%")

        print(f"\n📋 DÉTAIL:")
        for name, details in analysis["details"].items():
            print(f"   {details['status']} {name}")
            print(f"      📊 {details['signals']} signaux")
            if details['errors'] > 0:
                print(f"      ❌ {details['errors']} erreur(s)")
            if details['warnings'] > 0:
                print(f"      ⚠️ {details['warnings']} warning(s)")

        print(f"\n🎯 PRIORITÉS POUR FIABILISATION:")

        # Stratégies à corriger en priorité
        problematic = [name for name, details in analysis["details"].items()
                      if details["status"] == "❌ PROBLÈME"]
        partial = [name for name, details in analysis["details"].items()
                  if details["status"] == "⚠️ PARTIEL"]
        working = [name for name, details in analysis["details"].items()
                  if details["status"] == "✅ FONCTIONNEL"]

        if working:
            print(f"✅ Stratégies fonctionnelles ({len(working)}): {', '.join(working)}")

        if partial:
            print(f"⚠️ À optimiser ({len(partial)}): {', '.join(partial)}")
            print("   → Focus: logique de génération de signaux")

        if problematic:
            print(f"❌ À corriger ({len(problematic)}): {', '.join(problematic)}")
            print("   → Focus: signatures et dépendances")

        print(f"\n⏱️ Fin: {datetime.now().strftime('%H:%M:%S')}")

        # Retourner le succès si au moins une stratégie fonctionne
        return len(working) > 0

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)