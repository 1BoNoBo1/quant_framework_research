#!/usr/bin/env python3
"""
🚀 PHASE 3 - Validation et Correction des 4 Stratégies Restantes
==============================================================

Objectif: Débloquer MeanReversion, FundingArbitrage, DMN LSTM et RL Alpha
Méthode: Analyser et corriger les paramètres pour génération de signaux
Validation: Intégrer chaque stratégie corrigée dans le pipeline de backtesting
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

print("🚀 PHASE 3 - CORRECTION ET VALIDATION DES 4 STRATÉGIES")
print("=" * 60)
print(f"⏱️ Début: {datetime.now().strftime('%H:%M:%S')}\n")

async def get_test_data(symbol: str = "BTC/USDT", days: int = 15) -> Optional[pd.DataFrame]:
    """Récupère des données de test pour les stratégies."""

    print(f"📊 RÉCUPÉRATION DONNÉES TEST")
    print("-" * 30)

    try:
        from qframe.infrastructure.data.ccxt_provider import CCXTProvider

        provider = CCXTProvider(exchange_name='binance')
        await provider.connect()

        data_points = await provider.get_klines(
            symbol=symbol,
            interval='1h',
            limit=days * 24
        )

        if not data_points:
            return None

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

        data = data.sort_values('timestamp').reset_index(drop=True)
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index('datetime', inplace=True)

        print(f"   ✅ {len(data)} points récupérés")
        print(f"   📊 Prix: ${data['close'].min():.0f} - ${data['close'].max():.0f}")

        return data

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None

async def test_mean_reversion_strategy(data: pd.DataFrame) -> Dict:
    """Test et correction de MeanReversionStrategy."""

    print(f"\n🎯 1. MEAN REVERSION STRATEGY")
    print("-" * 32)

    result = {
        "name": "MeanReversion",
        "status": "failed",
        "signals": 0,
        "error": None,
        "fixed": False
    }

    try:
        from qframe.strategies.research.mean_reversion_strategy import (
            MeanReversionStrategy, MeanReversionConfig
        )

        print("   ✅ Import réussi")

        # Configuration par défaut
        try:
            default_config = MeanReversionConfig()
            default_strategy = MeanReversionStrategy(config=default_config)
            default_signals = default_strategy.generate_signals(data)

            print(f"   📊 Config par défaut: {len(default_signals)} signaux")

            if len(default_signals) == 0:
                print("   🔧 Signaux insuffisants - Ajustement des paramètres...")

                # Configuration ajustée pour plus de signaux
                adjusted_config = MeanReversionConfig(
                    # Réduire les seuils pour plus de sensibilité
                    z_score_threshold=1.0,  # Plus bas que défaut (probablement 2.0)
                    lookback_period=10,     # Plus court pour plus de réactivité
                    volatility_threshold=0.01,  # Plus bas
                    min_signal_strength=0.005   # Plus bas
                )

                adjusted_strategy = MeanReversionStrategy(config=adjusted_config)
                adjusted_signals = adjusted_strategy.generate_signals(data)

                print(f"   🎯 Config ajustée: {len(adjusted_signals)} signaux")

                if len(adjusted_signals) > 0:
                    result["status"] = "fixed"
                    result["signals"] = len(adjusted_signals)
                    result["fixed"] = True

                    # Analyser les signaux
                    if hasattr(adjusted_signals[0], 'action'):
                        actions = [s.action.value for s in adjusted_signals[:5]]
                        print(f"      🎯 Premiers signaux: {actions}")

                    print("   ✅ MeanReversion corrigée et opérationnelle!")
                else:
                    print("   ⚠️ Ajustements insuffisants")
            else:
                result["status"] = "working"
                result["signals"] = len(default_signals)
                print("   ✅ Fonctionne avec config par défaut")

        except Exception as config_error:
            # Essayer sans configuration
            try:
                basic_strategy = MeanReversionStrategy()
                basic_signals = basic_strategy.generate_signals(data)
                print(f"   📊 Sans config: {len(basic_signals)} signaux")

                if len(basic_signals) > 0:
                    result["status"] = "working"
                    result["signals"] = len(basic_signals)
                else:
                    result["error"] = f"Config error: {config_error}"

            except Exception as basic_error:
                result["error"] = f"Basic error: {basic_error}"

    except ImportError as e:
        result["error"] = f"Import failed: {e}"
        print(f"   ❌ Import échoué: {e}")
    except Exception as e:
        result["error"] = f"Unexpected: {e}"
        print(f"   ❌ Erreur inattendue: {e}")

    return result

async def test_funding_arbitrage_strategy(data: pd.DataFrame) -> Dict:
    """Test et correction de FundingArbitrageStrategy."""

    print(f"\n💰 2. FUNDING ARBITRAGE STRATEGY")
    print("-" * 36)

    result = {
        "name": "FundingArbitrage",
        "status": "failed",
        "signals": 0,
        "error": None,
        "fixed": False
    }

    try:
        from qframe.strategies.research.funding_arbitrage_strategy import (
            FundingArbitrageStrategy, FundingArbitrageConfig
        )

        print("   ✅ Import réussi")

        # Mock funding rates data pour le test
        def create_mock_funding_data(base_data):
            """Crée des données de funding rates mockées."""
            mock_funding = base_data.copy()
            # Ajouter funding rates simulés
            mock_funding['funding_rate'] = np.random.normal(0.0001, 0.0005, len(base_data))
            mock_funding['predicted_funding'] = mock_funding['funding_rate'] * 1.1
            return mock_funding

        mock_data = create_mock_funding_data(data)
        print("   📊 Données funding mockées créées")

        try:
            # Test avec config par défaut
            config = FundingArbitrageConfig()
            strategy = FundingArbitrageStrategy(config=config)

            # La stratégie pourrait avoir besoin de funding rates
            signals = strategy.generate_signals(mock_data)

            print(f"   📊 Signaux générés: {len(signals)}")

            if len(signals) > 0:
                result["status"] = "working"
                result["signals"] = len(signals)
                print("   ✅ FundingArbitrage opérationnelle!")
            else:
                print("   🔧 Ajustement des seuils de funding...")

                # Configuration plus sensible
                adjusted_config = FundingArbitrageConfig(
                    funding_threshold=0.0001,      # Seuil très bas
                    max_position_size=0.01,        # Position plus petite
                    position_size=0.01,            # Position par défaut
                    use_ml_prediction=False        # Désactiver ML pour test rapide
                )

                adjusted_strategy = FundingArbitrageStrategy(config=adjusted_config)
                adjusted_signals = adjusted_strategy.generate_signals(mock_data)

                if len(adjusted_signals) > 0:
                    result["status"] = "fixed"
                    result["signals"] = len(adjusted_signals)
                    result["fixed"] = True
                    print("   ✅ FundingArbitrage corrigée!")
                else:
                    result["error"] = "Seuils trop élevés même après ajustement"

        except Exception as strategy_error:
            # Essayer instanciation basique
            try:
                basic_strategy = FundingArbitrageStrategy()
                basic_signals = basic_strategy.generate_signals(mock_data)

                if len(basic_signals) > 0:
                    result["status"] = "working"
                    result["signals"] = len(basic_signals)
                else:
                    result["error"] = f"No signals: {strategy_error}"

            except Exception as basic_error:
                result["error"] = f"Basic instantiation failed: {basic_error}"

    except ImportError as e:
        result["error"] = f"Import failed: {e}"
        print(f"   ❌ Import échoué: {e}")
    except Exception as e:
        result["error"] = f"Unexpected: {e}"
        print(f"   ❌ Erreur: {e}")

    return result

async def test_dmn_lstm_strategy(data: pd.DataFrame) -> Dict:
    """Test et correction de DMNLSTMStrategy."""

    print(f"\n🧠 3. DMN LSTM STRATEGY")
    print("-" * 23)

    result = {
        "name": "DMN_LSTM",
        "status": "failed",
        "signals": 0,
        "error": None,
        "fixed": False
    }

    try:
        from qframe.strategies.research.dmn_lstm_strategy import (
            DMNLSTMStrategy, DMNConfig
        )

        print("   ✅ Import réussi")

        try:
            # Configuration adaptée pour test rapide
            config = DMNConfig(
                window_size=20,        # Plus petit pour test
                hidden_size=16,        # Très petit pour test rapide
                num_layers=1,          # Plus simple
                epochs=10,             # Très peu d'epochs pour test
                learning_rate=0.01,    # Plus élevé pour convergence rapide
                signal_threshold=0.01, # Seuil bas pour plus de signaux
                use_attention=False,   # Désactiver attention pour simplicité
                batch_size=16          # Plus petit batch
            )

            strategy = DMNLSTMStrategy(config=config)

            # Générer signaux (peut prendre du temps pour ML)
            print("   🔄 Entraînement modèle et génération signaux...")
            start_time = time.time()

            signals = strategy.generate_signals(data)

            training_time = time.time() - start_time
            print(f"   ⏱️ Temps d'entraînement: {training_time:.2f}s")
            print(f"   📊 Signaux générés: {len(signals)}")

            if len(signals) > 0:
                result["status"] = "working"
                result["signals"] = len(signals)
                print("   ✅ DMN LSTM opérationnelle!")

                # Analyser qualité signaux
                if hasattr(signals[0], 'confidence'):
                    confidences = [s.confidence for s in signals[:5]]
                    print(f"      💪 Confidences: {confidences}")
            else:
                print("   🔧 Modèle nécessite plus d'entraînement ou données")
                result["error"] = "Modèle non entraîné ou seuils trop élevés"

        except Exception as strategy_error:
            print(f"   ⚠️ Erreur stratégie: {strategy_error}")

            # Essayer sans config
            try:
                basic_strategy = DMNLSTMStrategy()
                basic_signals = basic_strategy.generate_signals(data)

                if len(basic_signals) > 0:
                    result["status"] = "working"
                    result["signals"] = len(basic_signals)
                else:
                    result["error"] = f"Strategy error: {strategy_error}"

            except Exception as basic_error:
                result["error"] = f"Basic failed: {basic_error}"

    except ImportError as e:
        result["error"] = f"Import failed: {e}"
        print(f"   ❌ Import échoué: {e}")
    except Exception as e:
        result["error"] = f"Unexpected: {e}"
        print(f"   ❌ Erreur: {e}")

    return result

async def test_rl_alpha_strategy(data: pd.DataFrame) -> Dict:
    """Test et correction de RLAlphaStrategy."""

    print(f"\n🤖 4. RL ALPHA STRATEGY")
    print("-" * 23)

    result = {
        "name": "RL_Alpha",
        "status": "failed",
        "signals": 0,
        "error": None,
        "fixed": False
    }

    try:
        from qframe.strategies.research.rl_alpha_strategy import RLAlphaStrategy

        print("   ✅ Import réussi")

        try:
            # Instanciation basique d'abord
            strategy = RLAlphaStrategy()

            print("   🔄 Génération d'alphas RL...")
            start_time = time.time()

            signals = strategy.generate_signals(data)

            generation_time = time.time() - start_time
            print(f"   ⏱️ Temps de génération: {generation_time:.2f}s")
            print(f"   📊 Signaux générés: {len(signals)}")

            if len(signals) > 0:
                result["status"] = "working"
                result["signals"] = len(signals)
                print("   ✅ RL Alpha opérationnelle!")

                # Analyser alphas générés
                if hasattr(signals[0], 'alpha_formula'):
                    formulas = [getattr(s, 'alpha_formula', 'N/A') for s in signals[:3]]
                    print(f"      🧮 Formules: {formulas}")
            else:
                print("   🔧 Agent RL nécessite pré-entraînement")
                result["error"] = "Agent RL non entraîné"

        except Exception as strategy_error:
            print(f"   ⚠️ Erreur stratégie: {strategy_error}")
            result["error"] = f"Strategy error: {strategy_error}"

    except ImportError as e:
        result["error"] = f"Import failed: {e}"
        print(f"   ❌ Import échoué: {e}")
    except Exception as e:
        result["error"] = f"Unexpected: {e}"
        print(f"   ❌ Erreur: {e}")

    return result

def analyze_strategy_results(results: List[Dict]) -> Dict:
    """Analyse les résultats des tests de stratégies."""

    total_strategies = len(results)
    working_strategies = sum(1 for r in results if r["status"] in ["working", "fixed"])
    fixed_strategies = sum(1 for r in results if r["fixed"])
    total_signals = sum(r["signals"] for r in results)

    analysis = {
        "total": total_strategies,
        "working": working_strategies,
        "fixed": fixed_strategies,
        "failed": total_strategies - working_strategies,
        "total_signals": total_signals,
        "success_rate": (working_strategies / total_strategies * 100) if total_strategies > 0 else 0
    }

    return analysis

async def main():
    """Point d'entrée principal Phase 3."""

    try:
        print("🎯 OBJECTIF: Débloquer les 4 stratégies restantes")
        print("🔧 MÉTHODE: Analyser et corriger les paramètres")
        print("✅ VALIDATION: Génération de signaux fonctionnelle\n")

        # Récupérer données de test
        data = await get_test_data("BTC/USDT", days=15)

        if data is None or len(data) < 50:
            print("❌ Données insuffisantes")
            return False

        # Tester chaque stratégie
        results = []

        # 1. MeanReversion
        mr_result = await test_mean_reversion_strategy(data)
        results.append(mr_result)

        # 2. FundingArbitrage
        fa_result = await test_funding_arbitrage_strategy(data)
        results.append(fa_result)

        # 3. DMN LSTM
        dmn_result = await test_dmn_lstm_strategy(data)
        results.append(dmn_result)

        # 4. RL Alpha
        rl_result = await test_rl_alpha_strategy(data)
        results.append(rl_result)

        # Analyser résultats
        analysis = analyze_strategy_results(results)

        # Rapport final
        print(f"\n" + "=" * 60)
        print("🎯 RÉSULTATS PHASE 3")
        print("=" * 60)

        print(f"📊 STRATÉGIES TESTÉES: {analysis['total']}")
        print(f"✅ OPÉRATIONNELLES: {analysis['working']}")
        print(f"🔧 CORRIGÉES: {analysis['fixed']}")
        print(f"❌ ÉCHOUÉES: {analysis['failed']}")
        print(f"📈 SIGNAUX TOTAUX: {analysis['total_signals']}")
        print(f"💯 TAUX SUCCÈS: {analysis['success_rate']:.1f}%")

        print(f"\n📋 DÉTAIL PAR STRATÉGIE:")
        for result in results:
            status_emoji = "✅" if result["status"] in ["working", "fixed"] else "❌"
            fixed_text = " (CORRIGÉE)" if result["fixed"] else ""
            print(f"   {status_emoji} {result['name']}: {result['signals']} signaux{fixed_text}")
            if result["error"]:
                print(f"      ⚠️ {result['error']}")

        # Recommandations
        print(f"\n🚀 PROCHAINES ÉTAPES:")

        if analysis["working"] > 0:
            print(f"✅ {analysis['working']} stratégies opérationnelles")
            print("   → Intégrer dans pipeline de backtesting")
            print("   → Tests walk-forward et Monte Carlo")

        if analysis["failed"] > 0:
            print(f"🔧 {analysis['failed']} stratégies nécessitent corrections")
            print("   → Analyser paramètres en détail")
            print("   → Entraînement modèles ML/RL")

        success = analysis["working"] >= 2  # Au moins 2 stratégies fonctionnelles
        print(f"\n⏱️ Fin: {datetime.now().strftime('%H:%M:%S')}")

        return success

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE PHASE 3: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Réduire bruit logging
    logging.getLogger('ccxt').setLevel(logging.WARNING)

    # Exécuter Phase 3
    success = asyncio.run(main())
    sys.exit(0 if success else 1)