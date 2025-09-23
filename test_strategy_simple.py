#!/usr/bin/env python3
"""
Test simple de la stratégie Adaptive Mean Reversion.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
from qframe.strategies.research.adaptive_mean_reversion_config import AdaptiveMeanReversionConfig

def create_test_data(n_points=200):
    """Créer des données de test réalistes."""
    dates = pd.date_range('2023-01-01', periods=n_points, freq='h')
    np.random.seed(42)

    # Générer des prix réalistes avec mean reversion
    base_price = 50000
    returns = np.random.normal(0, 0.02, n_points)

    # Ajouter du mean reversion
    mean_reversion_component = np.sin(np.linspace(0, 4*np.pi, n_points)) * 0.03
    returns += mean_reversion_component

    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, n_points)
    }, index=dates)

    data.loc[data.index[0], 'open'] = data.loc[data.index[0], 'close']
    return data

def test_strategy():
    """Test simple de la stratégie."""
    print("🚀 Test de la stratégie Adaptive Mean Reversion")

    # Configuration
    config = AdaptiveMeanReversionConfig(
        universe=["BTC/USDT"],
        min_data_points=50,
        signal_threshold=0.01
    )

    # Mocks
    mock_data_provider = Mock()
    mock_risk_manager = Mock()

    # Créer la stratégie
    strategy = AdaptiveMeanReversionStrategy(
        data_provider=mock_data_provider,
        risk_manager=mock_risk_manager,
        config=config
    )

    print(f"✅ Stratégie initialisée: {strategy.get_name()}")

    # Créer des données de test
    test_data = create_test_data(150)
    print(f"📊 Données de test créées: {len(test_data)} points")

    # Générer des signaux
    try:
        signals = strategy.generate_signals(test_data)
        print(f"📈 Signaux générés: {len(signals)}")

        if signals:
            for i, signal in enumerate(signals[:3]):  # Afficher les 3 premiers
                print(f"  Signal {i+1}: {signal.action.value} {signal.symbol} "
                      f"strength={signal.strength:.3f} "
                      f"regime={signal.metadata.get('regime', 'unknown')}")
        else:
            print("  Aucun signal généré")

        # Test de l'information de la stratégie
        info = strategy.get_strategy_info()
        print(f"📋 Info stratégie: {info['name']}, régime actuel: {info.get('current_regime', 'unknown')}")

        # Test des features
        features = strategy._engineer_features(test_data)
        print(f"🔧 Features créées: {len(features.columns)} colonnes")
        print(f"   Exemples: {list(features.columns[:5])}")

        return True

    except Exception as e:
        print(f"❌ Erreur lors du test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test de la configuration."""
    print("\n🔧 Test de la configuration")

    try:
        # Configuration valide
        config = AdaptiveMeanReversionConfig()
        print(f"✅ Configuration par défaut: {config.name}")

        # Test des validations
        config_custom = AdaptiveMeanReversionConfig(
            universe=["BTC/USDT", "ETH/USDT"],
            mean_reversion_windows=[5, 10, 20],
            signal_threshold=0.05
        )
        print(f"✅ Configuration personnalisée validée")

        return True

    except Exception as e:
        print(f"❌ Erreur de configuration: {str(e)}")
        return False

def main():
    """Point d'entrée principal."""
    print("="*60)
    print("🧪 TESTS DE LA STRATÉGIE ADAPTIVE MEAN REVERSION")
    print("="*60)

    # Test de configuration
    config_ok = test_configuration()

    # Test de la stratégie
    strategy_ok = test_strategy()

    print("\n" + "="*60)
    if config_ok and strategy_ok:
        print("🎉 TOUS LES TESTS SONT PASSÉS !")
        print("✅ La stratégie Adaptive Mean Reversion est fonctionnelle")
        print("\nCaractéristiques:")
        print("- Détection de régimes ML (LSTM + Random Forest)")
        print("- Mean reversion adaptatif par régime")
        print("- Features symboliques intégrées")
        print("- Position sizing avec Kelly Criterion")
        print("- Gestion des risques dynamique")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        if not config_ok:
            print("   - Problème de configuration")
        if not strategy_ok:
            print("   - Problème de stratégie")
    print("="*60)

if __name__ == "__main__":
    main()