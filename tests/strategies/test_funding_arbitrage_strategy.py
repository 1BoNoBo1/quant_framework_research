"""
Tests for Funding Arbitrage Strategy
===================================

Suite de tests complète pour la stratégie d'arbitrage de funding rate.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from decimal import Decimal

from qframe.strategies.research.funding_arbitrage_strategy import (
    FundingArbitrageStrategy,
    FundingArbitrageConfig,
    FundingRateCalculator,
    MLFundingPredictor
)
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence


class TestFundingArbitrageConfig:
    """Tests de configuration pour Funding Arbitrage Strategy"""

    def test_default_configuration(self):
        """Test la configuration par défaut"""
        config = FundingArbitrageConfig()

        assert config.funding_threshold == 0.001
        assert config.prediction_window == 24
        assert config.max_position_size == 0.5
        assert config.risk_aversion == 2.0
        assert config.use_ml_prediction == True
        assert config.funding_interval_hours == 8
        assert config.basis_risk_limit == 0.02
        assert config.position_size == 0.1

    def test_custom_configuration(self):
        """Test une configuration personnalisée"""
        config = FundingArbitrageConfig(
            funding_threshold=0.002,
            prediction_window=48,
            max_position_size=0.3,
            use_ml_prediction=False
        )

        assert config.funding_threshold == 0.002
        assert config.prediction_window == 48
        assert config.max_position_size == 0.3
        assert config.use_ml_prediction == False

    def test_config_validation(self):
        """Test la validation des paramètres de configuration"""
        # Test valeurs positives
        config = FundingArbitrageConfig(
            funding_threshold=0.005,
            max_position_size=0.8
        )
        assert config.funding_threshold > 0
        assert config.max_position_size > 0


class TestFundingRateCalculator:
    """Tests pour le calculateur de funding rate"""

    @pytest.fixture
    def sample_funding_data(self):
        """Données de funding pour les tests"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='8h')

        # Simuler des funding rates typiques (±0.5%)
        funding_rates = np.random.normal(0.0001, 0.002, 100)  # Moyenne 0.01%, std 0.2%

        data = pd.DataFrame({
            'timestamp': dates,
            'funding_rate': funding_rates,
            'open_interest': np.random.uniform(1000000, 5000000, 100),
            'basis': np.random.normal(0, 0.001, 100)  # Basis spread
        })
        data.set_index('timestamp', inplace=True)
        return data

    def test_funding_calculator_initialization(self):
        """Test l'initialisation du calculateur"""
        calculator = FundingRateCalculator()
        assert hasattr(calculator, 'calculate_funding_rate')

    def test_funding_rate_calculation(self, sample_funding_data):
        """Test le calcul du funding rate"""
        calculator = FundingRateCalculator()

        # Test avec des données réelles
        rate = calculator.calculate_funding_rate(
            index_price=50000.0,
            mark_price=50100.0,
            interest_rate=0.0001,  # 0.01% daily
            premium_window=8  # 8 heures
        )

        assert isinstance(rate, float)
        assert -0.01 <= rate <= 0.01  # Valeurs réalistes

    def test_premium_calculation(self):
        """Test le calcul de la prime"""
        calculator = FundingRateCalculator()

        premium = calculator.calculate_premium(
            futures_price=50100.0,
            spot_price=50000.0
        )

        expected_premium = (50100.0 - 50000.0) / 50000.0
        assert abs(premium - expected_premium) < 1e-6

    def test_funding_rate_statistics(self, sample_funding_data):
        """Test les statistiques des funding rates"""
        calculator = FundingRateCalculator()

        stats = calculator.calculate_funding_statistics(
            sample_funding_data['funding_rate']
        )

        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'percentile_95' in stats
        assert 'percentile_5' in stats

        # Vérifier les valeurs
        assert -0.1 <= stats['mean'] <= 0.1
        assert stats['std'] >= 0


class TestMLFundingPredictor:
    """Tests pour le prédicteur ML des funding rates"""

    @pytest.fixture
    def sample_training_data(self):
        """Données d'entraînement pour le ML"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='8h')

        data = pd.DataFrame({
            'timestamp': dates,
            'funding_rate': np.random.normal(0.0001, 0.002, 200),
            'volume': np.random.uniform(100000, 1000000, 200),
            'open_interest': np.random.uniform(1000000, 5000000, 200),
            'basis': np.random.normal(0, 0.001, 200),
            'volatility': np.random.uniform(0.01, 0.1, 200)
        })
        data.set_index('timestamp', inplace=True)
        return data

    def test_ml_predictor_initialization(self):
        """Test l'initialisation du prédicteur ML"""
        predictor = MLFundingPredictor()
        assert hasattr(predictor, 'model')
        assert hasattr(predictor, 'scaler')
        assert hasattr(predictor, 'feature_columns')

    def test_feature_engineering(self, sample_training_data):
        """Test l'engineering des features"""
        predictor = MLFundingPredictor()

        features = predictor.engineer_features(sample_training_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) <= len(sample_training_data)

        # Vérifier que les features incluent des lags et moyennes mobiles
        expected_features = ['volume_lag1', 'basis_ma_24', 'volatility_std_24']
        for feature in expected_features:
            if feature in features.columns:
                assert features[feature].notna().any()

    def test_model_training(self, sample_training_data):
        """Test l'entraînement du modèle"""
        predictor = MLFundingPredictor()

        # Préparer les données
        features = predictor.engineer_features(sample_training_data)
        target = sample_training_data['funding_rate'].shift(-1).dropna()

        # Aligner les données
        min_len = min(len(features), len(target))
        X = features.iloc[:min_len]
        y = target.iloc[:min_len]

        # Entraîner
        predictor.train(X, y)

        assert hasattr(predictor, 'model')
        assert predictor.is_trained == True

    def test_funding_prediction(self, sample_training_data):
        """Test la prédiction des funding rates"""
        predictor = MLFundingPredictor()

        # Entraîner rapidement
        features = predictor.engineer_features(sample_training_data)
        target = sample_training_data['funding_rate'].shift(-1).dropna()

        min_len = min(len(features), len(target))
        X = features.iloc[:min_len-20]  # Train set
        y = target.iloc[:min_len-20]

        predictor.train(X, y)

        # Prédire
        test_features = features.iloc[-10:]
        predictions = predictor.predict(test_features)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(test_features)
        assert all(np.isfinite(predictions))

    def test_prediction_confidence(self, sample_training_data):
        """Test le calcul de confiance des prédictions"""
        predictor = MLFundingPredictor()

        # Entraîner
        features = predictor.engineer_features(sample_training_data)
        target = sample_training_data['funding_rate'].shift(-1).dropna()

        min_len = min(len(features), len(target))
        X = features.iloc[:min_len-10]
        y = target.iloc[:min_len-10]

        predictor.train(X, y)

        # Calculer confiance
        test_features = features.iloc[-5:]
        confidence = predictor.calculate_prediction_confidence(test_features)

        assert isinstance(confidence, np.ndarray)
        assert len(confidence) == len(test_features)
        assert all(0 <= conf <= 1 for conf in confidence)


class TestFundingArbitrageStrategy:
    """Tests complets pour la stratégie Funding Arbitrage"""

    @pytest.fixture
    def config(self):
        """Configuration de test"""
        return FundingArbitrageConfig(
            funding_threshold=0.002,  # 0.2%
            prediction_window=24,
            max_position_size=0.3,
            use_ml_prediction=False  # Simplifier pour les tests
        )

    @pytest.fixture
    def sample_market_data(self):
        """Données de marché avec funding rates"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='8h')

        # Simuler des données de marché avec funding rates
        spot_prices = 50000 + np.cumsum(np.random.randn(200) * 100)
        futures_prices = spot_prices * (1 + np.random.normal(0.001, 0.0005, 200))

        data = pd.DataFrame({
            'timestamp': dates,
            'spot_price': spot_prices,
            'futures_price': futures_prices,
            'funding_rate': np.random.normal(0.0001, 0.002, 200),
            'open_interest': np.random.uniform(1000000, 5000000, 200),
            'volume': np.random.uniform(100000, 1000000, 200),
            'basis': (futures_prices - spot_prices) / spot_prices
        })
        data.set_index('timestamp', inplace=True)
        return data

    @pytest.fixture
    def strategy(self, config):
        """Stratégie initialisée pour les tests"""
        return FundingArbitrageStrategy(
            config=config,
            metrics_collector=Mock()
        )

    def test_strategy_initialization(self, strategy, config):
        """Test l'initialisation de la stratégie"""
        assert strategy.config == config
        assert hasattr(strategy, 'funding_calculator')
        assert hasattr(strategy, 'ml_predictor')

    def test_data_validation(self, strategy, sample_market_data):
        """Test la validation des données"""
        # Données complètes
        is_valid = strategy._validate_data(sample_market_data)
        assert is_valid == True

        # Données incomplètes
        incomplete_data = sample_market_data.drop('funding_rate', axis=1)
        is_valid = strategy._validate_data(incomplete_data)
        assert is_valid == False

    def test_arbitrage_opportunity_detection(self, strategy, sample_market_data):
        """Test la détection d'opportunités d'arbitrage"""
        opportunities = strategy._detect_arbitrage_opportunities(sample_market_data)

        assert isinstance(opportunities, pd.DataFrame)
        assert len(opportunities) <= len(sample_market_data)

        if len(opportunities) > 0:
            # Vérifier les colonnes importantes
            expected_cols = ['funding_rate', 'expected_profit', 'risk_adjusted_return']
            for col in expected_cols:
                if col in opportunities.columns:
                    assert opportunities[col].notna().any()

    def test_signal_generation(self, strategy, sample_market_data):
        """Test la génération de signaux"""
        signals = strategy.generate_signals(sample_market_data)

        assert isinstance(signals, list)

        # Vérifier le format des signaux
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
            assert isinstance(signal.strength, Decimal)

    def test_funding_rate_filtering(self, strategy):
        """Test le filtrage par seuil de funding rate"""
        # Funding rates en dessous du seuil
        low_funding = pd.Series([0.0001, 0.0005, 0.0008])  # < 0.2%
        high_funding = pd.Series([0.003, 0.005, 0.008])    # > 0.2%

        low_signals = strategy._filter_by_funding_threshold(low_funding)
        high_signals = strategy._filter_by_funding_threshold(high_funding)

        # Les funding rates élevés devraient générer plus de signaux
        assert len(high_signals) >= len(low_signals)

    def test_position_sizing(self, strategy, sample_market_data):
        """Test le calcul de taille de position"""
        # Données d'exemple
        funding_rate = 0.005  # 0.5%
        confidence = 0.8
        portfolio_value = 100000

        position_size = strategy._calculate_position_size(
            funding_rate=funding_rate,
            confidence=confidence,
            portfolio_value=portfolio_value,
            risk_metrics={}
        )

        assert isinstance(position_size, float)
        assert 0 <= position_size <= strategy.config.max_position_size

    def test_risk_management(self, strategy, sample_market_data):
        """Test la gestion des risques"""
        # Test du risque de base
        basis_risk = strategy._calculate_basis_risk(sample_market_data)

        assert isinstance(basis_risk, float)
        assert basis_risk >= 0

        # Test des limites de risque
        high_risk_data = sample_market_data.copy()
        high_risk_data['basis'] = 0.05  # Risque de base élevé

        signals_normal = strategy.generate_signals(sample_market_data)
        signals_high_risk = strategy.generate_signals(high_risk_data)

        # Avec un risque élevé, il devrait y avoir moins de signaux
        assert len(signals_high_risk) <= len(signals_normal)

    def test_ml_prediction_integration(self, config):
        """Test l'intégration avec la prédiction ML"""
        config.use_ml_prediction = True
        strategy = FundingArbitrageStrategy(
            config=config,
            metrics_collector=Mock()
        )

        assert hasattr(strategy, 'ml_predictor')
        assert strategy.config.use_ml_prediction == True

    def test_funding_interval_handling(self, strategy, sample_market_data):
        """Test la gestion des intervalles de funding"""
        # Vérifier que la stratégie respecte les intervalles de 8h
        next_funding_time = strategy._get_next_funding_time(
            sample_market_data.index[0]
        )

        assert isinstance(next_funding_time, datetime)

        # La différence devrait être de 8h max
        time_diff = next_funding_time - sample_market_data.index[0]
        assert time_diff <= timedelta(hours=8)

    def test_strategy_performance_metrics(self, strategy, sample_market_data):
        """Test les métriques de performance"""
        signals = strategy.generate_signals(sample_market_data)

        if signals:
            # Calculer des métriques simples
            total_signals = len(signals)
            buy_signals = len([s for s in signals if s.action == SignalAction.BUY])
            sell_signals = len([s for s in signals if s.action == SignalAction.SELL])

            assert total_signals >= 0
            assert buy_signals + sell_signals <= total_signals

    def test_error_handling(self, strategy):
        """Test la gestion d'erreurs"""
        # Données vides
        empty_data = pd.DataFrame()
        signals = strategy.generate_signals(empty_data)
        assert signals == []

        # Données corrompues
        corrupt_data = pd.DataFrame({
            'funding_rate': [np.nan, np.inf, -np.inf],
            'spot_price': [50000, 51000, 49000]
        })
        signals = strategy.generate_signals(corrupt_data)
        assert signals == []

    def test_strategy_state(self, strategy):
        """Test l'état de la stratégie"""
        state = strategy.get_strategy_state()

        assert isinstance(state, dict)
        assert 'config' in state
        assert 'funding_calculator' in state
        assert 'ml_predictor' in state

    @pytest.mark.performance
    def test_strategy_execution_speed(self, strategy, sample_market_data):
        """Test la vitesse d'exécution"""
        import time

        start_time = time.time()
        signals = strategy.generate_signals(sample_market_data)
        execution_time = time.time() - start_time

        # L'arbitrage devrait être rapide (< 2 secondes)
        assert execution_time < 2.0
        assert len(signals) >= 0

    def test_funding_rate_prediction_accuracy(self, strategy, sample_market_data):
        """Test la précision des prédictions (si ML activé)"""
        if strategy.config.use_ml_prediction:
            # Test de base pour vérifier que les prédictions sont raisonnables
            recent_data = sample_market_data.tail(50)
            predictions = strategy._predict_future_funding_rates(recent_data)

            assert isinstance(predictions, (list, np.ndarray))
            assert len(predictions) > 0

            # Les prédictions devraient être dans une plage raisonnable
            for pred in predictions:
                assert -0.01 <= pred <= 0.01  # ±1% par periode

    def test_arbitrage_profitability_calculation(self, strategy):
        """Test le calcul de profitabilité de l'arbitrage"""
        # Paramètres d'exemple
        funding_rate = 0.005
        position_size = 10000  # USD
        holding_period = 8  # heures

        expected_profit = strategy._calculate_expected_profit(
            funding_rate=funding_rate,
            position_size=position_size,
            holding_period=holding_period
        )

        assert isinstance(expected_profit, float)
        # Profit approximatif = funding_rate * position_size
        expected_approx = funding_rate * position_size
        assert abs(expected_profit - expected_approx) / expected_approx < 0.1  # 10% de tolérance