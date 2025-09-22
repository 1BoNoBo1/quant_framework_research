"""
Tests unitaires pour les opérateurs symboliques
==============================================

Tests des opérateurs du papier de recherche et du processeur de features.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from qframe.features.symbolic_operators import (
    SymbolicOperators,
    SymbolicFeatureProcessor
)


class TestSymbolicOperators:
    """Tests des opérateurs symboliques individuels"""

    @pytest.fixture
    def sample_series(self):
        """Série de test"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        values = np.cumsum(np.random.randn(50))
        return pd.Series(values, index=dates)

    def test_sign_operator(self, sample_series):
        """Test de l'opérateur sign"""
        result = SymbolicOperators.sign(sample_series)

        # Vérifier que les valeurs sont -1, 0, ou 1
        assert all(val in [-1, 0, 1] for val in result)

        # Test avec valeurs connues
        test_series = pd.Series([-2, -0.5, 0, 0.5, 2])
        expected = pd.Series([-1, -1, 0, 1, 1])
        result = SymbolicOperators.sign(test_series)
        pd.testing.assert_series_equal(result, expected)

    def test_cs_rank_operator(self, sample_series):
        """Test de l'opérateur cs_rank"""
        result = SymbolicOperators.cs_rank(sample_series)

        # Vérifier que les rangs sont entre 0 et 1
        assert all(0 <= val <= 1 for val in result)
        assert len(result) == len(sample_series)

    def test_product_operator(self, sample_series):
        """Test de l'opérateur product"""
        # Product sur 3 périodes
        result = SymbolicOperators.product(sample_series, 2)  # t=2 means 3 values

        assert len(result) == len(sample_series)
        assert not result.isna().all()  # Au moins quelques valeurs non-NaN

        # Test manuel pour les premières valeurs
        manual_product = sample_series.iloc[0] * sample_series.iloc[1] * sample_series.iloc[2]
        assert abs(result.iloc[2] - manual_product) < 1e-10

    def test_scale_operator(self, sample_series):
        """Test de l'opérateur scale"""
        result = SymbolicOperators.scale(sample_series)

        # La somme des valeurs absolues devrait être 1 (normalisation)
        assert abs(result.abs().sum() - 1.0) < 1e-10
        assert len(result) == len(sample_series)

    def test_scale_operator_zero_sum(self):
        """Test de l'opérateur scale avec somme nulle"""
        zero_series = pd.Series([0, 0, 0])
        result = SymbolicOperators.scale(zero_series)

        # Devrait retourner des zéros
        assert all(val == 0 for val in result)

    def test_pow_op_operator(self, sample_series):
        """Test de l'opérateur pow_op"""
        # Test avec exposant simple
        result = SymbolicOperators.pow_op(sample_series, 2)

        assert len(result) == len(sample_series)
        assert all(np.isfinite(val) for val in result)

        # Test avec exposant fractionnaire
        result_sqrt = SymbolicOperators.pow_op(sample_series.abs(), 0.5)
        assert all(np.isfinite(val) for val in result_sqrt)

    def test_skew_operator(self, sample_series):
        """Test de l'opérateur skew"""
        result = SymbolicOperators.skew(sample_series, window=10)

        assert len(result) == len(sample_series)
        # Les premières valeurs peuvent être NaN
        assert not result.dropna().empty

    def test_kurt_operator(self, sample_series):
        """Test de l'opérateur kurt"""
        result = SymbolicOperators.kurt(sample_series, window=10)

        assert len(result) == len(sample_series)
        # Les premières valeurs peuvent être NaN
        assert not result.dropna().empty

    def test_ts_rank_operator(self, sample_series):
        """Test de l'opérateur ts_rank"""
        result = SymbolicOperators.ts_rank(sample_series, 5)

        assert len(result) == len(sample_series)
        # Les rangs doivent être entre 0 et 1
        valid_values = result.dropna()
        assert all(0 <= val <= 1 for val in valid_values)

    def test_delta_operator(self, sample_series):
        """Test de l'opérateur delta"""
        result = SymbolicOperators.delta(sample_series, 3)

        assert len(result) == len(sample_series)

        # Test manuel: delta(t=3) = value[i] - value[i-3]
        manual_delta = sample_series.iloc[5] - sample_series.iloc[2]
        assert abs(result.iloc[5] - manual_delta) < 1e-10

    def test_argmax_operator(self, sample_series):
        """Test de l'opérateur argmax"""
        result = SymbolicOperators.argmax(sample_series, 10)

        assert len(result) == len(sample_series)
        # Les valeurs doivent être des indices valides
        valid_values = result.dropna()
        assert all(0 <= val < 10 for val in valid_values)

    def test_argmin_operator(self, sample_series):
        """Test de l'opérateur argmin"""
        result = SymbolicOperators.argmin(sample_series, 10)

        assert len(result) == len(sample_series)
        # Les valeurs doivent être des indices valides
        valid_values = result.dropna()
        assert all(0 <= val < 10 for val in valid_values)

    def test_cond_operator(self, sample_series):
        """Test de l'opérateur cond"""
        # Test simple: si sample_series > 0, retourner 1, sinon -1
        result = SymbolicOperators.cond(sample_series, 0, 1.0, -1.0)

        assert len(result) == len(sample_series)
        assert all(val in [1.0, -1.0] for val in result)

        # Vérifier la logique
        for i, val in enumerate(sample_series):
            expected = 1.0 if val > 0 else -1.0
            assert result.iloc[i] == expected

    def test_wma_operator(self, sample_series):
        """Test de l'opérateur WMA"""
        result = SymbolicOperators.wma(sample_series, 5)

        assert len(result) == len(sample_series)
        assert not result.isna().all()

    def test_ema_operator(self, sample_series):
        """Test de l'opérateur EMA"""
        result = SymbolicOperators.ema(sample_series, 10)

        assert len(result) == len(sample_series)
        assert not result.isna().all()

        # L'EMA devrait être plus lisse que les données originales
        assert result.std() <= sample_series.std()

    def test_mad_operator(self, sample_series):
        """Test de l'opérateur MAD"""
        result = SymbolicOperators.mad(sample_series, 10)

        assert len(result) == len(sample_series)
        # MAD doit être positive
        valid_values = result.dropna()
        assert all(val >= 0 for val in valid_values)


class TestSymbolicFeatureProcessor:
    """Tests du processeur de features symboliques"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Données OHLCV de test"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="1h")

        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.cumprod(1 + returns)

        data = pd.DataFrame({
            "open": prices * (1 + np.random.normal(0, 0.001, 100)),
            "high": prices * (1 + np.random.uniform(0, 0.02, 100)),
            "low": prices * (1 - np.random.uniform(0, 0.02, 100)),
            "close": prices,
            "volume": np.random.randint(1000, 10000, 100),
        }, index=dates)

        return data

    def test_processor_initialization(self):
        """Test d'initialisation du processeur"""
        processor = SymbolicFeatureProcessor()

        assert processor.ops is not None
        assert isinstance(processor._feature_names, list)
        assert len(processor._feature_names) == 0  # Vide au début

    def test_feature_generation(self, sample_ohlcv_data):
        """Test de génération de features"""
        processor = SymbolicFeatureProcessor()
        features = processor.process(sample_ohlcv_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
        assert len(features.columns) > 0

        # Vérifier que les noms de features sont mis à jour
        feature_names = processor.get_feature_names()
        assert len(feature_names) == len(features.columns)
        assert set(feature_names) == set(features.columns)

    def test_specific_features_presence(self, sample_ohlcv_data):
        """Test de présence de features spécifiques"""
        processor = SymbolicFeatureProcessor()
        features = processor.process(sample_ohlcv_data)

        expected_features = [
            "sign_returns",
            "cs_rank_volume",
            "product_close_5",
            "scale_volume",
            "skew_returns_20",
            "kurt_returns_20",
            "ts_rank_close_10",
            "delta_close_5",
            "argmax_high_20",
            "argmin_low_20",
            "wma_close_10",
            "ema_volume_20",
            "mad_close_15",
            "cond_high_low",
            "pow_volume_half",
            "alpha_006",
            "alpha_061",
            "alpha_099"
        ]

        for feature in expected_features:
            assert feature in features.columns, f"Feature {feature} missing"

    def test_alpha_formulas(self, sample_ohlcv_data):
        """Test des formules alpha spécifiques"""
        processor = SymbolicFeatureProcessor()

        # Test Alpha 006
        alpha_006 = processor.generate_alpha_006(sample_ohlcv_data)
        assert len(alpha_006) == len(sample_ohlcv_data)
        assert not alpha_006.isna().all()

        # Test Alpha 061
        alpha_061 = processor.generate_alpha_061(sample_ohlcv_data)
        assert len(alpha_061) == len(sample_ohlcv_data)

        # Test Alpha 099
        alpha_099 = processor.generate_alpha_099(sample_ohlcv_data)
        assert len(alpha_099) == len(sample_ohlcv_data)

    def test_data_quality(self, sample_ohlcv_data):
        """Test de qualité des données générées"""
        processor = SymbolicFeatureProcessor()
        features = processor.process(sample_ohlcv_data)

        # Pas d'infinis
        assert not np.isinf(features.values).any()

        # Gestion des NaN (remplacés par 0 ou forward fill)
        nan_count = features.isna().sum().sum()
        assert nan_count == 0 or nan_count < len(features) * 0.1  # Moins de 10% de NaN

    def test_processor_interface_compliance(self, sample_ohlcv_data):
        """Test de conformité à l'interface FeatureProcessor"""
        processor = SymbolicFeatureProcessor()

        # Test de la méthode process
        result = processor.process(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)

        # Test de get_feature_names
        feature_names = processor.get_feature_names()
        assert isinstance(feature_names, list)

        # Après processing, les noms doivent être disponibles
        if len(result.columns) > 0:
            assert len(feature_names) > 0

    def test_statistics_method(self, sample_ohlcv_data):
        """Test de la méthode get_statistics"""
        processor = SymbolicFeatureProcessor()
        processor.process(sample_ohlcv_data)  # Générer les features

        stats = processor.get_statistics()

        assert isinstance(stats, dict)
        assert "total_features" in stats
        assert "operators_available" in stats
        assert "alpha_formulas" in stats
        assert "feature_categories" in stats

        assert stats["total_features"] > 0
        assert stats["alpha_formulas"] == 3

    def test_empty_data_handling(self):
        """Test de gestion des données vides"""
        processor = SymbolicFeatureProcessor()
        empty_data = pd.DataFrame()

        result = processor.process(empty_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_minimal_data_handling(self):
        """Test de gestion des données minimales"""
        processor = SymbolicFeatureProcessor()

        # Données avec seulement quelques lignes
        minimal_data = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200]
        })

        result = processor.process(minimal_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_error_handling_in_alpha_formulas(self):
        """Test de gestion d'erreurs dans les formules alpha"""
        import warnings

        processor = SymbolicFeatureProcessor()

        # Données problématiques (que des NaN)
        problematic_data = pd.DataFrame({
            "open": [np.nan, np.nan, np.nan],
            "high": [np.nan, np.nan, np.nan],
            "low": [np.nan, np.nan, np.nan],
            "close": [np.nan, np.nan, np.nan],
            "volume": [np.nan, np.nan, np.nan]
        })

        # Supprimer temporairement les warnings pandas pour ce test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
            # Ne devrait pas lever d'exception
            result = processor.process(problematic_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3