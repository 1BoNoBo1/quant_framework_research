"""
Data Validation Framework for QFrame
====================================

Implémente la validation rigoureuse des données selon les standards Claude Flow,
utilisant Great Expectations et des contrôles spécialisés pour données financières.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings

try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite
    from great_expectations.dataset import PandasDataset
    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False
    warnings.warn("Great Expectations not available. Basic validation will be used.")

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Résultat de validation de données."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime

class FinancialDataValidator:
    """
    Validateur de données financières avec contrôles spécialisés.

    Implémente les meilleures pratiques Data Science pour assurer
    la qualité des données OHLCV et dérivées.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_suite = self._create_financial_expectations()
        self.anomaly_thresholds = self._get_anomaly_thresholds()

    def validate_ohlcv_data(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1h"
    ) -> ValidationResult:
        """
        Validation complète des données OHLCV.

        Args:
            data: DataFrame avec colonnes OHLCV
            symbol: Symbole trading
            timeframe: Timeframe des données

        Returns:
            ValidationResult avec score et détails
        """
        logger.info(f"Validation OHLCV pour {symbol} ({timeframe}): {len(data)} points")

        errors = []
        warnings = []
        metrics = {}

        try:
            # 1. Validation de structure
            structure_valid, structure_errors = self._validate_structure(data)
            errors.extend(structure_errors)

            # 2. Validation des contraintes OHLCV
            ohlcv_valid, ohlcv_errors = self._validate_ohlcv_constraints(data)
            errors.extend(ohlcv_errors)

            # 3. Détection d'anomalies
            anomalies, anomaly_warnings = self._detect_anomalies(data, symbol)
            warnings.extend(anomaly_warnings)

            # 4. Validation temporelle
            temporal_valid, temporal_errors = self._validate_temporal_consistency(data, timeframe)
            errors.extend(temporal_errors)

            # 5. Contrôles statistiques
            stats_valid, stats_warnings = self._validate_statistical_properties(data)
            warnings.extend(stats_warnings)

            # 6. Great Expectations (si disponible)
            if GE_AVAILABLE:
                ge_valid, ge_errors = self._validate_with_great_expectations(data)
                errors.extend(ge_errors)

            # Calcul du score de qualité
            quality_score = self._calculate_quality_score(data, errors, warnings)

            # Métriques détaillées
            metrics = self._calculate_detailed_metrics(data, anomalies)

            is_valid = len(errors) == 0
            if self.strict_mode:
                is_valid = is_valid and len(warnings) == 0

            logger.info(f"Validation terminée: score={quality_score:.3f}, erreurs={len(errors)}, warnings={len(warnings)}")

            return ValidationResult(
                is_valid=is_valid,
                score=quality_score,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Erreur lors de la validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                errors=[f"Erreur critique de validation: {str(e)}"],
                warnings=[],
                metrics={},
                timestamp=datetime.now()
            )

    def _validate_structure(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validation de la structure du DataFrame."""
        errors = []

        # Colonnes requises
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            errors.append(f"Colonnes manquantes: {missing_columns}")

        # Types de données
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Colonne {col} doit être numérique")

        # Index temporel
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Index doit être de type DatetimeIndex")

        # Taille minimale
        if len(data) < 10:
            errors.append(f"Données insuffisantes: {len(data)} points (minimum: 10)")

        return len(errors) == 0, errors

    def _validate_ohlcv_constraints(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validation des contraintes spécifiques OHLCV."""
        errors = []

        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return False, ["Colonnes OHLC manquantes"]

        # High >= max(Open, Low, Close)
        high_valid = (data['high'] >= data[['open', 'low', 'close']].max(axis=1)).all()
        if not high_valid:
            invalid_count = (~(data['high'] >= data[['open', 'low', 'close']].max(axis=1))).sum()
            errors.append(f"High invalide: {invalid_count} points où High < max(Open, Low, Close)")

        # Low <= min(Open, High, Close)
        low_valid = (data['low'] <= data[['open', 'high', 'close']].min(axis=1)).all()
        if not low_valid:
            invalid_count = (~(data['low'] <= data[['open', 'high', 'close']].min(axis=1))).sum()
            errors.append(f"Low invalide: {invalid_count} points où Low > min(Open, High, Close)")

        # Prix positifs
        for col in ['open', 'high', 'low', 'close']:
            if (data[col] <= 0).any():
                invalid_count = (data[col] <= 0).sum()
                errors.append(f"Prix négatifs/zéro détectés dans {col}: {invalid_count} points")

        # Volume positif
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                invalid_count = (data['volume'] < 0).sum()
                errors.append(f"Volume négatif détecté: {invalid_count} points")

        return len(errors) == 0, errors

    def _detect_anomalies(self, data: pd.DataFrame, symbol: str) -> Tuple[List[Dict], List[str]]:
        """Détection d'anomalies dans les données financières."""
        anomalies = []
        warnings = []

        if len(data) < 20:
            return anomalies, warnings

        # Calcul des returns
        returns = data['close'].pct_change().dropna()

        # 1. Outliers de returns
        return_threshold = self.anomaly_thresholds['return_outlier']
        extreme_returns = returns[np.abs(returns) > return_threshold]

        if len(extreme_returns) > 0:
            anomalies.extend([
                {
                    'type': 'extreme_return',
                    'timestamp': idx,
                    'value': val,
                    'threshold': return_threshold
                }
                for idx, val in extreme_returns.items()
            ])
            warnings.append(f"{len(extreme_returns)} returns extrêmes détectés (>{return_threshold:.1%})")

        # 2. Gaps de prix
        price_gaps = data['open'] / data['close'].shift(1) - 1
        gap_threshold = self.anomaly_thresholds['price_gap']
        large_gaps = price_gaps[np.abs(price_gaps) > gap_threshold].dropna()

        if len(large_gaps) > 0:
            anomalies.extend([
                {
                    'type': 'price_gap',
                    'timestamp': idx,
                    'value': val,
                    'threshold': gap_threshold
                }
                for idx, val in large_gaps.items()
            ])
            warnings.append(f"{len(large_gaps)} gaps de prix détectés (>{gap_threshold:.1%})")

        # 3. Volume anormal
        if 'volume' in data.columns:
            volume_median = data['volume'].median()
            volume_outliers = data['volume'][data['volume'] > volume_median * 10]

            if len(volume_outliers) > 0:
                anomalies.extend([
                    {
                        'type': 'volume_spike',
                        'timestamp': idx,
                        'value': val,
                        'median': volume_median
                    }
                    for idx, val in volume_outliers.items()
                ])
                warnings.append(f"{len(volume_outliers)} pics de volume détectés (>10x médiane)")

        # 4. Périodes de volatilité extrême
        rolling_vol = returns.rolling(20).std()
        vol_threshold = rolling_vol.quantile(0.95)
        high_vol_periods = rolling_vol[rolling_vol > vol_threshold].dropna()

        if len(high_vol_periods) > len(data) * 0.1:
            warnings.append(f"Volatilité élevée prolongée: {len(high_vol_periods)} périodes")

        return anomalies, warnings

    def _validate_temporal_consistency(self, data: pd.DataFrame, timeframe: str) -> Tuple[bool, List[str]]:
        """Validation de la cohérence temporelle."""
        errors = []

        if len(data) < 2:
            return True, errors

        # Parsing du timeframe
        timeframe_mapping = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }

        expected_delta = timeframe_mapping.get(timeframe)
        if expected_delta is None:
            errors.append(f"Timeframe non reconnu: {timeframe}")
            return False, errors

        # Vérification des intervalles
        time_diffs = data.index.to_series().diff().dropna()

        # S'assurer que les comparaisons sont en timedelta
        if len(time_diffs) > 0:
            # Tolérance de 10% sur l'intervalle attendu
            tolerance = expected_delta * 0.1
            min_delta = expected_delta - tolerance
            max_delta = expected_delta + tolerance

            # Convertir en secondes pour la comparaison si nécessaire
            try:
                invalid_intervals = time_diffs[(time_diffs < min_delta) | (time_diffs > max_delta)]
            except TypeError:
                # Fallback en cas de problème de type
                time_diffs_seconds = time_diffs.dt.total_seconds()
                expected_seconds = expected_delta.total_seconds()
                tolerance_seconds = tolerance.total_seconds()
                min_seconds = expected_seconds - tolerance_seconds
                max_seconds = expected_seconds + tolerance_seconds
                invalid_mask = (time_diffs_seconds < min_seconds) | (time_diffs_seconds > max_seconds)
                invalid_intervals = time_diffs[invalid_mask]
        else:
            invalid_intervals = pd.Series(dtype='timedelta64[ns]')

        if len(invalid_intervals) > 0:
            errors.append(f"Intervalles temporels invalides: {len(invalid_intervals)} points")

        # Vérification des doublons temporels
        duplicated_times = data.index.duplicated()
        if duplicated_times.any():
            errors.append(f"Timestamps dupliqués: {duplicated_times.sum()} points")

        # Ordre chronologique
        if not data.index.is_monotonic_increasing:
            errors.append("Données non triées chronologiquement")

        return len(errors) == 0, errors

    def _validate_statistical_properties(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validation des propriétés statistiques."""
        warnings = []

        if len(data) < 50:
            return True, warnings

        returns = data['close'].pct_change().dropna()

        # 1. Test de normalité (returns ne doivent pas être parfaitement normaux)
        from scipy import stats
        _, p_value = stats.jarque_bera(returns)
        if p_value > 0.5:  # Trop normal pour des données financières
            warnings.append("Returns anormalement normaux (possibles données synthétiques)")

        # 2. Autocorrélation excessive
        if len(returns) > 20:
            autocorr_1 = returns.autocorr(lag=1)
            if abs(autocorr_1) > 0.3:
                warnings.append(f"Autocorrélation élevée (lag=1): {autocorr_1:.3f}")

        # 3. Variance constante
        if len(returns) > 100:
            # Test de Breusch-Pagan simple
            variance_test = self._simple_variance_test(returns)
            if variance_test['p_value'] < 0.05:
                warnings.append("Hétéroscédasticité détectée dans les returns")

        # 4. Clustering de volatilité
        rolling_vol = returns.rolling(20).std()
        if len(rolling_vol.dropna()) > 50:
            vol_autocorr = rolling_vol.autocorr(lag=1)
            if vol_autocorr > 0.7:
                warnings.append(f"Clustering de volatilité fort: {vol_autocorr:.3f}")

        return True, warnings

    def _validate_with_great_expectations(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validation avec Great Expectations."""
        if not GE_AVAILABLE:
            return True, []

        errors = []

        try:
            # Créer un dataset Great Expectations
            ge_df = PandasDataset(data)

            # Appliquer les expectations
            validation_result = ge_df.validate(expectation_suite=self.validation_suite)

            # Extraire les erreurs
            for result in validation_result.results:
                if not result.success:
                    errors.append(f"GE: {result.expectation_config.expectation_type}")

        except Exception as e:
            logger.warning(f"Erreur Great Expectations: {str(e)}")

        return len(errors) == 0, errors

    def _calculate_quality_score(
        self,
        data: pd.DataFrame,
        errors: List[str],
        warnings: List[str]
    ) -> float:
        """Calcul du score de qualité (0-1)."""

        base_score = 1.0

        # Pénalités pour erreurs (critiques)
        error_penalty = len(errors) * 0.2

        # Pénalités pour warnings (modérées)
        warning_penalty = len(warnings) * 0.05

        # Bonus pour complétude
        completeness = 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
        completeness_bonus = completeness * 0.1

        # Score final
        score = base_score - error_penalty - warning_penalty + completeness_bonus

        return max(0.0, min(1.0, score))

    def _calculate_detailed_metrics(
        self,
        data: pd.DataFrame,
        anomalies: List[Dict]
    ) -> Dict[str, Any]:
        """Calcul de métriques détaillées."""

        metrics = {
            'data_points': len(data),
            'completeness': 1 - data.isnull().sum().sum() / (len(data) * len(data.columns)),
            'anomaly_count': len(anomalies),
            'anomaly_rate': len(anomalies) / len(data) if len(data) > 0 else 0,
        }

        if len(data) > 1:
            returns = data['close'].pct_change().dropna()
            metrics.update({
                'return_volatility': returns.std(),
                'max_return': returns.max(),
                'min_return': returns.min(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            })

        if 'volume' in data.columns:
            metrics.update({
                'avg_volume': data['volume'].mean(),
                'volume_volatility': data['volume'].std() / data['volume'].mean()
            })

        return metrics

    def _create_financial_expectations(self):
        """Créer la suite d'expectations pour données financières."""
        if not GE_AVAILABLE:
            return None

        suite = ExpectationSuite(expectation_suite_name="financial_data_suite")

        # Expectations de base
        expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "close"}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "close", "min_value": 0, "max_value": None}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "volume", "min_value": 0, "max_value": None}
            }
        ]

        for exp in expectations:
            suite.add_expectation(exp)

        return suite

    def _get_anomaly_thresholds(self) -> Dict[str, float]:
        """Seuils de détection d'anomalies."""
        return {
            'return_outlier': 0.15,  # 15% return
            'price_gap': 0.10,       # 10% gap
            'volume_multiplier': 10,  # 10x volume normal
            'volatility_percentile': 0.95
        }

    def _simple_variance_test(self, returns: pd.Series) -> Dict[str, float]:
        """Test simple d'hétéroscédasticité."""

        # Régression simple de |returns| sur time
        y = np.abs(returns).values
        x = np.arange(len(y))

        # Corrélation simple
        correlation = np.corrcoef(x, y)[0, 1]

        # P-value approximative
        n = len(y)
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))

        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        return {
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

# Fonctions utilitaires
def validate_strategy_data(
    data: pd.DataFrame,
    strategy_name: str = "unknown",
    strict: bool = True
) -> ValidationResult:
    """
    Fonction simple pour valider les données d'une stratégie.

    Args:
        data: Données OHLCV
        strategy_name: Nom de la stratégie
        strict: Mode strict (warnings = erreurs)

    Returns:
        ValidationResult
    """
    validator = FinancialDataValidator(strict_mode=strict)
    return validator.validate_ohlcv_data(data, strategy_name)

def create_data_quality_report(
    validation_result: ValidationResult,
    output_path: Optional[str] = None
) -> str:
    """
    Génère un rapport de qualité des données.

    Args:
        validation_result: Résultat de validation
        output_path: Chemin de sauvegarde (optionnel)

    Returns:
        Rapport en format HTML
    """

    report_html = f"""
    <html>
    <head><title>Rapport de Qualité des Données</title></head>
    <body>
        <h1>Rapport de Qualité des Données</h1>
        <h2>Score Global: {validation_result.score:.2%}</h2>
        <h3>Statut: {'✅ VALIDE' if validation_result.is_valid else '❌ INVALIDE'}</h3>

        <h3>Erreurs ({len(validation_result.errors)}):</h3>
        <ul>
        {''.join(f'<li>{error}</li>' for error in validation_result.errors)}
        </ul>

        <h3>Avertissements ({len(validation_result.warnings)}):</h3>
        <ul>
        {''.join(f'<li>{warning}</li>' for warning in validation_result.warnings)}
        </ul>

        <h3>Métriques:</h3>
        <ul>
        {''.join(f'<li>{k}: {v}</li>' for k, v in validation_result.metrics.items())}
        </ul>

        <p>Généré le: {validation_result.timestamp}</p>
    </body>
    </html>
    """

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_html)

    return report_html

# Exemple d'usage
if __name__ == "__main__":
    # Créer des données de test
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    test_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Validation
    result = validate_strategy_data(test_data, "test_strategy")
    print(f"Validation: {result.is_valid}, Score: {result.score:.2%}")
    print(f"Erreurs: {len(result.errors)}, Warnings: {len(result.warnings)}")

    # Rapport
    report = create_data_quality_report(result)
    print("Rapport généré!")