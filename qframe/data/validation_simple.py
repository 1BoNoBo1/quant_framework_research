"""
Simple Data Validation for QFrame
================================

Version simplifiée du validateur de données pour éviter les problèmes de type
et fournir une validation basique robuste.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SimpleValidationResult:
    """Résultat simplifié de validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

class SimpleFinancialValidator:
    """
    Validateur simplifié pour données financières.

    Version robuste qui évite les problèmes de type complexes.
    """

    def __init__(self):
        self.min_data_points = 10

    def validate_ohlcv_data(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> SimpleValidationResult:
        """
        Validation simplifiée des données OHLCV.

        Args:
            data: DataFrame avec colonnes OHLCV
            symbol: Symbole trading

        Returns:
            SimpleValidationResult avec validation basique
        """
        logger.info(f"Validation simple OHLCV pour {symbol}: {len(data)} points")

        errors = []
        warnings = []
        metrics = {}

        try:
            # 1. Validation de structure basique
            structure_valid, structure_errors = self._validate_basic_structure(data)
            errors.extend(structure_errors)

            # 2. Validation des contraintes OHLCV simples
            if structure_valid:
                ohlcv_valid, ohlcv_errors = self._validate_ohlcv_simple(data)
                errors.extend(ohlcv_errors)

                # 3. Calcul de métriques basiques
                metrics = self._calculate_basic_metrics(data)

            # 4. Calcul du score
            score = self._calculate_score(len(errors), len(warnings), len(data))

            return SimpleValidationResult(
                is_valid=len(errors) == 0,
                score=score,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Erreur lors de la validation simple: {str(e)}")
            return SimpleValidationResult(
                is_valid=False,
                score=0.0,
                errors=[f"Erreur de validation: {str(e)}"],
                warnings=[],
                metrics={}
            )

    def _validate_basic_structure(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validation de structure basique."""
        errors = []

        # Vérifier que c'est bien un DataFrame
        if not isinstance(data, pd.DataFrame):
            errors.append("Les données doivent être un pandas DataFrame")
            return False, errors

        # Colonnes requises
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            errors.append(f"Colonnes manquantes: {missing_columns}")

        # Types de données numériques
        for col in required_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    errors.append(f"Colonne {col} doit être numérique")

        # Taille minimale
        if len(data) < self.min_data_points:
            errors.append(f"Données insuffisantes: {len(data)} points (minimum: {self.min_data_points})")

        # Données vides
        if data.empty:
            errors.append("DataFrame vide")

        return len(errors) == 0, errors

    def _validate_ohlcv_simple(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validation simple des contraintes OHLCV."""
        errors = []

        try:
            # High >= Low
            if 'high' in data.columns and 'low' in data.columns:
                violations = (data['high'] < data['low']).sum()
                if violations > 0:
                    errors.append(f"Violations High < Low: {violations} points")

            # High >= Open, Close
            if all(col in data.columns for col in ['high', 'open', 'close']):
                high_open_violations = (data['high'] < data['open']).sum()
                high_close_violations = (data['high'] < data['close']).sum()

                if high_open_violations > 0:
                    errors.append(f"Violations High < Open: {high_open_violations} points")
                if high_close_violations > 0:
                    errors.append(f"Violations High < Close: {high_close_violations} points")

            # Low <= Open, Close
            if all(col in data.columns for col in ['low', 'open', 'close']):
                low_open_violations = (data['low'] > data['open']).sum()
                low_close_violations = (data['low'] > data['close']).sum()

                if low_open_violations > 0:
                    errors.append(f"Violations Low > Open: {low_open_violations} points")
                if low_close_violations > 0:
                    errors.append(f"Violations Low > Close: {low_close_violations} points")

            # Volume >= 0
            if 'volume' in data.columns:
                negative_volumes = (data['volume'] < 0).sum()
                if negative_volumes > 0:
                    errors.append(f"Volumes négatifs: {negative_volumes} points")

            # Valeurs nulles/infinies
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    null_count = data[col].isnull().sum()
                    inf_count = np.isinf(data[col]).sum()

                    if null_count > 0:
                        errors.append(f"Valeurs nulles en {col}: {null_count}")
                    if inf_count > 0:
                        errors.append(f"Valeurs infinies en {col}: {inf_count}")

        except Exception as e:
            errors.append(f"Erreur validation OHLCV: {str(e)}")

        return len(errors) == 0, errors

    def _calculate_basic_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcul de métriques basiques."""
        metrics = {}

        try:
            metrics['total_points'] = len(data)
            metrics['total_columns'] = len(data.columns)

            # Métriques par colonne
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        metrics[f'{col}_mean'] = float(col_data.mean())
                        metrics[f'{col}_std'] = float(col_data.std())
                        metrics[f'{col}_min'] = float(col_data.min())
                        metrics[f'{col}_max'] = float(col_data.max())
                        metrics[f'{col}_null_count'] = int(data[col].isnull().sum())

            # Métriques dérivées simples
            if all(col in data.columns for col in ['high', 'low']):
                metrics['avg_range'] = float((data['high'] - data['low']).mean())

            if 'close' in data.columns:
                close_data = data['close'].dropna()
                if len(close_data) > 1:
                    returns = close_data.pct_change().dropna()
                    if len(returns) > 0:
                        metrics['avg_return'] = float(returns.mean())
                        metrics['volatility'] = float(returns.std())

        except Exception as e:
            logger.warning(f"Erreur calcul métriques: {str(e)}")
            metrics['calculation_error'] = str(e)

        return metrics

    def _calculate_score(self, num_errors: int, num_warnings: int, data_size: int) -> float:
        """Calcul d'un score de qualité simple."""
        if num_errors > 0:
            # Pénalité pour les erreurs
            error_penalty = min(0.8, num_errors * 0.1)
            base_score = 1.0 - error_penalty
        else:
            base_score = 1.0

        # Bonus pour la taille des données
        size_bonus = min(0.1, data_size / 1000 * 0.1)

        # Pénalité légère pour les warnings
        warning_penalty = min(0.1, num_warnings * 0.02)

        final_score = max(0.0, min(1.0, base_score + size_bonus - warning_penalty))
        return final_score

def create_simple_validator() -> SimpleFinancialValidator:
    """Factory pour créer un validateur simple."""
    return SimpleFinancialValidator()