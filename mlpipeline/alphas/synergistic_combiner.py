"""
Système de Combinaison Synergique d'Alphas
Basé sur "Synergistic Formulaic Alpha Generation for Quantitative Trading"

Optimise les poids et combinaisons d'alphas pour maximiser les performances ensemble.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from mlpipeline.alphas.rl_alpha_generator import AlphaFormula

logger = logging.getLogger(__name__)

@dataclass
class AlphaCombination:
    """Représente une combinaison optimale d'alphas"""
    alphas: List[AlphaFormula]
    weights: np.ndarray
    combined_ic: float
    combined_rank_ic: float
    sharpe_ratio: float
    diversification_score: float
    total_score: float

class CombinationOptimizer:
    """Optimiseur pour la combinaison synergique d'alphas"""

    def __init__(self, market_data: pd.DataFrame):
        self.data = market_data
        self.scaler = StandardScaler()

    def optimize_alpha_combination(self,
                                 alphas: List[AlphaFormula],
                                 max_alphas: int = 10) -> AlphaCombination:
        """
        Optimise la combinaison d'alphas pour maximiser les performances synergiques
        """
        if len(alphas) == 0:
            raise ValueError("Aucun alpha fourni pour optimisation")

        logger.info(f"🎯 Optimisation combinaison de {len(alphas)} alphas")

        # 1. Filtrer les alphas de qualité
        valid_alphas = [a for a in alphas if a.ic > 0.01]
        if len(valid_alphas) == 0:
            logger.warning("Aucun alpha valide trouvé")
            return self._create_empty_combination()

        # 2. Limiter le nombre d'alphas
        top_alphas = sorted(valid_alphas, key=lambda x: x.ic, reverse=True)[:max_alphas]

        # 3. Évaluer les signaux alpha
        alpha_signals = self._evaluate_alphas(top_alphas)
        if alpha_signals.empty:
            return self._create_empty_combination()

        # 4. Optimiser les poids
        optimal_weights = self._optimize_weights(alpha_signals)

        # 5. Calculer les métriques de la combinaison
        combined_signal = (alpha_signals * optimal_weights).sum(axis=1)
        metrics = self._calculate_combination_metrics(combined_signal, alpha_signals)

        # 6. Score de diversification
        diversification = self._calculate_diversification_score(alpha_signals)

        # 7. Score total
        total_score = (
            0.4 * metrics['ic'] +
            0.3 * metrics['rank_ic'] +
            0.2 * metrics['sharpe'] +
            0.1 * diversification
        )

        combination = AlphaCombination(
            alphas=top_alphas,
            weights=optimal_weights,
            combined_ic=metrics['ic'],
            combined_rank_ic=metrics['rank_ic'],
            sharpe_ratio=metrics['sharpe'],
            diversification_score=diversification,
            total_score=total_score
        )

        logger.info(f"✅ Combinaison optimisée - IC: {metrics['ic']:.4f}, Score: {total_score:.4f}")
        return combination

    def _evaluate_alphas(self, alphas: List[AlphaFormula]) -> pd.DataFrame:
        """Évalue les signaux pour chaque alpha"""
        signals_dict = {}

        for i, alpha in enumerate(alphas):
            try:
                signal = self._generate_alpha_signal(alpha)
                if signal is not None and len(signal) > 0:
                    signals_dict[f'alpha_{i}'] = signal
            except Exception as e:
                logger.debug(f"Erreur évaluation alpha {i}: {e}")

        if not signals_dict:
            return pd.DataFrame()

        # Créer DataFrame avec alignement des indices
        signals_df = pd.DataFrame(signals_dict)

        # Normaliser les signaux
        for col in signals_df.columns:
            signals_df[col] = self._normalize_signal(signals_df[col])

        return signals_df.fillna(0)

    def _generate_alpha_signal(self, alpha: AlphaFormula) -> Optional[pd.Series]:
        """Génère le signal pour un alpha donné"""
        try:
            # Simuler la génération de signal basée sur la formule
            # En production, ceci utiliserait le vrai évaluateur de formules

            from mlpipeline.features.symbolic_operators import SymbolicOperators
            ops = SymbolicOperators()

            # Analyser la formule pour déterminer le type de signal
            formula_lower = alpha.formula.lower()

            if 'cs_rank' in formula_lower and 'volume' in self.data.columns:
                return ops.cs_rank(self.data['volume'])

            elif 'sign' in formula_lower and 'close' in self.data.columns:
                returns = self.data['close'].pct_change().fillna(0)
                return ops.sign(returns)

            elif 'delta' in formula_lower and 'close' in self.data.columns:
                return ops.delta(self.data['close'], 5)

            elif 'corr' in formula_lower:
                if 'open' in self.data.columns and 'volume' in self.data.columns:
                    return -1 * self.data['open'].rolling(10).corr(self.data['volume'])

            else:
                # Signal par défaut basé sur l'IC de l'alpha
                if 'close' in self.data.columns and 'volume' in self.data.columns:
                    base_signal = self.data['close'].rolling(10).corr(self.data['volume'])
                    # Moduler par l'IC de l'alpha
                    return base_signal * alpha.ic

            return None

        except Exception as e:
            logger.debug(f"Erreur génération signal: {e}")
            return None

    def _normalize_signal(self, signal: pd.Series) -> pd.Series:
        """Normalise un signal"""
        try:
            # Z-score normalization
            mean_val = signal.mean()
            std_val = signal.std()

            if std_val > 1e-8:
                normalized = (signal - mean_val) / std_val
                # Clipper pour éviter les outliers extrêmes
                return np.clip(normalized, -3, 3)
            else:
                return pd.Series(0, index=signal.index)

        except Exception:
            return signal.fillna(0)

    def _optimize_weights(self, alpha_signals: pd.DataFrame) -> np.ndarray:
        """Optimise les poids des alphas pour maximiser l'IC combiné"""
        n_alphas = len(alpha_signals.columns)

        if n_alphas == 1:
            return np.array([1.0])

        # Fonction objectif : maximiser l'IC du signal combiné
        def objective(weights):
            try:
                # Normaliser les poids
                weights = weights / (weights.sum() + 1e-8)

                # Signal combiné
                combined = (alpha_signals * weights).sum(axis=1)

                # Calculer IC avec les returns futurs
                ic = self._calculate_ic(combined)

                # Pénalité pour la concentration (encourager la diversification)
                concentration_penalty = np.sum(weights ** 2) * 0.1

                return -(ic - concentration_penalty)  # Minimiser le négatif

            except Exception:
                return 1000  # Pénalité pour erreur

        # Contraintes : poids positifs et somme = 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        ]

        bounds = [(0, 1) for _ in range(n_alphas)]

        # Poids initiaux uniformes
        x0 = np.ones(n_alphas) / n_alphas

        # Optimisation
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )

            if result.success:
                optimal_weights = result.x / result.x.sum()  # Re-normaliser
                return optimal_weights
            else:
                logger.warning("Optimisation échouée, utilisation poids uniformes")
                return np.ones(n_alphas) / n_alphas

        except Exception as e:
            logger.warning(f"Erreur optimisation: {e}")
            return np.ones(n_alphas) / n_alphas

    def _calculate_ic(self, signal: pd.Series) -> float:
        """Calcule l'Information Coefficient"""
        try:
            if 'ret' in self.data.columns:
                future_returns = self.data['ret'].shift(-1).fillna(0)
                aligned_signal = signal.reindex(future_returns.index).fillna(0)

                correlation = aligned_signal.corr(future_returns)
                return correlation if not np.isnan(correlation) else 0.0

            elif 'close' in self.data.columns:
                price_changes = self.data['close'].pct_change().shift(-1).fillna(0)
                aligned_signal = signal.reindex(price_changes.index).fillna(0)

                correlation = aligned_signal.corr(price_changes)
                return correlation if not np.isnan(correlation) else 0.0

            return 0.0

        except Exception:
            return 0.0

    def _calculate_combination_metrics(self, combined_signal: pd.Series, alpha_signals: pd.DataFrame) -> Dict:
        """Calcule les métriques de la combinaison"""
        try:
            # IC
            ic = self._calculate_ic(combined_signal)

            # Rank IC (corrélation de Spearman)
            rank_ic = self._calculate_rank_ic(combined_signal)

            # Sharpe approximé
            sharpe = self._calculate_sharpe(combined_signal)

            return {
                'ic': ic,
                'rank_ic': rank_ic,
                'sharpe': sharpe
            }

        except Exception as e:
            logger.debug(f"Erreur calcul métriques: {e}")
            return {'ic': 0.0, 'rank_ic': 0.0, 'sharpe': 0.0}

    def _calculate_rank_ic(self, signal: pd.Series) -> float:
        """Calcule le Rank IC (corrélation de Spearman)"""
        try:
            if 'ret' in self.data.columns:
                future_returns = self.data['ret'].shift(-1).fillna(0)
                aligned_signal = signal.reindex(future_returns.index).fillna(0)

                # Corrélation de rang
                correlation = aligned_signal.corr(future_returns, method='spearman')
                return correlation if not np.isnan(correlation) else 0.0

            return 0.0

        except Exception:
            return 0.0

    def _calculate_sharpe(self, signal: pd.Series) -> float:
        """Calcule un Sharpe ratio approximé"""
        try:
            if 'ret' in self.data.columns:
                future_returns = self.data['ret'].shift(-1).fillna(0)
                aligned_signal = signal.reindex(future_returns.index).fillna(0)

                # Strategy returns (signal * future returns)
                strategy_returns = aligned_signal * future_returns

                mean_return = strategy_returns.mean()
                std_return = strategy_returns.std()

                if std_return > 1e-8:
                    return mean_return / std_return * np.sqrt(252)  # Annualisé
                else:
                    return 0.0

            return 0.0

        except Exception:
            return 0.0

    def _calculate_diversification_score(self, alpha_signals: pd.DataFrame) -> float:
        """Calcule un score de diversification"""
        try:
            if len(alpha_signals.columns) < 2:
                return 0.0

            # Matrice de corrélation
            corr_matrix = alpha_signals.corr().abs()

            # Score basé sur les corrélations moyennes (plus faible = meilleur)
            # Exclure la diagonale
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_correlation = corr_matrix.values[mask].mean()

            # Convertir en score (1 - corrélation moyenne)
            diversification_score = max(0, 1 - avg_correlation)

            return diversification_score

        except Exception:
            return 0.0

    def _create_empty_combination(self) -> AlphaCombination:
        """Crée une combinaison vide en cas d'échec"""
        return AlphaCombination(
            alphas=[],
            weights=np.array([]),
            combined_ic=0.0,
            combined_rank_ic=0.0,
            sharpe_ratio=0.0,
            diversification_score=0.0,
            total_score=0.0
        )

class SynergisticAlphaEngine:
    """Engine principal pour la génération et combinaison synergique d'alphas"""

    def __init__(self, market_data: pd.DataFrame):
        self.data = market_data
        self.combiner = CombinationOptimizer(market_data)
        self.alpha_history: List[AlphaCombination] = []

    def generate_synergistic_alphas(self,
                                  generated_alphas: List[AlphaFormula],
                                  max_combinations: int = 5) -> List[AlphaCombination]:
        """
        Génère des combinaisons synergiques optimales d'alphas
        """
        logger.info(f"🔬 Génération de {max_combinations} combinaisons synergiques")

        combinations = []

        for i in range(max_combinations):
            try:
                # Sélectionner un sous-ensemble d'alphas pour cette combinaison
                subset_size = min(10, len(generated_alphas))
                if subset_size < 2:
                    break

                # Sélection aléatoire avec biais vers les meilleurs alphas
                if i == 0:
                    # Premier : prendre les meilleurs
                    alpha_subset = sorted(generated_alphas, key=lambda x: x.ic, reverse=True)[:subset_size]
                else:
                    # Autres : mélange aléatoire pondéré
                    weights = [a.ic + 0.01 for a in generated_alphas]  # +0.01 pour éviter 0
                    alpha_subset = np.random.choice(
                        generated_alphas,
                        size=min(subset_size, len(generated_alphas)),
                        replace=False,
                        p=np.array(weights) / sum(weights)
                    ).tolist()

                # Optimiser la combinaison
                combination = self.combiner.optimize_alpha_combination(alpha_subset)

                if combination.total_score > 0.01:  # Seuil minimum
                    combinations.append(combination)
                    logger.info(f"✅ Combinaison {i+1}: Score={combination.total_score:.4f}")

            except Exception as e:
                logger.debug(f"Erreur combinaison {i+1}: {e}")

        # Trier par score total
        combinations.sort(key=lambda x: x.total_score, reverse=True)

        # Ajouter à l'historique
        self.alpha_history.extend(combinations)

        logger.info(f"🎯 {len(combinations)} combinaisons synergiques générées")
        return combinations

    def get_best_combination(self) -> Optional[AlphaCombination]:
        """Retourne la meilleure combinaison générée"""
        if not self.alpha_history:
            return None

        return max(self.alpha_history, key=lambda x: x.total_score)

    def get_combination_stats(self) -> Dict:
        """Retourne les statistiques des combinaisons"""
        if not self.alpha_history:
            return {"total": 0}

        scores = [c.total_score for c in self.alpha_history]
        ics = [c.combined_ic for c in self.alpha_history]

        return {
            "total_combinations": len(self.alpha_history),
            "mean_score": np.mean(scores),
            "best_score": max(scores),
            "mean_ic": np.mean(ics),
            "best_ic": max(ics),
            "avg_alphas_per_combination": np.mean([len(c.alphas) for c in self.alpha_history])
        }

def test_synergistic_combiner():
    """Test du système de combinaison synergique"""
    print("🧪 Test du combinateur synergique")

    # Données de test
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')

    data = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.cumsum(np.random.randn(200) * 100),
        'high': 50100 + np.cumsum(np.random.randn(200) * 100),
        'low': 49900 + np.cumsum(np.random.randn(200) * 100),
        'close': 50000 + np.cumsum(np.random.randn(200) * 100),
        'volume': np.random.randint(1000, 10000, 200),
        'ret': np.random.randn(200) * 0.02
    }, index=dates)

    # Créer des alphas de test
    test_alphas = [
        AlphaFormula("cs_rank(volume)", ["cs_rank"], 0.05, 0.04, 2, 1),
        AlphaFormula("sign(returns)", ["sign"], 0.03, 0.025, 1, 1),
        AlphaFormula("delta(close, 5)", ["delta"], 0.04, 0.035, 2, 1),
        AlphaFormula("corr(open, volume)", ["corr"], 0.06, 0.05, 3, 1),
    ]

    # Test du combinateur
    engine = SynergisticAlphaEngine(data)
    combinations = engine.generate_synergistic_alphas(test_alphas, 3)

    print(f"✅ Combinaisons générées: {len(combinations)}")

    if combinations:
        best = combinations[0]
        print(f"🏆 Meilleure combinaison:")
        print(f"   Score total: {best.total_score:.4f}")
        print(f"   IC combiné: {best.combined_ic:.4f}")
        print(f"   Sharpe: {best.sharpe_ratio:.4f}")
        print(f"   Diversification: {best.diversification_score:.4f}")
        print(f"   Alphas: {len(best.alphas)}")

    # Stats
    stats = engine.get_combination_stats()
    print(f"📊 Stats: {stats}")

    return engine

if __name__ == "__main__":
    test_synergistic_combiner()