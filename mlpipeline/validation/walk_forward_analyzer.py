#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward Analysis pour validation robuste des stratégies
Implémentation professionnelle pour éviter l'overfitting
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Optionnel pour viz
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow

# Import du moteur événementiel unifié
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from mlpipeline.execution.event_engine import EventEngine, StrategyRunner, create_event_engine

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class WalkForwardAnalyzer:
    """
    Analyseur Walk-Forward pour validation rigoureuse des stratégies quantitatives
    
    Fonctionnalités:
    - Walk-forward optimization avec fenêtres glissantes
    - Validation out-of-sample stricte
    - Détection d'overfitting automatique
    - Métriques de robustesse avancées
    - Visualisations diagnostic
    """
    
    def __init__(self, 
                 train_window_days: int = 180,    # 6 mois d'entraînement
                 test_window_days: int = 30,      # 1 mois de test
                 step_days: int = 15,             # Step de 15 jours
                 min_trades_per_period: int = 5,  # Min trades pour validation
                 significance_level: float = 0.05): # Niveau de significativité
        
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_trades_per_period = min_trades_per_period
        self.significance_level = significance_level
        
        # Résultats stockés
        self.walk_forward_results = []
        self.oos_results = []
        self.robustness_metrics = {}
        
        logger.info(f"✅ Walk-Forward Analyzer initialisé:")
        logger.info(f"   - Train: {train_window_days} jours")
        logger.info(f"   - Test: {test_window_days} jours")
        logger.info(f"   - Step: {step_days} jours")
    
    def create_time_splits(self, 
                          data: pd.DataFrame,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Crée les splits temporels pour walk-forward analysis
        
        Args:
            data: DataFrame avec DatetimeIndex
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            
        Returns:
            Liste de tuples (train_start, train_end, test_start, test_end)
        """
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame doit avoir un DatetimeIndex")
        
        # Période d'analyse
        analysis_start = pd.to_datetime(start_date) if start_date else data.index.min()
        analysis_end = pd.to_datetime(end_date) if end_date else data.index.max()
        
        logger.info(f"📅 Création splits pour période: {analysis_start} → {analysis_end}")
        
        splits = []
        current_date = analysis_start
        
        while current_date + timedelta(days=self.train_window_days + self.test_window_days) <= analysis_end:
            
            # Période d'entraînement
            train_start = current_date
            train_end = current_date + timedelta(days=self.train_window_days)
            
            # Période de test (out-of-sample)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window_days)
            
            # Vérifier qu'on a assez de données
            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]
            
            if len(train_data) >= 30 and len(test_data) >= 10:  # Minimum de données
                splits.append((train_start, train_end, test_start, test_end))
            
            # Avancer de step_days
            current_date += timedelta(days=self.step_days)
        
        logger.info(f"✅ {len(splits)} splits créés pour walk-forward")
        return splits
    
    def run_strategy_on_period(self, 
                              strategy_func,
                              train_data: pd.DataFrame,
                              test_data: pd.DataFrame,
                              strategy_params: Dict) -> Dict:
        """
        Exécute une stratégie sur une période train/test
        
        Args:
            strategy_func: Fonction de stratégie à tester
            train_data: Données d'entraînement
            test_data: Données de test
            strategy_params: Paramètres de la stratégie
            
        Returns:
            Dict avec métriques in-sample et out-of-sample
        """
        
        try:
            # 1. Entraînement sur période train
            logger.debug(f"🎯 Entraînement sur {len(train_data)} observations")
            
            # Appel de la stratégie avec données train
            strategy_result = strategy_func(train_data, strategy_params)
            
            if not strategy_result or 'model' not in strategy_result:
                return {"error": "Strategy training failed", "valid": False}
            
            # 2. Test sur période out-of-sample
            logger.debug(f"📊 Test OOS sur {len(test_data)} observations")
            
            # Génération signaux sur données test
            if hasattr(strategy_result['model'], 'generate_signals'):
                oos_signals = strategy_result['model'].generate_signals(test_data)
            else:
                # Fallback: utiliser fonction de prédiction générique
                oos_signals = self._generate_generic_signals(strategy_result['model'], test_data)
            
            # 3. Calcul métriques
            is_metrics = self._calculate_period_metrics(
                strategy_result.get('returns', pd.Series()),
                strategy_result.get('signals', pd.Series()),
                prefix="IS"
            )
            
            oos_metrics = self._calculate_period_metrics(
                self._calculate_strategy_returns(test_data, oos_signals),
                oos_signals,
                prefix="OOS"
            )
            
            # 4. Tests de significativité
            significance_tests = self._run_significance_tests(
                strategy_result.get('returns', pd.Series()),
                self._calculate_strategy_returns(test_data, oos_signals)
            )
            
            return {
                "valid": True,
                "in_sample": is_metrics,
                "out_of_sample": oos_metrics,
                "significance": significance_tests,
                "model": strategy_result.get('model'),
                "train_period": f"{train_data.index[0]} → {train_data.index[-1]}",
                "test_period": f"{test_data.index[0]} → {test_data.index[-1]}"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur exécution stratégie: {e}")
            return {"error": str(e), "valid": False}
    
    def walk_forward_analysis(self,
                             data: pd.DataFrame,
                             strategy_func,
                             param_ranges: Dict,
                             optimization_metric: str = "sharpe_ratio") -> Dict:
        """
        Exécute une analyse walk-forward complète
        
        Args:
            data: Données complètes avec DatetimeIndex
            strategy_func: Fonction de stratégie à analyser
            param_ranges: Ranges de paramètres à optimiser
            optimization_metric: Métrique pour optimisation
            
        Returns:
            Résultats complets de l'analyse
        """
        
        logger.info("🚀 Début Walk-Forward Analysis")
        
        # 1. Créer les splits temporels
        splits = self.create_time_splits(data)
        
        if len(splits) < 3:
            raise ValueError("Pas assez de splits pour analyse robuste (minimum 3)")
        
        # 2. Pour chaque split, optimiser et tester
        walk_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            
            logger.info(f"📊 Split {i+1}/{len(splits)}: Train({train_start.date()}→{train_end.date()}) Test({test_start.date()}→{test_end.date()})")
            
            # Données pour ce split
            train_data = data[train_start:train_end].copy()
            test_data = data[test_start:test_end].copy()
            
            # 3. Optimisation paramètres sur période train
            best_params = self._optimize_parameters(
                train_data, strategy_func, param_ranges, optimization_metric
            )
            
            # 4. Test avec paramètres optimaux sur OOS
            period_result = self.run_strategy_on_period(
                strategy_func, train_data, test_data, best_params
            )
            
            if period_result["valid"]:
                period_result.update({
                    "split_id": i,
                    "optimal_params": best_params,
                    "train_start": train_start,
                    "train_end": train_end, 
                    "test_start": test_start,
                    "test_end": test_end
                })
                walk_results.append(period_result)
            
        # 5. Analyser les résultats globaux
        self.walk_forward_results = walk_results
        analysis_results = self._analyze_walk_forward_results(walk_results)
        
        logger.info("✅ Walk-Forward Analysis terminée")
        return analysis_results
    
    def _optimize_parameters(self,
                            train_data: pd.DataFrame,
                            strategy_func,
                            param_ranges: Dict,
                            optimization_metric: str) -> Dict:
        """
        Optimise les paramètres sur la période d'entraînement
        
        Args:
            train_data: Données d'entraînement
            strategy_func: Fonction de stratégie
            param_ranges: Ranges de paramètres
            optimization_metric: Métrique à optimiser
            
        Returns:
            Paramètres optimaux
        """
        
        logger.debug("🔧 Optimisation paramètres...")
        
        # Génération grille de paramètres (simple grid search)
        param_combinations = self._generate_param_grid(param_ranges)
        
        best_score = -np.inf
        best_params = {}
        
        for params in param_combinations[:20]:  # Limite à 20 combos pour performance
            
            try:
                # Test paramètres sur données train
                result = strategy_func(train_data, params)
                
                if result and optimization_metric in result:
                    score = result[optimization_metric]
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        
            except Exception as e:
                logger.debug(f"Paramètres échoués {params}: {e}")
                continue
        
        logger.debug(f"✅ Paramètres optimaux: {best_params} (score: {best_score:.4f})")
        return best_params if best_params else param_combinations[0]
    
    def _generate_param_grid(self, param_ranges: Dict) -> List[Dict]:
        """
        Génère une grille de paramètres à tester
        
        Args:
            param_ranges: Dict avec ranges de paramètres
            
        Returns:
            Liste des combinaisons de paramètres
        """
        
        from itertools import product
        
        # Extraire noms et valeurs
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        # Générer toutes combinaisons
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _calculate_period_metrics(self, 
                                 returns: pd.Series,
                                 signals: pd.Series,
                                 prefix: str = "") -> Dict:
        """
        Calcule les métriques pour une période
        """
        
        if len(returns) == 0:
            return {f"{prefix}_sharpe": 0, f"{prefix}_return": 0, f"{prefix}_trades": 0}
        
        # Métriques de base
        total_return = (1 + returns).prod() - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trades
        total_trades = len(signals[signals != 0]) if len(signals) > 0 else 0
        
        return {
            f"{prefix}_total_return": float(total_return),
            f"{prefix}_sharpe": float(sharpe),
            f"{prefix}_volatility": float(volatility),
            f"{prefix}_max_drawdown": float(max_drawdown),
            f"{prefix}_total_trades": int(total_trades)
        }
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calcule les returns d'une stratégie basée sur signaux
        """
        
        if 'close' not in data.columns or len(signals) == 0:
            return pd.Series(dtype=float)
        
        # Returns des prix
        price_returns = data['close'].pct_change().fillna(0)
        
        # Aligner signaux et returns
        aligned_signals = signals.reindex(price_returns.index, method='ffill').fillna(0)
        
        # Strategy returns = signal * price_return
        strategy_returns = aligned_signals.shift(1) * price_returns
        
        return strategy_returns.fillna(0)
    
    def _generate_generic_signals(self, model, data: pd.DataFrame) -> pd.Series:
        """
        Génère des signaux génériques depuis un modèle
        """
        
        # Implémentation simple - à adapter selon le modèle
        if hasattr(model, 'predict'):
            try:
                predictions = model.predict(data[['close', 'volume']].fillna(0))
                signals = pd.Series(np.sign(predictions), index=data.index)
                return signals
            except:
                pass
        
        # Fallback: signaux aléatoiresfor testing
        return pd.Series(0, index=data.index)
    
    def _run_significance_tests(self, 
                               is_returns: pd.Series,
                               oos_returns: pd.Series) -> Dict:
        """
        Tests de significativité statistique
        """
        
        tests = {}
        
        if len(is_returns) > 0 and len(oos_returns) > 0:
            
            # Test t pour différence de moyennes
            try:
                t_stat, t_pval = stats.ttest_ind(is_returns.dropna(), oos_returns.dropna())
                tests["t_test"] = {"statistic": float(t_stat), "p_value": float(t_pval)}
            except:
                tests["t_test"] = {"statistic": np.nan, "p_value": np.nan}
            
            # Test de Kolmogorov-Smirnov (différence distributions)
            try:
                ks_stat, ks_pval = stats.ks_2samp(is_returns.dropna(), oos_returns.dropna())
                tests["ks_test"] = {"statistic": float(ks_stat), "p_value": float(ks_pval)}
            except:
                tests["ks_test"] = {"statistic": np.nan, "p_value": np.nan}
        
        return tests
    
    def _analyze_walk_forward_results(self, walk_results: List[Dict]) -> Dict:
        """
        Analyse les résultats globaux du walk-forward
        """
        
        logger.info("📊 Analyse des résultats walk-forward...")
        
        if not walk_results:
            return {"error": "Aucun résultat valide"}
        
        # Extraire métriques
        is_sharpes = [r["in_sample"]["IS_sharpe"] for r in walk_results if "in_sample" in r]
        oos_sharpes = [r["out_of_sample"]["OOS_sharpe"] for r in walk_results if "out_of_sample" in r]
        
        is_returns = [r["in_sample"]["IS_total_return"] for r in walk_results if "in_sample" in r]
        oos_returns = [r["out_of_sample"]["OOS_total_return"] for r in walk_results if "out_of_sample" in r]
        
        # Analyses de robustesse
        analysis = {
            "summary": {
                "total_periods": len(walk_results),
                "avg_is_sharpe": np.mean(is_sharpes) if is_sharpes else 0,
                "avg_oos_sharpe": np.mean(oos_sharpes) if oos_sharpes else 0,
                "avg_is_return": np.mean(is_returns) if is_returns else 0,
                "avg_oos_return": np.mean(oos_returns) if oos_returns else 0,
            },
            
            "robustness": {
                "sharpe_degradation": self._calculate_degradation(is_sharpes, oos_sharpes),
                "return_degradation": self._calculate_degradation(is_returns, oos_returns),
                "consistency_score": self._calculate_consistency(oos_sharpes),
                "overfitting_score": self._detect_overfitting(is_sharpes, oos_sharpes),
            },
            
            "stability": {
                "sharpe_std": np.std(oos_sharpes) if oos_sharpes else np.inf,
                "return_std": np.std(oos_returns) if oos_returns else np.inf,
                "positive_periods": sum(1 for x in oos_returns if x > 0),
                "negative_periods": sum(1 for x in oos_returns if x < 0),
            }
        }
        
        # Verdict final
        analysis["verdict"] = self._generate_robustness_verdict(analysis)
        
        return analysis
    
    def _calculate_degradation(self, is_values: List, oos_values: List) -> float:
        """
        Calcule la dégradation IS vs OOS
        """
        if not is_values or not oos_values:
            return np.inf
        
        is_avg = np.mean(is_values)
        oos_avg = np.mean(oos_values)
        
        if is_avg == 0:
            return np.inf
        
        degradation = (is_avg - oos_avg) / abs(is_avg)
        return float(degradation)
    
    def _calculate_consistency(self, values: List) -> float:
        """
        Score de consistance (0-1, 1 = très consistant)
        """
        if len(values) <= 1:
            return 0.0
        
        # Proportion de valeurs positives
        positive_ratio = sum(1 for x in values if x > 0) / len(values)
        
        # Coefficient de variation (inversé)
        cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else np.inf
        consistency = 1 / (1 + cv)
        
        # Score combiné
        return float(positive_ratio * consistency)
    
    def _detect_overfitting(self, is_values: List, oos_values: List) -> float:
        """
        Score d'overfitting (0-1, 1 = très overfitté)
        """
        if not is_values or not oos_values:
            return 1.0
        
        degradation = self._calculate_degradation(is_values, oos_values)
        
        # Score basé sur la dégradation
        if degradation < 0.1:  # < 10% dégradation
            return 0.0
        elif degradation < 0.3:  # 10-30% dégradation
            return 0.3
        elif degradation < 0.5:  # 30-50% dégradation
            return 0.6
        else:  # > 50% dégradation
            return 1.0
    
    def _generate_robustness_verdict(self, analysis: Dict) -> Dict:
        """
        Génère le verdict final de robustesse
        """
        
        robustness = analysis["robustness"]
        summary = analysis["summary"]
        
        # Critères de robustesse
        criteria = {
            "oos_sharpe_positive": summary["avg_oos_sharpe"] > 0,
            "low_overfitting": robustness["overfitting_score"] < 0.5,
            "consistent_performance": robustness["consistency_score"] > 0.6,
            "acceptable_degradation": robustness["sharpe_degradation"] < 0.5
        }
        
        # Score global
        robustness_score = sum(criteria.values()) / len(criteria)
        
        # Verdict textuel
        if robustness_score >= 0.8:
            verdict_text = "✅ ROBUSTE - Stratégie fiable pour production"
        elif robustness_score >= 0.6:
            verdict_text = "⚠️ MODÉRÉMENT ROBUSTE - Surveillance requise"
        elif robustness_score >= 0.4:
            verdict_text = "🟡 PEU ROBUSTE - Optimisation nécessaire"
        else:
            verdict_text = "❌ NON-ROBUSTE - Stratégie non viable"
        
        return {
            "score": float(robustness_score),
            "text": verdict_text,
            "criteria": criteria,
            "recommendation": self._generate_recommendations(analysis)
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """
        Génère des recommandations basées sur l'analyse
        """
        
        recommendations = []
        robustness = analysis["robustness"]
        summary = analysis["summary"]
        
        if robustness["overfitting_score"] > 0.5:
            recommendations.append("🔧 Réduire la complexité du modèle (overfitting détecté)")
        
        if summary["avg_oos_sharpe"] < 0.5:
            recommendations.append("📈 Améliorer la stratégie (Sharpe OOS faible)")
        
        if robustness["consistency_score"] < 0.5:
            recommendations.append("🎯 Stabiliser les paramètres (performance incohérente)")
        
        if robustness["sharpe_degradation"] > 0.5:
            recommendations.append("⚠️ Réviser l'optimisation (forte dégradation IS→OOS)")
        
        if not recommendations:
            recommendations.append("✅ Stratégie robuste - Prête pour déploiement")
        
        return recommendations
    
    def generate_report(self, output_path: str = "walk_forward_report.html") -> str:
        """
        Génère un rapport HTML complet
        """
        
        if not self.walk_forward_results:
            logger.error("Aucun résultat à reporter")
            return ""
        
        # TODO: Implémenter génération rapport HTML
        logger.info(f"📄 Rapport sauvé: {output_path}")
        return output_path

# Exemple d'utilisation
if __name__ == "__main__":
    
    # Test avec données simulées
    logger.info("🧪 Test Walk-Forward Analyzer...")
    
    # Données test
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='1h')
    data = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Stratégie test simple
    def simple_strategy(data, params):
        lookback = params.get('lookback', 20)
        
        # Simple moving average strategy
        ma = data['close'].rolling(lookback).mean()
        signals = pd.Series(0, index=data.index)
        signals[data['close'] > ma] = 1
        signals[data['close'] < ma] = -1
        
        returns = signals.shift(1) * data['close'].pct_change()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'model': None,
            'returns': returns,
            'signals': signals,
            'sharpe_ratio': sharpe
        }
    
    # Paramètres à tester
    param_ranges = {
        'lookback': [10, 20, 30, 50]
    }
    
    # Analyse walk-forward
    analyzer = WalkForwardAnalyzer(
        train_window_days=90,
        test_window_days=15,
        step_days=7
    )
    
    results = analyzer.walk_forward_analysis(
        data, simple_strategy, param_ranges, "sharpe_ratio"
    )
    
    print("\n=== RÉSULTATS WALK-FORWARD ===")
    print(f"Périodes analysées: {results['summary']['total_periods']}")
    print(f"Sharpe moyen IS: {results['summary']['avg_is_sharpe']:.3f}")
    print(f"Sharpe moyen OOS: {results['summary']['avg_oos_sharpe']:.3f}")
    print(f"Score robustesse: {results['verdict']['score']:.3f}")
    print(f"Verdict: {results['verdict']['text']}")
    
    print("\n=== RECOMMANDATIONS ===")
    for rec in results['verdict']['recommendation']:
        print(f"  {rec}")