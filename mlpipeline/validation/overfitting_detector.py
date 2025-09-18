#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overfitting Detector - Système de détection avancé d'overfitting
Détection multi-niveaux et automatisée pour stratégies quantitatives
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# import seaborn as sns  # Optionnel pour viz

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class OverfittingDetector:
    """
    Détecteur avancé d'overfitting pour stratégies quantitatives
    
    Méthodes de détection:
    1. Performance Drop Analysis (IS vs OOS)
    2. Parameter Sensitivity Analysis
    3. White Reality Check (Hansen, 2005)
    4. Rolling Window Stability
    5. Complexity Penalty Analysis
    6. Bootstrap Reality Check
    7. Monte Carlo Permutation Tests
    8. Regime Change Analysis
    """
    
    def __init__(self,
                 confidence_level: float = 0.95,
                 bootstrap_trials: int = 1000,
                 sensitivity_threshold: float = 0.3,
                 stability_window: int = 60):
        
        self.confidence_level = confidence_level
        self.bootstrap_trials = bootstrap_trials
        self.sensitivity_threshold = sensitivity_threshold
        self.stability_window = stability_window
        
        # Résultats stockés
        self.detection_results = {}
        
        logger.info(f"✅ OverfittingDetector initialisé:")
        logger.info(f"   - Confiance: {confidence_level*100:.1f}%")
        logger.info(f"   - Bootstrap trials: {bootstrap_trials}")
    
    def comprehensive_overfitting_analysis(self,
                                         strategy_func,
                                         data: pd.DataFrame,
                                         param_ranges: Dict,
                                         base_params: Dict,
                                         split_ratio: float = 0.7) -> Dict:
        """
        Analyse complète de détection d'overfitting
        
        Args:
            strategy_func: Fonction de stratégie à analyser
            data: Données historiques complètes
            param_ranges: Ranges de paramètres à tester
            base_params: Paramètres de base de référence
            split_ratio: Ratio IS/OOS pour split temporel
            
        Returns:
            Analyse complète d'overfitting
        """
        
        logger.info("🔍 Début analyse complète d'overfitting")
        start_time = datetime.now()
        
        try:
            # 1. Préparation des données
            is_data, oos_data = self._temporal_split(data, split_ratio)
            
            # 2. Performance Drop Analysis
            logger.info("📊 Analyse dégradation performance...")
            performance_analysis = self._performance_drop_analysis(
                strategy_func, is_data, oos_data, base_params
            )
            
            # 3. Parameter Sensitivity Analysis
            logger.info("🎛️ Analyse sensibilité paramètres...")
            sensitivity_analysis = self._parameter_sensitivity_analysis(
                strategy_func, is_data, param_ranges, base_params
            )
            
            # 4. White Reality Check
            logger.info("🎯 White Reality Check...")
            white_reality_check = self._white_reality_check(
                strategy_func, data, param_ranges, base_params
            )
            
            # 5. Rolling Window Stability
            logger.info("📈 Analyse stabilité temporelle...")
            stability_analysis = self._rolling_stability_analysis(
                strategy_func, data, base_params
            )
            
            # 6. Complexity Analysis
            logger.info("🧮 Analyse complexité modèle...")
            complexity_analysis = self._complexity_analysis(
                param_ranges, performance_analysis
            )
            
            # 7. Bootstrap Reality Check
            logger.info("🎲 Bootstrap Reality Check...")
            bootstrap_analysis = self._bootstrap_reality_check(
                strategy_func, is_data, oos_data, base_params
            )
            
            # 8. Regime Change Analysis
            logger.info("📊 Analyse changements de régime...")
            regime_analysis = self._regime_change_analysis(
                strategy_func, data, base_params
            )
            
            # 9. Score global d'overfitting
            overfitting_score = self._calculate_overfitting_score({
                "performance": performance_analysis,
                "sensitivity": sensitivity_analysis,
                "white_reality": white_reality_check,
                "stability": stability_analysis,
                "complexity": complexity_analysis,
                "bootstrap": bootstrap_analysis,
                "regime": regime_analysis
            })
            
            # 10. Compilation des résultats
            execution_time = (datetime.now() - start_time).total_seconds()
            
            comprehensive_analysis = {
                "metadata": {
                    "timestamp": start_time.isoformat(),
                    "execution_time": execution_time,
                    "data_periods": len(data),
                    "is_periods": len(is_data),
                    "oos_periods": len(oos_data),
                    "parameters_tested": len(list(ParameterGrid(param_ranges)))
                },
                "performance_drop": performance_analysis,
                "parameter_sensitivity": sensitivity_analysis,
                "white_reality_check": white_reality_check,
                "stability_analysis": stability_analysis,
                "complexity_analysis": complexity_analysis,
                "bootstrap_reality_check": bootstrap_analysis,
                "regime_analysis": regime_analysis,
                "overfitting_score": overfitting_score,
                "final_verdict": self._generate_overfitting_verdict(overfitting_score)
            }
            
            self.detection_results = comprehensive_analysis
            
            logger.info(f"✅ Analyse overfitting terminée en {execution_time:.1f}s")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse overfitting: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "valid": False}
    
    def _temporal_split(self, data: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split temporel strict pour validation
        """
        split_index = int(len(data) * split_ratio)
        
        is_data = data.iloc[:split_index].copy()
        oos_data = data.iloc[split_index:].copy()
        
        logger.debug(f"Split: IS={len(is_data)} OOS={len(oos_data)}")
        return is_data, oos_data
    
    def _performance_drop_analysis(self,
                                 strategy_func,
                                 is_data: pd.DataFrame,
                                 oos_data: pd.DataFrame,
                                 params: Dict) -> Dict:
        """
        Analyse de la dégradation IS → OOS
        """
        
        try:
            # Performance in-sample
            is_results = strategy_func(is_data, params)
            is_returns = is_results.get('returns', pd.Series())
            is_sharpe = self._calculate_sharpe(is_returns)
            is_total_return = (1 + is_returns).prod() - 1 if len(is_returns) > 0 else 0
            
            # Performance out-of-sample
            oos_results = strategy_func(oos_data, params)
            oos_returns = oos_results.get('returns', pd.Series())
            oos_sharpe = self._calculate_sharpe(oos_returns)
            oos_total_return = (1 + oos_returns).prod() - 1 if len(oos_returns) > 0 else 0
            
            # Calcul des dégradations
            sharpe_degradation = self._calculate_degradation(is_sharpe, oos_sharpe)
            return_degradation = self._calculate_degradation(is_total_return, oos_total_return)
            
            # Tests statistiques
            statistical_significance = self._test_performance_difference(is_returns, oos_returns)
            
            # Classification de l'overfitting
            overfitting_level = self._classify_overfitting_level(sharpe_degradation, return_degradation)
            
            return {
                "is_sharpe": float(is_sharpe),
                "oos_sharpe": float(oos_sharpe),
                "is_total_return": float(is_total_return),
                "oos_total_return": float(oos_total_return),
                "sharpe_degradation": float(sharpe_degradation),
                "return_degradation": float(return_degradation),
                "statistical_significance": statistical_significance,
                "overfitting_level": overfitting_level
            }
            
        except Exception as e:
            logger.error(f"Erreur performance drop analysis: {e}")
            return {"error": str(e)}
    
    def _parameter_sensitivity_analysis(self,
                                      strategy_func,
                                      data: pd.DataFrame,
                                      param_ranges: Dict,
                                      base_params: Dict) -> Dict:
        """
        Analyse de sensibilité aux paramètres
        """
        
        try:
            # Performance avec paramètres de base
            base_results = strategy_func(data, base_params)
            base_sharpe = self._calculate_sharpe(base_results.get('returns', pd.Series()))
            
            # Tester variations de paramètres
            sensitivity_scores = {}
            
            for param_name, param_range in param_ranges.items():
                if param_name in base_params:
                    param_sharpes = []
                    
                    for param_value in param_range:
                        test_params = base_params.copy()
                        test_params[param_name] = param_value
                        
                        try:
                            test_results = strategy_func(data, test_params)
                            test_sharpe = self._calculate_sharpe(test_results.get('returns', pd.Series()))
                            param_sharpes.append(test_sharpe)
                        except:
                            param_sharpes.append(0)
                    
                    # Calcul sensibilité (coefficient de variation)
                    if len(param_sharpes) > 1 and np.mean(param_sharpes) != 0:
                        sensitivity = np.std(param_sharpes) / abs(np.mean(param_sharpes))
                    else:
                        sensitivity = 0
                    
                    sensitivity_scores[param_name] = {
                        "sensitivity_score": float(sensitivity),
                        "param_sharpes": [float(x) for x in param_sharpes],
                        "best_sharpe": float(max(param_sharpes)),
                        "worst_sharpe": float(min(param_sharpes))
                    }
            
            # Sensibilité globale
            all_sensitivities = [scores["sensitivity_score"] for scores in sensitivity_scores.values()]
            global_sensitivity = np.mean(all_sensitivities) if all_sensitivities else 0
            
            # Détection overfitting basée sur sensibilité
            high_sensitivity_params = [
                param for param, scores in sensitivity_scores.items()
                if scores["sensitivity_score"] > self.sensitivity_threshold
            ]
            
            return {
                "global_sensitivity": float(global_sensitivity),
                "parameter_sensitivity": sensitivity_scores,
                "high_sensitivity_params": high_sensitivity_params,
                "overfitting_detected": len(high_sensitivity_params) > len(param_ranges) / 2
            }
            
        except Exception as e:
            logger.error(f"Erreur sensitivity analysis: {e}")
            return {"error": str(e)}
    
    def _white_reality_check(self,
                           strategy_func,
                           data: pd.DataFrame,
                           param_ranges: Dict,
                           base_params: Dict) -> Dict:
        """
        White Reality Check (Hansen, 2005) - Test de multiple testing
        """
        
        try:
            # Split temporel
            split_point = len(data) // 2
            selection_data = data.iloc[:split_point]
            evaluation_data = data.iloc[split_point:]
            
            # Générer toutes les combinaisons de paramètres
            param_grid = list(ParameterGrid(param_ranges))
            n_strategies = len(param_grid)
            
            logger.debug(f"Testing {n_strategies} parameter combinations")
            
            # Phase 1: Sélection sur première période
            selection_performances = []
            
            for params in param_grid:
                combined_params = {**base_params, **params}
                try:
                    results = strategy_func(selection_data, combined_params)
                    returns = results.get('returns', pd.Series())
                    sharpe = self._calculate_sharpe(returns)
                    selection_performances.append(sharpe)
                except:
                    selection_performances.append(0)
            
            # Sélectionner la meilleure stratégie
            best_idx = np.argmax(selection_performances)
            best_params = {**base_params, **param_grid[best_idx]}
            best_selection_sharpe = selection_performances[best_idx]
            
            # Phase 2: Évaluation sur deuxième période
            evaluation_results = strategy_func(evaluation_data, best_params)
            evaluation_returns = evaluation_results.get('returns', pd.Series())
            evaluation_sharpe = self._calculate_sharpe(evaluation_returns)
            
            # Monte Carlo pour p-value
            mc_sharpes = []
            
            for _ in range(self.bootstrap_trials):
                # Permutation aléatoire
                shuffled_data = evaluation_data.copy()
                shuffled_data['close'] = evaluation_data['close'].sample(frac=1).values
                
                try:
                    mc_results = strategy_func(shuffled_data, best_params)
                    mc_returns = mc_results.get('returns', pd.Series())
                    mc_sharpe = self._calculate_sharpe(mc_returns)
                    mc_sharpes.append(mc_sharpe)
                except:
                    mc_sharpes.append(0)
            
            # P-value ajustée pour multiple testing
            p_value = (np.array(mc_sharpes) >= evaluation_sharpe).mean()
            
            # Correction de Bonferroni
            bonferroni_p_value = min(1.0, p_value * n_strategies)
            
            return {
                "n_strategies_tested": n_strategies,
                "best_selection_sharpe": float(best_selection_sharpe),
                "evaluation_sharpe": float(evaluation_sharpe),
                "raw_p_value": float(p_value),
                "bonferroni_p_value": float(bonferroni_p_value),
                "significant": bonferroni_p_value < (1 - self.confidence_level),
                "overfitting_likely": bonferroni_p_value > 0.5,
                "best_params": best_params
            }
            
        except Exception as e:
            logger.error(f"Erreur White Reality Check: {e}")
            return {"error": str(e)}
    
    def _rolling_stability_analysis(self,
                                  strategy_func,
                                  data: pd.DataFrame,
                                  params: Dict) -> Dict:
        """
        Analyse de stabilité sur fenêtres glissantes
        """
        
        try:
            if len(data) < self.stability_window * 3:
                return {"error": "Not enough data for stability analysis"}
            
            # Performance sur fenêtres glissantes
            window_performances = []
            window_dates = []
            
            for i in range(0, len(data) - self.stability_window + 1, self.stability_window // 4):
                window_data = data.iloc[i:i + self.stability_window]
                
                if len(window_data) >= self.stability_window:
                    try:
                        results = strategy_func(window_data, params)
                        returns = results.get('returns', pd.Series())
                        sharpe = self._calculate_sharpe(returns)
                        
                        window_performances.append(sharpe)
                        window_dates.append(window_data.index[-1])
                    except:
                        continue
            
            if len(window_performances) < 3:
                return {"error": "Not enough windows for stability analysis"}
            
            # Métriques de stabilité
            mean_performance = np.mean(window_performances)
            std_performance = np.std(window_performances)
            
            # Coefficient de variation (stabilité inverse)
            if mean_performance != 0:
                stability_score = 1 - (std_performance / abs(mean_performance))
                stability_score = max(0, stability_score)  # Borné à [0, 1]
            else:
                stability_score = 0
            
            # Trend analysis
            if len(window_performances) > 2:
                time_trend = stats.linregress(range(len(window_performances)), window_performances).slope
            else:
                time_trend = 0
            
            # Détection de dégradation temporelle
            first_half = window_performances[:len(window_performances)//2]
            second_half = window_performances[len(window_performances)//2:]
            
            if len(first_half) > 0 and len(second_half) > 0:
                performance_degradation = (np.mean(first_half) - np.mean(second_half)) / abs(np.mean(first_half)) if np.mean(first_half) != 0 else 0
            else:
                performance_degradation = 0
            
            return {
                "window_count": len(window_performances),
                "mean_performance": float(mean_performance),
                "performance_std": float(std_performance),
                "stability_score": float(stability_score),
                "time_trend": float(time_trend),
                "performance_degradation": float(performance_degradation),
                "window_performances": [float(x) for x in window_performances],
                "unstable": stability_score < 0.5 or abs(time_trend) > 0.1
            }
            
        except Exception as e:
            logger.error(f"Erreur stability analysis: {e}")
            return {"error": str(e)}
    
    def _complexity_analysis(self, param_ranges: Dict, performance_analysis: Dict) -> Dict:
        """
        Analyse de la complexité du modèle vs performance
        """
        
        try:
            # Calcul de complexité basé sur nombre de paramètres
            n_parameters = len(param_ranges)
            total_combinations = np.prod([len(range_vals) for range_vals in param_ranges.values()])
            
            # Score de complexité normalisé
            complexity_score = min(1.0, n_parameters / 10)  # Normalize à [0,1]
            
            # Rapport performance/complexité
            performance_score = performance_analysis.get("is_sharpe", 0)
            
            if complexity_score > 0:
                efficiency_ratio = performance_score / complexity_score
            else:
                efficiency_ratio = performance_score
            
            # Risque d'overfitting basé sur complexité
            overfitting_risk = self._assess_complexity_overfitting_risk(
                n_parameters, total_combinations, performance_analysis
            )
            
            return {
                "n_parameters": n_parameters,
                "total_combinations": int(total_combinations),
                "complexity_score": float(complexity_score),
                "performance_efficiency_ratio": float(efficiency_ratio),
                "overfitting_risk": overfitting_risk,
                "too_complex": complexity_score > 0.7 and overfitting_risk["risk_level"] == "High"
            }
            
        except Exception as e:
            logger.error(f"Erreur complexity analysis: {e}")
            return {"error": str(e)}
    
    def _bootstrap_reality_check(self,
                               strategy_func,
                               is_data: pd.DataFrame,
                               oos_data: pd.DataFrame,
                               params: Dict) -> Dict:
        """
        Bootstrap Reality Check pour validation robuste
        """
        
        try:
            # Performance observée OOS
            observed_results = strategy_func(oos_data, params)
            observed_returns = observed_results.get('returns', pd.Series())
            observed_sharpe = self._calculate_sharpe(observed_returns)
            
            # Bootstrap simulations
            bootstrap_sharpes = []
            
            for trial in range(self.bootstrap_trials):
                # Sample avec remplacement des returns OOS
                if len(observed_returns) > 10:
                    bootstrap_returns = observed_returns.sample(n=len(observed_returns), replace=True)
                    bootstrap_sharpe = self._calculate_sharpe(bootstrap_returns)
                    bootstrap_sharpes.append(bootstrap_sharpe)
            
            if not bootstrap_sharpes:
                return {"error": "No valid bootstrap samples"}
            
            # Analyse des résultats bootstrap
            bootstrap_sharpes = np.array(bootstrap_sharpes)
            
            # Intervalles de confiance
            ci_lower = np.percentile(bootstrap_sharpes, 2.5)
            ci_upper = np.percentile(bootstrap_sharpes, 97.5)
            
            # P-value (probabilité d'obtenir performance >= observée par hasard)
            p_value = (bootstrap_sharpes >= observed_sharpe).mean()
            
            # Stabilité bootstrap (variance des résultats)
            bootstrap_stability = 1 - (np.std(bootstrap_sharpes) / abs(np.mean(bootstrap_sharpes))) if np.mean(bootstrap_sharpes) != 0 else 0
            bootstrap_stability = max(0, bootstrap_stability)
            
            return {
                "observed_sharpe": float(observed_sharpe),
                "bootstrap_mean": float(np.mean(bootstrap_sharpes)),
                "bootstrap_std": float(np.std(bootstrap_sharpes)),
                "confidence_interval": [float(ci_lower), float(ci_upper)],
                "p_value": float(p_value),
                "bootstrap_stability": float(bootstrap_stability),
                "statistically_significant": p_value < (1 - self.confidence_level),
                "bootstrap_trials": len(bootstrap_sharpes)
            }
            
        except Exception as e:
            logger.error(f"Erreur bootstrap reality check: {e}")
            return {"error": str(e)}
    
    def _regime_change_analysis(self,
                              strategy_func,
                              data: pd.DataFrame,
                              params: Dict) -> Dict:
        """
        Analyse performance selon les régimes de marché
        """
        
        try:
            if len(data) < 200:
                return {"error": "Not enough data for regime analysis"}
            
            # Détecter régimes (simple: basé sur volatilité)
            returns = data['close'].pct_change().dropna()
            rolling_vol = returns.rolling(20).std()
            
            vol_median = rolling_vol.median()
            high_vol_regime = rolling_vol > vol_median * 1.5
            low_vol_regime = rolling_vol < vol_median * 0.5
            normal_regime = ~(high_vol_regime | low_vol_regime)
            
            # Performance par régime
            regime_performances = {}
            
            for regime_name, regime_mask in [
                ("high_volatility", high_vol_regime),
                ("low_volatility", low_vol_regime),
                ("normal_volatility", normal_regime)
            ]:
                regime_data = data[regime_mask.reindex(data.index, fill_value=False)]
                
                if len(regime_data) > 30:
                    try:
                        regime_results = strategy_func(regime_data, params)
                        regime_returns = regime_results.get('returns', pd.Series())
                        regime_sharpe = self._calculate_sharpe(regime_returns)
                        
                        regime_performances[regime_name] = {
                            "periods": len(regime_data),
                            "sharpe": float(regime_sharpe),
                            "total_return": float((1 + regime_returns).prod() - 1) if len(regime_returns) > 0 else 0
                        }
                    except:
                        regime_performances[regime_name] = {"periods": 0, "sharpe": 0, "total_return": 0}
            
            # Consistance entre régimes
            regime_sharpes = [perf["sharpe"] for perf in regime_performances.values() if perf["periods"] > 0]
            
            if len(regime_sharpes) > 1:
                regime_consistency = 1 - (np.std(regime_sharpes) / abs(np.mean(regime_sharpes))) if np.mean(regime_sharpes) != 0 else 0
                regime_consistency = max(0, regime_consistency)
            else:
                regime_consistency = 0
            
            # Détection de biais de régime
            positive_regimes = sum(1 for sharpe in regime_sharpes if sharpe > 0)
            regime_bias_detected = positive_regimes < len(regime_sharpes) * 0.5
            
            return {
                "regime_performances": regime_performances,
                "regime_consistency": float(regime_consistency),
                "regime_bias_detected": regime_bias_detected,
                "robust_across_regimes": regime_consistency > 0.6 and not regime_bias_detected
            }
            
        except Exception as e:
            logger.error(f"Erreur regime analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_overfitting_score(self, analyses: Dict) -> Dict:
        """
        Score global d'overfitting basé sur toutes les analyses
        """
        
        # Poids des différentes analyses
        weights = {
            "performance": 0.25,
            "sensitivity": 0.15,
            "white_reality": 0.20,
            "stability": 0.15,
            "complexity": 0.10,
            "bootstrap": 0.10,
            "regime": 0.05
        }
        
        # Scores individuels (0-1, 1 = overfitting sévère)
        scores = {}
        
        # Performance drop score
        perf_analysis = analyses.get("performance", {})
        if "sharpe_degradation" in perf_analysis:
            degradation = perf_analysis["sharpe_degradation"]
            scores["performance"] = min(1.0, max(0.0, degradation))
        else:
            scores["performance"] = 0
        
        # Sensitivity score
        sens_analysis = analyses.get("sensitivity", {})
        if "global_sensitivity" in sens_analysis:
            sensitivity = sens_analysis["global_sensitivity"]
            scores["sensitivity"] = min(1.0, sensitivity / 2)  # Normalise
        else:
            scores["sensitivity"] = 0
        
        # White reality check score
        white_analysis = analyses.get("white_reality", {})
        if "bonferroni_p_value" in white_analysis:
            p_val = white_analysis["bonferroni_p_value"]
            scores["white_reality"] = p_val  # P-value élevée = overfitting probable
        else:
            scores["white_reality"] = 0
        
        # Stability score (inverser car stabilité élevée = bon)
        stab_analysis = analyses.get("stability", {})
        if "stability_score" in stab_analysis:
            stability = stab_analysis["stability_score"]
            scores["stability"] = 1 - stability  # Inverser
        else:
            scores["stability"] = 0
        
        # Complexity score
        comp_analysis = analyses.get("complexity", {})
        if "complexity_score" in comp_analysis:
            scores["complexity"] = comp_analysis["complexity_score"]
        else:
            scores["complexity"] = 0
        
        # Bootstrap score
        boot_analysis = analyses.get("bootstrap", {})
        if "p_value" in boot_analysis:
            p_val = boot_analysis["p_value"]
            scores["bootstrap"] = p_val  # P-value élevée = performance non significative
        else:
            scores["bootstrap"] = 0
        
        # Regime score
        regime_analysis = analyses.get("regime", {})
        if "regime_consistency" in regime_analysis:
            consistency = regime_analysis["regime_consistency"]
            scores["regime"] = 1 - consistency  # Faible consistance = overfitting
        else:
            scores["regime"] = 0
        
        # Score global pondéré
        global_score = sum(scores[key] * weights[key] for key in scores.keys())
        
        return {
            "global_overfitting_score": float(global_score),
            "component_scores": scores,
            "weights": weights,
            "overfitting_level": self._score_to_level(global_score)
        }
    
    def _generate_overfitting_verdict(self, overfitting_score: Dict) -> Dict:
        """
        Verdict final sur l'overfitting
        """
        
        score = overfitting_score["global_overfitting_score"]
        level = overfitting_score["overfitting_level"]
        
        # Verdict textuel
        if score < 0.2:
            verdict = "✅ PAS D'OVERFITTING - Stratégie robuste"
            color = "green"
            action = "Prêt pour production"
        elif score < 0.4:
            verdict = "⚠️ OVERFITTING LÉGER - Surveillance recommandée"
            color = "orange"
            action = "Déploiement avec monitoring renforcé"
        elif score < 0.6:
            verdict = "🟡 OVERFITTING MODÉRÉ - Correction nécessaire"
            color = "yellow"
            action = "Optimisation et re-validation requises"
        else:
            verdict = "❌ OVERFITTING SÉVÈRE - Stratégie non viable"
            color = "red"
            action = "Refonte complète de la stratégie"
        
        # Recommandations spécifiques
        recommendations = self._generate_specific_recommendations(overfitting_score)
        
        return {
            "overfitting_score": float(score),
            "overfitting_level": level,
            "verdict": verdict,
            "color": color,
            "recommended_action": action,
            "specific_recommendations": recommendations,
            "production_suitable": score < 0.4
        }
    
    # Méthodes utilitaires
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calcul Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    def _calculate_degradation(self, is_value: float, oos_value: float) -> float:
        """Calcul dégradation IS->OOS"""
        if is_value == 0:
            return 1.0 if oos_value < 0 else 0.0
        return (is_value - oos_value) / abs(is_value)
    
    def _test_performance_difference(self, is_returns: pd.Series, oos_returns: pd.Series) -> Dict:
        """Test statistique différence IS vs OOS"""
        try:
            if len(is_returns) > 10 and len(oos_returns) > 10:
                t_stat, p_val = stats.ttest_ind(is_returns.dropna(), oos_returns.dropna())
                return {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "significant_difference": p_val < 0.05
                }
        except:
            pass
        return {"error": "Could not compute statistical test"}
    
    def _classify_overfitting_level(self, sharpe_deg: float, return_deg: float) -> str:
        """Classification niveau overfitting"""
        avg_deg = (sharpe_deg + return_deg) / 2
        
        if avg_deg < 0.2:
            return "None"
        elif avg_deg < 0.4:
            return "Mild"
        elif avg_deg < 0.7:
            return "Moderate"
        else:
            return "Severe"
    
    def _assess_complexity_overfitting_risk(self, n_params: int, n_combinations: int, perf_analysis: Dict) -> Dict:
        """Évaluation risque overfitting lié à complexité"""
        
        # Facteurs de risque
        high_param_count = n_params > 5
        high_combinations = n_combinations > 100
        high_performance = perf_analysis.get("is_sharpe", 0) > 3.0
        
        risk_factors = [high_param_count, high_combinations, high_performance]
        risk_score = sum(risk_factors) / len(risk_factors)
        
        if risk_score >= 0.7:
            risk_level = "High"
        elif risk_score >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "risk_factors": {
                "high_param_count": high_param_count,
                "high_combinations": high_combinations,
                "high_performance": high_performance
            }
        }
    
    def _score_to_level(self, score: float) -> str:
        """Conversion score vers niveau"""
        if score < 0.2:
            return "None"
        elif score < 0.4:
            return "Mild"
        elif score < 0.6:
            return "Moderate"
        else:
            return "Severe"
    
    def _generate_specific_recommendations(self, overfitting_score: Dict) -> List[str]:
        """Recommandations spécifiques basées sur l'analyse"""
        
        recommendations = []
        component_scores = overfitting_score["component_scores"]
        
        if component_scores.get("performance", 0) > 0.5:
            recommendations.append("📊 Réduire la dégradation IS->OOS (simplifier modèle)")
        
        if component_scores.get("sensitivity", 0) > 0.5:
            recommendations.append("🎛️ Stabiliser paramètres (réduire sensibilité)")
        
        if component_scores.get("white_reality", 0) > 0.5:
            recommendations.append("🎯 Corriger multiple testing bias")
        
        if component_scores.get("stability", 0) > 0.5:
            recommendations.append("📈 Améliorer stabilité temporelle")
        
        if component_scores.get("complexity", 0) > 0.7:
            recommendations.append("🧮 Simplifier modèle (moins de paramètres)")
        
        if component_scores.get("bootstrap", 0) > 0.5:
            recommendations.append("🎲 Améliorer significativité statistique")
        
        if component_scores.get("regime", 0) > 0.5:
            recommendations.append("📊 Améliorer robustesse multi-régimes")
        
        if not recommendations:
            recommendations.append("✅ Stratégie robuste - Continue monitoring")
        
        return recommendations

# Exemple d'utilisation
if __name__ == "__main__":
    
    logger.info("🧪 Test OverfittingDetector...")
    
    # Données simulées avec overfitting intentionnel
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-06-30', freq='1h')
    
    # Simuler données avec patterns qui ne persistent pas
    n = len(dates)
    returns_is = np.random.normal(0.001, 0.02, n//2)  # Bon sur première période
    returns_oos = np.random.normal(-0.0005, 0.025, n - n//2)  # Mauvais sur deuxième
    
    all_returns = np.concatenate([returns_is, returns_oos])
    data = pd.DataFrame({
        'close': (1 + all_returns).cumprod() * 100,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Stratégie avec overfitting (fonctionne bien sur première période seulement)
    def overfitted_strategy(data, params):
        lookback = params.get('lookback', 20)
        threshold = params.get('threshold', 0.01)
        
        returns = data['close'].pct_change()
        
        # Pattern qui marche sur 2023 mais pas 2024
        if data.index[0].year == 2023:
            signals = pd.Series(1, index=data.index)  # Always long en 2023
        else:
            signals = pd.Series(-1, index=data.index)  # Always short en 2024
        
        strategy_returns = signals.shift(1) * returns
        
        return {
            'returns': strategy_returns,
            'signals': signals
        }
    
    # Paramètres à tester
    param_ranges = {
        'lookback': [10, 20, 30, 50],
        'threshold': [0.005, 0.01, 0.02, 0.03]
    }
    base_params = {'lookback': 20, 'threshold': 0.01}
    
    # Test détecteur overfitting
    detector = OverfittingDetector(bootstrap_trials=100)
    
    analysis = detector.comprehensive_overfitting_analysis(
        overfitted_strategy, data, param_ranges, base_params
    )
    
    if analysis and not analysis.get("error"):
        print("\n=== ANALYSE OVERFITTING ===")
        
        score = analysis["overfitting_score"]["global_overfitting_score"]
        verdict = analysis["final_verdict"]
        
        print(f"Score overfitting: {score:.3f}")
        print(f"Niveau: {analysis['overfitting_score']['overfitting_level']}")
        print(f"Verdict: {verdict['verdict']}")
        print(f"Production suitable: {verdict['production_suitable']}")
        
        print(f"\nRecommandations:")
        for rec in verdict['specific_recommendations']:
            print(f"  {rec}")
        
        # Détails par composant
        print(f"\nScores par composant:")
        for component, score in analysis["overfitting_score"]["component_scores"].items():
            print(f"  {component}: {score:.3f}")
        
    else:
        print(f"❌ Erreur analyse: {analysis.get('error')}")