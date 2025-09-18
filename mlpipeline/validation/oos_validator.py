#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Out-of-Sample Validator pour validation stricte sans data leakage
Implémentation rigoureuse pour détecter l'overfitting
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
import mlflow

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class OutOfSampleValidator:
    """
    Validateur Out-of-Sample strict pour stratégies quantitatives
    
    Fonctionnalités:
    - Validation OOS avec séparation temporelle stricte
    - Détection de data leakage automatique
    - Tests de significativité statistique
    - Monte Carlo permutation tests
    - Métriques de performance ajustées au risque
    """
    
    def __init__(self,
                 oos_ratio: float = 0.3,              # 30% pour OOS
                 min_oos_periods: int = 100,          # Min 100 périodes OOS
                 confidence_level: float = 0.95,      # Niveau de confiance
                 monte_carlo_trials: int = 1000):     # Nombre de simulations MC
        
        self.oos_ratio = oos_ratio
        self.min_oos_periods = min_oos_periods
        self.confidence_level = confidence_level
        self.monte_carlo_trials = monte_carlo_trials
        
        # Résultats stockés
        self.validation_results = {}
        self.monte_carlo_results = {}
        
        logger.info(f"✅ OOS Validator initialisé:")
        logger.info(f"   - Ratio OOS: {oos_ratio*100:.1f}%")
        logger.info(f"   - Min périodes OOS: {min_oos_periods}")
        logger.info(f"   - Confiance: {confidence_level*100:.1f}%")
    
    def create_temporal_split(self, 
                             data: pd.DataFrame,
                             split_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Crée un split temporel strict pour validation OOS
        
        Args:
            data: DataFrame avec DatetimeIndex
            split_date: Date de split (optionnel, sinon calculé)
            
        Returns:
            Tuple (in_sample_data, out_of_sample_data)
        """
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame doit avoir un DatetimeIndex")
        
        # Calculer date de split
        if split_date:
            split_timestamp = pd.to_datetime(split_date)
        else:
            # Split automatique selon ratio
            total_periods = len(data)
            split_index = int(total_periods * (1 - self.oos_ratio))
            split_timestamp = data.index[split_index]
        
        # Créer splits
        in_sample = data[data.index < split_timestamp].copy()
        out_of_sample = data[data.index >= split_timestamp].copy()
        
        # Validation des splits
        if len(out_of_sample) < self.min_oos_periods:
            logger.warning(f"⚠️ OOS insuffisant: {len(out_of_sample)} < {self.min_oos_periods}")
        
        # Gap temporel pour éviter look-ahead
        gap_hours = 24  # 24h de gap minimum
        gap_timestamp = split_timestamp + timedelta(hours=gap_hours)
        out_of_sample = out_of_sample[out_of_sample.index >= gap_timestamp]
        
        logger.info(f"📊 Split créé: IS={len(in_sample)} OOS={len(out_of_sample)} (gap={gap_hours}h)")
        logger.info(f"   IS: {in_sample.index[0]} → {in_sample.index[-1]}")
        logger.info(f"   OOS: {out_of_sample.index[0]} → {out_of_sample.index[-1]}")
        
        return in_sample, out_of_sample
    
    def validate_strategy(self,
                         strategy_func,
                         data: pd.DataFrame,
                         strategy_params: Dict,
                         split_date: Optional[str] = None) -> Dict:
        """
        Valide une stratégie avec protocole OOS strict
        
        Args:
            strategy_func: Fonction de stratégie à valider
            data: Données complètes
            strategy_params: Paramètres de la stratégie
            split_date: Date de split (optionnel)
            
        Returns:
            Résultats de validation complets
        """
        
        logger.info("🔬 Début validation Out-of-Sample")
        
        # 1. Créer split temporel
        is_data, oos_data = self.create_temporal_split(data, split_date)
        
        try:
            # 2. Entraîner sur in-sample uniquement
            logger.info("🎯 Entraînement sur données in-sample...")
            is_results = strategy_func(is_data, strategy_params)
            
            if not is_results:
                return {"error": "Strategy training failed", "valid": False}
            
            # 3. Tester sur out-of-sample (JAMAIS VU)
            logger.info("📊 Test sur données out-of-sample...")
            oos_results = self._test_on_oos(is_results, oos_data, strategy_params)
            
            # 4. Analyse comparative
            comparison = self._compare_is_vs_oos(is_results, oos_results)
            
            # 5. Tests statistiques
            statistical_tests = self._run_statistical_tests(is_results, oos_results)
            
            # 6. Monte Carlo validation
            mc_results = self._monte_carlo_validation(
                is_results, oos_data, strategy_params, strategy_func
            )
            
            # 7. Détection data leakage
            leakage_tests = self._detect_data_leakage(is_results, oos_results, data)
            
            # 8. Compilation résultats
            validation_results = {
                "valid": True,
                "timestamp": datetime.now().isoformat(),
                "split_info": {
                    "is_periods": len(is_data),
                    "oos_periods": len(oos_data),
                    "oos_ratio": len(oos_data) / len(data),
                    "split_date": split_date or is_data.index[-1].isoformat()
                },
                "performance": {
                    "in_sample": self._extract_performance_metrics(is_results),
                    "out_of_sample": self._extract_performance_metrics(oos_results),
                    "comparison": comparison
                },
                "statistical_tests": statistical_tests,
                "monte_carlo": mc_results,
                "data_leakage": leakage_tests,
                "verdict": self._generate_oos_verdict(comparison, statistical_tests, mc_results)
            }
            
            self.validation_results = validation_results
            
            logger.info("✅ Validation OOS terminée")
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Erreur validation OOS: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "valid": False}
    
    def _test_on_oos(self, 
                    is_results: Dict,
                    oos_data: pd.DataFrame,
                    strategy_params: Dict) -> Dict:
        """
        Teste la stratégie entraînée sur données OOS
        """
        
        # Extraire le modèle ou la logique de la stratégie
        if 'model' in is_results and is_results['model']:
            model = is_results['model']
            
            # Générer signaux sur OOS
            if hasattr(model, 'generate_signals'):
                oos_signals = model.generate_signals(oos_data)
            elif hasattr(model, 'predict'):
                # Pour modèles ML
                try:
                    features = self._extract_features_for_prediction(oos_data)
                    predictions = model.predict(features)
                    oos_signals = pd.Series(np.sign(predictions), index=oos_data.index)
                except Exception as e:
                    logger.warning(f"Erreur prédiction ML: {e}")
                    oos_signals = pd.Series(0, index=oos_data.index)
            else:
                # Signaux par défaut
                oos_signals = pd.Series(0, index=oos_data.index)
        else:
            # Pas de modèle - utiliser logique de signaux directe
            oos_signals = self._generate_signals_from_params(oos_data, strategy_params)
        
        # Calculer performance sur OOS
        oos_returns = self._calculate_strategy_returns(oos_data, oos_signals)
        oos_metrics = self._calculate_performance_metrics(oos_returns, oos_signals)
        
        return {
            "signals": oos_signals,
            "returns": oos_returns,
            "metrics": oos_metrics,
            "data_period": f"{oos_data.index[0]} → {oos_data.index[-1]}"
        }
    
    def _extract_features_for_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrait les features nécessaires pour prédiction ML
        """
        
        # Features basiques - à adapter selon la stratégie
        features = pd.DataFrame(index=data.index)
        
        if 'close' in data.columns:
            features['price'] = data['close']
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
        
        if 'volume' in data.columns:
            features['volume'] = data['volume']
            features['volume_ma'] = data['volume'].rolling(20).mean()
        
        # RSI si disponible
        if 'rsi_14' in data.columns:
            features['rsi'] = data['rsi_14']
        
        # Nettoyage
        features = features.fillna(0)
        
        return features
    
    def _generate_signals_from_params(self, 
                                    data: pd.DataFrame, 
                                    params: Dict) -> pd.Series:
        """
        Génère des signaux basés sur paramètres de stratégie
        """
        
        signals = pd.Series(0, index=data.index)
        
        if 'close' not in data.columns:
            return signals
        
        # Simple moving average strategy (exemple)
        lookback = params.get('lookback_window', 20)
        z_entry = params.get('z_entry_threshold', 1.5)
        
        if len(data) > lookback:
            returns = data['close'].pct_change()
            ma = returns.rolling(lookback).mean()
            std = returns.rolling(lookback).std()
            zscore = (returns - ma) / (std + 1e-8)
            
            signals[zscore < -z_entry] = 1   # Long sur oversold
            signals[zscore > z_entry] = -1   # Short sur overbought
        
        return signals
    
    def _calculate_strategy_returns(self, 
                                  data: pd.DataFrame, 
                                  signals: pd.Series) -> pd.Series:
        """
        Calcule les returns d'une stratégie
        """
        
        if 'close' not in data.columns or len(signals) == 0:
            return pd.Series(dtype=float)
        
        # Returns des prix
        price_returns = data['close'].pct_change().fillna(0)
        
        # Aligner signaux
        aligned_signals = signals.reindex(price_returns.index, method='ffill').fillna(0)
        
        # Strategy returns avec lag (signal t-1 * return t)
        strategy_returns = aligned_signals.shift(1) * price_returns
        
        return strategy_returns.fillna(0)
    
    def _calculate_performance_metrics(self, 
                                     returns: pd.Series, 
                                     signals: pd.Series) -> Dict:
        """
        Calcule métriques de performance complètes
        """
        
        if len(returns) == 0:
            return {"error": "No returns to analyze"}
        
        # Returns statistics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trading metrics
        total_trades = len(signals[signals != 0])
        win_trades = len(returns[returns > 0])
        hit_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # Skew & Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # VaR
        var_5pct = returns.quantile(0.05)
        cvar_5pct = returns[returns <= var_5pct].mean()
        
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "hit_rate": float(hit_rate),
            "total_trades": int(total_trades),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "var_5pct": float(var_5pct),
            "cvar_5pct": float(cvar_5pct)
        }
    
    def _extract_performance_metrics(self, results: Dict) -> Dict:
        """
        Extrait métriques de performance depuis résultats stratégie
        """
        
        if 'metrics' in results:
            return results['metrics']
        elif 'returns' in results:
            signals = results.get('signals', pd.Series())
            return self._calculate_performance_metrics(results['returns'], signals)
        else:
            # Fallback sur métriques directes
            return {key: results.get(key, 0) for key in [
                'total_return', 'sharpe_ratio', 'max_drawdown', 'hit_rate'
            ]}
    
    def _compare_is_vs_oos(self, is_results: Dict, oos_results: Dict) -> Dict:
        """
        Compare performance in-sample vs out-of-sample
        """
        
        is_metrics = self._extract_performance_metrics(is_results)
        oos_metrics = self._extract_performance_metrics(oos_results)
        
        # Calcul dégradations
        degradations = {}
        for metric in ['sharpe_ratio', 'total_return', 'hit_rate']:
            is_val = is_metrics.get(metric, 0)
            oos_val = oos_metrics.get(metric, 0)
            
            if is_val != 0:
                degradation = (is_val - oos_val) / abs(is_val)
            else:
                degradation = np.inf if oos_val < 0 else 0
            
            degradations[f"{metric}_degradation"] = float(degradation)
        
        # Score de robustesse global
        avg_degradation = np.mean(list(degradations.values()))
        robustness_score = max(0, 1 - avg_degradation)
        
        return {
            "degradations": degradations,
            "robustness_score": float(robustness_score),
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics
        }
    
    def _run_statistical_tests(self, is_results: Dict, oos_results: Dict) -> Dict:
        """
        Tests statistiques IS vs OOS
        """
        
        tests = {}
        
        # Récupérer returns
        is_returns = is_results.get('returns', pd.Series())
        oos_returns = oos_results.get('returns', pd.Series())
        
        if len(is_returns) > 10 and len(oos_returns) > 10:
            
            # Test t de Student
            try:
                t_stat, t_pval = stats.ttest_ind(is_returns.dropna(), oos_returns.dropna())
                tests["t_test"] = {
                    "statistic": float(t_stat),
                    "p_value": float(t_pval),
                    "significant": t_pval < (1 - self.confidence_level)
                }
            except Exception as e:
                tests["t_test"] = {"error": str(e)}
            
            # Test de Kolmogorov-Smirnov
            try:
                ks_stat, ks_pval = stats.ks_2samp(is_returns.dropna(), oos_returns.dropna())
                tests["ks_test"] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_pval),
                    "significant": ks_pval < (1 - self.confidence_level)
                }
            except Exception as e:
                tests["ks_test"] = {"error": str(e)}
            
            # Test de Wilcoxon (non-paramétrique)
            try:
                # Prendre des échantillons de même taille
                min_len = min(len(is_returns), len(oos_returns))
                is_sample = is_returns.dropna().iloc[:min_len]
                oos_sample = oos_returns.dropna().iloc[:min_len]
                
                if len(is_sample) > 5 and len(oos_sample) > 5:
                    wilcox_stat, wilcox_pval = stats.wilcoxon(is_sample, oos_sample)
                    tests["wilcoxon_test"] = {
                        "statistic": float(wilcox_stat),
                        "p_value": float(wilcox_pval),
                        "significant": wilcox_pval < (1 - self.confidence_level)
                    }
            except Exception as e:
                tests["wilcoxon_test"] = {"error": str(e)}
        
        return tests
    
    def _monte_carlo_validation(self, 
                               is_results: Dict,
                               oos_data: pd.DataFrame,
                               strategy_params: Dict,
                               strategy_func) -> Dict:
        """
        Validation Monte Carlo avec permutations
        """
        
        logger.info(f"🎲 Monte Carlo validation ({self.monte_carlo_trials} trials)...")
        
        # Performance OOS observée
        oos_results = self._test_on_oos(is_results, oos_data, strategy_params)
        observed_sharpe = oos_results['metrics'].get('sharpe_ratio', 0)
        
        # Simulations Monte Carlo
        mc_sharpes = []
        
        for trial in range(self.monte_carlo_trials):
            try:
                # Permutation aléatoire des returns
                shuffled_data = oos_data.copy()
                shuffled_data['close'] = oos_data['close'].sample(frac=1).values
                
                # Test sur données permutées
                mc_result = self._test_on_oos(is_results, shuffled_data, strategy_params)
                mc_sharpe = mc_result['metrics'].get('sharpe_ratio', 0)
                mc_sharpes.append(mc_sharpe)
                
            except Exception:
                mc_sharpes.append(0)  # Fallback
        
        # Analyse résultats Monte Carlo
        mc_sharpes = np.array(mc_sharpes)
        percentile_rank = stats.percentileofscore(mc_sharpes, observed_sharpe)
        
        # P-value (bilatéral)
        p_value = 2 * min(percentile_rank, 100 - percentile_rank) / 100
        
        # Intervalles de confiance
        ci_lower = np.percentile(mc_sharpes, 2.5)
        ci_upper = np.percentile(mc_sharpes, 97.5)
        
        return {
            "observed_sharpe": float(observed_sharpe),
            "mc_mean_sharpe": float(np.mean(mc_sharpes)),
            "mc_std_sharpe": float(np.std(mc_sharpes)),
            "percentile_rank": float(percentile_rank),
            "p_value": float(p_value),
            "confidence_interval": [float(ci_lower), float(ci_upper)],
            "significant": p_value < (1 - self.confidence_level),
            "trials": len(mc_sharpes)
        }
    
    def _detect_data_leakage(self, 
                           is_results: Dict,
                           oos_results: Dict,
                           full_data: pd.DataFrame) -> Dict:
        """
        Détecte les signes de data leakage
        """
        
        leakage_indicators = {}
        
        # 1. Performance trop parfaite
        is_metrics = self._extract_performance_metrics(is_results)
        oos_metrics = self._extract_performance_metrics(oos_results)
        
        is_sharpe = is_metrics.get('sharpe_ratio', 0)
        is_hit_rate = is_metrics.get('hit_rate', 0)
        
        leakage_indicators["perfect_performance"] = {
            "suspicious_sharpe": is_sharpe > 5.0,  # Sharpe > 5 suspect
            "perfect_hit_rate": is_hit_rate > 0.9,  # Hit rate > 90% suspect
            "zero_drawdown": is_metrics.get('max_drawdown', 0) == 0
        }
        
        # 2. Dégradation anormale
        comparison = self._compare_is_vs_oos(is_results, oos_results)
        degradations = comparison['degradations']
        
        leakage_indicators["degradation_patterns"] = {
            "severe_sharpe_drop": degradations.get('sharpe_ratio_degradation', 0) > 0.8,
            "severe_return_drop": degradations.get('total_return_degradation', 0) > 0.8,
            "negative_oos_performance": oos_metrics.get('sharpe_ratio', 0) < -1.0
        }
        
        # 3. Patterns temporels suspects
        if 'returns' in is_results and 'returns' in oos_results:
            is_returns = is_results['returns']
            oos_returns = oos_results['returns']
            
            # Corrélation anormale avec période future
            if len(is_returns) > 50 and len(oos_returns) > 50:
                try:
                    # Corrélation avec returns futurs
                    future_corr = is_returns.tail(50).corr(oos_returns.head(50))
                    leakage_indicators["temporal_correlation"] = {
                        "future_correlation": float(future_corr) if not np.isnan(future_corr) else 0,
                        "suspicious_correlation": abs(future_corr) > 0.3 if not np.isnan(future_corr) else False
                    }
                except Exception:
                    leakage_indicators["temporal_correlation"] = {"error": "Could not compute"}
        
        # 4. Score global de leakage
        all_flags = []
        for category in leakage_indicators.values():
            if isinstance(category, dict):
                all_flags.extend([v for v in category.values() if isinstance(v, bool)])
        
        leakage_score = sum(all_flags) / len(all_flags) if all_flags else 0
        
        leakage_indicators["leakage_score"] = float(leakage_score)
        leakage_indicators["leakage_detected"] = leakage_score > 0.3  # > 30% de flags
        
        return leakage_indicators
    
    def _generate_oos_verdict(self, 
                            comparison: Dict,
                            statistical_tests: Dict,
                            mc_results: Dict) -> Dict:
        """
        Génère le verdict final de validation OOS
        """
        
        # Critères de validation
        criteria = {
            "positive_oos_performance": comparison['oos_metrics'].get('sharpe_ratio', 0) > 0,
            "acceptable_degradation": comparison['robustness_score'] > 0.5,
            "statistically_significant": mc_results.get('significant', False),
            "no_severe_overfitting": comparison['degradations'].get('sharpe_ratio_degradation', 1) < 0.7
        }
        
        # Score de validation
        validation_score = sum(criteria.values()) / len(criteria)
        
        # Verdict textuel
        if validation_score >= 0.8:
            verdict_text = "✅ VALIDATION RÉUSSIE - Stratégie robuste OOS"
        elif validation_score >= 0.6:
            verdict_text = "⚠️ VALIDATION PARTIELLE - Surveillance nécessaire"
        elif validation_score >= 0.4:
            verdict_text = "🟡 VALIDATION FAIBLE - Optimisation requise"
        else:
            verdict_text = "❌ ÉCHEC VALIDATION - Stratégie non viable"
        
        # Recommandations
        recommendations = []
        if not criteria["positive_oos_performance"]:
            recommendations.append("📈 Améliorer performance OOS")
        if not criteria["acceptable_degradation"]:
            recommendations.append("🔧 Réduire overfitting")
        if not criteria["statistically_significant"]:
            recommendations.append("📊 Améliorer significativité statistique")
        if not criteria["no_severe_overfitting"]:
            recommendations.append("⚠️ Réviser optimisation paramètres")
        
        return {
            "validation_score": float(validation_score),
            "verdict": verdict_text,
            "criteria": criteria,
            "recommendations": recommendations,
            "ready_for_production": validation_score >= 0.7
        }
    
    def save_results(self, filepath: str = "oos_validation_results.json") -> str:
        """
        Sauvegarde les résultats de validation
        """
        
        if not self.validation_results:
            logger.warning("Aucun résultat à sauvegarder")
            return ""
        
        # Conversion pour JSON
        results_json = self._make_json_serializable(self.validation_results)
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"💾 Résultats sauvés: {filepath}")
        return filepath
    
    def _make_json_serializable(self, obj):
        """Convertit objets non-JSON en format sérialisable"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

# Exemple d'utilisation
if __name__ == "__main__":
    
    # Test avec données simulées
    logger.info("🧪 Test OOS Validator...")
    
    # Génération données test
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='1h')
    
    # Simuler une stratégie avec overfitting
    returns_is = np.random.normal(0.001, 0.02, len(dates)//2)  # Bon IS
    returns_oos = np.random.normal(-0.0005, 0.025, len(dates) - len(dates)//2)  # Mauvais OOS
    all_returns = np.concatenate([returns_is, returns_oos])
    
    data = pd.DataFrame({
        'close': (1 + all_returns).cumprod() * 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Stratégie test avec overfitting simulé
    def overfitted_strategy(data, params):
        # Stratégie qui fonctionne bien sur première moitié seulement
        lookback = params.get('lookback_window', 20)
        
        returns = data['close'].pct_change().fillna(0)
        
        # "Overfitting" sur première période
        if data.index[0].year == 2023:
            # Bon performance sur 2023 (IS)
            signals = pd.Series(1, index=data.index)  # Always long
        else:
            # Mauvais performance sur 2024 (OOS)
            signals = pd.Series(-1, index=data.index)  # Always short
        
        strategy_returns = signals.shift(1) * returns
        
        return {
            'returns': strategy_returns,
            'signals': signals,
            'model': None
        }
    
    # Test validation OOS
    validator = OutOfSampleValidator(
        oos_ratio=0.3,
        monte_carlo_trials=100  # Réduit pour demo
    )
    
    results = validator.validate_strategy(
        overfitted_strategy,
        data,
        {'lookback_window': 20}
    )
    
    if results['valid']:
        print("\n=== RÉSULTATS VALIDATION OOS ===")
        print(f"IS Sharpe: {results['performance']['in_sample']['sharpe_ratio']:.3f}")
        print(f"OOS Sharpe: {results['performance']['out_of_sample']['sharpe_ratio']:.3f}")
        print(f"Dégradation: {results['performance']['comparison']['degradations']['sharpe_ratio_degradation']:.1%}")
        print(f"Score robustesse: {results['performance']['comparison']['robustness_score']:.3f}")
        print(f"Monte Carlo p-value: {results['monte_carlo']['p_value']:.3f}")
        print(f"Verdict: {results['verdict']['verdict']}")
        
        # Sauvegarde
        validator.save_results("test_oos_results.json")
    else:
        print(f"❌ Erreur validation: {results.get('error')}")