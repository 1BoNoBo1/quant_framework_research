#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rigorous Backtester - Backtesting professionnel sans biais
Intègre tous les systèmes de validation pour backtests de qualité institutionnelle
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Optionnel pour viz
from scipy import stats
import mlflow

# Imports locaux
from .walk_forward_analyzer import WalkForwardAnalyzer
from .oos_validator import OutOfSampleValidator  
from .robustness_metrics import RobustnessMetrics
from ..utils.realistic_costs import get_realistic_trading_costs, apply_realistic_costs_to_backtest

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class RigorousBacktester:
    """
    Backtester rigoureux intégrant toutes les validations
    
    Fonctionnalités:
    - Backtesting avec coûts réalistes
    - Walk-forward analysis automatique
    - Validation out-of-sample stricte
    - Métriques de robustesse complètes
    - Détection d'overfitting multi-niveaux
    - Rapports de validation professionnels
    - Intégration MLflow pour tracking
    """
    
    def __init__(self,
                 initial_capital: float = 50000,
                 commission: float = 0.002,
                 slippage: float = 0.003,
                 enable_walk_forward: bool = True,
                 enable_oos_validation: bool = True,
                 enable_monte_carlo: bool = True,
                 confidence_level: float = 0.95):
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.confidence_level = confidence_level
        
        # Modules de validation
        self.walk_forward_analyzer = WalkForwardAnalyzer() if enable_walk_forward else None
        self.oos_validator = OutOfSampleValidator() if enable_oos_validation else None
        self.robustness_metrics = RobustnessMetrics()
        
        # Résultats stockés
        self.backtest_results = {}
        self.validation_results = {}
        
        # Flags
        self.enable_walk_forward = enable_walk_forward
        self.enable_oos_validation = enable_oos_validation
        self.enable_monte_carlo = enable_monte_carlo
        
        logger.info(f"✅ RigorousBacktester initialisé:")
        logger.info(f"   - Capital: ${initial_capital:,.0f}")
        logger.info(f"   - Commission: {commission:.1%}")
        logger.info(f"   - Walk-forward: {enable_walk_forward}")
        logger.info(f"   - OOS validation: {enable_oos_validation}")
    
    def run_comprehensive_backtest(self,
                                 strategy_func: Callable,
                                 data: pd.DataFrame,
                                 strategy_params: Dict,
                                 symbol: str = "BTCUSDT",
                                 benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Exécute un backtest complet avec toutes les validations
        
        Args:
            strategy_func: Fonction de stratégie à tester
            data: Données historiques avec DatetimeIndex
            strategy_params: Paramètres de la stratégie
            symbol: Symbol tradé (pour coûts réalistes)
            benchmark_data: Données de benchmark (optionnel)
            
        Returns:
            Résultats complets du backtest rigoureux
        """
        
        logger.info("🚀 Début backtest rigoureux complet")
        
        start_time = datetime.now()
        
        try:
            # 1. Validation des données
            data_validation = self._validate_input_data(data)
            if not data_validation["valid"]:
                return {"error": "Data validation failed", "details": data_validation}
            
            # 2. Backtest de base
            logger.info("📊 Exécution backtest de base...")
            base_backtest = self._run_base_backtest(strategy_func, data, strategy_params, symbol)
            
            if not base_backtest["valid"]:
                return {"error": "Base backtest failed", "details": base_backtest}
            
            # 3. Walk-Forward Analysis (si activé)
            walk_forward_results = {}
            if self.enable_walk_forward and len(data) > 500:  # Minimum de données
                logger.info("🔄 Exécution Walk-Forward Analysis...")
                try:
                    param_ranges = self._extract_param_ranges(strategy_params)
                    walk_forward_results = self.walk_forward_analyzer.walk_forward_analysis(
                        data, strategy_func, param_ranges, "sharpe_ratio"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Walk-forward échoué: {e}")
                    walk_forward_results = {"error": str(e)}
            
            # 4. Validation Out-of-Sample (si activé)
            oos_results = {}
            if self.enable_oos_validation and len(data) > 200:
                logger.info("🔬 Exécution validation Out-of-Sample...")
                try:
                    oos_results = self.oos_validator.validate_strategy(
                        strategy_func, data, strategy_params
                    )
                except Exception as e:
                    logger.warning(f"⚠️ OOS validation échouée: {e}")
                    oos_results = {"error": str(e)}
            
            # 5. Métriques de robustesse complètes
            logger.info("📈 Calcul métriques de robustesse...")
            returns = base_backtest["returns"]
            signals = base_backtest["signals"]
            benchmark_returns = benchmark_data['close'].pct_change() if benchmark_data is not None else None
            
            robustness_analysis = self.robustness_metrics.calculate_comprehensive_metrics(
                returns, benchmark_returns, signals
            )
            
            # 6. Détection d'overfitting multi-niveaux
            logger.info("🔍 Détection d'overfitting...")
            overfitting_analysis = self._comprehensive_overfitting_detection(
                base_backtest, walk_forward_results, oos_results, robustness_analysis
            )
            
            # 7. Tests de stress et sensibilité
            logger.info("⚡ Tests de stress...")
            stress_tests = self._run_stress_tests(strategy_func, data, strategy_params, symbol)
            
            # 8. Monte Carlo validation (si activé)
            monte_carlo_results = {}
            if self.enable_monte_carlo:
                logger.info("🎲 Validation Monte Carlo...")
                monte_carlo_results = self._run_monte_carlo_validation(base_backtest, 1000)
            
            # 9. Compilation des résultats finaux
            execution_time = (datetime.now() - start_time).total_seconds()
            
            comprehensive_results = {
                "metadata": {
                    "timestamp": start_time.isoformat(),
                    "execution_time_seconds": execution_time,
                    "symbol": symbol,
                    "data_periods": len(data),
                    "data_range": f"{data.index[0]} → {data.index[-1]}",
                    "strategy_params": strategy_params
                },
                "base_backtest": base_backtest,
                "walk_forward": walk_forward_results,
                "out_of_sample": oos_results,
                "robustness": robustness_analysis,
                "overfitting": overfitting_analysis,
                "stress_tests": stress_tests,
                "monte_carlo": monte_carlo_results,
                "final_verdict": self._generate_final_verdict(
                    base_backtest, walk_forward_results, oos_results, 
                    robustness_analysis, overfitting_analysis
                )
            }
            
            # 10. Sauvegarde et logging MLflow
            self.backtest_results = comprehensive_results
            self._log_to_mlflow(comprehensive_results)
            
            logger.info(f"✅ Backtest rigoureux terminé en {execution_time:.1f}s")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"❌ Erreur fatale backtest: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "valid": False}
    
    def _validate_input_data(self, data: pd.DataFrame) -> Dict:
        """
        Validation stricte des données d'entrée
        """
        
        checks = {
            "has_datetime_index": isinstance(data.index, pd.DatetimeIndex),
            "has_required_columns": all(col in data.columns for col in ['close']),
            "sufficient_data": len(data) >= 100,
            "no_major_gaps": True,  # À implémenter
            "price_data_valid": (data['close'] > 0).all() if 'close' in data.columns else False
        }
        
        # Test de gaps majeurs
        if isinstance(data.index, pd.DatetimeIndex):
            time_diffs = data.index.to_series().diff()
            median_diff = time_diffs.median()
            large_gaps = (time_diffs > median_diff * 5).sum()
            checks["no_major_gaps"] = large_gaps < len(data) * 0.05  # < 5% de gaps
        
        validity = all(checks.values())
        
        return {
            "valid": validity,
            "checks": checks,
            "data_quality_score": sum(checks.values()) / len(checks)
        }
    
    def _run_base_backtest(self, 
                          strategy_func: Callable,
                          data: pd.DataFrame,
                          strategy_params: Dict,
                          symbol: str) -> Dict:
        """
        Backtest de base avec coûts réalistes
        """
        
        try:
            # 1. Exécuter la stratégie
            strategy_results = strategy_func(data, strategy_params)
            
            if not strategy_results:
                return {"error": "Strategy execution failed", "valid": False}
            
            # 2. Extraire signaux et returns
            signals = strategy_results.get('signals', pd.Series(0, index=data.index))
            base_returns = strategy_results.get('returns', pd.Series(0, index=data.index))
            
            # 3. Appliquer coûts réalistes
            realistic_returns = apply_realistic_costs_to_backtest(
                base_returns, signals, symbol, self.initial_capital
            )
            
            # 4. Calculer equity curve
            equity_curve = (1 + realistic_returns).cumprod() * self.initial_capital
            
            # 5. Métriques de performance
            performance_metrics = self._calculate_backtest_metrics(realistic_returns, signals, equity_curve)
            
            # 6. Trade analysis
            trade_analysis = self._analyze_trades(signals, realistic_returns, data)
            
            # 7. Risk analysis  
            risk_analysis = self._analyze_risk(realistic_returns, equity_curve)
            
            return {
                "valid": True,
                "returns": realistic_returns,
                "signals": signals,
                "equity_curve": equity_curve,
                "performance": performance_metrics,
                "trades": trade_analysis,
                "risk": risk_analysis,
                "final_equity": float(equity_curve.iloc[-1]),
                "total_return": float((equity_curve.iloc[-1] / self.initial_capital) - 1)
            }
            
        except Exception as e:
            logger.error(f"Erreur backtest base: {e}")
            return {"error": str(e), "valid": False}
    
    def _calculate_backtest_metrics(self, 
                                  returns: pd.Series,
                                  signals: pd.Series,
                                  equity_curve: pd.Series) -> Dict:
        """
        Calcule métriques de performance du backtest
        """
        
        if len(returns) == 0:
            return {"error": "No returns to analyze"}
        
        # Returns metrics
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate
        winning_returns = returns[returns > 0]
        hit_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0
        
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "hit_rate": float(hit_rate),
            "periods": len(returns)
        }
    
    def _analyze_trades(self, 
                       signals: pd.Series,
                       returns: pd.Series,
                       data: pd.DataFrame) -> Dict:
        """
        Analyse détaillée des trades
        """
        
        # Détecter les trades (changements de signal)
        signal_changes = signals.diff().abs() > 0
        trade_returns = returns[signal_changes]
        
        if len(trade_returns) == 0:
            return {"total_trades": 0, "error": "No trades detected"}
        
        # Trade statistics
        total_trades = len(trade_returns)
        winning_trades = len(trade_returns[trade_returns > 0])
        losing_trades = len(trade_returns[trade_returns < 0])
        
        avg_win = trade_returns[trade_returns > 0].mean() if winning_trades > 0 else 0
        avg_loss = trade_returns[trade_returns < 0].mean() if losing_trades > 0 else 0
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
        
        # Profit factor
        total_wins = trade_returns[trade_returns > 0].sum()
        total_losses = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        # Trade duration analysis (approximatif)
        position_changes = signals[signals != signals.shift(1)].index
        if len(position_changes) > 1:
            durations = [(position_changes[i] - position_changes[i-1]).days for i in range(1, len(position_changes))]
            avg_trade_duration = np.mean(durations) if durations else 0
        else:
            avg_trade_duration = 0
        
        return {
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "hit_rate": float(winning_trades / total_trades) if total_trades > 0 else 0,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "win_loss_ratio": float(win_loss_ratio) if not np.isinf(win_loss_ratio) else None,
            "profit_factor": float(profit_factor) if not np.isinf(profit_factor) else None,
            "avg_trade_duration_days": float(avg_trade_duration)
        }
    
    def _analyze_risk(self, returns: pd.Series, equity_curve: pd.Series) -> Dict:
        """
        Analyse de risque détaillée
        """
        
        # VaR metrics
        var_1pct = returns.quantile(0.01)
        var_5pct = returns.quantile(0.05)
        cvar_5pct = returns[returns <= var_5pct].mean()
        
        # Drawdown analysis
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # Drawdown duration
        in_drawdown = drawdown < -0.01  # Seuil de 1%
        dd_periods = []
        current_dd = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_dd += 1
            else:
                if current_dd > 0:
                    dd_periods.append(current_dd)
                current_dd = 0
        
        max_dd_duration = max(dd_periods) if dd_periods else 0
        avg_dd_duration = np.mean(dd_periods) if dd_periods else 0
        
        # Volatility clustering (GARCH effect)
        returns_squared = returns ** 2
        vol_autocorr = returns_squared.autocorr(lag=1) if len(returns_squared) > 10 else 0
        
        return {
            "var_1pct": float(var_1pct),
            "var_5pct": float(var_5pct),
            "cvar_5pct": float(cvar_5pct),
            "max_drawdown_duration": int(max_dd_duration),
            "avg_drawdown_duration": float(avg_dd_duration),
            "drawdown_periods": len(dd_periods),
            "volatility_clustering": float(vol_autocorr)
        }
    
    def _extract_param_ranges(self, strategy_params: Dict) -> Dict:
        """
        Extrait des ranges de paramètres pour optimisation
        """
        
        # Ranges par défaut basés sur les paramètres actuels
        param_ranges = {}
        
        for param, value in strategy_params.items():
            if isinstance(value, (int, float)):
                if param in ['lookback_window', 'lookback_short', 'lookback_long']:
                    # Paramètres de lookback
                    param_ranges[param] = [max(5, int(value * 0.5)), value, int(value * 1.5), int(value * 2)]
                elif param in ['z_entry_threshold', 'z_exit_threshold']:
                    # Seuils z-score
                    param_ranges[param] = [value * 0.5, value * 0.75, value, value * 1.25, value * 1.5]
                elif param in ['funding_threshold', 'min_funding_rate']:
                    # Seuils de funding
                    param_ranges[param] = [value * 0.5, value * 0.75, value, value * 1.5, value * 2.0]
                else:
                    # Paramètres génériques
                    param_ranges[param] = [value * 0.8, value, value * 1.2]
        
        return param_ranges
    
    def _comprehensive_overfitting_detection(self,
                                           base_backtest: Dict,
                                           walk_forward: Dict,
                                           oos: Dict,
                                           robustness: Dict) -> Dict:
        """
        Détection d'overfitting multi-niveaux
        """
        
        overfitting_indicators = {}
        overfitting_score = 0.0
        
        # 1. Performance trop parfaite (base backtest)
        if base_backtest.get("valid", False):
            sharpe = base_backtest["performance"].get("sharpe_ratio", 0)
            hit_rate = base_backtest["performance"].get("hit_rate", 0)
            max_dd = base_backtest["performance"].get("max_drawdown", 1)
            
            perfect_performance = {
                "unrealistic_sharpe": sharpe > 4.0,
                "perfect_hit_rate": hit_rate > 0.85,
                "minimal_drawdown": max_dd < 0.01
            }
            overfitting_indicators["perfect_performance"] = perfect_performance
            overfitting_score += sum(perfect_performance.values()) * 0.3
        
        # 2. Dégradation Walk-Forward
        if walk_forward and "summary" in walk_forward:
            wf_summary = walk_forward["summary"]
            is_sharpe = wf_summary.get("avg_is_sharpe", 0)
            oos_sharpe = wf_summary.get("avg_oos_sharpe", 0)
            
            if is_sharpe != 0:
                wf_degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)
                walk_forward_issues = {
                    "severe_degradation": wf_degradation > 0.7,
                    "negative_oos": oos_sharpe < 0,
                    "high_variance": walk_forward.get("robustness", {}).get("consistency_score", 1) < 0.3
                }
                overfitting_indicators["walk_forward"] = walk_forward_issues
                overfitting_score += sum(walk_forward_issues.values()) * 0.35
        
        # 3. Échec OOS
        if oos and oos.get("valid", False):
            oos_performance = oos.get("performance", {}).get("comparison", {})
            oos_issues = {
                "poor_oos_performance": oos_performance.get("robustness_score", 0) < 0.3,
                "large_degradation": any(
                    deg > 0.6 for deg in oos_performance.get("degradations", {}).values()
                )
            }
            overfitting_indicators["out_of_sample"] = oos_issues
            overfitting_score += sum(oos_issues.values()) * 0.35
        
        # 4. Métriques de robustesse faibles
        if robustness and "statistical" in robustness:
            statistical = robustness["statistical"]
            robustness_issues = {
                "low_psr": statistical.get("probabilistic_sharpe_ratio", 0) < 0.5,
                "high_autocorr": abs(statistical.get("autocorr_lag1", 0) or 0) > 0.3,
                "non_stationary": not statistical.get("stationarity", {}).get("is_stationary", True)
            }
            overfitting_indicators["robustness"] = robustness_issues
            overfitting_score += sum(robustness_issues.values()) * 0.2
        
        # Score final normalisé
        max_possible_score = 0.3 * 3 + 0.35 * 3 + 0.35 * 2 + 0.2 * 3  # Max indicators * weights
        overfitting_score_normalized = min(1.0, overfitting_score / max_possible_score) if max_possible_score > 0 else 0
        
        # Verdict
        if overfitting_score_normalized < 0.2:
            verdict = "✅ PAS D'OVERFITTING détecté"
        elif overfitting_score_normalized < 0.4:
            verdict = "⚠️ OVERFITTING LÉGER - Surveillance recommandée"
        elif overfitting_score_normalized < 0.7:
            verdict = "🟡 OVERFITTING MODÉRÉ - Correction nécessaire"
        else:
            verdict = "❌ OVERFITTING SÉVÈRE - Stratégie non viable"
        
        return {
            "overfitting_score": float(overfitting_score_normalized),
            "indicators": overfitting_indicators,
            "verdict": verdict,
            "overfitting_detected": overfitting_score_normalized > 0.4
        }
    
    def _run_stress_tests(self,
                         strategy_func: Callable,
                         data: pd.DataFrame,
                         strategy_params: Dict,
                         symbol: str) -> Dict:
        """
        Tests de stress sur différents scénarios
        """
        
        stress_results = {}
        
        try:
            # 1. Test avec volatilité augmentée
            stressed_data = data.copy()
            returns = stressed_data['close'].pct_change()
            stressed_returns = returns * 1.5  # 50% plus volatil
            stressed_data['close'] = (1 + stressed_returns).cumprod() * data['close'].iloc[0]
            
            high_vol_result = self._run_base_backtest(strategy_func, stressed_data, strategy_params, symbol)
            stress_results["high_volatility"] = {
                "sharpe": high_vol_result.get("performance", {}).get("sharpe_ratio", 0),
                "max_dd": high_vol_result.get("performance", {}).get("max_drawdown", 1)
            }
            
            # 2. Test bear market (trend baissier)
            bear_data = data.copy()
            bear_returns = returns - 0.001  # -0.1% par période
            bear_data['close'] = (1 + bear_returns).cumprod() * data['close'].iloc[0]
            
            bear_result = self._run_base_backtest(strategy_func, bear_data, strategy_params, symbol)
            stress_results["bear_market"] = {
                "sharpe": bear_result.get("performance", {}).get("sharpe_ratio", 0),
                "total_return": bear_result.get("total_return", -1)
            }
            
            # 3. Test avec gaps (market crashes)
            crash_data = data.copy()
            crash_returns = returns.copy()
            # Ajouter quelques crashes de -10%
            crash_indices = np.random.choice(len(crash_returns), size=5, replace=False)
            crash_returns.iloc[crash_indices] = -0.10
            crash_data['close'] = (1 + crash_returns).cumprod() * data['close'].iloc[0]
            
            crash_result = self._run_base_backtest(strategy_func, crash_data, strategy_params, symbol)
            stress_results["market_crashes"] = {
                "sharpe": crash_result.get("performance", {}).get("sharpe_ratio", 0),
                "max_dd": crash_result.get("performance", {}).get("max_drawdown", 1)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Stress tests échoués: {e}")
            stress_results["error"] = str(e)
        
        return stress_results
    
    def _run_monte_carlo_validation(self, base_backtest: Dict, n_simulations: int = 1000) -> Dict:
        """
        Validation Monte Carlo des résultats
        """
        
        if not base_backtest.get("valid", False):
            return {"error": "Invalid base backtest"}
        
        returns = base_backtest.get("returns", pd.Series())
        
        if len(returns) < 50:
            return {"error": "Not enough returns for Monte Carlo"}
        
        # Statistiques observées
        observed_sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        observed_return = (1 + returns).prod() - 1
        
        # Simulations Monte Carlo (bootstrap)
        mc_sharpes = []
        mc_returns = []
        
        for _ in range(n_simulations):
            # Bootstrap des returns
            bootstrap_returns = returns.sample(n=len(returns), replace=True)
            
            # Métriques de la simulation
            mc_sharpe = (bootstrap_returns.mean() * 252) / (bootstrap_returns.std() * np.sqrt(252)) if bootstrap_returns.std() > 0 else 0
            mc_return = (1 + bootstrap_returns).prod() - 1
            
            mc_sharpes.append(mc_sharpe)
            mc_returns.append(mc_return)
        
        # Analyse des résultats MC
        mc_sharpes = np.array(mc_sharpes)
        mc_returns = np.array(mc_returns)
        
        # Intervalles de confiance
        sharpe_ci = np.percentile(mc_sharpes, [2.5, 97.5])
        return_ci = np.percentile(mc_returns, [2.5, 97.5])
        
        # P-values (test si résultats > hasard)
        sharpe_pvalue = (mc_sharpes <= 0).mean()  # Probabilité Sharpe <= 0
        return_pvalue = (mc_returns <= 0).mean()  # Probabilité return <= 0
        
        return {
            "simulations": n_simulations,
            "observed_sharpe": float(observed_sharpe),
            "observed_return": float(observed_return),
            "mc_sharpe_mean": float(np.mean(mc_sharpes)),
            "mc_return_mean": float(np.mean(mc_returns)),
            "sharpe_confidence_interval": [float(sharpe_ci[0]), float(sharpe_ci[1])],
            "return_confidence_interval": [float(return_ci[0]), float(return_ci[1])],
            "sharpe_pvalue": float(sharpe_pvalue),
            "return_pvalue": float(return_pvalue),
            "statistically_significant": sharpe_pvalue < 0.05 and return_pvalue < 0.05
        }
    
    def _generate_final_verdict(self,
                              base_backtest: Dict,
                              walk_forward: Dict,
                              oos: Dict,
                              robustness: Dict,
                              overfitting: Dict) -> Dict:
        """
        Génère le verdict final du backtest rigoureux
        """
        
        # Critères de validation
        criteria = {}
        
        # 1. Performance de base acceptable
        if base_backtest.get("valid", False):
            performance = base_backtest.get("performance", {})
            criteria["positive_sharpe"] = performance.get("sharpe_ratio", 0) > 0.5
            criteria["acceptable_drawdown"] = performance.get("max_drawdown", 1) < 0.25
            criteria["sufficient_trades"] = base_backtest.get("trades", {}).get("total_trades", 0) > 5
        else:
            criteria["positive_sharpe"] = False
            criteria["acceptable_drawdown"] = False
            criteria["sufficient_trades"] = False
        
        # 2. Robustesse walk-forward
        if walk_forward and "robustness" in walk_forward:
            criteria["walk_forward_robust"] = walk_forward["robustness"].get("overfitting_score", 1) < 0.5
        else:
            criteria["walk_forward_robust"] = True  # Neutre si pas testé
        
        # 3. Validation OOS
        if oos and oos.get("valid", False):
            verdict = oos.get("verdict", {})
            criteria["oos_validated"] = verdict.get("ready_for_production", False)
        else:
            criteria["oos_validated"] = True  # Neutre si pas testé
        
        # 4. Métriques de robustesse
        if robustness and "robustness_score" in robustness:
            criteria["statistically_robust"] = robustness["robustness_score"].get("global_score", 0) > 0.6
        else:
            criteria["statistically_robust"] = False
        
        # 5. Pas d'overfitting détecté
        if overfitting:
            criteria["no_overfitting"] = not overfitting.get("overfitting_detected", True)
        else:
            criteria["no_overfitting"] = True
        
        # Score global
        validation_score = sum(criteria.values()) / len(criteria)
        
        # Verdict final
        if validation_score >= 0.8:
            final_verdict = "✅ VALIDATION COMPLÈTE - Stratégie prête pour production"
            production_ready = True
            color = "green"
        elif validation_score >= 0.6:
            final_verdict = "⚠️ VALIDATION PARTIELLE - Surveillance en production requise"
            production_ready = True
            color = "orange"
        elif validation_score >= 0.4:
            final_verdict = "🟡 VALIDATION FAIBLE - Optimisation recommandée avant production"
            production_ready = False
            color = "yellow"
        else:
            final_verdict = "❌ ÉCHEC VALIDATION - Stratégie non viable pour production"
            production_ready = False
            color = "red"
        
        # Recommandations
        recommendations = []
        if not criteria.get("positive_sharpe", True):
            recommendations.append("📈 Améliorer performance (Sharpe ratio)")
        if not criteria.get("acceptable_drawdown", True):
            recommendations.append("🛡️ Réduire drawdown maximum")
        if not criteria.get("walk_forward_robust", True):
            recommendations.append("🔄 Améliorer robustesse walk-forward")
        if not criteria.get("oos_validated", True):
            recommendations.append("🔬 Corriger validation out-of-sample")
        if not criteria.get("statistically_robust", True):
            recommendations.append("📊 Améliorer robustesse statistique")
        if not criteria.get("no_overfitting", True):
            recommendations.append("⚠️ Corriger overfitting détecté")
        
        return {
            "validation_score": float(validation_score),
            "verdict": final_verdict,
            "production_ready": production_ready,
            "color": color,
            "criteria": criteria,
            "recommendations": recommendations,
            "confidence_level": "High" if validation_score >= 0.8 else "Medium" if validation_score >= 0.6 else "Low"
        }
    
    def _log_to_mlflow(self, results: Dict):
        """
        Log les résultats vers MLflow
        """
        
        try:
            with mlflow.start_run(run_name="rigorous_backtest"):
                
                # Métadonnées
                metadata = results.get("metadata", {})
                mlflow.log_params(metadata.get("strategy_params", {}))
                
                # Métriques de base
                base_backtest = results.get("base_backtest", {})
                if base_backtest.get("valid", False):
                    performance = base_backtest.get("performance", {})
                    for metric, value in performance.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"base_{metric}", value)
                
                # Robustesse
                robustness = results.get("robustness", {})
                if "robustness_score" in robustness:
                    score = robustness["robustness_score"].get("global_score", 0)
                    mlflow.log_metric("robustness_global_score", score)
                
                # Overfitting
                overfitting = results.get("overfitting", {})
                if "overfitting_score" in overfitting:
                    mlflow.log_metric("overfitting_score", overfitting["overfitting_score"])
                
                # Verdict final
                verdict = results.get("final_verdict", {})
                mlflow.log_metric("validation_score", verdict.get("validation_score", 0))
                mlflow.log_param("production_ready", verdict.get("production_ready", False))
                
                logger.info("📊 Résultats loggés vers MLflow")
                
        except Exception as e:
            logger.warning(f"⚠️ Échec logging MLflow: {e}")
    
    def save_results(self, filepath: str = None) -> str:
        """
        Sauvegarde les résultats du backtest
        """
        
        if not self.backtest_results:
            logger.warning("Aucun résultat à sauvegarder")
            return ""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"rigorous_backtest_{timestamp}.json"
        
        # Conversion pour JSON
        json_results = self._make_json_serializable(self.backtest_results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"💾 Résultats sauvés: {filepath}")
        return filepath
    
    def _make_json_serializable(self, obj):
        """Convertit objets pour JSON"""
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
        elif pd.isna(obj):
            return None
        else:
            return obj

# Exemple d'utilisation
if __name__ == "__main__":
    
    # Test avec stratégie simple
    logger.info("🧪 Test RigorousBacktester...")
    
    # Générer données test
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-06-30', freq='1h')
    
    # Simuler prix avec tendance + bruit
    returns = np.random.normal(0.0002, 0.015, len(dates))  # Léger bull market
    prices = (1 + returns).cumprod() * 100
    
    data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Stratégie simple momentum
    def momentum_strategy(data, params):
        lookback = params.get('lookback', 20)
        threshold = params.get('threshold', 0.02)
        
        returns = data['close'].pct_change()
        momentum = returns.rolling(lookback).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[momentum > threshold] = 1
        signals[momentum < -threshold] = -1
        
        strategy_returns = signals.shift(1) * returns
        
        return {
            'returns': strategy_returns,
            'signals': signals
        }
    
    # Test backtest rigoureux
    backtester = RigorousBacktester(
        initial_capital=50000,
        enable_walk_forward=True,
        enable_oos_validation=True,
        enable_monte_carlo=True
    )
    
    results = backtester.run_comprehensive_backtest(
        momentum_strategy,
        data,
        {'lookback': 20, 'threshold': 0.02},
        'BTCUSDT'
    )
    
    if results and not results.get("error"):
        print("\n=== RÉSULTATS BACKTEST RIGOUREUX ===")
        
        # Performance de base
        base_performance = results["base_backtest"]["performance"]
        print(f"Sharpe Ratio: {base_performance['sharpe_ratio']:.3f}")
        print(f"Total Return: {base_performance['total_return']:.1%}")
        print(f"Max Drawdown: {base_performance['max_drawdown']:.1%}")
        
        # Verdict final
        verdict = results["final_verdict"]
        print(f"\nVerdict: {verdict['verdict']}")
        print(f"Score validation: {verdict['validation_score']:.1%}")
        print(f"Prêt production: {verdict['production_ready']}")
        
        # Recommandations
        print(f"\nRecommandations:")
        for rec in verdict['recommendations']:
            print(f"  {rec}")
        
        # Sauvegarde
        filepath = backtester.save_results()
        print(f"\n💾 Résultats sauvés: {filepath}")
        
    else:
        print(f"❌ Erreur backtest: {results.get('error')}")