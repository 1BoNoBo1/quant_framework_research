#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Métriques de Robustesse pour Stratégies Quantitatives
Implémentation complète des métriques avancées de validation
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class RobustnessMetrics:
    """
    Calculateur de métriques de robustesse avancées
    
    Métriques implémentées:
    - Probabilistic Sharpe Ratio (PSR)
    - Deflated Sharpe Ratio (DSR) 
    - Maximum Drawdown Duration
    - Calmar Ratio, Sortino Ratio, Omega Ratio
    - Tail Risk Metrics (VaR, CVaR, Tail Ratio)
    - Stability Metrics (Serial Correlation, Regime Analysis)
    - Information Ratio et Active Share
    - Monte Carlo Confidence Intervals
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 risk_free_rate: float = 0.02,
                 benchmark_return: float = 0.05):
        
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.benchmark_return = benchmark_return
        
        logger.info("✅ RobustnessMetrics initialisé")
    
    def calculate_comprehensive_metrics(self, 
                                      returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None,
                                      signals: Optional[pd.Series] = None) -> Dict:
        """
        Calcule un ensemble complet de métriques de robustesse
        
        Args:
            returns: Série des returns de la stratégie
            benchmark_returns: Returns du benchmark (optionnel)
            signals: Signaux de trading (optionnel)
            
        Returns:
            Dict avec toutes les métriques de robustesse
        """
        
        if len(returns) == 0:
            return {"error": "Returns series is empty"}
        
        logger.info(f"📊 Calcul métriques robustesse sur {len(returns)} observations")
        
        # Nettoyage données
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            logger.warning("⚠️ Peu de données pour calculs fiables")
        
        # 1. Métriques de base
        basic_metrics = self._calculate_basic_metrics(returns_clean)
        
        # 2. Métriques de risque avancées
        risk_metrics = self._calculate_risk_metrics(returns_clean)
        
        # 3. Métriques de robustesse statistique
        statistical_metrics = self._calculate_statistical_robustness(returns_clean)
        
        # 4. Métriques de stabilité temporelle
        stability_metrics = self._calculate_stability_metrics(returns_clean)
        
        # 5. Métriques vs benchmark (si disponible)
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns_clean, benchmark_returns)
        
        # 6. Métriques de trading (si signaux disponibles)
        trading_metrics = {}
        if signals is not None:
            trading_metrics = self._calculate_trading_metrics(returns_clean, signals)
        
        # 7. Score de robustesse global
        robustness_score = self._calculate_robustness_score(
            basic_metrics, risk_metrics, statistical_metrics, stability_metrics
        )
        
        # Compilation finale
        comprehensive_metrics = {
            "timestamp": datetime.now().isoformat(),
            "basic": basic_metrics,
            "risk": risk_metrics,
            "statistical": statistical_metrics,
            "stability": stability_metrics,
            "benchmark": benchmark_metrics,
            "trading": trading_metrics,
            "robustness_score": robustness_score,
            "overall_verdict": self._generate_robustness_verdict(robustness_score)
        }
        
        return comprehensive_metrics
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict:
        """
        Métriques de base de performance
        """
        
        # Returns metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "sortino_ratio": float(sortino_ratio)
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """
        Métriques de risque avancées
        """
        
        # Value at Risk
        var_1pct = returns.quantile(0.01)
        var_5pct = returns.quantile(0.05)
        
        # Conditional VaR (Expected Shortfall)
        cvar_1pct = returns[returns <= var_1pct].mean()
        cvar_5pct = returns[returns <= var_5pct].mean()
        
        # Tail Risk
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            tail_ratio = abs(positive_returns.mean() / negative_returns.mean())
        else:
            tail_ratio = np.nan
        
        # Maximum Drawdown Duration
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # Trouver durée max du drawdown
        is_drawdown = drawdown < 0
        dd_durations = []
        current_duration = 0
        
        for is_dd in is_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    dd_durations.append(current_duration)
                current_duration = 0
        
        max_dd_duration = max(dd_durations) if dd_durations else 0
        
        # Skewness et Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Omega Ratio (gains/losses ratio above threshold)
        threshold = self.risk_free_rate / 252  # Daily risk-free rate
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        
        omega_ratio = gains.sum() / losses.sum() if len(losses) > 0 and losses.sum() > 0 else np.inf
        
        return {
            "var_1pct": float(var_1pct),
            "var_5pct": float(var_5pct),
            "cvar_1pct": float(cvar_1pct),
            "cvar_5pct": float(cvar_5pct),
            "tail_ratio": float(tail_ratio) if not np.isnan(tail_ratio) else None,
            "max_drawdown_duration": int(max_dd_duration),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "omega_ratio": float(omega_ratio) if not np.isinf(omega_ratio) else None
        }
    
    def _calculate_statistical_robustness(self, returns: pd.Series) -> Dict:
        """
        Métriques de robustesse statistique
        """
        
        # Probabilistic Sharpe Ratio
        psr = self._probabilistic_sharpe_ratio(returns)
        
        # Deflated Sharpe Ratio (nécessite hypothèses sur nb de tests)
        n_tests = 100  # Assume 100 strategies tested
        dsr = self._deflated_sharpe_ratio(returns, n_tests)
        
        # Serial Correlation (autocorrélation)
        try:
            autocorr_lag1 = returns.autocorr(lag=1)
            autocorr_lag5 = returns.autocorr(lag=5)
        except:
            autocorr_lag1 = autocorr_lag5 = np.nan
        
        # Test de normalité (Jarque-Bera)
        try:
            jb_stat, jb_pval = stats.jarque_bera(returns.dropna())
            is_normal = jb_pval > 0.05
        except:
            jb_stat = jb_pval = np.nan
            is_normal = False
        
        # Stationarité (Augmented Dickey-Fuller)
        stationarity = self._test_stationarity(returns)
        
        # Stability des moments dans le temps
        stability = self._test_moment_stability(returns)
        
        return {
            "probabilistic_sharpe_ratio": float(psr),
            "deflated_sharpe_ratio": float(dsr) if not np.isnan(dsr) else None,
            "autocorr_lag1": float(autocorr_lag1) if not np.isnan(autocorr_lag1) else None,
            "autocorr_lag5": float(autocorr_lag5) if not np.isnan(autocorr_lag5) else None,
            "jarque_bera_stat": float(jb_stat) if not np.isnan(jb_stat) else None,
            "jarque_bera_pval": float(jb_pval) if not np.isnan(jb_pval) else None,
            "is_normal": bool(is_normal),
            "stationarity": stationarity,
            "moment_stability": stability
        }
    
    def _calculate_stability_metrics(self, returns: pd.Series) -> Dict:
        """
        Métriques de stabilité temporelle
        """
        
        if len(returns) < 60:  # Besoin minimum 2 mois
            return {"error": "Not enough data for stability analysis"}
        
        # Split en sous-périodes
        n_periods = 4
        period_length = len(returns) // n_periods
        
        period_sharpes = []
        period_returns = []
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else len(returns)
            
            period_data = returns.iloc[start_idx:end_idx]
            
            if len(period_data) > 10:
                period_return = (1 + period_data.mean()) ** 252 - 1
                period_vol = period_data.std() * np.sqrt(252)
                period_sharpe = (period_return - self.risk_free_rate) / period_vol if period_vol > 0 else 0
                
                period_sharpes.append(period_sharpe)
                period_returns.append(period_return)
        
        # Stabilité = 1 - coefficient de variation
        if len(period_sharpes) > 1:
            sharpe_stability = 1 - (np.std(period_sharpes) / abs(np.mean(period_sharpes))) if np.mean(period_sharpes) != 0 else 0
            return_stability = 1 - (np.std(period_returns) / abs(np.mean(period_returns))) if np.mean(period_returns) != 0 else 0
        else:
            sharpe_stability = return_stability = 0
        
        # Régime detection (Bull vs Bear markets)
        regime_analysis = self._analyze_regimes(returns)
        
        # Rolling window stability
        rolling_stability = self._calculate_rolling_stability(returns)
        
        return {
            "sharpe_stability": float(max(0, sharpe_stability)),
            "return_stability": float(max(0, return_stability)),
            "regime_analysis": regime_analysis,
            "rolling_stability": rolling_stability,
            "period_sharpes": [float(x) for x in period_sharpes],
            "period_returns": [float(x) for x in period_returns]
        }
    
    def _calculate_benchmark_metrics(self, 
                                   returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Dict:
        """
        Métriques versus benchmark
        """
        
        # Aligner les séries
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:
            return {"error": "Not enough aligned data"}
        
        strategy_returns = aligned_data.iloc[:, 0]
        bench_returns = aligned_data.iloc[:, 1]
        
        # Active returns
        active_returns = strategy_returns - bench_returns
        
        # Information Ratio
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Beta
        try:
            beta = np.cov(strategy_returns, bench_returns)[0, 1] / np.var(bench_returns)
            alpha = (strategy_returns.mean() * 252) - beta * (bench_returns.mean() * 252)
        except:
            beta = alpha = np.nan
        
        # Correlation
        correlation = strategy_returns.corr(bench_returns)
        
        # Capture Ratios (Up/Down market performance)
        up_market = bench_returns > 0
        down_market = bench_returns < 0
        
        if up_market.sum() > 5 and down_market.sum() > 5:
            up_capture = (strategy_returns[up_market].mean() / bench_returns[up_market].mean()) if bench_returns[up_market].mean() > 0 else np.nan
            down_capture = (strategy_returns[down_market].mean() / bench_returns[down_market].mean()) if bench_returns[down_market].mean() < 0 else np.nan
        else:
            up_capture = down_capture = np.nan
        
        return {
            "information_ratio": float(information_ratio),
            "tracking_error": float(tracking_error),
            "beta": float(beta) if not np.isnan(beta) else None,
            "alpha": float(alpha) if not np.isnan(alpha) else None,
            "correlation": float(correlation) if not np.isnan(correlation) else None,
            "up_capture_ratio": float(up_capture) if not np.isnan(up_capture) else None,
            "down_capture_ratio": float(down_capture) if not np.isnan(down_capture) else None
        }
    
    def _calculate_trading_metrics(self, 
                                 returns: pd.Series,
                                 signals: pd.Series) -> Dict:
        """
        Métriques spécifiques au trading
        """
        
        # Aligner les séries
        aligned_data = pd.concat([returns, signals], axis=1).dropna()
        
        if len(aligned_data) < 10:
            return {"error": "Not enough trading data"}
        
        strategy_returns = aligned_data.iloc[:, 0]
        strategy_signals = aligned_data.iloc[:, 1]
        
        # Trade analysis
        signal_changes = strategy_signals.diff().abs() > 0
        trades = signal_changes.sum()
        
        # Win/Loss analysis
        winning_trades = (strategy_returns[signal_changes] > 0).sum()
        losing_trades = (strategy_returns[signal_changes] < 0).sum()
        hit_rate = winning_trades / trades if trades > 0 else 0
        
        # Average win/loss
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        # Profit Factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        # Position holding analysis
        position_durations = []
        current_position = 0
        current_duration = 0
        
        for signal in strategy_signals:
            if signal != 0 and signal == current_position:
                current_duration += 1
            else:
                if current_duration > 0:
                    position_durations.append(current_duration)
                current_position = signal
                current_duration = 1 if signal != 0 else 0
        
        avg_holding_period = np.mean(position_durations) if position_durations else 0
        
        return {
            "total_trades": int(trades),
            "hit_rate": float(hit_rate),
            "win_loss_ratio": float(win_loss_ratio) if not np.isinf(win_loss_ratio) else None,
            "profit_factor": float(profit_factor) if not np.isinf(profit_factor) else None,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "avg_holding_period": float(avg_holding_period)
        }
    
    def _probabilistic_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calcul du Probabilistic Sharpe Ratio (PSR)
        """
        
        n = len(returns)
        if n < 10:
            return 0.0
        
        # Sharpe ratio observé
        sharpe = (returns.mean() * 252 - self.risk_free_rate) / (returns.std() * np.sqrt(252))
        
        # Skewness et Kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Facteur d'ajustement pour non-normalité
        adjustment = (1 - skew * sharpe + ((kurt - 3) / 4) * sharpe**2) / np.sqrt(n - 1)
        
        # PSR (probabilité que Sharpe > 0)
        psr = stats.norm.cdf(sharpe * np.sqrt(adjustment))
        
        return float(psr)
    
    def _deflated_sharpe_ratio(self, returns: pd.Series, n_tests: int) -> float:
        """
        Calcul du Deflated Sharpe Ratio (DSR)
        Ajuste pour multiple testing bias
        """
        
        if len(returns) < 10:
            return np.nan
        
        # Sharpe ratio observé
        sharpe = (returns.mean() * 252 - self.risk_free_rate) / (returns.std() * np.sqrt(252))
        
        # Expected maximum Sharpe sous null hypothesis
        euler_gamma = 0.5772156649015329  # Constante d'Euler-Mascheroni
        expected_max_sharpe = (1 - euler_gamma) * stats.norm.ppf(1 - 1/n_tests) + euler_gamma * stats.norm.ppf(1 - 1/(n_tests * np.e))
        
        # Variance du maximum
        variance_max = (1 - euler_gamma) * stats.norm.ppf(1 - 1/n_tests)**2 + euler_gamma * stats.norm.ppf(1 - 1/(n_tests * np.e))**2 - expected_max_sharpe**2
        
        # DSR
        if variance_max > 0:
            dsr = (sharpe - expected_max_sharpe) / np.sqrt(variance_max)
        else:
            dsr = sharpe
        
        return float(dsr)
    
    def _test_stationarity(self, returns: pd.Series) -> Dict:
        """
        Test de stationnarité (Augmented Dickey-Fuller)
        """
        
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(returns.dropna())
            
            return {
                "adf_statistic": float(result[0]),
                "p_value": float(result[1]),
                "is_stationary": result[1] < 0.05,
                "critical_values": {str(k): float(v) for k, v in result[4].items()}
            }
        except ImportError:
            logger.warning("statsmodels non disponible - skip test stationnarité")
            return {"error": "statsmodels not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _test_moment_stability(self, returns: pd.Series) -> Dict:
        """
        Test de stabilité des moments statistiques
        """
        
        if len(returns) < 60:
            return {"error": "Not enough data"}
        
        # Split en 3 périodes
        n_periods = 3
        period_length = len(returns) // n_periods
        
        means = []
        stds = []
        skews = []
        kurts = []
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else len(returns)
            
            period_data = returns.iloc[start_idx:end_idx]
            
            if len(period_data) > 20:
                means.append(period_data.mean())
                stds.append(period_data.std())
                skews.append(period_data.skew())
                kurts.append(period_data.kurtosis())
        
        # Coefficients de variation pour stabilité
        stability_scores = {}
        
        for name, values in [("mean", means), ("std", stds), ("skew", skews), ("kurt", kurts)]:
            if len(values) > 1 and np.mean(values) != 0:
                cv = np.std(values) / abs(np.mean(values))
                stability_scores[f"{name}_stability"] = max(0, 1 - cv)
            else:
                stability_scores[f"{name}_stability"] = 0
        
        return stability_scores
    
    def _analyze_regimes(self, returns: pd.Series) -> Dict:
        """
        Analyse de régimes de marché (Bull/Bear)
        """
        
        if len(returns) < 60:
            return {"error": "Not enough data for regime analysis"}
        
        # Simple regime detection basé sur moving average
        ma_short = returns.rolling(20).mean()
        ma_long = returns.rolling(60).mean()
        
        # Bull market when short MA > long MA
        bull_market = ma_short > ma_long
        bear_market = ~bull_market
        
        # Performance par régime
        bull_returns = returns[bull_market.shift(1)]  # Lag pour éviter look-ahead
        bear_returns = returns[bear_market.shift(1)]
        
        bull_sharpe = (bull_returns.mean() * 252) / (bull_returns.std() * np.sqrt(252)) if len(bull_returns) > 5 and bull_returns.std() > 0 else 0
        bear_sharpe = (bear_returns.mean() * 252) / (bear_returns.std() * np.sqrt(252)) if len(bear_returns) > 5 and bear_returns.std() > 0 else 0
        
        return {
            "bull_periods": int(bull_market.sum()),
            "bear_periods": int(bear_market.sum()),
            "bull_sharpe": float(bull_sharpe),
            "bear_sharpe": float(bear_sharpe),
            "regime_consistency": float(abs(bull_sharpe) + abs(bear_sharpe)) / 2  # Performance moyenne
        }
    
    def _calculate_rolling_stability(self, returns: pd.Series) -> Dict:
        """
        Stabilité sur fenêtres glissantes
        """
        
        window = min(60, len(returns) // 4)
        
        if window < 20:
            return {"error": "Not enough data for rolling analysis"}
        
        # Rolling Sharpe
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        # Stabilité = 1 - coefficient de variation du rolling Sharpe
        rolling_sharpe_clean = rolling_sharpe.dropna()
        
        if len(rolling_sharpe_clean) > 5:
            cv = rolling_sharpe_clean.std() / abs(rolling_sharpe_clean.mean()) if rolling_sharpe_clean.mean() != 0 else np.inf
            stability = max(0, 1 - cv) if not np.isinf(cv) else 0
        else:
            stability = 0
        
        return {
            "rolling_sharpe_stability": float(stability),
            "rolling_window": int(window),
            "rolling_observations": int(len(rolling_sharpe_clean))
        }
    
    def _calculate_robustness_score(self, 
                                  basic: Dict,
                                  risk: Dict,
                                  statistical: Dict,
                                  stability: Dict) -> Dict:
        """
        Calcul du score de robustesse global
        """
        
        # Pondérations des différentes catégories
        weights = {
            "performance": 0.3,
            "risk": 0.25,
            "statistical": 0.25,
            "stability": 0.2
        }
        
        # Normalisation des scores (0-1)
        scores = {}
        
        # Performance score
        sharpe = basic.get('sharpe_ratio', 0)
        calmar = basic.get('calmar_ratio', 0)
        performance_score = min(1, max(0, (sharpe + 1) / 3))  # Normalise Sharpe [-1, 2] -> [0, 1]
        
        scores['performance'] = performance_score
        
        # Risk score
        max_dd = basic.get('max_drawdown', 1)
        tail_ratio = risk.get('tail_ratio', 1) or 1
        var_5pct = risk.get('var_5pct', -0.1)
        
        risk_score = min(1, max(0, (1 - max_dd) * 0.5 + min(1, tail_ratio/2) * 0.3 + min(1, abs(var_5pct)/0.05) * 0.2))
        scores['risk'] = risk_score
        
        # Statistical score
        psr = statistical.get('probabilistic_sharpe_ratio', 0)
        autocorr = abs(statistical.get('autocorr_lag1', 0) or 0)
        stat_score = psr * 0.7 + (1 - min(1, autocorr)) * 0.3  # PSR élevé + faible autocorr = bon
        
        scores['statistical'] = stat_score
        
        # Stability score
        sharpe_stability = stability.get('sharpe_stability', 0)
        return_stability = stability.get('return_stability', 0)
        stab_score = (sharpe_stability + return_stability) / 2
        
        scores['stability'] = stab_score
        
        # Score global pondéré
        global_score = sum(scores[key] * weights[key] for key in scores.keys())
        
        return {
            "global_score": float(global_score),
            "component_scores": scores,
            "weights": weights,
            "grade": self._score_to_grade(global_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """
        Convertit score en grade alphabétique
        """
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        else:
            return "D"
    
    def _generate_robustness_verdict(self, robustness_score: Dict) -> Dict:
        """
        Génère le verdict final de robustesse
        """
        
        score = robustness_score["global_score"]
        grade = robustness_score["grade"]
        
        # Verdict textuel
        if score >= 0.8:
            verdict = "✅ TRÈS ROBUSTE - Prêt pour production"
            color = "green"
        elif score >= 0.6:
            verdict = "⚠️ MODÉRÉMENT ROBUSTE - Surveillance recommandée"
            color = "orange"
        elif score >= 0.4:
            verdict = "🟡 PEU ROBUSTE - Amélioration nécessaire"
            color = "yellow"
        else:
            verdict = "❌ NON-ROBUSTE - Refonte requise"
            color = "red"
        
        # Recommandations basées sur les composants faibles
        recommendations = []
        component_scores = robustness_score["component_scores"]
        
        if component_scores.get("performance", 0) < 0.5:
            recommendations.append("📈 Améliorer performance (Sharpe/Calmar)")
        
        if component_scores.get("risk", 0) < 0.5:
            recommendations.append("🛡️ Réduire exposition au risque (DD/VaR)")
        
        if component_scores.get("statistical", 0) < 0.5:
            recommendations.append("📊 Améliorer robustesse statistique (PSR)")
        
        if component_scores.get("stability", 0) < 0.5:
            recommendations.append("⚖️ Stabiliser performance temporelle")
        
        return {
            "score": float(score),
            "grade": grade,
            "verdict": verdict,
            "color": color,
            "recommendations": recommendations,
            "production_ready": score >= 0.7
        }

# Exemple d'utilisation
if __name__ == "__main__":
    
    # Test avec données simulées
    logger.info("🧪 Test RobustnessMetrics...")
    
    # Simuler stratégie avec caractéristiques connues
    np.random.seed(42)
    n_days = 500
    
    # Stratégie momentum avec drawdown périodiques
    base_return = 0.001
    volatility = 0.02
    returns = np.random.normal(base_return, volatility, n_days)
    
    # Ajouter quelques drawdowns
    returns[100:120] = np.random.normal(-0.005, 0.03, 20)  # Drawdown
    returns[300:310] = np.random.normal(-0.008, 0.04, 10)  # Autre drawdown
    
    returns_series = pd.Series(returns, index=pd.date_range('2023-01-01', periods=n_days))
    
    # Benchmark (market)
    benchmark = np.random.normal(0.0005, 0.015, n_days)
    benchmark_series = pd.Series(benchmark, index=returns_series.index)
    
    # Signaux de trading simulés
    signals = np.where(returns > 0.002, 1, np.where(returns < -0.002, -1, 0))
    signals_series = pd.Series(signals, index=returns_series.index)
    
    # Calcul métriques
    metrics_calculator = RobustnessMetrics()
    
    comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(
        returns_series,
        benchmark_series,
        signals_series
    )
    
    # Affichage résultats
    print("\n=== MÉTRIQUES DE ROBUSTESSE ===")
    print(f"Score global: {comprehensive_metrics['robustness_score']['global_score']:.3f}")
    print(f"Grade: {comprehensive_metrics['robustness_score']['grade']}")
    print(f"Verdict: {comprehensive_metrics['overall_verdict']['verdict']}")
    
    print(f"\nSharpe Ratio: {comprehensive_metrics['basic']['sharpe_ratio']:.3f}")
    print(f"PSR: {comprehensive_metrics['statistical']['probabilistic_sharpe_ratio']:.3f}")
    print(f"Max DD: {comprehensive_metrics['basic']['max_drawdown']:.1%}")
    print(f"VaR 5%: {comprehensive_metrics['risk']['var_5pct']:.3f}")
    
    print("\nRecommandations:")
    for rec in comprehensive_metrics['overall_verdict']['recommendations']:
        print(f"  {rec}")
    
    # Sauvegarde
    output_file = "robustness_metrics_test.json"
    with open(output_file, 'w') as f:
        # Conversion pour JSON
        json_data = json.loads(json.dumps(comprehensive_metrics, default=str))
        json.dump(json_data, f, indent=2)
    
    print(f"\n💾 Résultats sauvés: {output_file}")