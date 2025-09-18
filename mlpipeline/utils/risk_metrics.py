#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Métriques de risque quantitatives - Version Production
Migration et amélioration du utils_risque.py original
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional
import warnings

def ratio_sharpe(frequence_annuelle: int, returns: Union[np.ndarray, pd.Series], 
                risk_free_rate: float = 0.0) -> float:
    """
    Calcule le ratio de Sharpe annualisé
    
    Args:
        frequence_annuelle: Nombre de périodes par an (365 pour daily, 8760 pour hourly)
        returns: Série des returns
        risk_free_rate: Taux sans risque (défaut: 0)
        
    Returns:
        Ratio de Sharpe annualisé
    """
    returns = np.asarray(returns)
    
    if len(returns) == 0:
        return 0.0
    
    # Returns excédentaires
    excess_returns = returns - risk_free_rate / frequence_annuelle
    
    # Moyennes et volatilité
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)
    
    if std_return == 0 or np.isnan(std_return):
        return 0.0
    
    # Annualisation
    sharpe = (mean_return / std_return) * np.sqrt(frequence_annuelle)
    
    return float(sharpe) if not np.isnan(sharpe) else 0.0

def drawdown_max(equity_curve: Union[np.ndarray, pd.Series]) -> float:
    """
    Calcule le drawdown maximum
    
    Args:
        equity_curve: Courbe d'équité cumulative
        
    Returns:
        Maximum drawdown (positif)
    """
    equity = np.asarray(equity_curve)
    
    if len(equity) <= 1:
        return 0.0
    
    # Peak running (sommet courant)
    peaks = np.maximum.accumulate(equity)
    
    # Drawdowns
    drawdowns = (peaks - equity) / peaks
    
    max_dd = np.max(drawdowns)
    
    return float(max_dd) if not np.isnan(max_dd) else 0.0

def probabilistic_sharpe_ratio(frequence_annuelle: int, 
                              returns: Union[np.ndarray, pd.Series],
                              benchmark_sharpe: float = 0.0,
                              confidence_level: float = 0.95) -> float:
    """
    Calcule le Probabilistic Sharpe Ratio (PSR) selon López de Prado
    
    Args:
        frequence_annuelle: Fréquence annuelle des returns
        returns: Série des returns
        benchmark_sharpe: Sharpe de référence à battre
        confidence_level: Niveau de confiance
        
    Returns:
        PSR (probabilité que le Sharpe soit > benchmark)
    """
    returns = np.asarray(returns)
    n_obs = len(returns)
    
    if n_obs <= 2:
        return 0.0
    
    # Sharpe ratio observé
    sharpe_obs = ratio_sharpe(frequence_annuelle, returns)
    
    if sharpe_obs <= benchmark_sharpe:
        return 0.0
    
    # Skewness et kurtosis des returns
    skewness = stats.skew(returns)
    excess_kurtosis = stats.kurtosis(returns, fisher=True)  # fisher=True pour excess kurtosis
    
    # Variance du Sharpe ratio estimé (formule López de Prado)
    sharpe_variance = (1 / n_obs) * (1 + 0.5 * sharpe_obs**2 - 
                                    skewness * sharpe_obs + 
                                    (excess_kurtosis / 4) * sharpe_obs**2)
    
    if sharpe_variance <= 0:
        return 0.0
    
    # Test statistique
    z_score = (sharpe_obs - benchmark_sharpe) / np.sqrt(sharpe_variance)
    
    # PSR : probabilité que Sharpe > benchmark
    psr = stats.norm.cdf(z_score)
    
    return float(psr) if not np.isnan(psr) else 0.0

def calmar_ratio(returns: Union[np.ndarray, pd.Series], 
                frequence_annuelle: int) -> float:
    """
    Ratio de Calmar : Return annualisé / Max Drawdown
    """
    returns_array = np.asarray(returns)
    
    if len(returns_array) == 0:
        return 0.0
    
    # Return annualisé
    cumul_return = np.prod(1 + returns_array) - 1
    n_periods = len(returns_array)
    annualized_return = (1 + cumul_return) ** (frequence_annuelle / n_periods) - 1
    
    # Equity curve pour drawdown
    equity = np.cumprod(1 + returns_array)
    max_dd = drawdown_max(equity)
    
    if max_dd == 0:
        return np.inf if annualized_return > 0 else 0.0
    
    calmar = annualized_return / max_dd
    return float(calmar) if not np.isnan(calmar) else 0.0

def sortino_ratio(returns: Union[np.ndarray, pd.Series],
                 frequence_annuelle: int,
                 target_return: float = 0.0) -> float:
    """
    Ratio de Sortino : utilise seulement la volatilité des returns négatifs
    """
    returns_array = np.asarray(returns)
    
    if len(returns_array) == 0:
        return 0.0
    
    # Returns excédentaires
    excess_returns = returns_array - target_return / frequence_annuelle
    mean_excess = np.mean(excess_returns)
    
    # Volatilité downside (seulement returns < target)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    downside_vol = np.std(downside_returns, ddof=1)
    
    if downside_vol == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    # Annualisation
    sortino = (mean_excess / downside_vol) * np.sqrt(frequence_annuelle)
    
    return float(sortino) if not np.isnan(sortino) else 0.0

def omega_ratio(returns: Union[np.ndarray, pd.Series],
               threshold: float = 0.0) -> float:
    """
    Ratio Omega : Probabilité de gains / Probabilité de pertes
    """
    returns_array = np.asarray(returns)
    
    if len(returns_array) == 0:
        return 1.0
    
    # Returns au-dessus et en-dessous du seuil
    above_threshold = returns_array - threshold
    gains = above_threshold[above_threshold > 0]
    losses = above_threshold[above_threshold < 0]
    
    gains_sum = np.sum(gains) if len(gains) > 0 else 0
    losses_sum = abs(np.sum(losses)) if len(losses) > 0 else 1e-10  # Éviter division par 0
    
    omega = gains_sum / losses_sum
    return float(omega) if not np.isnan(omega) else 0.0

def var_cvar(returns: Union[np.ndarray, pd.Series], 
            confidence_level: float = 0.05) -> tuple:
    """
    Calcule VaR et CVaR (Expected Shortfall)
    
    Returns:
        (VaR, CVaR) - valeurs positives représentant les pertes
    """
    returns_array = np.asarray(returns)
    
    if len(returns_array) == 0:
        return 0.0, 0.0
    
    # VaR (quantile des pertes)
    var = -np.percentile(returns_array, confidence_level * 100)
    
    # CVaR (moyenne des pertes au-delà du VaR)
    threshold_return = -var
    tail_returns = returns_array[returns_array <= threshold_return]
    
    if len(tail_returns) == 0:
        cvar = var
    else:
        cvar = -np.mean(tail_returns)
    
    return float(var), float(cvar)

def information_ratio(portfolio_returns: Union[np.ndarray, pd.Series],
                     benchmark_returns: Union[np.ndarray, pd.Series],
                     frequence_annuelle: int) -> float:
    """
    Ratio d'information : Alpha / Tracking Error
    """
    portfolio = np.asarray(portfolio_returns)
    benchmark = np.asarray(benchmark_returns)
    
    if len(portfolio) != len(benchmark) or len(portfolio) == 0:
        return 0.0
    
    # Excess returns (alpha)
    excess_returns = portfolio - benchmark
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    # Annualisation
    ir = (mean_excess / std_excess) * np.sqrt(frequence_annuelle)
    
    return float(ir) if not np.isnan(ir) else 0.0

def tail_ratio(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Tail Ratio : 95ème percentile / 5ème percentile
    Mesure l'asymétrie des queues de distribution
    """
    returns_array = np.asarray(returns)
    
    if len(returns_array) < 20:  # Minimum pour percentiles fiables
        return 1.0
    
    p95 = np.percentile(returns_array, 95)
    p5 = np.percentile(returns_array, 5)
    
    if abs(p5) < 1e-10:  # Éviter division par 0
        return np.inf if p95 > 0 else 1.0
    
    tail_r = abs(p95 / p5)
    return float(tail_r) if not np.isnan(tail_r) else 1.0

def stability_ratio(returns: Union[np.ndarray, pd.Series],
                   window_size: int = 252) -> float:
    """
    Mesure la stabilité des returns sur des fenêtres glissantes
    """
    returns_array = np.asarray(returns)
    
    if len(returns_array) < window_size * 2:
        return 0.0
    
    # Sharpe ratios sur fenêtres glissantes
    rolling_sharpes = []
    
    for i in range(window_size, len(returns_array)):
        window_returns = returns_array[i-window_size:i]
        window_sharpe = ratio_sharpe(252, window_returns)  # Assumé daily
        rolling_sharpes.append(window_sharpe)
    
    if len(rolling_sharpes) == 0:
        return 0.0
    
    # Stabilité = 1 - coefficient de variation des Sharpe ratios
    rolling_sharpes = np.array(rolling_sharpes)
    mean_sharpe = np.mean(rolling_sharpes)
    std_sharpe = np.std(rolling_sharpes)
    
    if abs(mean_sharpe) < 1e-10:
        return 0.0
    
    cv = std_sharpe / abs(mean_sharpe)
    stability = max(0.0, 1.0 - cv)
    
    return float(stability) if not np.isnan(stability) else 0.0

def comprehensive_metrics(returns: Union[np.ndarray, pd.Series],
                         frequence_annuelle: int = 365,
                         benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None) -> dict:
    """
    Calcule un ensemble complet de métriques de risque-rendement
    
    Args:
        returns: Série des returns de la stratégie
        frequence_annuelle: Fréquence des données (365=daily, 8760=hourly, etc.)
        benchmark_returns: Returns du benchmark (optionnel)
        
    Returns:
        Dictionnaire complet des métriques
    """
    returns_array = np.asarray(returns)
    
    if len(returns_array) == 0:
        return {}
    
    # Equity curve
    equity = np.cumprod(1 + returns_array)
    
    # Métriques de base
    metrics = {
        "total_return": float(equity[-1] - 1),
        "annualized_return": float((equity[-1] ** (frequence_annuelle / len(returns_array))) - 1),
        "volatility": float(np.std(returns_array) * np.sqrt(frequence_annuelle)),
        "sharpe_ratio": ratio_sharpe(frequence_annuelle, returns_array),
        "max_drawdown": drawdown_max(equity),
        "calmar_ratio": calmar_ratio(returns_array, frequence_annuelle),
        "sortino_ratio": sortino_ratio(returns_array, frequence_annuelle),
        "psr": probabilistic_sharpe_ratio(frequence_annuelle, returns_array),
    }
    
    # Métriques additionnelles
    var_5, cvar_5 = var_cvar(returns_array, 0.05)
    metrics.update({
        "omega_ratio": omega_ratio(returns_array),
        "tail_ratio": tail_ratio(returns_array),
        "var_5pct": var_5,
        "cvar_5pct": cvar_5,
        "skewness": float(stats.skew(returns_array)),
        "kurtosis": float(stats.kurtosis(returns_array)),
        "hit_rate": float((returns_array > 0).mean()),
        "stability": stability_ratio(returns_array)
    })
    
    # Métriques vs benchmark
    if benchmark_returns is not None:
        benchmark_array = np.asarray(benchmark_returns)
        if len(benchmark_array) == len(returns_array):
            metrics["information_ratio"] = information_ratio(
                returns_array, benchmark_array, frequence_annuelle
            )
            
            # Beta et Alpha (CAPM)
            if np.std(benchmark_array) > 0:
                beta = np.cov(returns_array, benchmark_array)[0,1] / np.var(benchmark_array)
                alpha = np.mean(returns_array) - beta * np.mean(benchmark_array)
                metrics["beta"] = float(beta)
                metrics["alpha_capm"] = float(alpha * frequence_annuelle)
    
    return metrics

# ==============================================
# FONCTIONS DE VALIDATION
# ==============================================

def validate_metrics(metrics: dict, min_sharpe: float = 0.5, 
                     max_drawdown: float = 0.3) -> dict:
    """
    Valide les métriques selon des seuils de qualité
    """
    validation = {
        "is_valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Vérifications critiques
    if metrics.get("sharpe_ratio", 0) < min_sharpe:
        validation["warnings"].append(f"Sharpe ratio faible: {metrics.get('sharpe_ratio', 0):.3f}")
    
    if metrics.get("max_drawdown", 1) > max_drawdown:
        validation["errors"].append(f"Drawdown excessif: {metrics.get('max_drawdown', 0):.3f}")
        validation["is_valid"] = False
    
    if metrics.get("hit_rate", 0) < 0.4:
        validation["warnings"].append(f"Hit rate faible: {metrics.get('hit_rate', 0):.3f}")
    
    if metrics.get("psr", 0) < 0.6:
        validation["warnings"].append(f"PSR faible: {metrics.get('psr', 0):.3f}")
    
    return validation