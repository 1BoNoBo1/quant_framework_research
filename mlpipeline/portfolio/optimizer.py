#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimiseur de Portefeuille Quantitatif Avancé
Allocation optimale entre alphas selon Markowitz + Kelly + Risk Parity
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import cvxpy as cp
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# MLflow pour tracking
import mlflow
import mlflow.sklearn

from mlpipeline.utils.risk_metrics import (
    ratio_sharpe, 
    drawdown_max,
    probabilistic_sharpe_ratio,
    comprehensive_metrics
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class QuantPortfolioOptimizer:
    """
    Optimiseur de portefeuille multi-objectif pour alphas quantitatifs
    
    Méthodes supportées:
    - Mean-Variance (Markowitz)
    - Kelly Criterion
    - Risk Parity 
    - Maximum Diversification
    - Black-Litterman
    - Regime-Aware Allocation
    """
    
    def __init__(self,
                 method: str = "kelly_markowitz",
                 risk_aversion: float = 2.0,
                 max_weight: float = 0.4,
                 min_weight: float = 0.0,
                 rebalancing_frequency: str = "weekly",
                 transaction_cost: float = 0.001,
                 use_regime_detection: bool = True):
        
        self.method = method
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.rebalancing_frequency = rebalancing_frequency
        self.transaction_cost = transaction_cost
        self.use_regime_detection = use_regime_detection
        
        # Caches et modèles
        self.covariance_estimator = LedoitWolf()
        self.regime_weights = {}
        self.last_optimization_time = None
        self.historical_allocations = []
        
        logger.info(f"📊 PortfolioOptimizer initialisé: méthode={method}, "
                   f"risk_aversion={risk_aversion}, max_weight={max_weight}")
    
    def optimize_portfolio(self, 
                          alpha_returns: pd.DataFrame,
                          regime_info: Optional[Dict] = None,
                          current_positions: Optional[Dict] = None) -> Dict:
        """
        Optimisation principale du portefeuille
        
        Args:
            alpha_returns: DataFrame avec returns de chaque alpha (colonnes = alphas)
            regime_info: Information sur régime de marché actuel
            current_positions: Positions actuelles pour calcul coûts transaction
            
        Returns:
            Dict avec poids optimaux et métriques
        """
        
        logger.info(f"🎯 Optimisation portfolio: {len(alpha_returns.columns)} alphas, "
                   f"{len(alpha_returns)} observations")
        
        try:
            # 1. Préparation données
            clean_returns = self._prepare_returns_data(alpha_returns)
            
            if clean_returns.empty or len(clean_returns.columns) == 0:
                logger.warning("⚠️ Données insuffisantes pour optimisation")
                return self._get_equal_weight_fallback(alpha_returns.columns)
            
            # 2. Estimation paramètres
            expected_returns = self._estimate_expected_returns(clean_returns, regime_info)
            covariance_matrix = self._estimate_covariance_matrix(clean_returns)
            
            # 3. Optimisation selon méthode choisie
            if self.method == "kelly_markowitz":
                weights = self._optimize_kelly_markowitz(expected_returns, covariance_matrix)
            elif self.method == "risk_parity":
                weights = self._optimize_risk_parity(covariance_matrix)
            elif self.method == "max_diversification":
                weights = self._optimize_max_diversification(covariance_matrix)
            elif self.method == "black_litterman":
                weights = self._optimize_black_litterman(expected_returns, covariance_matrix, regime_info)
            else:
                logger.warning(f"⚠️ Méthode inconnue {self.method}, fallback Markowitz")
                weights = self._optimize_kelly_markowitz(expected_returns, covariance_matrix)
            
            # 4. Post-traitement et validation
            weights = self._apply_constraints(weights, current_positions)
            
            # 5. Calcul métriques portfolio
            portfolio_metrics = self._calculate_portfolio_metrics(
                weights, expected_returns, covariance_matrix, clean_returns
            )
            
            # 6. Logging MLflow
            self._log_to_mlflow(weights, portfolio_metrics, clean_returns)
            
            result = {
                'weights': weights,
                'expected_returns': expected_returns,
                'covariance_matrix': covariance_matrix,
                'portfolio_metrics': portfolio_metrics,
                'optimization_time': datetime.now(),
                'method': self.method,
                'alpha_names': list(clean_returns.columns)
            }
            
            # Sauvegarde historique
            self.historical_allocations.append(result)
            
            logger.info(f"✅ Optimisation terminée: expected_return={portfolio_metrics.get('expected_return', 0):.4f}, "
                       f"volatility={portfolio_metrics.get('volatility', 0):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur optimisation portfolio: {e}")
            return self._get_equal_weight_fallback(alpha_returns.columns)
    
    def _prepare_returns_data(self, alpha_returns: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage et préparation des données de returns"""
        
        # Suppression colonnes avec trop de NaN
        clean_returns = alpha_returns.dropna(thresh=len(alpha_returns) * 0.7, axis=1)
        
        # Suppression lignes avec NaN
        clean_returns = clean_returns.dropna()
        
        # Vérification minimum de données
        if len(clean_returns) < 30:
            logger.warning(f"⚠️ Peu d'observations: {len(clean_returns)}")
        
        # Suppression outliers extrêmes (>5 sigma)
        for col in clean_returns.columns:
            mean_ret = clean_returns[col].mean()
            std_ret = clean_returns[col].std()
            clean_returns = clean_returns[
                np.abs(clean_returns[col] - mean_ret) <= 5 * std_ret
            ]
        
        logger.debug(f"📊 Données nettoyées: {clean_returns.shape}")
        return clean_returns
    
    def _estimate_expected_returns(self, returns: pd.DataFrame, regime_info: Optional[Dict] = None) -> pd.Series:
        """Estimation des returns attendus avec ajustement régime"""
        
        # Méthode de base: moyenne historique
        expected_returns = returns.mean()
        
        # Ajustement selon régime de marché
        if regime_info and self.use_regime_detection:
            current_regime = regime_info.get('current_regime', 0)
            regime_multipliers = regime_info.get('regime_multipliers', {})
            
            if current_regime in regime_multipliers:
                multiplier = regime_multipliers[current_regime]
                expected_returns *= multiplier
                logger.debug(f"📊 Ajustement régime {current_regime}: multiplier={multiplier}")
        
        # Décroissance temporelle (plus de poids aux observations récentes)
        decay_factor = 0.94  # Half-life ~30 jours
        weights = np.array([decay_factor ** i for i in range(len(returns))])[::-1]
        weights /= weights.sum()
        
        # Expected returns pondérés
        expected_returns_weighted = (returns * weights.reshape(-1, 1)).sum()
        
        return expected_returns_weighted
    
    def _estimate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimation robuste de la matrice de covariance"""
        
        # Shrinkage Ledoit-Wolf pour stabilité
        cov_shrunk, _ = self.covariance_estimator.fit(returns.values).covariance_, self.covariance_estimator.shrinkage_
        
        # Conversion en DataFrame
        cov_matrix = pd.DataFrame(
            cov_shrunk, 
            index=returns.columns, 
            columns=returns.columns
        )
        
        # Vérification positive definite
        eigenvals = np.linalg.eigvals(cov_matrix.values)
        if np.min(eigenvals) <= 0:
            logger.warning("⚠️ Matrice covariance non définie positive, régularisation")
            regularization = 1e-6
            cov_matrix += regularization * np.eye(len(cov_matrix))
        
        return cov_matrix
    
    def _optimize_kelly_markowitz(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Optimisation Kelly-Markowitz hybride
        Combine critère de Kelly avec contraintes de variance
        """
        
        n_assets = len(expected_returns)
        
        # Variables CVXPY
        w = cp.Variable(n_assets)
        
        # Return attendu du portfolio
        portfolio_return = expected_returns.values @ w
        
        # Variance du portfolio
        portfolio_variance = cp.quad_form(w, cov_matrix.values)
        
        # Objectif Kelly modifié: log(1 + return) - risk_aversion * variance
        # Approximation: return - 0.5 * variance - risk_aversion * variance
        objective = portfolio_return - 0.5 * portfolio_variance - self.risk_aversion * portfolio_variance
        
        # Contraintes
        constraints = [
            cp.sum(w) == 1,  # Somme = 1
            w >= self.min_weight,  # Poids minimum
            w <= self.max_weight,  # Poids maximum
            portfolio_variance <= 0.25,  # Volatilité max 50%
        ]
        
        # Résolution
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = pd.Series(w.value, index=expected_returns.index)
            weights = weights.clip(0, 1)  # Sécurité
            weights /= weights.sum()  # Normalisation
            return weights
        else:
            logger.warning("⚠️ Optimisation Kelly-Markowitz échouée, fallback égal")
            return pd.Series(1/n_assets, index=expected_returns.index)
    
    def _optimize_risk_parity(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Optimisation Risk Parity
        Chaque alpha contribue également au risque total
        """
        
        n_assets = len(cov_matrix)
        
        def risk_parity_objective(weights):
            """Objectif: minimiser différences entre contributions au risque"""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            
            # Contributions marginales au risque
            marginal_contrib = (cov_matrix.values @ weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Égalisation des contributions
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Somme = 1
        ]
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Point initial
        x0 = np.ones(n_assets) / n_assets
        
        # Optimisation
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            weights = pd.Series(result.x, index=cov_matrix.index)
            weights = weights.clip(0, 1)
            weights /= weights.sum()
            return weights
        else:
            logger.warning("⚠️ Risk Parity échoué, fallback égal")
            return pd.Series(1/n_assets, index=cov_matrix.index)
    
    def _optimize_max_diversification(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Optimisation Maximum Diversification
        Maximise ratio: volatilité moyenne pondérée / volatilité portfolio
        """
        
        n_assets = len(cov_matrix)
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        
        # Variables CVXPY
        w = cp.Variable(n_assets)
        
        # Volatilité moyenne pondérée
        weighted_avg_vol = volatilities @ w
        
        # Volatilité du portfolio
        portfolio_vol = cp.sqrt(cp.quad_form(w, cov_matrix.values))
        
        # Objectif: maximiser diversification ratio
        # Équivalent: minimiser volatilité portfolio / volatilité moyenne
        objective = portfolio_vol / (weighted_avg_vol + 1e-8)
        
        # Contraintes
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight,
        ]
        
        # Résolution
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = pd.Series(w.value, index=cov_matrix.index)
            weights = weights.clip(0, 1)
            weights /= weights.sum()
            return weights
        else:
            logger.warning("⚠️ Max Diversification échoué, fallback égal")
            return pd.Series(1/n_assets, index=cov_matrix.index)
    
    def _optimize_black_litterman(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, 
                                 regime_info: Optional[Dict] = None) -> pd.Series:
        """
        Optimisation Black-Litterman avec vues de marché
        """
        
        # Pour l'instant, simplification: Markowitz avec prior égal
        # TODO: Implémenter vraies vues Black-Litterman
        logger.debug("📊 Black-Litterman simplifié (Markowitz avec prior)")
        
        return self._optimize_kelly_markowitz(expected_returns, cov_matrix)
    
    def _apply_constraints(self, weights: pd.Series, current_positions: Optional[Dict] = None) -> pd.Series:
        """Application contraintes finales et coûts de transaction"""
        
        # Normalisation
        weights = weights.clip(self.min_weight, self.max_weight)
        weights /= weights.sum()
        
        # Pénalité coûts de transaction si positions actuelles
        if current_positions is not None:
            current_weights = pd.Series(current_positions).reindex(weights.index, fill_value=0)
            turnover = np.abs(weights - current_weights).sum()
            
            if turnover > 0.1:  # Turnover > 10%
                logger.info(f"📊 Turnover élevé: {turnover:.2%}, ajustement pour coûts")
                # Réduction turnover par déplacement vers positions actuelles
                adjustment_factor = 0.8
                weights = adjustment_factor * weights + (1 - adjustment_factor) * current_weights
                weights /= weights.sum()
        
        return weights
    
    def _calculate_portfolio_metrics(self, weights: pd.Series, expected_returns: pd.Series,
                                   cov_matrix: pd.DataFrame, historical_returns: pd.DataFrame) -> Dict:
        """Calcul métriques complètes du portfolio optimisé"""
        
        # Métriques ex-ante (prévisionnelles)
        expected_return = (weights * expected_returns).sum()
        volatility = np.sqrt(weights @ cov_matrix @ weights)
        sharpe_expected = expected_return / (volatility + 1e-8)
        
        # Métriques ex-post (historiques)
        portfolio_returns = (historical_returns * weights).sum(axis=1)
        
        metrics = {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_expected': sharpe_expected,
            'weights_entropy': -np.sum(weights * np.log(weights + 1e-8)),  # Diversification
            'max_weight': weights.max(),
            'min_weight': weights.min(),
            'n_active_positions': (weights > 0.01).sum(),
        }
        
        # Métriques historiques si suffisamment de données
        if len(portfolio_returns) > 30:
            historical_metrics = comprehensive_metrics(portfolio_returns.values)
            metrics.update({
                'historical_sharpe': historical_metrics.get('sharpe_ratio', 0),
                'historical_volatility': portfolio_returns.std(),
                'max_drawdown': historical_metrics.get('max_drawdown', 0),
                'calmar_ratio': historical_metrics.get('calmar_ratio', 0),
                'psr': historical_metrics.get('psr', 0),
            })
        
        return metrics
    
    def _log_to_mlflow(self, weights: pd.Series, metrics: Dict, returns: pd.DataFrame):
        """Logging MLflow pour suivi optimisations"""
        
        try:
            with mlflow.start_run(nested=True):
                # Paramètres optimisation
                mlflow.log_param("optimization_method", self.method)
                mlflow.log_param("risk_aversion", self.risk_aversion)
                mlflow.log_param("max_weight", self.max_weight)
                mlflow.log_param("n_alphas", len(weights))
                mlflow.log_param("n_observations", len(returns))
                
                # Métriques portfolio
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"portfolio_{key}", value)
                
                # Poids individuels
                for alpha, weight in weights.items():
                    mlflow.log_metric(f"weight_{alpha}", weight)
                
                logger.debug("📊 Optimisation loggée dans MLflow")
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur logging MLflow: {e}")
    
    def _get_equal_weight_fallback(self, alpha_names: List[str]) -> Dict:
        """Fallback: allocation égale en cas d'échec optimisation"""
        
        n_alphas = len(alpha_names)
        equal_weight = 1.0 / n_alphas
        
        weights = pd.Series(equal_weight, index=alpha_names)
        
        return {
            'weights': weights,
            'expected_returns': pd.Series(0.0, index=alpha_names),
            'covariance_matrix': pd.DataFrame(np.eye(n_alphas), index=alpha_names, columns=alpha_names),
            'portfolio_metrics': {
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_expected': 0.0,
                'method': 'equal_weight_fallback'
            },
            'optimization_time': datetime.now(),
            'method': 'equal_weight_fallback',
            'alpha_names': alpha_names
        }
    
    def rebalance_check(self, current_time: datetime) -> bool:
        """Vérifie si rebalancement nécessaire selon fréquence"""
        
        if self.last_optimization_time is None:
            return True
        
        time_diff = current_time - self.last_optimization_time
        
        if self.rebalancing_frequency == "daily":
            return time_diff >= timedelta(days=1)
        elif self.rebalancing_frequency == "weekly":
            return time_diff >= timedelta(weeks=1)
        elif self.rebalancing_frequency == "monthly":
            return time_diff >= timedelta(days=30)
        else:
            return time_diff >= timedelta(hours=1)  # Par défaut
    
    def get_allocation_summary(self) -> pd.DataFrame:
        """Résumé des allocations historiques"""
        
        if not self.historical_allocations:
            return pd.DataFrame()
        
        summary_data = []
        for allocation in self.historical_allocations:
            row = {
                'timestamp': allocation['optimization_time'],
                'method': allocation['method'],
                'expected_return': allocation['portfolio_metrics'].get('expected_return', 0),
                'volatility': allocation['portfolio_metrics'].get('volatility', 0),
                'sharpe': allocation['portfolio_metrics'].get('sharpe_expected', 0),
                'max_weight': allocation['portfolio_metrics'].get('max_weight', 0),
                'n_active': allocation['portfolio_metrics'].get('n_active_positions', 0),
            }
            
            # Poids individuels
            for alpha, weight in allocation['weights'].items():
                row[f'weight_{alpha}'] = weight
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


# ==============================================
# FONCTIONS UTILITAIRES
# ==============================================

def optimize_portfolio_simple(alpha_returns: pd.DataFrame, 
                              method: str = "kelly_markowitz",
                              **kwargs) -> Dict:
    """
    Interface simplifiée pour optimisation rapide
    """
    
    optimizer = QuantPortfolioOptimizer(method=method, **kwargs)
    return optimizer.optimize_portfolio(alpha_returns)


def backtest_portfolio_allocation(alpha_returns: pd.DataFrame,
                                 method: str = "kelly_markowitz",
                                 rebalancing_frequency: str = "weekly",
                                 initial_capital: float = 10000) -> pd.DataFrame:
    """
    Backtest d'une stratégie d'allocation
    """
    
    logger.info(f"📊 Backtest allocation: {method}, rebalancing={rebalancing_frequency}")
    
    optimizer = QuantPortfolioOptimizer(
        method=method,
        rebalancing_frequency=rebalancing_frequency
    )
    
    results = []
    current_capital = initial_capital
    current_weights = None
    
    # Fenêtre glissante pour optimisation
    lookback_window = 252  # 1 an
    
    for i in range(lookback_window, len(alpha_returns)):
        current_date = alpha_returns.index[i]
        
        # Données pour optimisation
        train_data = alpha_returns.iloc[i-lookback_window:i]
        
        # Rebalancement si nécessaire
        if optimizer.rebalance_check(current_date):
            allocation_result = optimizer.optimize_portfolio(train_data)
            current_weights = allocation_result['weights']
            optimizer.last_optimization_time = current_date
        
        # Calcul return du portfolio ce jour
        if current_weights is not None:
            daily_returns = alpha_returns.iloc[i]
            portfolio_return = (current_weights * daily_returns).sum()
            current_capital *= (1 + portfolio_return)
            
            results.append({
                'date': current_date,
                'portfolio_value': current_capital,
                'portfolio_return': portfolio_return,
                'weights': current_weights.to_dict()
            })
    
    backtest_df = pd.DataFrame(results)
    backtest_df.set_index('date', inplace=True)
    
    logger.info(f"✅ Backtest terminé: return total={(current_capital/initial_capital-1)*100:.2f}%")
    
    return backtest_df