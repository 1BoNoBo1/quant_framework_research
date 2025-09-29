"""
Risk Analyzer pour QFrame
========================

Module d'analyse de risque avancée pour les stratégies quantitatives.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

class RiskAnalyzer:
    """Analyseur de risque pour stratégies quantitatives"""

    def __init__(self, confidence_levels: List[float] = [0.90, 0.95, 0.99]):
        """Initialise l'analyseur de risque"""
        self.confidence_levels = confidence_levels
        self.risk_free_rate = 0.02  # 2% taux sans risque annuel

    def calculate_var_cvar(self, returns: pd.Series, method: str = 'historical') -> Dict[str, Dict[str, float]]:
        """Calcule VaR et CVaR pour différents niveaux de confiance"""

        if len(returns) == 0:
            return {}

        var_cvar_results = {}

        for confidence in self.confidence_levels:
            if method == 'historical':
                # VaR historique
                var_value = np.percentile(returns, (1 - confidence) * 100)

                # CVaR (Expected Shortfall)
                cvar_value = returns[returns <= var_value].mean()

            elif method == 'parametric':
                # VaR paramétrique (distribution normale)
                mean_return = returns.mean()
                std_return = returns.std()
                z_score = stats.norm.ppf(1 - confidence)
                var_value = mean_return + z_score * std_return

                # CVaR paramétrique
                phi_z = stats.norm.pdf(z_score)
                cvar_value = mean_return - std_return * phi_z / (1 - confidence)

            else:
                raise ValueError("Method must be 'historical' or 'parametric'")

            var_cvar_results[f'confidence_{int(confidence*100)}'] = {
                'var': var_value,
                'cvar': cvar_value,
                'var_pct': var_value * 100,
                'cvar_pct': cvar_value * 100
            }

        return var_cvar_results

    def calculate_maximum_drawdown_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyse détaillée du drawdown maximum"""

        if len(returns) == 0:
            return {}

        # Calcul des drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()

        # Durée du maximum drawdown
        max_dd_start = running_max[running_max == running_max.loc[max_dd_date]].index[0]
        recovery_mask = cumulative >= running_max.loc[max_dd_date]
        recovery_dates = cumulative[recovery_mask & (cumulative.index > max_dd_date)]

        if len(recovery_dates) > 0:
            recovery_date = recovery_dates.index[0]
            max_dd_duration = (recovery_date - max_dd_start).days
        else:
            recovery_date = None
            max_dd_duration = (returns.index[-1] - max_dd_start).days

        # Statistiques des drawdowns
        drawdown_periods = self._identify_drawdown_periods(drawdown)

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_date': max_dd_date,
            'max_drawdown_start': max_dd_start,
            'recovery_date': recovery_date,
            'max_drawdown_duration_days': max_dd_duration,
            'current_drawdown': drawdown.iloc[-1],
            'current_drawdown_pct': drawdown.iloc[-1] * 100,
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'drawdown_frequency': len(drawdown_periods),
            'avg_recovery_time_days': np.mean([period['duration'] for period in drawdown_periods]) if drawdown_periods else 0
        }

    def _identify_drawdown_periods(self, drawdown: pd.Series) -> List[Dict[str, Any]]:
        """Identifie les périodes de drawdown distinctes"""

        periods = []
        in_drawdown = False
        start_date = None

        for date, dd_value in drawdown.items():
            if dd_value < 0 and not in_drawdown:
                # Début d'une période de drawdown
                in_drawdown = True
                start_date = date
                min_dd = dd_value
                min_dd_date = date

            elif dd_value < 0 and in_drawdown:
                # Continuation du drawdown
                if dd_value < min_dd:
                    min_dd = dd_value
                    min_dd_date = date

            elif dd_value >= 0 and in_drawdown:
                # Fin de la période de drawdown
                in_drawdown = False
                duration = (date - start_date).days

                periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'duration': duration,
                    'max_drawdown': min_dd,
                    'max_drawdown_date': min_dd_date
                })

        # Si on est encore en drawdown à la fin
        if in_drawdown:
            duration = (drawdown.index[-1] - start_date).days
            periods.append({
                'start_date': start_date,
                'end_date': drawdown.index[-1],
                'duration': duration,
                'max_drawdown': min_dd,
                'max_drawdown_date': min_dd_date
            })

        return periods

    def calculate_volatility_analysis(self, returns: pd.Series, windows: List[int] = [30, 60, 252]) -> Dict[str, Any]:
        """Analyse de volatilité sur différentes fenêtres"""

        if len(returns) == 0:
            return {}

        volatility_analysis = {
            'current_volatility_annualized': returns.std() * np.sqrt(252),
            'volatility_windows': {}
        }

        for window in windows:
            if len(returns) >= window:
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)

                volatility_analysis['volatility_windows'][f'{window}d'] = {
                    'current': rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0,
                    'mean': rolling_vol.mean(),
                    'min': rolling_vol.min(),
                    'max': rolling_vol.max(),
                    'std': rolling_vol.std()
                }

        # Détection de clusters de volatilité
        vol_30d = returns.rolling(30).std() * np.sqrt(252) if len(returns) >= 30 else pd.Series()

        if len(vol_30d) > 0:
            vol_median = vol_30d.median()
            high_vol_threshold = vol_median * 1.5
            low_vol_threshold = vol_median * 0.5

            volatility_analysis['volatility_regimes'] = {
                'low_volatility_periods': (vol_30d < low_vol_threshold).sum(),
                'normal_volatility_periods': ((vol_30d >= low_vol_threshold) & (vol_30d <= high_vol_threshold)).sum(),
                'high_volatility_periods': (vol_30d > high_vol_threshold).sum(),
                'current_regime': 'high' if vol_30d.iloc[-1] > high_vol_threshold else 'low' if vol_30d.iloc[-1] < low_vol_threshold else 'normal'
            }

        return volatility_analysis

    def calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcule les métriques de risque de queue"""

        if len(returns) == 0:
            return {}

        # Statistiques de queue
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        excess_kurtosis = kurtosis - 3

        # Analyse des queues
        left_tail = returns[returns < returns.quantile(0.05)]
        right_tail = returns[returns > returns.quantile(0.95)]

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'excess_kurtosis': excess_kurtosis,
            'left_tail_mean': left_tail.mean() if len(left_tail) > 0 else 0,
            'right_tail_mean': right_tail.mean() if len(right_tail) > 0 else 0,
            'tail_ratio': abs(left_tail.mean() / right_tail.mean()) if len(right_tail) > 0 and right_tail.mean() != 0 else 0,
            'extreme_loss_days': (returns < returns.quantile(0.01)).sum(),
            'extreme_gain_days': (returns > returns.quantile(0.99)).sum()
        }

    def calculate_beta_analysis(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calcule l'analyse bêta vs marché"""

        if market_returns is None or len(returns) == 0:
            return {'beta': None, 'alpha': None, 'correlation': None}

        # Aligner les séries
        aligned_data = pd.DataFrame({
            'strategy': returns,
            'market': market_returns
        }).dropna()

        if len(aligned_data) < 30:  # Minimum de données pour un calcul fiable
            return {'beta': None, 'alpha': None, 'correlation': None}

        # Calcul du bêta
        covariance = aligned_data['strategy'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        beta = covariance / market_variance if market_variance != 0 else 0

        # Calcul de l'alpha
        strategy_mean = aligned_data['strategy'].mean() * 252  # Annualisé
        market_mean = aligned_data['market'].mean() * 252  # Annualisé
        alpha = strategy_mean - beta * market_mean

        # Corrélation
        correlation = aligned_data['strategy'].corr(aligned_data['market'])

        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'tracking_error': (aligned_data['strategy'] - aligned_data['market']).std() * np.sqrt(252),
            'information_ratio': alpha / (aligned_data['strategy'] - aligned_data['market']).std() / np.sqrt(252) if (aligned_data['strategy'] - aligned_data['market']).std() != 0 else 0
        }

    def analyze_comprehensive_risk(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Analyse complète de risque"""

        analysis = {}

        # VaR/CVaR
        analysis['var_cvar_historical'] = self.calculate_var_cvar(returns, method='historical')
        analysis['var_cvar_parametric'] = self.calculate_var_cvar(returns, method='parametric')

        # Analyse des drawdowns
        analysis['drawdown_analysis'] = self.calculate_maximum_drawdown_analysis(returns)

        # Analyse de volatilité
        analysis['volatility_analysis'] = self.calculate_volatility_analysis(returns)

        # Métriques de queue
        analysis['tail_risk_metrics'] = self.calculate_tail_risk_metrics(returns)

        # Analyse bêta (si données marché disponibles)
        analysis['beta_analysis'] = self.calculate_beta_analysis(returns, market_returns)

        # Métriques de risque additionnelles
        analysis['additional_risk_metrics'] = self._calculate_additional_risk_metrics(returns)

        # Résumé du profil de risque
        analysis['risk_profile_summary'] = self._generate_risk_profile_summary(analysis)

        return analysis

    def _calculate_additional_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcule des métriques de risque additionnelles"""

        if len(returns) == 0:
            return {}

        # Downside deviation
        downside_returns = returns[returns < returns.mean()]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0

        # Ulcer Index
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ulcer_index = np.sqrt((drawdown ** 2).mean()) if len(drawdown) > 0 else 0

        # Pain Index (moyenne des drawdowns)
        pain_index = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0

        return {
            'downside_deviation_annualized': downside_deviation * np.sqrt(252),
            'ulcer_index': ulcer_index,
            'pain_index': pain_index,
            'calmar_ratio': (returns.mean() * 252) / abs(drawdown.min()) if drawdown.min() < 0 else 0,
            'sterling_ratio': (returns.mean() * 252) / (abs(drawdown.min()) + 0.1) if drawdown.min() < 0 else 0
        }

    def _generate_risk_profile_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Génère un résumé du profil de risque"""

        summary = {
            'overall_risk_level': 'Unknown',
            'primary_risk_factors': [],
            'risk_warnings': [],
            'risk_score': 0  # Sur 100
        }

        # Calcul du score de risque basé sur plusieurs facteurs
        risk_factors = []

        # Facteur 1: Volatilité
        vol_analysis = analysis.get('volatility_analysis', {})
        current_vol = vol_analysis.get('current_volatility_annualized', 0)

        if current_vol < 0.1:  # < 10%
            risk_factors.append(10)  # Faible risque
        elif current_vol < 0.2:  # < 20%
            risk_factors.append(30)  # Risque modéré
        elif current_vol < 0.4:  # < 40%
            risk_factors.append(60)  # Risque élevé
        else:
            risk_factors.append(90)  # Risque très élevé

        # Facteur 2: Maximum Drawdown
        dd_analysis = analysis.get('drawdown_analysis', {})
        max_dd = abs(dd_analysis.get('max_drawdown', 0))

        if max_dd < 0.05:  # < 5%
            risk_factors.append(10)
        elif max_dd < 0.15:  # < 15%
            risk_factors.append(40)
        elif max_dd < 0.30:  # < 30%
            risk_factors.append(70)
        else:
            risk_factors.append(95)

        # Facteur 3: VaR 95%
        var_hist = analysis.get('var_cvar_historical', {}).get('confidence_95', {}).get('var', 0)
        if abs(var_hist) < 0.02:  # < 2%
            risk_factors.append(15)
        elif abs(var_hist) < 0.05:  # < 5%
            risk_factors.append(45)
        else:
            risk_factors.append(80)

        # Score global
        risk_score = np.mean(risk_factors) if risk_factors else 50

        # Classification du niveau de risque
        if risk_score < 25:
            summary['overall_risk_level'] = 'Conservative'
        elif risk_score < 50:
            summary['overall_risk_level'] = 'Moderate'
        elif risk_score < 75:
            summary['overall_risk_level'] = 'Aggressive'
        else:
            summary['overall_risk_level'] = 'High Risk'

        summary['risk_score'] = risk_score

        # Identification des facteurs de risque principaux
        if current_vol > 0.3:
            summary['primary_risk_factors'].append('High Volatility')
        if max_dd > 0.2:
            summary['primary_risk_factors'].append('Large Drawdowns')

        tail_metrics = analysis.get('tail_risk_metrics', {})
        if abs(tail_metrics.get('skewness', 0)) > 1:
            summary['primary_risk_factors'].append('Skewed Returns')
        if tail_metrics.get('excess_kurtosis', 0) > 2:
            summary['primary_risk_factors'].append('Fat Tails')

        return summary