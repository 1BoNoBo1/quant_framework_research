"""
Statistical Validator pour QFrame
=================================

Module de validation statistique pour les stratégies quantitatives.
Implémente des tests de robustesse et de signification statistique.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import warnings
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

class StatisticalValidator:
    """Validateur statistique pour stratégies quantitatives"""

    def __init__(self, significance_level: float = 0.05):
        """Initialise le validateur statistique"""
        self.significance_level = significance_level
        self.min_sample_size = 30

    def test_normality(self, returns: pd.Series) -> Dict[str, Any]:
        """Tests de normalité des retours"""

        if len(returns) < self.min_sample_size:
            return {'error': 'Sample size too small for normality tests'}

        results = {}

        # Test de Shapiro-Wilk (pour échantillons < 5000)
        if len(returns) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(returns)
            results['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > self.significance_level,
                'interpretation': 'Normal' if shapiro_p > self.significance_level else 'Non-normal'
            }

        # Test de Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(returns)
        results['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'is_normal': jb_p > self.significance_level,
            'interpretation': 'Normal' if jb_p > self.significance_level else 'Non-normal'
        }

        # Test de Kolmogorov-Smirnov vs distribution normale
        fitted_mean = returns.mean()
        fitted_std = returns.std()
        ks_stat, ks_p = stats.kstest(returns, lambda x: stats.norm.cdf(x, fitted_mean, fitted_std))
        results['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_normal': ks_p > self.significance_level,
            'interpretation': 'Normal' if ks_p > self.significance_level else 'Non-normal'
        }

        # Test de D'Agostino-Pearson
        try:
            dag_stat, dag_p = stats.normaltest(returns)
            results['dagostino_pearson'] = {
                'statistic': dag_stat,
                'p_value': dag_p,
                'is_normal': dag_p > self.significance_level,
                'interpretation': 'Normal' if dag_p > self.significance_level else 'Non-normal'
            }
        except:
            results['dagostino_pearson'] = {'error': 'Test failed'}

        # Résumé
        normal_tests = [test for test in results.values() if 'is_normal' in test]
        normal_count = sum(1 for test in normal_tests if test['is_normal'])
        results['summary'] = {
            'tests_passed': normal_count,
            'total_tests': len(normal_tests),
            'overall_normality': normal_count >= len(normal_tests) // 2
        }

        return results

    def test_stationarity(self, returns: pd.Series) -> Dict[str, Any]:
        """Tests de stationnarité des retours"""

        if len(returns) < self.min_sample_size:
            return {'error': 'Sample size too small for stationarity tests'}

        results = {}

        # Test de Dickey-Fuller Augmenté
        try:
            from statsmodels.tsa.stattools import adfuller

            adf_result = adfuller(returns.dropna(), autolag='AIC')
            results['augmented_dickey_fuller'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < self.significance_level,
                'interpretation': 'Stationary' if adf_result[1] < self.significance_level else 'Non-stationary'
            }
        except ImportError:
            results['augmented_dickey_fuller'] = {'error': 'statsmodels not available'}

        # Test KPSS
        try:
            from statsmodels.tsa.stattools import kpss

            kpss_result = kpss(returns.dropna(), regression='c')
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > self.significance_level,  # KPSS: H0 = stationary
                'interpretation': 'Stationary' if kpss_result[1] > self.significance_level else 'Non-stationary'
            }
        except ImportError:
            results['kpss'] = {'error': 'statsmodels not available'}

        return results

    def test_autocorrelation(self, returns: pd.Series, max_lags: int = 20) -> Dict[str, Any]:
        """Tests d'autocorrélation des retours"""

        if len(returns) < max_lags + 10:
            return {'error': 'Sample size too small for autocorrelation tests'}

        results = {
            'autocorrelations': {},
            'ljung_box_tests': {},
            'portmanteau_test': {}
        }

        # Calcul des autocorrélations
        for lag in range(1, min(max_lags + 1, len(returns) // 4)):
            autocorr = returns.autocorr(lag)
            results['autocorrelations'][f'lag_{lag}'] = autocorr

        # Test de Ljung-Box pour différents lags
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb_result = acorr_ljungbox(returns, lags=min(10, len(returns) // 4), return_df=True)

            for lag in lb_result.index:
                results['ljung_box_tests'][f'lag_{lag}'] = {
                    'statistic': lb_result.loc[lag, 'lb_stat'],
                    'p_value': lb_result.loc[lag, 'lb_pvalue'],
                    'no_autocorrelation': lb_result.loc[lag, 'lb_pvalue'] > self.significance_level
                }

        except ImportError:
            results['ljung_box_tests'] = {'error': 'statsmodels not available'}

        # Test de Breusch-Godfrey (pour autocorrélation d'ordre supérieur)
        significant_autocorrs = [abs(autocorr) for autocorr in results['autocorrelations'].values() if abs(autocorr) > 0.05]

        results['summary'] = {
            'max_autocorrelation': max(abs(autocorr) for autocorr in results['autocorrelations'].values()),
            'significant_lags': len(significant_autocorrs),
            'has_significant_autocorrelation': len(significant_autocorrs) > 0
        }

        return results

    def test_heteroskedasticity(self, returns: pd.Series) -> Dict[str, Any]:
        """Tests d'hétéroscédasticité (volatilité non-constante)"""

        if len(returns) < 50:
            return {'error': 'Sample size too small for heteroskedasticity tests'}

        results = {}

        # Test de Breusch-Pagan
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            from statsmodels.regression.linear_model import OLS
            import statsmodels.api as sm

            # Régression simple pour obtenir les résidus
            X = sm.add_constant(np.arange(len(returns)))
            model = OLS(returns, X).fit()

            bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
            results['breusch_pagan'] = {
                'statistic': bp_stat,
                'p_value': bp_p,
                'homoskedastic': bp_p > self.significance_level,
                'interpretation': 'Homoskedastic' if bp_p > self.significance_level else 'Heteroskedastic'
            }

        except ImportError:
            results['breusch_pagan'] = {'error': 'statsmodels not available'}

        # Test de White
        try:
            from statsmodels.stats.diagnostic import het_white

            X = sm.add_constant(np.arange(len(returns)))
            model = OLS(returns, X).fit()

            white_stat, white_p, _, _ = het_white(model.resid, model.model.exog)
            results['white'] = {
                'statistic': white_stat,
                'p_value': white_p,
                'homoskedastic': white_p > self.significance_level,
                'interpretation': 'Homoskedastic' if white_p > self.significance_level else 'Heteroskedastic'
            }

        except ImportError:
            results['white'] = {'error': 'statsmodels not available'}

        # Test simple basé sur la volatilité roulante
        if len(returns) >= 60:
            rolling_vol = returns.rolling(30).std()
            vol_changes = rolling_vol.pct_change().dropna()

            # Test de variance des changements de volatilité
            vol_stability_stat, vol_stability_p = stats.levene(
                vol_changes[:len(vol_changes)//2],
                vol_changes[len(vol_changes)//2:]
            )

            results['volatility_stability'] = {
                'statistic': vol_stability_stat,
                'p_value': vol_stability_p,
                'stable_volatility': vol_stability_p > self.significance_level,
                'interpretation': 'Stable volatility' if vol_stability_p > self.significance_level else 'Changing volatility'
            }

        return results

    def test_outliers(self, returns: pd.Series, method: str = 'iqr') -> Dict[str, Any]:
        """Détection d'outliers dans les retours"""

        if len(returns) < self.min_sample_size:
            return {'error': 'Sample size too small for outlier detection'}

        results = {
            'outliers_detected': [],
            'outlier_indices': [],
            'outlier_values': []
        }

        if method == 'iqr':
            # Méthode IQR (Interquartile Range)
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = returns[(returns < lower_bound) | (returns > upper_bound)]

        elif method == 'zscore':
            # Méthode Z-score
            z_scores = np.abs(stats.zscore(returns))
            outliers = returns[z_scores > 3]

        elif method == 'modified_zscore':
            # Z-score modifié (plus robuste)
            median = returns.median()
            mad = np.median(np.abs(returns - median))
            modified_z_scores = 0.6745 * (returns - median) / mad
            outliers = returns[np.abs(modified_z_scores) > 3.5]

        else:
            return {'error': f'Unknown method: {method}'}

        results['outliers_detected'] = outliers.tolist()
        results['outlier_indices'] = outliers.index.tolist()
        results['outlier_values'] = outliers.values.tolist()
        results['outlier_count'] = len(outliers)
        results['outlier_percentage'] = len(outliers) / len(returns) * 100

        # Classification de la sévérité
        if results['outlier_percentage'] < 1:
            severity = 'Low'
        elif results['outlier_percentage'] < 5:
            severity = 'Moderate'
        else:
            severity = 'High'

        results['outlier_severity'] = severity

        return results

    def walk_forward_analysis(self, returns: pd.Series, n_splits: int = 5) -> Dict[str, Any]:
        """Analyse walk-forward pour tester la robustesse"""

        if len(returns) < n_splits * 30:
            return {'error': 'Not enough data for walk-forward analysis'}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(returns)):
            train_returns = returns.iloc[train_idx]
            test_returns = returns.iloc[test_idx]

            # Métriques sur chaque fold
            fold_result = {
                'fold': fold + 1,
                'train_size': len(train_returns),
                'test_size': len(test_returns),
                'train_mean': train_returns.mean(),
                'test_mean': test_returns.mean(),
                'train_std': train_returns.std(),
                'test_std': test_returns.std(),
                'train_sharpe': train_returns.mean() / train_returns.std() * np.sqrt(252) if train_returns.std() > 0 else 0,
                'test_sharpe': test_returns.mean() / test_returns.std() * np.sqrt(252) if test_returns.std() > 0 else 0
            }

            fold_results.append(fold_result)

        # Statistiques agrégées
        test_sharpes = [fold['test_sharpe'] for fold in fold_results]
        train_sharpes = [fold['train_sharpe'] for fold in fold_results]

        stability_analysis = {
            'fold_results': fold_results,
            'mean_test_sharpe': np.mean(test_sharpes),
            'std_test_sharpe': np.std(test_sharpes),
            'min_test_sharpe': np.min(test_sharpes),
            'max_test_sharpe': np.max(test_sharpes),
            'sharpe_stability': np.std(test_sharpes) / (np.mean(test_sharpes) + 1e-8),  # CV des Sharpe ratios
            'positive_folds': sum(1 for sharpe in test_sharpes if sharpe > 0),
            'consistency_rate': sum(1 for sharpe in test_sharpes if sharpe > 0) / len(test_sharpes)
        }

        # Classification de la robustesse
        if stability_analysis['consistency_rate'] >= 0.8 and stability_analysis['sharpe_stability'] < 0.5:
            robustness = 'High'
        elif stability_analysis['consistency_rate'] >= 0.6 and stability_analysis['sharpe_stability'] < 1.0:
            robustness = 'Moderate'
        else:
            robustness = 'Low'

        stability_analysis['robustness_classification'] = robustness

        return stability_analysis

    def monte_carlo_simulation(self, returns: pd.Series, n_simulations: int = 1000, period_length: Optional[int] = None) -> Dict[str, Any]:
        """Simulation Monte Carlo pour tester la robustesse"""

        if len(returns) < 30:
            return {'error': 'Not enough data for Monte Carlo simulation'}

        if period_length is None:
            period_length = len(returns)

        # Paramètres empiriques
        mean_return = returns.mean()
        std_return = returns.std()

        # Générer des simulations
        simulated_returns = []
        simulated_sharpes = []
        simulated_max_dds = []

        np.random.seed(42)  # Pour la reproductibilité

        for _ in range(n_simulations):
            # Simulation avec distribution normale
            sim_returns = np.random.normal(mean_return, std_return, period_length)
            sim_series = pd.Series(sim_returns)

            # Métriques pour cette simulation
            sim_sharpe = sim_returns.mean() / sim_returns.std() * np.sqrt(252) if sim_returns.std() > 0 else 0

            # Drawdown maximum
            sim_cumulative = (1 + sim_series).cumprod()
            sim_running_max = sim_cumulative.expanding().max()
            sim_drawdown = (sim_cumulative - sim_running_max) / sim_running_max
            sim_max_dd = sim_drawdown.min()

            simulated_returns.append(sim_returns)
            simulated_sharpes.append(sim_sharpe)
            simulated_max_dds.append(sim_max_dd)

        # Statistiques des simulations
        actual_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calcul du drawdown actuel
        actual_cumulative = (1 + returns).cumprod()
        actual_running_max = actual_cumulative.expanding().max()
        actual_drawdown = (actual_cumulative - actual_running_max) / actual_running_max
        actual_max_dd = actual_drawdown.min()

        results = {
            'n_simulations': n_simulations,
            'simulated_sharpe_mean': np.mean(simulated_sharpes),
            'simulated_sharpe_std': np.std(simulated_sharpes),
            'actual_sharpe': actual_sharpe,
            'sharpe_percentile': stats.percentileofscore(simulated_sharpes, actual_sharpe),
            'simulated_max_dd_mean': np.mean(simulated_max_dds),
            'simulated_max_dd_std': np.std(simulated_max_dds),
            'actual_max_dd': actual_max_dd,
            'max_dd_percentile': stats.percentileofscore(simulated_max_dds, actual_max_dd),
            'outperformance_probability': sum(1 for sharpe in simulated_sharpes if sharpe < actual_sharpe) / n_simulations
        }

        # Classification de la performance vs simulation
        if results['sharpe_percentile'] > 75:
            performance_vs_random = 'Above Average'
        elif results['sharpe_percentile'] > 50:
            performance_vs_random = 'Average'
        else:
            performance_vs_random = 'Below Average'

        results['performance_vs_random'] = performance_vs_random

        return results

    def comprehensive_validation(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Validation statistique complète"""

        validation_results = {
            'sample_info': {
                'sample_size': len(returns),
                'start_date': returns.index[0] if len(returns) > 0 else None,
                'end_date': returns.index[-1] if len(returns) > 0 else None,
                'valid_sample': len(returns) >= self.min_sample_size
            }
        }

        if len(returns) < self.min_sample_size:
            validation_results['error'] = f'Sample size too small. Minimum required: {self.min_sample_size}'
            return validation_results

        # Tests statistiques
        validation_results['normality_tests'] = self.test_normality(returns)
        validation_results['stationarity_tests'] = self.test_stationarity(returns)
        validation_results['autocorrelation_tests'] = self.test_autocorrelation(returns)
        validation_results['heteroskedasticity_tests'] = self.test_heteroskedasticity(returns)
        validation_results['outlier_analysis'] = self.test_outliers(returns)

        # Analyses de robustesse
        validation_results['walk_forward_analysis'] = self.walk_forward_analysis(returns)
        validation_results['monte_carlo_simulation'] = self.monte_carlo_simulation(returns)

        # Score global de validation
        validation_results['overall_validation_score'] = self._calculate_overall_validation_score(validation_results)

        return validation_results

    def _calculate_overall_validation_score(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule un score global de validation"""

        score_components = []

        # Score normalité (20%)
        normality = validation_results.get('normality_tests', {}).get('summary', {})
        if 'tests_passed' in normality and 'total_tests' in normality:
            normality_score = (normality['tests_passed'] / normality['total_tests']) * 100
            score_components.append(('normality', normality_score, 0.2))

        # Score stationnarité (15%)
        stationarity = validation_results.get('stationarity_tests', {})
        stationarity_count = sum(1 for test in stationarity.values()
                               if isinstance(test, dict) and test.get('is_stationary', False))
        stationarity_total = sum(1 for test in stationarity.values()
                               if isinstance(test, dict) and 'is_stationary' in test)
        if stationarity_total > 0:
            stationarity_score = (stationarity_count / stationarity_total) * 100
            score_components.append(('stationarity', stationarity_score, 0.15))

        # Score autocorrélation (15%)
        autocorr = validation_results.get('autocorrelation_tests', {}).get('summary', {})
        if 'has_significant_autocorrelation' in autocorr:
            autocorr_score = 0 if autocorr['has_significant_autocorrelation'] else 100  # Pas d'autocorrélation = bon
            score_components.append(('autocorrelation', autocorr_score, 0.15))

        # Score outliers (10%)
        outliers = validation_results.get('outlier_analysis', {})
        if 'outlier_percentage' in outliers:
            outlier_pct = outliers['outlier_percentage']
            outlier_score = max(0, 100 - outlier_pct * 10)  # Pénalité pour les outliers
            score_components.append(('outliers', outlier_score, 0.1))

        # Score robustesse walk-forward (20%)
        wf = validation_results.get('walk_forward_analysis', {})
        if 'consistency_rate' in wf:
            wf_score = wf['consistency_rate'] * 100
            score_components.append(('walk_forward', wf_score, 0.2))

        # Score Monte Carlo (20%)
        mc = validation_results.get('monte_carlo_simulation', {})
        if 'sharpe_percentile' in mc:
            mc_score = mc['sharpe_percentile']
            score_components.append(('monte_carlo', mc_score, 0.2))

        # Calcul du score pondéré
        if score_components:
            weighted_score = sum(score * weight for _, score, weight in score_components)
            total_weight = sum(weight for _, _, weight in score_components)
            overall_score = weighted_score / total_weight if total_weight > 0 else 0
        else:
            overall_score = 0

        # Classification
        if overall_score >= 80:
            classification = 'Excellent'
        elif overall_score >= 70:
            classification = 'Good'
        elif overall_score >= 60:
            classification = 'Fair'
        else:
            classification = 'Poor'

        return {
            'overall_score': overall_score,
            'classification': classification,
            'score_components': {name: score for name, score, _ in score_components},
            'recommendation': self._get_validation_recommendation(classification, validation_results)
        }

    def _get_validation_recommendation(self, classification: str, validation_results: Dict[str, Any]) -> str:
        """Génère une recommandation basée sur les résultats de validation"""

        if classification == 'Excellent':
            return "Strategy shows excellent statistical properties. Recommended for production deployment."

        elif classification == 'Good':
            return "Strategy shows good statistical properties. Monitor performance closely in production."

        elif classification == 'Fair':
            recommendations = []

            # Recommandations spécifiques
            normality = validation_results.get('normality_tests', {}).get('summary', {})
            if not normality.get('overall_normality', True):
                recommendations.append("Consider non-parametric risk models due to non-normal returns")

            wf = validation_results.get('walk_forward_analysis', {})
            if wf.get('consistency_rate', 1) < 0.6:
                recommendations.append("Strategy lacks consistency - consider parameter optimization")

            outliers = validation_results.get('outlier_analysis', {})
            if outliers.get('outlier_percentage', 0) > 5:
                recommendations.append("High outlier percentage - implement robust risk controls")

            base_rec = "Strategy has acceptable properties but needs improvements: "
            return base_rec + "; ".join(recommendations) if recommendations else base_rec + "Monitor closely."

        else:  # Poor
            return "Strategy fails multiple statistical tests. Not recommended for production without significant improvements."