"""
📊 Probabilistic Metrics

Implements advanced statistical metrics for strategy validation:
- Probabilistic Sharpe Ratio (PSR)
- Deflated Sharpe Ratio (DSR)
- Information Coefficient statistics
- Bootstrap confidence intervals
- Bayesian performance estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.stats import norm, t
from scipy.special import gamma
import logging

logger = logging.getLogger(__name__)


class ProbabilisticMetrics:
    """
    🎯 Advanced Probabilistic Metrics for Strategy Validation

    Implements cutting-edge statistical methods used by institutional
    quantitative funds for rigorous strategy assessment.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate_probabilistic_sharpe(self,
                                     returns: pd.Series,
                                     benchmark_sr: float = 0.0,
                                     skewness: Optional[float] = None,
                                     kurtosis: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate Probabilistic Sharpe Ratio (PSR)

        PSR estimates the probability that the true Sharpe ratio exceeds a benchmark.
        Higher PSR indicates more statistical confidence in the strategy's performance.

        Args:
            returns: Strategy returns
            benchmark_sr: Benchmark Sharpe ratio to compare against
            skewness: Optional skewness override
            kurtosis: Optional kurtosis override

        Returns:
            Dictionary with PSR and related statistics
        """
        if len(returns) == 0:
            return {'probabilistic_sharpe': 0.0, 'error': 'No returns data'}

        try:
            # Calculate basic statistics
            n = len(returns)
            sample_sr = self._calculate_sharpe_ratio(returns)

            # Calculate higher moments if not provided
            if skewness is None:
                skewness = returns.skew()
            if kurtosis is None:
                kurtosis = returns.kurtosis()

            # PSR calculation with skewness and kurtosis adjustments
            sr_std = np.sqrt((1 + (sample_sr**2)/2 - skewness*sample_sr + (kurtosis-3)/4*(sample_sr**2)) / (n-1))

            # Z-score for PSR
            z_score = (sample_sr - benchmark_sr) / sr_std

            # Probabilistic Sharpe Ratio
            psr = norm.cdf(z_score)

            # Additional statistics
            confidence_interval = self._calculate_sr_confidence_interval(returns, self.confidence_level)

            return {
                'probabilistic_sharpe': psr,
                'sample_sharpe': sample_sr,
                'benchmark_sharpe': benchmark_sr,
                'z_score': z_score,
                'sharpe_std': sr_std,
                'confidence_interval_lower': confidence_interval[0],
                'confidence_interval_upper': confidence_interval[1],
                'skewness': skewness,
                'kurtosis': kurtosis,
                'sample_size': n
            }

        except Exception as e:
            logger.error(f"PSR calculation failed: {e}")
            return {'probabilistic_sharpe': 0.0, 'error': str(e)}

    def calculate_deflated_sharpe(self,
                                returns: pd.Series,
                                n_trials: int,
                                n_observations: Optional[int] = None,
                                expected_return_variance: float = 0.0) -> Dict[str, float]:
        """
        Calculate Deflated Sharpe Ratio (DSR)

        DSR adjusts for multiple testing bias when many strategies are tested.
        Essential for preventing false discoveries in strategy research.

        Args:
            returns: Strategy returns
            n_trials: Number of strategies tested (for multiple testing correction)
            n_observations: Number of observations (defaults to len(returns))
            expected_return_variance: Expected variance of returns under null hypothesis

        Returns:
            Dictionary with DSR and related statistics
        """
        try:
            if n_observations is None:
                n_observations = len(returns)

            sample_sr = self._calculate_sharpe_ratio(returns)

            # Expected maximum Sharpe ratio under null hypothesis
            gamma_val = -np.log(2) - np.log(np.log(n_trials))
            expected_max_sr = np.sqrt(gamma_val * (2 * np.log(n_trials) + gamma_val) / n_observations)

            # Variance of maximum Sharpe ratio
            var_max_sr = np.pi**2 / (6 * n_observations)

            # Deflated Sharpe Ratio
            dsr = (sample_sr - expected_max_sr) / np.sqrt(var_max_sr)

            # P-value for DSR
            p_value = 1 - norm.cdf(dsr)

            return {
                'deflated_sharpe': dsr,
                'sample_sharpe': sample_sr,
                'expected_max_sharpe': expected_max_sr,
                'variance_max_sharpe': var_max_sr,
                'p_value': p_value,
                'n_trials': n_trials,
                'n_observations': n_observations,
                'significant': p_value < 0.05
            }

        except Exception as e:
            logger.error(f"DSR calculation failed: {e}")
            return {'deflated_sharpe': 0.0, 'error': str(e)}

    def calculate_information_coefficient(self,
                                        predictions: pd.Series,
                                        actual_returns: pd.Series,
                                        method: str = 'pearson') -> Dict[str, float]:
        """
        Calculate Information Coefficient (IC) between predictions and actual returns

        IC measures the correlation between forecasted and actual returns,
        indicating the predictive power of the strategy.

        Args:
            predictions: Predicted returns or signals
            actual_returns: Actual market returns
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Dictionary with IC statistics
        """
        try:
            # Align series
            aligned_pred, aligned_actual = predictions.align(actual_returns, join='inner')

            if len(aligned_pred) == 0:
                return {'information_coefficient': 0.0, 'error': 'No aligned data'}

            # Calculate correlation
            if method == 'pearson':
                ic, p_value = stats.pearsonr(aligned_pred, aligned_actual)
            elif method == 'spearman':
                ic, p_value = stats.spearmanr(aligned_pred, aligned_actual)
            elif method == 'kendall':
                ic, p_value = stats.kendalltau(aligned_pred, aligned_actual)
            else:
                raise ValueError(f"Unknown correlation method: {method}")

            # Calculate IC statistics
            ic_abs = abs(ic)
            ic_squared = ic**2

            # Rolling IC for stability assessment
            window = min(252, len(aligned_pred) // 4)  # Quarterly windows
            if window >= 20:
                rolling_ic = aligned_pred.rolling(window).corr(aligned_actual).dropna()
                ic_stability = 1 - rolling_ic.std() / max(abs(rolling_ic.mean()), 0.01)
            else:
                ic_stability = 0.0

            # IC quality assessment
            if ic_abs > 0.1:
                quality = "Excellent"
            elif ic_abs > 0.05:
                quality = "Good"
            elif ic_abs > 0.02:
                quality = "Fair"
            else:
                quality = "Poor"

            return {
                'information_coefficient': ic,
                'ic_absolute': ic_abs,
                'ic_squared': ic_squared,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'ic_stability': ic_stability,
                'quality_assessment': quality,
                'method': method,
                'sample_size': len(aligned_pred)
            }

        except Exception as e:
            logger.error(f"IC calculation failed: {e}")
            return {'information_coefficient': 0.0, 'error': str(e)}

    def bootstrap_confidence_intervals(self,
                                     returns: pd.Series,
                                     metric_func: callable,
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for any metric

        Args:
            returns: Return series
            metric_func: Function to calculate metric (e.g., Sharpe ratio)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with confidence interval bounds and statistics
        """
        try:
            bootstrap_values = []

            for _ in range(n_bootstrap):
                # Bootstrap sample
                bootstrap_sample = returns.sample(n=len(returns), replace=True)
                metric_value = metric_func(bootstrap_sample)
                bootstrap_values.append(metric_value)

            bootstrap_values = np.array(bootstrap_values)

            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_values, lower_percentile)
            ci_upper = np.percentile(bootstrap_values, upper_percentile)

            # Calculate statistics
            mean_value = np.mean(bootstrap_values)
            std_value = np.std(bootstrap_values)
            median_value = np.median(bootstrap_values)

            return {
                'mean': mean_value,
                'median': median_value,
                'std': std_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'confidence_level': confidence_level,
                'n_bootstrap': n_bootstrap,
                'ci_width': ci_upper - ci_lower
            }

        except Exception as e:
            logger.error(f"Bootstrap CI calculation failed: {e}")
            return {'error': str(e)}

    def bayesian_sharpe_estimation(self,
                                  returns: pd.Series,
                                  prior_mean: float = 0.0,
                                  prior_precision: float = 1.0) -> Dict[str, float]:
        """
        Bayesian estimation of Sharpe ratio with uncertainty quantification

        Uses Bayesian inference to estimate Sharpe ratio with credible intervals,
        incorporating prior beliefs about strategy performance.

        Args:
            returns: Strategy returns
            prior_mean: Prior mean for Sharpe ratio
            prior_precision: Prior precision (1/variance)

        Returns:
            Dictionary with Bayesian Sharpe estimates
        """
        try:
            if len(returns) == 0:
                return {'posterior_mean': 0.0, 'error': 'No returns data'}

            # Calculate sample statistics
            n = len(returns)
            sample_mean = returns.mean() * 252  # Annualized
            sample_var = returns.var() * 252   # Annualized
            sample_sr = sample_mean / np.sqrt(sample_var) if sample_var > 0 else 0

            # Bayesian update (assuming normal-gamma conjugate prior)
            # Posterior precision
            posterior_precision = prior_precision + n

            # Posterior mean
            posterior_mean = (prior_precision * prior_mean + n * sample_sr) / posterior_precision

            # Posterior variance (approximate)
            posterior_var = 1 / posterior_precision

            # Credible intervals
            alpha = 1 - self.confidence_level
            t_critical = t.ppf(1 - alpha/2, df=n-1)

            ci_lower = posterior_mean - t_critical * np.sqrt(posterior_var)
            ci_upper = posterior_mean + t_critical * np.sqrt(posterior_var)

            # Probability that Sharpe > 0
            prob_positive = 1 - norm.cdf(0, loc=posterior_mean, scale=np.sqrt(posterior_var))

            return {
                'posterior_mean': posterior_mean,
                'posterior_variance': posterior_var,
                'credible_interval_lower': ci_lower,
                'credible_interval_upper': ci_upper,
                'probability_positive': prob_positive,
                'sample_sharpe': sample_sr,
                'prior_mean': prior_mean,
                'prior_precision': prior_precision,
                'sample_size': n
            }

        except Exception as e:
            logger.error(f"Bayesian Sharpe estimation failed: {e}")
            return {'posterior_mean': 0.0, 'error': str(e)}

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))

    def _calculate_sr_confidence_interval(self,
                                        returns: pd.Series,
                                        confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for Sharpe ratio"""
        n = len(returns)
        sr = self._calculate_sharpe_ratio(returns)

        # Standard error of Sharpe ratio
        sr_se = np.sqrt((1 + sr**2/2) / (n-1))

        # Critical value
        alpha = 1 - confidence_level
        z_critical = norm.ppf(1 - alpha/2)

        # Confidence interval
        ci_lower = sr - z_critical * sr_se
        ci_upper = sr + z_critical * sr_se

        return ci_lower, ci_upper

    def generate_probabilistic_report(self,
                                    returns: pd.Series,
                                    n_trials: int = 1) -> str:
        """Generate comprehensive probabilistic metrics report"""

        psr_result = self.calculate_probabilistic_sharpe(returns)
        dsr_result = self.calculate_deflated_sharpe(returns, n_trials)

        # Bootstrap confidence intervals for Sharpe ratio
        sharpe_bootstrap = self.bootstrap_confidence_intervals(
            returns, self._calculate_sharpe_ratio
        )

        # Bayesian estimation
        bayesian_result = self.bayesian_sharpe_estimation(returns)

        report = f"""
📊 PROBABILISTIC METRICS REPORT
{'=' * 50}

PROBABILISTIC SHARPE RATIO:
- PSR: {psr_result.get('probabilistic_sharpe', 0):.3f}
- Sample Sharpe: {psr_result.get('sample_sharpe', 0):.3f}
- 95% CI: [{psr_result.get('confidence_interval_lower', 0):.3f}, {psr_result.get('confidence_interval_upper', 0):.3f}]

DEFLATED SHARPE RATIO:
- DSR: {dsr_result.get('deflated_sharpe', 0):.3f}
- P-value: {dsr_result.get('p_value', 1):.4f}
- Significant: {dsr_result.get('significant', False)}

BOOTSTRAP ANALYSIS:
- Mean Sharpe: {sharpe_bootstrap.get('mean', 0):.3f}
- 95% CI: [{sharpe_bootstrap.get('ci_lower', 0):.3f}, {sharpe_bootstrap.get('ci_upper', 0):.3f}]
- CI Width: {sharpe_bootstrap.get('ci_width', 0):.3f}

BAYESIAN ESTIMATION:
- Posterior Mean: {bayesian_result.get('posterior_mean', 0):.3f}
- Credible Interval: [{bayesian_result.get('credible_interval_lower', 0):.3f}, {bayesian_result.get('credible_interval_upper', 0):.3f}]
- P(Sharpe > 0): {bayesian_result.get('probability_positive', 0):.3f}

ASSESSMENT:
"""

        # Add assessment
        psr = psr_result.get('probabilistic_sharpe', 0)
        if psr > 0.95:
            report += "✅ EXCELLENT: Very high confidence in positive performance\n"
        elif psr > 0.8:
            report += "✅ GOOD: High confidence in positive performance\n"
        elif psr > 0.6:
            report += "⚠️ MODERATE: Some confidence in positive performance\n"
        else:
            report += "❌ LOW: Low confidence in positive performance\n"

        return report