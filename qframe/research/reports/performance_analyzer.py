"""
Performance Analyzer pour QFrame
===============================

Module d'analyse de performance avancée pour les stratégies quantitatives.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """Analyseur de performance pour stratégies quantitatives"""

    def __init__(self):
        """Initialise l'analyseur de performance"""
        self.risk_free_rate = 0.02  # 2% taux sans risque annuel

    def calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcule les métriques de performance de base"""

        if len(returns) == 0:
            return {}

        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0
        }

    def calculate_risk_metrics(self, returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Any]:
        """Calcule les métriques de risque avancées"""

        if len(returns) == 0:
            return {}

        risk_metrics = {}

        # Value at Risk (VaR)
        for confidence in confidence_levels:
            var_value = np.percentile(returns, (1 - confidence) * 100)
            risk_metrics[f'var_{int(confidence*100)}'] = var_value

            # Conditional VaR (CVaR)
            cvar_value = returns[returns <= var_value].mean()
            risk_metrics[f'cvar_{int(confidence*100)}'] = cvar_value

        # Downside deviation (pour Sortino)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Sortino ratio
        annualized_return = self.calculate_basic_metrics(returns)['annualized_return']
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        risk_metrics.update({
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        })

        return risk_metrics

    def calculate_trading_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calcule les métriques de trading"""

        if len(trades_df) == 0:
            return {'total_trades': 0}

        total_trades = len(trades_df)

        # Assuming trades_df has 'pnl' column
        if 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if avg_loss > 0 else 0

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_trade_return': trades_df['pnl'].mean() if 'pnl' in trades_df.columns else 0
            }
        else:
            return {'total_trades': total_trades}

    def calculate_rolling_metrics(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """Calcule les métriques roulantes"""

        if len(returns) < window:
            return pd.DataFrame()

        rolling_returns = returns.rolling(window)

        rolling_metrics = pd.DataFrame(index=returns.index)
        rolling_metrics['rolling_return'] = rolling_returns.sum()
        rolling_metrics['rolling_volatility'] = rolling_returns.std() * np.sqrt(252)
        rolling_metrics['rolling_sharpe'] = (rolling_returns.mean() * 252 - self.risk_free_rate) / (rolling_returns.std() * np.sqrt(252))

        # Rolling drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_metrics['rolling_drawdown'] = (cumulative - rolling_max) / rolling_max

        return rolling_metrics.dropna()

    def analyze_comprehensive(self, returns: pd.Series, trades_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyse complète de performance"""

        analysis = {}

        # Métriques de base
        analysis['basic_metrics'] = self.calculate_basic_metrics(returns)

        # Métriques de risque
        analysis['risk_metrics'] = self.calculate_risk_metrics(returns)

        # Métriques de trading
        if trades_df is not None:
            analysis['trading_metrics'] = self.calculate_trading_metrics(trades_df)

        # Métriques roulantes (sample sur les 252 derniers points si disponibles)
        if len(returns) >= 252:
            analysis['rolling_metrics'] = self.calculate_rolling_metrics(returns).tail(1).to_dict('records')[0] if not self.calculate_rolling_metrics(returns).empty else {}

        # Statistiques descriptives
        analysis['descriptive_stats'] = {
            'mean': returns.mean(),
            'median': returns.median(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }

        # Période d'analyse
        analysis['period_info'] = {
            'start_date': returns.index[0] if len(returns) > 0 else None,
            'end_date': returns.index[-1] if len(returns) > 0 else None,
            'total_periods': len(returns),
            'analysis_date': datetime.now()
        }

        return analysis

    def compare_strategies(self, strategies_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """Compare plusieurs stratégies"""

        if not strategies_returns:
            return pd.DataFrame()

        comparison_data = {}

        for strategy_name, returns in strategies_returns.items():
            metrics = self.calculate_basic_metrics(returns)
            risk_metrics = self.calculate_risk_metrics(returns)

            comparison_data[strategy_name] = {
                **metrics,
                **risk_metrics
            }

        comparison_df = pd.DataFrame(comparison_data).T

        # Ajout de ranking
        for metric in ['total_return', 'sharpe_ratio', 'sortino_ratio']:
            if metric in comparison_df.columns:
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)

        return comparison_df

    def generate_performance_summary(self, returns: pd.Series, strategy_name: str = "Strategy") -> str:
        """Génère un résumé textuel de performance"""

        analysis = self.analyze_comprehensive(returns)
        basic = analysis.get('basic_metrics', {})
        risk = analysis.get('risk_metrics', {})

        summary = f"""
Performance Summary - {strategy_name}
{'=' * (20 + len(strategy_name))}

📊 Performance Metrics:
• Total Return: {basic.get('total_return', 0):.2%}
• Annualized Return: {basic.get('annualized_return', 0):.2%}
• Volatility: {basic.get('volatility', 0):.2%}
• Sharpe Ratio: {basic.get('sharpe_ratio', 0):.3f}

⚠️ Risk Metrics:
• Max Drawdown: {basic.get('max_drawdown', 0):.2%}
• Sortino Ratio: {risk.get('sortino_ratio', 0):.3f}
• VaR 95%: {risk.get('var_95', 0):.2%}
• CVaR 95%: {risk.get('cvar_95', 0):.2%}

📈 Statistical Properties:
• Skewness: {risk.get('skewness', 0):.3f}
• Kurtosis: {risk.get('kurtosis', 0):.3f}
"""

        return summary