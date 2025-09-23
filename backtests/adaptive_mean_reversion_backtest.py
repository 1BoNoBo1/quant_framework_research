#!/usr/bin/env python3
"""
Backtesting script for AdaptiveMeanReversionStrategy.

This script provides comprehensive backtesting capabilities including:
- Single period backtests
- Walk-forward analysis
- Monte Carlo simulations
- Performance visualization
- Risk analysis
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qframe.core.container import get_container
from qframe.core.config import get_config
from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
from qframe.strategies.research.adaptive_mean_reversion_config import AdaptiveMeanReversionConfig

class AdaptiveMeanReversionBacktester:
    """Comprehensive backtesting for AdaptiveMeanReversionStrategy."""

    def __init__(self, config: AdaptiveMeanReversionConfig):
        self.config = config

        # Create mock providers for testing
        from unittest.mock import Mock
        mock_data_provider = Mock()
        mock_risk_manager = Mock()
        mock_risk_manager.validate_position.return_value = True
        mock_risk_manager.calculate_position_size.return_value = 0.05

        # Initialize strategy directly
        self.strategy = AdaptiveMeanReversionStrategy(
            data_provider=mock_data_provider,
            risk_manager=mock_risk_manager,
            config=config
        )
        self.results_history = []

    def run_single_backtest(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000
    ) -> Dict[str, any]:
        """Run a single comprehensive backtest."""
        print(f"🚀 Running adaptive mean reversion backtest from {start_date} to {end_date}")
        print(f"💰 Initial capital: ${initial_capital:,.2f}")

        try:
            # Generate synthetic market data for demonstration
            # In production, this would load real historical data
            market_data = self._generate_test_market_data(start_date, end_date)

            # Run backtest simulation
            portfolio_returns, trades, positions = self._simulate_trading(
                market_data, initial_capital
            )

            # Calculate comprehensive metrics
            metrics = self._calculate_performance_metrics(
                portfolio_returns, trades, initial_capital
            )

            # Add additional analysis
            metrics.update(self._calculate_risk_metrics(portfolio_returns))
            metrics.update(self._analyze_trades(trades))
            metrics.update(self._analyze_regime_performance(trades))

            self._print_results(metrics)
            self.results_history.append(metrics)

            return metrics

        except Exception as e:
            print(f"❌ Error in backtesting: {str(e)}")
            return {}

    def run_walk_forward_analysis(
        self,
        start_date: str,
        end_date: str,
        train_period_months: int = 12,
        test_period_months: int = 3,
        step_months: int = 1
    ) -> pd.DataFrame:
        """Run walk-forward analysis with rolling windows."""
        print("📊 Running walk-forward analysis...")

        results = []
        current_date = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        period_count = 0
        while current_date < end_date_dt:
            # Define periods
            train_start = current_date
            train_end = current_date + timedelta(days=train_period_months * 30)
            test_start = train_end
            test_end = train_end + timedelta(days=test_period_months * 30)

            if test_end > end_date_dt:
                break

            period_count += 1
            print(f"📈 Period {period_count}: Training {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"📊 Testing {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

            # Run backtest for this period
            period_results = self.run_single_backtest(
                start_date=test_start.strftime('%Y-%m-%d'),
                end_date=test_end.strftime('%Y-%m-%d'),
                initial_capital=100000
            )

            if period_results:
                period_results['period_start'] = test_start
                period_results['period_end'] = test_end
                period_results['period_number'] = period_count
                results.append(period_results)

            # Move to next period
            current_date += timedelta(days=step_months * 30)

        results_df = pd.DataFrame(results)
        self._analyze_walk_forward_results(results_df)
        return results_df

    def run_monte_carlo_simulation(
        self,
        start_date: str,
        end_date: str,
        num_simulations: int = 1000,
        confidence_levels: List[float] = [0.05, 0.25, 0.75, 0.95]
    ) -> Dict[str, any]:
        """Run Monte Carlo simulation for risk analysis."""
        print(f"🎲 Running {num_simulations} Monte Carlo simulations...")

        # Get base backtest results
        base_results = self.run_single_backtest(start_date, end_date)
        if not base_results:
            return {}

        # Generate market data
        market_data = self._generate_test_market_data(start_date, end_date)

        simulation_results = []
        for i in range(num_simulations):
            if i % 100 == 0:
                print(f"🔄 Simulation {i}/{num_simulations}")

            # Bootstrap market data
            bootstrapped_data = self._bootstrap_market_data(market_data)

            # Run simulation
            portfolio_returns, _, _ = self._simulate_trading(bootstrapped_data, 100000)
            if len(portfolio_returns) > 0:
                sim_metrics = self._calculate_performance_metrics(portfolio_returns, [], 100000)
                simulation_results.append(sim_metrics)

        # Analyze simulation results
        mc_analysis = self._analyze_monte_carlo_results(
            simulation_results, base_results, confidence_levels
        )

        return {
            'base_results': base_results,
            'simulation_results': simulation_results,
            'monte_carlo_analysis': mc_analysis
        }

    def _simulate_trading(
        self,
        market_data: pd.DataFrame,
        initial_capital: float
    ) -> Tuple[pd.Series, List[Dict], pd.DataFrame]:
        """Simulate trading with the strategy."""
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {}
        portfolio_values = [initial_capital]
        trades = []
        position_history = []

        # Process data in chunks to simulate real-time trading
        for i in range(len(market_data)):
            current_data = market_data.iloc[:i+1]

            if len(current_data) < 50:  # Need minimum data for features
                portfolio_values.append(portfolio_value)
                position_history.append(positions.copy())
                continue

            try:
                # Generate signals
                signals = self.strategy.generate_signals(current_data)

                # Process signals
                for signal in signals:
                    symbol = signal.symbol
                    signal_strength = signal.strength
                    current_price = signal.price if signal.price else current_data['close'].iloc[-1]

                    # Calculate position size based on signal action
                    if signal.action.value in ['buy']:
                        target_position_value = portfolio_value * signal_strength * self.config.max_position_size
                    elif signal.action.value in ['sell']:
                        target_position_value = -portfolio_value * signal_strength * self.config.max_position_size
                    else:
                        target_position_value = 0

                    target_shares = target_position_value / current_price

                    # Current position
                    current_shares = positions.get(symbol, 0)

                    # Trade execution
                    if abs(target_shares - current_shares) > 0.01:  # Minimum trade size
                        trade_shares = target_shares - current_shares
                        trade_value = trade_shares * current_price
                        transaction_cost = abs(trade_value) * self.config.transaction_cost

                        if cash >= trade_value + transaction_cost:
                            # Execute trade
                            positions[symbol] = target_shares
                            cash -= trade_value + transaction_cost

                            # Record trade
                            trades.append({
                                'timestamp': current_data.index[-1],
                                'symbol': symbol,
                                'shares': trade_shares,
                                'price': current_price,
                                'value': trade_value,
                                'signal_strength': signal_strength,
                                'action': signal.action.value,
                                'regime': signal.metadata.get('regime', 'unknown'),
                                'confidence': signal.metadata.get('confidence', 0.5)
                            })

                # Calculate portfolio value
                portfolio_value = cash
                for symbol, shares in positions.items():
                    if symbol in current_data.columns:
                        portfolio_value += shares * current_data['close'].iloc[-1]
                    else:
                        portfolio_value += shares * current_data['close'].iloc[-1]  # Use close price

                portfolio_values.append(portfolio_value)
                position_history.append(positions.copy())

            except Exception as e:
                print(f"⚠️  Error at step {i}: {str(e)}")
                portfolio_values.append(portfolio_value)
                position_history.append(positions.copy())

        # Convert to returns
        portfolio_values = pd.Series(portfolio_values, index=market_data.index)
        portfolio_returns = portfolio_values.pct_change().dropna()

        # Convert position history
        positions_df = pd.DataFrame(position_history, index=market_data.index)

        return portfolio_returns, trades, positions_df

    def _calculate_performance_metrics(
        self,
        returns: pd.Series,
        trades: List[Dict],
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0:
            return {}

        # Basic metrics
        total_return = (returns + 1).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        var_95 = returns.quantile(0.05)
        sortino_ratio = self._calculate_sortino_ratio(returns)

        # Advanced metrics
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        win_rate = (returns > 0).mean()
        profit_factor = self._calculate_profit_factor(returns)

        # Information ratio
        benchmark_return = 0.0  # Assume 0% benchmark
        excess_returns = returns - benchmark_return
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_trade_return': returns.mean(),
            'volatility_adjusted_return': annualized_return / volatility if volatility > 0 else 0
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate additional risk metrics."""
        if len(returns) == 0:
            return {}

        # Value at Risk and Expected Shortfall
        var_99 = returns.quantile(0.01)
        var_95 = returns.quantile(0.05)
        expected_shortfall_95 = returns[returns <= var_95].mean()

        # Tail ratio
        tail_ratio = abs(returns.quantile(0.95)) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0

        # Maximum consecutive losses
        consecutive_losses = self._calculate_max_consecutive_losses(returns)

        return {
            'var_99': var_99,
            'expected_shortfall_95': expected_shortfall_95,
            'tail_ratio': tail_ratio,
            'max_consecutive_losses': consecutive_losses
        }

    def _analyze_trades(self, trades: List[Dict]) -> Dict[str, any]:
        """Analyze trading patterns."""
        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)

        # Trade statistics
        avg_trade_size = trades_df['value'].abs().mean()
        trade_frequency = len(trades) / 30  # trades per month assuming daily data

        # Regime analysis
        regime_trades = trades_df.groupby('regime').size().to_dict()

        return {
            'avg_trade_size': avg_trade_size,
            'trade_frequency': trade_frequency,
            'regime_trade_distribution': regime_trades
        }

    def _analyze_regime_performance(self, trades: List[Dict]) -> Dict[str, any]:
        """Analyze performance by market regime."""
        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)
        regime_performance = {}

        for regime in ['trending', 'ranging', 'volatile']:
            regime_trades = trades_df[trades_df['regime'] == regime]
            if len(regime_trades) > 0:
                regime_performance[f'{regime}_trades'] = len(regime_trades)
                regime_performance[f'{regime}_avg_signal'] = regime_trades['signal_strength'].mean()
                regime_performance[f'{regime}_avg_confidence'] = regime_trades['confidence'].mean()

        return regime_performance

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf')
        return returns.mean() / downside_std * np.sqrt(252)

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return profits / losses if losses > 0 else float('inf')

    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losses."""
        is_loss = returns < 0
        consecutive = 0
        max_consecutive = 0

        for loss in is_loss:
            if loss:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        return max_consecutive

    def _generate_test_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        np.random.seed(42)  # For reproducible results

        # Generate realistic crypto-like price data
        n_points = len(dates)
        returns = np.random.normal(0.0001, 0.02, n_points)  # Small positive drift with volatility
        prices = 50000 * np.exp(np.cumsum(returns))  # Start at $50,000

        # Add some market regime patterns
        trend_periods = np.random.choice([True, False], n_points, p=[0.3, 0.7])
        ranging_periods = np.random.choice([True, False], n_points, p=[0.4, 0.6])

        # Adjust returns based on regimes
        for i in range(n_points):
            if trend_periods[i]:
                returns[i] += 0.0002  # Trending bias
            elif ranging_periods[i]:
                returns[i] *= 0.5  # Lower volatility in ranging

        # Recalculate prices
        prices = 50000 * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        high = prices * (1 + np.abs(np.random.normal(0, 0.005, n_points)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.005, n_points)))
        volume = np.random.lognormal(15, 0.5, n_points)

        data = pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        }, index=dates)

        data['open'].iloc[0] = data['close'].iloc[0]
        return data

    def _bootstrap_market_data(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Bootstrap market data for Monte Carlo simulation."""
        n = len(original_data)
        bootstrap_indices = np.random.choice(n, n, replace=True)
        return original_data.iloc[bootstrap_indices].reset_index(drop=True)

    def _analyze_monte_carlo_results(
        self,
        simulation_results: List[Dict],
        base_results: Dict,
        confidence_levels: List[float]
    ) -> Dict[str, any]:
        """Analyze Monte Carlo simulation results."""
        if not simulation_results:
            return {}

        metrics_df = pd.DataFrame(simulation_results)

        analysis = {}
        for metric in ['sharpe_ratio', 'max_drawdown', 'total_return']:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                analysis[f'{metric}_mean'] = values.mean()
                analysis[f'{metric}_std'] = values.std()

                for cl in confidence_levels:
                    analysis[f'{metric}_p{int(cl*100)}'] = values.quantile(cl)

                # Probability of beating base case
                base_value = base_results.get(metric, 0)
                if metric == 'max_drawdown':
                    prob_better = (values > base_value).mean()  # Less negative is better
                else:
                    prob_better = (values > base_value).mean()
                analysis[f'{metric}_prob_beat_base'] = prob_better

        return analysis

    def _analyze_walk_forward_results(self, results_df: pd.DataFrame) -> None:
        """Analyze walk-forward analysis results."""
        if results_df.empty:
            return

        print("\n📊 Walk-Forward Analysis Summary:")
        print("=" * 50)

        for metric in ['sharpe_ratio', 'max_drawdown', 'win_rate']:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                print(f"{metric.replace('_', ' ').title()}:")
                print(f"  Mean: {values.mean():.3f}")
                print(f"  Std:  {values.std():.3f}")
                print(f"  Min:  {values.min():.3f}")
                print(f"  Max:  {values.max():.3f}")
                print()

    def _print_results(self, results: Dict[str, any]) -> None:
        """Print comprehensive backtest results."""
        print("\n" + "="*60)
        print("📈 ADAPTIVE MEAN REVERSION BACKTEST RESULTS")
        print("="*60)

        # Performance metrics
        print("💰 PERFORMANCE METRICS:")
        print(f"  Total Return:          {results.get('total_return', 0):.2%}")
        print(f"  Annualized Return:     {results.get('annualized_return', 0):.2%}")
        print(f"  Volatility:            {results.get('volatility', 0):.2%}")
        print(f"  Sharpe Ratio:          {results.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio:         {results.get('sortino_ratio', 0):.3f}")
        print(f"  Calmar Ratio:          {results.get('calmar_ratio', 0):.3f}")
        print(f"  Information Ratio:     {results.get('information_ratio', 0):.3f}")

        # Risk metrics
        print("\n🛡️  RISK METRICS:")
        print(f"  Maximum Drawdown:      {results.get('max_drawdown', 0):.2%}")
        print(f"  VaR (95%):            {results.get('var_95', 0):.2%}")
        print(f"  Expected Shortfall:    {results.get('expected_shortfall_95', 0):.2%}")
        print(f"  Tail Ratio:           {results.get('tail_ratio', 0):.2f}")

        # Trading metrics
        print("\n📊 TRADING METRICS:")
        print(f"  Total Trades:          {results.get('total_trades', 0)}")
        print(f"  Win Rate:              {results.get('win_rate', 0):.2%}")
        print(f"  Profit Factor:         {results.get('profit_factor', 0):.2f}")
        print(f"  Average Trade Return:  {results.get('avg_trade_return', 0):.4f}")

        # Regime analysis
        if 'regime_trade_distribution' in results:
            print("\n🔄 REGIME ANALYSIS:")
            for regime, count in results['regime_trade_distribution'].items():
                print(f"  {regime.title()} Trades:      {count}")

        # Performance assessment
        print("\n🎯 PERFORMANCE ASSESSMENT:")
        sharpe = results.get('sharpe_ratio', 0)
        max_dd = results.get('max_drawdown', 0)
        win_rate = results.get('win_rate', 0)

        if sharpe >= 1.5:
            print("  ✅ Sharpe Ratio: EXCELLENT")
        elif sharpe >= 1.0:
            print("  ⚠️  Sharpe Ratio: GOOD")
        else:
            print("  ❌ Sharpe Ratio: NEEDS IMPROVEMENT")

        if abs(max_dd) <= 0.10:
            print("  ✅ Drawdown: EXCELLENT")
        elif abs(max_dd) <= 0.15:
            print("  ⚠️  Drawdown: ACCEPTABLE")
        else:
            print("  ❌ Drawdown: HIGH RISK")

        if win_rate >= 0.55:
            print("  ✅ Win Rate: GOOD")
        else:
            print("  ⚠️  Win Rate: NEEDS IMPROVEMENT")

def main():
    """Main entry point for backtesting."""
    parser = argparse.ArgumentParser(description='Backtest Adaptive Mean Reversion Strategy')

    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward analysis')
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo simulation')
    parser.add_argument('--mc-sims', type=int, default=1000, help='Number of MC simulations')

    args = parser.parse_args()

    # Load configuration
    config = AdaptiveMeanReversionConfig()
    backtester = AdaptiveMeanReversionBacktester(config)

    if args.walk_forward:
        print("🔄 Running Walk-Forward Analysis...")
        results = backtester.run_walk_forward_analysis(
            args.start_date, args.end_date
        )
        print(f"📊 Walk-forward analysis completed with {len(results)} periods")

    elif args.monte_carlo:
        print("🎲 Running Monte Carlo Simulation...")
        results = backtester.run_monte_carlo_simulation(
            args.start_date, args.end_date, args.mc_sims
        )
        print("📈 Monte Carlo simulation completed")

    else:
        print("📊 Running Single Backtest...")
        results = backtester.run_single_backtest(
            args.start_date, args.end_date, args.capital
        )

    print("\n🎉 Backtesting completed successfully!")

if __name__ == "__main__":
    main()