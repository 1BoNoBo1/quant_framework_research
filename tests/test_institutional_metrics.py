#!/usr/bin/env python
"""
🧪 Tests pour le service de métriques institutionnelles

Tests complets pour InstitutionalMetricsService avec toutes les métriques
avancées pour hedge funds et institutions financières.
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from qframe.domain.services.institutional_metrics_service import InstitutionalMetricsService
from qframe.domain.entities.backtest import BacktestResult, BacktestMetrics, BacktestStatus
from qframe.domain.entities.order import Order, OrderSide, OrderType


@pytest.mark.unit
@pytest.mark.critical
class TestInstitutionalMetricsService:
    """Tests for institutional metrics calculations"""

    @pytest.fixture
    def service(self):
        """Create service instance"""
        return InstitutionalMetricsService()

    @pytest.fixture
    def sample_returns_data(self):
        """Generate sample returns data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=252, freq='B')

        # Strategy returns with some predictability
        strategy_returns = np.random.normal(0.0005, 0.015, 252)
        benchmark_returns = np.random.normal(0.0003, 0.012, 252)

        # Predictions with some correlation to returns
        predictions = []
        for i in range(len(strategy_returns)):
            if i >= 5:
                recent_perf = np.mean(strategy_returns[i-5:i])
                prediction = recent_perf * 1.5 + np.random.normal(0, 0.008)
            else:
                prediction = np.random.normal(0, 0.008)
            predictions.append(prediction)

        return {
            'strategy_returns': pd.Series(strategy_returns, index=dates),
            'benchmark_returns': pd.Series(benchmark_returns, index=dates),
            'predictions': pd.Series(predictions, index=dates)
        }

    @pytest.fixture
    def sample_backtest_result(self, sample_returns_data):
        """Create sample backtest result"""
        returns = sample_returns_data['strategy_returns']
        benchmark_returns = sample_returns_data['benchmark_returns']

        # Calculate basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0

        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        metrics = BacktestMetrics(
            total_return=Decimal(str(total_return)),
            annualized_return=Decimal(str(annual_return)),
            volatility=Decimal(str(volatility)),
            sharpe_ratio=Decimal(str(sharpe)),
            max_drawdown=Decimal(str(max_drawdown)),
            total_trades=25,
            winning_trades=15,
            losing_trades=10,
            win_rate=Decimal("0.6")
        )

        # Generate mock trades
        trades = []
        for i in range(0, len(returns), 10):
            trade = Order(
                id=f"trade-{i}",
                portfolio_id="test-portfolio",
                symbol="BTC/USD",
                side=OrderSide.BUY if i % 20 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                price=Decimal(str(50000 + i * 10)),
                created_time=returns.index[i]
            )
            trades.append(trade)

        portfolio_values = 100000 * cumulative

        return BacktestResult(
            id="test-backtest",
            name="Test Backtest",
            status=BacktestStatus.COMPLETED,
            initial_capital=Decimal("100000"),
            final_capital=Decimal(str(100000 * (1 + total_return))),
            metrics=metrics,
            returns=returns,
            portfolio_values=portfolio_values,
            trades=trades,
            benchmark_returns=benchmark_returns,
            start_time=returns.index[0],
            end_time=returns.index[-1]
        )

    def test_information_coefficient_calculation(self, service, sample_returns_data):
        """Test Information Coefficient calculation"""
        predictions = sample_returns_data['predictions']
        strategy_returns = sample_returns_data['strategy_returns']
        benchmark_returns = sample_returns_data['benchmark_returns']

        result = service.calculate_information_metrics(
            predictions, strategy_returns, benchmark_returns
        )

        # Verify structure
        assert hasattr(result, 'information_coefficient')
        assert hasattr(result, 'information_ratio')
        assert hasattr(result, 'ic_ir')
        assert hasattr(result, 'hit_rate')
        assert hasattr(result, 'predictive_power')

        # Verify values are reasonable
        assert -1 <= result.information_coefficient <= 1
        assert 0 <= result.hit_rate <= 1
        assert result.predictive_power in ['EXCELLENT', 'GOOD', 'MODERATE', 'POOR']

    def test_information_ratio_calculation(self, service, sample_returns_data):
        """Test Information Ratio calculation"""
        predictions = sample_returns_data['predictions']
        strategy_returns = sample_returns_data['strategy_returns']
        benchmark_returns = sample_returns_data['benchmark_returns']

        result = service.calculate_information_metrics(
            predictions, strategy_returns, benchmark_returns
        )

        # Information ratio should be meaningful
        assert isinstance(result.information_ratio, float)
        assert -5 <= result.information_ratio <= 5  # Reasonable range

    @pytest.mark.slow
    def test_excursion_metrics_calculation(self, service, sample_backtest_result):
        """Test MAE/MFE excursion metrics"""
        # Create mock price data
        price_data = pd.DataFrame({
            'close': np.random.randn(len(sample_backtest_result.returns)) * 0.02 + 1
        }).cumsum() * 50000
        price_data.index = sample_backtest_result.returns.index

        result = service.calculate_excursion_metrics(
            sample_backtest_result.trades, price_data
        )

        # Verify structure
        assert hasattr(result, 'mae_average')
        assert hasattr(result, 'mae_maximum')
        assert hasattr(result, 'mfe_average')
        assert hasattr(result, 'mfe_maximum')
        assert hasattr(result, 'efficiency_ratio')

        # Verify values are reasonable
        assert 0 <= result.mae_average <= 1
        assert 0 <= result.mfe_average <= 1
        assert result.mae_maximum >= result.mae_average
        assert result.mfe_maximum >= result.mfe_average

    def test_attribution_metrics_calculation(self, service, sample_returns_data):
        """Test performance attribution metrics"""
        strategy_returns = sample_returns_data['strategy_returns']
        benchmark_returns = sample_returns_data['benchmark_returns']

        result = service.calculate_attribution_metrics(
            strategy_returns, benchmark_returns
        )

        # Verify structure
        assert hasattr(result, 'stock_selection')
        assert hasattr(result, 'market_timing')
        assert hasattr(result, 'interaction_effect')
        assert hasattr(result, 'total_active_return')
        assert hasattr(result, 'tracking_error')
        assert hasattr(result, 'attribution_quality')

        # Verify mathematical consistency
        total_calculated = result.stock_selection + result.market_timing + result.interaction_effect
        assert abs(total_calculated - result.total_active_return) < 0.001

    def test_risk_adjusted_metrics_calculation(self, service, sample_returns_data):
        """Test advanced risk-adjusted metrics"""
        strategy_returns = sample_returns_data['strategy_returns']
        benchmark_returns = sample_returns_data['benchmark_returns']

        result = service.calculate_risk_adjusted_metrics(
            strategy_returns, benchmark_returns
        )

        # Verify structure
        expected_metrics = [
            'calmar_ratio', 'sterling_ratio', 'burke_ratio',
            'pain_index', 'ulcer_index', 'martin_ratio',
            'kappa_three', 'omega_ratio'
        ]

        for metric in expected_metrics:
            assert hasattr(result, metric)
            assert isinstance(getattr(result, metric), (int, float))

    @pytest.mark.performance
    def test_comprehensive_report_generation(self, service, sample_backtest_result, sample_returns_data):
        """Test comprehensive report generation performance"""
        predictions = sample_returns_data['predictions']
        benchmark_returns = sample_returns_data['benchmark_returns']

        import time
        start_time = time.time()

        report = service.generate_comprehensive_report(
            sample_backtest_result, predictions, benchmark_returns
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete reasonably quickly
        assert execution_time < 5.0  # Less than 5 seconds

        # Report should contain all sections
        assert "INFORMATION METRICS:" in report
        assert "EXCURSION ANALYSIS:" in report
        assert "PERFORMANCE ATTRIBUTION:" in report
        assert "ADVANCED RISK METRICS:" in report

    def test_edge_cases_empty_data(self, service):
        """Test behavior with empty or invalid data"""
        empty_series = pd.Series([], dtype=float)

        # Service should handle empty data gracefully
        result = service.calculate_information_metrics(
            empty_series, empty_series, empty_series
        )

        # Should return reasonable defaults for empty data
        assert result.information_coefficient == 0.0
        assert result.hit_rate == 0.0  # No data hit rate
        assert result.predictive_power == 'NO_DATA'

    def test_edge_cases_constant_data(self, service):
        """Test behavior with constant data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        constant_series = pd.Series([0.01] * 100, index=dates)

        result = service.calculate_information_metrics(
            constant_series, constant_series, constant_series
        )

        # Should handle constant data gracefully
        assert isinstance(result.information_coefficient, (int, float))

    def test_numerical_precision(self, service, sample_returns_data):
        """Test numerical precision and stability"""
        predictions = sample_returns_data['predictions']
        strategy_returns = sample_returns_data['strategy_returns']
        benchmark_returns = sample_returns_data['benchmark_returns']

        # Run calculation multiple times
        results = []
        for _ in range(5):
            result = service.calculate_information_metrics(
                predictions, strategy_returns, benchmark_returns
            )
            results.append(result.information_coefficient)

        # Results should be identical (deterministic)
        assert all(abs(r - results[0]) < 1e-10 for r in results)


@pytest.mark.integration
@pytest.mark.critical
class TestInstitutionalMetricsIntegration:
    """Integration tests for institutional metrics with real workflows"""

    def test_end_to_end_metrics_workflow(self):
        """Test complete metrics workflow from strategy to report"""
        # This would test the complete pipeline from strategy execution
        # to final institutional metrics reporting
        pass

    def test_metrics_with_real_market_data(self):
        """Test metrics calculation with real market data patterns"""
        # This would test with realistic market data characteristics
        pass


if __name__ == "__main__":
    pytest.main([__file__])