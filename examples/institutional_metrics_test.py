#!/usr/bin/env python
"""
📊 Institutional Metrics Test

Demonstrates advanced institutional metrics:
- Information Coefficient (IC) and Information Ratio (IR)
- Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
- Performance attribution analysis
- Advanced risk-adjusted measures
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qframe.domain.services.institutional_metrics_service import InstitutionalMetricsService
from qframe.domain.entities.backtest import BacktestResult, BacktestMetrics, BacktestStatus
from qframe.domain.entities.order import Order, OrderSide, OrderType
import warnings

warnings.filterwarnings('ignore')


def generate_test_data() -> tuple:
    """Generate realistic test data for metrics calculation"""

    # Generate 252 trading days (1 year)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='B')

    # Generate realistic returns with some predictability
    np.random.seed(42)
    market_factor = np.random.normal(0.0005, 0.015, 252)  # Market component
    alpha_component = np.random.normal(0.0002, 0.008, 252)  # Alpha component

    # Strategy returns with some skill
    strategy_returns = market_factor + alpha_component + np.random.normal(0, 0.005, 252)
    strategy_returns = pd.Series(strategy_returns, index=dates)

    # Benchmark returns (market only)
    benchmark_returns = pd.Series(market_factor, index=dates)

    # Generate predictions with some skill
    # Strategy predictions are somewhat correlated with future returns
    predictions = []
    for i in range(len(strategy_returns)):
        # Prediction based on recent performance + noise
        if i >= 5:
            recent_perf = strategy_returns.iloc[i-5:i].mean()
            prediction = recent_perf * 2 + np.random.normal(0, 0.01)
        else:
            prediction = np.random.normal(0, 0.01)
        predictions.append(prediction)

    predictions = pd.Series(predictions, index=dates)

    return strategy_returns, benchmark_returns, predictions, dates


def create_mock_backtest_result(returns: pd.Series, benchmark_returns: pd.Series) -> BacktestResult:
    """Create a mock backtest result for testing"""

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

    # Create metrics
    metrics = BacktestMetrics(
        total_return=Decimal(str(total_return)),
        annualized_return=Decimal(str(annual_return)),
        volatility=Decimal(str(volatility)),
        sharpe_ratio=Decimal(str(sharpe)),
        max_drawdown=Decimal(str(max_drawdown)),
        total_trades=len(returns) // 10,  # Assume some trades
        winning_trades=int(len(returns) // 10 * 0.6),
        losing_trades=int(len(returns) // 10 * 0.4),
        win_rate=Decimal("0.6")
    )

    # Generate some mock trades
    trades = []
    for i in range(0, len(returns), 10):  # Trade every 10 days
        trade = Order(
            id=f"trade-{i}",
            portfolio_id="test-portfolio",
            symbol="BTC/USD",
            side=OrderSide.BUY if i % 20 == 0 else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal(str(50000 + i * 10)),  # Mock price progression
            created_time=returns.index[i]
        )
        trades.append(trade)

    # Portfolio values
    initial_value = 100000
    portfolio_values = initial_value * cumulative

    # Create result
    result = BacktestResult(
        id="test-backtest-metrics",
        name="Institutional Metrics Test",
        status=BacktestStatus.COMPLETED,
        initial_capital=Decimal(str(initial_value)),
        final_capital=Decimal(str(initial_value * (1 + total_return))),
        metrics=metrics,
        returns=returns,
        portfolio_values=portfolio_values,
        trades=trades,
        benchmark_returns=benchmark_returns,
        start_time=returns.index[0],
        end_time=returns.index[-1]
    )

    return result


def test_information_metrics():
    """Test Information Coefficient and Information Ratio calculations"""
    print("🔍 Testing Information Metrics...")

    strategy_returns, benchmark_returns, predictions, dates = generate_test_data()

    service = InstitutionalMetricsService()
    info_metrics = service.calculate_information_metrics(
        predictions, strategy_returns, benchmark_returns
    )

    print(f"  📊 Information Coefficient: {info_metrics.information_coefficient:.4f}")
    print(f"  📊 Information Ratio: {info_metrics.information_ratio:.4f}")
    print(f"  📊 IC Stability (IR): {info_metrics.ic_ir:.4f}")
    print(f"  📊 Hit Rate: {info_metrics.hit_rate:.2%}")
    print(f"  📊 Predictive Power: {info_metrics.predictive_power}")

    return info_metrics


def test_excursion_metrics():
    """Test Maximum Adverse/Favorable Excursion calculations"""
    print("\n📈 Testing Excursion Metrics...")

    strategy_returns, benchmark_returns, predictions, dates = generate_test_data()
    backtest_result = create_mock_backtest_result(strategy_returns, benchmark_returns)

    # Create mock price data
    price_data = pd.DataFrame({
        'close': np.random.randn(len(strategy_returns)) * 0.02 + 1
    }).cumsum() * 50000  # Mock BTC prices
    price_data.index = dates

    service = InstitutionalMetricsService()
    excursion_metrics = service.calculate_excursion_metrics(
        backtest_result.trades, price_data
    )

    print(f"  📊 Average MAE: {excursion_metrics.mae_average:.2%}")
    print(f"  📊 Maximum MAE: {excursion_metrics.mae_maximum:.2%}")
    print(f"  📊 MAE 95th Percentile: {excursion_metrics.mae_percentile_95:.2%}")
    print(f"  📊 Average MFE: {excursion_metrics.mfe_average:.2%}")
    print(f"  📊 Maximum MFE: {excursion_metrics.mfe_maximum:.2%}")
    print(f"  📊 MFE 95th Percentile: {excursion_metrics.mfe_percentile_95:.2%}")
    print(f"  📊 MAE/MFE Ratio: {excursion_metrics.mae_mfe_ratio:.3f}")
    print(f"  📊 Efficiency Ratio: {excursion_metrics.efficiency_ratio:.3f}")

    return excursion_metrics


def test_attribution_metrics():
    """Test Performance Attribution calculations"""
    print("\n🎯 Testing Attribution Metrics...")

    strategy_returns, benchmark_returns, predictions, dates = generate_test_data()

    service = InstitutionalMetricsService()
    attribution_metrics = service.calculate_attribution_metrics(
        strategy_returns, benchmark_returns
    )

    print(f"  📊 Stock Selection: {attribution_metrics.stock_selection:.2%}")
    print(f"  📊 Market Timing: {attribution_metrics.market_timing:.2%}")
    print(f"  📊 Interaction Effect: {attribution_metrics.interaction_effect:.2%}")
    print(f"  📊 Total Active Return: {attribution_metrics.total_active_return:.2%}")
    print(f"  📊 Tracking Error: {attribution_metrics.tracking_error:.2%}")
    print(f"  📊 Attribution Quality: {attribution_metrics.attribution_quality}")

    return attribution_metrics


def test_risk_adjusted_metrics():
    """Test Advanced Risk-Adjusted Metrics"""
    print("\n⚖️ Testing Risk-Adjusted Metrics...")

    strategy_returns, benchmark_returns, predictions, dates = generate_test_data()

    service = InstitutionalMetricsService()
    risk_metrics = service.calculate_risk_adjusted_metrics(
        strategy_returns, benchmark_returns
    )

    print(f"  📊 Calmar Ratio: {risk_metrics.calmar_ratio:.3f}")
    print(f"  📊 Sterling Ratio: {risk_metrics.sterling_ratio:.3f}")
    print(f"  📊 Burke Ratio: {risk_metrics.burke_ratio:.3f}")
    print(f"  📊 Pain Index: {risk_metrics.pain_index:.3f}")
    print(f"  📊 Ulcer Index: {risk_metrics.ulcer_index:.3f}")
    print(f"  📊 Martin Ratio: {risk_metrics.martin_ratio:.3f}")
    print(f"  📊 Kappa Three: {risk_metrics.kappa_three:.3f}")
    print(f"  📊 Omega Ratio: {risk_metrics.omega_ratio:.3f}")

    return risk_metrics


def test_comprehensive_report():
    """Test Comprehensive Institutional Metrics Report"""
    print("\n📋 Testing Comprehensive Report...")

    strategy_returns, benchmark_returns, predictions, dates = generate_test_data()
    backtest_result = create_mock_backtest_result(strategy_returns, benchmark_returns)

    service = InstitutionalMetricsService()
    report = service.generate_comprehensive_report(
        backtest_result, predictions, benchmark_returns
    )

    print(report)

    return report


def main():
    """Main test function"""
    print("📊 QFrame Institutional Metrics Test")
    print("=" * 60)

    try:
        # Test individual metric components
        info_metrics = test_information_metrics()
        excursion_metrics = test_excursion_metrics()
        attribution_metrics = test_attribution_metrics()
        risk_metrics = test_risk_adjusted_metrics()

        # Test comprehensive report
        report = test_comprehensive_report()

        print("\n" + "=" * 60)
        print("✅ INSTITUTIONAL METRICS TEST COMPLETED")
        print("=" * 60)

        print("\n🏆 Summary:")
        print(f"✅ Information Metrics: IC={info_metrics.information_coefficient:.4f}, Hit Rate={info_metrics.hit_rate:.2%}")
        print(f"✅ Excursion Metrics: MAE={excursion_metrics.mae_average:.2%}, MFE={excursion_metrics.mfe_average:.2%}")
        print(f"✅ Attribution Metrics: Active Return={attribution_metrics.total_active_return:.2%}")
        print(f"✅ Risk Metrics: Calmar={risk_metrics.calmar_ratio:.3f}, Omega={risk_metrics.omega_ratio:.3f}")

        print("\n💡 All institutional metrics calculated successfully!")
        print("🎯 Framework now has hedge fund-grade performance analysis!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())