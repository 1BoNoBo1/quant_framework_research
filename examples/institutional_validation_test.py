#!/usr/bin/env python
"""
🛡️ Institutional Validation Test

Demonstrates the comprehensive institutional validation suite
for quantitative trading strategies.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qframe.validation.institutional_validator import InstitutionalValidator, InstitutionalValidationConfig
from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
from qframe.core.interfaces import Strategy
from typing import List
import warnings

warnings.filterwarnings('ignore')


class MockStrategy(Strategy):
    """Mock strategy for testing validation"""

    def __init__(self, name: str = "Mock Strategy"):
        self.name = name

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame = None) -> List:
        """Generate mock signals"""
        # Simple momentum strategy for testing
        if len(data) < 2:
            return []

        returns = data['close'].pct_change().dropna()
        signals = []

        for i, ret in enumerate(returns):
            if ret > 0.01:  # Strong positive return
                signals.append({
                    'timestamp': data.index[i+1],
                    'type': 'BUY',
                    'strength': min(abs(ret) * 10, 1.0)
                })
            elif ret < -0.01:  # Strong negative return
                signals.append({
                    'timestamp': data.index[i+1],
                    'type': 'SELL',
                    'strength': min(abs(ret) * 10, 1.0)
                })

        return signals


def generate_test_data(n_periods: int = 1000) -> pd.DataFrame:
    """Generate realistic test market data"""
    np.random.seed(42)

    # Generate realistic price series
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')

    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, n_periods)  # 0.05% daily mean, 2% volatility

    # Add some autocorrelation and volatility clustering
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Momentum
        if abs(returns[i-1]) > 0.03:  # Volatility clustering
            returns[i] *= 1.5

    # Generate price series
    base_price = 100
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * np.random.uniform(0.995, 1.005, n_periods),
        'high': prices * np.random.uniform(1.000, 1.020, n_periods),
        'low': prices * np.random.uniform(0.980, 1.000, n_periods),
        'close': prices,
        'volume': np.random.lognormal(15, 1, n_periods)
    }, index=dates)

    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))

    return data


def test_institutional_validation():
    """Test the institutional validation suite"""

    print("🛡️ QFrame Institutional Validation Test")
    print("=" * 60)

    # Generate test data
    print("\n📊 Generating test market data...")
    test_data = generate_test_data(1000)
    print(f"Generated {len(test_data)} periods of market data")

    # Create test strategy
    print("\n🎯 Creating test strategy...")
    strategy = MockStrategy("Test Momentum Strategy")

    # Configure validation
    print("\n⚙️ Configuring institutional validation...")
    config = InstitutionalValidationConfig(
        walk_forward_periods=30,  # Reduced for testing
        out_of_sample_ratio=0.3,
        bootstrap_iterations=100,  # Reduced for speed
        min_trade_count=20,  # Reduced for testing
        max_drawdown_threshold=0.20,
        min_sharpe_threshold=0.5
    )

    # Initialize validator
    validator = InstitutionalValidator(config)

    # Run validation
    print("\n🔍 Running institutional validation suite...")
    print("This may take a few minutes...")

    result = validator.validate_strategy(strategy, test_data, "Test Momentum Strategy")

    # Display results
    print("\n" + "=" * 60)
    print("🏆 INSTITUTIONAL VALIDATION RESULTS")
    print("=" * 60)

    print(f"\n📊 Overall Assessment:")
    print(f"Strategy: {result.strategy_name}")
    print(f"Validation Score: {result.validation_score:.1f}/100")
    print(f"Tests Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")

    print(f"\n📈 Detailed Test Results:")
    for test_name, test_result in result.metrics.items():
        status = "✅" if test_result.get('passed', False) else "❌"
        score = test_result.get('score', 0)
        print(f"{status} {test_name.replace('_', ' ').title()}: {score:.1f}/10")

    # Critical Issues
    if result.critical_issues:
        print(f"\n🚨 Critical Issues:")
        for issue in result.critical_issues:
            print(f"  • {issue}")

    # Warnings
    if result.warnings:
        print(f"\n⚠️ Warnings:")
        for warning in result.warnings:
            print(f"  • {warning}")

    # Recommendations
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")

    # Test specific components
    print("\n" + "=" * 60)
    print("🔬 COMPONENT TESTING")
    print("=" * 60)

    # Test Walk-Forward Analysis
    print("\n📈 Testing Walk-Forward Analysis...")
    wf_analyzer = validator.walk_forward_analyzer
    wf_result = wf_analyzer.analyze(strategy, test_data, periods=10)

    print(f"Walk-Forward Results:")
    print(f"  • Periods Tested: {wf_result.periods_tested}")
    print(f"  • Mean Sharpe: {wf_result.mean_sharpe:.3f} ± {wf_result.sharpe_std:.3f}")
    print(f"  • Stability Score: {wf_result.stability_score:.3f}")
    print(f"  • Recommendation: {wf_result.recommendation}")

    # Test Overfitting Detection
    print("\n🔍 Testing Overfitting Detection...")
    overfitting_results = validator.overfitting_detector.detect_overfitting(strategy, test_data)

    passed_overfitting = sum(1 for r in overfitting_results.values() if r.get('passed', False))
    total_overfitting = len(overfitting_results)

    print(f"Overfitting Detection Results:")
    print(f"  • Tests Passed: {passed_overfitting}/{total_overfitting}")
    print(f"  • Assessment: {'✅ LOW RISK' if passed_overfitting >= 6 else '⚠️ MODERATE' if passed_overfitting >= 4 else '❌ HIGH RISK'}")

    # Test Probabilistic Metrics
    print("\n📊 Testing Probabilistic Metrics...")

    # Generate mock returns for testing
    mock_returns = test_data['close'].pct_change().dropna() * np.random.choice([-1, 0, 1], size=len(test_data)-1, p=[0.3, 0.4, 0.3])

    psr_result = validator.probabilistic_metrics.calculate_probabilistic_sharpe(mock_returns)

    print(f"Probabilistic Metrics Results:")
    print(f"  • Probabilistic Sharpe Ratio: {psr_result.get('probabilistic_sharpe', 0):.3f}")
    print(f"  • Sample Sharpe Ratio: {psr_result.get('sample_sharpe', 0):.3f}")
    print(f"  • 95% Confidence Interval: [{psr_result.get('confidence_interval_lower', 0):.3f}, {psr_result.get('confidence_interval_upper', 0):.3f}]")

    print("\n" + "=" * 60)
    print("✅ INSTITUTIONAL VALIDATION TEST COMPLETED")
    print("=" * 60)

    # Summary assessment
    if result.validation_score >= 70:
        print("🎉 Strategy meets institutional validation standards!")
    else:
        print("⚠️ Strategy requires improvements to meet institutional standards.")

    return result


if __name__ == "__main__":
    test_institutional_validation()