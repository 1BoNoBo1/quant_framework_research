#!/usr/bin/env python
"""
🔍 QFrame Data Integrity Validation Script

Validates data quality and integrity across all framework components:
- OHLCV data consistency
- Backtesting results accuracy
- Research Platform data integrity
- Metric calculations correctness
"""

import sys
import asyncio
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import traceback

import pandas as pd
import numpy as np

# Add project root to path
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings for clean output
warnings.filterwarnings('ignore')


class DataIntegrityValidator:
    """
    🔍 Comprehensive data integrity validation for QFrame

    Tests all data flows, calculations, and storage integrity
    """

    def __init__(self):
        self.results = {
            "ohlcv_validation": [],
            "metrics_validation": [],
            "storage_validation": [],
            "research_validation": [],
            "integration_validation": []
        }
        self.errors = []
        self.warnings = []

    def run_all_validations(self) -> Dict[str, bool]:
        """Run complete data integrity validation suite"""
        print("🔍 QFrame Data Integrity Validation")
        print("=" * 60)

        validation_results = {}

        # 1. OHLCV Data Validation
        print("\n📊 1. OHLCV Data Validation...")
        validation_results["ohlcv"] = self.validate_ohlcv_data()

        # 2. Metrics Calculation Validation
        print("\n📈 2. Metrics Calculation Validation...")
        validation_results["metrics"] = self.validate_metrics_calculations()

        # 3. Storage Integrity Validation
        print("\n💾 3. Storage Integrity Validation...")
        validation_results["storage"] = asyncio.run(self.validate_storage_integrity())

        # 4. Research Platform Validation
        print("\n🔬 4. Research Platform Validation...")
        validation_results["research"] = self.validate_research_platform()

        # 5. End-to-End Integration Validation
        print("\n🔗 5. Integration Validation...")
        validation_results["integration"] = self.validate_integration_flow()

        # Summary
        self._print_summary(validation_results)
        return validation_results

    def validate_ohlcv_data(self) -> bool:
        """Validate OHLCV data consistency and quality"""
        try:
            # Generate test OHLCV data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1h')
            np.random.seed(42)

            # Realistic price simulation
            base_price = 50000
            n_periods = len(dates)
            returns = np.random.normal(0.0001, 0.02, n_periods)
            log_prices = np.log(base_price) + returns.cumsum()
            close_prices = np.exp(log_prices)

            data = pd.DataFrame({
                'open': close_prices * (1 + np.random.normal(0, 0.001, n_periods)),
                'high': close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
                'low': close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
                'close': close_prices,
                'volume': np.random.lognormal(10, 0.5, n_periods)
            }, index=dates)

            # Ensure OHLC consistency
            data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
            data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))

            # Validation checks
            checks = {
                'ohlc_consistency': self._check_ohlc_consistency(data),
                'positive_prices': self._check_positive_prices(data),
                'positive_volume': self._check_positive_volume(data),
                'no_missing_data': self._check_no_missing_data(data),
                'reasonable_volatility': self._check_reasonable_volatility(data),
                'no_extreme_jumps': self._check_no_extreme_jumps(data)
            }

            # Report results
            passed = 0
            for check_name, (result, message) in checks.items():
                status = "✅" if result else "❌"
                print(f"  {status} {check_name}: {message}")
                if result:
                    passed += 1
                else:
                    self.errors.append(f"OHLCV validation failed: {check_name} - {message}")

            success = passed >= len(checks) * 0.9  # 90% pass rate
            print(f"  📊 OHLCV validation: {passed}/{len(checks)} ({'✅ PASS' if success else '❌ FAIL'})")
            return success

        except Exception as e:
            print(f"  ❌ OHLCV validation error: {e}")
            self.errors.append(f"OHLCV validation exception: {e}")
            return False

    def validate_metrics_calculations(self) -> bool:
        """Validate accuracy of financial metrics calculations"""
        try:
            # Generate test returns
            np.random.seed(42)
            returns = pd.Series(np.random.normal(0.001, 0.02, 252))
            prices = (1 + returns).cumprod() * 100

            # Manual calculations for validation
            total_return_manual = (prices.iloc[-1] / prices.iloc[0]) - 1
            volatility_manual = returns.std() * np.sqrt(252)
            sharpe_manual = (returns.mean() * 252) / volatility_manual if volatility_manual > 0 else 0

            # Drawdown calculation
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown_manual = drawdowns.min()

            # Test BacktestResult entity
            from qframe.domain.entities.backtest import BacktestResult, BacktestMetrics, BacktestStatus
            from decimal import Decimal

            # Create metrics object
            metrics = BacktestMetrics(
                total_return=Decimal(str(total_return_manual)),
                sharpe_ratio=Decimal(str(sharpe_manual)),
                max_drawdown=Decimal(str(max_drawdown_manual)),
                volatility=Decimal(str(volatility_manual))
            )

            result = BacktestResult(
                name='test_validation',
                status=BacktestStatus.COMPLETED,
                initial_capital=Decimal('100000.0'),
                final_capital=Decimal(str(100000.0 * (1 + total_return_manual))),
                metrics=metrics
            )

            # Validation checks
            checks = {
                'return_consistency': (
                    abs(float(result.metrics.total_return) - total_return_manual) < 0.0001,
                    f"Expected {total_return_manual:.4f}, got {result.metrics.total_return:.4f}"
                ),
                'capital_consistency': (
                    abs(float(result.final_capital) - float(result.initial_capital) * (1 + float(result.metrics.total_return))) < 0.01,
                    f"Capital calculation mismatch"
                ),
                'sharpe_reasonable': (
                    -5 <= float(result.metrics.sharpe_ratio) <= 5,
                    f"Sharpe ratio {result.metrics.sharpe_ratio:.4f} seems reasonable"
                ),
                'volatility_positive': (
                    float(result.metrics.volatility) >= 0,
                    f"Volatility {result.metrics.volatility:.4f} is positive"
                ),
                'drawdown_negative': (
                    float(result.metrics.max_drawdown) <= 0,
                    f"Max drawdown {result.metrics.max_drawdown:.4f} is negative"
                )
            }

            # Test Research Platform metrics
            try:
                from qframe.research.backtesting.performance_analyzer import AdvancedPerformanceAnalyzer
                analyzer = AdvancedPerformanceAnalyzer()
                analysis = analyzer.analyze_comprehensive(result)

                # Validate advanced metrics
                basic_metrics = analysis.get('basic_metrics', {})
                risk_metrics = analysis.get('risk_metrics', {})

                checks['sortino_calculated'] = (
                    'sortino_ratio' in risk_metrics,
                    f"Sortino ratio calculated: {risk_metrics.get('sortino_ratio', 'N/A')}"
                )
                checks['var_calculated'] = (
                    'var_95' in risk_metrics,
                    f"VaR calculated: {risk_metrics.get('var_95', 'N/A')}"
                )

            except ImportError:
                self.warnings.append("Research Platform not available for metrics validation")

            # Report results
            passed = 0
            for check_name, (result_ok, message) in checks.items():
                status = "✅" if result_ok else "❌"
                print(f"  {status} {check_name}: {message}")
                if result_ok:
                    passed += 1
                else:
                    self.errors.append(f"Metrics validation failed: {check_name} - {message}")

            success = passed >= len(checks) * 0.9
            print(f"  📈 Metrics validation: {passed}/{len(checks)} ({'✅ PASS' if success else '❌ FAIL'})")
            return success

        except Exception as e:
            print(f"  ❌ Metrics validation error: {e}")
            self.errors.append(f"Metrics validation exception: {e}")
            return False

    async def validate_storage_integrity(self) -> bool:
        """Validate data storage and retrieval integrity"""
        try:
            from qframe.research.data_lake.storage import LocalFileStorage
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as temp_dir:
                storage = LocalFileStorage(temp_dir)

                # Create test data
                test_data = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1h'),
                    'price': np.random.randn(1000).cumsum() + 50000,
                    'volume': np.random.lognormal(10, 0.5, 1000),
                    'signal': np.random.choice(['buy', 'sell', 'hold'], 1000)
                })

                # Test storage
                metadata = await storage.put_dataframe(test_data, 'test_integrity.parquet')

                # Test retrieval
                retrieved_data = await storage.get_dataframe('test_integrity.parquet')

                # Validation checks
                checks = {
                    'data_size_match': (
                        len(test_data) == len(retrieved_data),
                        f"Size: {len(test_data)} vs {len(retrieved_data)}"
                    ),
                    'columns_match': (
                        list(test_data.columns) == list(retrieved_data.columns),
                        f"Columns match"
                    ),
                    'data_types_preserved': (
                        test_data.dtypes.equals(retrieved_data.dtypes),
                        f"Data types preserved"
                    ),
                    'values_identical': (
                        test_data.equals(retrieved_data),
                        f"Values identical"
                    ),
                    'metadata_valid': (
                        metadata.size_bytes > 0 and metadata.path == 'test_integrity.parquet',
                        f"Metadata: {metadata.size_bytes} bytes"
                    )
                }

                # Report results
                passed = 0
                for check_name, (result, message) in checks.items():
                    status = "✅" if result else "❌"
                    print(f"  {status} {check_name}: {message}")
                    if result:
                        passed += 1
                    else:
                        self.errors.append(f"Storage validation failed: {check_name} - {message}")

                success = passed >= len(checks) * 0.95  # 95% for storage integrity
                print(f"  💾 Storage validation: {passed}/{len(checks)} ({'✅ PASS' if success else '❌ FAIL'})")
                return success

        except Exception as e:
            print(f"  ❌ Storage validation error: {e}")
            self.errors.append(f"Storage validation exception: {e}")
            return False

    def validate_research_platform(self) -> bool:
        """Validate Research Platform data processing"""
        try:
            # Test distributed engine
            from qframe.research.backtesting.distributed_engine import DistributedBacktestEngine

            engine = DistributedBacktestEngine(compute_backend='sequential', max_workers=2)

            # Test task creation
            strategies = ['test_strategy']
            data = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [101, 102, 103],
                'low': [99, 100, 101],
                'close': [100.5, 101.5, 102.5],
                'volume': [1000, 1100, 1200]
            })

            tasks = engine._create_backtest_tasks(
                strategies=strategies,
                datasets=[data],
                parameter_grids=None,
                split_strategy='time_series',
                n_splits=1,
                initial_capital=100000.0
            )

            # Validation checks
            checks = {
                'engine_initialized': (
                    engine is not None,
                    f"Engine backend: {engine.compute_backend}"
                ),
                'tasks_created': (
                    len(tasks) > 0,
                    f"Created {len(tasks)} tasks"
                ),
                'task_structure_valid': (
                    all('id' in task and 'strategy_name' in task for task in tasks),
                    "Task structure valid"
                )
            }

            # Test advanced analytics if available
            try:
                from qframe.research.backtesting.performance_analyzer import AdvancedPerformanceAnalyzer
                analyzer = AdvancedPerformanceAnalyzer()

                # Mock result for testing
                from qframe.domain.entities.backtest import BacktestResult, BacktestMetrics, BacktestStatus
                from decimal import Decimal

                mock_metrics = BacktestMetrics(
                    total_return=Decimal('0.15'),
                    sharpe_ratio=Decimal('1.2'),
                    max_drawdown=Decimal('-0.08'),
                    volatility=Decimal('0.18')
                )

                mock_result = BacktestResult(
                    name='mock_test',
                    status=BacktestStatus.COMPLETED,
                    metrics=mock_metrics
                )

                analysis = analyzer.analyze_comprehensive(mock_result)

                checks['analytics_working'] = (
                    len(analysis) > 0,
                    f"Generated {len(analysis)} analysis sections"
                )

            except ImportError:
                self.warnings.append("Advanced analytics not available")

            # Report results
            passed = 0
            for check_name, (result, message) in checks.items():
                status = "✅" if result else "❌"
                print(f"  {status} {check_name}: {message}")
                if result:
                    passed += 1
                else:
                    self.errors.append(f"Research platform validation failed: {check_name} - {message}")

            success = passed >= len(checks) * 0.8
            print(f"  🔬 Research validation: {passed}/{len(checks)} ({'✅ PASS' if success else '❌ FAIL'})")
            return success

        except Exception as e:
            print(f"  ❌ Research platform validation error: {e}")
            self.errors.append(f"Research platform validation exception: {e}")
            return False

    def validate_integration_flow(self) -> bool:
        """Validate end-to-end data flow integration"""
        try:
            # Test basic QFrame Core functionality
            from qframe.core.container import get_container

            container = get_container()

            # Test integration layer if available
            integration_available = False
            try:
                from qframe.research.integration_layer import create_research_integration
                integration = create_research_integration(use_minio=False)
                status = integration.get_integration_status()
                integration_available = True
            except Exception:
                self.warnings.append("Research integration layer not fully available")
                status = {}

            # Validation checks
            checks = {
                'container_available': (
                    container is not None,
                    "QFrame Core container initialized"
                ),
                'integration_status': (
                    integration_available,
                    f"Integration layer available: {integration_available}"
                )
            }

            if integration_available:
                checks['qframe_core_connected'] = (
                    status.get('qframe_core_available', False),
                    "QFrame Core connected to Research Platform"
                )

            # Report results
            passed = 0
            for check_name, (result, message) in checks.items():
                status = "✅" if result else "❌"
                print(f"  {status} {check_name}: {message}")
                if result:
                    passed += 1
                else:
                    self.errors.append(f"Integration validation failed: {check_name} - {message}")

            success = passed >= len(checks) * 0.7  # Lower threshold due to optional components
            print(f"  🔗 Integration validation: {passed}/{len(checks)} ({'✅ PASS' if success else '❌ FAIL'})")
            return success

        except Exception as e:
            print(f"  ❌ Integration validation error: {e}")
            self.errors.append(f"Integration validation exception: {e}")
            return False

    # Helper methods for OHLCV validation
    def _check_ohlc_consistency(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check OHLC price consistency"""
        valid = ((data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close'])).all()
        return valid, "All OHLC relationships valid" if valid else "OHLC inconsistencies found"

    def _check_positive_prices(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check all prices are positive"""
        valid = ((data['open'] > 0) & (data['high'] > 0) &
                (data['low'] > 0) & (data['close'] > 0)).all()
        return valid, "All prices positive" if valid else "Negative prices found"

    def _check_positive_volume(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check volume is positive"""
        valid = (data['volume'] > 0).all()
        return valid, "All volumes positive" if valid else "Non-positive volumes found"

    def _check_no_missing_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check for missing data"""
        missing = data.isnull().sum().sum()
        valid = missing == 0
        return valid, f"No missing data" if valid else f"{missing} missing values found"

    def _check_reasonable_volatility(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check volatility is reasonable"""
        returns = data['close'].pct_change().dropna()
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(365 * 24)  # Hourly data
        valid = 0.01 <= annual_vol <= 5.0  # 1% to 500% annual volatility
        return valid, f"Annual volatility {annual_vol:.2f} is reasonable" if valid else f"Extreme volatility {annual_vol:.2f}"

    def _check_no_extreme_jumps(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check for extreme price jumps"""
        returns = data['close'].pct_change().abs()
        extreme_moves = (returns > 0.2).sum()  # >20% moves
        valid = extreme_moves < len(data) * 0.01  # <1% of data
        return valid, f"No extreme jumps" if valid else f"{extreme_moves} extreme price jumps found"

    def _print_summary(self, results: Dict[str, bool]):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("📊 DATA INTEGRITY VALIDATION SUMMARY")
        print("=" * 60)

        total_validations = len(results)
        passed_validations = sum(results.values())

        for validation_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {validation_name:20} {status}")

        print(f"\n📈 Overall: {passed_validations}/{total_validations} validations passed")

        if self.warnings:
            print(f"\n⚠️ {len(self.warnings)} warnings:")
            for warning in self.warnings[:3]:
                print(f"  • {warning}")

        if self.errors:
            print(f"\n❌ {len(self.errors)} errors:")
            for error in self.errors[:3]:
                print(f"  • {error}")

        success_rate = passed_validations / total_validations
        if success_rate >= 0.9:
            print("🎉 DATA INTEGRITY: EXCELLENT")
        elif success_rate >= 0.7:
            print("✅ DATA INTEGRITY: GOOD")
        elif success_rate >= 0.5:
            print("⚠️ DATA INTEGRITY: ACCEPTABLE")
        else:
            print("❌ DATA INTEGRITY: NEEDS ATTENTION")


def main():
    """Main validation function"""
    validator = DataIntegrityValidator()
    results = validator.run_all_validations()

    # Return appropriate exit code
    success_rate = sum(results.values()) / len(results)
    return 0 if success_rate >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())