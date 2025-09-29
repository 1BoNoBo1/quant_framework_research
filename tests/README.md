# 🧪 QFrame Test Suite

Comprehensive test suite for the QFrame quantitative trading framework with institutional-grade validation.

## 📋 Test Organization

### Test Categories

| Category | Marker | Description | Duration | Criticality |
|----------|--------|-------------|-----------|-------------|
| **Unit** | `unit` | Fast, isolated component tests | < 1s | Critical |
| **Integration** | `integration` | Component interaction tests | 1-10s | Critical |
| **UI** | `ui` | Web interface component tests | 2-15s | Important |
| **Strategies** | `strategies` | Trading strategy algorithm tests | 5-30s | Critical |
| **Backtesting** | `backtesting` | Historical backtesting engine tests | 10-60s | Important |
| **Data** | `data` | External data provider tests | 5-20s | Important |
| **Risk** | `risk` | Risk calculation and management tests | 2-10s | Critical |
| **Performance** | `performance` | System performance benchmarks | 30-300s | Optional |
| **Slow** | `slow` | Long-running tests | > 30s | Optional |

### Test Structure

```
tests/
├── README.md                          # This file
├── conftest.py                        # Shared fixtures and configuration
├── pytest.ini                        # Pytest configuration
├── test_organization.py              # Test organization utilities
├── test_institutional_metrics.py     # Institutional metrics tests
├──
├── unit/                              # Unit tests (fast, isolated)
│   ├── test_config.py                 # Configuration tests
│   ├── test_container.py              # Dependency injection tests
│   ├── test_symbolic_operators.py     # Financial operators tests
│   └── ...
├──
├── integration/                       # Integration tests
│   ├── test_strategy_workflow.py      # End-to-end strategy tests
│   ├── test_phase7_research_platform.py
│   └── test_phase8_ecosystem.py
├──
├── strategies/                        # Strategy-specific tests
│   └── test_dmn_lstm_strategy.py
├──
├── ui/                               # UI component tests
│   ├── conftest.py
│   ├── test_backtesting_integration.py
│   └── components/
├──
├── data/                             # Data provider tests
│   └── test_binance_provider.py
├──
├── risk/                             # Risk management tests
│   └── test_risk_calculation_service.py
├──
├── backtesting/                      # Backtesting tests
│   └── test_backtest_engine.py
└──
└── execution/                        # Order execution tests
    └── test_order_execution.py
```

## 🚀 Quick Start

### Essential Commands

```bash
# Install dependencies
poetry install

# Quick development tests (< 2 minutes)
make test-quick

# Critical tests for production validation (< 5 minutes)
make test-critical

# Complete test suite (< 30 minutes)
make test-all

# Institutional metrics validation
make test-institutional

# Test organization information
make test-organization
```

### Test Execution Plans

#### Quick Development (`make test-quick`)
- **Purpose**: Fast feedback during development
- **Includes**: Unit tests only
- **Excludes**: Slow and performance tests
- **Duration**: ~120 seconds
- **Parallel**: Yes

#### CI/CD Pipeline (`make test-ci`)
- **Purpose**: Complete automated testing
- **Includes**: Unit, integration, strategies, risk
- **Excludes**: Slow and performance tests
- **Duration**: ~10 minutes
- **Parallel**: Yes

#### Critical Production (`make test-critical`)
- **Purpose**: Production readiness validation
- **Includes**: All critical tests
- **Excludes**: Optional tests
- **Duration**: ~5 minutes
- **Parallel**: Yes

#### Full Suite (`make test-all`)
- **Purpose**: Complete validation including slow tests
- **Includes**: All test categories
- **Duration**: ~30 minutes
- **Parallel**: Yes

#### Performance Benchmarks (`make test-performance`)
- **Purpose**: Performance and benchmark testing
- **Includes**: Performance tests only
- **Duration**: ~60 minutes
- **Parallel**: No (for accurate benchmarks)

## 📊 Coverage and Quality

### Coverage Targets

- **Unit Tests**: 95%+ coverage
- **Integration Tests**: 80%+ coverage
- **Overall Target**: 85%+ coverage

### Quality Standards

- All tests must have appropriate markers (`@pytest.mark.unit`, `@pytest.mark.critical`, etc.)
- Test names should be descriptive and follow pattern: `test_<functionality>_<scenario>`
- Use fixtures for common test data and setup
- Mock external dependencies in unit tests
- Include edge case testing
- Performance tests should have benchmarks and thresholds

## 🧪 Writing Tests

### Test Markers

Always mark your tests with appropriate categories and criticality:

```python
@pytest.mark.unit
@pytest.mark.critical
class TestMyComponent:
    def test_basic_functionality(self):
        pass

@pytest.mark.integration
@pytest.mark.important
def test_component_interaction():
    pass

@pytest.mark.performance
@pytest.mark.optional
def test_performance_benchmark():
    pass
```

### Using Fixtures

Create reusable test data with fixtures:

```python
@pytest.fixture
def sample_market_data():
    """Generate realistic market data for testing"""
    return generate_ohlcv_data(periods=100)

@pytest.fixture
def mock_data_provider():
    """Mock data provider for isolated testing"""
    with patch('qframe.data.BinanceProvider') as mock:
        yield mock
```

### Test Patterns

#### Unit Test Pattern
```python
@pytest.mark.unit
@pytest.mark.critical
def test_information_coefficient_calculation():
    # Arrange
    service = InstitutionalMetricsService()
    predictions = generate_predictions()
    actual_returns = generate_returns()

    # Act
    result = service.calculate_information_metrics(predictions, actual_returns)

    # Assert
    assert -1 <= result.information_coefficient <= 1
    assert result.predictive_power in ['EXCELLENT', 'GOOD', 'MODERATE', 'POOR']
```

#### Integration Test Pattern
```python
@pytest.mark.integration
@pytest.mark.critical
def test_strategy_execution_workflow():
    # Test complete workflow from data ingestion to signal generation
    strategy = AdaptiveMeanReversionStrategy()
    data_provider = MockDataProvider()

    # Execute complete workflow
    data = data_provider.fetch_ohlcv("BTC/USD", "1h")
    signals = strategy.generate_signals(data)

    # Validate workflow results
    assert len(signals) > 0
    assert all(s.symbol == "BTC/USD" for s in signals)
```

## 🔧 Test Configuration

### Environment Variables

Set these for testing:

```bash
export QFRAME_ENV=testing
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Pytest Configuration

Key settings in `pytest.ini`:

- **Test Discovery**: Automatic discovery of `test_*.py` files
- **Markers**: Strict marker validation
- **Coverage**: Minimum 75% requirement
- **Parallel**: Automatic parallel execution support
- **Warnings**: Filtered common warnings

### CI/CD Integration

The test suite is designed for CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Critical Tests
  run: make test-critical

- name: Run Full Test Suite
  run: make test-all
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

## 📈 Continuous Improvement

### Test Metrics Tracking

Monitor these metrics:

- **Test Count**: Track growth of test coverage
- **Execution Time**: Monitor test performance
- **Flakiness**: Track and fix unstable tests
- **Coverage**: Maintain high coverage standards

### Adding New Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Mark appropriately** with category and criticality
3. **Include edge cases** and error conditions
4. **Add to appropriate test plan** (quick, ci, full)
5. **Update documentation** if needed

### Performance Monitoring

- Track slow tests with `--durations=10`
- Optimize or mark as `@pytest.mark.slow`
- Set realistic timeouts
- Use profiling for complex test scenarios

## 🔍 Debugging Tests

### Common Commands

```bash
# Run specific test with verbose output
poetry run pytest tests/unit/test_config.py::TestDatabaseConfig::test_default_values -v

# Run with debugging
poetry run pytest tests/unit/test_config.py -v -s --tb=long

# Run with coverage report
poetry run pytest tests/unit/ --cov=qframe --cov-report=html

# Run tests matching pattern
poetry run pytest tests/ -k "information_coefficient"

# Run with profiling
poetry run pytest tests/unit/ --profile
```

### Test Debugging Tips

1. **Use `--tb=long`** for detailed tracebacks
2. **Add `print()` statements** or use `pytest.set_trace()`
3. **Run single tests** to isolate issues
4. **Check fixtures** with `--setup-show`
5. **Mock external dependencies** to avoid network issues

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [QFrame Testing Guidelines](../CLAUDE.md#tests--qualité)
- [Institutional Metrics Testing](./test_institutional_metrics.py)

---

*This test suite ensures QFrame meets institutional-grade quality standards for quantitative trading systems.*