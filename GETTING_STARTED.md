# 🚀 Getting Started with QFrame

**Quick setup guide for QFrame quantitative trading framework**

---

## ⚡ 2-Minute Setup

### 1. Installation

```bash
# Clone repository
git clone https://github.com/1BoNoBo1/quant_framework_research.git
cd quant_framework_research

# Install dependencies
poetry install

# Verify installation
poetry run python demo_framework.py
```

**Expected output:**
```
============================================================
  🚀 QFrame Framework - Operational Demo
============================================================

✅ All systems operational
```

### 2. Your First Trading Strategy

```bash
# Run minimal example
poetry run python examples/minimal_example.py
```

This will:
- Create a portfolio with $10,000
- Load sample market data
- Execute a simple moving average crossover strategy
- Show buy/sell signals and final results

### 3. ✨ Advanced Example - Complete Framework Demo

```bash
# Run the enhanced example showing full capabilities
poetry run python examples/enhanced_example.py
```

This demonstrates:
- **Multi-Portfolio**: 3 portfolios with different strategies
- **262 Orders**: Generated across Mean Reversion, Momentum, and Arbitrage strategies
- **Complete Order Repository**: All 20+ methods working (statistics, archiving, etc.)
- **Realistic Market Data**: 721 periods with Bitcoin-like price movements
- **Advanced Features**: Priority orders, time-in-force, order fills simulation

**Sample Output:**
```
🚀 QFrame Enhanced Trading Example
✓ Created 3 portfolios with total capital: $30000.00
✓ Generated realistic market data: 721 periods
📈 Mean Reversion Strategy: 89 orders created
📈 Momentum Strategy: 87 orders created
📈 Arbitrage Strategy: 86 orders created
📊 Total Orders: 262 orders across 3 portfolios
🎯 Framework Status: All repository methods working perfectly
```

### 4. Explore the CLI

```bash
# Framework information
poetry run python qframe_cli.py info

# List available strategies
poetry run python qframe_cli.py strategies

# Run framework tests
poetry run python qframe_cli.py test
```

---

## 📊 Understanding Your First Strategy

The minimal example demonstrates core QFrame concepts:

### Portfolio Management
```python
from qframe.domain.entities.portfolio import Portfolio

portfolio = Portfolio(
    id="portfolio-001",
    name="Test Portfolio",
    initial_capital=Decimal("10000.00"),
    base_currency="USD"
)
```

### Order Creation
```python
from qframe.domain.entities.order import Order, OrderSide, OrderType

order = Order(
    id=f"order-{i:03d}",
    portfolio_id=portfolio.id,
    symbol="BTC/USD",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal("0.01"),
    price=price,
    created_time=timestamp
)
```

### Trading Logic
```python
# Simple moving average crossover
for i in range(5, len(data), 10):
    current_price = data.iloc[i]['close']
    moving_average = data['close'].iloc[i-5:i].mean()

    if current_price < moving_average:
        # Create BUY order
    elif current_price > moving_average * 1.02:
        # Create SELL order
```

---

## 🧪 Testing Your Setup

### Run Tests
```bash
# All tests (173/232 passing)
poetry run pytest

# Quick summary
poetry run pytest --tb=no -q

# Specific categories
poetry run pytest tests/unit/ -v
poetry run pytest tests/integration/ -v
```

### Verify Components
```bash
# Check imports work
poetry run python -c "import qframe; print('✅ QFrame imported successfully')"

# Check configuration
poetry run python -c "from qframe.core.config import FrameworkConfig; print('✅ Configuration loaded')"

# Check repositories
poetry run python -c "from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository; print('✅ Repositories working')"
```

---

## 📈 Next Steps

### 1. Explore Research Strategies

```bash
# Check available strategies
poetry run python qframe_cli.py strategies
```

Available strategies:
- **DMN LSTM**: Deep Market Networks with LSTM
- **Mean Reversion**: Statistical mean reversion with ML
- **Funding Arbitrage**: Cross-exchange funding arbitrage
- **RL Alpha**: Reinforcement Learning alpha generation

### 2. Create Your Own Strategy

Create `my_strategy.py`:
```python
import pandas as pd
from qframe.domain.entities.portfolio import Portfolio
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository

async def my_trading_strategy():
    # Your strategy logic here
    portfolio_repo = MemoryPortfolioRepository()

    portfolio = Portfolio(
        name="My Strategy",
        initial_capital=Decimal("5000.00")
    )

    await portfolio_repo.save(portfolio)
    print(f"Created portfolio: {portfolio.name}")

# Run your strategy
if __name__ == "__main__":
    import asyncio
    asyncio.run(my_trading_strategy())
```

### 3. Work with Real Data

```python
# Example: Load crypto data (when providers are working)
from qframe.infrastructure.data.binance_provider import BinanceProvider

# Note: Provider tests are currently failing, but basic structure exists
provider = BinanceProvider()
# data = await provider.get_ohlcv("BTCUSDT", "1h", limit=100)
```

### 4. Backtesting

```python
# Basic backtesting structure exists
from qframe.domain.services.backtesting_service import BacktestingService

# Note: Full backtesting needs more development
# service = BacktestingService()
```

---

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
If you see import errors:
```bash
# Reinstall dependencies
poetry install --no-cache

# Check Python version
python --version  # Should be 3.11+
```

#### Test Failures
If tests fail:
```bash
# Run only working tests
poetry run pytest tests/unit/test_config.py -v
poetry run pytest tests/unit/test_container.py -v
```

#### CLI Issues
If original CLI doesn't work:
```bash
# Use alternative CLI
poetry run python qframe_cli.py --help
```

### Known Limitations

1. **Binance Provider**: Some tests fail (mocking issues)
2. **Risk Calculations**: Advanced VaR/CVaR need fixes
3. **Order Repository**: Missing some methods
4. **Original CLI**: Typer compatibility issue

See [Functional Audit Report](FUNCTIONAL_AUDIT_REPORT.md) for complete status.

---

## 📚 Learning Resources

### Documentation
- [📊 Functional Audit Report](FUNCTIONAL_AUDIT_REPORT.md) - Current status
- [🏗️ Infrastructure Audit](INFRASTRUCTURE_AUDIT.md) - Architecture details
- [📘 Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical overview

### Code Examples
- `examples/minimal_example.py` - Basic portfolio and orders
- `examples/enhanced_example.py` - ✨ **NEW** Complete framework demonstration (262 orders)
- `test_order_repository.py` - Complete Order Repository testing
- `examples/backtest_example.py` - Backtesting structure
- `qframe_cli.py` - CLI implementation
- `demo_framework.py` - Framework verification

### Key Concepts
- **Domain Entities**: Portfolio, Order, Strategy
- **Repository Pattern**: Memory and PostgreSQL implementations
- **Dependency Injection**: IoC container for loose coupling
- **Configuration**: Pydantic-based type-safe config

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Verify setup
poetry run python demo_framework.py

# Run example
poetry run python examples/minimal_example.py

# Framework info
poetry run python qframe_cli.py info

# Test framework
poetry run pytest tests/unit/ -v
```

### Import Patterns
```python
# Core framework
from qframe.core.config import FrameworkConfig
from qframe.core.container import get_container

# Domain entities
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.order import Order, OrderSide, OrderType

# Infrastructure
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository
```

---

**You're ready to start quantitative trading with QFrame! 🎉**

For questions, check the [Functional Audit Report](FUNCTIONAL_AUDIT_REPORT.md) or create an issue on GitHub.