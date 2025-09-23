# QFrame - Quick Start Guide

Get started with QFrame in 5 minutes! 🚀

## Installation

```bash
# Clone the repository
git clone https://github.com/1BoNoBo1/quant_framework_research.git
cd quant_framework_research

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

## Your First Strategy

### 1. Create a Simple Strategy

```python
from examples.strategies.ma_crossover_strategy import MovingAverageCrossoverStrategy
import pandas as pd
import numpy as np

# Create sample price data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
prices = 50000 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod()

price_data = pd.DataFrame({
    'close': prices,
    'open': prices,
    'high': prices * 1.01,
    'low': prices * 0.99,
    'volume': np.random.randint(1000, 10000, 100)
}, index=dates)

# Initialize strategy
strategy = MovingAverageCrossoverStrategy(
    fast_period=10,
    slow_period=20,
    symbol='BTC/USDT'
)

# Generate signals
signals = strategy.generate_signals(price_data)

print(f"Generated {len(signals)} trading signals")
for signal in signals[:3]:  # Show first 3
    print(f"{signal.signal_type.value} @ ${signal.price}")
```

### 2. Run a Backtest with Real Data

```python
import asyncio
from examples.backtest_real_data import run_backtest

# Run backtest with real Binance data
asyncio.run(run_backtest())
```

Output:
```
📊 Fetching historical data from Binance...
✅ Fetched 730 days of data
   Period: 2023-01-01 to 2024-12-31

💰 Portfolio Performance:
   Initial Capital: $10,000.00
   Final Value: $15,243.50
   Total Return: 52.44%
   Sharpe Ratio: 1.42
   Max Drawdown: -18.32%

📊 Trading Statistics:
   Total Trades: 23
   Win Rate: 65.22%
   Average Return per Trade: 2.28%
```

### 3. Use the Framework with DI Container

```python
from qframe.infrastructure.config.service_configuration import ServiceConfiguration
from qframe.infrastructure.config.environment_config import ApplicationConfig, Environment
from qframe.application.handlers.strategy_command_handler import StrategyCommandHandler
from qframe.application.commands.strategy_commands import CreateStrategyCommand

# Setup
config = ApplicationConfig(environment=Environment.DEVELOPMENT)
service_config = ServiceConfiguration(config)
service_config.configure()

# Get handler from DI container
command_handler = service_config.container.resolve(StrategyCommandHandler)

# Create strategy
command = CreateStrategyCommand(
    name="My MA Strategy",
    description="Moving average crossover",
    parameters={
        "fast_period": 10,
        "slow_period": 20,
        "symbol": "BTC/USDT"
    }
)

strategy = await command_handler.handle(command)
print(f"Created strategy: {strategy.name} with ID {strategy.id}")
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=qframe --cov-report=html

# Run specific test suite
poetry run pytest tests/unit/test_strategy.py -v

# Run integration tests only
poetry run pytest tests/integration/ -v
```

## Key Concepts

### 1. **Hexagonal Architecture**
- **Domain Layer**: Core business logic (entities, value objects, repositories)
- **Application Layer**: Use cases (commands, queries, handlers)
- **Infrastructure Layer**: External adapters (databases, APIs, message queues)

### 2. **Event-Driven Design**
```python
from qframe.infrastructure.events import EventBus, DomainEvent

# Publish events
event_bus = get_event_bus()
await event_bus.publish(StrategyActivatedEvent(strategy_id=strategy.id))

# Subscribe to events
@event_bus.subscribe(StrategyActivatedEvent)
async def on_strategy_activated(event):
    print(f"Strategy {event.strategy_id} activated!")
```

### 3. **CQRS Pattern**
- **Commands**: Modify state (CreateStrategy, UpdateStrategy)
- **Queries**: Read state (GetStrategyById, ListActiveStrategies)

### 4. **Observability**
```python
from qframe.infrastructure.observability import StructuredLogger, get_business_metrics

# Logging with correlation
logger = StructuredLogger("my_strategy")
logger.trade("Executed BUY order", symbol="BTC/USDT", quantity=0.5)

# Metrics
metrics = get_business_metrics()
metrics.collector.increment_counter("trades.executed", labels={"side": "buy"})
```

## Next Steps

1. **Explore Examples**: Check `/examples` directory for more strategies
2. **Read API Docs**: See `/docs/api` for detailed API reference
3. **Deploy**: Use Docker/Kubernetes configs in `/deployment`
4. **Monitor**: Set up Grafana dashboards from `/monitoring`

## Common Tasks

### Add a New Strategy
1. Create strategy class in `/examples/strategies`
2. Implement `generate_signals()` method
3. Backtest with historical data
4. Deploy via DI container

### Connect to Exchange
```python
from qframe.infrastructure.data.binance_provider import BinanceProvider

provider = BinanceProvider(
    api_key="your_key",
    api_secret="your_secret"
)

await provider.start()
await provider.subscribe_ticker("BTC/USDT")
```

### Paper Trading
```python
from qframe.infrastructure.config.environment_config import ApplicationConfig, Environment

config = ApplicationConfig(
    environment=Environment.DEVELOPMENT,
    paper_trading=True  # Enable paper trading mode
)
```

## Need Help?

- 📚 **Documentation**: `/docs`
- 💬 **Issues**: [GitHub Issues](https://github.com/1BoNoBo1/quant_framework_research/issues)
- 📧 **Contact**: research@qframe.dev

Happy Trading! 📈
