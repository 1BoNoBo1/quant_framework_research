# qframe.strategies.orchestration.multi_strategy_manager


Multi-Strategy Manager
======================

Orchestrates multiple trading strategies with dynamic allocation
and performance-based optimization.


::: qframe.strategies.orchestration.multi_strategy_manager
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      members_order: alphabetical
      filters:
        - "!^_"
        - "!^__"
      group_by_category: true
      show_category_heading: true

## Composants

### Classes

- `AllocationMethod`
- `StrategyMetrics`
- `StrategyAllocation`
- `MultiStrategyManager`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
