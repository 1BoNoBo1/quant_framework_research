# qframe.application.queries.strategy_queries


Strategy Queries
================

Query objects for strategy-related read operations in CQRS architecture.


::: qframe.application.queries.strategy_queries
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

- `GetStrategyByIdQuery`
- `GetActiveStrategiesQuery`
- `GetStrategiesByTypeQuery`
- `GetStrategyPerformanceQuery`
- `SearchStrategiesQuery`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
