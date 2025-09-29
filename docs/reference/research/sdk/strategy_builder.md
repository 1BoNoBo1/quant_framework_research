# qframe.research.sdk.strategy_builder


🏗️ QFrame Strategy Builder - Interactive Strategy Creation

High-level interface for building custom trading strategies using QFrame components.


::: qframe.research.sdk.strategy_builder
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

- `StrategyComponent`
- `SignalRule`
- `RiskRule`
- `StrategyBuilder`

### Fonctions

- `mean_reversion_strategy`
- `momentum_strategy`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
