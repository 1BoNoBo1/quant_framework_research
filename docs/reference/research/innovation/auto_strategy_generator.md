# qframe.research.innovation.auto_strategy_generator


Auto Strategy Generator
======================

Revolutionary automated strategy generation using genetic algorithms,
machine learning, and symbolic evolution.


::: qframe.research.innovation.auto_strategy_generator
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

- `StrategyType`
- `ComplexityLevel`
- `StrategyComponent`
- `StrategyTemplate`
- `GenerationConfig`
- `AutoStrategyGenerator`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
