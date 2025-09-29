# qframe.strategies.research.mean_reversion_strategy


Adaptive Mean Reversion Strategy
===============================

Migration de votre stratégie Mean Reversion avec améliorations ML
et détection de régimes. Préserve toute la logique sophistiquée.


::: qframe.strategies.research.mean_reversion_strategy
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

- `MeanReversionConfig`
- `RegimeDetector`
- `MLOptimizer`
- `MeanReversionStrategy`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
