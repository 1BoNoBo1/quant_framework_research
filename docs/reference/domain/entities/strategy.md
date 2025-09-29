# qframe.domain.entities.strategy


Domain Entity: Strategy
======================

Entité centrale représentant une stratégie de trading dans l'architecture hexagonale.
Cette entité contient la logique métier pure, indépendante de l'infrastructure.


::: qframe.domain.entities.strategy
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
- `StrategyStatus`
- `Strategy`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
