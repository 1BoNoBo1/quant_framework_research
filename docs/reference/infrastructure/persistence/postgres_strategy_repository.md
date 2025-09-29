# qframe.infrastructure.persistence.postgres_strategy_repository


Infrastructure: PostgresStrategyRepository
==========================================

Implémentation PostgreSQL du repository de stratégies.
Version placeholder pour la configuration DI.


::: qframe.infrastructure.persistence.postgres_strategy_repository
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

- `PostgresStrategyRepository`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
