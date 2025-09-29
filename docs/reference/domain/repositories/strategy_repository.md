# qframe.domain.repositories.strategy_repository


Repository Interface: StrategyRepository
======================================

Interface abstraite pour la persistance des stratégies.
Définit le contrat sans dépendance à l'implémentation.


::: qframe.domain.repositories.strategy_repository
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

- `StrategyRepository`
- `RepositoryError`
- `StrategyNotFoundError`
- `DuplicateStrategyError`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
