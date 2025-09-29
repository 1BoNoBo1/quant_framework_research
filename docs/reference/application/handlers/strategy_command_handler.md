# qframe.application.handlers.strategy_command_handler


Application Handler: StrategyCommandHandler
==========================================

Handler pour les commandes liées aux stratégies.
Orchestre les use cases de création, modification et suppression des stratégies.


::: qframe.application.handlers.strategy_command_handler
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

- `StrategyCommandHandler`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
