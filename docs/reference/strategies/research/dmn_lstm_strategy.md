# qframe.strategies.research.dmn_lstm_strategy


Deep Market Network LSTM Strategy
=================================

Migration propre de votre alpha DMN existant vers l'architecture moderne.
Préserve toute la logique métier tout en utilisant les nouvelles interfaces.


::: qframe.strategies.research.dmn_lstm_strategy
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

- `DMNConfig`
- `MarketDataset`
- `DMNModel`
- `DMNLSTMStrategy`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
