# qframe.strategies.research.adaptive_mean_reversion_strategy


Stratégie de Mean Reversion Adaptative avec Machine Learning

Mathematical Foundation:
Cette stratégie utilise l'apprentissage automatique pour détecter les régimes de marché
et adapter dynamiquement les paramètres de mean reversion. Le modèle principal utilise
un ensemble LSTM + Random Forest pour classifier les régimes de marché (trending, ranging, volatile)
et ajuste les seuils de mean reversion en conséquence.

Expected Performance:
- Sharpe Ratio: 1.8-2.2
- Max Drawdown: 8-12%
- Win Rate: 58-65%
- Information Coefficient: 0.06-0.09


::: qframe.strategies.research.adaptive_mean_reversion_strategy
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

- `AdaptiveMeanReversionSignal`
- `AdaptiveMeanReversionStrategy`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
