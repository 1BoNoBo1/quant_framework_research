# qframe.strategies.research.rl_alpha_strategy


Reinforcement Learning Alpha Strategy
====================================

Migration du générateur RL d'alphas vers l'architecture moderne.
Basé sur "Synergistic Formulaic Alpha Generation for Quantitative Trading".


::: qframe.strategies.research.rl_alpha_strategy
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

- `AlphaFormula`
- `RLAlphaConfig`
- `SearchSpace`
- `FormulaEnvironment`
- `PPOAgent`
- `RLAlphaStrategy`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
