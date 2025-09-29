# qframe.strategies.research.funding_arbitrage_strategy


Advanced Funding Rate Arbitrage Strategy
=======================================

Migration de votre stratégie Funding Rate sophistiquée avec:
- Calcul réel des funding rates crypto
- Prédiction ML des funding rates futurs
- Arbitrage spot-futures intelligent
- Gestion du risque de base


::: qframe.strategies.research.funding_arbitrage_strategy
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

- `FundingArbitrageConfig`
- `FundingRateCalculator`
- `MLFundingPredictor`
- `FundingArbitrageStrategy`

## Exemples d'usage


```python
from qframe.core.container import get_container

# Utilisation avec DI Container
container = get_container()
strategy = container.resolve(YourStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
```
