# qframe.domain.entities.portfolio


Domain Entity: Portfolio
========================

Entité agrégat représentant un portfolio de trading.
Encapsule les positions, la valorisation, et les performances.


::: qframe.domain.entities.portfolio
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

- `PortfolioStatus`
- `PortfolioType`
- `RebalancingFrequency`
- `PortfolioConstraints`
- `PortfolioSnapshot`
- `Portfolio`

### Fonctions

- `create_trading_portfolio`
- `create_backtesting_portfolio`
- `create_paper_trading_portfolio`

