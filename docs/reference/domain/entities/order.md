# qframe.domain.entities.order


Domain Entity: Order
===================

Entité représentant un ordre de trading.
Encapsule le cycle de vie complet d'un ordre depuis sa création jusqu'à son exécution.


::: qframe.domain.entities.order
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

- `OrderType`
- `OrderSide`
- `OrderStatus`
- `TimeInForce`
- `OrderPriority`
- `OrderExecution`
- `OrderReject`
- `Order`

### Fonctions

- `create_market_order`
- `create_limit_order`
- `create_stop_order`
- `create_stop_limit_order`

