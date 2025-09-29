# qframe.domain.repositories.order_repository


Repository Interface: Order Repository
=====================================

Interface de repository pour la persistance des ordres.
Définit le contrat pour l'accès aux données selon l'architecture hexagonale.


::: qframe.domain.repositories.order_repository
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

- `OrderRepository`
- `OrderQuery`
- `OrderAggregateQuery`
- `RepositoryError`
- `OrderNotFoundError`
- `DuplicateOrderError`
- `OrderStateError`

