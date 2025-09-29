# qframe.infrastructure.events.saga


Infrastructure Layer: Saga Pattern
==================================

Implémentation du pattern Saga pour orchestrer des transactions distribuées
avec gestion des compensations et des échecs.


::: qframe.infrastructure.events.saga
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

- `SagaStepResult`
- `SagaState`
- `CompensationAction`
- `SagaStep`
- `SagaStepHandler`
- `SagaDefinition`
- `SagaInstance`
- `Saga`
- `SagaManager`

### Fonctions

- `get_saga_manager`

