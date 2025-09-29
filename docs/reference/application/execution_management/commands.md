# qframe.application.execution_management.commands


Application Commands: Execution Management
==========================================

Commandes pour les opérations de gestion d'exécution des ordres.
Implémente le pattern CQRS pour la séparation des responsabilités.


::: qframe.application.execution_management.commands
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

- `CreateOrderCommand`
- `SubmitOrderCommand`
- `ModifyOrderCommand`
- `CancelOrderCommand`
- `ExecuteOrderCommand`
- `AddExecutionCommand`
- `CreateExecutionPlanCommand`
- `CreateChildOrdersCommand`
- `BulkCancelOrdersCommand`
- `SetOrderPriorityCommand`
- `CreateOrderHandler`
- `SubmitOrderHandler`
- `ModifyOrderHandler`
- `CancelOrderHandler`
- `ExecuteOrderHandler`
- `AddExecutionHandler`
- `CreateExecutionPlanHandler`
- `CreateChildOrdersHandler`
- `BulkCancelOrdersHandler`

