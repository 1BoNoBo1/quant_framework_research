# qframe.infrastructure.persistence.database


Infrastructure Layer: Database Management
=========================================

Gestionnaire de base de données avec connection pooling,
transaction management et monitoring.


::: qframe.infrastructure.persistence.database
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

- `IsolationLevel`
- `DatabaseConfig`
- `ConnectionStats`
- `ConnectionPool`
- `TransactionManager`
- `DatabaseManager`

### Fonctions

- `get_database_manager`
- `create_database_manager`

