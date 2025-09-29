# qframe.infrastructure.persistence.migrations


Infrastructure Layer: Database Migrations
=========================================

Système de migration de base de données avec versioning,
rollback et support SQL/Python.


::: qframe.infrastructure.persistence.migrations
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

- `MigrationStatus`
- `MigrationInfo`
- `Migration`
- `SQLMigration`
- `PythonMigration`
- `MigrationManager`

### Fonctions

- `get_migration_manager`
- `create_migration_manager`

