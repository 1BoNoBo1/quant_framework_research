# qframe.application.backtesting.commands


Application Layer: Backtesting Command Handlers
==============================================

Handlers pour les commandes liées au backtesting.
Implémente le pattern CQRS pour les opérations de modification.


::: qframe.application.backtesting.commands
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

- `CreateBacktestConfigurationCommand`
- `UpdateBacktestConfigurationCommand`
- `DeleteBacktestConfigurationCommand`
- `RunBacktestCommand`
- `StopBacktestCommand`
- `DeleteBacktestResultCommand`
- `ArchiveBacktestResultCommand`
- `RestoreBacktestResultCommand`
- `CleanupOldBacktestResultsCommand`
- `ExportBacktestResultsCommand`
- `ImportBacktestResultsCommand`
- `CreateBacktestConfigurationHandler`
- `UpdateBacktestConfigurationHandler`
- `DeleteBacktestConfigurationHandler`
- `RunBacktestHandler`
- `StopBacktestHandler`
- `DeleteBacktestResultHandler`
- `ArchiveBacktestResultHandler`
- `RestoreBacktestResultHandler`
- `CleanupOldBacktestResultsHandler`
- `ExportBacktestResultsHandler`
- `ImportBacktestResultsHandler`

