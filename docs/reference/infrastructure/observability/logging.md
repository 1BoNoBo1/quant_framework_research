# qframe.infrastructure.observability.logging


Infrastructure Layer: Structured Logging System
==============================================

Système de logging structuré avec correlation IDs, contexte métier,
et formatage JSON pour analyse dans ELK Stack ou DataDog.


::: qframe.infrastructure.observability.logging
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

- `LogLevel`
- `LogContext`
- `StructuredLogger`
- `JsonFormatter`
- `ConsoleFormatter`
- `PerformanceTimer`
- `LoggerFactory`

