# qframe.infrastructure.observability.tracing


Infrastructure Layer: Distributed Tracing System
===============================================

Système de tracing distribué pour suivre les flux à travers les services.
Compatible avec OpenTelemetry, Jaeger, Zipkin, etc.


::: qframe.infrastructure.observability.tracing
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

- `SpanKind`
- `SpanStatus`
- `SpanContext`
- `Span`
- `SpanEvent`
- `SpanLink`
- `Trace`
- `Tracer`
- `SpanContext`
- `NoOpSpan`
- `TradingTracer`

### Fonctions

- `get_tracer`
- `trace`

