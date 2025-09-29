# qframe.infrastructure.observability.metrics


Infrastructure Layer: Metrics Collection System
==============================================

Système de collecte de métriques pour monitoring temps réel.
Compatible avec Prometheus, StatsD, DataDog, etc.


::: qframe.infrastructure.observability.metrics
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

- `MetricType`
- `MetricUnit`
- `MetricDefinition`
- `MetricPoint`
- `MetricStorage`
- `MetricsCollector`
- `TimeMeasurement`
- `BusinessMetrics`

### Fonctions

- `get_metrics_collector`
- `get_business_metrics`

