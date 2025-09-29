# qframe.infrastructure.monitoring.metrics_collector


Advanced Metrics Collection and Monitoring for QFrame Enterprise
===============================================================

Système de collecte de métriques et monitoring enterprise avec
intégration Prometheus, alerting, et tableaux de bord temps réel.


::: qframe.infrastructure.monitoring.metrics_collector
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
- `AlertSeverity`
- `MetricDefinition`
- `MetricValue`
- `AlertRule`
- `Alert`
- `MetricStorage`
- `InMemoryMetricStorage`
- `PrometheusAdapter`
- `SystemMetricsCollector`
- `MetricsCollector`
- `TimerContext`

### Fonctions

- `monitored`
- `get_default_collector`
- `configure_monitoring`

