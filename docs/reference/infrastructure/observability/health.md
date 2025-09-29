# qframe.infrastructure.observability.health


Infrastructure Layer: Health Monitoring System
=============================================

Système de health checks et monitoring de la santé des services.
Circuit breakers, readiness/liveness probes, et alerting.


::: qframe.infrastructure.observability.health
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

- `HealthStatus`
- `ComponentType`
- `CircuitBreakerState`
- `HealthCheckResult`
- `ComponentHealth`
- `HealthCheck`
- `DatabaseHealthCheck`
- `MarketDataHealthCheck`
- `BrokerHealthCheck`
- `CircuitBreaker`
- `HealthMonitor`
- `ReadinessProbe`
- `LivenessProbe`

### Fonctions

- `get_health_monitor`

