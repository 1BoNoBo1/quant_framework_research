# qframe.infrastructure.observability.alerting


Infrastructure Layer: Intelligent Alerting System
================================================

Système d'alerting intelligent avec ML pour détecter les anomalies,
groupement d'alertes, et suppression de bruit.


::: qframe.infrastructure.observability.alerting
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

- `AlertSeverity`
- `AlertCategory`
- `AlertStatus`
- `Alert`
- `AlertRule`
- `AlertChannel`
- `LogChannel`
- `EmailChannel`
- `SlackChannel`
- `AnomalyDetector`
- `AlertCorrelator`
- `AlertManager`

### Fonctions

- `get_alert_manager`

