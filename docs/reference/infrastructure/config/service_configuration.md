# qframe.infrastructure.config.service_configuration


Service Configuration for Dependency Injection
==============================================

Configuration centralisée de tous les services pour l'architecture hexagonale.
Intègre les repositories, services domain, et adapters infrastructure.


::: qframe.infrastructure.config.service_configuration
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

- `ServiceConfiguration`
- `ServiceModule`

### Fonctions

- `create_trading_module`
- `create_data_module`
- `create_risk_module`
- `configure_production_services`
- `configure_testing_services`
- `get_service_statistics`

## Exemples d'usage


```python
from qframe.core.config import get_config, FrameworkConfig

# Configuration automatique
config = get_config()

# Configuration personnalisée
config = FrameworkConfig(
    environment=Environment.DEVELOPMENT,
    database=DatabaseConfig(url="...")
)
```
