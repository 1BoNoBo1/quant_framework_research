# qframe.core.config


Configuration centralisée avec Pydantic
=====================================

Système de configuration type-safe et validé pour tout le framework.
Supporte les environnements multiples et la validation automatique.


::: qframe.core.config
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

- `Environment`
- `LogLevel`
- `DatabaseConfig`
- `RedisConfig`
- `MLFlowConfig`
- `DataProviderConfig`
- `StrategyConfig`
- `RiskManagementConfig`
- `BacktestConfig`
- `TradingConfig`
- `AlertConfig`
- `FrameworkConfig`
- `DevelopmentConfig`
- `ProductionConfig`
- `TestingConfig`

### Fonctions

- `get_config`
- `load_config_from_file`
- `update_config`
- `get_config_for_environment`
- `load_environment_config`
- `get_enhanced_config`
- `is_feature_enabled_legacy`

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
