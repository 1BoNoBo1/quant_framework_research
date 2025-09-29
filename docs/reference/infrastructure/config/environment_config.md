# qframe.infrastructure.config.environment_config


Environment Configuration System
===============================

Système de configuration avancé pour multi-environnement avec:
- Support fichiers YAML/JSON par environnement
- Variables d'environnement sécurisées
- Feature flags dynamiques
- Configuration secrets


::: qframe.infrastructure.config.environment_config
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

- `EnvironmentType`
- `FeatureFlag`
- `SecretConfig`
- `ConfigurationManager`

### Fonctions

- `get_configuration_manager`
- `reload_configuration`
- `get_current_config`
- `is_feature_enabled`

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
