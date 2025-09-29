# qframe.examples.config_usage_example


Example: Utilisation du système de configuration multi-environnement
===================================================================

Exemple d'utilisation du nouveau système de configuration avec:
- Chargement par environnement
- Feature flags
- Secrets sécurisés
- Variables d'environnement


::: qframe.examples.config_usage_example
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

### Fonctions

- `example_basic_config_loading`
- `example_environment_specific_config`
- `example_feature_flags`
- `example_secrets_handling`
- `example_environment_variables_override`
- `example_global_configuration_manager`
- `example_config_validation`
- `example_configuration_differences`

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
