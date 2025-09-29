# qframe.strategies.research.adaptive_mean_reversion_config


Configuration for AdaptiveMeanReversionStrategy.

This configuration supports adaptive mean reversion with ML-based regime detection.
The strategy adapts its parameters based on detected market regimes.


::: qframe.strategies.research.adaptive_mean_reversion_config
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

- `AdaptiveMeanReversionConfig`

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
