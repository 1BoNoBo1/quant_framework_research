# qframe.ui.streamlit_app.components.backtesting.backtest_configurator


Backtest Configurator - Composant pour configuration de backtests


::: qframe.ui.streamlit_app.components.backtesting.backtest_configurator
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

- `BacktestConfigurator`

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
