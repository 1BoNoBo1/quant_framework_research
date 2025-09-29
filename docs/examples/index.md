# 📚 Exemples Complets

Exemples pratiques d'utilisation du framework QFrame pour démarrer rapidement.

## Exemple Minimal

```python
# examples/minimal_example.py
from qframe.core.container import get_container
from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy

# Configuration automatique
container = get_container()
strategy = container.resolve(AdaptiveMeanReversionStrategy)

# Génération signaux
import pandas as pd
data = pd.read_csv('market_data.csv')
signals = strategy.generate_signals(data)
print(f"Generated {len(signals)} signals")
```

## Exemple Avancé

```python
# examples/enhanced_example.py
# Simulation multi-stratégies avec 262+ ordres
from qframe.core.container import get_container
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository

# Setup complet
container = get_container()
portfolio_repo = container.resolve(MemoryPortfolioRepository)

# Création portfolios
portfolios = [
    portfolio_repo.create(Portfolio(
        id=f"portfolio-{i}",
        name=f"Strategy {i}",
        initial_capital=Decimal("10000.00")
    )) for i in range(3)
]

# Exécution stratégies
strategies = [
    container.resolve(AdaptiveMeanReversionStrategy),
    container.resolve(DMNLSTMStrategy),
    container.resolve(RLAlphaStrategy)
]

# Simulation et analytics
for portfolio, strategy in zip(portfolios, strategies):
    signals = strategy.generate_signals(sample_data)
    orders = execute_signals(portfolio, signals)
    stats = portfolio_repo.get_portfolio_statistics(portfolio.id)
    print(f"Portfolio {portfolio.name}: {stats}")
```

## Exemples Spécialisés

### Backtesting Distribué
```python
# Research Platform
from qframe.research.backtesting import DistributedBacktestEngine

engine = DistributedBacktestEngine(compute_backend="dask")
results = await engine.run_distributed_backtest(
    strategies=["mean_reversion", "dmn_lstm"],
    parameter_grids={"lookback": [10, 20, 30]}
)
```

### Interface Web
```bash
# Dashboard Streamlit
cd qframe/ui && ./deploy-simple.sh test
# Interface sur http://localhost:8502
```

### CLI Avancée
```bash
# Information framework
poetry run python qframe_cli.py info

# Liste stratégies
poetry run python qframe_cli.py strategies

# Backtest
poetry run python qframe_cli.py backtest --strategy mean_reversion
```

## Validation Installation

```bash
# Framework de base
poetry run python demo_framework.py

# Exemple minimal
poetry run python examples/minimal_example.py

# Exemple complet
poetry run python examples/enhanced_example.py

# Tests
poetry run pytest tests/unit/ -v
```

## Voir aussi

- [Quickstart](../getting-started/quickstart.md) - Démarrage rapide
- [Tutoriels](tutorials.md) - Guides détaillés
- [Architecture](../architecture/overview.md) - Concepts techniques