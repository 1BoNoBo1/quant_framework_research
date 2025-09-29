# 🔬 Research Platform Overview

Infrastructure distribuée de recherche quantitative avec Data Lake, computing distribué et MLOps.

## Architecture Research Platform

```
QFrame Research Platform
├── 🏗️ QFrame Core (validé 100%)
│   ├── Container DI + Configuration Pydantic
│   ├── Stratégies : DMN LSTM, Mean Reversion, RL Alpha
│   ├── Feature Engineering : 18+ opérateurs symboliques
│   └── Services : Backtesting, Portfolio, Risk
├── 🔬 Research Layer (6/8 validé)
│   ├── ✅ Data Lake Storage (S3/MinIO/Local)
│   ├── ✅ Data Catalog + Feature Store
│   ├── ✅ Distributed Backtesting (Dask/Ray)
│   ├── ✅ Integration Layer QFrame ↔ Research
│   ├── ✅ Advanced Performance Analytics
│   └── ⚠️ SDK Modules (en développement)
└── 🐳 Infrastructure Docker (validé 100%)
```

## Composants Opérationnels

### Data Lake Infrastructure
```python
from qframe.research.data_lake import DataLakeStorage, FeatureStore

# Multi-backend storage avec fallback
storage = LocalFileStorage("/data/lake")  # ou S3Storage, MinIOStorage
feature_store = FeatureStore(storage)
catalog = DataCatalog(db_url="postgresql://...")
```

### Distributed Backtesting
```python
from qframe.research.backtesting import DistributedBacktestEngine

engine = DistributedBacktestEngine(
    compute_backend="dask",  # ou "ray", "sequential"
    max_workers=4
)

# Backtesting multi-stratégies distribué
results = await engine.run_distributed_backtest(
    strategies=["adaptive_mean_reversion", "dmn_lstm"],
    datasets=[data1, data2],
    parameter_grids={"lookback": [10, 20, 30]},
    n_splits=5
)
```

## Docker Research Stack

### Services Déployés
- ✅ **JupyterHub** : Multi-user research environment
- ✅ **MLflow** : Experiment tracking avec PostgreSQL + S3
- ✅ **Dask Cluster** : Distributed computing
- ✅ **Ray Cluster** : Distributed ML
- ✅ **TimescaleDB** : Time-series data storage
- ✅ **MinIO** : S3-compatible object storage
- ✅ **Elasticsearch** : Search et analytics
- ✅ **Superset** : Data visualization

### Démarrage
```bash
# Stack complète
docker-compose -f docker-compose.research.yml up -d

# Validation
poetry run python scripts/validate_installation.py
```

## Integration Layer

```python
from qframe.research.integration_layer import create_research_integration

integration = create_research_integration(use_minio=False)
status = integration.get_integration_status()
features = await integration.compute_research_features(data)
```

## Performance Analytics

```python
from qframe.research.backtesting import AdvancedPerformanceAnalyzer

analyzer = AdvancedPerformanceAnalyzer()
analysis = analyzer.analyze_comprehensive(backtest_result)

# Métriques avancées : Sortino, Calmar, VaR/CVaR, Skewness, Kurtosis
# Rolling metrics, Drawdown analysis, Confidence intervals
```

## Voir aussi

- [Data Lake](data-lake.md) - Storage et catalog
- [Distributed Computing](distributed.md) - Dask et Ray
- [MLOps Pipeline](mlops.md) - Experiment tracking