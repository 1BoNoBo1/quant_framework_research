# 🏗️ QFrame Research Platform - Architecture Guide

**Version**: 1.0
**Date**: 28 Septembre 2025
**Status**: ✅ **VALIDÉ & OPÉRATIONNEL**

---

## 🎯 **Vue d'Ensemble**

Le **QFrame Research Platform** étend QFrame Core avec une infrastructure de recherche quantitative distribuée de niveau entreprise, permettant :

- 🔬 **Recherche collaborative** avec JupyterHub multi-user
- 🚀 **Computing distribué** avec Dask/Ray scaling automatique
- 💾 **Data Lake intelligent** multi-backend (S3/MinIO/Local)
- 📊 **Experiment tracking** professionnel avec MLflow
- 🎯 **Analytics avancées** avec métriques sophistiquées
- 🐳 **Infrastructure containerisée** production-ready

---

## 🏗️ **Architecture en Couches**

```
┌─────────────────────────────────────────────────────────────┐
│                    🔬 RESEARCH PLATFORM                     │
├─────────────────────────────────────────────────────────────┤
│  JupyterHub  │  MLflow   │  Superset │    Optuna            │
│  Multi-User  │ Tracking  │Analytics  │ Optimization         │
├─────────────────────────────────────────────────────────────┤
│      📊 COMPUTE LAYER                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Dask Cluster   │  │   Ray Cluster   │  │  Sequential  │ │
│  │ Scheduler+Works │  │  Head+Workers   │  │   Fallback   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│      💾 DATA LAYER                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Data Lake     │  │  Feature Store  │  │ Data Catalog │ │
│  │ S3/MinIO/Local  │  │   Centralized   │  │  Metadata    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│      🔗 INTEGRATION LAYER                                   │
│            QFrame Core ←→ Research Platform                 │
├─────────────────────────────────────────────────────────────┤
│                    🏗️ QFRAME CORE                          │
│   Container DI  │ Strategies │ Services │  Infrastructure   │
│   Configuration │ Research   │Portfolio │  Data Providers   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 **Composants Research Platform**

### **1. Data Lake Infrastructure**

#### **Multi-Backend Storage**
```python
from qframe.research.data_lake import DataLakeStorage, StorageMetadata

# Local Development
storage = LocalFileStorage("/data/lake")

# Production S3
storage = S3Storage(
    bucket_name="qframe-prod",
    aws_access_key_id="AKIA...",
    region_name="us-east-1"
)

# Self-hosted MinIO
storage = MinIOStorage(
    endpoint="minio.company.com:9000",
    access_key="admin",
    secret_key="password",
    bucket_name="qframe-data"
)
```

#### **Data Catalog & Metadata**
```python
from qframe.research.data_lake import DataCatalog, DatasetMetadata

catalog = DataCatalog(db_url="postgresql://user:pass@db:5432/catalog")

# Enregistrement automatique avec métadonnées
metadata = await catalog.register_dataset(
    name="BTC_1H_2024",
    dataset_type=DatasetType.RAW_MARKET_DATA,
    quality_level=DataQuality.VALIDATED,
    schema={"open": "float64", "high": "float64", ...},
    tags={"exchange": "binance", "symbol": "BTC/USDT"}
)
```

#### **Feature Store Centralisé**
```python
from qframe.research.data_lake import FeatureStore

feature_store = FeatureStore(storage, catalog)

# Stockage features avec versioning
await feature_store.store_features(
    feature_group="technical_indicators",
    features_df=df_with_features,
    version="v1.2",
    description="RSI, MACD, Bollinger Bands"
)

# Récupération features par version
features = await feature_store.get_features(
    feature_group="technical_indicators",
    version="latest",
    symbols=["BTC/USDT", "ETH/USDT"]
)
```

### **2. Distributed Computing Engine**

#### **Auto-Detection & Fallback**
```python
from qframe.research.backtesting import DistributedBacktestEngine

# Auto-détection Dask → Ray → Sequential
engine = DistributedBacktestEngine(
    compute_backend="dask",  # Préféré
    max_workers=8
)

# Si Dask indisponible:
# ⚠️ Dask not available, falling back to sequential processing
# ✅ Engine créé en mode sequential
```

#### **Distributed Backtesting**
```python
# Multi-stratégies avec parameter grids
results = await engine.run_distributed_backtest(
    strategies=["adaptive_mean_reversion", "dmn_lstm", "rl_alpha"],
    datasets=[btc_data, eth_data, ada_data],
    parameter_grids={
        "adaptive_mean_reversion": {
            "lookback_short": [10, 20, 30],
            "lookback_long": [50, 100, 200],
            "z_entry_base": [1.0, 1.5, 2.0]
        },
        "dmn_lstm": {
            "window_size": [32, 64, 128],
            "hidden_size": [32, 64, 128],
            "learning_rate": [0.001, 0.01]
        }
    },
    n_splits=5,  # Cross-validation
    initial_capital=100000.0
)

# Résultats automatiquement agrégés
summary_df = engine.get_performance_summary(results)
```

#### **Task Distribution**
```python
# Dask Implementation
def _run_dask_backtest(self, tasks):
    futures = []
    for task in tasks:
        future = self.dask_client.submit(self._run_single_backtest, task)
        futures.append(future)

    results = await asyncio.gather(*[
        self._wait_for_dask_future(future) for future in futures
    ])
    return results

# Ray Implementation
@ray.remote
def _run_single_backtest_ray(task):
    engine = DistributedBacktestEngine(compute_backend="ray")
    return engine._run_single_backtest(task)
```

### **3. Advanced Performance Analytics**

#### **Métriques Sophistiquées**
```python
from qframe.research.backtesting import AdvancedPerformanceAnalyzer

analyzer = AdvancedPerformanceAnalyzer()
analysis = analyzer.analyze_comprehensive(backtest_result)

# Métriques disponibles:
metrics = {
    'basic_metrics': {
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'sortino_ratio': 1.8,
        'calmar_ratio': 0.9
    },
    'risk_metrics': {
        'var_95': -0.025,
        'cvar_95': -0.035,
        'max_consecutive_losses': 5,
        'skewness': -0.2,
        'kurtosis': 3.1
    },
    'drawdown_analysis': {
        'max_drawdown': -0.08,
        'avg_drawdown': -0.02,
        'time_in_drawdown': 0.25,
        'num_drawdowns': 12
    }
}
```

#### **Rolling Metrics & Confidence Intervals**
```python
# Rolling performance windows
rolling_metrics = analysis['rolling_metrics']
rolling_sharpe = rolling_metrics['rolling_sharpe']  # 30-day rolling
rolling_volatility = rolling_metrics['rolling_volatility']

# Monte Carlo confidence intervals
confidence_intervals = {
    "95%": (0.08, 0.22),  # 95% confident returns between 8-22%
    "75%": (0.12, 0.18)   # 75% confident returns between 12-18%
}
```

### **4. Integration Layer**

#### **Transparent QFrame Bridge**
```python
from qframe.research.integration_layer import create_research_integration

# Connexion automatique à QFrame Core existant
integration = create_research_integration(use_minio=False)

# Utilise automatiquement:
# - Container DI de QFrame Core
# - Strategies déjà enregistrées
# - Data providers configurés
# - Services de backtesting existants

# Status complet
status = integration.get_integration_status()
# {
#     'qframe_core_available': True,
#     'container_initialized': True,
#     'data_lake_backend': 'local',
#     'feature_store_ready': True,
#     'strategies_discovered': 4
# }
```

#### **Research Features Pipeline**
```python
# Enrichissement données avec features research
enhanced_data = await integration.compute_research_features(
    data=market_data,
    include_symbolic=True,  # Opérateurs symboliques QFrame
    include_ml=True,        # Features ML avancées
    feature_groups=['technical', 'statistical', 'regime']
)

# Combine automatiquement:
# - Features QFrame Core (SymbolicFeatureProcessor)
# - Features Research Platform spécialisées
# - Metadata tracking pour reproducibilité
```

---

## 🐳 **Infrastructure Docker**

### **Services Architecture**

```yaml
# docker-compose.research.yml
services:
  # 🔬 Research Environment
  jupyterhub:         # Multi-user notebooks (port 8888)
  research-notebook:  # Custom QFrame notebook image

  # 📊 Experiment Tracking
  mlflow:            # Tracking server (port 5000)
  postgres-mlflow:   # MLflow metadata DB

  # 🚀 Distributed Computing
  dask-scheduler:    # Dask coordinator (port 8786)
  dask-worker:       # Dask workers (replicas: 2)
  ray-head:          # Ray head node (port 8265)
  ray-worker:        # Ray workers (replicas: 2)

  # 💾 Data Storage
  minio:             # S3-compatible storage (port 9000)
  timescaledb:       # Time-series DB (port 5433)
  elasticsearch:     # Search & analytics (port 9200)

  # 📈 Analytics & Visualization
  superset:          # BI dashboards (port 8088)
  optuna-dashboard:  # Hyperparameter optimization (port 8080)

  # 🔄 Streaming & Messaging
  kafka:             # Event streaming (port 9092)
  zookeeper:         # Kafka coordination
  redis:             # Caching & sessions (port 6380)
```

### **Resource Management**
```yaml
# Auto-scaling configuration
dask-worker:
  deploy:
    replicas: 2
    resources:
      limits:
        cpus: '2'
        memory: 4G

ray-worker:
  deploy:
    replicas: 2
    resources:
      limits:
        cpus: '4'
        memory: 8G
```

### **Network & Security**
```yaml
networks:
  qframe-research:
    driver: bridge

volumes:
  # Persistent storage
  minio-data:
  postgres-mlflow-data:
  timescale-data:
  elasticsearch-data:
```

---

## 🔄 **Workflows de Recherche**

### **1. Strategy Development Workflow**

```python
# 1. Data Ingestion
from qframe.research.data_lake import DataIngestionPipeline

pipeline = DataIngestionPipeline(storage, catalog)
await pipeline.ingest_market_data(
    provider="binance",
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframe="1h",
    start_date="2024-01-01"
)

# 2. Feature Engineering
features = await integration.compute_research_features(data)

# 3. Strategy Development (Jupyter)
# - Interactive notebook development
# - Real-time visualization
# - Iterative testing

# 4. Distributed Backtesting
results = await engine.run_distributed_backtest(
    strategies=new_strategies,
    datasets=test_datasets,
    parameter_grids=optimization_grids
)

# 5. MLflow Tracking
mlflow.log_metrics(performance_metrics)
mlflow.log_artifacts(["strategy_code.py", "results.parquet"])
```

### **2. Research Collaboration Workflow**

```bash
# 1. Team Setup
docker-compose -f docker-compose.research.yml up -d

# 2. JupyterHub Access (http://localhost:8888)
# - Multi-user authentication
# - Shared notebooks workspace
# - Git integration

# 3. Experiment Tracking (http://localhost:5000)
# - MLflow central tracking
# - Experiment comparison
# - Model registry

# 4. Resource Monitoring
# - Dask Dashboard (http://localhost:8787)
# - Ray Dashboard (http://localhost:8265)
# - System metrics
```

### **3. Production Deployment Workflow**

```python
# 1. Model Selection from MLflow
best_model = mlflow.search_runs(
    experiment_id="strategy_optimization",
    order_by=["metrics.sharpe_ratio DESC"],
    max_results=1
)

# 2. Strategy Registration in QFrame Core
container.register_singleton(
    Strategy,
    production_strategy,
    name="production_v1"
)

# 3. Live Trading Deployment
# - QFrame Core production pipeline
# - Real-time monitoring
# - Risk management integration
```

---

## ⚡ **Performance & Scaling**

### **Compute Performance**

| **Backend** | **Tasks/Min** | **Memory** | **CPU** | **Scaling** |
|-------------|---------------|------------|---------|-------------|
| Sequential  | 1-5           | Low        | Single  | None        |
| Dask        | 50-200        | Medium     | Multi   | Horizontal  |
| Ray         | 100-500       | High       | Multi   | Auto        |

### **Storage Performance**

| **Backend** | **Throughput** | **Latency** | **Scalability** |
|-------------|----------------|-------------|-----------------|
| Local       | ~100 MB/s      | <10ms       | Single node     |
| MinIO       | ~500 MB/s      | ~50ms       | Multi-node      |
| AWS S3      | ~1000 MB/s     | ~100ms      | Unlimited       |

### **Auto-Scaling Configuration**

```python
# Dask adaptive scaling
from dask.distributed import Client
from dask_kubernetes import KubeCluster

cluster = KubeCluster.from_yaml('dask-cluster.yaml')
cluster.adapt(minimum=2, maximum=20)  # Auto-scale 2-20 workers

# Ray autoscaler
ray.init(
    address="ray://head:10001",
    runtime_env={
        "pip": ["qframe", "pandas", "numpy"]
    }
)
```

---

## 🔒 **Sécurité & Bonnes Pratiques**

### **Data Security**

```python
# Encryption at rest
storage = S3Storage(
    bucket_name="qframe-encrypted",
    encryption="AES256",
    kms_key_id="arn:aws:kms:us-east-1:123456789012:key/..."
)

# Access control
catalog = DataCatalog(
    db_url="postgresql://...",
    auth_provider="oauth2",
    permissions={"read": ["research_team"], "write": ["admin"]}
)
```

### **Compute Security**

```yaml
# Docker security
jupyterhub:
  user: 1000:1000  # Non-root
  read_only: true
  cap_drop: [ALL]
  security_opt: [no-new-privileges]
```

### **Network Security**

```yaml
# Internal communication only
networks:
  qframe-research:
    internal: true

# Reverse proxy for external access
nginx:
  ports: [80, 443]
  ssl_certificate: /certs/qframe.crt
```

---

## 📊 **Monitoring & Observability**

### **System Metrics**

```python
# Prometheus metrics endpoint
from prometheus_client import CollectorRegistry, Counter, Histogram

BACKTESTS_TOTAL = Counter('backtests_total', 'Total backtests run')
BACKTEST_DURATION = Histogram('backtest_duration_seconds', 'Backtest execution time')

# Custom QFrame metrics
STRATEGY_PERFORMANCE = Histogram(
    'strategy_sharpe_ratio',
    'Strategy Sharpe ratios',
    buckets=[0, 0.5, 1.0, 1.5, 2.0, float('inf')]
)
```

### **Application Logs**

```python
# Structured logging with correlation IDs
import structlog

logger = structlog.get_logger("qframe.research")

logger.info(
    "Distributed backtest started",
    correlation_id=correlation_id,
    strategies=strategy_names,
    n_tasks=len(tasks),
    compute_backend=self.compute_backend
)
```

### **Health Checks**

```python
# Docker health checks
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

---

## 🚀 **Déploiement Production**

### **Kubernetes Manifests** (À venir Phase 8)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qframe-research-platform
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: qframe/research:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### **CI/CD Pipeline** (À venir Phase 8)

```yaml
# .github/workflows/research-platform.yml
name: QFrame Research Platform
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Validate Research Platform
      run: poetry run python scripts/validate_installation.py

  deploy:
    if: github.ref == 'refs/heads/main'
    needs: validate
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to Kubernetes
      run: kubectl apply -f k8s/
```

---

## 🎯 **Roadmap & Extensions**

### **Phase 8 - Production Deployment**
- [ ] Kubernetes manifests
- [ ] Auto-scaling policies
- [ ] CI/CD pipeline GitHub Actions
- [ ] Production monitoring stack

### **Phase 9 - Advanced Research**
- [ ] AutoML strategy generation
- [ ] Ensemble methods
- [ ] Market regime detection
- [ ] Real-time feature streaming

### **Phase 10 - Ecosystem**
- [ ] Plugin marketplace
- [ ] Third-party integrations
- [ ] Academic partnerships
- [ ] Open source community

---

## 📚 **Ressources & Documentation**

### **Liens Utiles**
- [Phase 7 Validation Report](PHASE7_RESEARCH_PLATFORM_VALIDATION_REPORT.md)
- [QFrame Core Guide](CLAUDE.md)
- [Docker Research Stack](docker-compose.research.yml)
- [Validation Script](scripts/validate_installation.py)

### **Support & Communauté**
- **Issues**: [GitHub Issues](https://github.com/qframe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/qframe/discussions)
- **Documentation**: [Docs Site](https://docs.qframe.io)

---

**QFrame Research Platform** - *Enterprise quantitative research infrastructure for financial independence* 🚀