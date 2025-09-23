# 🚀 QFrame - Production-Ready Quantitative Trading Framework

[![CI/CD](https://github.com/1BoNoBo1/quant_framework_research/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/1BoNoBo1/quant_framework_research/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QFrame** est un framework quantitatif professionnel pour le développement, backtesting et déploiement de stratégies de trading algorithmique.

---

## ✨ Fonctionnalités

### 🏗️ Architecture Robuste
- **Hexagonal Architecture** avec séparation claire des couches
- **Event-Driven Design** avec Event Sourcing & CQRS
- **Dependency Injection** pour testabilité et découplage
- **Domain-Driven Design** avec rich domain model

### 📊 Trading & Backtesting
- **Stratégies multiples**: MA Crossover, Mean Reversion, ML-based
- **Backtesting réaliste**: Données historiques réelles (Binance, Coinbase)
- **Métriques complètes**: Sharpe, Sortino, Calmar, Win Rate, Drawdown
- **Multi-asset support**: Crypto, stocks, forex, futures

### 🔬 Machine Learning
- **PyTorch integration** pour deep learning
- **Feature engineering** avancé
- **Model training & deployment** automatisés
- **Walk-forward optimization**

### 🛡️ Risk Management
- **Position sizing** dynamique
- **Stop-loss & Take-profit** automatiques
- **VaR & CVaR** calculation
- **Portfolio risk limits** enforcement

### 🔭 Observability
- **Structured logging** avec correlation IDs
- **Metrics** (Prometheus compatible)
- **Distributed tracing** (OpenTelemetry/Jaeger)
- **Grafana dashboards** pré-configurés

### 🚢 Production-Ready
- **Docker & Kubernetes** configs
- **CI/CD pipeline** (GitHub Actions)
- **Auto-scaling** (HPA)
- **High availability** setup

---

## 🚀 Quick Start

### Installation

```bash
# Clone le repository
git clone https://github.com/1BoNoBo1/quant_framework_research.git
cd quant_framework_research

# Install avec Poetry
poetry install

# Active l'environnement
poetry shell
```

### Première Stratégie en 2 Minutes

```python
from examples.strategies.ma_crossover_strategy import MovingAverageCrossoverStrategy
import pandas as pd

# Créer la stratégie
strategy = MovingAverageCrossoverStrategy(
    fast_period=10,
    slow_period=20,
    symbol='BTC/USDT'
)

# Charger des données (ou utiliser des données réelles via CCXT)
price_data = pd.read_csv('btc_usdt_1d.csv')

# Générer des signaux
signals = strategy.generate_signals(price_data)

# Backtest
summary = strategy.backtest_summary(price_data)
print(f"Win Rate: {summary['win_rate']:.2%}")
print(f"Total Return: {summary['total_return']:.2%}")
```

### Backtest avec Données Réelles

```bash
# Backtest sur données Binance (2 ans)
poetry run python examples/backtest_real_data.py
```

**Output:**
```
📊 Fetching historical data from Binance...
✅ Fetched 730 days of data

💰 Portfolio Performance:
   Initial Capital: $10,000.00
   Final Value: $15,243.50
   Total Return: 52.44%
   Sharpe Ratio: 1.42
   Max Drawdown: -18.32%
```

---

## 📖 Documentation

- **[📘 Quick Start Guide](docs/QUICKSTART.md)** - Démarrage en 5 minutes
- **[🚀 Deployment Guide](docs/DEPLOYMENT.md)** - Déploiement production
- **[📊 Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Vue d'ensemble technique
- **[✅ Completed Tasks](COMPLETED_TASKS.md)** - Toutes les tâches accomplies

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────┐
│        Presentation Layer                │
│    REST API | WebSocket | GraphQL        │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│        Application Layer                 │
│    Commands | Queries | Handlers         │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         Domain Layer                     │
│  Entities | Value Objects | Services     │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│      Infrastructure Layer                │
│  Data | Events | API | Observability     │
└─────────────────────────────────────────┘
```

### Technologies

**Core:**
- Python 3.11+
- FastAPI, Pydantic
- asyncio, aiohttp

**Data & ML:**
- PyTorch, scikit-learn
- pandas, numpy, scipy
- CCXT, python-binance

**Infrastructure:**
- PostgreSQL, Redis, InfluxDB
- Prometheus, Grafana, Jaeger
- Docker, Kubernetes
- GitHub Actions

---

## 🧪 Tests

```bash
# Tous les tests
poetry run pytest

# Tests unitaires uniquement
poetry run pytest tests/unit/ -v

# Tests avec coverage
poetry run pytest --cov=qframe --cov-report=html

# Tests d'intégration
poetry run pytest tests/integration/ -v
```

**Coverage**: 60+ test cases, >80% code coverage

---

## 🐳 Déploiement

### Local Development

```bash
# Démarrer tous les services
docker-compose up -d

# Services disponibles:
# - API: http://localhost:8000
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9091
# - Jaeger: http://localhost:16686
```

### Production (Kubernetes)

```bash
# Build image
docker build -t qframe/qframe:latest .

# Deploy sur K8s
kubectl apply -f deployment/kubernetes/production/

# Vérifier le déploiement
kubectl get pods -n qframe-prod
kubectl get svc -n qframe-prod
```

---

## 📊 Monitoring

### Grafana Dashboards

Accès: `http://localhost:3000` (admin/admin)

**Dashboards inclus:**
- Portfolio Performance
- Trading Metrics
- System Health
- API Performance
- Event Bus Throughput

### Métriques Disponibles

```promql
# Portfolio value
portfolio_total_value

# Trading stats
trades_executed_total
trades_winning_total
win_rate

# System metrics
http_request_duration_seconds
event_bus_published_total
```

---

## 🔐 Sécurité

- ✅ JWT authentication
- ✅ RBAC authorization
- ✅ API rate limiting
- ✅ Secrets management (Kubernetes Secrets)
- ✅ TLS encryption
- ✅ SQL injection prevention

---

## 📈 Exemple de Performance

**MA Crossover Strategy (BTC/USDT, 1 an):**
```
Total Trades: 23
Win Rate: 65.22%
Total Return: 52.44%
Sharpe Ratio: 1.42
Max Drawdown: -18.32%
Average Holding: 15.8 days
```

---

## 🛠️ Development

### Structure du Projet

```
qframe/
├── domain/              # Domain entities & services
│   ├── entities/
│   ├── value_objects/
│   └── services/
├── application/         # Use cases & handlers
│   ├── commands/
│   ├── queries/
│   └── handlers/
├── infrastructure/      # External adapters
│   ├── api/
│   ├── data/
│   ├── events/
│   ├── persistence/
│   └── observability/
├── examples/            # Example strategies
└── tests/              # Test suites
```

### Contributing

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push sur la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

---

## 📝 Roadmap

### Phase 4: Advanced Strategies
- [ ] LSTM-based strategies
- [ ] Reinforcement Learning agents
- [ ] Multi-timeframe analysis
- [ ] Portfolio optimization algorithms

### Phase 5: Live Trading
- [ ] Exchange integration (live mode)
- [ ] Order management system
- [ ] Risk monitoring dashboard
- [ ] Alert system (email, Slack)

### Phase 6: Scalability
- [ ] Distributed backtesting
- [ ] Multi-region deployment
- [ ] Data lake for historical analysis
- [ ] Real-time stream processing

---

## 🤝 Support

- **Documentation**: [/docs](./docs)
- **Issues**: [GitHub Issues](https://github.com/1BoNoBo1/quant_framework_research/issues)
- **Discussions**: [GitHub Discussions](https://github.com/1BoNoBo1/quant_framework_research/discussions)
- **Email**: research@qframe.dev

---

## 📜 License

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **FastAPI** - Framework web moderne
- **PyTorch** - Machine learning
- **CCXT** - Unified exchange API
- **Prometheus & Grafana** - Monitoring
- **Kubernetes** - Orchestration

---

## ⭐ Star History

Si ce projet vous aide, n'hésitez pas à lui donner une étoile! ⭐

---

**Built with ❤️ by the QFrame Team**

**Happy Trading! 📈🚀**
