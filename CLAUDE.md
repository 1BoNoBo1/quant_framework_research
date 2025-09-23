# 📋 CLAUDE.md - Guide pour Instances Claude Futures

> **Instructions complètes pour la compréhension et le développement du framework QFrame**

## 🎯 Mission & Contexte

### Objectif Principal
**Développer un framework quantitatif professionnel pour l'autonomie financière** en combinant :
- Recherche sophistiquée (DMN LSTM, Mean Reversion, Funding Arbitrage, RL Alpha Generation)
- Architecture technique moderne (DI, interfaces propres, configuration type-safe)
- Pipeline production-ready (backtesting, trading live, monitoring)

### Historique du Projet
Le projet a évolué depuis un framework initial (`quant_framework_base`) vers une architecture moderne (`quant_framework_research`) en préservant entièrement les recherches existantes tout en appliquant les meilleures pratiques de développement.

## 🏗️ Architecture Technique

### Principes Fondamentaux

1. **Architecture Hexagonale** : Séparation claire domaine métier / infrastructure
2. **Dependency Injection** : Container IoC avec lifecycles (singleton, transient, scoped)
3. **Interfaces Protocol** : Contrats Python modernes avec duck typing
4. **Configuration Pydantic** : Type-safe avec validation et environnements multiples
5. **Tests Complets** : Suite pytest avec mocks et fixtures

### Structure du Code

```
qframe/
├── core/                   # Coeur du framework
│   ├── interfaces.py       # Protocols et contrats de base
│   ├── container.py        # DI Container avec thread safety
│   └── config.py          # Configuration Pydantic centralisée
├── strategies/research/    # Stratégies de recherche migrées
├── features/              # Feature engineering avec opérateurs symboliques
├── data/                  # Data providers (Binance, YFinance, etc.)
├── execution/             # Order execution et portfolio management
├── risk/                  # Risk management et position sizing
└── apps/                  # CLI et applications web
```

## 🧠 Stratégies de Recherche

### 1. DMN LSTM Strategy (`dmn_lstm_strategy.py`)
**Architecture** : Deep Market Networks avec LSTM + Attention optionnelle
**Features** :
- Model PyTorch avec 64+ hidden units
- Dataset temporel avec sliding windows
- Entraînement avec TimeSeriesSplit
- Prédiction de returns futurs avec activation Tanh

**Configuration** :
```python
DMNConfig(
    window_size=64,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    use_attention=False,
    learning_rate=0.001,
    signal_threshold=0.1
)
```

### 2. Mean Reversion Strategy (`mean_reversion_strategy.py`)
**Logique** : Mean reversion adaptatif avec détection de régimes
**Features** :
- Calcul z-score avec seuils adaptatifs
- Détection régimes volatilité (low_vol, normal, high_vol)
- Optimisation ML des seuils d'entrée/sortie
- Position sizing Kelly

**Paramètres** :
```python
MeanReversionConfig(
    lookback_short=10,
    lookback_long=50,
    z_entry_base=1.0,
    z_exit_base=0.2,
    regime_window=252,
    use_ml_optimization=True
)
```

### 3. Funding Arbitrage Strategy (`funding_arbitrage_strategy.py`)
**Méthode** : Arbitrage de taux de financement avec prédiction ML
**Sophistication** :
- Collecte funding rates multi-exchanges
- Prédiction ML (Random Forest) des taux futurs
- Calcul spreads et détection opportunités
- Gestion risque de contrepartie

### 4. RL Alpha Strategy (`rl_alpha_strategy.py`)
**Innovation** : Génération automatique d'alphas via Reinforcement Learning
**Basé sur** : Papier "Synergistic Formulaic Alpha Generation for Quantitative Trading"

**Architecture RL** :
- **Agent** : PPO (Proximal Policy Optimization)
- **Environnement** : Génération de formules avec 42 actions possibles
- **État** : 50 dimensions (structure formule + stats marché)
- **Reward** : Information Coefficient (IC) avec pénalité complexité

**SearchSpace** :
```python
operators = ["sign", "cs_rank", "product", "scale", "pow_op", "skew", "kurt",
             "ts_rank", "delta", "argmax", "argmin", "cond", "wma", "ema", "mad"]
features = ["open", "high", "low", "close", "volume", "vwap"]
constants = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0]
time_deltas = [5, 10, 20, 30, 40, 50, 60, 120]
```

## 🔬 Opérateurs Symboliques

### Implémentation du Papier de Recherche
Le fichier `symbolic_operators.py` implémente intégralement les opérateurs du papier académique :

**Opérateurs Temporels** :
- `ts_rank(x, t)` : Rang temporel sur t périodes
- `delta(x, t)` : Différence avec t périodes passées
- `argmax/argmin(x, t)` : Index du max/min sur t périodes

**Opérateurs Statistiques** :
- `skew(x, window)` : Asymétrie de distribution
- `kurt(x, window)` : Kurtosis (peakedness)
- `mad(x, window)` : Mean Absolute Deviation

**Opérateurs Cross-Sectionnels** :
- `cs_rank(x)` : Rang cross-sectionnel (simulé avec rolling rank)
- `scale(x)` : Normalisation par somme absolue

**Formules Alpha du Papier** :
- `alpha_006` : `(-1 * Corr(open, volume, 10))`
- `alpha_061` : `Less(CSRank((vwap - Min(vwap, 16))), CSRank(Corr(vwap, Mean(volume, 180), 17)))`
- `alpha_099` : Formule complexe avec corrélations et rangs

### SymbolicFeatureProcessor
```python
@injectable
class SymbolicFeatureProcessor(FeatureProcessor):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        # Génère 18+ features symboliques avancées

    def get_feature_names(self) -> List[str]:
        # Retourne noms des features générées
```

## 🛠️ Infrastructure Technique

### DI Container (`container.py`)
**Fonctionnalités** :
- Thread-safe avec RLock
- Lifecycles : singleton, transient, scoped
- Injection automatique via annotations de type
- Détection dépendances circulaires
- Auto-registration avec décorateurs

**Usage** :
```python
@injectable
class MyStrategy(BaseStrategy):
    def __init__(self, data_provider: DataProvider, metrics: MetricsCollector): ...

container = get_container()
container.register_singleton(DataProvider, BinanceProvider)
strategy = container.resolve(MyStrategy)  # Injection automatique
```

### Configuration (`config.py`)
**Structure Pydantic** :
```python
class FrameworkConfig(BaseSettings):
    # Métadonnées
    app_name: str = "Quant Framework Research"
    environment: Environment = Environment.DEVELOPMENT

    # Configurations spécialisées
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    mlflow: MLFlowConfig = MLFlowConfig()
    risk_management: RiskManagementConfig = RiskManagementConfig()

    # Stratégies configurées
    strategies: Dict[str, StrategyConfig] = {...}
```

**Environnements** :
- `DevelopmentConfig` : Log DEBUG, base locale
- `ProductionConfig` : Log INFO, sécurité renforcée
- `TestingConfig` : SQLite memory, mocks

### Interfaces (`interfaces.py`)
**Protocols Principaux** :
```python
class Strategy(Protocol):
    def generate_signals(self, data: pd.DataFrame, features: Optional[pd.DataFrame]) -> List[Signal]: ...

class DataProvider(Protocol):
    async def fetch_ohlcv(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame: ...

class FeatureProcessor(Protocol):
    def process(self, data: pd.DataFrame) -> pd.DataFrame: ...
```

## 🧪 Tests & Qualité

### Organisation des Tests
```
tests/
├── conftest.py           # Fixtures partagées (sample_data, mock_container, etc.)
├── unit/                 # Tests unitaires
│   ├── test_container.py     # DI Container complet
│   ├── test_config.py        # Configuration Pydantic
│   └── test_symbolic_operators.py  # Opérateurs symboliques
└── integration/          # Tests d'intégration
```

### Fixtures Importantes
```python
@pytest.fixture
def sample_ohlcv_data():
    # Génère données OHLCV réalistes avec random walk

@pytest.fixture
def mock_container():
    # Container DI mocké pour isolation tests

@pytest.fixture
def mock_metrics_collector():
    # Collecteur de métriques pour validation
```

### Standards Qualité
- **Coverage** : >90% avec pytest-cov
- **Linting** : Black + Ruff + MyPy
- **Type Safety** : Annotations strictes, no `Any`
- **Documentation** : Docstrings complètes

## 🚀 Commandes Essentielles

### Développement
```bash
# Installation
poetry install

# Tests complets
poetry run pytest tests/ -v --cov=qframe

# Qualité code
poetry run black qframe/
poetry run ruff check qframe/
poetry run mypy qframe/

# Tests spécifiques
poetry run pytest tests/unit/test_container.py::TestDIContainer::test_dependency_injection -v
```

### Usage CLI
```bash
# CLI principal
poetry run qframe --help

# Génération alphas RL
poetry run python -m qframe.strategies.research.rl_alpha_strategy --data-path data/BTCUSDT_1h.parquet

# Test opérateurs symboliques
poetry run python -m qframe.features.symbolic_operators
```

## 🎯 Objectifs de Développement

### Priorités Immédiates
1. **Grid Trading Strategy** : Stratégie revenue-generating stable
2. **Freqtrade Integration** : Backend de trading production
3. **Backtesting Engine** : Framework de test historique complet
4. **WebUI Dashboard** : Interface monitoring et contrôle

### Recherche Continue
1. **Ensemble Methods** : Combinaison synergique des alphas
2. **Multi-Asset** : Extension crypto → stocks → forex
3. **Real-Time Pipeline** : Stream processing avec Kafka/Redis
4. **Cloud Native** : Déploiement Kubernetes scalable

### Innovation RL
1. **Alpha Ensemble** : Méta-learning pour combinaison optimale
2. **Market Regime Detection** : Classification automatique des conditions
3. **Risk-Aware RL** : Agents avec contraintes de risque intégrées
4. **Multi-Objective** : Optimisation Pareto (return, Sharpe, drawdown)

## 📚 Ressources de Référence

### Papiers Académiques

#### 🎯 **Référence Principale**
- **[2401.02710v2]** "Synergistic Formulaic Alpha Generation for Quantitative Trading"
  - Implémentation complète des 15 opérateurs symboliques
  - Méthodologie de génération et évaluation (IC, Rank IC)
  - Base théorique du RL Alpha Generator

#### 📚 **Papiers Prometteurs pour Développements Futurs**

**Machine Learning & Deep Learning pour Finance :**
- **"Deep Learning for Multivariate Financial Time Series"** - Chen et al.
  - Architectures LSTM-CNN hybrides pour prédiction multi-asset
  - Mécanismes d'attention temporelle avancés
- **"Attention-based Deep Multiple Instance Learning"** - Ilse et al.
  - Applications aux séries financières avec attention bags
  - Pertinent pour améliorer notre DMN LSTM

**Reinforcement Learning Avancé :**
- **"Deep Reinforcement Learning for Trading"** - Deng et al.
  - Agents multi-objectifs (return + Sharpe + drawdown)
  - Environnements avec contraintes de risque intégrées
- **"Model-Agnostic Meta-Learning for Portfolio Management"** - MAML Finance
  - Méta-apprentissage pour adaptation rapide nouveaux marchés
  - Technique pour ensembles d'alphas auto-adaptatifs

**Alpha Discovery & Feature Engineering :**
- **"Genetic Programming for Financial Feature Engineering"** - Koza et al.
  - Évolution génétique de features alternatives aux opérateurs symboliques
  - Complémentaire à notre approche RL
- **"Information-Theoretic Alpha Discovery"** - Mutual Information approach
  - Sélection de features basée sur théorie de l'information
  - Optimisation de notre pipeline SymbolicFeatureProcessor

**Risk Management Sophistiqué :**
- **"Dynamic Risk Budgeting for Portfolio Management"** - Modern Portfolio Theory++
  - Allocation de risque adaptative en temps réel
  - Extension de notre RiskManagementConfig
- **"Regime-Aware Portfolio Construction"** - Hidden Markov Models
  - Détection de régimes plus sophistiquée que notre mean reversion
  - Applicable à toutes nos stratégies

**Multi-Asset & Cross-Market :**
- **"Cross-Asset Momentum and Contrarian Strategies"** - Cross-market arbitrage
  - Extension de funding arbitrage vers autres asset classes
  - Corrélations crypto-forex-équities
- **"High-Frequency Cross-Exchange Arbitrage"** - Latency arbitrage
  - Évolution de notre funding arbitrage vers HFT
  - Techniques d'optimisation réseau et execution

**Validation & Backtesting :**
- **"Combinatorially Symmetric Cross Validation"** - Purged cross-validation
  - Méthodes de validation pour séries temporelles financières
  - Critical pour éviter le data leakage en backtesting
- **"The Deflated Sharpe Ratio"** - Lopez de Prado
  - Métriques de performance ajustées pour multiple testing
  - Essential pour validation statistique de nos alphas

#### 🔍 **Sources de Veille Continue**
- **arXiv.org** : cs.LG + q-fin.CP + stat.ML
- **SSRN** : Finance et Machine Learning sections
- **Journal of Financial Data Science** - CFA Institute
- **Quantitative Finance Journal** - Taylor & Francis
- **IEEE Transactions on Computational Intelligence and AI in Games** - RL applications

### Documentation Technique
- **Pydantic** : [pydantic-docs.helpmanual.io](https://pydantic-docs.helpmanual.io)
- **PyTorch** : [pytorch.org/docs](https://pytorch.org/docs)
- **Poetry** : [python-poetry.org/docs](https://python-poetry.org/docs)

### Architecture Patterns
- **Hexagonal Architecture** : Ports & Adapters
- **Dependency Injection** : IoC Container patterns
- **CQRS** : Command Query Responsibility Segregation

## ⚠️ Points d'Attention

### Sécurité
- **API Keys** : Toujours via variables d'environnement
- **Secrets** : Jamais en dur dans le code
- **Production** : Validation obligatoire du SECRET_KEY

### Performance
- **Memory** : Attention aux leaks dans les DataFrames PyTorch
- **Compute** : RL training peut être intensif (GPU recommandé)
- **I/O** : Cache Redis pour data providers API rate limits

### Trading Live
- **Testnet First** : Toujours tester en paper trading
- **Position Sizing** : Respecter les limites de risque configurées
- **Error Handling** : Logging complet pour audit et debug

## 🔄 Workflow de Développement

### Nouvelles Fonctionnalités
1. **Design** : Interfaces Protocol first
2. **Implementation** : Classes concrètes avec DI
3. **Tests** : Unitaires + intégration
4. **Integration** : Container registration
5. **Configuration** : Pydantic config schema
6. **Documentation** : Docstrings + README update

### Nouvelles Stratégies
1. **Hériter** : `BaseStrategy` pour interface standard
2. **Configurer** : Dataclass config avec validation
3. **Injecter** : Dependencies via constructor typing
4. **Tester** : Fixtures avec données réalistes
5. **Enregistrer** : Container DI avec lifecycle approprié

---

## 💡 Claude, utilisez ce guide pour :

1. **Comprendre** l'architecture et les choix techniques
2. **Naviguer** le code existant et ses patterns
3. **Développer** en respectant les conventions établies
4. **Tester** avec les fixtures et mocks appropriés
5. **Innover** en préservant la sophistication de recherche existante

Le framework est conçu pour **l'autonomie financière via la recherche quantitative**, en combinant innovation académique et qualité professionnelle. Chaque décision technique sert cet objectif ultime.
- les donnees des marchés crypto seront recuperer soit par un framework soit par ccxt.