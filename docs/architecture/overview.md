# 🏗️ Architecture du Framework QFrame

QFrame implémente une **architecture hexagonale moderne** avec injection de dépendances, configuration type-safe et séparation claire des responsabilités.

## Vue d'Ensemble Architecturale

```mermaid
graph TB
    subgraph "🎯 Presentation Layer"
        CLI[CLI Application]
        WEB[Web Interface]
        API[REST API]
        GQL[GraphQL API]
    end

    subgraph "📱 Application Layer"
        UC[Use Cases]
        CMD[Commands]
        QRY[Queries]
        HDLR[Handlers]
    end

    subgraph "💼 Domain Layer"
        ENT[Entities]
        VO[Value Objects]
        SERV[Domain Services]
        REPO[Repository Interfaces]
    end

    subgraph "🛠️ Infrastructure Layer"
        IMPL[Repository Implementations]
        EXT[External Services]
        CONF[Configuration]
        OBS[Observability]
    end

    subgraph "🏗️ Core Framework"
        DI[DI Container]
        INT[Interfaces/Protocols]
        CFG[Type-Safe Config]
    end

    CLI --> UC
    WEB --> UC
    API --> UC
    GQL --> UC

    UC --> CMD
    UC --> QRY
    CMD --> HDLR
    QRY --> HDLR

    HDLR --> ENT
    HDLR --> SERV
    SERV --> REPO

    REPO -.-> IMPL
    SERV -.-> EXT

    DI --> UC
    DI --> SERV
    DI --> IMPL
    DI --> EXT

    CFG --> DI
    INT --> DI

    style DI fill:#7c4dff,color:#fff
    style ENT fill:#9c27b0,color:#fff
    style UC fill:#4caf50,color:#fff
```

## Principes Architecturaux

### 1. **Dependency Injection (DI)**

Le cœur de QFrame utilise un container DI thread-safe qui gère automatiquement les dépendances :

```python
from qframe.core.container import get_container, injectable

@injectable
class TradingStrategy:
    def __init__(self,
                 data_provider: DataProvider,  # (1)
                 risk_manager: RiskManager,    # (2)
                 metrics: MetricsCollector):   # (3)
        self.data_provider = data_provider
        self.risk_manager = risk_manager
        self.metrics = metrics

# Utilisation
container = get_container()
strategy = container.resolve(TradingStrategy)  # (4)
```

1. **DataProvider** injecté automatiquement
2. **RiskManager** résolu via le container
3. **MetricsCollector** singleton partagé
4. **Résolution automatique** de toutes les dépendances

### 2. **Configuration Type-Safe**

Configuration Pydantic avec validation et environnements multiples :

```python
from qframe.core.config import FrameworkConfig, Environment

class MyStrategyConfig(BaseSettings):
    lookback_window: int = Field(default=20, ge=5, le=200)
    risk_threshold: float = Field(default=0.02, ge=0.001, le=0.1)

    class Config:
        env_prefix = "QFRAME_STRATEGY_"

# Auto-loading depuis variables d'environnement
config = FrameworkConfig()
print(f"Environment: {config.environment}")  # (1)
print(f"Database URL: {config.database.url}")  # (2)
```

1. **Environnement** détecté automatiquement (dev/prod/test)
2. **Validation** automatique des types et contraintes

### 3. **Interfaces Protocol**

Contrats modernes avec duck typing Python :

```python
from typing import Protocol
from qframe.domain.entities import Signal

class Strategy(Protocol):
    def generate_signals(self,
                        data: pd.DataFrame,
                        features: Optional[pd.DataFrame] = None) -> List[Signal]:
        """Génère des signaux de trading."""
        ...

class DataProvider(Protocol):
    async def fetch_ohlcv(self,
                          symbol: str,
                          timeframe: TimeFrame,
                          limit: int = 1000) -> pd.DataFrame:
        """Récupère des données OHLCV."""
        ...
```

## Layers Détaillés

### 🎯 Presentation Layer

**Responsabilité** : Interfaces utilisateur et APIs

=== "Web Interface"

    ```python
    # Streamlit moderne avec session state
    import streamlit as st
    from qframe.ui.components import StrategyDashboard

    dashboard = StrategyDashboard()
    dashboard.render_performance_metrics()
    dashboard.render_portfolio_overview()
    ```

=== "REST API"

    ```python
    # FastAPI avec injection automatique
    from fastapi import FastAPI, Depends
    from qframe.infrastructure.api.dependencies import get_strategy_service

    @app.post("/strategies/{strategy_id}/signals")
    async def generate_signals(
        strategy_id: str,
        service: StrategyService = Depends(get_strategy_service)
    ):
        return await service.generate_signals(strategy_id)
    ```

=== "CLI"

    ```python
    # Typer avec container DI
    import typer
    from qframe.core.container import get_container

    @app.command()
    def backtest(strategy: str = "mean_reversion"):
        container = get_container()
        engine = container.resolve(BacktestEngine)
        results = engine.run(strategy)
        typer.echo(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    ```

### 💼 Domain Layer

**Responsabilité** : Logique métier pure, indépendante de l'infrastructure

```python
# Entités métier
@dataclass
class Portfolio:
    id: str
    name: str
    initial_capital: Decimal
    base_currency: str
    positions: List[Position] = field(default_factory=list)

    def add_position(self, position: Position) -> None:
        """Ajoute une position avec validation métier."""
        if position.quantity <= 0:
            raise ValueError("Quantity must be positive")
        self.positions.append(position)

# Services de domaine
class RiskCalculationService:
    def calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calcule la Value at Risk."""
        return returns.quantile(confidence)

    def calculate_position_size(self,
                               signal_strength: float,
                               account_balance: Decimal,
                               risk_per_trade: float) -> Decimal:
        """Calcule la taille de position selon Kelly Criterion."""
        return account_balance * Decimal(str(risk_per_trade * signal_strength))
```

### 🛠️ Infrastructure Layer

**Responsabilité** : Implémentations concrètes et services externes

=== "Data Providers"

    ```python
    @injectable
    class BinanceProvider:
        def __init__(self, config: ExchangeConfig):
            self.client = Client(config.api_key, config.secret_key)

        async def fetch_ohlcv(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
            klines = await self.client.get_historical_klines(
                symbol, timeframe.value, limit=1000
            )
            return self._convert_to_dataframe(klines)
    ```

=== "Persistence"

    ```python
    @injectable
    class PostgresPortfolioRepository:
        def __init__(self, db: AsyncSession):
            self.db = db

        async def save(self, portfolio: Portfolio) -> Portfolio:
            db_portfolio = PortfolioModel.from_entity(portfolio)
            self.db.add(db_portfolio)
            await self.db.commit()
            return portfolio
    ```

=== "Observability"

    ```python
    @injectable
    class StructuredLogger:
        def __init__(self, config: LoggingConfig):
            self.logger = structlog.get_logger("qframe")

        def log_trade_execution(self, order: Order, result: ExecutionResult):
            self.logger.info(
                "trade_executed",
                order_id=order.id,
                symbol=order.symbol,
                quantity=float(order.quantity),
                execution_time=result.execution_time.isoformat()
            )
    ```

## Patterns Architecturaux

### 1. **Repository Pattern**

Abstraction de la persistance des données :

```python
class PortfolioRepository(Protocol):
    async def save(self, portfolio: Portfolio) -> Portfolio: ...
    async def get_by_id(self, portfolio_id: str) -> Optional[Portfolio]: ...
    async def list_all(self) -> List[Portfolio]: ...

# Implémentations multiples
@injectable
class MemoryPortfolioRepository: ...  # Pour les tests

@injectable
class PostgresPortfolioRepository: ...  # Pour la production

@injectable
class MongoPortfolioRepository: ...  # Pour NoSQL
```

### 2. **CQRS (Command Query Responsibility Segregation)**

Séparation des commandes et requêtes :

```python
# Commands (mutations)
@dataclass
class CreatePortfolioCommand:
    name: str
    initial_capital: Decimal
    base_currency: str

class CreatePortfolioHandler:
    async def handle(self, command: CreatePortfolioCommand) -> Portfolio:
        portfolio = Portfolio(
            id=generate_id(),
            name=command.name,
            initial_capital=command.initial_capital,
            base_currency=command.base_currency
        )
        return await self.repository.save(portfolio)

# Queries (lectures)
@dataclass
class GetPortfolioQuery:
    portfolio_id: str

class GetPortfolioHandler:
    async def handle(self, query: GetPortfolioQuery) -> Optional[Portfolio]:
        return await self.repository.get_by_id(query.portfolio_id)
```

### 3. **Event-Driven Architecture**

Découplage via événements :

```python
@dataclass
class OrderExecutedEvent:
    order_id: str
    portfolio_id: str
    execution_time: datetime
    execution_price: Decimal

class PortfolioService:
    async def handle_order_executed(self, event: OrderExecutedEvent):
        portfolio = await self.repository.get_by_id(event.portfolio_id)
        portfolio.update_from_execution(event)
        await self.repository.save(portfolio)

# Event bus avec DI
@injectable
class EventBus:
    def publish(self, event: DomainEvent): ...
    def subscribe(self, event_type: Type, handler: Callable): ...
```

## Configuration des Environnements

QFrame supporte plusieurs environnements avec configuration automatique :

=== "Development"

    ```python
    class DevelopmentConfig(FrameworkConfig):
        environment: Environment = Environment.DEVELOPMENT

        # Database locale
        database: DatabaseConfig = DatabaseConfig(
            url="sqlite:///./qframe_dev.db",
            echo_sql=True
        )

        # Logging verbeux
        logging: LoggingConfig = LoggingConfig(
            level="DEBUG",
            structured=False
        )
    ```

=== "Production"

    ```python
    class ProductionConfig(FrameworkConfig):
        environment: Environment = Environment.PRODUCTION

        # Database distribuée
        database: DatabaseConfig = DatabaseConfig(
            url="postgresql://user:pass@cluster/qframe",
            pool_size=20,
            echo_sql=False
        )

        # Sécurité renforcée
        security: SecurityConfig = SecurityConfig(
            secret_key=os.environ["SECRET_KEY"],
            require_https=True
        )
    ```

=== "Testing"

    ```python
    class TestingConfig(FrameworkConfig):
        environment: Environment = Environment.TESTING

        # Base en mémoire
        database: DatabaseConfig = DatabaseConfig(
            url="sqlite:///:memory:",
            echo_sql=False
        )

        # Mocks activés
        testing: TestingConfig = TestingConfig(
            use_mocks=True,
            fast_mode=True
        )
    ```

## Extensibilité

L'architecture permet d'étendre facilement le framework :

### 1. **Nouvelles Stratégies**

```python
@injectable
class MyCustomStrategy:
    def __init__(self,
                 data_provider: DataProvider,
                 feature_processor: FeatureProcessor):
        self.data_provider = data_provider
        self.feature_processor = feature_processor

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        # Votre logique personnalisée
        features = self.feature_processor.process(data)
        return self._my_algorithm(data, features)

# Enregistrement automatique
container.register_singleton(Strategy, MyCustomStrategy)
```

### 2. **Nouveaux Data Providers**

```python
@injectable
class AlpacaProvider:
    async def fetch_ohlcv(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
        # Implémentation Alpaca
        ...

# Le container résout automatiquement
container.register_singleton(DataProvider, AlpacaProvider)
```

### 3. **Nouveaux Modules**

```python
# Module de sentiment analysis
@injectable
class SentimentAnalyzer:
    def analyze_news(self, symbol: str) -> float: ...
    def analyze_social_media(self, symbol: str) -> float: ...

# Intégration dans une stratégie
@injectable
class SentimentAwareStrategy:
    def __init__(self,
                 sentiment: SentimentAnalyzer,
                 base_strategy: Strategy):
        self.sentiment = sentiment
        self.base_strategy = base_strategy
```

## Performance et Scalabilité

L'architecture QFrame est conçue pour la performance :

- **Container DI** : Thread-safe avec cache de résolution
- **Configuration** : Lazy loading et validation asynchrone
- **Database** : Connection pooling et requêtes optimisées
- **Caching** : Redis pour les données fréquemment accédées
- **Async/Await** : Support natif pour I/O non-bloquante

## Sécurité

Sécurité intégrée à tous les niveaux :

- **Secrets Management** : Variables d'environnement et vault
- **API Authentication** : JWT tokens avec rotation
- **Data Encryption** : Chiffrement des données sensibles
- **Input Validation** : Pydantic pour toutes les entrées
- **Audit Logging** : Traçabilité complète des actions

---

*Cette architecture garantit maintenabilité, testabilité et extensibilité pour le framework QFrame.*