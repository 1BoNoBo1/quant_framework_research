"""
Configuration centralisée avec Pydantic
=====================================

Système de configuration type-safe et validé pour tout le framework.
Supporte les environnements multiples et la validation automatique.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import os
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Environnements d'exécution"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    BACKTESTING = "backtesting"


class LogLevel(str, Enum):
    """Niveaux de logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseSettings):
    """Configuration base de données"""
    host: str = "localhost"
    port: int = 5432
    database: str = "quant_framework"
    username: str = Field("postgres", alias="DB_USERNAME")
    password: str = Field("", alias="DB_PASSWORD")
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    model_config = {
        'env_prefix': 'DB_',
        'case_sensitive': False,
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'extra': 'ignore'
    }

    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v


class RedisConfig(BaseSettings):
    """Configuration Redis pour cache"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = Field(None, alias="REDIS_PASSWORD")
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

    model_config = {
        'env_prefix': 'REDIS_',
        'case_sensitive': False,
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'extra': 'ignore'
    }


class MLFlowConfig(BaseModel):
    """Configuration MLflow"""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "quant_framework"
    artifact_location: Optional[str] = None
    registry_uri: Optional[str] = None

    @field_validator('tracking_uri')
    @classmethod
    def validate_tracking_uri(cls, v):
        if not (v.startswith('http://') or v.startswith('https://') or v.startswith('file://')):
            raise ValueError('tracking_uri must be a valid URL or file path')
        return v


class DataProviderConfig(BaseModel):
    """Configuration pour fournisseurs de données"""
    name: str
    api_key: Optional[str] = Field(None, env="DATA_API_KEY")
    api_secret: Optional[str] = Field(None, env="DATA_API_SECRET")
    base_url: Optional[str] = None
    rate_limit: int = 1200  # Requêtes par minute
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class StrategyConfig(BaseModel):
    """Configuration de base pour les stratégies"""
    name: str
    enabled: bool = True
    position_size: float = Field(0.02, ge=0.001, le=1.0)
    max_positions: int = Field(5, ge=1, le=50)
    risk_per_trade: float = Field(0.02, ge=0.001, le=0.1)

    @field_validator('position_size')
    @classmethod
    def validate_position_size(cls, v):
        if not 0.001 <= v <= 1.0:
            raise ValueError('position_size must be between 0.001 and 1.0')
        return v


class RiskManagementConfig(BaseModel):
    """Configuration gestion des risques"""
    max_portfolio_risk: float = Field(0.1, ge=0.01, le=0.5)
    max_correlation: float = Field(0.7, ge=0.1, le=1.0)
    var_confidence: float = Field(0.95, ge=0.9, le=0.99)
    drawdown_limit: float = Field(0.2, ge=0.05, le=0.5)
    position_timeout_hours: int = Field(24, ge=1, le=168)

    # Stop-loss et take-profit par défaut
    default_stop_loss: float = Field(0.05, ge=0.01, le=0.2)
    default_take_profit: float = Field(0.1, ge=0.02, le=0.5)


class BacktestConfig(BaseModel):
    """Configuration backtesting"""
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = Field(100000.0, ge=1000.0)
    commission: float = Field(0.001, ge=0.0, le=0.01)
    slippage: float = Field(0.0005, ge=0.0, le=0.01)

    # Benchmark
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = Field(0.02, ge=0.0, le=0.1)


class TradingConfig(BaseModel):
    """Configuration trading live"""
    broker: str = "binance"
    testnet: bool = True
    api_key: Optional[str] = Field(None, env="TRADING_API_KEY")
    api_secret: Optional[str] = Field(None, env="TRADING_API_SECRET")

    # Paramètres de trading
    min_order_size: float = Field(10.0, ge=1.0)
    max_order_size: float = Field(10000.0, ge=100.0)
    order_timeout: int = Field(300, ge=30, le=3600)  # secondes


class AlertConfig(BaseModel):
    """Configuration alertes"""
    enabled: bool = True
    email_enabled: bool = False
    slack_enabled: bool = False

    # Email
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = 587
    smtp_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")

    # Slack
    slack_webhook_url: Optional[str] = Field(None, env="SLACK_WEBHOOK_URL")
    slack_channel: str = "#trading-alerts"


class FrameworkConfig(BaseSettings):
    """Configuration principale du framework"""

    # Métadonnées de base
    app_name: str = "Quant Framework Research"
    app_version: str = "2.0.0"
    environment: Environment = Environment.DEVELOPMENT

    # Chemins
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))
    artifacts_dir: Path = Field(default_factory=lambda: Path("artifacts"))

    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None

    # Sécurité
    secret_key: str = Field("dev-secret-key-change-in-production", env="SECRET_KEY")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Performance
    max_workers: int = Field(4, ge=1, le=32)
    cache_ttl: int = 3600  # secondes

    # Configurations des composants
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    mlflow: MLFlowConfig = MLFlowConfig()
    risk_management: RiskManagementConfig = RiskManagementConfig()
    backtest: BacktestConfig = BacktestConfig()
    trading: TradingConfig = TradingConfig()
    alerts: AlertConfig = AlertConfig()

    # Fournisseurs de données
    data_providers: Dict[str, DataProviderConfig] = Field(default_factory=lambda: {
        "binance": DataProviderConfig(
            name="binance",
            base_url="https://api.binance.com",
            rate_limit=1200
        ),
        "yfinance": DataProviderConfig(
            name="yfinance",
            rate_limit=2000
        )
    })

    # Stratégies configurées
    strategies: Dict[str, StrategyConfig] = Field(default_factory=lambda: {
        "dmn_lstm": StrategyConfig(
            name="DMN_LSTM",
            position_size=0.02,
            max_positions=3
        ),
        "mean_reversion": StrategyConfig(
            name="Adaptive_Mean_Reversion",
            position_size=0.015,
            max_positions=5
        ),
        "funding_arbitrage": StrategyConfig(
            name="Funding_Arbitrage",
            position_size=0.01,
            max_positions=2
        ),
        "rl_alpha": StrategyConfig(
            name="RL_Alpha_Generator",
            position_size=0.02,
            max_positions=3
        )
    })

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env file

    @model_validator(mode='after')
    def validate_paths(self):
        """Valide et crée les répertoires nécessaires"""
        if self.project_root:
            for path_name in ['data_dir', 'logs_dir', 'artifacts_dir']:
                path_value = getattr(self, path_name, None)
                if path_value and not path_value.is_absolute():
                    setattr(self, path_name, self.project_root / path_value)

                # Créer le répertoire s'il n'existe pas
                full_path = getattr(self, path_name)
                if full_path:
                    full_path.mkdir(parents=True, exist_ok=True)

        return self

    @model_validator(mode='after')
    def validate_environment_settings(self):
        """Valide les paramètres selon l'environnement"""
        if self.environment == Environment.PRODUCTION:
            if self.secret_key == "dev-secret-key-change-in-production":
                raise ValueError('SECRET_KEY must be changed in production')
        return self

    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Récupère la configuration d'une stratégie"""
        return self.strategies.get(strategy_name)

    def get_data_provider_config(self, provider_name: str) -> Optional[DataProviderConfig]:
        """Récupère la configuration d'un fournisseur de données"""
        return self.data_providers.get(provider_name)

    def is_development(self) -> bool:
        """Vérifie si on est en mode développement"""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Vérifie si on est en mode production"""
        return self.environment == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Vérifie si on est en mode test"""
        return self.environment == Environment.TESTING


# Instance globale de configuration
config = FrameworkConfig()


def get_config() -> FrameworkConfig:
    """Retourne l'instance de configuration globale"""
    return config


def load_config_from_file(file_path: str) -> FrameworkConfig:
    """Charge la configuration depuis un fichier"""
    import json
    import yaml

    path = Path(file_path)

    if path.suffix.lower() == '.json':
        with open(path, 'r') as f:
            config_data = json.load(f)
    elif path.suffix.lower() in ['.yml', '.yaml']:
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    return FrameworkConfig(**config_data)


def update_config(**kwargs):
    """Met à jour la configuration globale"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)


# Configuration spécialisée pour chaque environnement
class DevelopmentConfig(FrameworkConfig):
    """Configuration développement"""

    def __init__(self, **data):
        super().__init__(**data)
        self.environment = Environment.DEVELOPMENT
        self.log_level = LogLevel.DEBUG

    class Config:
        env_prefix = "DEV_"


class ProductionConfig(FrameworkConfig):
    """Configuration production"""

    def __init__(self, **data):
        super().__init__(**data)
        self.environment = Environment.PRODUCTION
        self.log_level = LogLevel.INFO

    class Config:
        env_prefix = "PROD_"


class TestingConfig(FrameworkConfig):
    """Configuration test"""

    def __init__(self, **data):
        super().__init__(**data)
        self.environment = Environment.TESTING
        self.log_level = LogLevel.DEBUG
        # Base de données en mémoire pour les tests
        self.database = DatabaseConfig(
            host="sqlite",
            database=":memory:"
        )

    class Config:
        env_prefix = "TEST_"


def get_config_for_environment(env: str) -> FrameworkConfig:
    """Retourne la configuration pour un environnement spécifique"""
    env_configs = {
        Environment.DEVELOPMENT: DevelopmentConfig(),
        Environment.PRODUCTION: ProductionConfig(),
        Environment.TESTING: TestingConfig(),
        Environment.BACKTESTING: FrameworkConfig(environment=Environment.BACKTESTING)
    }

    try:
        env_enum = Environment(env)
        return env_configs.get(env_enum, FrameworkConfig())
    except ValueError:
        # Environnement inconnu, retourne config de base
        return FrameworkConfig()