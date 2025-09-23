"""
Infrastructure Layer: Advanced Persistence
==========================================

Système de persistence avancé avec PostgreSQL, Redis cache,
time-series DB et gestion des migrations.
"""

from .database import (
    DatabaseManager,
    DatabaseConfig,
    ConnectionPool,
    TransactionManager,
    get_database_manager,
    create_database_manager
)

from .repositories import (
    PostgresStrategyRepository,
    PostgresPortfolioRepository,
    PostgresOrderRepository,
    PostgresRiskAssessmentRepository,
    PostgresBacktestRepository
)

from .cache import (
    CacheManager,
    CacheConfig,
    RedisCache,
    InMemoryCache,
    get_cache_manager,
    create_cache_manager
)

from .timeseries import (
    TimeSeriesDB,
    TimeSeriesConfig as InfluxDBConfig,
    InfluxDBManager,
    MarketDataStorage,
    get_timeseries_db,
    create_timeseries_db
)

# Note: migrations module is a package, import from parent
try:
    from .migrations import (
        Migration,
        SQLMigration,
        PythonMigration,
        MigrationManager,
        get_migration_manager,
        create_migration_manager
    )
except ImportError:
    # Migrations are optional
    Migration = None
    SQLMigration = None
    PythonMigration = None
    MigrationManager = None
    get_migration_manager = lambda: None
    create_migration_manager = lambda *args, **kwargs: None

__all__ = [
    # Database management
    'DatabaseManager',
    'DatabaseConfig',
    'ConnectionPool',
    'TransactionManager',
    'get_database_manager',
    'create_database_manager',

    # Repository implementations
    'PostgresStrategyRepository',
    'PostgresPortfolioRepository',
    'PostgresOrderRepository',
    'PostgresRiskAssessmentRepository',
    'PostgresBacktestRepository',

    # Caching
    'CacheManager',
    'CacheConfig',
    'RedisCache',
    'InMemoryCache',
    'get_cache_manager',
    'create_cache_manager',

    # Time-series
    'TimeSeriesDB',
    'InfluxDBConfig',
    'InfluxDBManager',
    'MarketDataStorage',
    'get_timeseries_db',
    'create_timeseries_db',

    # Migrations
    'Migration',
    'SQLMigration',
    'PythonMigration',
    'MigrationManager',
    'get_migration_manager',
    'create_migration_manager'
]
