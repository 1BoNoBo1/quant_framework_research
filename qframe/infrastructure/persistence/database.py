"""
Infrastructure Layer: Database Management
=========================================

Gestionnaire de base de données avec connection pooling,
transaction management et monitoring.
"""

import asyncio
try:
    import asyncpg
except ImportError:
    asyncpg = None
try:
    import psycopg2
    from psycopg2 import pool
except ImportError:
    psycopg2 = None
    pool = None
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncContextManager, Callable
from enum import Enum

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace


class IsolationLevel(str, Enum):
    """Niveaux d'isolation des transactions"""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


@dataclass
class DatabaseConfig:
    """Configuration de la base de données"""
    host: str = "localhost"
    port: int = 5432
    database: str = "qframe"
    user: str = "qframe"
    password: str = "qframe"
    
    # Pool configuration
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    idle_timeout: int = 300
    
    # Query configuration
    query_timeout: int = 30
    statement_timeout: int = 0
    
    # SSL configuration
    ssl_mode: str = "prefer"
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None

    def get_dsn(self) -> str:
        """Obtenir la chaîne de connexion"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_async_dsn(self) -> str:
        """Obtenir la chaîne de connexion async"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class ConnectionStats:
    """Statistiques de connexion"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_queries: int = 0
    failed_queries: int = 0
    avg_query_time_ms: float = 0.0
    connection_errors: int = 0


class ConnectionPool:
    """
    Pool de connexions PostgreSQL avec monitoring.
    Gère les connexions asynchrones et synchrones.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

        # Pools
        self._async_pool: Optional[asyncpg.Pool] = None
        self._sync_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

        # Statistiques
        self._stats = ConnectionStats()
        self._query_times: List[float] = []

    async def initialize_async_pool(self):
        """Initialiser le pool asynchrone"""
        if not asyncpg:
            self.logger.warning("asyncpg not installed, skipping async pool initialization")
            return

        try:
            self._async_pool = await asyncpg.create_pool(
                dsn=self.config.get_async_dsn(),
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.query_timeout,
                server_settings={
                    'application_name': 'qframe',
                    'timezone': 'UTC'
                }
            )

            self.logger.info(
                f"Async connection pool initialized",
                min_connections=self.config.min_connections,
                max_connections=self.config.max_connections
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize async pool: {e}")
            raise

    def initialize_sync_pool(self):
        """Initialiser le pool synchrone"""
        if not psycopg2 or not pool:
            self.logger.warning("psycopg2 not installed, skipping sync pool initialization")
            return

        try:
            self._sync_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                connect_timeout=self.config.connection_timeout
            )

            self.logger.info(
                f"Sync connection pool initialized",
                min_connections=self.config.min_connections,
                max_connections=self.config.max_connections
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize sync pool: {e}")
            raise

    @asynccontextmanager
    async def get_async_connection(self):
        """Obtenir une connexion asynchrone du pool"""
        if not self._async_pool:
            raise RuntimeError("Async pool not initialized")

        start_time = time.time()
        connection = None

        try:
            connection = await self._async_pool.acquire()
            self._stats.active_connections += 1

            yield connection

        except Exception as e:
            self._stats.connection_errors += 1
            self.logger.error(f"Connection error: {e}")
            raise

        finally:
            if connection:
                await self._async_pool.release(connection)
                self._stats.active_connections -= 1

            # Statistiques
            connection_time = (time.time() - start_time) * 1000
            self.metrics.collector.record_histogram(
                "database.connection_time",
                connection_time,
                labels={"pool_type": "async"}
            )

    @asynccontextmanager
    async def get_sync_connection(self):
        """Obtenir une connexion synchrone du pool"""
        if not self._sync_pool:
            raise RuntimeError("Sync pool not initialized")

        connection = None
        try:
            connection = self._sync_pool.getconn()
            self._stats.active_connections += 1

            yield connection

        except Exception as e:
            self._stats.connection_errors += 1
            self.logger.error(f"Sync connection error: {e}")
            raise

        finally:
            if connection:
                self._sync_pool.putconn(connection)
                self._stats.active_connections -= 1

    async def execute_query(
        self,
        query: str,
        params: tuple = None,
        fetch: str = "none"  # "none", "one", "all"
    ) -> Any:
        """Exécuter une requête asynchrone"""
        start_time = time.time()

        try:
            async with self.get_async_connection() as conn:
                if fetch == "all":
                    result = await conn.fetch(query, *(params or ()))
                elif fetch == "one":
                    result = await conn.fetchrow(query, *(params or ()))
                else:
                    result = await conn.execute(query, *(params or ()))

                # Statistiques
                query_time = (time.time() - start_time) * 1000
                self._query_times.append(query_time)
                self._stats.total_queries += 1

                # Maintenir seulement les 1000 derniers temps
                if len(self._query_times) > 1000:
                    self._query_times = self._query_times[-1000:]

                self._stats.avg_query_time_ms = sum(self._query_times) / len(self._query_times)

                # Métriques
                self.metrics.collector.increment_counter("database.queries_total", labels={"status": "success"})
                self.metrics.collector.record_histogram("database.query_duration", query_time, labels={"fetch_type": fetch})

                return result

        except Exception as e:
            self._stats.failed_queries += 1
            self.metrics.collector.increment_counter("database.queries_total", labels={"status": "error"})
            self.logger.error(f"Query failed: {e}", query=query[:200])
            raise

    async def close(self):
        """Fermer tous les pools"""
        if self._async_pool:
            await self._async_pool.close()
            self.logger.info("Async pool closed")

        if self._sync_pool:
            self._sync_pool.closeall()
            self.logger.info("Sync pool closed")

    def get_stats(self) -> ConnectionStats:
        """Obtenir les statistiques"""
        if self._async_pool:
            self._stats.total_connections = self._async_pool.get_size()
            self._stats.idle_connections = self._async_pool.get_idle_size()
        
        return self._stats


class TransactionManager:
    """
    Gestionnaire de transactions avec support des savepoints
    et des transactions distribuées.
    """

    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.logger = LoggerFactory.get_logger(__name__)
        self.tracer = get_tracer()

    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        read_only: bool = False
    ) -> AsyncContextManager:
        """Gestionnaire de contexte pour les transactions"""
        transaction_id = str(uuid.uuid4())
        start_time = time.time()

        self.logger.debug(
            f"Starting transaction {transaction_id}",
            isolation_level=isolation_level.value,
            read_only=read_only
        )

        async with self.pool.get_async_connection() as conn:
            # Commencer la transaction
            if isolation_level == IsolationLevel.SERIALIZABLE:
                await conn.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
            elif isolation_level == IsolationLevel.REPEATABLE_READ:
                await conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            elif isolation_level == IsolationLevel.READ_UNCOMMITTED:
                await conn.execute("BEGIN ISOLATION LEVEL READ UNCOMMITTED")
            else:
                await conn.execute("BEGIN ISOLATION LEVEL READ COMMITTED")

            if read_only:
                await conn.execute("SET TRANSACTION READ ONLY")

            try:
                yield conn
                
                # Commit automatique
                await conn.execute("COMMIT")
                
                duration = (time.time() - start_time) * 1000
                self.logger.debug(
                    f"Transaction {transaction_id} committed",
                    duration_ms=duration
                )

                # Métriques
                self.pool.metrics.collector.increment_counter(
                    "database.transactions_total",
                    labels={"status": "committed"}
                )
                self.pool.metrics.collector.record_histogram(
                    "database.transaction_duration",
                    duration
                )

            except Exception as e:
                # Rollback automatique
                await conn.execute("ROLLBACK")
                
                duration = (time.time() - start_time) * 1000
                self.logger.error(
                    f"Transaction {transaction_id} rolled back: {e}",
                    duration_ms=duration
                )

                # Métriques
                self.pool.metrics.collector.increment_counter(
                    "database.transactions_total",
                    labels={"status": "rolled_back"}
                )

                raise

    @asynccontextmanager
    async def savepoint(self, connection, name: str = None):
        """Créer un savepoint dans une transaction"""
        savepoint_name = name or f"sp_{int(time.time())}"
        
        try:
            await connection.execute(f"SAVEPOINT {savepoint_name}")
            self.logger.debug(f"Savepoint {savepoint_name} created")
            
            yield savepoint_name
            
        except Exception as e:
            await connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            self.logger.warning(f"Rolled back to savepoint {savepoint_name}: {e}")
            raise

    async def execute_in_transaction(
        self,
        func: Callable,
        *args,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        **kwargs
    ) -> Any:
        """Exécuter une fonction dans une transaction"""
        async with self.transaction(isolation_level=isolation_level) as conn:
            return await func(conn, *args, **kwargs)


class DatabaseManager:
    """
    Gestionnaire principal de la base de données.
    Orchestre le pool de connexions et les transactions.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

        self.connection_pool = ConnectionPool(config)
        self.transaction_manager = TransactionManager(self.connection_pool)

        # État
        self._initialized = False

    async def initialize(self):
        """Initialiser le gestionnaire"""
        if self._initialized:
            return

        try:
            # Initialiser les pools
            await self.connection_pool.initialize_async_pool()
            self.connection_pool.initialize_sync_pool()

            # Tester la connexion
            await self._test_connection()

            self._initialized = True
            self.logger.info("Database manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize database manager: {e}")
            raise

    async def _test_connection(self):
        """Tester la connexion à la base de données"""
        try:
            result = await self.connection_pool.execute_query(
                "SELECT version(), current_database(), current_user",
                fetch="one"
            )
            
            self.logger.info(
                f"Database connection test successful",
                version=result[0],
                database=result[1],
                user=result[2]
            )

        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            raise

    async def execute_query(
        self,
        query: str,
        params: tuple = None,
        fetch: str = "none"
    ) -> Any:
        """Exécuter une requête"""
        return await self.connection_pool.execute_query(query, params, fetch)

    async def execute_script(self, script: str):
        """Exécuter un script SQL (migrations, etc.)"""
        statements = [stmt.strip() for stmt in script.split(';') if stmt.strip()]
        
        async with self.transaction_manager.transaction() as conn:
            for statement in statements:
                if statement:
                    await conn.execute(statement)

    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Obtenir les informations d'une table"""
        query = """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_name = $1
            ORDER BY ordinal_position
        """
        
        columns = await self.execute_query(query, (table_name,), fetch="all")
        
        return {
            "table_name": table_name,
            "columns": [dict(col) for col in columns]
        }

    async def check_table_exists(self, table_name: str) -> bool:
        """Vérifier si une table existe"""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = $1
            )
        """
        
        result = await self.execute_query(query, (table_name,), fetch="one")
        return result[0]

    async def get_database_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de la base de données"""
        stats_query = """
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_tuples,
                n_dead_tup as dead_tuples
            FROM pg_stat_user_tables
            ORDER BY schemaname, tablename
        """
        
        table_stats = await self.execute_query(stats_query, fetch="all")
        
        size_query = """
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """
        
        size_stats = await self.execute_query(size_query, fetch="all")
        
        return {
            "connection_stats": self.connection_pool.get_stats().__dict__,
            "table_stats": [dict(stat) for stat in table_stats],
            "size_stats": [dict(stat) for stat in size_stats]
        }

    async def close(self):
        """Fermer le gestionnaire"""
        await self.connection_pool.close()
        self._initialized = False
        self.logger.info("Database manager closed")


# Instance globale
_global_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> Optional[DatabaseManager]:
    """Obtenir l'instance globale du gestionnaire de base de données"""
    return _global_database_manager


def create_database_manager(config: DatabaseConfig) -> DatabaseManager:
    """Créer l'instance globale du gestionnaire de base de données"""
    global _global_database_manager
    _global_database_manager = DatabaseManager(config)
    return _global_database_manager
