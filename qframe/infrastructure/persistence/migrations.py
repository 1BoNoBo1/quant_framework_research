"""
Infrastructure Layer: Database Migrations
=========================================

Système de migration de base de données avec versioning,
rollback et support SQL/Python.
"""

import asyncio
import hashlib
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from enum import Enum

from ..observability.logging import LoggerFactory
from ..observability.tracing import trace
from .database import DatabaseManager


class MigrationStatus(str, Enum):
    """Statuts de migration"""
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationInfo:
    """Information sur une migration"""
    version: str
    name: str
    description: str
    applied_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    checksum: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class Migration(ABC):
    """
    Classe de base pour les migrations.
    Chaque migration doit implémenter up() et down().
    """

    def __init__(self, version: str, name: str, description: str = ""):
        self.version = version
        self.name = name
        self.description = description
        self.logger = LoggerFactory.get_logger(__name__)

    @abstractmethod
    async def up(self, db_manager: DatabaseManager) -> None:
        """Appliquer la migration"""
        pass

    @abstractmethod
    async def down(self, db_manager: DatabaseManager) -> None:
        """Annuler la migration"""
        pass

    def get_checksum(self) -> str:
        """Calculer le checksum de la migration"""
        content = f"{self.version}:{self.name}:{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()


class SQLMigration(Migration):
    """Migration basée sur du SQL"""

    def __init__(
        self,
        version: str,
        name: str,
        up_sql: str,
        down_sql: str,
        description: str = ""
    ):
        super().__init__(version, name, description)
        self.up_sql = up_sql
        self.down_sql = down_sql

    async def up(self, db_manager: DatabaseManager) -> None:
        """Exécuter le SQL d'upgrade"""
        await db_manager.execute_script(self.up_sql)
        self.logger.info(f"Applied SQL migration {self.version}: {self.name}")

    async def down(self, db_manager: DatabaseManager) -> None:
        """Exécuter le SQL de rollback"""
        await db_manager.execute_script(self.down_sql)
        self.logger.info(f"Rolled back SQL migration {self.version}: {self.name}")

    def get_checksum(self) -> str:
        """Checksum incluant le SQL"""
        content = f"{self.version}:{self.name}:{self.up_sql}:{self.down_sql}"
        return hashlib.sha256(content.encode()).hexdigest()


class PythonMigration(Migration):
    """Migration basée sur du code Python"""

    def __init__(
        self,
        version: str,
        name: str,
        up_func: Callable,
        down_func: Callable,
        description: str = ""
    ):
        super().__init__(version, name, description)
        self.up_func = up_func
        self.down_func = down_func

    async def up(self, db_manager: DatabaseManager) -> None:
        """Exécuter la fonction d'upgrade"""
        if asyncio.iscoroutinefunction(self.up_func):
            await self.up_func(db_manager)
        else:
            self.up_func(db_manager)
        self.logger.info(f"Applied Python migration {self.version}: {self.name}")

    async def down(self, db_manager: DatabaseManager) -> None:
        """Exécuter la fonction de rollback"""
        if asyncio.iscoroutinefunction(self.down_func):
            await self.down_func(db_manager)
        else:
            self.down_func(db_manager)
        self.logger.info(f"Rolled back Python migration {self.version}: {self.name}")


class MigrationManager:
    """
    Gestionnaire de migrations avec tracking et rollback.
    """

    def __init__(self, db_manager: DatabaseManager, migrations_path: Optional[Path] = None):
        self.db_manager = db_manager
        self.migrations_path = migrations_path
        self.logger = LoggerFactory.get_logger(__name__)

        self._migrations: List[Migration] = []
        self._migration_history: Dict[str, MigrationInfo] = {}

    async def initialize(self):
        """Initialiser le système de migration"""
        await self._create_migration_table()
        await self._load_migration_history()

        if self.migrations_path:
            await self._discover_migrations()

        self.logger.info(
            "Migration manager initialized",
            total_migrations=len(self._migrations),
            applied_migrations=sum(
                1 for info in self._migration_history.values()
                if info.status == MigrationStatus.APPLIED
            )
        )

    async def _create_migration_table(self):
        """Créer la table de tracking des migrations"""
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(50) NOT NULL,
                checksum VARCHAR(64),
                execution_time_ms FLOAT,
                error_message TEXT
            )
        """
        await self.db_manager.execute_script(create_table_sql)
        self.logger.debug("Migration tracking table created")

    async def _load_migration_history(self):
        """Charger l'historique des migrations"""
        query = """
            SELECT 
                version, name, description, applied_at, 
                status, checksum, execution_time_ms, error_message
            FROM schema_migrations
            ORDER BY version
        """

        rows = await self.db_manager.execute_query(query, fetch="all")

        for row in rows:
            info = MigrationInfo(
                version=row["version"],
                name=row["name"],
                description=row["description"] or "",
                applied_at=row["applied_at"],
                status=MigrationStatus(row["status"]),
                checksum=row["checksum"],
                execution_time_ms=row["execution_time_ms"],
                error_message=row["error_message"]
            )
            self._migration_history[info.version] = info

    async def _discover_migrations(self):
        """Découvrir les fichiers de migration"""
        if not self.migrations_path or not self.migrations_path.exists():
            return

        for migration_file in sorted(self.migrations_path.glob("*.py")):
            if migration_file.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    migration_file.stem,
                    migration_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "migration"):
                    self.register_migration(module.migration)

            except Exception as e:
                self.logger.error(
                    f"Failed to load migration from {migration_file}: {e}"
                )

    def register_migration(self, migration: Migration):
        """Enregistrer une migration"""
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)
        self.logger.debug(f"Registered migration {migration.version}: {migration.name}")

    @trace("migration.apply")
    async def apply_migration(self, migration: Migration) -> bool:
        """Appliquer une migration unique"""
        import time

        if migration.version in self._migration_history:
            existing = self._migration_history[migration.version]
            if existing.status == MigrationStatus.APPLIED:
                self.logger.warning(
                    f"Migration {migration.version} already applied, skipping"
                )
                return True

        start_time = time.time()

        try:
            async with self.db_manager.transaction_manager.transaction():
                await migration.up(self.db_manager)

                execution_time = (time.time() - start_time) * 1000
                checksum = migration.get_checksum()

                await self._record_migration(
                    migration,
                    MigrationStatus.APPLIED,
                    checksum,
                    execution_time
                )

                self.logger.info(
                    f"Applied migration {migration.version}: {migration.name}",
                    execution_time_ms=execution_time
                )

                return True

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            await self._record_migration(
                migration,
                MigrationStatus.FAILED,
                migration.get_checksum(),
                execution_time,
                str(e)
            )

            self.logger.error(
                f"Failed to apply migration {migration.version}: {e}",
                execution_time_ms=execution_time
            )

            return False

    @trace("migration.rollback")
    async def rollback_migration(self, migration: Migration) -> bool:
        """Annuler une migration"""
        import time

        if migration.version not in self._migration_history:
            self.logger.warning(
                f"Migration {migration.version} not found in history, skipping rollback"
            )
            return True

        start_time = time.time()

        try:
            async with self.db_manager.transaction_manager.transaction():
                await migration.down(self.db_manager)

                execution_time = (time.time() - start_time) * 1000

                await self._update_migration_status(
                    migration.version,
                    MigrationStatus.ROLLED_BACK,
                    execution_time
                )

                self.logger.info(
                    f"Rolled back migration {migration.version}: {migration.name}",
                    execution_time_ms=execution_time
                )

                return True

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            self.logger.error(
                f"Failed to rollback migration {migration.version}: {e}",
                execution_time_ms=execution_time
            )

            return False

    async def apply_all_pending(self) -> int:
        """Appliquer toutes les migrations en attente"""
        applied_count = 0

        for migration in self._migrations:
            if migration.version in self._migration_history:
                info = self._migration_history[migration.version]
                if info.status == MigrationStatus.APPLIED:
                    continue

            success = await self.apply_migration(migration)
            if success:
                applied_count += 1
            else:
                self.logger.error(
                    f"Stopping migration process due to failure at {migration.version}"
                )
                break

        self.logger.info(f"Applied {applied_count} migrations")
        return applied_count

    async def rollback_to_version(self, target_version: str) -> int:
        """Annuler les migrations jusqu'à une version cible"""
        rolled_back_count = 0

        for migration in reversed(self._migrations):
            if migration.version <= target_version:
                break

            if migration.version in self._migration_history:
                info = self._migration_history[migration.version]
                if info.status == MigrationStatus.APPLIED:
                    success = await self.rollback_migration(migration)
                    if success:
                        rolled_back_count += 1
                    else:
                        self.logger.error(
                            f"Stopping rollback process due to failure at {migration.version}"
                        )
                        break

        self.logger.info(f"Rolled back {rolled_back_count} migrations")
        return rolled_back_count

    async def _record_migration(
        self,
        migration: Migration,
        status: MigrationStatus,
        checksum: str,
        execution_time_ms: float,
        error_message: Optional[str] = None
    ):
        """Enregistrer une migration dans l'historique"""
        query = """
            INSERT INTO schema_migrations 
                (version, name, description, status, checksum, execution_time_ms, error_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (version) 
            DO UPDATE SET
                status = EXCLUDED.status,
                checksum = EXCLUDED.checksum,
                execution_time_ms = EXCLUDED.execution_time_ms,
                error_message = EXCLUDED.error_message,
                applied_at = CURRENT_TIMESTAMP
        """

        await self.db_manager.execute_query(
            query,
            (
                migration.version,
                migration.name,
                migration.description,
                status.value,
                checksum,
                execution_time_ms,
                error_message
            )
        )

        info = MigrationInfo(
            version=migration.version,
            name=migration.name,
            description=migration.description,
            applied_at=datetime.now(),
            status=status,
            checksum=checksum,
            execution_time_ms=execution_time_ms,
            error_message=error_message
        )
        self._migration_history[migration.version] = info

    async def _update_migration_status(
        self,
        version: str,
        status: MigrationStatus,
        execution_time_ms: float
    ):
        """Mettre à jour le statut d'une migration"""
        query = """
            UPDATE schema_migrations
            SET status = $1, execution_time_ms = $2, applied_at = CURRENT_TIMESTAMP
            WHERE version = $3
        """

        await self.db_manager.execute_query(
            query,
            (status.value, execution_time_ms, version)
        )

        if version in self._migration_history:
            self._migration_history[version].status = status
            self._migration_history[version].execution_time_ms = execution_time_ms
            self._migration_history[version].applied_at = datetime.now()

    def get_pending_migrations(self) -> List[Migration]:
        """Obtenir les migrations en attente"""
        pending = []

        for migration in self._migrations:
            if migration.version not in self._migration_history:
                pending.append(migration)
            else:
                info = self._migration_history[migration.version]
                if info.status != MigrationStatus.APPLIED:
                    pending.append(migration)

        return pending

    def get_applied_migrations(self) -> List[MigrationInfo]:
        """Obtenir les migrations appliquées"""
        return [
            info for info in self._migration_history.values()
            if info.status == MigrationStatus.APPLIED
        ]

    def get_current_version(self) -> Optional[str]:
        """Obtenir la version actuelle du schéma"""
        applied = self.get_applied_migrations()
        if not applied:
            return None

        return max(applied, key=lambda info: info.version).version

    def get_migration_status(self) -> Dict[str, Any]:
        """Obtenir le statut complet des migrations"""
        return {
            "current_version": self.get_current_version(),
            "total_migrations": len(self._migrations),
            "pending_migrations": len(self.get_pending_migrations()),
            "applied_migrations": len(self.get_applied_migrations()),
            "failed_migrations": sum(
                1 for info in self._migration_history.values()
                if info.status == MigrationStatus.FAILED
            ),
            "migrations": [
                {
                    "version": m.version,
                    "name": m.name,
                    "status": self._migration_history.get(m.version, MigrationInfo(
                        version=m.version,
                        name=m.name,
                        description=m.description
                    )).status.value
                }
                for m in self._migrations
            ]
        }


_global_migration_manager: Optional[MigrationManager] = None


def get_migration_manager() -> Optional[MigrationManager]:
    """Obtenir l'instance globale du gestionnaire de migrations"""
    return _global_migration_manager


def create_migration_manager(
    db_manager: DatabaseManager,
    migrations_path: Optional[Path] = None
) -> MigrationManager:
    """Créer l'instance globale du gestionnaire de migrations"""
    global _global_migration_manager
    _global_migration_manager = MigrationManager(db_manager, migrations_path)
    return _global_migration_manager
