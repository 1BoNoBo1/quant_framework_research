"""
Infrastructure: MemoryStrategyRepository
=======================================

Implémentation en mémoire du repository de stratégies.
Utilisée pour les tests et le développement.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import logging

from ...domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from ...domain.repositories.strategy_repository import (
    StrategyRepository,
    RepositoryError,
    StrategyNotFoundError,
    DuplicateStrategyError
)

logger = logging.getLogger(__name__)


class MemoryStrategyRepository(StrategyRepository):
    """
    Implémentation en mémoire du repository de stratégies.

    Stocke les stratégies dans un dictionnaire en mémoire.
    Idéal pour les tests et le prototypage.
    """

    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}
        self._backups: Dict[str, Strategy] = {}

    async def save(self, strategy: Strategy) -> None:
        """Sauvegarde une stratégie en mémoire."""
        try:
            # Vérifier si une stratégie avec ce nom existe déjà (sauf si c'est une mise à jour)
            existing_by_name = await self.find_by_name(strategy.name)
            if existing_by_name and existing_by_name.id != strategy.id:
                raise DuplicateStrategyError(f"Strategy with name '{strategy.name}' already exists")

            # Créer une copie pour éviter les modifications externes
            strategy_copy = self._deep_copy_strategy(strategy)
            self._strategies[strategy.id] = strategy_copy

            logger.debug(f"💾 Strategy saved: {strategy.id}")

        except Exception as e:
            logger.error(f"❌ Error saving strategy {strategy.id}: {e}")
            raise RepositoryError(f"Failed to save strategy: {e}")

    async def find_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """Trouve une stratégie par son ID."""
        try:
            strategy = self._strategies.get(strategy_id)
            return self._deep_copy_strategy(strategy) if strategy else None

        except Exception as e:
            logger.error(f"❌ Error finding strategy {strategy_id}: {e}")
            raise RepositoryError(f"Failed to find strategy: {e}")

    async def find_by_name(self, name: str) -> Optional[Strategy]:
        """Trouve une stratégie par son nom."""
        try:
            for strategy in self._strategies.values():
                if strategy.name == name:
                    return self._deep_copy_strategy(strategy)
            return None

        except Exception as e:
            logger.error(f"❌ Error finding strategy by name '{name}': {e}")
            raise RepositoryError(f"Failed to find strategy by name: {e}")

    async def find_all(self) -> List[Strategy]:
        """Récupère toutes les stratégies."""
        try:
            return [self._deep_copy_strategy(s) for s in self._strategies.values()]

        except Exception as e:
            logger.error(f"❌ Error finding all strategies: {e}")
            raise RepositoryError(f"Failed to find all strategies: {e}")

    async def find_by_status(self, status: StrategyStatus) -> List[Strategy]:
        """Trouve les stratégies par statut."""
        try:
            matching_strategies = [
                self._deep_copy_strategy(s)
                for s in self._strategies.values()
                if s.status == status
            ]
            return matching_strategies

        except Exception as e:
            logger.error(f"❌ Error finding strategies by status {status}: {e}")
            raise RepositoryError(f"Failed to find strategies by status: {e}")

    async def find_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        """Trouve les stratégies par type."""
        try:
            matching_strategies = [
                self._deep_copy_strategy(s)
                for s in self._strategies.values()
                if s.strategy_type == strategy_type
            ]
            return matching_strategies

        except Exception as e:
            logger.error(f"❌ Error finding strategies by type {strategy_type}: {e}")
            raise RepositoryError(f"Failed to find strategies by type: {e}")

    async def find_active_strategies(self) -> List[Strategy]:
        """Récupère toutes les stratégies actives."""
        return await self.find_by_status(StrategyStatus.ACTIVE)

    async def find_by_universe(self, symbol: str) -> List[Strategy]:
        """Trouve les stratégies qui tradent un symbole donné."""
        try:
            matching_strategies = [
                self._deep_copy_strategy(s)
                for s in self._strategies.values()
                if symbol in s.universe
            ]
            return matching_strategies

        except Exception as e:
            logger.error(f"❌ Error finding strategies by universe symbol '{symbol}': {e}")
            raise RepositoryError(f"Failed to find strategies by universe: {e}")

    async def update(self, strategy: Strategy) -> None:
        """Met à jour une stratégie existante."""
        try:
            if strategy.id not in self._strategies:
                raise StrategyNotFoundError(f"Strategy {strategy.id} not found")

            # Vérifier le conflit de nom avec d'autres stratégies
            existing_by_name = await self.find_by_name(strategy.name)
            if existing_by_name and existing_by_name.id != strategy.id:
                raise DuplicateStrategyError(f"Strategy with name '{strategy.name}' already exists")

            # Mettre à jour
            strategy_copy = self._deep_copy_strategy(strategy)
            strategy_copy.updated_at = datetime.utcnow()
            self._strategies[strategy.id] = strategy_copy

            logger.debug(f"🔄 Strategy updated: {strategy.id}")

        except (StrategyNotFoundError, DuplicateStrategyError):
            raise  # Re-raise les exceptions métier
        except Exception as e:
            logger.error(f"❌ Error updating strategy {strategy.id}: {e}")
            raise RepositoryError(f"Failed to update strategy: {e}")

    async def delete(self, strategy_id: str) -> bool:
        """Supprime une stratégie."""
        try:
            if strategy_id in self._strategies:
                del self._strategies[strategy_id]
                logger.debug(f"🗑️ Strategy deleted: {strategy_id}")
                return True
            else:
                logger.warning(f"⚠️ Strategy {strategy_id} not found for deletion")
                return False

        except Exception as e:
            logger.error(f"❌ Error deleting strategy {strategy_id}: {e}")
            raise RepositoryError(f"Failed to delete strategy: {e}")

    async def exists(self, strategy_id: str) -> bool:
        """Vérifie si une stratégie existe."""
        return strategy_id in self._strategies

    async def count(self) -> int:
        """Compte le nombre total de stratégies."""
        return len(self._strategies)

    async def count_by_status(self, status: StrategyStatus) -> int:
        """Compte les stratégies par statut."""
        return len([s for s in self._strategies.values() if s.status == status])

    async def find_with_performance_above(
        self,
        min_sharpe: float,
        min_return: float
    ) -> List[Strategy]:
        """Trouve les stratégies avec performance minimale."""
        try:
            matching_strategies = []

            for strategy in self._strategies.values():
                if (strategy.sharpe_ratio and strategy.sharpe_ratio >= min_sharpe and
                    strategy.total_pnl >= min_return):
                    matching_strategies.append(self._deep_copy_strategy(strategy))

            return matching_strategies

        except Exception as e:
            logger.error(f"❌ Error finding high-performance strategies: {e}")
            raise RepositoryError(f"Failed to find high-performance strategies: {e}")

    async def find_created_between(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Strategy]:
        """Trouve les stratégies créées dans une période."""
        try:
            matching_strategies = [
                self._deep_copy_strategy(s)
                for s in self._strategies.values()
                if start_date <= s.created_at <= end_date
            ]
            return matching_strategies

        except Exception as e:
            logger.error(f"❌ Error finding strategies by creation date: {e}")
            raise RepositoryError(f"Failed to find strategies by creation date: {e}")

    async def find_updated_since(self, since: datetime) -> List[Strategy]:
        """Trouve les stratégies modifiées depuis une date."""
        try:
            matching_strategies = [
                self._deep_copy_strategy(s)
                for s in self._strategies.values()
                if s.updated_at >= since
            ]
            return matching_strategies

        except Exception as e:
            logger.error(f"❌ Error finding strategies updated since {since}: {e}")
            raise RepositoryError(f"Failed to find strategies updated since date: {e}")

    async def bulk_update_status(
        self,
        strategy_ids: List[str],
        new_status: StrategyStatus
    ) -> int:
        """Met à jour le statut de plusieurs stratégies."""
        try:
            updated_count = 0

            for strategy_id in strategy_ids:
                if strategy_id in self._strategies:
                    strategy = self._strategies[strategy_id]

                    # Créer une copie mise à jour
                    updated_strategy = self._deep_copy_strategy(strategy)
                    updated_strategy.status = new_status
                    updated_strategy.updated_at = datetime.utcnow()

                    self._strategies[strategy_id] = updated_strategy
                    updated_count += 1

            logger.debug(f"📦 Bulk updated {updated_count} strategies to status {new_status}")
            return updated_count

        except Exception as e:
            logger.error(f"❌ Error bulk updating strategies: {e}")
            raise RepositoryError(f"Failed to bulk update strategies: {e}")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Strategy]:
        """Recherche de stratégies avec critères."""
        try:
            query_lower = query.lower()
            matching_strategies = []

            for strategy in self._strategies.values():
                # Recherche dans le nom et la description
                if (query_lower in strategy.name.lower() or
                    query_lower in strategy.description.lower()):

                    # Appliquer les filtres additionnels
                    if self._matches_filters(strategy, filters):
                        matching_strategies.append(self._deep_copy_strategy(strategy))

            return matching_strategies

        except Exception as e:
            logger.error(f"❌ Error searching strategies: {e}")
            raise RepositoryError(f"Failed to search strategies: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques globales des stratégies."""
        try:
            total_strategies = len(self._strategies)

            if total_strategies == 0:
                return {
                    "total_strategies": 0,
                    "active_strategies": 0,
                    "avg_sharpe_ratio": 0,
                    "total_trades": 0
                }

            active_count = len([s for s in self._strategies.values()
                              if s.status == StrategyStatus.ACTIVE])

            # Statistiques de performance
            strategies_with_sharpe = [s for s in self._strategies.values()
                                    if s.sharpe_ratio is not None]
            avg_sharpe = (sum(s.sharpe_ratio for s in strategies_with_sharpe) /
                         len(strategies_with_sharpe)) if strategies_with_sharpe else 0

            total_trades = sum(s.total_trades for s in self._strategies.values())
            total_pnl = sum(s.total_pnl for s in self._strategies.values())

            # Distribution par statut
            status_distribution = {}
            for status in StrategyStatus:
                count = len([s for s in self._strategies.values() if s.status == status])
                status_distribution[status.value] = count

            # Distribution par type
            type_distribution = {}
            for strategy_type in StrategyType:
                count = len([s for s in self._strategies.values() if s.strategy_type == strategy_type])
                type_distribution[strategy_type.value] = count

            return {
                "total_strategies": total_strategies,
                "active_strategies": active_count,
                "avg_sharpe_ratio": float(avg_sharpe),
                "total_trades": total_trades,
                "total_pnl": float(total_pnl),
                "status_distribution": status_distribution,
                "type_distribution": type_distribution,
                "avg_trades_per_strategy": total_trades / total_strategies,
                "avg_pnl_per_strategy": float(total_pnl) / total_strategies
            }

        except Exception as e:
            logger.error(f"❌ Error getting strategy statistics: {e}")
            raise RepositoryError(f"Failed to get strategy statistics: {e}")

    async def backup_strategy(self, strategy_id: str) -> str:
        """Crée une sauvegarde d'une stratégie."""
        try:
            if strategy_id not in self._strategies:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

            backup_id = str(uuid.uuid4())
            strategy_copy = self._deep_copy_strategy(self._strategies[strategy_id])
            self._backups[backup_id] = strategy_copy

            logger.debug(f"💾 Strategy backup created: {backup_id}")
            return backup_id

        except StrategyNotFoundError:
            raise
        except Exception as e:
            logger.error(f"❌ Error backing up strategy {strategy_id}: {e}")
            raise RepositoryError(f"Failed to backup strategy: {e}")

    async def restore_strategy(self, backup_id: str) -> Strategy:
        """Restaure une stratégie depuis une sauvegarde."""
        try:
            if backup_id not in self._backups:
                raise RepositoryError(f"Backup {backup_id} not found")

            restored_strategy = self._deep_copy_strategy(self._backups[backup_id])

            # Générer un nouvel ID pour éviter les conflits
            restored_strategy.id = str(uuid.uuid4())
            restored_strategy.created_at = datetime.utcnow()
            restored_strategy.updated_at = datetime.utcnow()

            # Sauvegarder la stratégie restaurée
            await self.save(restored_strategy)

            logger.debug(f"🔄 Strategy restored from backup {backup_id}: {restored_strategy.id}")
            return restored_strategy

        except Exception as e:
            logger.error(f"❌ Error restoring strategy from backup {backup_id}: {e}")
            raise RepositoryError(f"Failed to restore strategy: {e}")

    def _deep_copy_strategy(self, strategy: Strategy) -> Strategy:
        """Crée une copie profonde d'une stratégie."""
        if strategy is None:
            return None

        # Utiliser from_dict/to_dict pour une copie profonde
        return Strategy.from_dict(strategy.to_dict())

    def _matches_filters(self, strategy: Strategy, filters: Optional[Dict[str, Any]]) -> bool:
        """Vérifie si une stratégie correspond aux filtres."""
        if not filters:
            return True

        for key, value in filters.items():
            if key == "status" and strategy.status.value != value:
                return False
            elif key == "type" and strategy.strategy_type.value != value:
                return False
            elif key == "min_sharpe" and (not strategy.sharpe_ratio or strategy.sharpe_ratio < value):
                return False
            elif key == "symbol" and value not in strategy.universe:
                return False

        return True

    # Méthodes utilitaires pour les tests

    def clear(self) -> None:
        """Vide le repository (pour les tests)."""
        self._strategies.clear()
        self._backups.clear()

    def get_all_ids(self) -> List[str]:
        """Retourne tous les IDs de stratégies (pour les tests)."""
        return list(self._strategies.keys())

    def get_backup_count(self) -> int:
        """Retourne le nombre de sauvegardes (pour les tests)."""
        return len(self._backups)