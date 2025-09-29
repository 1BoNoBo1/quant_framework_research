"""
Infrastructure: MemoryPortfolioRepository
========================================

Implémentation en mémoire du repository de portfolios.
Utilisée pour les tests et le développement.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal
import uuid
import logging
import copy

from ...domain.entities.portfolio import Portfolio, PortfolioStatus, PortfolioType
from ...domain.value_objects.position import Position
from ...domain.repositories.portfolio_repository import (
    PortfolioRepository,
    RepositoryError,
    PortfolioNotFoundError,
    PortfolioAlreadyExistsError,
    DuplicatePortfolioNameError
)

logger = logging.getLogger(__name__)


class MemoryPortfolioRepository(PortfolioRepository):
    """
    Implémentation en mémoire du repository de portfolios.

    Stocke les portfolios dans un dictionnaire en mémoire.
    Idéal pour les tests et le prototypage.
    """

    def __init__(self):
        self._portfolios: Dict[str, Portfolio] = {}
        self._snapshots: Dict[str, List[Dict[str, Any]]] = {}

    async def save(self, portfolio: Portfolio) -> None:
        """Sauvegarde un portfolio en mémoire."""
        try:
            # Vérifier si un portfolio avec ce nom existe déjà (sauf si c'est une mise à jour)
            existing_by_name = await self.find_by_name(portfolio.name)
            if existing_by_name and existing_by_name.id != portfolio.id:
                raise DuplicatePortfolioNameError(portfolio.name)

            # Créer une copie pour éviter les modifications externes
            portfolio_copy = self._deep_copy_portfolio(portfolio)
            self._portfolios[portfolio.id] = portfolio_copy

            logger.debug(f"💾 Portfolio saved: {portfolio.id}")

        except Exception as e:
            if isinstance(e, (DuplicatePortfolioNameError, PortfolioAlreadyExistsError)):
                raise
            raise RepositoryError(f"Failed to save portfolio {portfolio.id}", e)

    async def find_by_id(self, portfolio_id: str) -> Optional[Portfolio]:
        """Trouve un portfolio par son ID."""
        portfolio = self._portfolios.get(portfolio_id)
        return self._deep_copy_portfolio(portfolio) if portfolio else None

    async def find_by_name(self, name: str) -> Optional[Portfolio]:
        """Trouve un portfolio par son nom."""
        for portfolio in self._portfolios.values():
            if portfolio.name == name:
                return self._deep_copy_portfolio(portfolio)
        return None

    async def find_all(self) -> List[Portfolio]:
        """Retourne tous les portfolios."""
        return [self._deep_copy_portfolio(p) for p in self._portfolios.values()]

    async def find_by_status(self, status: PortfolioStatus) -> List[Portfolio]:
        """Trouve tous les portfolios avec un statut donné."""
        return [
            self._deep_copy_portfolio(p)
            for p in self._portfolios.values()
            if p.status == status
        ]

    async def find_by_type(self, portfolio_type: PortfolioType) -> List[Portfolio]:
        """Trouve tous les portfolios d'un type donné."""
        return [
            self._deep_copy_portfolio(p)
            for p in self._portfolios.values()
            if p.portfolio_type == portfolio_type
        ]

    async def find_active_portfolios(self) -> List[Portfolio]:
        """Trouve tous les portfolios actifs."""
        return await self.find_by_status(PortfolioStatus.ACTIVE)

    async def find_by_strategy(self, strategy_id: str) -> List[Portfolio]:
        """Trouve tous les portfolios utilisant une stratégie donnée."""
        return [
            self._deep_copy_portfolio(p)
            for p in self._portfolios.values()
            if hasattr(p, 'strategy_id') and p.strategy_id == strategy_id
        ]

    async def find_by_symbol(self, symbol: str) -> List[Portfolio]:
        """Trouve tous les portfolios ayant une position sur un symbole."""
        result = []
        for portfolio in self._portfolios.values():
            if hasattr(portfolio, 'positions'):
                for position in portfolio.positions.values():
                    if position.symbol == symbol:
                        result.append(self._deep_copy_portfolio(portfolio))
                        break
        return result

    async def find_portfolios_needing_rebalancing(self, threshold: Decimal = Decimal("0.05")) -> List[Portfolio]:
        """Trouve tous les portfolios nécessitant un rééquilibrage."""
        result = []
        for portfolio in self._portfolios.values():
            # Logique simplifiée: un portfolio a besoin d'un rééquilibrage si
            # la différence entre la valeur actuelle et la valeur cible dépasse le seuil
            if hasattr(portfolio, 'needs_rebalancing') and portfolio.needs_rebalancing(threshold):
                result.append(self._deep_copy_portfolio(portfolio))
        return result

    async def find_portfolios_violating_constraints(self) -> List[Portfolio]:
        """Trouve tous les portfolios violant leurs contraintes."""
        result = []
        for portfolio in self._portfolios.values():
            if hasattr(portfolio, 'violates_constraints') and portfolio.violates_constraints():
                result.append(self._deep_copy_portfolio(portfolio))
        return result

    async def find_by_value_range(
        self,
        min_value: Optional[Decimal] = None,
        max_value: Optional[Decimal] = None
    ) -> List[Portfolio]:
        """Trouve tous les portfolios dans une fourchette de valeur."""
        result = []
        for portfolio in self._portfolios.values():
            value = portfolio.total_value
            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue
            result.append(self._deep_copy_portfolio(portfolio))
        return result

    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        date_field: str = "created_at"
    ) -> List[Portfolio]:
        """Trouve tous les portfolios dans une période donnée."""
        result = []
        for portfolio in self._portfolios.values():
            date_value = getattr(portfolio, date_field, None)
            if date_value and start_date <= date_value <= end_date:
                result.append(self._deep_copy_portfolio(portfolio))
        return result

    async def update(self, portfolio: Portfolio) -> None:
        """Met à jour un portfolio existant."""
        if portfolio.id not in self._portfolios:
            raise PortfolioNotFoundError(portfolio.id)

        # Vérifier les contraintes de nom unique
        existing_by_name = await self.find_by_name(portfolio.name)
        if existing_by_name and existing_by_name.id != portfolio.id:
            raise DuplicatePortfolioNameError(portfolio.name)

        try:
            portfolio_copy = self._deep_copy_portfolio(portfolio)
            portfolio_copy.updated_at = datetime.utcnow()
            self._portfolios[portfolio.id] = portfolio_copy
            logger.debug(f"🔄 Portfolio updated: {portfolio.id}")
        except Exception as e:
            raise RepositoryError(f"Failed to update portfolio {portfolio.id}", e)

    async def delete(self, portfolio_id: str) -> bool:
        """Supprime un portfolio."""
        if portfolio_id in self._portfolios:
            del self._portfolios[portfolio_id]
            if portfolio_id in self._snapshots:
                del self._snapshots[portfolio_id]
            logger.debug(f"🗑️ Portfolio deleted: {portfolio_id}")
            return True
        return False

    async def archive(self, portfolio_id: str) -> bool:
        """Archive un portfolio (change son statut à ARCHIVED)."""
        portfolio = self._portfolios.get(portfolio_id)
        if portfolio:
            portfolio.status = PortfolioStatus.ARCHIVED
            portfolio.updated_at = datetime.utcnow()
            logger.debug(f"📦 Portfolio archived: {portfolio_id}")
            return True
        return False

    async def count_by_status(self) -> Dict[PortfolioStatus, int]:
        """Compte les portfolios par statut."""
        counts = {}
        for portfolio in self._portfolios.values():
            status = portfolio.status
            counts[status] = counts.get(status, 0) + 1
        return counts

    async def count_by_type(self) -> Dict[PortfolioType, int]:
        """Compte les portfolios par type."""
        counts = {}
        for portfolio in self._portfolios.values():
            portfolio_type = portfolio.portfolio_type
            counts[portfolio_type] = counts.get(portfolio_type, 0) + 1
        return counts

    async def get_total_value_by_type(self) -> Dict[PortfolioType, Decimal]:
        """Calcule la valeur totale par type de portfolio."""
        totals = {}
        for portfolio in self._portfolios.values():
            portfolio_type = portfolio.portfolio_type
            totals[portfolio_type] = totals.get(portfolio_type, Decimal("0")) + portfolio.total_value
        return totals

    async def get_portfolio_statistics(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Calcule des statistiques pour un portfolio."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return None

        return {
            "id": portfolio.id,
            "name": portfolio.name,
            "total_value": portfolio.total_value,
            "cash_balance": portfolio.cash_balance,
            "position_count": len(portfolio.positions) if hasattr(portfolio, 'positions') else 0,
            "status": portfolio.status.value,
            "type": portfolio.portfolio_type.value,
            "created_at": portfolio.created_at.isoformat(),
            "updated_at": portfolio.updated_at.isoformat() if portfolio.updated_at else None,
        }

    async def get_global_statistics(self) -> Dict[str, Any]:
        """Calcule des statistiques globales sur tous les portfolios."""
        if not self._portfolios:
            return {
                "total_portfolios": 0,
                "total_value": Decimal("0"),
                "average_value": Decimal("0"),
                "status_counts": {},
                "type_counts": {}
            }

        total_value = sum(p.total_value for p in self._portfolios.values())
        count = len(self._portfolios)

        return {
            "total_portfolios": count,
            "total_value": total_value,
            "average_value": total_value / count if count > 0 else Decimal("0"),
            "status_counts": await self.count_by_status(),
            "type_counts": await self.count_by_type()
        }

    async def update_portfolio_snapshot(self, portfolio_id: str) -> None:
        """Met à jour le snapshot d'un portfolio et l'ajoute à l'historique."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            raise PortfolioNotFoundError(portfolio_id)

        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_value": float(portfolio.total_value),
            "cash_balance": float(portfolio.cash_balance),
            "position_count": len(portfolio.positions) if hasattr(portfolio, 'positions') else 0,
            "status": portfolio.status.value
        }

        if portfolio_id not in self._snapshots:
            self._snapshots[portfolio_id] = []

        self._snapshots[portfolio_id].append(snapshot)
        logger.debug(f"📸 Portfolio snapshot created: {portfolio_id}")

    async def bulk_update_snapshots(self, portfolio_ids: Optional[List[str]] = None) -> int:
        """Met à jour les snapshots de plusieurs portfolios en une fois."""
        ids_to_update = portfolio_ids or list(self._portfolios.keys())
        updated_count = 0

        for portfolio_id in ids_to_update:
            if portfolio_id in self._portfolios:
                await self.update_portfolio_snapshot(portfolio_id)
                updated_count += 1

        logger.debug(f"📸 Bulk snapshots updated: {updated_count}")
        return updated_count

    async def cleanup_old_snapshots(
        self,
        retention_days: int = 365,
        max_snapshots_per_portfolio: int = 1000
    ) -> int:
        """Nettoie les anciens snapshots pour optimiser le stockage."""
        deleted_count = 0
        cutoff_date = datetime.utcnow().timestamp() - (retention_days * 24 * 60 * 60)

        for portfolio_id, snapshots in self._snapshots.items():
            original_count = len(snapshots)

            # Filtrer par date et limiter le nombre
            filtered_snapshots = [
                s for s in snapshots
                if datetime.fromisoformat(s["timestamp"]).timestamp() > cutoff_date
            ]

            # Garder seulement les plus récents si trop nombreux
            if len(filtered_snapshots) > max_snapshots_per_portfolio:
                filtered_snapshots = sorted(
                    filtered_snapshots,
                    key=lambda x: x["timestamp"],
                    reverse=True
                )[:max_snapshots_per_portfolio]

            self._snapshots[portfolio_id] = filtered_snapshots
            deleted_count += original_count - len(filtered_snapshots)

        logger.debug(f"🧹 Cleaned up {deleted_count} old snapshots")
        return deleted_count

    def _deep_copy_portfolio(self, portfolio: Optional[Portfolio]) -> Optional[Portfolio]:
        """Crée une copie profonde d'un portfolio pour éviter les modifications externes."""
        if portfolio is None:
            return None

        try:
            return copy.deepcopy(portfolio)
        except Exception as e:
            logger.warning(f"⚠️ Could not deep copy portfolio {portfolio.id}: {e}")
            # Fallback: retourner l'original (pas idéal mais évite les crashes)
            return portfolio