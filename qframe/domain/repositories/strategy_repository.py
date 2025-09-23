"""
Repository Interface: StrategyRepository
======================================

Interface abstraite pour la persistance des stratégies.
Définit le contrat sans dépendance à l'implémentation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..entities.strategy import Strategy, StrategyStatus, StrategyType


class StrategyRepository(ABC):
    """
    Interface abstraite pour la persistance des stratégies.

    Définit tous les contrats nécessaires pour la gestion
    des stratégies sans être couplé à une implémentation
    spécifique (base de données, fichiers, etc.).
    """

    @abstractmethod
    async def save(self, strategy: Strategy) -> None:
        """
        Sauvegarde une stratégie.

        Args:
            strategy: Stratégie à sauvegarder

        Raises:
            RepositoryError: En cas d'erreur de persistance
        """
        pass

    @abstractmethod
    async def find_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """
        Trouve une stratégie par son ID.

        Args:
            strategy_id: ID unique de la stratégie

        Returns:
            Strategy si trouvée, None sinon

        Raises:
            RepositoryError: En cas d'erreur d'accès
        """
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Strategy]:
        """
        Trouve une stratégie par son nom.

        Args:
            name: Nom de la stratégie

        Returns:
            Strategy si trouvée, None sinon
        """
        pass

    @abstractmethod
    async def find_all(self) -> List[Strategy]:
        """
        Récupère toutes les stratégies.

        Returns:
            Liste de toutes les stratégies
        """
        pass

    @abstractmethod
    async def find_by_status(self, status: StrategyStatus) -> List[Strategy]:
        """
        Trouve les stratégies par statut.

        Args:
            status: Statut recherché

        Returns:
            Liste des stratégies avec ce statut
        """
        pass

    @abstractmethod
    async def find_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        """
        Trouve les stratégies par type.

        Args:
            strategy_type: Type de stratégie recherché

        Returns:
            Liste des stratégies de ce type
        """
        pass

    @abstractmethod
    async def find_active_strategies(self) -> List[Strategy]:
        """
        Récupère toutes les stratégies actives.

        Returns:
            Liste des stratégies avec status ACTIVE
        """
        pass

    @abstractmethod
    async def find_by_universe(self, symbol: str) -> List[Strategy]:
        """
        Trouve les stratégies qui tradent un symbole donné.

        Args:
            symbol: Symbole recherché (ex: "BTC/USDT")

        Returns:
            Liste des stratégies qui tradent ce symbole
        """
        pass

    @abstractmethod
    async def update(self, strategy: Strategy) -> None:
        """
        Met à jour une stratégie existante.

        Args:
            strategy: Stratégie avec les nouvelles données

        Raises:
            RepositoryError: Si la stratégie n'existe pas
        """
        pass

    @abstractmethod
    async def delete(self, strategy_id: str) -> bool:
        """
        Supprime une stratégie.

        Args:
            strategy_id: ID de la stratégie à supprimer

        Returns:
            True si supprimée, False si non trouvée

        Raises:
            RepositoryError: En cas d'erreur de suppression
        """
        pass

    @abstractmethod
    async def exists(self, strategy_id: str) -> bool:
        """
        Vérifie si une stratégie existe.

        Args:
            strategy_id: ID de la stratégie

        Returns:
            True si existe, False sinon
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """
        Compte le nombre total de stratégies.

        Returns:
            Nombre de stratégies
        """
        pass

    @abstractmethod
    async def count_by_status(self, status: StrategyStatus) -> int:
        """
        Compte les stratégies par statut.

        Args:
            status: Statut à compter

        Returns:
            Nombre de stratégies avec ce statut
        """
        pass

    @abstractmethod
    async def find_with_performance_above(
        self,
        min_sharpe: float,
        min_return: float
    ) -> List[Strategy]:
        """
        Trouve les stratégies avec performance minimale.

        Args:
            min_sharpe: Sharpe ratio minimum
            min_return: Rendement total minimum

        Returns:
            Liste des stratégies performantes
        """
        pass

    @abstractmethod
    async def find_created_between(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Strategy]:
        """
        Trouve les stratégies créées dans une période.

        Args:
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Liste des stratégies créées dans cette période
        """
        pass

    @abstractmethod
    async def find_updated_since(self, since: datetime) -> List[Strategy]:
        """
        Trouve les stratégies modifiées depuis une date.

        Args:
            since: Date de référence

        Returns:
            Liste des stratégies modifiées depuis cette date
        """
        pass

    @abstractmethod
    async def bulk_update_status(
        self,
        strategy_ids: List[str],
        new_status: StrategyStatus
    ) -> int:
        """
        Met à jour le statut de plusieurs stratégies.

        Args:
            strategy_ids: Liste des IDs des stratégies
            new_status: Nouveau statut à appliquer

        Returns:
            Nombre de stratégies mises à jour
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Strategy]:
        """
        Recherche de stratégies avec critères.

        Args:
            query: Terme de recherche (nom, description)
            filters: Filtres optionnels (type, statut, etc.)

        Returns:
            Liste des stratégies correspondantes
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Récupère les statistiques globales des stratégies.

        Returns:
            Dictionnaire avec les statistiques:
            - total_strategies
            - active_strategies
            - avg_sharpe_ratio
            - total_trades
            - etc.
        """
        pass

    @abstractmethod
    async def backup_strategy(self, strategy_id: str) -> str:
        """
        Crée une sauvegarde d'une stratégie.

        Args:
            strategy_id: ID de la stratégie à sauvegarder

        Returns:
            ID ou chemin de la sauvegarde

        Raises:
            RepositoryError: Si la stratégie n'existe pas
        """
        pass

    @abstractmethod
    async def restore_strategy(self, backup_id: str) -> Strategy:
        """
        Restaure une stratégie depuis une sauvegarde.

        Args:
            backup_id: ID de la sauvegarde

        Returns:
            Stratégie restaurée

        Raises:
            RepositoryError: Si la sauvegarde n'existe pas
        """
        pass


class RepositoryError(Exception):
    """Exception levée par les repositories en cas d'erreur."""
    pass


class StrategyNotFoundError(RepositoryError):
    """Exception levée quand une stratégie n'est pas trouvée."""
    pass


class DuplicateStrategyError(RepositoryError):
    """Exception levée en cas de tentative de création d'une stratégie existante."""
    pass