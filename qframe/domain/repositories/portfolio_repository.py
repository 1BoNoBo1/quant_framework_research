"""
Repository Interface: Portfolio Repository
=========================================

Interface de repository pour la persistance des portfolios.
Définit le contrat pour l'accès aux données selon l'architecture hexagonale.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

from ..entities.portfolio import Portfolio, PortfolioStatus, PortfolioType
from ..value_objects.position import Position


class PortfolioRepository(ABC):
    """
    Interface de repository pour les portfolios.

    Définit les opérations de persistance et de requête pour les
    portfolios selon les principes DDD.
    """

    @abstractmethod
    async def save(self, portfolio: Portfolio) -> None:
        """
        Sauvegarde un portfolio.

        Args:
            portfolio: Portfolio à sauvegarder

        Raises:
            RepositoryError: Si la sauvegarde échoue
        """
        pass

    @abstractmethod
    async def find_by_id(self, portfolio_id: str) -> Optional[Portfolio]:
        """
        Trouve un portfolio par son ID.

        Args:
            portfolio_id: ID du portfolio

        Returns:
            Portfolio trouvé ou None
        """
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Portfolio]:
        """
        Trouve un portfolio par son nom.

        Args:
            name: Nom du portfolio

        Returns:
            Portfolio trouvé ou None
        """
        pass

    @abstractmethod
    async def find_all(self) -> List[Portfolio]:
        """
        Retourne tous les portfolios.

        Returns:
            Liste de tous les portfolios
        """
        pass

    @abstractmethod
    async def find_by_status(self, status: PortfolioStatus) -> List[Portfolio]:
        """
        Trouve tous les portfolios avec un statut donné.

        Args:
            status: Statut recherché

        Returns:
            Liste des portfolios avec ce statut
        """
        pass

    @abstractmethod
    async def find_by_type(self, portfolio_type: PortfolioType) -> List[Portfolio]:
        """
        Trouve tous les portfolios d'un type donné.

        Args:
            portfolio_type: Type de portfolio recherché

        Returns:
            Liste des portfolios de ce type
        """
        pass

    @abstractmethod
    async def find_active_portfolios(self) -> List[Portfolio]:
        """
        Trouve tous les portfolios actifs.

        Returns:
            Liste des portfolios actifs
        """
        pass

    @abstractmethod
    async def find_by_strategy(self, strategy_id: str) -> List[Portfolio]:
        """
        Trouve tous les portfolios utilisant une stratégie donnée.

        Args:
            strategy_id: ID de la stratégie

        Returns:
            Liste des portfolios utilisant cette stratégie
        """
        pass

    @abstractmethod
    async def find_by_symbol(self, symbol: str) -> List[Portfolio]:
        """
        Trouve tous les portfolios ayant une position sur un symbole.

        Args:
            symbol: Symbole recherché

        Returns:
            Liste des portfolios ayant une position sur ce symbole
        """
        pass

    @abstractmethod
    async def find_portfolios_needing_rebalancing(self, threshold: Decimal = Decimal("0.05")) -> List[Portfolio]:
        """
        Trouve tous les portfolios nécessitant un rééquilibrage.

        Args:
            threshold: Seuil d'écart pour déclencher le rééquilibrage

        Returns:
            Liste des portfolios à rééquilibrer
        """
        pass

    @abstractmethod
    async def find_portfolios_violating_constraints(self) -> List[Portfolio]:
        """
        Trouve tous les portfolios violant leurs contraintes.

        Returns:
            Liste des portfolios en violation
        """
        pass

    @abstractmethod
    async def find_by_value_range(
        self,
        min_value: Optional[Decimal] = None,
        max_value: Optional[Decimal] = None
    ) -> List[Portfolio]:
        """
        Trouve tous les portfolios dans une fourchette de valeur.

        Args:
            min_value: Valeur minimale (optionnel)
            max_value: Valeur maximale (optionnel)

        Returns:
            Liste des portfolios dans la fourchette
        """
        pass

    @abstractmethod
    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        date_field: str = "created_at"
    ) -> List[Portfolio]:
        """
        Trouve tous les portfolios dans une période donnée.

        Args:
            start_date: Date de début
            end_date: Date de fin
            date_field: Champ de date à utiliser (created_at, updated_at, etc.)

        Returns:
            Liste des portfolios dans la période
        """
        pass

    @abstractmethod
    async def update(self, portfolio: Portfolio) -> None:
        """
        Met à jour un portfolio existant.

        Args:
            portfolio: Portfolio à mettre à jour

        Raises:
            RepositoryError: Si le portfolio n'existe pas ou si la mise à jour échoue
        """
        pass

    @abstractmethod
    async def delete(self, portfolio_id: str) -> bool:
        """
        Supprime un portfolio.

        Args:
            portfolio_id: ID du portfolio à supprimer

        Returns:
            True si supprimé, False si introuvable

        Raises:
            RepositoryError: Si la suppression échoue
        """
        pass

    @abstractmethod
    async def archive(self, portfolio_id: str) -> bool:
        """
        Archive un portfolio (change son statut à ARCHIVED).

        Args:
            portfolio_id: ID du portfolio à archiver

        Returns:
            True si archivé, False si introuvable
        """
        pass

    @abstractmethod
    async def count_by_status(self) -> Dict[PortfolioStatus, int]:
        """
        Compte les portfolios par statut.

        Returns:
            Dictionnaire statut -> nombre de portfolios
        """
        pass

    @abstractmethod
    async def count_by_type(self) -> Dict[PortfolioType, int]:
        """
        Compte les portfolios par type.

        Returns:
            Dictionnaire type -> nombre de portfolios
        """
        pass

    @abstractmethod
    async def get_total_value_by_type(self) -> Dict[PortfolioType, Decimal]:
        """
        Calcule la valeur totale par type de portfolio.

        Returns:
            Dictionnaire type -> valeur totale
        """
        pass

    @abstractmethod
    async def get_portfolio_statistics(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """
        Calcule des statistiques pour un portfolio.

        Args:
            portfolio_id: ID du portfolio

        Returns:
            Dictionnaire de statistiques ou None si portfolio introuvable
        """
        pass

    @abstractmethod
    async def get_global_statistics(self) -> Dict[str, Any]:
        """
        Calcule des statistiques globales sur tous les portfolios.

        Returns:
            Dictionnaire de statistiques globales
        """
        pass

    @abstractmethod
    async def update_portfolio_snapshot(self, portfolio_id: str) -> None:
        """
        Met à jour le snapshot d'un portfolio et l'ajoute à l'historique.

        Args:
            portfolio_id: ID du portfolio à snapshotter

        Raises:
            RepositoryError: Si le portfolio n'existe pas
        """
        pass

    @abstractmethod
    async def bulk_update_snapshots(self, portfolio_ids: Optional[List[str]] = None) -> int:
        """
        Met à jour les snapshots de plusieurs portfolios en une fois.

        Args:
            portfolio_ids: IDs des portfolios (tous si None)

        Returns:
            Nombre de portfolios mis à jour
        """
        pass

    @abstractmethod
    async def cleanup_old_snapshots(
        self,
        retention_days: int = 365,
        max_snapshots_per_portfolio: int = 1000
    ) -> int:
        """
        Nettoie les anciens snapshots pour optimiser le stockage.

        Args:
            retention_days: Nombre de jours à conserver
            max_snapshots_per_portfolio: Nombre maximum de snapshots par portfolio

        Returns:
            Nombre de snapshots supprimés
        """
        pass


class PortfolioQuery:
    """
    Classe utilitaire pour construire des requêtes complexes sur les portfolios.
    """

    def __init__(self):
        self.filters = {}
        self.sort_by = None
        self.sort_desc = False
        self.limit = None
        self.offset = 0

    def by_status(self, status: PortfolioStatus) -> 'PortfolioQuery':
        """Filtre par statut"""
        self.filters['status'] = status
        return self

    def by_type(self, portfolio_type: PortfolioType) -> 'PortfolioQuery':
        """Filtre par type"""
        self.filters['type'] = portfolio_type
        return self

    def by_name_pattern(self, pattern: str) -> 'PortfolioQuery':
        """Filtre par motif dans le nom"""
        self.filters['name_pattern'] = pattern
        return self

    def by_value_range(self, min_value: Optional[Decimal], max_value: Optional[Decimal]) -> 'PortfolioQuery':
        """Filtre par fourchette de valeur"""
        if min_value is not None:
            self.filters['min_value'] = min_value
        if max_value is not None:
            self.filters['max_value'] = max_value
        return self

    def by_position_count_range(self, min_count: Optional[int], max_count: Optional[int]) -> 'PortfolioQuery':
        """Filtre par nombre de positions"""
        if min_count is not None:
            self.filters['min_positions'] = min_count
        if max_count is not None:
            self.filters['max_positions'] = max_count
        return self

    def with_strategy(self, strategy_id: str) -> 'PortfolioQuery':
        """Filtre les portfolios utilisant une stratégie"""
        self.filters['strategy_id'] = strategy_id
        return self

    def with_symbol(self, symbol: str) -> 'PortfolioQuery':
        """Filtre les portfolios ayant une position sur un symbole"""
        self.filters['symbol'] = symbol
        return self

    def active_only(self) -> 'PortfolioQuery':
        """Filtre uniquement les portfolios actifs"""
        self.filters['active_only'] = True
        return self

    def needs_rebalancing(self, threshold: Decimal = Decimal("0.05")) -> 'PortfolioQuery':
        """Filtre les portfolios nécessitant un rééquilibrage"""
        self.filters['needs_rebalancing'] = threshold
        return self

    def violating_constraints(self) -> 'PortfolioQuery':
        """Filtre les portfolios violant leurs contraintes"""
        self.filters['violating_constraints'] = True
        return self

    def created_after(self, date: datetime) -> 'PortfolioQuery':
        """Filtre les portfolios créés après une date"""
        self.filters['created_after'] = date
        return self

    def updated_after(self, date: datetime) -> 'PortfolioQuery':
        """Filtre les portfolios mis à jour après une date"""
        self.filters['updated_after'] = date
        return self

    def sort_by_value(self, descending: bool = True) -> 'PortfolioQuery':
        """Trie par valeur du portfolio"""
        self.sort_by = 'total_value'
        self.sort_desc = descending
        return self

    def sort_by_return(self, descending: bool = True) -> 'PortfolioQuery':
        """Trie par rendement"""
        self.sort_by = 'return'
        self.sort_desc = descending
        return self

    def sort_by_creation_date(self, descending: bool = True) -> 'PortfolioQuery':
        """Trie par date de création"""
        self.sort_by = 'created_at'
        self.sort_desc = descending
        return self

    def sort_by_update_date(self, descending: bool = True) -> 'PortfolioQuery':
        """Trie par date de mise à jour"""
        self.sort_by = 'updated_at'
        self.sort_desc = descending
        return self

    def paginate(self, limit: int, offset: int = 0) -> 'PortfolioQuery':
        """Ajoute la pagination"""
        self.limit = limit
        self.offset = offset
        return self


class PortfolioAggregateQuery:
    """
    Classe pour construire des requêtes d'agrégation sur les portfolios.
    """

    def __init__(self):
        self.group_by = []
        self.aggregations = []
        self.filters = {}

    def group_by_status(self) -> 'PortfolioAggregateQuery':
        """Grouper par statut"""
        self.group_by.append('status')
        return self

    def group_by_type(self) -> 'PortfolioAggregateQuery':
        """Grouper par type"""
        self.group_by.append('type')
        return self

    def group_by_date(self, date_field: str = 'created_at', interval: str = 'day') -> 'PortfolioAggregateQuery':
        """Grouper par date"""
        self.group_by.append(f'{date_field}_{interval}')
        return self

    def count(self) -> 'PortfolioAggregateQuery':
        """Compter les portfolios"""
        self.aggregations.append('count')
        return self

    def sum_value(self) -> 'PortfolioAggregateQuery':
        """Sommer les valeurs"""
        self.aggregations.append('sum_value')
        return self

    def avg_value(self) -> 'PortfolioAggregateQuery':
        """Moyenne des valeurs"""
        self.aggregations.append('avg_value')
        return self

    def avg_return(self) -> 'PortfolioAggregateQuery':
        """Moyenne des rendements"""
        self.aggregations.append('avg_return')
        return self

    def filter_by_type(self, portfolio_type: PortfolioType) -> 'PortfolioAggregateQuery':
        """Filtrer par type"""
        self.filters['type'] = portfolio_type
        return self

    def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> 'PortfolioAggregateQuery':
        """Filtrer par période"""
        self.filters['start_date'] = start_date
        self.filters['end_date'] = end_date
        return self


class RepositoryError(Exception):
    """Exception pour les erreurs de repository"""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class PortfolioNotFoundError(RepositoryError):
    """Exception quand un portfolio n'est pas trouvé"""

    def __init__(self, portfolio_id: str):
        super().__init__(f"Portfolio not found: {portfolio_id}")
        self.portfolio_id = portfolio_id


class PortfolioAlreadyExistsError(RepositoryError):
    """Exception quand un portfolio existe déjà"""

    def __init__(self, identifier: str):
        super().__init__(f"Portfolio already exists: {identifier}")
        self.identifier = identifier


class DuplicatePortfolioNameError(RepositoryError):
    """Exception quand le nom de portfolio existe déjà"""

    def __init__(self, name: str):
        super().__init__(f"Portfolio name already exists: {name}")
        self.name = name