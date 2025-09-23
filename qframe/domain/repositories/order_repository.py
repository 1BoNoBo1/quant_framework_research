"""
Repository Interface: Order Repository
=====================================

Interface de repository pour la persistance des ordres.
Définit le contrat pour l'accès aux données selon l'architecture hexagonale.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal

from ..entities.order import Order, OrderStatus, OrderType, OrderSide, TimeInForce, OrderPriority


class OrderRepository(ABC):
    """
    Interface de repository pour les ordres.

    Définit les opérations de persistance et de requête pour les
    ordres selon les principes DDD.
    """

    @abstractmethod
    async def save(self, order: Order) -> None:
        """
        Sauvegarde un ordre.

        Args:
            order: Ordre à sauvegarder

        Raises:
            RepositoryError: Si la sauvegarde échoue
        """
        pass

    @abstractmethod
    async def find_by_id(self, order_id: str) -> Optional[Order]:
        """
        Trouve un ordre par son ID.

        Args:
            order_id: ID de l'ordre

        Returns:
            Ordre trouvé ou None
        """
        pass

    @abstractmethod
    async def find_by_client_order_id(self, client_order_id: str) -> Optional[Order]:
        """
        Trouve un ordre par son ID client.

        Args:
            client_order_id: ID client de l'ordre

        Returns:
            Ordre trouvé ou None
        """
        pass

    @abstractmethod
    async def find_by_broker_order_id(self, broker_order_id: str) -> Optional[Order]:
        """
        Trouve un ordre par son ID broker.

        Args:
            broker_order_id: ID broker de l'ordre

        Returns:
            Ordre trouvé ou None
        """
        pass

    @abstractmethod
    async def find_by_status(self, status: OrderStatus) -> List[Order]:
        """
        Trouve tous les ordres avec un statut donné.

        Args:
            status: Statut recherché

        Returns:
            Liste des ordres avec ce statut
        """
        pass

    @abstractmethod
    async def find_active_orders(self) -> List[Order]:
        """
        Trouve tous les ordres actifs.

        Returns:
            Liste des ordres actifs (submitted, accepted, partially_filled)
        """
        pass

    @abstractmethod
    async def find_by_symbol(self, symbol: str) -> List[Order]:
        """
        Trouve tous les ordres pour un symbole donné.

        Args:
            symbol: Symbole recherché

        Returns:
            Liste des ordres pour ce symbole
        """
        pass

    @abstractmethod
    async def find_by_portfolio(self, portfolio_id: str) -> List[Order]:
        """
        Trouve tous les ordres d'un portfolio.

        Args:
            portfolio_id: ID du portfolio

        Returns:
            Liste des ordres du portfolio
        """
        pass

    @abstractmethod
    async def find_by_strategy(self, strategy_id: str) -> List[Order]:
        """
        Trouve tous les ordres d'une stratégie.

        Args:
            strategy_id: ID de la stratégie

        Returns:
            Liste des ordres de la stratégie
        """
        pass

    @abstractmethod
    async def find_by_symbol_and_side(self, symbol: str, side: OrderSide) -> List[Order]:
        """
        Trouve tous les ordres pour un symbole et un côté.

        Args:
            symbol: Symbole recherché
            side: Côté de l'ordre (BUY/SELL)

        Returns:
            Liste des ordres correspondants
        """
        pass

    @abstractmethod
    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        date_field: str = "created_time"
    ) -> List[Order]:
        """
        Trouve tous les ordres dans une période donnée.

        Args:
            start_date: Date de début
            end_date: Date de fin
            date_field: Champ de date à utiliser

        Returns:
            Liste des ordres dans la période
        """
        pass

    @abstractmethod
    async def find_expired_orders(self, current_time: Optional[datetime] = None) -> List[Order]:
        """
        Trouve tous les ordres expirés.

        Args:
            current_time: Timestamp actuel (utilise datetime.utcnow() si None)

        Returns:
            Liste des ordres expirés
        """
        pass

    @abstractmethod
    async def find_orders_by_priority(self, priority: OrderPriority) -> List[Order]:
        """
        Trouve tous les ordres avec une priorité donnée.

        Args:
            priority: Priorité recherchée

        Returns:
            Liste des ordres avec cette priorité
        """
        pass

    @abstractmethod
    async def find_orders_by_type(self, order_type: OrderType) -> List[Order]:
        """
        Trouve tous les ordres d'un type donné.

        Args:
            order_type: Type d'ordre recherché

        Returns:
            Liste des ordres de ce type
        """
        pass

    @abstractmethod
    async def find_orders_by_time_in_force(self, time_in_force: TimeInForce) -> List[Order]:
        """
        Trouve tous les ordres avec une durée de validité donnée.

        Args:
            time_in_force: Durée de validité recherchée

        Returns:
            Liste des ordres avec cette durée
        """
        pass

    @abstractmethod
    async def find_parent_orders(self) -> List[Order]:
        """
        Trouve tous les ordres parents (sans parent_order_id).

        Returns:
            Liste des ordres parents
        """
        pass

    @abstractmethod
    async def find_child_orders(self, parent_order_id: str) -> List[Order]:
        """
        Trouve tous les ordres enfants d'un ordre parent.

        Args:
            parent_order_id: ID de l'ordre parent

        Returns:
            Liste des ordres enfants
        """
        pass

    @abstractmethod
    async def find_orders_with_tag(self, tag_key: str, tag_value: Optional[str] = None) -> List[Order]:
        """
        Trouve tous les ordres avec un tag donné.

        Args:
            tag_key: Clé du tag
            tag_value: Valeur du tag (optionnel)

        Returns:
            Liste des ordres avec ce tag
        """
        pass

    @abstractmethod
    async def update(self, order: Order) -> None:
        """
        Met à jour un ordre existant.

        Args:
            order: Ordre à mettre à jour

        Raises:
            RepositoryError: Si l'ordre n'existe pas ou si la mise à jour échoue
        """
        pass

    @abstractmethod
    async def delete(self, order_id: str) -> bool:
        """
        Supprime un ordre.

        Args:
            order_id: ID de l'ordre à supprimer

        Returns:
            True si supprimé, False si introuvable

        Raises:
            RepositoryError: Si la suppression échoue
        """
        pass

    @abstractmethod
    async def count_by_status(self) -> Dict[OrderStatus, int]:
        """
        Compte les ordres par statut.

        Returns:
            Dictionnaire statut -> nombre d'ordres
        """
        pass

    @abstractmethod
    async def count_by_symbol(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Compte les ordres par symbole.

        Args:
            limit: Nombre maximum de symboles à retourner

        Returns:
            Dictionnaire symbole -> nombre d'ordres
        """
        pass

    @abstractmethod
    async def get_order_statistics(
        self,
        symbol: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les ordres.

        Args:
            symbol: Filtrer par symbole (optionnel)
            portfolio_id: Filtrer par portfolio (optionnel)
            strategy_id: Filtrer par stratégie (optionnel)

        Returns:
            Dictionnaire de statistiques
        """
        pass

    @abstractmethod
    async def get_execution_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calcule des statistiques d'exécution.

        Args:
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)

        Returns:
            Dictionnaire de statistiques d'exécution
        """
        pass

    @abstractmethod
    async def archive_old_orders(
        self,
        cutoff_date: datetime,
        statuses_to_archive: Optional[List[OrderStatus]] = None
    ) -> int:
        """
        Archive les anciens ordres.

        Args:
            cutoff_date: Date limite pour l'archivage
            statuses_to_archive: Statuts à archiver (tous les terminaux si None)

        Returns:
            Nombre d'ordres archivés
        """
        pass

    @abstractmethod
    async def cleanup_expired_orders(self) -> int:
        """
        Nettoie les ordres expirés en les marquant comme expirés.

        Returns:
            Nombre d'ordres marqués comme expirés
        """
        pass


class OrderQuery:
    """
    Classe utilitaire pour construire des requêtes complexes sur les ordres.
    """

    def __init__(self):
        self.filters = {}
        self.sort_by = None
        self.sort_desc = False
        self.limit = None
        self.offset = 0

    def by_status(self, status: OrderStatus) -> 'OrderQuery':
        """Filtre par statut"""
        self.filters['status'] = status
        return self

    def by_symbol(self, symbol: str) -> 'OrderQuery':
        """Filtre par symbole"""
        self.filters['symbol'] = symbol
        return self

    def by_side(self, side: OrderSide) -> 'OrderQuery':
        """Filtre par côté"""
        self.filters['side'] = side
        return self

    def by_order_type(self, order_type: OrderType) -> 'OrderQuery':
        """Filtre par type d'ordre"""
        self.filters['order_type'] = order_type
        return self

    def by_portfolio(self, portfolio_id: str) -> 'OrderQuery':
        """Filtre par portfolio"""
        self.filters['portfolio_id'] = portfolio_id
        return self

    def by_strategy(self, strategy_id: str) -> 'OrderQuery':
        """Filtre par stratégie"""
        self.filters['strategy_id'] = strategy_id
        return self

    def by_priority(self, priority: OrderPriority) -> 'OrderQuery':
        """Filtre par priorité"""
        self.filters['priority'] = priority
        return self

    def by_time_in_force(self, time_in_force: TimeInForce) -> 'OrderQuery':
        """Filtre par durée de validité"""
        self.filters['time_in_force'] = time_in_force
        return self

    def active_only(self) -> 'OrderQuery':
        """Filtre uniquement les ordres actifs"""
        self.filters['active_only'] = True
        return self

    def terminal_only(self) -> 'OrderQuery':
        """Filtre uniquement les ordres terminaux"""
        self.filters['terminal_only'] = True
        return self

    def filled_only(self) -> 'OrderQuery':
        """Filtre uniquement les ordres exécutés"""
        self.filters['filled_only'] = True
        return self

    def parent_orders_only(self) -> 'OrderQuery':
        """Filtre uniquement les ordres parents"""
        self.filters['parent_orders_only'] = True
        return self

    def child_orders_only(self) -> 'OrderQuery':
        """Filtre uniquement les ordres enfants"""
        self.filters['child_orders_only'] = True
        return self

    def by_quantity_range(self, min_qty: Optional[Decimal], max_qty: Optional[Decimal]) -> 'OrderQuery':
        """Filtre par fourchette de quantité"""
        if min_qty is not None:
            self.filters['min_quantity'] = min_qty
        if max_qty is not None:
            self.filters['max_quantity'] = max_qty
        return self

    def by_value_range(self, min_value: Optional[Decimal], max_value: Optional[Decimal]) -> 'OrderQuery':
        """Filtre par fourchette de valeur notionnelle"""
        if min_value is not None:
            self.filters['min_value'] = min_value
        if max_value is not None:
            self.filters['max_value'] = max_value
        return self

    def created_after(self, date: datetime) -> 'OrderQuery':
        """Filtre les ordres créés après une date"""
        self.filters['created_after'] = date
        return self

    def created_before(self, date: datetime) -> 'OrderQuery':
        """Filtre les ordres créés avant une date"""
        self.filters['created_before'] = date
        return self

    def submitted_after(self, date: datetime) -> 'OrderQuery':
        """Filtre les ordres soumis après une date"""
        self.filters['submitted_after'] = date
        return self

    def with_tag(self, tag_key: str, tag_value: Optional[str] = None) -> 'OrderQuery':
        """Filtre les ordres avec un tag"""
        self.filters['tag_key'] = tag_key
        if tag_value:
            self.filters['tag_value'] = tag_value
        return self

    def by_destination(self, destination: str) -> 'OrderQuery':
        """Filtre par destination d'exécution"""
        self.filters['destination'] = destination
        return self

    def sort_by_creation_time(self, descending: bool = True) -> 'OrderQuery':
        """Trie par temps de création"""
        self.sort_by = 'created_time'
        self.sort_desc = descending
        return self

    def sort_by_quantity(self, descending: bool = True) -> 'OrderQuery':
        """Trie par quantité"""
        self.sort_by = 'quantity'
        self.sort_desc = descending
        return self

    def sort_by_value(self, descending: bool = True) -> 'OrderQuery':
        """Trie par valeur notionnelle"""
        self.sort_by = 'notional_value'
        self.sort_desc = descending
        return self

    def sort_by_priority(self, descending: bool = True) -> 'OrderQuery':
        """Trie par priorité"""
        self.sort_by = 'priority'
        self.sort_desc = descending
        return self

    def paginate(self, limit: int, offset: int = 0) -> 'OrderQuery':
        """Ajoute la pagination"""
        self.limit = limit
        self.offset = offset
        return self


class OrderAggregateQuery:
    """
    Classe pour construire des requêtes d'agrégation sur les ordres.
    """

    def __init__(self):
        self.group_by = []
        self.aggregations = []
        self.filters = {}

    def group_by_status(self) -> 'OrderAggregateQuery':
        """Grouper par statut"""
        self.group_by.append('status')
        return self

    def group_by_symbol(self) -> 'OrderAggregateQuery':
        """Grouper par symbole"""
        self.group_by.append('symbol')
        return self

    def group_by_side(self) -> 'OrderAggregateQuery':
        """Grouper par côté"""
        self.group_by.append('side')
        return self

    def group_by_order_type(self) -> 'OrderAggregateQuery':
        """Grouper par type d'ordre"""
        self.group_by.append('order_type')
        return self

    def group_by_date(self, date_field: str = 'created_time', interval: str = 'day') -> 'OrderAggregateQuery':
        """Grouper par date"""
        self.group_by.append(f'{date_field}_{interval}')
        return self

    def count(self) -> 'OrderAggregateQuery':
        """Compter les ordres"""
        self.aggregations.append('count')
        return self

    def sum_quantity(self) -> 'OrderAggregateQuery':
        """Sommer les quantités"""
        self.aggregations.append('sum_quantity')
        return self

    def sum_value(self) -> 'OrderAggregateQuery':
        """Sommer les valeurs notionnelles"""
        self.aggregations.append('sum_value')
        return self

    def avg_quantity(self) -> 'OrderAggregateQuery':
        """Moyenne des quantités"""
        self.aggregations.append('avg_quantity')
        return self

    def avg_fill_time(self) -> 'OrderAggregateQuery':
        """Temps moyen d'exécution"""
        self.aggregations.append('avg_fill_time')
        return self

    def fill_rate(self) -> 'OrderAggregateQuery':
        """Taux d'exécution"""
        self.aggregations.append('fill_rate')
        return self

    def filter_by_symbol(self, symbol: str) -> 'OrderAggregateQuery':
        """Filtrer par symbole"""
        self.filters['symbol'] = symbol
        return self

    def filter_by_portfolio(self, portfolio_id: str) -> 'OrderAggregateQuery':
        """Filtrer par portfolio"""
        self.filters['portfolio_id'] = portfolio_id
        return self

    def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> 'OrderAggregateQuery':
        """Filtrer par période"""
        self.filters['start_date'] = start_date
        self.filters['end_date'] = end_date
        return self


class RepositoryError(Exception):
    """Exception pour les erreurs de repository"""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class OrderNotFoundError(RepositoryError):
    """Exception quand un ordre n'est pas trouvé"""

    def __init__(self, order_identifier: str):
        super().__init__(f"Order not found: {order_identifier}")
        self.order_identifier = order_identifier


class DuplicateOrderError(RepositoryError):
    """Exception quand un ordre existe déjà"""

    def __init__(self, order_identifier: str):
        super().__init__(f"Order already exists: {order_identifier}")
        self.order_identifier = order_identifier


class OrderStateError(RepositoryError):
    """Exception pour les erreurs d'état d'ordre"""

    def __init__(self, order_id: str, current_state: str, attempted_action: str):
        super().__init__(f"Cannot {attempted_action} order {order_id} in state {current_state}")
        self.order_id = order_id
        self.current_state = current_state
        self.attempted_action = attempted_action