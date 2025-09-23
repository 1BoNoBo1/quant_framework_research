"""
Domain Layer: Backtest Repository Interface
==========================================

Interface du repository pour la persistance des backtests.
Définit les opérations CRUD et de recherche pour les backtests.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal

from ..entities.backtest import (
    BacktestConfiguration, BacktestResult, BacktestStatus, BacktestType, BacktestMetrics
)


class BacktestRepository(ABC):
    """Interface du repository pour les backtests"""

    # Configuration Management

    @abstractmethod
    async def save_configuration(self, config: BacktestConfiguration) -> None:
        """Sauvegarde une configuration de backtest"""
        pass

    @abstractmethod
    async def get_configuration(self, config_id: str) -> Optional[BacktestConfiguration]:
        """Récupère une configuration par son ID"""
        pass

    @abstractmethod
    async def find_configurations_by_name(self, name: str) -> List[BacktestConfiguration]:
        """Trouve des configurations par nom"""
        pass

    @abstractmethod
    async def get_all_configurations(self) -> List[BacktestConfiguration]:
        """Récupère toutes les configurations"""
        pass

    @abstractmethod
    async def delete_configuration(self, config_id: str) -> bool:
        """Supprime une configuration"""
        pass

    # Result Management

    @abstractmethod
    async def save_result(self, result: BacktestResult) -> None:
        """Sauvegarde un résultat de backtest"""
        pass

    @abstractmethod
    async def get_result(self, result_id: str) -> Optional[BacktestResult]:
        """Récupère un résultat par son ID"""
        pass

    @abstractmethod
    async def find_results_by_configuration(self, config_id: str) -> List[BacktestResult]:
        """Trouve tous les résultats pour une configuration"""
        pass

    @abstractmethod
    async def find_results_by_status(self, status: BacktestStatus) -> List[BacktestResult]:
        """Trouve tous les résultats par statut"""
        pass

    @abstractmethod
    async def find_results_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[BacktestResult]:
        """Trouve les résultats créés dans une plage de dates"""
        pass

    @abstractmethod
    async def find_results_by_strategy(self, strategy_id: str) -> List[BacktestResult]:
        """Trouve tous les résultats qui utilisent une stratégie donnée"""
        pass

    @abstractmethod
    async def get_latest_results(self, limit: int = 10) -> List[BacktestResult]:
        """Récupère les derniers résultats"""
        pass

    @abstractmethod
    async def delete_result(self, result_id: str) -> bool:
        """Supprime un résultat"""
        pass

    # Advanced Queries

    @abstractmethod
    async def find_best_performing_results(
        self,
        metric: str = "sharpe_ratio",
        limit: int = 10,
        min_trades: int = 10
    ) -> List[BacktestResult]:
        """Trouve les backtests les plus performants selon une métrique"""
        pass

    @abstractmethod
    async def find_results_by_metrics_criteria(
        self,
        min_sharpe_ratio: Optional[Decimal] = None,
        max_drawdown: Optional[Decimal] = None,
        min_win_rate: Optional[Decimal] = None,
        min_return: Optional[Decimal] = None
    ) -> List[BacktestResult]:
        """Trouve les résultats selon des critères de métriques"""
        pass

    @abstractmethod
    async def get_performance_comparison(
        self,
        result_ids: List[str]
    ) -> Dict[str, BacktestMetrics]:
        """Compare les métriques de plusieurs backtests"""
        pass

    @abstractmethod
    async def search_results(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[BacktestResult]:
        """Recherche textuelle dans les backtests"""
        pass

    # Statistics and Analytics

    @abstractmethod
    async def get_backtest_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques globales sur les backtests"""
        pass

    @abstractmethod
    async def get_strategy_performance_summary(
        self,
        strategy_id: str
    ) -> Dict[str, Any]:
        """Retourne un résumé de performance pour une stratégie"""
        pass

    @abstractmethod
    async def get_monthly_performance_summary(
        self,
        year: int
    ) -> Dict[str, Any]:
        """Retourne un résumé mensuel des performances"""
        pass

    @abstractmethod
    async def find_similar_configurations(
        self,
        config: BacktestConfiguration,
        similarity_threshold: float = 0.8
    ) -> List[BacktestConfiguration]:
        """Trouve des configurations similaires"""
        pass

    # Data Management

    @abstractmethod
    async def cleanup_old_results(
        self,
        days_old: int = 90,
        keep_best: int = 10
    ) -> int:
        """Nettoie les anciens résultats en gardant les meilleurs"""
        pass

    @abstractmethod
    async def get_storage_usage(self) -> Dict[str, Any]:
        """Retourne l'utilisation de l'espace de stockage"""
        pass

    @abstractmethod
    async def export_results(
        self,
        result_ids: List[str],
        format: str = "json"
    ) -> bytes:
        """Exporte des résultats dans un format donné"""
        pass

    @abstractmethod
    async def import_results(
        self,
        data: bytes,
        format: str = "json"
    ) -> List[str]:
        """Importe des résultats depuis des données"""
        pass

    # Batch Operations

    @abstractmethod
    async def batch_save_results(self, results: List[BacktestResult]) -> None:
        """Sauvegarde plusieurs résultats en lot"""
        pass

    @abstractmethod
    async def batch_update_status(
        self,
        result_ids: List[str],
        status: BacktestStatus
    ) -> int:
        """Met à jour le statut de plusieurs résultats"""
        pass

    @abstractmethod
    async def get_results_count_by_status(self) -> Dict[BacktestStatus, int]:
        """Compte les résultats par statut"""
        pass

    @abstractmethod
    async def get_results_count_by_type(self) -> Dict[BacktestType, int]:
        """Compte les résultats par type de backtest"""
        pass

    # Archive and Restore

    @abstractmethod
    async def archive_result(self, result_id: str) -> bool:
        """Archive un résultat (le garde mais le marque comme archivé)"""
        pass

    @abstractmethod
    async def restore_result(self, result_id: str) -> bool:
        """Restaure un résultat archivé"""
        pass

    @abstractmethod
    async def get_archived_results(self) -> List[BacktestResult]:
        """Récupère tous les résultats archivés"""
        pass

    # Validation and Health

    @abstractmethod
    async def validate_result_integrity(self, result_id: str) -> List[str]:
        """Valide l'intégrité d'un résultat et retourne les erreurs"""
        pass

    @abstractmethod
    async def get_repository_health(self) -> Dict[str, Any]:
        """Retourne l'état de santé du repository"""
        pass