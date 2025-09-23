"""
Repository Interface: Risk Assessment Repository
==============================================

Interface de repository pour la persistance des évaluations de risque.
Définit le contrat pour l'accès aux données selon l'architecture hexagonale.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

from ..entities.risk_assessment import RiskAssessment, RiskLevel, RiskType
from ..value_objects.position import Position


class RiskAssessmentRepository(ABC):
    """
    Interface de repository pour les évaluations de risque.

    Définit les opérations de persistance et de requête pour les
    évaluations de risque selon les principes DDD.
    """

    @abstractmethod
    async def save(self, assessment: RiskAssessment) -> None:
        """
        Sauvegarde une évaluation de risque.

        Args:
            assessment: Évaluation à sauvegarder

        Raises:
            RepositoryError: Si la sauvegarde échoue
        """
        pass

    @abstractmethod
    async def find_by_id(self, assessment_id: str) -> Optional[RiskAssessment]:
        """
        Trouve une évaluation par son ID.

        Args:
            assessment_id: ID de l'évaluation

        Returns:
            Évaluation trouvée ou None
        """
        pass

    @abstractmethod
    async def find_by_target_id(self, target_id: str) -> List[RiskAssessment]:
        """
        Trouve toutes les évaluations pour une cible (portfolio/stratégie/position).

        Args:
            target_id: ID de la cible évaluée

        Returns:
            Liste des évaluations pour cette cible
        """
        pass

    @abstractmethod
    async def find_latest_by_target_id(self, target_id: str) -> Optional[RiskAssessment]:
        """
        Trouve la dernière évaluation pour une cible.

        Args:
            target_id: ID de la cible évaluée

        Returns:
            Dernière évaluation ou None
        """
        pass

    @abstractmethod
    async def find_by_assessment_type(self, assessment_type: str) -> List[RiskAssessment]:
        """
        Trouve toutes les évaluations d'un type donné.

        Args:
            assessment_type: Type d'évaluation (portfolio, strategy, position)

        Returns:
            Liste des évaluations du type spécifié
        """
        pass

    @abstractmethod
    async def find_by_risk_level(self, risk_level: RiskLevel) -> List[RiskAssessment]:
        """
        Trouve toutes les évaluations avec un niveau de risque donné.

        Args:
            risk_level: Niveau de risque recherché

        Returns:
            Liste des évaluations avec ce niveau de risque
        """
        pass

    @abstractmethod
    async def find_critical_assessments(self) -> List[RiskAssessment]:
        """
        Trouve toutes les évaluations critiques.

        Returns:
            Liste des évaluations avec risques critiques
        """
        pass

    @abstractmethod
    async def find_expired_assessments(self, current_time: datetime) -> List[RiskAssessment]:
        """
        Trouve toutes les évaluations expirées.

        Args:
            current_time: Timestamp actuel pour comparaison

        Returns:
            Liste des évaluations expirées
        """
        pass

    @abstractmethod
    async def find_breached_metrics(self) -> List[RiskAssessment]:
        """
        Trouve toutes les évaluations avec des métriques en dépassement.

        Returns:
            Liste des évaluations avec métriques breachées
        """
        pass

    @abstractmethod
    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[RiskAssessment]:
        """
        Trouve toutes les évaluations dans une période donnée.

        Args:
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Liste des évaluations dans la période
        """
        pass

    @abstractmethod
    async def update(self, assessment: RiskAssessment) -> None:
        """
        Met à jour une évaluation existante.

        Args:
            assessment: Évaluation à mettre à jour

        Raises:
            RepositoryError: Si l'évaluation n'existe pas ou si la mise à jour échoue
        """
        pass

    @abstractmethod
    async def delete(self, assessment_id: str) -> bool:
        """
        Supprime une évaluation.

        Args:
            assessment_id: ID de l'évaluation à supprimer

        Returns:
            True si supprimée, False si introuvable

        Raises:
            RepositoryError: Si la suppression échoue
        """
        pass

    @abstractmethod
    async def count_by_risk_level(self) -> Dict[RiskLevel, int]:
        """
        Compte les évaluations par niveau de risque.

        Returns:
            Dictionnaire niveau_risque -> nombre_évaluations
        """
        pass

    @abstractmethod
    async def get_risk_statistics(
        self,
        target_id: Optional[str] = None,
        assessment_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les évaluations de risque.

        Args:
            target_id: Filtrer par cible (optionnel)
            assessment_type: Filtrer par type (optionnel)

        Returns:
            Dictionnaire de statistiques
        """
        pass

    @abstractmethod
    async def archive_old_assessments(
        self,
        cutoff_date: datetime,
        keep_latest: bool = True
    ) -> int:
        """
        Archive les anciennes évaluations.

        Args:
            cutoff_date: Date limite pour l'archivage
            keep_latest: Conserver la dernière évaluation par cible

        Returns:
            Nombre d'évaluations archivées
        """
        pass


class RiskMetricQuery:
    """
    Classe utilitaire pour construire des requêtes sur les métriques de risque.
    """

    def __init__(self):
        self.filters = {}
        self.sort_by = None
        self.sort_desc = False
        self.limit = None

    def by_risk_type(self, risk_type: RiskType) -> 'RiskMetricQuery':
        """Filtre par type de risque"""
        self.filters['risk_type'] = risk_type
        return self

    def by_risk_level(self, risk_level: RiskLevel) -> 'RiskMetricQuery':
        """Filtre par niveau de risque"""
        self.filters['risk_level'] = risk_level
        return self

    def breached_only(self) -> 'RiskMetricQuery':
        """Filtre uniquement les métriques en dépassement"""
        self.filters['breached'] = True
        return self

    def above_threshold(self, percentage: float) -> 'RiskMetricQuery':
        """Filtre les métriques au-dessus d'un pourcentage du seuil"""
        self.filters['threshold_percentage'] = percentage
        return self

    def with_confidence_above(self, confidence: float) -> 'RiskMetricQuery':
        """Filtre par niveau de confiance minimum"""
        self.filters['min_confidence'] = confidence
        return self

    def sort_by_risk_score(self, descending: bool = True) -> 'RiskMetricQuery':
        """Trie par score de risque"""
        self.sort_by = 'risk_score'
        self.sort_desc = descending
        return self

    def sort_by_breach_percentage(self, descending: bool = True) -> 'RiskMetricQuery':
        """Trie par pourcentage de dépassement"""
        self.sort_by = 'breach_percentage'
        self.sort_desc = descending
        return self

    def limit_results(self, limit: int) -> 'RiskMetricQuery':
        """Limite le nombre de résultats"""
        self.limit = limit
        return self


class RiskAssessmentQuery:
    """
    Classe utilitaire pour construire des requêtes complexes sur les évaluations.
    """

    def __init__(self):
        self.filters = {}
        self.metric_queries = []
        self.sort_by = None
        self.sort_desc = False
        self.limit = None
        self.offset = 0

    def by_target_id(self, target_id: str) -> 'RiskAssessmentQuery':
        """Filtre par ID de cible"""
        self.filters['target_id'] = target_id
        return self

    def by_assessment_type(self, assessment_type: str) -> 'RiskAssessmentQuery':
        """Filtre par type d'évaluation"""
        self.filters['assessment_type'] = assessment_type
        return self

    def by_risk_level_range(
        self,
        min_level: RiskLevel,
        max_level: RiskLevel
    ) -> 'RiskAssessmentQuery':
        """Filtre par plage de niveaux de risque"""
        self.filters['min_risk_level'] = min_level
        self.filters['max_risk_level'] = max_level
        return self

    def by_risk_score_range(
        self,
        min_score: Decimal,
        max_score: Decimal
    ) -> 'RiskAssessmentQuery':
        """Filtre par plage de scores de risque"""
        self.filters['min_risk_score'] = min_score
        self.filters['max_risk_score'] = max_score
        return self

    def with_alerts(self) -> 'RiskAssessmentQuery':
        """Filtre les évaluations avec alertes"""
        self.filters['has_alerts'] = True
        return self

    def with_breached_metrics(self) -> 'RiskAssessmentQuery':
        """Filtre les évaluations avec métriques en dépassement"""
        self.filters['has_breached_metrics'] = True
        return self

    def valid_only(self) -> 'RiskAssessmentQuery':
        """Filtre uniquement les évaluations valides"""
        self.filters['valid_only'] = True
        return self

    def recent_only(self, hours: int = 24) -> 'RiskAssessmentQuery':
        """Filtre les évaluations récentes"""
        self.filters['recent_hours'] = hours
        return self

    def with_metric_query(self, metric_query: RiskMetricQuery) -> 'RiskAssessmentQuery':
        """Ajoute une requête sur les métriques"""
        self.metric_queries.append(metric_query)
        return self

    def sort_by_timestamp(self, descending: bool = True) -> 'RiskAssessmentQuery':
        """Trie par timestamp"""
        self.sort_by = 'assessment_time'
        self.sort_desc = descending
        return self

    def sort_by_risk_score(self, descending: bool = True) -> 'RiskAssessmentQuery':
        """Trie par score de risque"""
        self.sort_by = 'risk_score'
        self.sort_desc = descending
        return self

    def paginate(self, limit: int, offset: int = 0) -> 'RiskAssessmentQuery':
        """Ajoute la pagination"""
        self.limit = limit
        self.offset = offset
        return self


class RepositoryError(Exception):
    """Exception pour les erreurs de repository"""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class RiskAssessmentNotFoundError(RepositoryError):
    """Exception quand une évaluation n'est pas trouvée"""

    def __init__(self, assessment_id: str):
        super().__init__(f"Risk assessment not found: {assessment_id}")
        self.assessment_id = assessment_id


class RiskAssessmentAlreadyExistsError(RepositoryError):
    """Exception quand une évaluation existe déjà"""

    def __init__(self, assessment_id: str):
        super().__init__(f"Risk assessment already exists: {assessment_id}")
        self.assessment_id = assessment_id