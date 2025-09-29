"""
Application Queries: Risk Management
===================================

Requêtes pour la consultation des données de gestion des risques.
Implémente le pattern CQRS pour la séparation des responsabilités.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, field

from ..base.query import Query, QueryHandler, QueryResult
from ...domain.entities.risk_assessment import RiskAssessment, RiskLevel, RiskType
from ...domain.repositories.risk_assessment_repository import (
    RiskAssessmentRepository,
    RiskAssessmentQuery,
    RiskMetricQuery
)


@dataclass
class GetRiskAssessmentQuery(Query):
    """Requête pour récupérer une évaluation de risque par ID"""
    assessment_id: str


@dataclass
class GetLatestRiskAssessmentQuery(Query):
    """Requête pour récupérer la dernière évaluation d'une cible"""
    target_id: str


@dataclass
class GetRiskAssessmentsByTargetQuery(Query):
    """Requête pour récupérer toutes les évaluations d'une cible"""
    target_id: str
    limit: Optional[int] = None


@dataclass
class GetCriticalRiskAssessmentsQuery(Query):
    """Requête pour récupérer toutes les évaluations critiques"""
    include_expired: bool = False


@dataclass
class GetRiskAssessmentsByLevelQuery(Query):
    """Requête pour récupérer les évaluations par niveau de risque"""
    risk_level: RiskLevel
    assessment_type: Optional[str] = None


@dataclass
class GetBreachedMetricsQuery(Query):
    """Requête pour récupérer les évaluations avec métriques en dépassement"""
    risk_type: Optional[RiskType] = None
    severity_threshold: Optional[float] = None


@dataclass
class GetRiskStatisticsQuery(Query):
    """Requête pour récupérer les statistiques de risque"""
    target_id: Optional[str] = None
    assessment_type: Optional[str] = None
    date_range: Optional[tuple[datetime, datetime]] = None


@dataclass
class GetRiskTrendsQuery(Query):
    """Requête pour récupérer les tendances de risque"""
    target_id: str
    days_back: int = 30


@dataclass
class SearchRiskAssessmentsQuery(Query):
    """Requête de recherche avancée d'évaluations"""
    query_builder: RiskAssessmentQuery


@dataclass
class GetRiskDashboardQuery(Query):
    """Requête pour récupérer les données du tableau de bord risque"""
    assessment_types: Optional[List[str]] = None
    include_trends: bool = True


class GetRiskAssessmentHandler(QueryHandler[GetRiskAssessmentQuery]):
    """Handler pour récupérer une évaluation de risque"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: GetRiskAssessmentQuery) -> QueryResult:
        """
        Traite la requête de récupération d'évaluation.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec l'évaluation ou erreur si introuvable
        """
        try:
            assessment = await self.repository.find_by_id(query.assessment_id)

            if not assessment:
                return QueryResult(
                    success=False,
                    error_message=f"Évaluation introuvable: {query.assessment_id}"
                )

            return QueryResult(
                success=True,
                data=assessment.to_dict(),
                message="Évaluation récupérée avec succès"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors de la récupération: {str(e)}"
            )


class GetLatestRiskAssessmentHandler(QueryHandler[GetLatestRiskAssessmentQuery]):
    """Handler pour récupérer la dernière évaluation d'une cible"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: GetLatestRiskAssessmentQuery) -> QueryResult:
        """
        Traite la requête de récupération de la dernière évaluation.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec la dernière évaluation
        """
        try:
            assessment = await self.repository.find_latest_by_target_id(query.target_id)

            if not assessment:
                return QueryResult(
                    success=False,
                    error_message=f"Aucune évaluation trouvée pour: {query.target_id}"
                )

            return QueryResult(
                success=True,
                data=assessment.to_dict(),
                message="Dernière évaluation récupérée avec succès"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors de la récupération: {str(e)}"
            )


class GetRiskAssessmentsByTargetHandler(QueryHandler[GetRiskAssessmentsByTargetQuery]):
    """Handler pour récupérer toutes les évaluations d'une cible"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: GetRiskAssessmentsByTargetQuery) -> QueryResult:
        """
        Traite la requête de récupération des évaluations par cible.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec la liste des évaluations
        """
        try:
            assessments = await self.repository.find_by_target_id(query.target_id)

            # Limiter les résultats si demandé
            if query.limit and len(assessments) > query.limit:
                assessments = assessments[:query.limit]

            assessments_data = [assessment.to_dict() for assessment in assessments]

            return QueryResult(
                success=True,
                data={
                    "assessments": assessments_data,
                    "count": len(assessments_data),
                    "target_id": query.target_id
                },
                message=f"{len(assessments_data)} évaluations trouvées pour {query.target_id}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors de la récupération: {str(e)}"
            )


class GetCriticalRiskAssessmentsHandler(QueryHandler[GetCriticalRiskAssessmentsQuery]):
    """Handler pour récupérer les évaluations critiques"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: GetCriticalRiskAssessmentsQuery) -> QueryResult:
        """
        Traite la requête de récupération des évaluations critiques.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les évaluations critiques
        """
        try:
            critical_assessments = await self.repository.find_critical_assessments()

            # Filtrer les évaluations expirées si demandé
            if not query.include_expired:
                critical_assessments = [
                    assessment for assessment in critical_assessments
                    if assessment.is_valid()
                ]

            # Trier par niveau de risque (critiques en premier)
            critical_assessments.sort(
                key=lambda x: (
                    x.overall_risk_level != RiskLevel.CRITICAL,
                    -float(x.risk_score),
                    x.assessment_time
                ),
                reverse=True
            )

            assessments_data = [assessment.to_dict() for assessment in critical_assessments]

            return QueryResult(
                success=True,
                data={
                    "critical_assessments": assessments_data,
                    "count": len(assessments_data),
                    "include_expired": query.include_expired
                },
                message=f"{len(assessments_data)} évaluations critiques trouvées"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors de la récupération: {str(e)}"
            )


class GetRiskAssessmentsByLevelHandler(QueryHandler[GetRiskAssessmentsByLevelQuery]):
    """Handler pour récupérer les évaluations par niveau de risque"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: GetRiskAssessmentsByLevelQuery) -> QueryResult:
        """
        Traite la requête de récupération par niveau de risque.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les évaluations du niveau spécifié
        """
        try:
            assessments = await self.repository.find_by_risk_level(query.risk_level)

            # Filtrer par type d'évaluation si spécifié
            if query.assessment_type:
                assessments = [
                    assessment for assessment in assessments
                    if assessment.assessment_type == query.assessment_type
                ]

            assessments_data = [assessment.to_dict() for assessment in assessments]

            return QueryResult(
                success=True,
                data={
                    "assessments": assessments_data,
                    "count": len(assessments_data),
                    "risk_level": query.risk_level.value,
                    "assessment_type": query.assessment_type
                },
                message=f"{len(assessments_data)} évaluations trouvées pour le niveau {query.risk_level.value}"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors de la récupération: {str(e)}"
            )


class GetBreachedMetricsHandler(QueryHandler[GetBreachedMetricsQuery]):
    """Handler pour récupérer les évaluations avec métriques en dépassement"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: GetBreachedMetricsQuery) -> QueryResult:
        """
        Traite la requête de récupération des métriques en dépassement.

        Args:
            query: Requête de récupération

        Returns:
            Résultat avec les évaluations ayant des métriques breachées
        """
        try:
            assessments = await self.repository.find_breached_metrics()

            # Analyser les métriques breachées
            breached_data = []
            for assessment in assessments:
                breached_metrics = assessment.get_breached_metrics()

                # Filtrer par type de risque si spécifié
                if query.risk_type:
                    breached_metrics = [
                        metric for metric in breached_metrics
                        if metric.risk_type == query.risk_type
                    ]

                # Filtrer par seuil de sévérité si spécifié
                if query.severity_threshold:
                    breached_metrics = [
                        metric for metric in breached_metrics
                        if float(metric.breach_percentage) >= query.severity_threshold
                    ]

                if breached_metrics:
                    breached_data.append({
                        "assessment": assessment.to_dict(),
                        "breached_metrics": [metric.to_dict() for metric in breached_metrics],
                        "breach_count": len(breached_metrics)
                    })

            return QueryResult(
                success=True,
                data={
                    "breached_assessments": breached_data,
                    "total_assessments": len(breached_data),
                    "total_breached_metrics": sum(item["breach_count"] for item in breached_data),
                    "risk_type_filter": query.risk_type.value if query.risk_type else None,
                    "severity_threshold": query.severity_threshold
                },
                message=f"{len(breached_data)} évaluations avec métriques en dépassement"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors de la récupération: {str(e)}"
            )


class GetRiskStatisticsHandler(QueryHandler[GetRiskStatisticsQuery]):
    """Handler pour récupérer les statistiques de risque"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: GetRiskStatisticsQuery) -> QueryResult:
        """
        Traite la requête de récupération des statistiques.

        Args:
            query: Requête de statistiques

        Returns:
            Résultat avec les statistiques calculées
        """
        try:
            # Récupérer les statistiques de base
            stats = await self.repository.get_risk_statistics(
                query.target_id,
                query.assessment_type
            )

            # Ajouter des statistiques par niveau de risque
            risk_level_counts = await self.repository.count_by_risk_level()

            # Calculer des métriques supplémentaires
            total_assessments = sum(risk_level_counts.values())
            critical_percentage = (
                risk_level_counts.get(RiskLevel.CRITICAL, 0) / total_assessments * 100
                if total_assessments > 0 else 0
            )

            high_risk_count = (
                risk_level_counts.get(RiskLevel.HIGH, 0) +
                risk_level_counts.get(RiskLevel.VERY_HIGH, 0) +
                risk_level_counts.get(RiskLevel.CRITICAL, 0)
            )

            high_risk_percentage = (
                high_risk_count / total_assessments * 100
                if total_assessments > 0 else 0
            )

            enhanced_stats = {
                **stats,
                "risk_level_distribution": {
                    level.value: count for level, count in risk_level_counts.items()
                },
                "total_assessments": total_assessments,
                "critical_percentage": round(critical_percentage, 2),
                "high_risk_percentage": round(high_risk_percentage, 2),
                "filters": {
                    "target_id": query.target_id,
                    "assessment_type": query.assessment_type,
                    "date_range": [
                        query.date_range[0].isoformat(),
                        query.date_range[1].isoformat()
                    ] if query.date_range else None
                }
            }

            return QueryResult(
                success=True,
                data=enhanced_stats,
                message="Statistiques de risque calculées avec succès"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors du calcul des statistiques: {str(e)}"
            )


class GetRiskTrendsHandler(QueryHandler[GetRiskTrendsQuery]):
    """Handler pour récupérer les tendances de risque"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: GetRiskTrendsQuery) -> QueryResult:
        """
        Traite la requête de récupération des tendances.

        Args:
            query: Requête de tendances

        Returns:
            Résultat avec les données de tendance
        """
        try:
            # Calculer la date de début
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=query.days_back)

            # Récupérer les évaluations dans la période
            assessments = await self.repository.find_by_date_range(start_date, end_date)

            # Filtrer par target_id
            target_assessments = [
                assessment for assessment in assessments
                if assessment.target_id == query.target_id
            ]

            # Trier par date
            target_assessments.sort(key=lambda x: x.assessment_time)

            # Calculer les tendances
            risk_scores = [float(assessment.risk_score) for assessment in target_assessments]
            dates = [assessment.assessment_time.isoformat() for assessment in target_assessments]

            # Calculer la tendance (variation moyenne)
            trend_direction = "stable"
            trend_percentage = 0.0

            if len(risk_scores) >= 2:
                first_score = risk_scores[0]
                last_score = risk_scores[-1]
                trend_percentage = ((last_score - first_score) / first_score * 100) if first_score > 0 else 0

                if trend_percentage > 5:
                    trend_direction = "increasing"
                elif trend_percentage < -5:
                    trend_direction = "decreasing"

            # Analyser les niveaux de risque
            risk_levels_over_time = [
                {
                    "date": assessment.assessment_time.isoformat(),
                    "risk_level": assessment.overall_risk_level.value,
                    "risk_score": float(assessment.risk_score)
                }
                for assessment in target_assessments
            ]

            return QueryResult(
                success=True,
                data={
                    "target_id": query.target_id,
                    "period_days": query.days_back,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "assessments_count": len(target_assessments),
                    "risk_scores": risk_scores,
                    "dates": dates,
                    "trend": {
                        "direction": trend_direction,
                        "percentage": round(trend_percentage, 2),
                        "current_score": risk_scores[-1] if risk_scores else 0,
                        "average_score": round(sum(risk_scores) / len(risk_scores), 2) if risk_scores else 0
                    },
                    "risk_levels_timeline": risk_levels_over_time
                },
                message=f"Tendances calculées pour {query.target_id} sur {query.days_back} jours"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors du calcul des tendances: {str(e)}"
            )


class SearchRiskAssessmentsHandler(QueryHandler[SearchRiskAssessmentsQuery]):
    """Handler pour la recherche avancée d'évaluations"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: SearchRiskAssessmentsQuery) -> QueryResult:
        """
        Traite la requête de recherche avancée.

        Args:
            query: Requête de recherche

        Returns:
            Résultat avec les évaluations correspondantes
        """
        try:
            # Pour l'instant, on simule une recherche basique
            # Dans une vraie implémentation, on utiliserait le query_builder
            # pour construire une requête complexe

            # Récupérer tous les filtres du query builder
            filters = query.query_builder.filters

            assessments = []

            # Appliquer les filtres un par un
            if "target_id" in filters:
                assessments = await self.repository.find_by_target_id(filters["target_id"])
            elif "assessment_type" in filters:
                assessments = await self.repository.find_by_assessment_type(filters["assessment_type"])
            elif "has_alerts" in filters and filters["has_alerts"]:
                all_assessments = await self.repository.find_critical_assessments()
                assessments = [a for a in all_assessments if len(a.alerts) > 0]
            else:
                # Recherche générale - limiter les résultats
                assessments = await self.repository.find_by_risk_level(RiskLevel.MEDIUM)

            # Appliquer la pagination
            if query.query_builder.limit:
                offset = query.query_builder.offset
                limit = query.query_builder.limit
                assessments = assessments[offset:offset + limit]

            assessments_data = [assessment.to_dict() for assessment in assessments]

            return QueryResult(
                success=True,
                data={
                    "assessments": assessments_data,
                    "count": len(assessments_data),
                    "filters_applied": filters,
                    "pagination": {
                        "limit": query.query_builder.limit,
                        "offset": query.query_builder.offset
                    }
                },
                message=f"Recherche terminée: {len(assessments_data)} résultats"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors de la recherche: {str(e)}"
            )


class GetRiskDashboardHandler(QueryHandler[GetRiskDashboardQuery]):
    """Handler pour récupérer les données du tableau de bord risque"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, query: GetRiskDashboardQuery) -> QueryResult:
        """
        Traite la requête de tableau de bord.

        Args:
            query: Requête de tableau de bord

        Returns:
            Résultat avec les données du dashboard
        """
        try:
            # Récupérer les évaluations critiques
            critical_assessments = await self.repository.find_critical_assessments()

            # Récupérer les évaluations avec métriques breachées
            breached_assessments = await self.repository.find_breached_metrics()

            # Statistiques par niveau de risque
            risk_level_counts = await self.repository.count_by_risk_level()

            # Récupérer les statistiques générales
            overall_stats = await self.repository.get_risk_statistics()

            dashboard_data = {
                "summary": {
                    "total_assessments": sum(risk_level_counts.values()),
                    "critical_count": len(critical_assessments),
                    "breached_metrics_count": len(breached_assessments),
                    "risk_level_distribution": {
                        level.value: count for level, count in risk_level_counts.items()
                    }
                },
                "alerts": {
                    "critical_assessments": [
                        {
                            "id": assessment.id,
                            "target_id": assessment.target_id,
                            "risk_score": float(assessment.risk_score),
                            "alerts": assessment.alerts[:3],  # Limiter à 3 alertes
                            "assessment_time": assessment.assessment_time.isoformat()
                        }
                        for assessment in critical_assessments[:10]  # Top 10
                    ],
                    "recent_breaches": [
                        {
                            "assessment_id": assessment.id,
                            "target_id": assessment.target_id,
                            "breached_metrics": len(assessment.get_breached_metrics()),
                            "assessment_time": assessment.assessment_time.isoformat()
                        }
                        for assessment in breached_assessments[:10]  # Top 10
                    ]
                },
                "statistics": overall_stats
            }

            # Ajouter les tendances si demandé
            if query.include_trends:
                # Calculer des tendances basiques pour le dashboard
                # (dans une vraie implémentation, on ferait des requêtes plus complexes)
                dashboard_data["trends"] = {
                    "risk_score_trend": "stable",  # Placeholder
                    "critical_assessments_trend": "increasing" if len(critical_assessments) > 5 else "stable",
                    "breach_frequency": "normal"
                }

            return QueryResult(
                success=True,
                data=dashboard_data,
                message="Données du tableau de bord récupérées avec succès"
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Erreur lors de la récupération du dashboard: {str(e)}"
            )