"""
Infrastructure: MemoryRiskAssessmentRepository
============================================

Implémentation en mémoire du repository d'évaluations de risque.
Utilisée pour les tests et le développement.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import logging
import copy

from ...domain.entities.risk_assessment import RiskAssessment, RiskLevel, RiskType
from ...domain.repositories.risk_assessment_repository import RiskAssessmentRepository
from ...domain.value_objects.position import Position

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Exception de base pour les erreurs de repository."""
    pass


class RiskAssessmentNotFoundError(RepositoryError):
    """Évaluation de risque non trouvée."""
    pass


class MemoryRiskAssessmentRepository(RiskAssessmentRepository):
    """
    Implémentation en mémoire du repository d'évaluations de risque.

    Stocke les évaluations de risque dans un dictionnaire en mémoire.
    Idéal pour les tests et le prototypage.
    """

    def __init__(self):
        self._assessments: Dict[str, RiskAssessment] = {}
        self._target_index: Dict[str, List[str]] = {}  # target_id -> [assessment_ids]
        self._type_index: Dict[str, List[str]] = {}  # assessment_type -> [assessment_ids]
        self._risk_level_index: Dict[RiskLevel, List[str]] = {}  # risk_level -> [assessment_ids]

    async def save(self, assessment: RiskAssessment) -> None:
        """Sauvegarde une évaluation de risque en mémoire."""
        try:
            # Créer une copie profonde pour éviter les modifications externes
            assessment_copy = self._deep_copy_assessment(assessment)

            # Si c'est une mise à jour, supprimer les anciens index
            if assessment.id in self._assessments:
                await self._remove_from_indexes(assessment.id)

            # Sauvegarder l'évaluation
            self._assessments[assessment.id] = assessment_copy

            # Mettre à jour les index
            self._add_to_indexes(assessment_copy)

            logger.debug(f"💾 Risk assessment saved: {assessment.id}")

        except Exception as e:
            logger.error(f"❌ Failed to save risk assessment {assessment.id}: {e}")
            raise RepositoryError(f"Failed to save assessment: {e}")

    async def find_by_id(self, assessment_id: str) -> Optional[RiskAssessment]:
        """Trouve une évaluation par son ID."""
        try:
            assessment = self._assessments.get(assessment_id)
            if assessment:
                return self._deep_copy_assessment(assessment)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to find assessment by ID {assessment_id}: {e}")
            raise RepositoryError(f"Failed to find assessment: {e}")

    async def find_by_target_id(self, target_id: str) -> List[RiskAssessment]:
        """Trouve toutes les évaluations pour une cible."""
        try:
            assessment_ids = self._target_index.get(target_id, [])
            assessments = []

            for assessment_id in assessment_ids:
                if assessment_id in self._assessments:
                    assessments.append(self._deep_copy_assessment(self._assessments[assessment_id]))

            # Trier par date de création (plus récent en premier)
            assessments.sort(key=lambda x: x.created_at, reverse=True)
            return assessments

        except Exception as e:
            logger.error(f"❌ Failed to find assessments for target {target_id}: {e}")
            raise RepositoryError(f"Failed to find assessments: {e}")

    async def find_latest_by_target_id(self, target_id: str) -> Optional[RiskAssessment]:
        """Trouve la dernière évaluation pour une cible."""
        try:
            assessments = await self.find_by_target_id(target_id)
            return assessments[0] if assessments else None
        except Exception as e:
            logger.error(f"❌ Failed to find latest assessment for target {target_id}: {e}")
            raise RepositoryError(f"Failed to find latest assessment: {e}")

    async def find_by_assessment_type(self, assessment_type: str) -> List[RiskAssessment]:
        """Trouve toutes les évaluations d'un type donné."""
        try:
            assessment_ids = self._type_index.get(assessment_type, [])
            assessments = []

            for assessment_id in assessment_ids:
                if assessment_id in self._assessments:
                    assessments.append(self._deep_copy_assessment(self._assessments[assessment_id]))

            return assessments

        except Exception as e:
            logger.error(f"❌ Failed to find assessments by type {assessment_type}: {e}")
            raise RepositoryError(f"Failed to find assessments by type: {e}")

    async def find_by_risk_level(self, risk_level: RiskLevel) -> List[RiskAssessment]:
        """Trouve toutes les évaluations avec un niveau de risque donné."""
        try:
            assessment_ids = self._risk_level_index.get(risk_level, [])
            assessments = []

            for assessment_id in assessment_ids:
                if assessment_id in self._assessments:
                    assessments.append(self._deep_copy_assessment(self._assessments[assessment_id]))

            return assessments

        except Exception as e:
            logger.error(f"❌ Failed to find assessments by risk level {risk_level}: {e}")
            raise RepositoryError(f"Failed to find assessments by risk level: {e}")

    async def find_critical_assessments(self) -> List[RiskAssessment]:
        """Trouve toutes les évaluations critiques."""
        try:
            critical_assessments = []
            critical_assessments.extend(await self.find_by_risk_level(RiskLevel.CRITICAL))
            critical_assessments.extend(await self.find_by_risk_level(RiskLevel.HIGH))

            # Supprimer les doublons et trier
            seen = set()
            unique_assessments = []
            for assessment in critical_assessments:
                if assessment.id not in seen:
                    seen.add(assessment.id)
                    unique_assessments.append(assessment)

            unique_assessments.sort(key=lambda x: x.created_at, reverse=True)
            return unique_assessments

        except Exception as e:
            logger.error(f"❌ Failed to find critical assessments: {e}")
            raise RepositoryError(f"Failed to find critical assessments: {e}")

    async def find_expired_assessments(self, current_time: datetime) -> List[RiskAssessment]:
        """Trouve toutes les évaluations expirées."""
        try:
            expired_assessments = []

            for assessment in self._assessments.values():
                if assessment.expires_at and assessment.expires_at < current_time:
                    expired_assessments.append(self._deep_copy_assessment(assessment))

            expired_assessments.sort(key=lambda x: x.expires_at or datetime.min)
            return expired_assessments

        except Exception as e:
            logger.error(f"❌ Failed to find expired assessments: {e}")
            raise RepositoryError(f"Failed to find expired assessments: {e}")

    async def find_breached_metrics(self) -> List[RiskAssessment]:
        """Trouve toutes les évaluations avec des métriques en dépassement."""
        try:
            breached_assessments = []

            for assessment in self._assessments.values():
                # Vérifier si une métrique dépasse ses limites
                has_breach = False
                for metric_name, metric_value in assessment.metrics.items():
                    if isinstance(metric_value, dict) and 'breach_threshold' in metric_value:
                        current_value = metric_value.get('current_value', 0)
                        threshold = metric_value.get('breach_threshold', float('inf'))
                        if current_value > threshold:
                            has_breach = True
                            break

                if has_breach:
                    breached_assessments.append(self._deep_copy_assessment(assessment))

            breached_assessments.sort(key=lambda x: x.created_at, reverse=True)
            return breached_assessments

        except Exception as e:
            logger.error(f"❌ Failed to find breached assessments: {e}")
            raise RepositoryError(f"Failed to find breached assessments: {e}")

    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[RiskAssessment]:
        """Trouve toutes les évaluations dans une période donnée."""
        try:
            filtered_assessments = []

            for assessment in self._assessments.values():
                if start_date <= assessment.created_at <= end_date:
                    filtered_assessments.append(self._deep_copy_assessment(assessment))

            filtered_assessments.sort(key=lambda x: x.created_at, reverse=True)
            return filtered_assessments

        except Exception as e:
            logger.error(f"❌ Failed to find assessments in date range: {e}")
            raise RepositoryError(f"Failed to find assessments in date range: {e}")

    async def delete(self, assessment_id: str) -> bool:
        """Supprime une évaluation de risque."""
        try:
            if assessment_id not in self._assessments:
                return False

            # Supprimer des index
            await self._remove_from_indexes(assessment_id)

            # Supprimer l'évaluation
            del self._assessments[assessment_id]

            logger.debug(f"🗑️ Risk assessment deleted: {assessment_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete assessment {assessment_id}: {e}")
            raise RepositoryError(f"Failed to delete assessment: {e}")

    async def count_by_target_id(self, target_id: str) -> int:
        """Compte le nombre d'évaluations pour une cible."""
        try:
            return len(self._target_index.get(target_id, []))
        except Exception as e:
            logger.error(f"❌ Failed to count assessments for target {target_id}: {e}")
            raise RepositoryError(f"Failed to count assessments: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les évaluations."""
        try:
            total_assessments = len(self._assessments)

            # Compter par niveau de risque
            risk_level_counts = {}
            for level in RiskLevel:
                risk_level_counts[level.value] = len(self._risk_level_index.get(level, []))

            # Compter par type
            type_counts = {k: len(v) for k, v in self._type_index.items()}

            return {
                'total_assessments': total_assessments,
                'risk_level_distribution': risk_level_counts,
                'assessment_type_distribution': type_counts,
                'targets_with_assessments': len(self._target_index),
            }

        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            raise RepositoryError(f"Failed to get statistics: {e}")

    def _add_to_indexes(self, assessment: RiskAssessment) -> None:
        """Ajoute une évaluation aux index."""
        # Index par target_id
        if assessment.target_id not in self._target_index:
            self._target_index[assessment.target_id] = []
        self._target_index[assessment.target_id].append(assessment.id)

        # Index par assessment_type
        if assessment.assessment_type not in self._type_index:
            self._type_index[assessment.assessment_type] = []
        self._type_index[assessment.assessment_type].append(assessment.id)

        # Index par risk_level
        if assessment.risk_level not in self._risk_level_index:
            self._risk_level_index[assessment.risk_level] = []
        self._risk_level_index[assessment.risk_level].append(assessment.id)

    async def _remove_from_indexes(self, assessment_id: str) -> None:
        """Supprime une évaluation des index."""
        if assessment_id not in self._assessments:
            return

        assessment = self._assessments[assessment_id]

        # Supprimer de l'index target_id
        if assessment.target_id in self._target_index:
            if assessment_id in self._target_index[assessment.target_id]:
                self._target_index[assessment.target_id].remove(assessment_id)
            if not self._target_index[assessment.target_id]:
                del self._target_index[assessment.target_id]

        # Supprimer de l'index assessment_type
        if assessment.assessment_type in self._type_index:
            if assessment_id in self._type_index[assessment.assessment_type]:
                self._type_index[assessment.assessment_type].remove(assessment_id)
            if not self._type_index[assessment.assessment_type]:
                del self._type_index[assessment.assessment_type]

        # Supprimer de l'index risk_level
        if assessment.risk_level in self._risk_level_index:
            if assessment_id in self._risk_level_index[assessment.risk_level]:
                self._risk_level_index[assessment.risk_level].remove(assessment_id)
            if not self._risk_level_index[assessment.risk_level]:
                del self._risk_level_index[assessment.risk_level]

    def _deep_copy_assessment(self, assessment: RiskAssessment) -> RiskAssessment:
        """Crée une copie profonde d'une évaluation de risque."""
        return copy.deepcopy(assessment)

    async def clear_all(self) -> None:
        """Supprime toutes les évaluations (utile pour les tests)."""
        self._assessments.clear()
        self._target_index.clear()
        self._type_index.clear()
        self._risk_level_index.clear()
        logger.debug("🧹 All risk assessments cleared")