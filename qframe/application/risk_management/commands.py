"""
Application Commands: Risk Management
====================================

Commandes pour les opérations de gestion des risques.
Implémente le pattern CQRS pour la séparation des responsabilités.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, field
import uuid

from ..base.command import Command, CommandHandler, CommandResult
from ...domain.entities.risk_assessment import RiskAssessment, RiskLevel, RiskType
from ...domain.repositories.risk_assessment_repository import RiskAssessmentRepository
from ...domain.services.risk_calculation_service import (
    RiskCalculationService,
    RiskCalculationParams,
    MarketData
)
from ...domain.value_objects.position import Position


@dataclass
class CreateRiskAssessmentCommand(Command):
    """Commande pour créer une nouvelle évaluation de risque"""
    target_id: str
    assessment_type: str  # portfolio, strategy, position
    positions: Dict[str, Position]
    market_data: Dict[str, MarketData]
    calculation_params: Optional[RiskCalculationParams] = None

    def __post_init__(self):
        super().__init__()


@dataclass
class UpdateRiskAssessmentCommand(Command):
    """Commande pour mettre à jour une évaluation de risque"""
    assessment_id: str
    positions: Optional[Dict[str, Position]] = None
    market_data: Optional[Dict[str, MarketData]] = None
    force_recalculation: bool = False


@dataclass
class AddRiskMetricCommand(Command):
    """Commande pour ajouter une métrique de risque"""
    assessment_id: str
    metric_name: str
    value: Decimal
    threshold: Decimal
    risk_type: RiskType
    confidence: float = 0.95


@dataclass
class SetRiskLimitsCommand(Command):
    """Commande pour définir les limites de risque"""
    assessment_id: str
    position_limits: Optional[Dict[str, Decimal]] = None
    exposure_limits: Optional[Dict[str, Decimal]] = None


@dataclass
class AddRiskRecommendationCommand(Command):
    """Commande pour ajouter une recommandation"""
    assessment_id: str
    recommendation: str


@dataclass
class AddRiskAlertCommand(Command):
    """Commande pour ajouter une alerte de risque"""
    assessment_id: str
    alert_message: str
    severity: RiskLevel = RiskLevel.MEDIUM


@dataclass
class PerformStressTestCommand(Command):
    """Commande pour effectuer un test de stress"""
    assessment_id: str
    scenario_name: str
    market_shock_percentage: Decimal
    affected_symbols: Optional[List[str]] = None


@dataclass
class ArchiveOldAssessmentsCommand(Command):
    """Commande pour archiver les anciennes évaluations"""
    cutoff_date: datetime
    keep_latest_per_target: bool = True


class CreateRiskAssessmentHandler(CommandHandler[CreateRiskAssessmentCommand]):
    """Handler pour créer une nouvelle évaluation de risque"""

    def __init__(
        self,
        repository: RiskAssessmentRepository,
        risk_service: RiskCalculationService
    ):
        self.repository = repository
        self.risk_service = risk_service

    async def handle(self, command: CreateRiskAssessmentCommand) -> CommandResult:
        """
        Traite la commande de création d'évaluation de risque.

        Args:
            command: Commande de création

        Returns:
            Résultat avec l'ID de l'évaluation créée

        Raises:
            ValidationError: Si les données sont invalides
            RepositoryError: Si la sauvegarde échoue
        """
        try:
            # Configurer les paramètres de calcul
            if command.calculation_params:
                self.risk_service.params = command.calculation_params

            # Calculer l'évaluation de risque
            if command.assessment_type == "portfolio":
                assessment = self.risk_service.calculate_portfolio_risk(
                    command.positions,
                    command.market_data,
                    command.target_id
                )
            elif command.assessment_type in ["strategy", "position"]:
                # Pour une position unique, prendre la première
                if command.positions:
                    symbol, position = next(iter(command.positions.items()))
                    market_data = command.market_data.get(symbol)

                    if market_data:
                        assessment = self.risk_service.calculate_position_risk(
                            position,
                            market_data,
                            command.target_id
                        )
                    else:
                        # Créer une évaluation de base sans données de marché
                        assessment = RiskAssessment(
                            assessment_type=command.assessment_type,
                            target_id=command.target_id
                        )
                else:
                    assessment = RiskAssessment(
                        assessment_type=command.assessment_type,
                        target_id=command.target_id
                    )
            else:
                raise ValueError(f"Type d'évaluation non supporté: {command.assessment_type}")

            # Sauvegarder l'évaluation
            await self.repository.save(assessment)

            return CommandResult(
                success=True,
                result_data={"assessment_id": assessment.id, "risk_score": float(assessment.risk_score)},
                message=f"Évaluation de risque créée avec succès pour {command.target_id}"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Erreur lors de la création de l'évaluation: {str(e)}"
            )


class UpdateRiskAssessmentHandler(CommandHandler[UpdateRiskAssessmentCommand]):
    """Handler pour mettre à jour une évaluation de risque"""

    def __init__(
        self,
        repository: RiskAssessmentRepository,
        risk_service: RiskCalculationService
    ):
        self.repository = repository
        self.risk_service = risk_service

    async def handle(self, command: UpdateRiskAssessmentCommand) -> CommandResult:
        """
        Traite la commande de mise à jour d'évaluation.

        Args:
            command: Commande de mise à jour

        Returns:
            Résultat de la mise à jour
        """
        try:
            # Récupérer l'évaluation existante
            assessment = await self.repository.find_by_id(command.assessment_id)
            if not assessment:
                return CommandResult(
                    success=False,
                    error_message=f"Évaluation introuvable: {command.assessment_id}"
                )

            # Mettre à jour le timestamp
            assessment.assessment_time = datetime.utcnow()

            # Recalculer si demandé ou si nouvelles données fournies
            if command.force_recalculation or command.positions or command.market_data:
                positions = command.positions or {}
                market_data = command.market_data or {}

                if assessment.assessment_type == "portfolio" and positions and market_data:
                    # Recalculer l'évaluation complète
                    new_assessment = self.risk_service.calculate_portfolio_risk(
                        positions, market_data, assessment.target_id
                    )

                    # Transférer l'ID et les métadonnées
                    new_assessment.id = assessment.id
                    new_assessment.assessment_time = assessment.assessment_time
                    assessment = new_assessment

            # Sauvegarder les modifications
            await self.repository.update(assessment)

            return CommandResult(
                success=True,
                result_data={
                    "assessment_id": assessment.id,
                    "risk_score": float(assessment.risk_score),
                    "updated_at": assessment.assessment_time.isoformat()
                },
                message="Évaluation mise à jour avec succès"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Erreur lors de la mise à jour: {str(e)}"
            )


class AddRiskMetricHandler(CommandHandler[AddRiskMetricCommand]):
    """Handler pour ajouter une métrique de risque"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, command: AddRiskMetricCommand) -> CommandResult:
        """
        Traite la commande d'ajout de métrique.

        Args:
            command: Commande d'ajout de métrique

        Returns:
            Résultat de l'ajout
        """
        try:
            # Récupérer l'évaluation
            assessment = await self.repository.find_by_id(command.assessment_id)
            if not assessment:
                return CommandResult(
                    success=False,
                    error_message=f"Évaluation introuvable: {command.assessment_id}"
                )

            # Ajouter la métrique
            assessment.add_risk_metric(
                name=command.metric_name,
                value=command.value,
                threshold=command.threshold,
                risk_type=command.risk_type,
                confidence=command.confidence
            )

            # Sauvegarder
            await self.repository.update(assessment)

            return CommandResult(
                success=True,
                result_data={
                    "metric_name": command.metric_name,
                    "risk_level": assessment.risk_metrics[command.metric_name].risk_level.value
                },
                message=f"Métrique '{command.metric_name}' ajoutée avec succès"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Erreur lors de l'ajout de métrique: {str(e)}"
            )


class SetRiskLimitsHandler(CommandHandler[SetRiskLimitsCommand]):
    """Handler pour définir les limites de risque"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, command: SetRiskLimitsCommand) -> CommandResult:
        """
        Traite la commande de définition des limites.

        Args:
            command: Commande de définition des limites

        Returns:
            Résultat de la définition
        """
        try:
            # Récupérer l'évaluation
            assessment = await self.repository.find_by_id(command.assessment_id)
            if not assessment:
                return CommandResult(
                    success=False,
                    error_message=f"Évaluation introuvable: {command.assessment_id}"
                )

            # Définir les limites de position
            if command.position_limits:
                for symbol, limit in command.position_limits.items():
                    assessment.set_position_limit(symbol, limit)

            # Définir les limites d'exposition
            if command.exposure_limits:
                for asset_class, limit in command.exposure_limits.items():
                    assessment.set_exposure_limit(asset_class, limit)

            # Sauvegarder
            await self.repository.update(assessment)

            return CommandResult(
                success=True,
                result_data={
                    "position_limits_count": len(command.position_limits or {}),
                    "exposure_limits_count": len(command.exposure_limits or {})
                },
                message="Limites de risque définies avec succès"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Erreur lors de la définition des limites: {str(e)}"
            )


class AddRiskRecommendationHandler(CommandHandler[AddRiskRecommendationCommand]):
    """Handler pour ajouter une recommandation"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, command: AddRiskRecommendationCommand) -> CommandResult:
        """
        Traite la commande d'ajout de recommandation.

        Args:
            command: Commande d'ajout de recommandation

        Returns:
            Résultat de l'ajout
        """
        try:
            # Récupérer l'évaluation
            assessment = await self.repository.find_by_id(command.assessment_id)
            if not assessment:
                return CommandResult(
                    success=False,
                    error_message=f"Évaluation introuvable: {command.assessment_id}"
                )

            # Ajouter la recommandation
            assessment.add_recommendation(command.recommendation)

            # Sauvegarder
            await self.repository.update(assessment)

            return CommandResult(
                success=True,
                result_data={"recommendation": command.recommendation},
                message="Recommandation ajoutée avec succès"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Erreur lors de l'ajout de recommandation: {str(e)}"
            )


class AddRiskAlertHandler(CommandHandler[AddRiskAlertCommand]):
    """Handler pour ajouter une alerte de risque"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, command: AddRiskAlertCommand) -> CommandResult:
        """
        Traite la commande d'ajout d'alerte.

        Args:
            command: Commande d'ajout d'alerte

        Returns:
            Résultat de l'ajout
        """
        try:
            # Récupérer l'évaluation
            assessment = await self.repository.find_by_id(command.assessment_id)
            if not assessment:
                return CommandResult(
                    success=False,
                    error_message=f"Évaluation introuvable: {command.assessment_id}"
                )

            # Formater l'alerte avec la sévérité
            alert_with_severity = f"[{command.severity.value.upper()}] {command.alert_message}"
            assessment.add_alert(alert_with_severity)

            # Sauvegarder
            await self.repository.update(assessment)

            return CommandResult(
                success=True,
                result_data={
                    "alert": alert_with_severity,
                    "severity": command.severity.value
                },
                message="Alerte ajoutée avec succès"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Erreur lors de l'ajout d'alerte: {str(e)}"
            )


class PerformStressTestHandler(CommandHandler[PerformStressTestCommand]):
    """Handler pour effectuer un test de stress"""

    def __init__(
        self,
        repository: RiskAssessmentRepository,
        risk_service: RiskCalculationService
    ):
        self.repository = repository
        self.risk_service = risk_service

    async def handle(self, command: PerformStressTestCommand) -> CommandResult:
        """
        Traite la commande de test de stress.

        Args:
            command: Commande de test de stress

        Returns:
            Résultat du test de stress
        """
        try:
            # Récupérer l'évaluation
            assessment = await self.repository.find_by_id(command.assessment_id)
            if not assessment:
                return CommandResult(
                    success=False,
                    error_message=f"Évaluation introuvable: {command.assessment_id}"
                )

            # Simuler l'impact du choc de marché
            # Pour simplifier, on calcule un impact proportionnel
            current_risk_score = assessment.risk_score
            shock_impact = current_risk_score * (command.market_shock_percentage / 100)

            # Ajouter le résultat du test de stress
            assessment.add_stress_test_result(command.scenario_name, shock_impact)

            # Ajouter des recommandations si l'impact est élevé
            if shock_impact > current_risk_score * Decimal("0.5"):
                assessment.add_recommendation(
                    f"Impact élevé détecté pour le scénario '{command.scenario_name}': "
                    f"réduire l'exposition aux actifs affectés"
                )
                assessment.add_alert(
                    f"STRESS TEST CRITIQUE: {command.scenario_name} - Impact: {float(shock_impact):.2f}%"
                )

            # Sauvegarder
            await self.repository.update(assessment)

            return CommandResult(
                success=True,
                result_data={
                    "scenario": command.scenario_name,
                    "impact": float(shock_impact),
                    "shock_percentage": float(command.market_shock_percentage)
                },
                message=f"Test de stress '{command.scenario_name}' effectué avec succès"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Erreur lors du test de stress: {str(e)}"
            )


class ArchiveOldAssessmentsHandler(CommandHandler[ArchiveOldAssessmentsCommand]):
    """Handler pour archiver les anciennes évaluations"""

    def __init__(self, repository: RiskAssessmentRepository):
        self.repository = repository

    async def handle(self, command: ArchiveOldAssessmentsCommand) -> CommandResult:
        """
        Traite la commande d'archivage.

        Args:
            command: Commande d'archivage

        Returns:
            Résultat de l'archivage
        """
        try:
            # Effectuer l'archivage
            archived_count = await self.repository.archive_old_assessments(
                command.cutoff_date,
                command.keep_latest_per_target
            )

            return CommandResult(
                success=True,
                result_data={
                    "archived_count": archived_count,
                    "cutoff_date": command.cutoff_date.isoformat()
                },
                message=f"{archived_count} évaluations archivées avec succès"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Erreur lors de l'archivage: {str(e)}"
            )