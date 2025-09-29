"""
Position Reconciler
===================

Reconciliation automatique des positions entre le framework
et le broker pour assurer la cohérence des données.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

from ...core.container import injectable
from ...domain.entities.position import Position
from ...domain.entities.portfolio import Portfolio
from .broker_adapters import BrokerAdapter


logger = logging.getLogger(__name__)


class DiscrepancyType(str, Enum):
    """Types de divergences détectées"""
    QUANTITY_MISMATCH = "quantity_mismatch"
    MISSING_POSITION = "missing_position"
    EXTRA_POSITION = "extra_position"
    PRICE_MISMATCH = "price_mismatch"
    PNL_MISMATCH = "pnl_mismatch"


@dataclass
class PositionDiscrepancy:
    """Divergence détectée entre positions"""
    symbol: str
    discrepancy_type: DiscrepancyType
    framework_position: Optional[Position]
    broker_position: Optional[Position]
    difference: Optional[Decimal] = None
    severity: str = "medium"  # "low", "medium", "high", "critical"
    detected_at: datetime = datetime.utcnow()


@dataclass
class ReconciliationReport:
    """Rapport de réconciliation"""
    timestamp: datetime
    total_positions_checked: int
    discrepancies_found: List[PositionDiscrepancy]
    reconciliation_actions_taken: List[str]
    success: bool
    error_message: Optional[str] = None

    @property
    def has_discrepancies(self) -> bool:
        return len(self.discrepancies_found) > 0

    @property
    def critical_discrepancies(self) -> List[PositionDiscrepancy]:
        return [d for d in self.discrepancies_found if d.severity == "critical"]


@injectable
class PositionReconciler:
    """
    Réconciliateur de positions.

    Responsabilités:
    - Comparaison périodique positions framework vs broker
    - Détection automatique des divergences
    - Actions correctives automatiques
    - Alertes pour divergences critiques
    """

    def __init__(
        self,
        broker_adapter: BrokerAdapter,
        tolerance_percentage: Decimal = Decimal("0.01"),  # 1% tolerance
        auto_reconcile: bool = True,
        critical_threshold: Decimal = Decimal("1000")  # $1000 pour divergence critique
    ):
        self.broker_adapter = broker_adapter
        self.tolerance_percentage = tolerance_percentage
        self.auto_reconcile = auto_reconcile
        self.critical_threshold = critical_threshold

        # État interne
        self.framework_positions: Dict[str, Position] = {}
        self.last_reconciliation: Optional[datetime] = None
        self.reconciliation_history: List[ReconciliationReport] = []

        # Callbacks pour alertes
        self.on_discrepancy_detected = None
        self.on_critical_discrepancy = None

    async def reconcile_positions(self) -> ReconciliationReport:
        """
        Effectue une réconciliation complète des positions.

        Returns:
            Rapport détaillé de la réconciliation
        """
        start_time = datetime.utcnow()

        try:
            # Récupérer positions depuis le broker
            broker_positions = await self.broker_adapter.get_positions()

            # Convertir en dictionnaire pour faciliter comparaison
            broker_pos_dict = {pos.symbol: pos for pos in broker_positions}

            # Détecter divergences
            discrepancies = await self._detect_discrepancies(
                self.framework_positions,
                broker_pos_dict
            )

            # Actions correctives
            actions_taken = []
            if self.auto_reconcile and discrepancies:
                actions_taken = await self._perform_auto_reconciliation(discrepancies)

            # Créer rapport
            report = ReconciliationReport(
                timestamp=start_time,
                total_positions_checked=len(set(
                    list(self.framework_positions.keys()) + list(broker_pos_dict.keys())
                )),
                discrepancies_found=discrepancies,
                reconciliation_actions_taken=actions_taken,
                success=True
            )

            # Enregistrer dans historique
            self.reconciliation_history.append(report)
            self.last_reconciliation = start_time

            # Alertes
            await self._process_alerts(report)

            # Log résultats
            if discrepancies:
                logger.warning(f"Reconciliation found {len(discrepancies)} discrepancies")
                for disc in discrepancies:
                    logger.warning(f"  - {disc.symbol}: {disc.discrepancy_type.value}")
            else:
                logger.info("Position reconciliation: no discrepancies found")

            return report

        except Exception as e:
            logger.error(f"Error during position reconciliation: {e}")

            error_report = ReconciliationReport(
                timestamp=start_time,
                total_positions_checked=0,
                discrepancies_found=[],
                reconciliation_actions_taken=[],
                success=False,
                error_message=str(e)
            )

            self.reconciliation_history.append(error_report)
            return error_report

    async def update_framework_position(self, position: Position) -> None:
        """Met à jour une position dans le framework"""
        self.framework_positions[position.symbol] = position

    async def remove_framework_position(self, symbol: str) -> None:
        """Supprime une position du framework"""
        if symbol in self.framework_positions:
            del self.framework_positions[symbol]

    async def get_reconciliation_status(self) -> Dict[str, Any]:
        """Retourne le statut de réconciliation actuel"""
        if not self.reconciliation_history:
            return {"status": "never_run"}

        latest_report = self.reconciliation_history[-1]

        return {
            "last_reconciliation": self.last_reconciliation.isoformat() if self.last_reconciliation else None,
            "total_reconciliations": len(self.reconciliation_history),
            "latest_success": latest_report.success,
            "latest_discrepancies_count": len(latest_report.discrepancies_found),
            "critical_discrepancies_count": len(latest_report.critical_discrepancies),
            "framework_positions_count": len(self.framework_positions),
            "auto_reconcile_enabled": self.auto_reconcile
        }

    async def get_discrepancy_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des divergences récentes"""
        if not self.reconciliation_history:
            return {"no_data": True}

        # Analyser les 10 derniers rapports
        recent_reports = self.reconciliation_history[-10:]

        total_discrepancies = sum(len(r.discrepancies_found) for r in recent_reports)
        critical_discrepancies = sum(len(r.critical_discrepancies) for r in recent_reports)

        # Compteur par type
        discrepancy_types = {}
        for report in recent_reports:
            for disc in report.discrepancies_found:
                discrepancy_types[disc.discrepancy_type.value] = \
                    discrepancy_types.get(disc.discrepancy_type.value, 0) + 1

        return {
            "total_discrepancies_recent": total_discrepancies,
            "critical_discrepancies_recent": critical_discrepancies,
            "discrepancy_types_breakdown": discrepancy_types,
            "reconciliation_success_rate": sum(1 for r in recent_reports if r.success) / len(recent_reports) * 100,
            "reports_analyzed": len(recent_reports)
        }

    # === Détection de divergences ===

    async def _detect_discrepancies(
        self,
        framework_positions: Dict[str, Position],
        broker_positions: Dict[str, Position]
    ) -> List[PositionDiscrepancy]:
        """Détecte les divergences entre les positions"""
        discrepancies = []

        # Obtenir tous les symboles
        all_symbols = set(framework_positions.keys()) | set(broker_positions.keys())

        for symbol in all_symbols:
            framework_pos = framework_positions.get(symbol)
            broker_pos = broker_positions.get(symbol)

            # Position manquante dans framework
            if broker_pos and not framework_pos:
                discrepancies.append(PositionDiscrepancy(
                    symbol=symbol,
                    discrepancy_type=DiscrepancyType.MISSING_POSITION,
                    framework_position=None,
                    broker_position=broker_pos,
                    severity=self._calculate_severity(None, broker_pos)
                ))

            # Position extra dans framework
            elif framework_pos and not broker_pos:
                discrepancies.append(PositionDiscrepancy(
                    symbol=symbol,
                    discrepancy_type=DiscrepancyType.EXTRA_POSITION,
                    framework_position=framework_pos,
                    broker_position=None,
                    severity=self._calculate_severity(framework_pos, None)
                ))

            # Positions existent dans les deux - vérifier différences
            elif framework_pos and broker_pos:
                position_discrepancies = await self._compare_positions(framework_pos, broker_pos)
                discrepancies.extend(position_discrepancies)

        return discrepancies

    async def _compare_positions(
        self,
        framework_pos: Position,
        broker_pos: Position
    ) -> List[PositionDiscrepancy]:
        """Compare deux positions et détecte les divergences"""
        discrepancies = []

        # Vérifier quantité
        qty_diff = abs(framework_pos.quantity - broker_pos.quantity)
        qty_tolerance = abs(framework_pos.quantity) * self.tolerance_percentage

        if qty_diff > qty_tolerance:
            discrepancies.append(PositionDiscrepancy(
                symbol=framework_pos.symbol,
                discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                framework_position=framework_pos,
                broker_position=broker_pos,
                difference=qty_diff,
                severity=self._calculate_severity(framework_pos, broker_pos)
            ))

        # Vérifier prix moyen (si disponible)
        if (framework_pos.average_price and broker_pos.average_price):
            price_diff = abs(framework_pos.average_price - broker_pos.average_price)
            price_tolerance = framework_pos.average_price * self.tolerance_percentage

            if price_diff > price_tolerance:
                discrepancies.append(PositionDiscrepancy(
                    symbol=framework_pos.symbol,
                    discrepancy_type=DiscrepancyType.PRICE_MISMATCH,
                    framework_position=framework_pos,
                    broker_position=broker_pos,
                    difference=price_diff,
                    severity="medium"
                ))

        # Vérifier PnL (si disponible)
        if (framework_pos.unrealized_pnl is not None and
            broker_pos.unrealized_pnl is not None):
            pnl_diff = abs(framework_pos.unrealized_pnl - broker_pos.unrealized_pnl)

            if pnl_diff > self.critical_threshold * Decimal("0.1"):  # 10% du seuil critique
                discrepancies.append(PositionDiscrepancy(
                    symbol=framework_pos.symbol,
                    discrepancy_type=DiscrepancyType.PNL_MISMATCH,
                    framework_position=framework_pos,
                    broker_position=broker_pos,
                    difference=pnl_diff,
                    severity=self._calculate_pnl_severity(pnl_diff)
                ))

        return discrepancies

    def _calculate_severity(
        self,
        framework_pos: Optional[Position],
        broker_pos: Optional[Position]
    ) -> str:
        """Calcule la sévérité d'une divergence"""
        # Calculer valeur de la divergence
        if framework_pos and broker_pos:
            # Divergence quantité
            qty_diff = abs(framework_pos.quantity - broker_pos.quantity)
            value_diff = qty_diff * max(framework_pos.average_price or Decimal("0"),
                                       broker_pos.average_price or Decimal("0"))
        elif framework_pos:
            value_diff = abs(framework_pos.market_value or
                           (framework_pos.quantity * (framework_pos.average_price or Decimal("0"))))
        elif broker_pos:
            value_diff = abs(broker_pos.market_value or
                           (broker_pos.quantity * (broker_pos.average_price or Decimal("0"))))
        else:
            value_diff = Decimal("0")

        # Déterminer sévérité basée sur valeur
        if value_diff >= self.critical_threshold:
            return "critical"
        elif value_diff >= self.critical_threshold * Decimal("0.5"):
            return "high"
        elif value_diff >= self.critical_threshold * Decimal("0.1"):
            return "medium"
        else:
            return "low"

    def _calculate_pnl_severity(self, pnl_diff: Decimal) -> str:
        """Calcule la sévérité basée sur divergence PnL"""
        if pnl_diff >= self.critical_threshold:
            return "critical"
        elif pnl_diff >= self.critical_threshold * Decimal("0.5"):
            return "high"
        else:
            return "medium"

    # === Actions correctives ===

    async def _perform_auto_reconciliation(
        self,
        discrepancies: List[PositionDiscrepancy]
    ) -> List[str]:
        """Effectue des actions correctives automatiques"""
        actions_taken = []

        for discrepancy in discrepancies:
            # Seulement traiter divergences non-critiques automatiquement
            if discrepancy.severity == "critical":
                actions_taken.append(f"MANUAL_REVIEW_REQUIRED: {discrepancy.symbol} - critical discrepancy")
                continue

            try:
                if discrepancy.discrepancy_type == DiscrepancyType.MISSING_POSITION:
                    # Ajouter position manquante au framework
                    if discrepancy.broker_position:
                        await self.update_framework_position(discrepancy.broker_position)
                        actions_taken.append(f"ADDED: {discrepancy.symbol} position to framework")

                elif discrepancy.discrepancy_type == DiscrepancyType.EXTRA_POSITION:
                    # Supprimer position extra du framework (si petite)
                    if (discrepancy.framework_position and
                        discrepancy.severity in ["low", "medium"]):
                        await self.remove_framework_position(discrepancy.symbol)
                        actions_taken.append(f"REMOVED: {discrepancy.symbol} extra position from framework")

                elif discrepancy.discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH:
                    # Synchroniser quantité (si divergence faible)
                    if (discrepancy.severity == "low" and
                        discrepancy.broker_position):
                        # Créer position corrigée
                        corrected_position = Position(
                            symbol=discrepancy.broker_position.symbol,
                            quantity=discrepancy.broker_position.quantity,
                            average_price=discrepancy.broker_position.average_price,
                            market_value=discrepancy.broker_position.market_value,
                            unrealized_pnl=discrepancy.broker_position.unrealized_pnl,
                            timestamp=datetime.utcnow()
                        )
                        await self.update_framework_position(corrected_position)
                        actions_taken.append(f"CORRECTED: {discrepancy.symbol} quantity mismatch")

            except Exception as e:
                actions_taken.append(f"ERROR: Failed to reconcile {discrepancy.symbol} - {e}")

        return actions_taken

    # === Alertes ===

    async def _process_alerts(self, report: ReconciliationReport) -> None:
        """Traite les alertes basées sur le rapport"""
        # Alertes pour divergences critiques
        critical_discrepancies = report.critical_discrepancies
        if critical_discrepancies:
            logger.critical(f"CRITICAL: {len(critical_discrepancies)} critical position discrepancies detected!")

            if self.on_critical_discrepancy:
                await self.on_critical_discrepancy(critical_discrepancies)

        # Alertes générales
        if report.has_discrepancies and self.on_discrepancy_detected:
            await self.on_discrepancy_detected(report.discrepancies_found)

        # Log détaillé des divergences
        for discrepancy in report.discrepancies_found:
            logger.warning(
                f"Position discrepancy - {discrepancy.symbol}: "
                f"{discrepancy.discrepancy_type.value} (severity: {discrepancy.severity})"
            )