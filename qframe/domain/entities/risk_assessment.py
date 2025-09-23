"""
Domain Entity: RiskAssessment
============================

Entité représentant une évaluation de risque complète.
Encapsule tous les calculs et métriques de risque pour un portfolio ou position.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import uuid

from ..value_objects.position import Position
from ..value_objects.performance_metrics import PerformanceMetrics


class RiskLevel(str, Enum):
    """Niveaux de risque"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class RiskType(str, Enum):
    """Types de risques"""
    MARKET = "market"
    CREDIT = "credit"
    LIQUIDITY = "liquidity"
    OPERATIONAL = "operational"
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"


@dataclass
class RiskMetric:
    """Métrique de risque individuelle"""
    name: str
    value: Decimal
    threshold: Decimal
    risk_level: RiskLevel
    risk_type: RiskType
    currency: str = "USD"
    confidence: float = 0.95
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_breached(self) -> bool:
        """Vérifie si la métrique dépasse le seuil"""
        return abs(self.value) > abs(self.threshold)

    @property
    def breach_percentage(self) -> Decimal:
        """Pourcentage de dépassement du seuil"""
        if self.threshold == 0:
            return Decimal("0")
        return (abs(self.value) / abs(self.threshold) - 1) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise la métrique en dictionnaire"""
        return {
            "name": self.name,
            "value": float(self.value),
            "threshold": float(self.threshold),
            "risk_level": self.risk_level.value,
            "risk_type": self.risk_type.value,
            "currency": self.currency,
            "confidence": self.confidence,
            "is_breached": self.is_breached,
            "breach_percentage": float(self.breach_percentage),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RiskAssessment:
    """
    Entité représentant une évaluation complète de risque.

    Encapsule tous les calculs de risque pour un portfolio,
    une stratégie ou une position spécifique.
    """

    # Identité
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_type: str = "portfolio"  # portfolio, strategy, position
    target_id: str = ""  # ID du portfolio/stratégie/position évalué

    # Métadonnées temporelles
    assessment_time: datetime = field(default_factory=datetime.utcnow)
    market_data_time: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Niveau de risque global
    overall_risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: Decimal = Decimal("50")  # 0-100

    # Métriques de risque
    risk_metrics: Dict[str, RiskMetric] = field(default_factory=dict)

    # Limites et seuils
    position_limits: Dict[str, Decimal] = field(default_factory=dict)
    exposure_limits: Dict[str, Decimal] = field(default_factory=dict)

    # Corrélations et concentrations
    correlations: Dict[str, Decimal] = field(default_factory=dict)
    concentrations: Dict[str, Decimal] = field(default_factory=dict)

    # Scénarios de stress
    stress_test_results: Dict[str, Decimal] = field(default_factory=dict)

    # Recommandations
    recommendations: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validation post-initialisation"""
        self._validate_invariants()

    def _validate_invariants(self):
        """Valide les invariants métier"""
        if not self.target_id:
            raise ValueError("target_id cannot be empty")

        if self.risk_score < 0 or self.risk_score > 100:
            raise ValueError("risk_score must be between 0 and 100")

        if self.assessment_type not in ["portfolio", "strategy", "position"]:
            raise ValueError("assessment_type must be portfolio, strategy, or position")

    def add_risk_metric(
        self,
        name: str,
        value: Decimal,
        threshold: Decimal,
        risk_type: RiskType,
        confidence: float = 0.95
    ) -> None:
        """Ajoute une métrique de risque"""
        # Déterminer le niveau de risque automatiquement
        risk_level = self._calculate_risk_level(value, threshold)

        metric = RiskMetric(
            name=name,
            value=value,
            threshold=threshold,
            risk_level=risk_level,
            risk_type=risk_type,
            confidence=confidence
        )

        self.risk_metrics[name] = metric
        self._update_overall_risk_level()

    def get_breached_metrics(self) -> List[RiskMetric]:
        """Retourne les métriques qui dépassent leurs seuils"""
        return [metric for metric in self.risk_metrics.values() if metric.is_breached]

    def get_metrics_by_type(self, risk_type: RiskType) -> List[RiskMetric]:
        """Retourne les métriques d'un type donné"""
        return [metric for metric in self.risk_metrics.values() if metric.risk_type == risk_type]

    def get_metrics_by_level(self, risk_level: RiskLevel) -> List[RiskMetric]:
        """Retourne les métriques d'un niveau de risque donné"""
        return [metric for metric in self.risk_metrics.values() if metric.risk_level == risk_level]

    def add_recommendation(self, recommendation: str) -> None:
        """Ajoute une recommandation de gestion des risques"""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)

    def add_alert(self, alert: str) -> None:
        """Ajoute une alerte de risque"""
        if alert not in self.alerts:
            self.alerts.append(alert)

    def set_position_limit(self, symbol: str, limit: Decimal) -> None:
        """Définit une limite de position pour un symbole"""
        if limit <= 0:
            raise ValueError("Position limit must be positive")
        self.position_limits[symbol] = limit

    def set_exposure_limit(self, asset_class: str, limit: Decimal) -> None:
        """Définit une limite d'exposition pour une classe d'actifs"""
        if limit <= 0:
            raise ValueError("Exposure limit must be positive")
        self.exposure_limits[asset_class] = limit

    def add_correlation(self, pair: str, correlation: Decimal) -> None:
        """Ajoute une corrélation entre deux actifs"""
        if correlation < -1 or correlation > 1:
            raise ValueError("Correlation must be between -1 and 1")
        self.correlations[pair] = correlation

    def add_stress_test_result(self, scenario: str, impact: Decimal) -> None:
        """Ajoute le résultat d'un test de stress"""
        self.stress_test_results[scenario] = impact

    def is_within_limits(self, positions: Dict[str, Position]) -> bool:
        """Vérifie si les positions respectent les limites"""
        for symbol, position in positions.items():
            if symbol in self.position_limits:
                position_size = abs(position.market_value)
                if position_size > self.position_limits[symbol]:
                    return False
        return True

    def calculate_total_exposure(self, positions: Dict[str, Position]) -> Decimal:
        """Calcule l'exposition totale du portfolio"""
        return sum(abs(position.market_value) for position in positions.values())

    def calculate_concentration_risk(self, positions: Dict[str, Position]) -> Dict[str, Decimal]:
        """Calcule le risque de concentration par position"""
        total_value = self.calculate_total_exposure(positions)

        if total_value == 0:
            return {}

        concentrations = {}
        for symbol, position in positions.items():
            concentration = abs(position.market_value) / total_value * 100
            concentrations[symbol] = concentration

        return concentrations

    def get_risk_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'évaluation de risque"""
        breached_count = len(self.get_breached_metrics())
        high_risk_count = len(self.get_metrics_by_level(RiskLevel.HIGH)) + \
                         len(self.get_metrics_by_level(RiskLevel.VERY_HIGH)) + \
                         len(self.get_metrics_by_level(RiskLevel.CRITICAL))

        return {
            "overall_risk_level": self.overall_risk_level.value,
            "risk_score": float(self.risk_score),
            "total_metrics": len(self.risk_metrics),
            "breached_metrics": breached_count,
            "high_risk_metrics": high_risk_count,
            "total_alerts": len(self.alerts),
            "total_recommendations": len(self.recommendations),
            "assessment_time": self.assessment_time.isoformat(),
            "is_valid": self.is_valid()
        }

    def is_valid(self) -> bool:
        """Vérifie si l'évaluation est encore valide"""
        if self.valid_until is None:
            return True
        return datetime.utcnow() < self.valid_until

    def is_critical(self) -> bool:
        """Vérifie si l'évaluation contient des risques critiques"""
        return (
            self.overall_risk_level == RiskLevel.CRITICAL or
            len(self.get_metrics_by_level(RiskLevel.CRITICAL)) > 0 or
            len(self.get_breached_metrics()) > 0
        )

    def _calculate_risk_level(self, value: Decimal, threshold: Decimal) -> RiskLevel:
        """Calcule le niveau de risque basé sur la valeur et le seuil"""
        if threshold == 0:
            return RiskLevel.LOW

        ratio = abs(value) / abs(threshold)

        if ratio < 0.3:
            return RiskLevel.VERY_LOW
        elif ratio < 0.6:
            return RiskLevel.LOW
        elif ratio < 0.8:
            return RiskLevel.MEDIUM
        elif ratio < 1.0:
            return RiskLevel.HIGH
        elif ratio < 1.5:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.CRITICAL

    def _update_overall_risk_level(self) -> None:
        """Met à jour le niveau de risque global basé sur les métriques"""
        if not self.risk_metrics:
            self.overall_risk_level = RiskLevel.LOW
            self.risk_score = Decimal("20")
            return

        # Calculer le score basé sur les métriques
        level_scores = {
            RiskLevel.VERY_LOW: 10,
            RiskLevel.LOW: 25,
            RiskLevel.MEDIUM: 50,
            RiskLevel.HIGH: 75,
            RiskLevel.VERY_HIGH: 90,
            RiskLevel.CRITICAL: 100
        }

        # Score pondéré par le nombre de métriques
        total_score = 0
        for metric in self.risk_metrics.values():
            total_score += level_scores[metric.risk_level]

        avg_score = total_score / len(self.risk_metrics)
        self.risk_score = Decimal(str(avg_score))

        # Déterminer le niveau global
        if avg_score < 20:
            self.overall_risk_level = RiskLevel.VERY_LOW
        elif avg_score < 40:
            self.overall_risk_level = RiskLevel.LOW
        elif avg_score < 60:
            self.overall_risk_level = RiskLevel.MEDIUM
        elif avg_score < 80:
            self.overall_risk_level = RiskLevel.HIGH
        elif avg_score < 95:
            self.overall_risk_level = RiskLevel.VERY_HIGH
        else:
            self.overall_risk_level = RiskLevel.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise l'évaluation en dictionnaire"""
        return {
            "id": self.id,
            "assessment_type": self.assessment_type,
            "target_id": self.target_id,
            "assessment_time": self.assessment_time.isoformat(),
            "market_data_time": self.market_data_time.isoformat() if self.market_data_time else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "overall_risk_level": self.overall_risk_level.value,
            "risk_score": float(self.risk_score),
            "risk_metrics": {name: metric.to_dict() for name, metric in self.risk_metrics.items()},
            "position_limits": {k: float(v) for k, v in self.position_limits.items()},
            "exposure_limits": {k: float(v) for k, v in self.exposure_limits.items()},
            "correlations": {k: float(v) for k, v in self.correlations.items()},
            "concentrations": {k: float(v) for k, v in self.concentrations.items()},
            "stress_test_results": {k: float(v) for k, v in self.stress_test_results.items()},
            "recommendations": self.recommendations,
            "alerts": self.alerts,
            "summary": self.get_risk_summary()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskAssessment":
        """Crée une évaluation depuis un dictionnaire"""
        assessment = cls(
            id=data.get("id", str(uuid.uuid4())),
            assessment_type=data.get("assessment_type", "portfolio"),
            target_id=data["target_id"],
            assessment_time=datetime.fromisoformat(data["assessment_time"].replace('Z', '+00:00')),
            market_data_time=datetime.fromisoformat(data["market_data_time"].replace('Z', '+00:00')) if data.get("market_data_time") else None,
            valid_until=datetime.fromisoformat(data["valid_until"].replace('Z', '+00:00')) if data.get("valid_until") else None,
            overall_risk_level=RiskLevel(data.get("overall_risk_level", "medium")),
            risk_score=Decimal(str(data.get("risk_score", "50"))),
            recommendations=data.get("recommendations", []),
            alerts=data.get("alerts", [])
        )

        # Reconstruire les métriques de risque
        for name, metric_data in data.get("risk_metrics", {}).items():
            metric = RiskMetric(
                name=metric_data["name"],
                value=Decimal(str(metric_data["value"])),
                threshold=Decimal(str(metric_data["threshold"])),
                risk_level=RiskLevel(metric_data["risk_level"]),
                risk_type=RiskType(metric_data["risk_type"]),
                confidence=metric_data.get("confidence", 0.95),
                timestamp=datetime.fromisoformat(metric_data["timestamp"].replace('Z', '+00:00'))
            )
            assessment.risk_metrics[name] = metric

        # Reconstruire les autres dictionnaires
        assessment.position_limits = {k: Decimal(str(v)) for k, v in data.get("position_limits", {}).items()}
        assessment.exposure_limits = {k: Decimal(str(v)) for k, v in data.get("exposure_limits", {}).items()}
        assessment.correlations = {k: Decimal(str(v)) for k, v in data.get("correlations", {}).items()}
        assessment.concentrations = {k: Decimal(str(v)) for k, v in data.get("concentrations", {}).items()}
        assessment.stress_test_results = {k: Decimal(str(v)) for k, v in data.get("stress_test_results", {}).items()}

        return assessment

    def __str__(self) -> str:
        return f"RiskAssessment(target={self.target_id}, level={self.overall_risk_level.value}, score={self.risk_score})"

    def __repr__(self) -> str:
        return f"RiskAssessment(id={self.id}, target_id={self.target_id})"


# Factory functions
def create_portfolio_risk_assessment(portfolio_id: str) -> RiskAssessment:
    """Factory pour créer une évaluation de risque de portfolio"""
    return RiskAssessment(
        assessment_type="portfolio",
        target_id=portfolio_id
    )


def create_strategy_risk_assessment(strategy_id: str) -> RiskAssessment:
    """Factory pour créer une évaluation de risque de stratégie"""
    return RiskAssessment(
        assessment_type="strategy",
        target_id=strategy_id
    )


def create_position_risk_assessment(position_id: str) -> RiskAssessment:
    """Factory pour créer une évaluation de risque de position"""
    return RiskAssessment(
        assessment_type="position",
        target_id=position_id
    )