"""
Value Object: PerformanceMetrics
==============================

Métriques de performance pour évaluer les stratégies et portfolios.
Value object immutable contenant toutes les métriques calculées.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal
import math


@dataclass(frozen=True)  # Immutable value object
class PerformanceMetrics:
    """
    Value Object contenant les métriques de performance.

    Encapsule toutes les métriques importantes pour évaluer
    la performance d'une stratégie ou d'un portfolio.
    """

    # Métriques de rendement
    total_return: Decimal
    annualized_return: Optional[Decimal] = None
    daily_returns_mean: Optional[Decimal] = None
    daily_returns_std: Optional[Decimal] = None

    # Métriques de risque
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Optional[Decimal] = None
    calmar_ratio: Optional[Decimal] = None
    max_drawdown: Decimal = Decimal("0")
    var_95: Optional[Decimal] = None  # Value at Risk 95%
    cvar_95: Optional[Decimal] = None  # Conditional VaR 95%

    # Métriques de trading
    total_trades: int = 0
    win_rate: Decimal = Decimal("0")
    profit_factor: Optional[Decimal] = None
    avg_win: Optional[Decimal] = None
    avg_loss: Optional[Decimal] = None

    # Métriques d'exposition
    current_exposure: Decimal = Decimal("0")
    max_exposure: Optional[Decimal] = None
    avg_exposure: Optional[Decimal] = None

    # Métriques temporelles
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    trading_days: Optional[int] = None

    # Métriques de volatilité
    volatility: Optional[Decimal] = None
    downside_volatility: Optional[Decimal] = None
    beta: Optional[Decimal] = None
    alpha: Optional[Decimal] = None

    # Métriques de consistance
    best_month: Optional[Decimal] = None
    worst_month: Optional[Decimal] = None
    positive_months: Optional[int] = None
    total_months: Optional[int] = None

    def __post_init__(self):
        """Validation des invariants"""
        self._validate_invariants()

    def _validate_invariants(self):
        """Valide les règles métier des métriques"""
        if self.total_trades < 0:
            raise ValueError("Total trades cannot be negative")

        if self.win_rate < 0 or self.win_rate > 1:
            raise ValueError("Win rate must be between 0 and 1")

        if self.max_drawdown < 0:
            raise ValueError("Max drawdown cannot be negative")

        if self.current_exposure < 0:
            raise ValueError("Current exposure cannot be negative")

    @property
    def loss_rate(self) -> Decimal:
        """Taux de perte (complément du win rate)"""
        return Decimal("1") - self.win_rate

    @property
    def expectancy(self) -> Optional[Decimal]:
        """Espérance mathématique par trade"""
        if not (self.avg_win and self.avg_loss and self.total_trades > 0):
            return None

        win_contribution = self.win_rate * self.avg_win
        loss_contribution = self.loss_rate * abs(self.avg_loss)

        return win_contribution - loss_contribution

    @property
    def period_days(self) -> Optional[int]:
        """Nombre de jours dans la période"""
        if not (self.period_start and self.period_end):
            return None

        return (self.period_end - self.period_start).days

    @property
    def annualized_sharpe(self) -> Decimal:
        """Sharpe ratio annualisé"""
        if not self.sharpe_ratio:
            return Decimal("0")

        # Supposons que le Sharpe est déjà annualisé
        return self.sharpe_ratio

    @property
    def risk_adjusted_return(self) -> Optional[Decimal]:
        """Rendement ajusté du risque"""
        if not (self.annualized_return and self.volatility and self.volatility > 0):
            return None

        return self.annualized_return / self.volatility

    @property
    def sterling_ratio(self) -> Optional[Decimal]:
        """Sterling ratio (rendement / drawdown moyen)"""
        if not (self.annualized_return and self.max_drawdown and self.max_drawdown > 0):
            return None

        return self.annualized_return / self.max_drawdown

    @property
    def kelly_criterion(self) -> Optional[Decimal]:
        """Critère de Kelly pour le sizing optimal"""
        if not (self.avg_win and self.avg_loss and self.win_rate > 0 and self.avg_loss > 0):
            return None

        win_loss_ratio = self.avg_win / abs(self.avg_loss)
        kelly = self.win_rate - (self.loss_rate / win_loss_ratio)

        # Limiter Kelly à des valeurs raisonnables
        return max(Decimal("0"), min(kelly, Decimal("0.25")))

    @property
    def consistency_score(self) -> Optional[Decimal]:
        """Score de consistance (0-100)"""
        if not (self.positive_months and self.total_months and self.total_months > 0):
            return None

        positive_ratio = Decimal(self.positive_months) / Decimal(self.total_months)
        return positive_ratio * 100

    @property
    def ulcer_index(self) -> Optional[Decimal]:
        """Ulcer Index - mesure de stress du drawdown"""
        if not self.max_drawdown:
            return None

        # Approximation simplifiée de l'Ulcer Index
        return self.max_drawdown * Decimal("100")

    def get_risk_score(self) -> Decimal:
        """Score de risque global (0-100, plus bas = moins risqué)"""
        risk_score = Decimal("0")

        # Contribution du drawdown (40% du score)
        if self.max_drawdown:
            drawdown_score = min(Decimal("40"), self.max_drawdown * 200)
            risk_score += drawdown_score

        # Contribution de la volatilité (30% du score)
        if self.volatility:
            vol_score = min(Decimal("30"), self.volatility * 100)
            risk_score += vol_score

        # Contribution du VaR (20% du score)
        if self.var_95:
            var_score = min(Decimal("20"), abs(self.var_95) * 100)
            risk_score += var_score

        # Contribution de l'exposition (10% du score)
        if self.current_exposure:
            exposure_score = min(Decimal("10"), self.current_exposure * 10)
            risk_score += exposure_score

        return min(risk_score, Decimal("100"))

    def get_performance_grade(self) -> str:
        """Note de performance (A+ à F)"""
        score = self._calculate_performance_score()

        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        elif score >= 40:
            return "D"
        else:
            return "F"

    def _calculate_performance_score(self) -> Decimal:
        """Calcule un score de performance global (0-100)"""
        score = Decimal("50")  # Score de base

        # Sharpe ratio (25% du score)
        if self.sharpe_ratio:
            sharpe_contribution = min(Decimal("25"), self.sharpe_ratio * 10)
            score += sharpe_contribution

        # Win rate (20% du score)
        win_rate_contribution = self.win_rate * 20
        score += win_rate_contribution

        # Rendement total (20% du score)
        if self.total_return > 0:
            return_contribution = min(Decimal("20"), self.total_return * 100)
            score += return_contribution
        else:
            score -= abs(self.total_return) * 50  # Pénalité pour pertes

        # Drawdown (15% du score, inversé)
        drawdown_penalty = self.max_drawdown * 30
        score -= drawdown_penalty

        # Consistance (10% du score)
        if self.consistency_score():
            consistency_contribution = self.consistency_score() * Decimal("0.1")
            score += consistency_contribution

        # Profit factor (10% du score)
        if self.profit_factor and self.profit_factor > 1:
            pf_contribution = min(Decimal("10"), (self.profit_factor - 1) * 5)
            score += pf_contribution

        return max(Decimal("0"), min(score, Decimal("100")))

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise les métriques en dictionnaire"""
        return {
            # Métriques de base
            "total_return": float(self.total_return),
            "annualized_return": float(self.annualized_return) if self.annualized_return else None,
            "sharpe_ratio": float(self.sharpe_ratio),
            "sortino_ratio": float(self.sortino_ratio) if self.sortino_ratio else None,
            "calmar_ratio": float(self.calmar_ratio) if self.calmar_ratio else None,
            "max_drawdown": float(self.max_drawdown),
            "volatility": float(self.volatility) if self.volatility else None,

            # Métriques de trading
            "total_trades": self.total_trades,
            "win_rate": float(self.win_rate),
            "loss_rate": float(self.loss_rate),
            "profit_factor": float(self.profit_factor) if self.profit_factor else None,
            "avg_win": float(self.avg_win) if self.avg_win else None,
            "avg_loss": float(self.avg_loss) if self.avg_loss else None,
            "expectancy": float(self.expectancy) if self.expectancy else None,

            # Métriques de risque
            "var_95": float(self.var_95) if self.var_95 else None,
            "cvar_95": float(self.cvar_95) if self.cvar_95 else None,
            "beta": float(self.beta) if self.beta else None,
            "alpha": float(self.alpha) if self.alpha else None,

            # Métriques d'exposition
            "current_exposure": float(self.current_exposure),
            "max_exposure": float(self.max_exposure) if self.max_exposure else None,
            "avg_exposure": float(self.avg_exposure) if self.avg_exposure else None,

            # Métriques dérivées
            "kelly_criterion": float(self.kelly_criterion) if self.kelly_criterion else None,
            "consistency_score": float(self.consistency_score) if self.consistency_score else None,
            "risk_score": float(self.get_risk_score()),
            "performance_grade": self.get_performance_grade(),
            "performance_score": float(self._calculate_performance_score()),

            # Métadonnées temporelles
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "period_days": self.period_days,
            "trading_days": self.trading_days
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Crée des métriques depuis un dictionnaire"""
        return cls(
            total_return=Decimal(str(data["total_return"])),
            annualized_return=Decimal(str(data["annualized_return"])) if data.get("annualized_return") else None,
            daily_returns_mean=Decimal(str(data["daily_returns_mean"])) if data.get("daily_returns_mean") else None,
            daily_returns_std=Decimal(str(data["daily_returns_std"])) if data.get("daily_returns_std") else None,
            sharpe_ratio=Decimal(str(data.get("sharpe_ratio", "0"))),
            sortino_ratio=Decimal(str(data["sortino_ratio"])) if data.get("sortino_ratio") else None,
            calmar_ratio=Decimal(str(data["calmar_ratio"])) if data.get("calmar_ratio") else None,
            max_drawdown=Decimal(str(data.get("max_drawdown", "0"))),
            var_95=Decimal(str(data["var_95"])) if data.get("var_95") else None,
            cvar_95=Decimal(str(data["cvar_95"])) if data.get("cvar_95") else None,
            total_trades=data.get("total_trades", 0),
            win_rate=Decimal(str(data.get("win_rate", "0"))),
            profit_factor=Decimal(str(data["profit_factor"])) if data.get("profit_factor") else None,
            avg_win=Decimal(str(data["avg_win"])) if data.get("avg_win") else None,
            avg_loss=Decimal(str(data["avg_loss"])) if data.get("avg_loss") else None,
            current_exposure=Decimal(str(data.get("current_exposure", "0"))),
            max_exposure=Decimal(str(data["max_exposure"])) if data.get("max_exposure") else None,
            avg_exposure=Decimal(str(data["avg_exposure"])) if data.get("avg_exposure") else None,
            period_start=datetime.fromisoformat(data["period_start"].replace('Z', '+00:00')) if data.get("period_start") else None,
            period_end=datetime.fromisoformat(data["period_end"].replace('Z', '+00:00')) if data.get("period_end") else None,
            trading_days=data.get("trading_days"),
            volatility=Decimal(str(data["volatility"])) if data.get("volatility") else None,
            downside_volatility=Decimal(str(data["downside_volatility"])) if data.get("downside_volatility") else None,
            beta=Decimal(str(data["beta"])) if data.get("beta") else None,
            alpha=Decimal(str(data["alpha"])) if data.get("alpha") else None,
            best_month=Decimal(str(data["best_month"])) if data.get("best_month") else None,
            worst_month=Decimal(str(data["worst_month"])) if data.get("worst_month") else None,
            positive_months=data.get("positive_months"),
            total_months=data.get("total_months")
        )

    def __str__(self) -> str:
        return f"PerformanceMetrics(return={self.total_return:.2%}, sharpe={self.sharpe_ratio:.2f}, dd={self.max_drawdown:.2%})"

    def __repr__(self) -> str:
        return f"PerformanceMetrics(trades={self.total_trades}, win_rate={self.win_rate:.2%})"


# Factory functions
def create_empty_metrics() -> PerformanceMetrics:
    """Crée des métriques vides"""
    return PerformanceMetrics(
        total_return=Decimal("0"),
        sharpe_ratio=Decimal("0"),
        max_drawdown=Decimal("0"),
        total_trades=0,
        win_rate=Decimal("0"),
        current_exposure=Decimal("0")
    )


def create_basic_metrics(
    total_return: Decimal,
    total_trades: int,
    winning_trades: int,
    max_drawdown: Decimal
) -> PerformanceMetrics:
    """Crée des métriques de base"""
    win_rate = Decimal(winning_trades) / Decimal(total_trades) if total_trades > 0 else Decimal("0")

    return PerformanceMetrics(
        total_return=total_return,
        sharpe_ratio=Decimal("0"),  # À calculer séparément
        max_drawdown=max_drawdown,
        total_trades=total_trades,
        win_rate=win_rate,
        current_exposure=Decimal("0")
    )