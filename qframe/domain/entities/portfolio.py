"""
Domain Entity: Portfolio
========================

Entité agrégat représentant un portfolio de trading.
Encapsule les positions, la valorisation, et les performances.
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import uuid

from ..value_objects.position import Position
from ..value_objects.performance_metrics import PerformanceMetrics
from ..entities.risk_assessment import RiskAssessment


class PortfolioStatus(str, Enum):
    """Statuts possibles d'un portfolio"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LIQUIDATING = "liquidating"
    FROZEN = "frozen"
    ARCHIVED = "archived"


class PortfolioType(str, Enum):
    """Types de portfolios"""
    TRADING = "trading"
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    RESEARCH = "research"


class RebalancingFrequency(str, Enum):
    """Fréquences de rééquilibrage"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    MANUAL = "manual"


@dataclass
class PortfolioConstraints:
    """Contraintes et limites du portfolio"""
    max_positions: int = 50
    max_position_weight: Decimal = Decimal("0.20")  # 20% max par position
    max_sector_weight: Decimal = Decimal("0.30")   # 30% max par secteur
    max_leverage: Decimal = Decimal("2.0")         # Levier maximum
    min_cash_percentage: Decimal = Decimal("0.05")  # 5% cash minimum
    max_drawdown: Decimal = Decimal("0.15")        # 15% drawdown max
    rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.WEEKLY

    def validate(self) -> List[str]:
        """Valide les contraintes et retourne les erreurs"""
        errors = []

        if self.max_positions <= 0:
            errors.append("max_positions must be positive")

        if not 0 < self.max_position_weight <= 1:
            errors.append("max_position_weight must be between 0 and 1")

        if not 0 < self.max_sector_weight <= 1:
            errors.append("max_sector_weight must be between 0 and 1")

        if self.max_leverage < 1:
            errors.append("max_leverage must be >= 1")

        if not 0 <= self.min_cash_percentage <= 1:
            errors.append("min_cash_percentage must be between 0 and 1")

        if not 0 < self.max_drawdown <= 1:
            errors.append("max_drawdown must be between 0 and 1")

        return errors


@dataclass
class PortfolioSnapshot:
    """Snapshot instantané d'un portfolio"""
    timestamp: datetime
    total_value: Decimal
    cash: Decimal
    positions_count: int
    largest_position_weight: Decimal
    risk_score: Optional[Decimal] = None
    daily_pnl: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_value": float(self.total_value),
            "cash": float(self.cash),
            "positions_count": self.positions_count,
            "largest_position_weight": float(self.largest_position_weight),
            "risk_score": float(self.risk_score) if self.risk_score else None,
            "daily_pnl": float(self.daily_pnl) if self.daily_pnl else None
        }


@dataclass
class Portfolio:
    """
    Entité agrégat Portfolio.

    Représente un portfolio de trading avec ses positions,
    sa valorisation, ses performances et ses contraintes.
    """

    # Identité et métadonnées
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    portfolio_type: PortfolioType = PortfolioType.TRADING
    status: PortfolioStatus = PortfolioStatus.ACTIVE

    # Dates importantes
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_rebalanced_at: Optional[datetime] = None

    # Capital et valorisation
    initial_capital: Decimal = Decimal("100000")
    current_cash: Decimal = field(init=False)
    total_value: Decimal = field(init=False)

    # Positions et allocations
    positions: Dict[str, Position] = field(default_factory=dict)
    target_allocations: Dict[str, Decimal] = field(default_factory=dict)  # Allocations cibles

    # Contraintes et limites
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)

    # Performance et historique
    performance_metrics: Optional[PerformanceMetrics] = None
    snapshots: List[PortfolioSnapshot] = field(default_factory=list)

    # Risk assessment référence
    current_risk_assessment_id: Optional[str] = None

    # Métadonnées de trading
    strategy_ids: Set[str] = field(default_factory=set)  # Stratégies actives
    benchmark_symbol: str = "SPY"
    currency: str = "USD"

    def __post_init__(self):
        """Initialisation post-création"""
        self.current_cash = self.initial_capital
        self.total_value = self.initial_capital
        self._validate_invariants()

    def _validate_invariants(self):
        """Valide les invariants métier du portfolio"""
        if not self.name:
            raise ValueError("Portfolio name cannot be empty")

        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        constraint_errors = self.constraints.validate()
        if constraint_errors:
            raise ValueError(f"Invalid constraints: {'; '.join(constraint_errors)}")

    # === Gestion des positions ===

    def add_position(self, position: Position) -> None:
        """
        Ajoute ou met à jour une position dans le portfolio.

        Args:
            position: Position à ajouter

        Raises:
            ValueError: Si la position viole les contraintes
        """
        # Vérifier les contraintes avant d'ajouter
        if len(self.positions) >= self.constraints.max_positions and position.symbol not in self.positions:
            raise ValueError(f"Cannot add position: max positions limit ({self.constraints.max_positions}) reached")

        # Calculer le poids de la position
        if self.total_value > 0:
            position_weight = abs(position.market_value) / self.total_value
            if position_weight > self.constraints.max_position_weight:
                raise ValueError(
                    f"Position weight {position_weight:.3f} exceeds maximum {self.constraints.max_position_weight:.3f}"
                )

        self.positions[position.symbol] = position
        self._update_portfolio_values()
        self.updated_at = datetime.utcnow()

    def remove_position(self, symbol: str) -> Optional[Position]:
        """
        Retire une position du portfolio.

        Args:
            symbol: Symbole de la position à retirer

        Returns:
            Position retirée ou None si inexistante
        """
        position = self.positions.pop(symbol, None)
        if position:
            # Ajouter la valeur de la position au cash
            self.current_cash += position.market_value
            self._update_portfolio_values()
            self.updated_at = datetime.utcnow()

        return position

    def get_position(self, symbol: str) -> Optional[Position]:
        """Récupère une position par symbole"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Vérifie si le portfolio a une position sur un symbole"""
        return symbol in self.positions

    def get_position_weight(self, symbol: str) -> Decimal:
        """Calcule le poids d'une position dans le portfolio"""
        if symbol not in self.positions or self.total_value == 0:
            return Decimal("0")

        return abs(self.positions[symbol].market_value) / self.total_value

    def get_largest_position_weight(self) -> Decimal:
        """Retourne le poids de la plus grosse position"""
        if not self.positions or self.total_value == 0:
            return Decimal("0")

        return max(
            abs(position.market_value) / self.total_value
            for position in self.positions.values()
        )

    # === Valorisation et cash ===

    def _update_portfolio_values(self) -> None:
        """Met à jour les valeurs calculées du portfolio"""
        positions_value = sum(position.market_value for position in self.positions.values())
        self.total_value = self.current_cash + positions_value

    def get_positions_value(self) -> Decimal:
        """Retourne la valeur totale des positions"""
        return sum(position.market_value for position in self.positions.values())

    def get_cash_percentage(self) -> Decimal:
        """Retourne le pourcentage de cash"""
        if self.total_value == 0:
            return Decimal("1")
        return self.current_cash / self.total_value

    def get_leverage(self) -> Decimal:
        """Calcule le levier du portfolio"""
        if self.total_value == 0:
            return Decimal("0")

        gross_exposure = sum(abs(position.market_value) for position in self.positions.values())
        return gross_exposure / self.total_value

    def adjust_cash(self, amount: Decimal, reason: str = "Manual adjustment") -> None:
        """
        Ajuste le montant de cash du portfolio.

        Args:
            amount: Montant à ajouter (positif) ou retirer (négatif)
            reason: Raison de l'ajustement
        """
        new_cash = self.current_cash + amount
        if new_cash < 0:
            raise ValueError(f"Insufficient cash: current={self.current_cash}, requested={amount}")

        self.current_cash = new_cash
        self._update_portfolio_values()
        self.updated_at = datetime.utcnow()

    # === Allocations et rééquilibrage ===

    def set_target_allocation(self, symbol: str, target_weight: Decimal) -> None:
        """
        Définit l'allocation cible pour un actif.

        Args:
            symbol: Symbole de l'actif
            target_weight: Poids cible (entre 0 et 1)
        """
        if not 0 <= target_weight <= 1:
            raise ValueError("Target weight must be between 0 and 1")

        self.target_allocations[symbol] = target_weight
        self.updated_at = datetime.utcnow()

    def get_allocation_drift(self) -> Dict[str, Decimal]:
        """
        Calcule l'écart entre allocations actuelles et cibles.

        Returns:
            Dictionnaire symbol -> écart (négatif = sous-pondéré, positif = sur-pondéré)
        """
        drifts = {}

        for symbol, target_weight in self.target_allocations.items():
            current_weight = self.get_position_weight(symbol)
            drifts[symbol] = current_weight - target_weight

        return drifts

    def needs_rebalancing(self, threshold: Decimal = Decimal("0.05")) -> bool:
        """
        Vérifie si le portfolio a besoin d'être rééquilibré.

        Args:
            threshold: Seuil d'écart pour déclencher le rééquilibrage

        Returns:
            True si rééquilibrage nécessaire
        """
        drifts = self.get_allocation_drift()
        return any(abs(drift) > threshold for drift in drifts.values())

    def calculate_rebalancing_trades(self) -> Dict[str, Decimal]:
        """
        Calcule les trades nécessaires pour rééquilibrer le portfolio.

        Returns:
            Dictionnaire symbol -> montant à trader (positif = acheter, négatif = vendre)
        """
        trades = {}
        drifts = self.get_allocation_drift()

        for symbol, drift in drifts.items():
            if abs(drift) > Decimal("0.01"):  # Seuil minimum de 1%
                trade_amount = -drift * self.total_value  # Inverse du drift
                trades[symbol] = trade_amount

        return trades

    # === Contraintes et validation ===

    def validate_constraints(self) -> List[str]:
        """
        Valide toutes les contraintes du portfolio.

        Returns:
            Liste des violations de contraintes
        """
        violations = []

        # Vérifier le nombre de positions
        if len(self.positions) > self.constraints.max_positions:
            violations.append(f"Too many positions: {len(self.positions)} > {self.constraints.max_positions}")

        # Vérifier les poids de positions
        for symbol, position in self.positions.items():
            weight = self.get_position_weight(symbol)
            if weight > self.constraints.max_position_weight:
                violations.append(f"Position {symbol} weight {weight:.3f} > {self.constraints.max_position_weight:.3f}")

        # Vérifier le levier
        leverage = self.get_leverage()
        if leverage > self.constraints.max_leverage:
            violations.append(f"Leverage {leverage:.2f} > {self.constraints.max_leverage:.2f}")

        # Vérifier le cash minimum
        cash_pct = self.get_cash_percentage()
        if cash_pct < self.constraints.min_cash_percentage:
            violations.append(f"Cash {cash_pct:.3f} < {self.constraints.min_cash_percentage:.3f}")

        return violations

    def is_valid(self) -> bool:
        """Vérifie si le portfolio respecte toutes ses contraintes"""
        return len(self.validate_constraints()) == 0

    # === Performance et snapshots ===

    def create_snapshot(self) -> PortfolioSnapshot:
        """
        Crée un snapshot instantané du portfolio.

        Returns:
            Snapshot du portfolio
        """
        snapshot = PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            total_value=self.total_value,
            cash=self.current_cash,
            positions_count=len(self.positions),
            largest_position_weight=self.get_largest_position_weight()
        )

        # Calculer le PnL journalier si on a un snapshot précédent
        if self.snapshots:
            previous = self.snapshots[-1]
            snapshot.daily_pnl = self.total_value - previous.total_value

        return snapshot

    def add_snapshot(self, snapshot: Optional[PortfolioSnapshot] = None) -> None:
        """
        Ajoute un snapshot à l'historique.

        Args:
            snapshot: Snapshot à ajouter (créé automatiquement si None)
        """
        if snapshot is None:
            snapshot = self.create_snapshot()

        self.snapshots.append(snapshot)

        # Limiter l'historique (garder 1000 snapshots max)
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-1000:]

    def calculate_return(self, days_back: int = 1) -> Optional[Decimal]:
        """
        Calcule le rendement sur une période.

        Args:
            days_back: Nombre de jours en arrière

        Returns:
            Rendement en pourcentage ou None si pas assez de données
        """
        if len(self.snapshots) < days_back + 1:
            return None

        current_value = self.total_value
        past_value = self.snapshots[-(days_back + 1)].total_value

        if past_value == 0:
            return None

        return (current_value - past_value) / past_value * 100

    def get_max_drawdown(self) -> Optional[Decimal]:
        """
        Calcule le drawdown maximum historique.

        Returns:
            Drawdown maximum en pourcentage
        """
        if len(self.snapshots) < 2:
            return None

        values = [snapshot.total_value for snapshot in self.snapshots]
        peak = values[0]
        max_drawdown = Decimal("0")

        for value in values[1:]:
            if value > peak:
                peak = value

            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown * 100  # En pourcentage

    # === Stratégies ===

    def add_strategy(self, strategy_id: str) -> None:
        """Ajoute une stratégie au portfolio"""
        self.strategy_ids.add(strategy_id)
        self.updated_at = datetime.utcnow()

    def remove_strategy(self, strategy_id: str) -> bool:
        """Retire une stratégie du portfolio"""
        if strategy_id in self.strategy_ids:
            self.strategy_ids.remove(strategy_id)
            self.updated_at = datetime.utcnow()
            return True
        return False

    def has_strategy(self, strategy_id: str) -> bool:
        """Vérifie si une stratégie est active"""
        return strategy_id in self.strategy_ids

    # === Sérialisation ===

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le portfolio en dictionnaire"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "portfolio_type": self.portfolio_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_rebalanced_at": self.last_rebalanced_at.isoformat() if self.last_rebalanced_at else None,
            "initial_capital": float(self.initial_capital),
            "current_cash": float(self.current_cash),
            "total_value": float(self.total_value),
            "positions": {symbol: position.to_dict() for symbol, position in self.positions.items()},
            "target_allocations": {symbol: float(weight) for symbol, weight in self.target_allocations.items()},
            "constraints": {
                "max_positions": self.constraints.max_positions,
                "max_position_weight": float(self.constraints.max_position_weight),
                "max_sector_weight": float(self.constraints.max_sector_weight),
                "max_leverage": float(self.constraints.max_leverage),
                "min_cash_percentage": float(self.constraints.min_cash_percentage),
                "max_drawdown": float(self.constraints.max_drawdown),
                "rebalancing_frequency": self.constraints.rebalancing_frequency.value
            },
            "performance_metrics": self.performance_metrics.to_dict() if self.performance_metrics else None,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots[-10:]],  # 10 derniers
            "current_risk_assessment_id": self.current_risk_assessment_id,
            "strategy_ids": list(self.strategy_ids),
            "benchmark_symbol": self.benchmark_symbol,
            "currency": self.currency,
            "statistics": {
                "positions_count": len(self.positions),
                "cash_percentage": float(self.get_cash_percentage()),
                "leverage": float(self.get_leverage()),
                "largest_position_weight": float(self.get_largest_position_weight()),
                "needs_rebalancing": self.needs_rebalancing(),
                "constraint_violations": self.validate_constraints(),
                "max_drawdown": float(self.get_max_drawdown()) if self.get_max_drawdown() else None
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Portfolio":
        """Crée un portfolio depuis un dictionnaire"""
        # Créer les contraintes
        constraints_data = data.get("constraints", {})
        constraints = PortfolioConstraints(
            max_positions=constraints_data.get("max_positions", 50),
            max_position_weight=Decimal(str(constraints_data.get("max_position_weight", "0.20"))),
            max_sector_weight=Decimal(str(constraints_data.get("max_sector_weight", "0.30"))),
            max_leverage=Decimal(str(constraints_data.get("max_leverage", "2.0"))),
            min_cash_percentage=Decimal(str(constraints_data.get("min_cash_percentage", "0.05"))),
            max_drawdown=Decimal(str(constraints_data.get("max_drawdown", "0.15"))),
            rebalancing_frequency=RebalancingFrequency(constraints_data.get("rebalancing_frequency", "weekly"))
        )

        # Créer le portfolio
        portfolio = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            portfolio_type=PortfolioType(data.get("portfolio_type", "trading")),
            status=PortfolioStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"].replace('Z', '+00:00')),
            initial_capital=Decimal(str(data["initial_capital"])),
            constraints=constraints,
            benchmark_symbol=data.get("benchmark_symbol", "SPY"),
            currency=data.get("currency", "USD")
        )

        # Mettre à jour les champs calculés
        portfolio.updated_at = datetime.fromisoformat(data["updated_at"].replace('Z', '+00:00'))
        portfolio.current_cash = Decimal(str(data["current_cash"]))
        portfolio.total_value = Decimal(str(data["total_value"]))

        # Recréer les positions
        for symbol, position_data in data.get("positions", {}).items():
            position = Position.from_dict(position_data)
            portfolio.positions[symbol] = position

        # Recréer les allocations cibles
        for symbol, weight in data.get("target_allocations", {}).items():
            portfolio.target_allocations[symbol] = Decimal(str(weight))

        # Ajouter les stratégies
        for strategy_id in data.get("strategy_ids", []):
            portfolio.strategy_ids.add(strategy_id)

        # Dates optionnelles
        if data.get("last_rebalanced_at"):
            portfolio.last_rebalanced_at = datetime.fromisoformat(data["last_rebalanced_at"].replace('Z', '+00:00'))

        return portfolio

    def __str__(self) -> str:
        return f"Portfolio(name={self.name}, value={self.total_value}, positions={len(self.positions)})"

    def __repr__(self) -> str:
        return f"Portfolio(id={self.id}, name={self.name})"


# Factory functions

def create_trading_portfolio(
    name: str,
    initial_capital: Decimal,
    max_positions: int = 20,
    max_position_weight: Decimal = Decimal("0.15")
) -> Portfolio:
    """Factory pour créer un portfolio de trading"""
    constraints = PortfolioConstraints(
        max_positions=max_positions,
        max_position_weight=max_position_weight,
        rebalancing_frequency=RebalancingFrequency.WEEKLY
    )

    return Portfolio(
        name=name,
        portfolio_type=PortfolioType.LIVE_TRADING,
        initial_capital=initial_capital,
        constraints=constraints
    )


def create_backtesting_portfolio(
    name: str,
    initial_capital: Decimal,
    benchmark: str = "SPY"
) -> Portfolio:
    """Factory pour créer un portfolio de backtesting"""
    constraints = PortfolioConstraints(
        max_positions=100,  # Plus de positions autorisées en backtest
        max_position_weight=Decimal("0.10"),
        rebalancing_frequency=RebalancingFrequency.DAILY
    )

    return Portfolio(
        name=name,
        portfolio_type=PortfolioType.BACKTESTING,
        initial_capital=initial_capital,
        constraints=constraints,
        benchmark_symbol=benchmark
    )


def create_paper_trading_portfolio(
    name: str,
    initial_capital: Decimal = Decimal("100000")
) -> Portfolio:
    """Factory pour créer un portfolio de paper trading"""
    constraints = PortfolioConstraints(
        max_positions=30,
        max_position_weight=Decimal("0.20"),
        rebalancing_frequency=RebalancingFrequency.MANUAL
    )

    return Portfolio(
        name=name,
        portfolio_type=PortfolioType.PAPER_TRADING,
        initial_capital=initial_capital,
        constraints=constraints
    )