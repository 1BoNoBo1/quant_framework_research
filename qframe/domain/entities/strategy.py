"""
Domain Entity: Strategy
======================

Entité centrale représentant une stratégie de trading dans l'architecture hexagonale.
Cette entité contient la logique métier pure, indépendante de l'infrastructure.
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import uuid

from ..value_objects.signal import Signal
from ..value_objects.position import Position
from ..value_objects.performance_metrics import PerformanceMetrics


class StrategyType(str, Enum):
    """Types de stratégies supportées"""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


class StrategyStatus(str, Enum):
    """Statuts possibles d'une stratégie"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    BACKTESTING = "backtesting"


@dataclass
class Strategy:
    """
    Entité Strategy représentant une stratégie de trading.

    Cette entité encapsule la logique métier pure d'une stratégie,
    indépendamment de son implémentation technique.
    """

    # Identité
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Métadonnées
    strategy_type: StrategyType = StrategyType.MEAN_REVERSION
    version: str = "1.0.0"
    author: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # État opérationnel
    status: StrategyStatus = StrategyStatus.INACTIVE
    universe: Set[str] = field(default_factory=set)  # Instruments tradés

    # Configuration de risque
    max_position_size: Decimal = Decimal("0.02")
    max_positions: int = 5
    risk_per_trade: Decimal = Decimal("0.01")

    # Métriques de performance
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: Optional[Decimal] = None

    # Positions actuelles
    positions: Dict[str, Position] = field(default_factory=dict)

    # Historique des signaux
    signal_history: List[Signal] = field(default_factory=list)

    def __post_init__(self):
        """Validation post-initialisation"""
        self._validate_invariants()

    def _validate_invariants(self):
        """Valide les invariants métier de la stratégie"""
        if not self.name:
            raise ValueError("Strategy name cannot be empty")

        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1")

        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")

        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            raise ValueError("risk_per_trade must be between 0 and 0.1")

    def activate(self) -> None:
        """Active la stratégie"""
        if self.status == StrategyStatus.ERROR:
            raise ValueError("Cannot activate strategy in error state")

        self.status = StrategyStatus.ACTIVE
        self.updated_at = datetime.utcnow()

    def pause(self) -> None:
        """Met en pause la stratégie"""
        if self.status == StrategyStatus.ACTIVE:
            self.status = StrategyStatus.PAUSED
            self.updated_at = datetime.utcnow()

    def stop(self) -> None:
        """Arrête la stratégie"""
        self.status = StrategyStatus.INACTIVE
        self.updated_at = datetime.utcnow()

    def set_error(self, error_msg: str) -> None:
        """Marque la stratégie en erreur"""
        self.status = StrategyStatus.ERROR
        self.updated_at = datetime.utcnow()
        # Log error would be handled by infrastructure layer

    def add_signal(self, signal: Signal) -> None:
        """Ajoute un signal à l'historique"""
        if not isinstance(signal, Signal):
            raise TypeError("signal must be a Signal instance")

        self.signal_history.append(signal)

        # Limite la taille de l'historique (business rule)
        if len(self.signal_history) > 10000:
            self.signal_history = self.signal_history[-5000:]

    def add_position(self, position: Position) -> None:
        """Ajoute une position active"""
        if not isinstance(position, Position):
            raise TypeError("position must be a Position instance")

        if len(self.positions) >= self.max_positions:
            raise ValueError(f"Cannot exceed max_positions ({self.max_positions})")

        self.positions[position.symbol] = position
        self.updated_at = datetime.utcnow()

    def close_position(self, symbol: str) -> Optional[Position]:
        """Ferme une position et la retourne"""
        position = self.positions.pop(symbol, None)
        if position:
            self.updated_at = datetime.utcnow()
            # Mise à jour des métriques
            self._update_trade_metrics(position)
        return position

    def _update_trade_metrics(self, position: Position) -> None:
        """Met à jour les métriques après fermeture d'une position"""
        self.total_trades += 1

        if position.pnl > 0:
            self.winning_trades += 1

        self.total_pnl += position.pnl

        # Calcul drawdown simplifié
        if position.pnl < 0:
            potential_drawdown = abs(position.pnl) / position.entry_price
            self.max_drawdown = max(self.max_drawdown, potential_drawdown)

    def get_win_rate(self) -> Decimal:
        """Calcule le taux de réussite"""
        if self.total_trades == 0:
            return Decimal("0")
        return Decimal(self.winning_trades) / Decimal(self.total_trades)

    def get_current_exposure(self) -> Decimal:
        """Calcule l'exposition actuelle"""
        total_exposure = sum(
            abs(position.size * position.current_price)
            for position in self.positions.values()
        )
        return Decimal(str(total_exposure))

    def can_add_position(self, position_value: Decimal) -> bool:
        """Vérifie si une nouvelle position peut être ajoutée"""
        # Vérification nombre max de positions
        if len(self.positions) >= self.max_positions:
            return False

        # Vérification taille max de position
        if position_value > self.max_position_size:
            return False

        # Vérification exposition totale
        current_exposure = self.get_current_exposure()
        max_total_exposure = self.max_position_size * self.max_positions

        if current_exposure + position_value > max_total_exposure:
            return False

        return True

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Retourne les métriques de performance actuelles"""
        return PerformanceMetrics(
            total_return=self.total_pnl,
            win_rate=self.get_win_rate(),
            total_trades=self.total_trades,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=self.sharpe_ratio or Decimal("0"),
            current_exposure=self.get_current_exposure()
        )

    def is_active(self) -> bool:
        """Vérifie si la stratégie est active"""
        return self.status == StrategyStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise la stratégie en dictionnaire"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type.value,
            "version": self.version,
            "status": self.status.value,
            "universe": list(self.universe),
            "max_position_size": float(self.max_position_size),
            "max_positions": self.max_positions,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": float(self.total_pnl),
            "win_rate": float(self.get_win_rate()),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Strategy":
        """Crée une stratégie depuis un dictionnaire"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            strategy_type=StrategyType(data.get("strategy_type", "mean_reversion")),
            version=data.get("version", "1.0.0"),
            status=StrategyStatus(data.get("status", "inactive")),
            universe=set(data.get("universe", [])),
            max_position_size=Decimal(str(data.get("max_position_size", "0.02"))),
            max_positions=data.get("max_positions", 5),
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            total_pnl=Decimal(str(data.get("total_pnl", "0"))),
            max_drawdown=Decimal(str(data.get("max_drawdown", "0")))
        )

    def __str__(self) -> str:
        return f"Strategy(name={self.name}, type={self.strategy_type.value}, status={self.status.value})"

    def __repr__(self) -> str:
        return f"Strategy(id={self.id}, name={self.name})"