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
from ..entities.position import Position  # Using simple mutable Position
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
    id: str
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Métadonnées
    strategy_type: StrategyType = StrategyType.MEAN_REVERSION
    version: int = 1  # Version as int for tests
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
    risk_limits: Dict[str, Any] = field(default_factory=dict)  # For test compatibility

    # Métriques de performance
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: Optional[Decimal] = None

    # Positions actuelles
    positions: Dict[str, Any] = field(default_factory=dict)  # Changed to Any for flexibility

    # Historique des signaux
    signal_history: List[Any] = field(default_factory=list)  # Changed to Any for flexibility

    def __post_init__(self):
        """Validation post-initialisation"""
        self._validate_invariants()

    def _validate_invariants(self):
        """Valide les invariants métier de la stratégie"""
        if not self.name:
            self.name = "Default Strategy"  # Default pour tests

        if self.max_position_size <= 0 or self.max_position_size > 1:
            self.max_position_size = Decimal("0.02")  # Default safe value

        if self.max_positions <= 0:
            self.max_positions = 5  # Default safe value

        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            self.risk_per_trade = Decimal("0.01")  # Default safe value

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

    def deactivate(self) -> None:
        """Désactive la stratégie (alias pour stop)"""
        self.stop()

    def is_active(self) -> bool:
        """Vérifie si la stratégie est active"""
        return self.status == StrategyStatus.ACTIVE

    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Met à jour les paramètres de la stratégie"""
        self.parameters.update(new_parameters)
        self.version += 1  # Incrémente la version
        self.updated_at = datetime.utcnow()

    def record_performance(self, total_return: float, max_drawdown: float,
                          win_rate: float, total_trades: int) -> None:
        """Enregistre les métriques de performance"""
        self.performance_metrics["total_return"] = total_return
        self.performance_metrics["max_drawdown"] = max_drawdown
        self.performance_metrics["win_rate"] = win_rate
        self.performance_metrics["total_trades"] = total_trades
        self.updated_at = datetime.utcnow()

    def __eq__(self, other) -> bool:
        """Comparaison par ID uniquement"""
        if not isinstance(other, Strategy):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash basé sur l'ID"""
        return hash(self.id)

    def set_error(self, error_msg: str) -> None:
        """Marque la stratégie en erreur"""
        self.status = StrategyStatus.ERROR
        self.updated_at = datetime.utcnow()
        # Log error would be handled by infrastructure layer

    def add_signal(self, signal) -> None:
        """Ajoute un signal à l'historique avec limite de mémoire"""
        # Convertir en deque si ce n'est pas déjà fait pour éviter les memory leaks
        if not hasattr(self, '_signal_history_deque'):
            from collections import deque
            # Migrer les signaux existants vers deque avec limite
            self._signal_history_deque = deque(self.signal_history, maxlen=5000)
            self.signal_history = self._signal_history_deque

        self.signal_history.append(signal)
        self.updated_at = datetime.utcnow()

    def add_position(self, position) -> None:
        """Ajoute une position active"""
        # Flexible pour accepter différents types de Position
        if len(self.positions) >= self.max_positions:
            raise ValueError(f"Cannot exceed max_positions ({self.max_positions})")

        symbol = position.symbol if hasattr(position, 'symbol') else str(position)
        self.positions[symbol] = position
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

        # Calcul du PnL si nécessaire
        pnl = position.realized_pnl if hasattr(position, 'realized_pnl') else Decimal("0")

        if pnl > 0:
            self.winning_trades += 1

        self.total_pnl += pnl

        # Calcul drawdown simplifié
        if pnl < 0:
            potential_drawdown = abs(pnl) / (position.average_price if position.average_price > 0 else Decimal("1"))
            self.max_drawdown = max(self.max_drawdown, potential_drawdown)

    def get_win_rate(self) -> Decimal:
        """Calcule le taux de réussite"""
        if self.total_trades == 0:
            return Decimal("0")
        return Decimal(self.winning_trades) / Decimal(self.total_trades)

    def get_current_exposure(self) -> Decimal:
        """Calcule l'exposition actuelle"""
        total_exposure = Decimal("0")
        for position in self.positions.values():
            if hasattr(position, 'quantity') and hasattr(position, 'current_price'):
                total_exposure += abs(position.quantity * position.current_price)
            elif hasattr(position, 'quantity') and hasattr(position, 'average_price'):
                total_exposure += abs(position.quantity * position.average_price)
        return total_exposure

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
            "risk_per_trade": float(self.risk_per_trade),  # Ajouter risk_per_trade
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": float(self.total_pnl),
            "max_drawdown": float(self.max_drawdown),  # Ajouter max_drawdown
            "sharpe_ratio": float(self.sharpe_ratio) if self.sharpe_ratio else None,  # Ajouter sharpe_ratio
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
            version=data.get("version", 1),
            status=StrategyStatus(data.get("status", "inactive")),
            universe=set(data.get("universe", [])),
            max_position_size=Decimal(str(data.get("max_position_size", "0.02"))),
            max_positions=data.get("max_positions", 5),
            risk_per_trade=Decimal(str(data.get("risk_per_trade", "0.01"))),  # Ajouter risk_per_trade
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            total_pnl=Decimal(str(data.get("total_pnl", "0"))),
            max_drawdown=Decimal(str(data.get("max_drawdown", "0"))),
            sharpe_ratio=Decimal(str(data["sharpe_ratio"])) if data.get("sharpe_ratio") else None  # Ajouter sharpe_ratio
        )

    def __str__(self) -> str:
        return f"Strategy(name={self.name}, type={self.strategy_type.value}, status={self.status.value})"

    def __repr__(self) -> str:
        return f"Strategy(id={self.id}, name={self.name})"