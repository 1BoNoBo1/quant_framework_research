"""
Value Object: Position
====================

Position représentant une position sur un instrument financier.
Value object immutable définissant l'état d'une position.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal


class PositionSide(str, Enum):
    """Côté de la position"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """Statut de la position"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


@dataclass(frozen=True)  # Immutable value object
class Position:
    """
    Value Object représentant une position de trading.

    Une position capture l'état d'un investissement sur un instrument
    à un moment donné, avec toutes ses métriques associées.
    """

    # Identité de la position
    symbol: str
    side: PositionSide

    # Quantités et prix
    size: Decimal  # Taille de la position (positive pour long, négative pour short)
    entry_price: Decimal  # Prix d'entrée moyen
    current_price: Decimal  # Prix actuel

    # Métadonnées temporelles
    entry_time: datetime
    last_update: datetime = None

    # Configuration de risque
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    # Métriques calculées
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Decimal = Decimal("0")

    # Métadonnées
    strategy_id: Optional[str] = None
    order_id: Optional[str] = None
    status: PositionStatus = PositionStatus.OPEN

    def __post_init__(self):
        """Validation des invariants et calculs automatiques"""
        self._validate_invariants()
        # Calcul automatique du PnL non réalisé si pas fourni
        if self.unrealized_pnl is None:
            object.__setattr__(self, 'unrealized_pnl', self._calculate_unrealized_pnl())

        # Mise à jour automatique du timestamp si pas fourni
        if self.last_update is None:
            object.__setattr__(self, 'last_update', datetime.utcnow())

    def _validate_invariants(self):
        """Valide les règles métier de la position"""
        if not self.symbol:
            raise ValueError("Position symbol cannot be empty")

        if self.size == 0:
            raise ValueError("Position size cannot be zero")

        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")

        if self.current_price <= 0:
            raise ValueError("Current price must be positive")

        # Validation cohérence side/size
        if self.side == PositionSide.LONG and self.size < 0:
            raise ValueError("Long position must have positive size")

        if self.side == PositionSide.SHORT and self.size > 0:
            raise ValueError("Short position must have negative size")

        # Validation stop loss et take profit
        if self.side == PositionSide.LONG:
            if self.stop_loss and self.stop_loss >= self.entry_price:
                raise ValueError("Stop loss must be below entry price for long position")
            if self.take_profit and self.take_profit <= self.entry_price:
                raise ValueError("Take profit must be above entry price for long position")

        elif self.side == PositionSide.SHORT:
            if self.stop_loss and self.stop_loss <= self.entry_price:
                raise ValueError("Stop loss must be above entry price for short position")
            if self.take_profit and self.take_profit >= self.entry_price:
                raise ValueError("Take profit must be below entry price for short position")

    def _calculate_unrealized_pnl(self) -> Decimal:
        """Calcule le PnL non réalisé"""
        if self.side == PositionSide.LONG:
            price_diff = self.current_price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - self.current_price

        return abs(self.size) * price_diff

    @property
    def pnl(self) -> Decimal:
        """PnL total (réalisé + non réalisé)"""
        return self.realized_pnl + (self.unrealized_pnl or Decimal("0"))

    @property
    def pnl_percentage(self) -> Decimal:
        """PnL en pourcentage"""
        if self.entry_price == 0:
            return Decimal("0")

        return (self.pnl / (abs(self.size) * self.entry_price)) * 100

    @property
    def market_value(self) -> Decimal:
        """Valeur de marché actuelle"""
        return abs(self.size) * self.current_price

    @property
    def entry_value(self) -> Decimal:
        """Valeur d'entrée"""
        return abs(self.size) * self.entry_price

    @property
    def is_profitable(self) -> bool:
        """Vérifie si la position est profitable"""
        return self.pnl > 0

    @property
    def should_stop_loss(self) -> bool:
        """Vérifie si le stop loss doit être déclenché"""
        if not self.stop_loss:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss

    @property
    def should_take_profit(self) -> bool:
        """Vérifie si le take profit doit être déclenché"""
        if not self.take_profit:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        else:  # SHORT
            return self.current_price <= self.take_profit

    @property
    def holding_period(self) -> int:
        """Durée de détention en secondes"""
        update_time = self.last_update or datetime.utcnow()
        return int((update_time - self.entry_time).total_seconds())

    @property
    def holding_days(self) -> Decimal:
        """Durée de détention en jours"""
        return Decimal(self.holding_period) / Decimal(86400)  # 24*60*60

    def update_price(self, new_price: Decimal) -> "Position":
        """Crée une nouvelle position avec prix mis à jour"""
        if new_price <= 0:
            raise ValueError("New price must be positive")

        new_unrealized_pnl = self._calculate_unrealized_pnl_with_price(new_price)

        return Position(
            symbol=self.symbol,
            side=self.side,
            size=self.size,
            entry_price=self.entry_price,
            current_price=new_price,
            entry_time=self.entry_time,
            last_update=datetime.utcnow(),
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            unrealized_pnl=new_unrealized_pnl,
            realized_pnl=self.realized_pnl,
            strategy_id=self.strategy_id,
            order_id=self.order_id,
            status=self.status
        )

    def _calculate_unrealized_pnl_with_price(self, price: Decimal) -> Decimal:
        """Calcule le PnL non réalisé avec un prix donné"""
        if self.side == PositionSide.LONG:
            price_diff = price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - price

        return abs(self.size) * price_diff

    def close_position(self, close_price: Decimal) -> "Position":
        """Crée une position fermée avec PnL réalisé"""
        if close_price <= 0:
            raise ValueError("Close price must be positive")

        final_pnl = self._calculate_unrealized_pnl_with_price(close_price)

        return Position(
            symbol=self.symbol,
            side=self.side,
            size=self.size,
            entry_price=self.entry_price,
            current_price=close_price,
            entry_time=self.entry_time,
            last_update=datetime.utcnow(),
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            unrealized_pnl=Decimal("0"),
            realized_pnl=self.realized_pnl + final_pnl,
            strategy_id=self.strategy_id,
            order_id=self.order_id,
            status=PositionStatus.CLOSED
        )

    def partial_close(self, close_size: Decimal, close_price: Decimal) -> tuple["Position", Decimal]:
        """Ferme partiellement la position et retourne (nouvelle_position, pnl_realisé)"""
        if close_size <= 0 or close_size >= abs(self.size):
            raise ValueError("Close size must be positive and less than position size")

        if close_price <= 0:
            raise ValueError("Close price must be positive")

        # Calcul du PnL réalisé sur la partie fermée
        if self.side == PositionSide.LONG:
            pnl_per_unit = close_price - self.entry_price
        else:  # SHORT
            pnl_per_unit = self.entry_price - close_price

        realized_pnl_amount = close_size * pnl_per_unit

        # Nouvelle taille de position
        new_size = self.size + close_size if self.side == PositionSide.SHORT else self.size - close_size

        # Nouvelle position
        new_position = Position(
            symbol=self.symbol,
            side=self.side,
            size=new_size,
            entry_price=self.entry_price,
            current_price=close_price,
            entry_time=self.entry_time,
            last_update=datetime.utcnow(),
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            unrealized_pnl=self._calculate_unrealized_pnl_with_price(close_price),
            realized_pnl=self.realized_pnl + realized_pnl_amount,
            strategy_id=self.strategy_id,
            order_id=self.order_id,
            status=PositionStatus.PARTIALLY_CLOSED
        )

        return new_position, realized_pnl_amount

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise la position en dictionnaire"""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "size": float(self.size),
            "entry_price": float(self.entry_price),
            "current_price": float(self.current_price),
            "entry_time": self.entry_time.isoformat(),
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "stop_loss": float(self.stop_loss) if self.stop_loss else None,
            "take_profit": float(self.take_profit) if self.take_profit else None,
            "unrealized_pnl": float(self.unrealized_pnl) if self.unrealized_pnl else None,
            "realized_pnl": float(self.realized_pnl),
            "total_pnl": float(self.pnl),
            "pnl_percentage": float(self.pnl_percentage),
            "market_value": float(self.market_value),
            "entry_value": float(self.entry_value),
            "is_profitable": self.is_profitable,
            "holding_days": float(self.holding_days),
            "strategy_id": self.strategy_id,
            "order_id": self.order_id,
            "status": self.status.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Crée une position depuis un dictionnaire"""
        return cls(
            symbol=data["symbol"],
            side=PositionSide(data["side"]),
            size=Decimal(str(data["size"])),
            entry_price=Decimal(str(data["entry_price"])),
            current_price=Decimal(str(data["current_price"])),
            entry_time=datetime.fromisoformat(data["entry_time"].replace('Z', '+00:00')),
            last_update=datetime.fromisoformat(data["last_update"].replace('Z', '+00:00')) if data.get("last_update") else None,
            stop_loss=Decimal(str(data["stop_loss"])) if data.get("stop_loss") else None,
            take_profit=Decimal(str(data["take_profit"])) if data.get("take_profit") else None,
            unrealized_pnl=Decimal(str(data["unrealized_pnl"])) if data.get("unrealized_pnl") is not None else None,
            realized_pnl=Decimal(str(data.get("realized_pnl", "0"))),
            strategy_id=data.get("strategy_id"),
            order_id=data.get("order_id"),
            status=PositionStatus(data.get("status", "open"))
        )

    def __str__(self) -> str:
        return f"Position({self.side.value} {abs(self.size)} {self.symbol} @ {self.entry_price}, PnL: {self.pnl})"

    def __repr__(self) -> str:
        return f"Position(symbol={self.symbol}, side={self.side.value}, size={self.size})"


# Factory functions
def create_long_position(
    symbol: str,
    size: Decimal,
    entry_price: Decimal,
    current_price: Optional[Decimal] = None,
    stop_loss: Optional[Decimal] = None,
    take_profit: Optional[Decimal] = None,
    strategy_id: Optional[str] = None
) -> Position:
    """Factory pour créer une position longue"""
    return Position(
        symbol=symbol,
        side=PositionSide.LONG,
        size=abs(size),  # Force positive pour long
        entry_price=entry_price,
        current_price=current_price or entry_price,
        entry_time=datetime.utcnow(),
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_id=strategy_id
    )


def create_short_position(
    symbol: str,
    size: Decimal,
    entry_price: Decimal,
    current_price: Optional[Decimal] = None,
    stop_loss: Optional[Decimal] = None,
    take_profit: Optional[Decimal] = None,
    strategy_id: Optional[str] = None
) -> Position:
    """Factory pour créer une position courte"""
    return Position(
        symbol=symbol,
        side=PositionSide.SHORT,
        size=-abs(size),  # Force négative pour short
        entry_price=entry_price,
        current_price=current_price or entry_price,
        entry_time=datetime.utcnow(),
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_id=strategy_id
    )