"""
Simple Position class for Portfolio
====================================

Position mutable pour utilisation dans Portfolio.
Séparée du value object immutable.
"""

from decimal import Decimal
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Position:
    """Position mutable pour Portfolio"""

    symbol: str
    quantity: Decimal
    average_price: Decimal

    # Optionnel
    current_price: Optional[Decimal] = None
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Optional[Decimal] = None

    @property
    def market_value(self) -> Decimal:
        """Calcule la valeur de marché de la position"""
        if self.current_price:
            return self.quantity * self.current_price
        return self.quantity * self.average_price

    def to_dict(self) -> Dict[str, Any]:
        """Convertit la position en dictionnaire"""
        return {
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "average_price": float(self.average_price),
            "current_price": float(self.current_price) if self.current_price else None,
            "realized_pnl": float(self.realized_pnl),
            "market_value": float(self.market_value)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Crée une position depuis un dictionnaire"""
        return cls(
            symbol=data["symbol"],
            quantity=Decimal(str(data["quantity"])),
            average_price=Decimal(str(data["average_price"])),
            current_price=Decimal(str(data["current_price"])) if data.get("current_price") else None,
            realized_pnl=Decimal(str(data.get("realized_pnl", "0")))
        )