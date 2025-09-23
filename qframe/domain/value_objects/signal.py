"""
Value Object: Signal
==================

Signal de trading représentant une recommandation d'achat/vente.
Les value objects sont immutables et définis par leurs valeurs.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal


class SignalAction(str, Enum):
    """Actions possibles pour un signal"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class SignalConfidence(str, Enum):
    """Niveaux de confiance du signal"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass(frozen=True)  # Immutable value object
class Signal:
    """
    Value Object représentant un signal de trading.

    Un signal est immutable et représente une recommandation
    à un moment donné pour un instrument spécifique.
    """

    # Identité du signal
    symbol: str
    action: SignalAction
    timestamp: datetime

    # Force et confiance du signal
    strength: Decimal  # Entre 0 et 1
    confidence: SignalConfidence

    # Détails du signal
    price: Decimal  # Prix au moment du signal
    quantity: Optional[Decimal] = None  # Quantité suggérée
    strategy_id: Optional[str] = None  # ID de la stratégie qui a généré le signal

    # Métadonnées optionnelles
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    expiry: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validation des invariants du value object"""
        self._validate_invariants()

    def _validate_invariants(self):
        """Valide les règles métier du signal"""
        if not self.symbol:
            raise ValueError("Signal symbol cannot be empty")

        if self.strength < 0 or self.strength > 1:
            raise ValueError("Signal strength must be between 0 and 1")

        if self.price <= 0:
            raise ValueError("Signal price must be positive")

        if self.quantity is not None and self.quantity <= 0:
            raise ValueError("Signal quantity must be positive if specified")

        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("Stop loss must be positive if specified")

        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("Take profit must be positive if specified")

        # Validation cohérence prix/stop/take profit
        if self.action == SignalAction.BUY:
            if self.stop_loss and self.stop_loss >= self.price:
                raise ValueError("Stop loss must be below entry price for BUY signal")
            if self.take_profit and self.take_profit <= self.price:
                raise ValueError("Take profit must be above entry price for BUY signal")

        elif self.action == SignalAction.SELL:
            if self.stop_loss and self.stop_loss <= self.price:
                raise ValueError("Stop loss must be above entry price for SELL signal")
            if self.take_profit and self.take_profit >= self.price:
                raise ValueError("Take profit must be below entry price for SELL signal")

    @property
    def is_valid(self) -> bool:
        """Vérifie si le signal est encore valide"""
        if self.expiry:
            return datetime.utcnow() < self.expiry
        return True

    @property
    def is_actionable(self) -> bool:
        """Vérifie si le signal peut être exécuté"""
        return (
            self.is_valid and
            self.action in [SignalAction.BUY, SignalAction.SELL] and
            self.strength >= Decimal("0.3")  # Seuil minimum de force
        )

    @property
    def risk_reward_ratio(self) -> Optional[Decimal]:
        """Calcule le ratio risque/récompense"""
        if not (self.stop_loss and self.take_profit):
            return None

        if self.action == SignalAction.BUY:
            risk = abs(self.price - self.stop_loss)
            reward = abs(self.take_profit - self.price)
        else:  # SELL
            risk = abs(self.stop_loss - self.price)
            reward = abs(self.price - self.take_profit)

        if risk == 0:
            return None

        return reward / risk

    def get_signal_score(self) -> Decimal:
        """Calcule un score global du signal (0-100)"""
        base_score = self.strength * 100

        # Bonus pour confidence élevée
        confidence_bonus = {
            SignalConfidence.LOW: 0,
            SignalConfidence.MEDIUM: 5,
            SignalConfidence.HIGH: 10,
            SignalConfidence.VERY_HIGH: 15
        }
        base_score += confidence_bonus.get(self.confidence, 0)

        # Bonus pour bon ratio risque/récompense
        rr_ratio = self.risk_reward_ratio
        if rr_ratio:
            if rr_ratio >= 2:
                base_score += 10
            elif rr_ratio >= 1.5:
                base_score += 5

        return min(Decimal("100"), base_score)

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le signal en dictionnaire"""
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
            "strength": float(self.strength),
            "confidence": self.confidence.value,
            "price": float(self.price),
            "quantity": float(self.quantity) if self.quantity else None,
            "strategy_id": self.strategy_id,
            "stop_loss": float(self.stop_loss) if self.stop_loss else None,
            "take_profit": float(self.take_profit) if self.take_profit else None,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "metadata": self.metadata,
            "signal_score": float(self.get_signal_score()),
            "risk_reward_ratio": float(self.risk_reward_ratio) if self.risk_reward_ratio else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Crée un signal depuis un dictionnaire"""
        return cls(
            symbol=data["symbol"],
            action=SignalAction(data["action"]),
            timestamp=datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00')),
            strength=Decimal(str(data["strength"])),
            confidence=SignalConfidence(data["confidence"]),
            price=Decimal(str(data["price"])),
            quantity=Decimal(str(data["quantity"])) if data.get("quantity") else None,
            strategy_id=data.get("strategy_id"),
            stop_loss=Decimal(str(data["stop_loss"])) if data.get("stop_loss") else None,
            take_profit=Decimal(str(data["take_profit"])) if data.get("take_profit") else None,
            expiry=datetime.fromisoformat(data["expiry"].replace('Z', '+00:00')) if data.get("expiry") else None,
            metadata=data.get("metadata")
        )

    def __str__(self) -> str:
        return f"Signal({self.action.value} {self.symbol} @ {self.price}, strength={self.strength})"

    def __repr__(self) -> str:
        return f"Signal(symbol={self.symbol}, action={self.action.value}, price={self.price})"


# Factory functions pour création de signaux communs
def create_buy_signal(
    symbol: str,
    price: Decimal,
    strength: Decimal,
    confidence: SignalConfidence = SignalConfidence.MEDIUM,
    stop_loss: Optional[Decimal] = None,
    take_profit: Optional[Decimal] = None,
    strategy_id: Optional[str] = None
) -> Signal:
    """Factory pour créer un signal d'achat"""
    return Signal(
        symbol=symbol,
        action=SignalAction.BUY,
        timestamp=datetime.utcnow(),
        strength=strength,
        confidence=confidence,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_id=strategy_id
    )


def create_sell_signal(
    symbol: str,
    price: Decimal,
    strength: Decimal,
    confidence: SignalConfidence = SignalConfidence.MEDIUM,
    stop_loss: Optional[Decimal] = None,
    take_profit: Optional[Decimal] = None,
    strategy_id: Optional[str] = None
) -> Signal:
    """Factory pour créer un signal de vente"""
    return Signal(
        symbol=symbol,
        action=SignalAction.SELL,
        timestamp=datetime.utcnow(),
        strength=strength,
        confidence=confidence,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_id=strategy_id
    )


def create_close_signal(
    symbol: str,
    price: Decimal,
    strategy_id: Optional[str] = None
) -> Signal:
    """Factory pour créer un signal de fermeture"""
    return Signal(
        symbol=symbol,
        action=SignalAction.CLOSE,
        timestamp=datetime.utcnow(),
        strength=Decimal("1.0"),  # Force max pour fermeture
        confidence=SignalConfidence.HIGH,
        price=price,
        strategy_id=strategy_id
    )