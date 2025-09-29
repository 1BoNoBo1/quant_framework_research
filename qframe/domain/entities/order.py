"""
Domain Entity: Order
===================

Entité représentant un ordre de trading.
Encapsule le cycle de vie complet d'un ordre depuis sa création jusqu'à son exécution.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import uuid

from ..entities.position import Position


class OrderType(str, Enum):
    """Types d'ordres"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(str, Enum):
    """Côtés d'ordres"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Statuts d'ordres"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Durée de validité d'un ordre"""
    GTC = "gtc"  # Good Till Cancelled
    DAY = "day"  # Good for Day
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date


class OrderPriority(str, Enum):
    """Priorité d'exécution"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class OrderExecution:
    """Représente une exécution partielle ou totale d'un ordre"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_time: datetime = field(default_factory=datetime.utcnow)
    executed_quantity: Decimal = Decimal("0")
    execution_price: Decimal = Decimal("0")
    execution_value: Decimal = field(init=False)
    commission: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    venue: str = ""
    liquidity_flag: str = ""  # "maker", "taker", "unknown"

    def __post_init__(self):
        self.execution_value = self.executed_quantity * self.execution_price

    def get_total_cost(self) -> Decimal:
        """Retourne le coût total incluant commissions et frais"""
        return self.execution_value + self.commission + self.fees

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "execution_time": self.execution_time.isoformat(),
            "executed_quantity": float(self.executed_quantity),
            "execution_price": float(self.execution_price),
            "execution_value": float(self.execution_value),
            "commission": float(self.commission),
            "fees": float(self.fees),
            "venue": self.venue,
            "liquidity_flag": self.liquidity_flag,
            "total_cost": float(self.get_total_cost())
        }


@dataclass
class OrderReject:
    """Représente un rejet d'ordre"""
    reject_time: datetime = field(default_factory=datetime.utcnow)
    reject_reason: str = ""
    reject_code: str = ""
    reject_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reject_time": self.reject_time.isoformat(),
            "reject_reason": self.reject_reason,
            "reject_code": self.reject_code,
            "reject_text": self.reject_text
        }


@dataclass
class Order:
    """
    Entité Order représentant un ordre de trading.

    Encapsule toutes les informations et le cycle de vie d'un ordre,
    de sa création à son exécution complète.
    """

    # Identité et métadonnées
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = field(default_factory=lambda: f"CLT_{uuid.uuid4().hex[:8]}")
    broker_order_id: Optional[str] = None

    # Informations de base
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal("0")

    # Prix et conditions
    price: Optional[Decimal] = None  # Pour ordres LIMIT
    stop_price: Optional[Decimal] = None  # Pour ordres STOP
    trail_amount: Optional[Decimal] = None  # Pour TRAILING_STOP
    trail_percent: Optional[Decimal] = None  # Pour TRAILING_STOP

    # Timing et validité
    time_in_force: TimeInForce = TimeInForce.GTC
    good_till_date: Optional[datetime] = None
    expire_time: Optional[datetime] = None

    # Statut et exécution
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = field(init=False)
    average_fill_price: Decimal = Decimal("0")

    # Timestamps
    created_time: datetime = field(default_factory=datetime.utcnow)
    submitted_time: Optional[datetime] = None
    accepted_time: Optional[datetime] = None
    last_update_time: datetime = field(default_factory=datetime.utcnow)

    # Exécutions et historique
    executions: List[OrderExecution] = field(default_factory=list)
    reject_info: Optional[OrderReject] = None

    # Configuration avancée
    priority: OrderPriority = OrderPriority.NORMAL
    hidden_quantity: Optional[Decimal] = None  # Pour ordres ICEBERG
    display_quantity: Optional[Decimal] = None  # Pour ordres ICEBERG
    min_quantity: Optional[Decimal] = None  # Quantité minimum d'exécution

    # Métadonnées de routing
    destination: Optional[str] = None  # Venue de destination
    route_strategy: Optional[str] = None  # Stratégie de routing

    # Références business
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # Pour ordres enfants

    # Contraintes et flags
    reduce_only: bool = False  # Pour fermer des positions uniquement
    post_only: bool = False  # Market maker uniquement
    allow_partial_fills: bool = True

    # Métadonnées custom
    tags: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        """Initialisation et validation post-création"""
        self.remaining_quantity = self.quantity - self.filled_quantity
        self._validate_order()

    def _validate_order(self):
        """Valide les invariants de l'ordre"""
        if not self.symbol:
            raise ValueError("Order symbol cannot be empty")

        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit order must have a price")

        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop order must have a stop price")

        if self.time_in_force == TimeInForce.GTD and self.good_till_date is None:
            raise ValueError("GTD order must have a good till date")

        if self.filled_quantity > self.quantity:
            raise ValueError("Filled quantity cannot exceed order quantity")

    # === Gestion des exécutions ===

    def add_execution(self, execution: OrderExecution) -> None:
        """
        Ajoute une exécution à l'ordre.

        Args:
            execution: Exécution à ajouter

        Raises:
            ValueError: Si l'exécution dépasse la quantité restante
        """
        if execution.executed_quantity > self.remaining_quantity:
            raise ValueError("Execution quantity exceeds remaining quantity")

        # Ajouter l'exécution
        self.executions.append(execution)

        # Mettre à jour les quantités
        self.filled_quantity += execution.executed_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity

        # Mettre à jour le prix moyen d'exécution
        self._update_average_fill_price()

        # Mettre à jour le statut
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED

        self.last_update_time = datetime.utcnow()

    def _update_average_fill_price(self) -> None:
        """Met à jour le prix moyen d'exécution"""
        if not self.executions:
            self.average_fill_price = Decimal("0")
            return

        total_value = sum(exec.execution_value for exec in self.executions)
        total_quantity = sum(exec.executed_quantity for exec in self.executions)

        if total_quantity > 0:
            self.average_fill_price = total_value / total_quantity
        else:
            self.average_fill_price = Decimal("0")

    def get_total_executed_value(self) -> Decimal:
        """Retourne la valeur totale exécutée"""
        return sum(exec.execution_value for exec in self.executions)

    def get_total_commission(self) -> Decimal:
        """Retourne le total des commissions"""
        return sum(exec.commission for exec in self.executions)

    def get_total_fees(self) -> Decimal:
        """Retourne le total des frais"""
        return sum(exec.fees for exec in self.executions)

    def get_total_cost(self) -> Decimal:
        """Retourne le coût total (valeur + commissions + frais)"""
        return self.get_total_executed_value() + self.get_total_commission() + self.get_total_fees()

    # === Gestion du statut ===

    def submit(self, broker_order_id: Optional[str] = None) -> None:
        """Marque l'ordre comme soumis"""
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit order in status {self.status}")

        self.status = OrderStatus.SUBMITTED
        self.submitted_time = datetime.utcnow()
        self.last_update_time = self.submitted_time

        if broker_order_id:
            self.broker_order_id = broker_order_id

    def accept(self) -> None:
        """Marque l'ordre comme accepté par le broker"""
        if self.status != OrderStatus.SUBMITTED:
            raise ValueError(f"Cannot accept order in status {self.status}")

        self.status = OrderStatus.ACCEPTED
        self.accepted_time = datetime.utcnow()
        self.last_update_time = self.accepted_time

    def cancel(self, reason: str = "User requested") -> None:
        """Annule l'ordre"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order in status {self.status}")

        self.status = OrderStatus.CANCELLED
        self.last_update_time = datetime.utcnow()

        # Ajouter une note sur l'annulation
        if self.notes:
            self.notes += f"; Cancelled: {reason}"
        else:
            self.notes = f"Cancelled: {reason}"

    def reject(self, reason: str, code: str = "", text: str = "") -> None:
        """Rejette l'ordre"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot reject order in status {self.status}")

        self.status = OrderStatus.REJECTED
        self.reject_info = OrderReject(
            reject_reason=reason,
            reject_code=code,
            reject_text=text
        )
        self.last_update_time = datetime.utcnow()

    def expire(self) -> None:
        """Marque l'ordre comme expiré"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot expire order in status {self.status}")

        self.status = OrderStatus.EXPIRED
        self.last_update_time = datetime.utcnow()

    # === Vérifications d'état ===

    def is_active(self) -> bool:
        """Vérifie si l'ordre est actif (peut être exécuté)"""
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]

    def is_terminal(self) -> bool:
        """Vérifie si l'ordre est dans un état terminal"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]

    def is_filled(self) -> bool:
        """Vérifie si l'ordre est complètement exécuté"""
        return self.status == OrderStatus.FILLED

    def is_expired(self) -> bool:
        """Vérifie si l'ordre a expiré"""
        if self.status == OrderStatus.EXPIRED:
            return True

        now = datetime.utcnow()

        # Vérifier expiration par date
        if self.expire_time and now >= self.expire_time:
            return True

        # Vérifier Good Till Date
        if self.time_in_force == TimeInForce.GTD and self.good_till_date and now >= self.good_till_date:
            return True

        # Vérifier Day order (expire à la fin de la session)
        if self.time_in_force == TimeInForce.DAY and self.created_time.date() < now.date():
            return True

        return False

    def get_fill_percentage(self) -> Decimal:
        """Retourne le pourcentage d'exécution"""
        if self.quantity == 0:
            return Decimal("0")
        return (self.filled_quantity / self.quantity) * 100

    # === Propriétés calculées ===

    @property
    def notional_value(self) -> Decimal:
        """Valeur notionnelle de l'ordre"""
        price = self.price or self.average_fill_price or Decimal("0")
        return self.quantity * price

    @property
    def remaining_value(self) -> Decimal:
        """Valeur restante à exécuter"""
        price = self.price or self.average_fill_price or Decimal("0")
        return self.remaining_quantity * price

    @property
    def age_seconds(self) -> int:
        """Âge de l'ordre en secondes"""
        return int((datetime.utcnow() - self.created_time).total_seconds())

    @property
    def time_to_expiry(self) -> Optional[timedelta]:
        """Temps restant avant expiration"""
        if self.expire_time:
            remaining = self.expire_time - datetime.utcnow()
            return remaining if remaining.total_seconds() > 0 else timedelta(0)
        return None

    # === Modification d'ordre ===

    def modify_quantity(self, new_quantity: Decimal) -> None:
        """Modifie la quantité de l'ordre"""
        if not self.is_active():
            raise ValueError("Cannot modify inactive order")

        if new_quantity <= self.filled_quantity:
            raise ValueError("New quantity must be greater than filled quantity")

        self.quantity = new_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        self.last_update_time = datetime.utcnow()

    def modify_price(self, new_price: Decimal) -> None:
        """Modifie le prix de l'ordre"""
        if not self.is_active():
            raise ValueError("Cannot modify inactive order")

        if self.order_type not in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            raise ValueError("Cannot modify price for this order type")

        self.price = new_price
        self.last_update_time = datetime.utcnow()

    # === Métadonnées et tags ===

    def add_tag(self, key: str, value: str) -> None:
        """Ajoute un tag à l'ordre"""
        self.tags[key] = value
        self.last_update_time = datetime.utcnow()

    def remove_tag(self, key: str) -> bool:
        """Retire un tag de l'ordre"""
        if key in self.tags:
            del self.tags[key]
            self.last_update_time = datetime.utcnow()
            return True
        return False

    def has_tag(self, key: str) -> bool:
        """Vérifie si l'ordre a un tag donné"""
        return key in self.tags

    def get_tag(self, key: str, default: str = "") -> str:
        """Récupère la valeur d'un tag"""
        return self.tags.get(key, default)

    # === Sérialisation ===

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise l'ordre en dictionnaire"""
        return {
            "id": self.id,
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": float(self.quantity),
            "price": float(self.price) if self.price else None,
            "stop_price": float(self.stop_price) if self.stop_price else None,
            "trail_amount": float(self.trail_amount) if self.trail_amount else None,
            "trail_percent": float(self.trail_percent) if self.trail_percent else None,
            "time_in_force": self.time_in_force.value,
            "good_till_date": self.good_till_date.isoformat() if self.good_till_date else None,
            "expire_time": self.expire_time.isoformat() if self.expire_time else None,
            "status": self.status.value,
            "filled_quantity": float(self.filled_quantity),
            "remaining_quantity": float(self.remaining_quantity),
            "average_fill_price": float(self.average_fill_price),
            "created_time": self.created_time.isoformat(),
            "submitted_time": self.submitted_time.isoformat() if self.submitted_time else None,
            "accepted_time": self.accepted_time.isoformat() if self.accepted_time else None,
            "last_update_time": self.last_update_time.isoformat(),
            "executions": [exec.to_dict() for exec in self.executions],
            "reject_info": self.reject_info.to_dict() if self.reject_info else None,
            "priority": self.priority.value,
            "hidden_quantity": float(self.hidden_quantity) if self.hidden_quantity else None,
            "display_quantity": float(self.display_quantity) if self.display_quantity else None,
            "min_quantity": float(self.min_quantity) if self.min_quantity else None,
            "destination": self.destination,
            "route_strategy": self.route_strategy,
            "portfolio_id": self.portfolio_id,
            "strategy_id": self.strategy_id,
            "parent_order_id": self.parent_order_id,
            "reduce_only": self.reduce_only,
            "post_only": self.post_only,
            "allow_partial_fills": self.allow_partial_fills,
            "tags": self.tags,
            "notes": self.notes,
            "statistics": {
                "notional_value": float(self.notional_value),
                "remaining_value": float(self.remaining_value),
                "total_executed_value": float(self.get_total_executed_value()),
                "total_commission": float(self.get_total_commission()),
                "total_fees": float(self.get_total_fees()),
                "total_cost": float(self.get_total_cost()),
                "fill_percentage": float(self.get_fill_percentage()),
                "age_seconds": self.age_seconds,
                "is_active": self.is_active(),
                "is_terminal": self.is_terminal(),
                "is_expired": self.is_expired()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """Crée un ordre depuis un dictionnaire"""
        # Créer l'ordre de base
        order = cls(
            id=data.get("id", str(uuid.uuid4())),
            client_order_id=data.get("client_order_id", f"CLT_{uuid.uuid4().hex[:8]}"),
            broker_order_id=data.get("broker_order_id"),
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            quantity=Decimal(str(data["quantity"])),
            price=Decimal(str(data["price"])) if data.get("price") else None,
            stop_price=Decimal(str(data["stop_price"])) if data.get("stop_price") else None,
            trail_amount=Decimal(str(data["trail_amount"])) if data.get("trail_amount") else None,
            trail_percent=Decimal(str(data["trail_percent"])) if data.get("trail_percent") else None,
            time_in_force=TimeInForce(data.get("time_in_force", "gtc")),
            good_till_date=datetime.fromisoformat(data["good_till_date"].replace('Z', '+00:00')) if data.get("good_till_date") else None,
            expire_time=datetime.fromisoformat(data["expire_time"].replace('Z', '+00:00')) if data.get("expire_time") else None,
            status=OrderStatus(data.get("status", "pending")),
            filled_quantity=Decimal(str(data.get("filled_quantity", "0"))),
            created_time=datetime.fromisoformat(data["created_time"].replace('Z', '+00:00')),
            portfolio_id=data.get("portfolio_id"),
            strategy_id=data.get("strategy_id"),
            parent_order_id=data.get("parent_order_id"),
            reduce_only=data.get("reduce_only", False),
            post_only=data.get("post_only", False),
            allow_partial_fills=data.get("allow_partial_fills", True),
            tags=data.get("tags", {}),
            notes=data.get("notes", "")
        )

        # Mettre à jour les timestamps
        if data.get("submitted_time"):
            order.submitted_time = datetime.fromisoformat(data["submitted_time"].replace('Z', '+00:00'))
        if data.get("accepted_time"):
            order.accepted_time = datetime.fromisoformat(data["accepted_time"].replace('Z', '+00:00'))
        if data.get("last_update_time"):
            order.last_update_time = datetime.fromisoformat(data["last_update_time"].replace('Z', '+00:00'))

        # Reconstruire les exécutions
        for exec_data in data.get("executions", []):
            execution = OrderExecution(
                execution_id=exec_data["execution_id"],
                execution_time=datetime.fromisoformat(exec_data["execution_time"].replace('Z', '+00:00')),
                executed_quantity=Decimal(str(exec_data["executed_quantity"])),
                execution_price=Decimal(str(exec_data["execution_price"])),
                commission=Decimal(str(exec_data.get("commission", "0"))),
                fees=Decimal(str(exec_data.get("fees", "0"))),
                venue=exec_data.get("venue", ""),
                liquidity_flag=exec_data.get("liquidity_flag", "")
            )
            order.executions.append(execution)

        # Reconstruire les informations de rejet
        if data.get("reject_info"):
            reject_data = data["reject_info"]
            order.reject_info = OrderReject(
                reject_time=datetime.fromisoformat(reject_data["reject_time"].replace('Z', '+00:00')),
                reject_reason=reject_data.get("reject_reason", ""),
                reject_code=reject_data.get("reject_code", ""),
                reject_text=reject_data.get("reject_text", "")
            )

        # Recalculer les champs dérivés
        order.remaining_quantity = order.quantity - order.filled_quantity
        order._update_average_fill_price()

        return order

    def __str__(self) -> str:
        return f"Order({self.symbol} {self.side.value} {self.quantity} @ {self.price or 'MKT'} - {self.status.value})"

    def __repr__(self) -> str:
        return f"Order(id={self.id}, symbol={self.symbol}, side={self.side.value})"


# Factory functions

def create_market_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    portfolio_id: Optional[str] = None,
    strategy_id: Optional[str] = None
) -> Order:
    """Factory pour créer un ordre au marché"""
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        time_in_force=TimeInForce.IOC,  # Immédiat pour ordres marché
        portfolio_id=portfolio_id,
        strategy_id=strategy_id
    )


def create_limit_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    time_in_force: TimeInForce = TimeInForce.GTC,
    portfolio_id: Optional[str] = None,
    strategy_id: Optional[str] = None
) -> Order:
    """Factory pour créer un ordre limite"""
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        time_in_force=time_in_force,
        portfolio_id=portfolio_id,
        strategy_id=strategy_id
    )


def create_stop_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    stop_price: Decimal,
    portfolio_id: Optional[str] = None,
    strategy_id: Optional[str] = None
) -> Order:
    """Factory pour créer un ordre stop"""
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.STOP,
        quantity=quantity,
        stop_price=stop_price,
        portfolio_id=portfolio_id,
        strategy_id=strategy_id
    )


def create_stop_limit_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    stop_price: Decimal,
    limit_price: Decimal,
    portfolio_id: Optional[str] = None,
    strategy_id: Optional[str] = None
) -> Order:
    """Factory pour créer un ordre stop-limite"""
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.STOP_LIMIT,
        quantity=quantity,
        stop_price=stop_price,
        price=limit_price,
        portfolio_id=portfolio_id,
        strategy_id=strategy_id
    )