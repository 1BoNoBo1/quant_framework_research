"""
Enhanced Portfolio Entity with Pydantic Validation
=================================================

Portfolio entity avec validation Pydantic avancée, métriques temps réel,
et sérialisation optimisée pour performance.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
import uuid

from pydantic import (
    BaseModel, Field, field_validator, model_validator,
    computed_field, ConfigDict
)
from pydantic.types import PositiveFloat, NonNegativeFloat

class PortfolioStatus(str, Enum):
    """Statuts du portfolio."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    LIQUIDATING = "liquidating"
    CLOSED = "closed"
    MARGIN_CALL = "margin_call"

class RiskLevel(str, Enum):
    """Niveaux de risque."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"

class CurrencyCode(str, Enum):
    """Codes de devises supportées."""
    USD = "USD"
    EUR = "EUR"
    BTC = "BTC"
    ETH = "ETH"
    USDT = "USDT"

class Position(BaseModel):
    """Position validée avec Pydantic."""
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    symbol: str = Field(..., min_length=3, max_length=20, description="Symbole trading")
    size: Decimal = Field(..., description="Taille position (+ long, - short)")
    entry_price: PositiveFloat = Field(..., description="Prix d'entrée")
    current_price: PositiveFloat = Field(..., description="Prix actuel")
    entry_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Métriques calculées
    unrealized_pnl: Optional[Decimal] = None
    unrealized_pnl_percent: Optional[float] = None

    # Métadonnées
    strategy_name: Optional[str] = Field(None, max_length=100)
    position_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tags: Dict[str, str] = Field(default_factory=dict)

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Valide le format du symbole."""
        if not v.replace('/', '').replace('-', '').isalnum():
            raise ValueError("Symbole doit contenir uniquement alphanumériques, / et -")
        return v.upper()

    @field_validator('size')
    @classmethod
    def validate_size(cls, v: Decimal) -> Decimal:
        """Valide que la taille n'est pas zéro."""
        if v == 0:
            raise ValueError("Taille de position ne peut pas être zéro")
        return v

    @computed_field
    @property
    def market_value(self) -> Decimal:
        """Valeur marché de la position."""
        return abs(self.size) * Decimal(str(self.current_price))

    @computed_field
    @property
    def is_long(self) -> bool:
        """Position longue?"""
        return self.size > 0

    @computed_field
    @property
    def pnl_percentage(self) -> float:
        """PnL en pourcentage."""
        if self.unrealized_pnl is None:
            return 0.0
        return float((self.unrealized_pnl / self.market_value) * 100)

    def update_current_price(self, new_price: float) -> 'Position':
        """Met à jour le prix avec nouveau PnL."""
        size_decimal = self.size
        entry_decimal = Decimal(str(self.entry_price))
        current_decimal = Decimal(str(new_price))

        # Calcul PnL
        if self.is_long:
            pnl = size_decimal * (current_decimal - entry_decimal)
        else:
            pnl = abs(size_decimal) * (entry_decimal - current_decimal)

        return self.model_copy(update={
            'current_price': new_price,
            'unrealized_pnl': pnl,
            'unrealized_pnl_percent': float((pnl / self.market_value) * 100)
        })

class RiskMetrics(BaseModel):
    """Métriques de risque du portfolio."""
    model_config = ConfigDict(validate_assignment=True)

    # Métriques de base
    total_exposure: NonNegativeFloat = 0.0
    leverage: NonNegativeFloat = 1.0
    var_1d: Optional[float] = Field(None, description="VaR 1 jour")
    var_1w: Optional[float] = Field(None, description="VaR 1 semaine")

    # Concentration
    max_position_weight: NonNegativeFloat = 0.0
    concentration_index: NonNegativeFloat = 0.0

    # Corrélations
    avg_correlation: Optional[float] = Field(None, ge=-1, le=1)
    correlation_risk: Optional[float] = None

    # Limites
    max_drawdown: NonNegativeFloat = 0.0
    risk_level: RiskLevel = RiskLevel.MODERATE

    # Timestamps
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('leverage')
    @classmethod
    def validate_leverage(cls, v: float) -> float:
        """Valide le levier."""
        if v > 10:
            raise ValueError("Levier ne peut pas dépasser 10x")
        return v

    @computed_field
    @property
    def risk_score(self) -> float:
        """Score de risque composite 0-100."""
        score = 0.0

        # Levier (30%)
        score += min(self.leverage / 5.0 * 30, 30)

        # Exposition (25%)
        score += min(self.total_exposure / 1000000 * 25, 25)

        # Concentration (25%)
        score += self.max_position_weight * 25

        # Drawdown (20%)
        score += min(self.max_drawdown * 100 * 20, 20)

        return min(score, 100.0)

class PerformanceMetrics(BaseModel):
    """Métriques de performance du portfolio."""
    model_config = ConfigDict(validate_assignment=True)

    # Returns
    total_return: float = 0.0
    daily_return: float = 0.0
    annualized_return: float = 0.0

    # Risque-ajusté
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Drawdown
    current_drawdown: NonNegativeFloat = 0.0
    max_drawdown: NonNegativeFloat = 0.0
    max_drawdown_duration: Optional[int] = Field(None, description="Durée max DD en jours")

    # Trading
    win_rate: Optional[NonNegativeFloat] = Field(None, ge=0, le=1)
    profit_factor: Optional[PositiveFloat] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None

    # Périodes
    period_start: datetime
    period_end: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('win_rate')
    @classmethod
    def validate_win_rate(cls, v: Optional[float]) -> Optional[float]:
        """Valide le taux de gain."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Win rate doit être entre 0 et 1")
        return v

    @computed_field
    @property
    def risk_adjusted_return(self) -> float:
        """Rendement ajusté du risque."""
        if self.sharpe_ratio:
            return self.annualized_return / max(self.sharpe_ratio, 0.1)
        return self.annualized_return

class EnhancedPortfolio(BaseModel):
    """Portfolio avancé avec validation Pydantic complète."""

    # Identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=100)
    owner_id: str = Field(..., min_length=1)

    # Configuration de base
    base_currency: CurrencyCode = CurrencyCode.USD
    initial_capital: Decimal = Field(..., gt=0, description="Capital initial")
    current_balance: Decimal = Field(..., ge=0)

    # Status et métadonnées
    status: PortfolioStatus = PortfolioStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Positions et ordres
    positions: List[Position] = Field(default_factory=list)
    position_count: int = Field(default=0, ge=0)

    # Métriques
    risk_metrics: RiskMetrics = Field(default_factory=RiskMetrics)
    performance_metrics: Optional[PerformanceMetrics] = None

    # Configuration trading
    max_position_size: Optional[Decimal] = Field(None, gt=0)
    max_daily_loss: Optional[Decimal] = Field(None, gt=0)
    margin_ratio: NonNegativeFloat = Field(default=1.0, le=10.0)

    # Métadonnées étendues
    tags: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = Field(None, max_length=1000)

    # Version pour optimistic locking
    version: int = Field(default=1, ge=1)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Valide le nom du portfolio."""
        if not v.strip():
            raise ValueError("Nom ne peut pas être vide")
        return v.strip()

    @model_validator(mode='after')
    def validate_balances(self) -> 'EnhancedPortfolio':
        """Valide la cohérence des balances."""
        if self.current_balance > self.initial_capital * 10:
            raise ValueError("Balance actuelle ne peut pas dépasser 10x le capital initial")
        return self

    @computed_field
    @property
    def total_equity(self) -> Decimal:
        """Équité totale du portfolio."""
        positions_value = sum(pos.market_value for pos in self.positions)
        return self.current_balance + positions_value

    @computed_field
    @property
    def unrealized_pnl(self) -> Decimal:
        """PnL non réalisé total."""
        return sum(pos.unrealized_pnl or Decimal('0') for pos in self.positions)

    @computed_field
    @property
    def total_return_percent(self) -> float:
        """Rendement total en pourcentage."""
        if self.initial_capital == 0:
            return 0.0
        return float(((self.total_equity - self.initial_capital) / self.initial_capital) * 100)

    @computed_field
    @property
    def position_symbols(self) -> List[str]:
        """Liste des symboles en position."""
        return [pos.symbol for pos in self.positions]

    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Portfolio en bonne santé?"""
        return (
            self.status == PortfolioStatus.ACTIVE and
            self.current_balance >= 0 and
            self.risk_metrics.risk_score < 80
        )

    def add_position(self, position: Position) -> 'EnhancedPortfolio':
        """Ajoute une position avec validation."""
        # Vérifier limites
        if self.max_position_size and position.market_value > self.max_position_size:
            raise ValueError(f"Position dépasse la limite: {self.max_position_size}")

        # Vérifier position existante
        existing_symbols = [pos.symbol for pos in self.positions]
        if position.symbol in existing_symbols:
            raise ValueError(f"Position existe déjà pour {position.symbol}")

        new_positions = self.positions + [position]
        return self.model_copy(update={
            'positions': new_positions,
            'position_count': len(new_positions),
            'updated_at': datetime.now(timezone.utc),
            'version': self.version + 1
        })

    def remove_position(self, symbol: str) -> 'EnhancedPortfolio':
        """Retire une position."""
        new_positions = [pos for pos in self.positions if pos.symbol != symbol]

        if len(new_positions) == len(self.positions):
            raise ValueError(f"Position {symbol} non trouvée")

        return self.model_copy(update={
            'positions': new_positions,
            'position_count': len(new_positions),
            'updated_at': datetime.now(timezone.utc),
            'version': self.version + 1
        })

    def update_position_price(self, symbol: str, new_price: float) -> 'EnhancedPortfolio':
        """Met à jour le prix d'une position."""
        updated_positions = []
        position_found = False

        for pos in self.positions:
            if pos.symbol == symbol:
                updated_positions.append(pos.update_current_price(new_price))
                position_found = True
            else:
                updated_positions.append(pos)

        if not position_found:
            raise ValueError(f"Position {symbol} non trouvée")

        return self.model_copy(update={
            'positions': updated_positions,
            'updated_at': datetime.now(timezone.utc),
            'version': self.version + 1
        })

    def calculate_risk_metrics(self) -> 'EnhancedPortfolio':
        """Recalcule les métriques de risque."""
        if not self.positions:
            return self

        total_exposure = sum(pos.market_value for pos in self.positions)
        max_position = max(pos.market_value for pos in self.positions) if self.positions else Decimal('0')
        max_weight = float(max_position / max(total_exposure, Decimal('1')))

        # Concentration index (Herfindahl)
        weights = [float(pos.market_value / max(total_exposure, Decimal('1'))) for pos in self.positions]
        concentration = sum(w * w for w in weights)

        leverage = float(total_exposure / max(self.current_balance, Decimal('1')))

        updated_risk = self.risk_metrics.model_copy(update={
            'total_exposure': float(total_exposure),
            'leverage': leverage,
            'max_position_weight': max_weight,
            'concentration_index': concentration,
            'last_updated': datetime.now(timezone.utc)
        })

        return self.model_copy(update={
            'risk_metrics': updated_risk,
            'updated_at': datetime.now(timezone.utc)
        })

    def to_summary_dict(self) -> Dict[str, Any]:
        """Résumé du portfolio pour APIs."""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'base_currency': self.base_currency,
            'total_equity': float(self.total_equity),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_return_percent': self.total_return_percent,
            'position_count': self.position_count,
            'risk_score': self.risk_metrics.risk_score,
            'is_healthy': self.is_healthy,
            'updated_at': self.updated_at.isoformat()
        }

    def to_detailed_dict(self) -> Dict[str, Any]:
        """Export détaillé pour analytics."""
        return {
            **self.to_summary_dict(),
            'positions': [pos.model_dump() for pos in self.positions],
            'risk_metrics': self.risk_metrics.model_dump(),
            'performance_metrics': self.performance_metrics.model_dump() if self.performance_metrics else None,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'version': self.version
        }

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        extra='forbid',
        json_encoders={
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        },
        json_schema_extra={
            "example": {
                "name": "Trading Portfolio Alpha",
                "owner_id": "user_123",
                "initial_capital": 10000.0,
                "current_balance": 9500.0,
                "base_currency": "USD",
                "status": "active"
            }
        }
    )


# Factory functions pour création facilitée
def create_portfolio(
    name: str,
    owner_id: str,
    initial_capital: Union[float, Decimal],
    base_currency: CurrencyCode = CurrencyCode.USD
) -> EnhancedPortfolio:
    """Factory pour créer un nouveau portfolio."""

    capital_decimal = Decimal(str(initial_capital))

    return EnhancedPortfolio(
        name=name,
        owner_id=owner_id,
        initial_capital=capital_decimal,
        current_balance=capital_decimal,
        base_currency=base_currency,
        performance_metrics=PerformanceMetrics(
            period_start=datetime.now(timezone.utc)
        )
    )

def create_demo_portfolio() -> EnhancedPortfolio:
    """Crée un portfolio de démonstration avec données."""

    portfolio = create_portfolio(
        name="Demo Trading Portfolio",
        owner_id="demo_user",
        initial_capital=50000,
        base_currency=CurrencyCode.USD
    )

    # Ajouter positions démo
    btc_position = Position(
        symbol="BTC/USD",
        size=Decimal("0.5"),
        entry_price=45000.0,
        current_price=47000.0,
        strategy_name="BTC Momentum"
    )

    eth_position = Position(
        symbol="ETH/USD",
        size=Decimal("5.0"),
        entry_price=3200.0,
        current_price=3350.0,
        strategy_name="ETH Mean Reversion"
    )

    portfolio = portfolio.add_position(btc_position)
    portfolio = portfolio.add_position(eth_position)
    portfolio = portfolio.calculate_risk_metrics()

    return portfolio