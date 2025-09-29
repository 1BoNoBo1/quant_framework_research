"""
📤 Response Models
Modèles de réponse pour l'API QFrame
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class StatusEnum(str, Enum):
    """Statuts de santé de l'API."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Réponse de vérification de santé."""
    status: StatusEnum
    timestamp: datetime
    services: Dict[str, bool] = Field(default_factory=dict)
    version: str
    uptime: str
    error: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ApiResponse(BaseModel):
    """Réponse API générique."""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    errors: Optional[List[str]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginatedResponse(BaseModel):
    """Réponse paginée."""
    data: List[Any]
    total: int
    page: int = Field(ge=1)
    per_page: int = Field(ge=1, le=100)
    pages: int
    has_next: bool
    has_prev: bool

    @classmethod
    def create(cls, data: List[Any], total: int, page: int, per_page: int):
        """Crée une réponse paginée."""
        pages = (total + per_page - 1) // per_page
        return cls(
            data=data,
            total=total,
            page=page,
            per_page=per_page,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1
        )


class MarketDataResponse(BaseModel):
    """Réponse de données de marché."""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    timestamp: datetime
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    open_24h: Optional[float] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OHLCVResponse(BaseModel):
    """Réponse OHLCV."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OrderResponse(BaseModel):
    """Réponse d'ordre."""
    id: str
    symbol: str
    side: str  # BUY, SELL
    type: str  # MARKET, LIMIT, STOP, etc.
    quantity: float
    price: Optional[float] = None
    filled_quantity: float = 0.0
    status: str  # PENDING, FILLED, CANCELLED, etc.
    created_at: datetime
    updated_at: datetime
    fees: Optional[float] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PositionResponse(BaseModel):
    """Réponse de position."""
    id: str
    symbol: str
    side: str  # LONG, SHORT
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin: float
    leverage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PortfolioResponse(BaseModel):
    """Réponse de portefeuille."""
    id: str
    name: str
    total_balance: float
    available_balance: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    positions_count: int
    active_orders_count: int
    updated_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RiskMetricsResponse(BaseModel):
    """Réponse de métriques de risque."""
    portfolio_var_95: float
    portfolio_var_99: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    leverage_ratio: float
    concentration_risk: float
    correlation_risk: float
    liquidity_risk: str  # LOW, MEDIUM, HIGH
    last_calculated: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StrategyResponse(BaseModel):
    """Réponse de stratégie."""
    id: str
    name: str
    type: str
    status: str  # ACTIVE, INACTIVE, PAUSED
    parameters: Dict[str, Any]
    performance: Dict[str, float]
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BacktestResponse(BaseModel):
    """Réponse de backtest."""
    id: str
    strategy_id: str
    status: str  # RUNNING, COMPLETED, FAILED
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    trades_count: int
    win_rate: float
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WebSocketMessage(BaseModel):
    """Message WebSocket."""
    type: str  # price_update, order_update, position_update, alert
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Réponse d'erreur."""
    error: bool = True
    message: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }