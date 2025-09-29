"""
📥 Request Models
Modèles de requête pour l'API QFrame
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class OrderSideEnum(str, Enum):
    """Côtés d'ordre."""
    BUY = "BUY"
    SELL = "SELL"


class OrderTypeEnum(str, Enum):
    """Types d'ordre."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class TimeframeEnum(str, Enum):
    """Timeframes pour les données de marché."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class CreateOrderRequest(BaseModel):
    """Requête de création d'ordre."""
    symbol: str = Field(..., description="Symbole de trading (ex: BTC/USDT)")
    side: OrderSideEnum = Field(..., description="Côté de l'ordre")
    type: OrderTypeEnum = Field(..., description="Type d'ordre")
    quantity: float = Field(..., gt=0, description="Quantité à trader")
    price: Optional[float] = Field(None, gt=0, description="Prix limite (requis pour LIMIT)")
    stop_price: Optional[float] = Field(None, gt=0, description="Prix stop")
    time_in_force: Optional[str] = Field("GTC", description="Durée de validité")

    # Risk management
    stop_loss: Optional[float] = Field(None, gt=0, description="Prix de stop loss")
    take_profit: Optional[float] = Field(None, gt=0, description="Prix de take profit")

    # Métadonnées
    client_order_id: Optional[str] = Field(None, description="ID client personnalisé")
    notes: Optional[str] = Field(None, max_length=500, description="Notes sur l'ordre")

    @validator('price')
    def price_required_for_limit(cls, v, values):
        """Valide que le prix est fourni pour les ordres LIMIT."""
        if values.get('type') in ['LIMIT', 'STOP_LIMIT'] and v is None:
            raise ValueError('Price is required for LIMIT and STOP_LIMIT orders')
        return v

    @validator('stop_price')
    def stop_price_required_for_stop(cls, v, values):
        """Valide que le prix stop est fourni pour les ordres STOP."""
        if values.get('type') in ['STOP', 'STOP_LIMIT', 'TRAILING_STOP'] and v is None:
            raise ValueError('Stop price is required for stop orders')
        return v


class UpdateOrderRequest(BaseModel):
    """Requête de modification d'ordre."""
    quantity: Optional[float] = Field(None, gt=0, description="Nouvelle quantité")
    price: Optional[float] = Field(None, gt=0, description="Nouveau prix")
    stop_price: Optional[float] = Field(None, gt=0, description="Nouveau prix stop")
    stop_loss: Optional[float] = Field(None, gt=0, description="Nouveau stop loss")
    take_profit: Optional[float] = Field(None, gt=0, description="Nouveau take profit")


class MarketDataRequest(BaseModel):
    """Requête de données de marché."""
    symbols: List[str] = Field(..., min_items=1, max_items=50, description="Symboles demandés")
    timeframe: Optional[TimeframeEnum] = Field(None, description="Timeframe pour historique")
    start_date: Optional[datetime] = Field(None, description="Date de début")
    end_date: Optional[datetime] = Field(None, description="Date de fin")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Nombre maximum de points")

    @validator('end_date')
    def end_date_after_start(cls, v, values):
        """Valide que la date de fin est après le début."""
        start_date = values.get('start_date')
        if start_date and v and v <= start_date:
            raise ValueError('End date must be after start date')
        return v


class CreateStrategyRequest(BaseModel):
    """Requête de création de stratégie."""
    name: str = Field(..., min_length=1, max_length=100, description="Nom de la stratégie")
    type: str = Field(..., description="Type de stratégie")
    parameters: Dict[str, Any] = Field(..., description="Paramètres de la stratégie")
    symbols: List[str] = Field(..., min_items=1, description="Symboles à trader")
    risk_parameters: Optional[Dict[str, Any]] = Field(None, description="Paramètres de risque")
    active: bool = Field(True, description="Stratégie active")


class BacktestRequest(BaseModel):
    """Requête de backtest."""
    strategy_id: str = Field(..., description="ID de la stratégie")
    start_date: datetime = Field(..., description="Date de début du backtest")
    end_date: datetime = Field(..., description="Date de fin du backtest")
    initial_capital: float = Field(..., gt=0, description="Capital initial")
    symbols: Optional[List[str]] = Field(None, description="Symboles spécifiques")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Paramètres personnalisés")

    @validator('end_date')
    def end_date_after_start(cls, v, values):
        """Valide que la date de fin est après le début."""
        start_date = values.get('start_date')
        if start_date and v <= start_date:
            raise ValueError('End date must be after start date')
        return v


class RiskConfigRequest(BaseModel):
    """Requête de configuration des risques."""
    max_portfolio_var: Optional[float] = Field(None, gt=0, description="VaR maximum du portefeuille")
    max_position_size: Optional[float] = Field(None, gt=0, description="Taille maximum d'une position")
    max_leverage: Optional[float] = Field(None, gt=0, description="Leverage maximum")
    max_correlation: Optional[float] = Field(None, ge=0, le=1, description="Corrélation maximum")
    max_drawdown: Optional[float] = Field(None, gt=0, description="Drawdown maximum")
    position_limit_pct: Optional[float] = Field(None, gt=0, le=100, description="% maximum du portefeuille par position")
    daily_loss_limit: Optional[float] = Field(None, gt=0, description="Perte journalière maximum")


class PaginationRequest(BaseModel):
    """Requête de pagination."""
    page: int = Field(1, ge=1, description="Numéro de page")
    per_page: int = Field(20, ge=1, le=100, description="Éléments par page")
    sort_by: Optional[str] = Field(None, description="Champ de tri")
    sort_order: Optional[str] = Field("desc", pattern="^(asc|desc)$", description="Ordre de tri")


class FilterRequest(BaseModel):
    """Requête de filtrage."""
    symbol: Optional[str] = Field(None, description="Filtrer par symbole")
    status: Optional[str] = Field(None, description="Filtrer par statut")
    side: Optional[str] = Field(None, description="Filtrer par côté")
    type: Optional[str] = Field(None, description="Filtrer par type")
    start_date: Optional[datetime] = Field(None, description="Date de début")
    end_date: Optional[datetime] = Field(None, description="Date de fin")


class BulkOrderRequest(BaseModel):
    """Requête d'ordres en lot."""
    orders: List[CreateOrderRequest] = Field(..., min_items=1, max_items=20, description="Liste d'ordres")
    fail_on_error: bool = Field(False, description="Arrêter si une erreur survient")


class WebSocketSubscriptionRequest(BaseModel):
    """Requête d'abonnement WebSocket."""
    channels: List[str] = Field(..., min_items=1, description="Canaux à suivre")
    symbols: Optional[List[str]] = Field(None, description="Symboles spécifiques")

    @validator('channels')
    def validate_channels(cls, v):
        """Valide les canaux disponibles."""
        valid_channels = ['price', 'order', 'position', 'trade', 'kline', 'depth']
        for channel in v:
            if channel not in valid_channels:
                raise ValueError(f'Invalid channel: {channel}. Valid channels: {valid_channels}')
        return v