"""
📊 Market Data Router
Endpoints pour les données de marché
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from qframe.api.models.requests import MarketDataRequest, TimeframeEnum
from qframe.api.models.responses import (
    ApiResponse, MarketDataResponse, OHLCVResponse, PaginatedResponse
)
from qframe.api.services.market_data_service import MarketDataService
from qframe.core.container import get_container

logger = logging.getLogger(__name__)
router = APIRouter()

# Injection de dépendance
def get_market_data_service() -> MarketDataService:
    container = get_container()
    return container.resolve(MarketDataService)


@router.get("/symbols", response_model=ApiResponse)
async def get_supported_symbols(
    exchange: Optional[str] = Query(None, description="Exchange spécifique"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère la liste des symboles supportés."""
    try:
        symbols = await market_data_service.get_supported_symbols(exchange)
        return ApiResponse(
            success=True,
            data=symbols,
            message=f"Found {len(symbols)} supported symbols"
        )
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price/{symbol}", response_model=ApiResponse)
async def get_current_price(
    symbol: str,
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère le prix actuel d'un symbole."""
    try:
        price_data = await market_data_service.get_current_price(symbol)
        if not price_data:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

        response = MarketDataResponse(**price_data)
        return ApiResponse(
            success=True,
            data=response,
            message=f"Current price for {symbol}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prices", response_model=ApiResponse)
async def get_multiple_prices(
    symbols: str = Query(..., description="Symboles séparés par des virgules"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère les prix de plusieurs symboles."""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        if len(symbol_list) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 symbols allowed")

        prices = await market_data_service.get_multiple_prices(symbol_list)
        responses = [MarketDataResponse(**price) for price in prices]

        return ApiResponse(
            success=True,
            data=responses,
            message=f"Prices for {len(responses)} symbols"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching multiple prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ohlcv/{symbol}", response_model=ApiResponse)
async def get_ohlcv_data(
    symbol: str,
    timeframe: TimeframeEnum = Query(TimeframeEnum.H1, description="Timeframe"),
    start_date: Optional[datetime] = Query(None, description="Date de début"),
    end_date: Optional[datetime] = Query(None, description="Date de fin"),
    limit: int = Query(100, ge=1, le=1000, description="Nombre de bougies"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère les données OHLCV pour un symbole."""
    try:
        # Validation des dates
        if start_date and end_date and end_date <= start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")

        # Par défaut, récupérer les dernières données
        if not start_date and not end_date:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=1)

        ohlcv_data = await market_data_service.get_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe.value,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        responses = [OHLCVResponse(**candle) for candle in ohlcv_data]

        return ApiResponse(
            success=True,
            data=responses,
            message=f"OHLCV data for {symbol} ({timeframe.value})"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ticker/{symbol}", response_model=ApiResponse)
async def get_ticker_data(
    symbol: str,
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère les données ticker complètes pour un symbole."""
    try:
        ticker_data = await market_data_service.get_ticker_data(symbol)
        if not ticker_data:
            raise HTTPException(status_code=404, detail=f"Ticker data for {symbol} not found")

        return ApiResponse(
            success=True,
            data=ticker_data,
            message=f"Ticker data for {symbol}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ticker for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/depth/{symbol}", response_model=ApiResponse)
async def get_order_book(
    symbol: str,
    limit: int = Query(20, ge=5, le=100, description="Nombre de niveaux"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère le carnet d'ordres pour un symbole."""
    try:
        order_book = await market_data_service.get_order_book(symbol, limit)
        if not order_book:
            raise HTTPException(status_code=404, detail=f"Order book for {symbol} not found")

        return ApiResponse(
            success=True,
            data=order_book,
            message=f"Order book for {symbol}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching order book for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/{symbol}", response_model=ApiResponse)
async def get_recent_trades(
    symbol: str,
    limit: int = Query(50, ge=1, le=500, description="Nombre de trades"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère les trades récents pour un symbole."""
    try:
        trades = await market_data_service.get_recent_trades(symbol, limit)

        return ApiResponse(
            success=True,
            data=trades,
            message=f"Recent trades for {symbol}"
        )
    except Exception as e:
        logger.error(f"Error fetching trades for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exchanges", response_model=ApiResponse)
async def get_supported_exchanges(
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère la liste des exchanges supportés."""
    try:
        exchanges = await market_data_service.get_supported_exchanges()
        return ApiResponse(
            success=True,
            data=exchanges,
            message=f"Found {len(exchanges)} supported exchanges"
        )
    except Exception as e:
        logger.error(f"Error fetching exchanges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=ApiResponse)
async def get_market_status(
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère le statut des marchés."""
    try:
        status = await market_data_service.get_market_status()
        return ApiResponse(
            success=True,
            data=status,
            message="Market status retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching market status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk", response_model=ApiResponse)
async def get_bulk_market_data(
    request: MarketDataRequest,
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère des données de marché en bulk."""
    try:
        if len(request.symbols) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 symbols allowed")

        bulk_data = await market_data_service.get_bulk_data(
            symbols=request.symbols,
            timeframe=request.timeframe.value if request.timeframe else None,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.limit
        )

        return ApiResponse(
            success=True,
            data=bulk_data,
            message=f"Bulk data for {len(request.symbols)} symbols"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching bulk market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{symbol}", response_model=ApiResponse)
async def get_market_stats(
    symbol: str,
    period: str = Query("24h", pattern="^(1h|4h|24h|7d|30d)$", description="Période pour les stats"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Récupère les statistiques de marché pour un symbole."""
    try:
        stats = await market_data_service.get_market_stats(symbol, period)
        if not stats:
            raise HTTPException(status_code=404, detail=f"Stats for {symbol} not found")

        return ApiResponse(
            success=True,
            data=stats,
            message=f"Market stats for {symbol} ({period})"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stats for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))