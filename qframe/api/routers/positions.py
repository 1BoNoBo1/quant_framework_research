"""
📊 Positions Router
Endpoints pour la gestion des positions
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional
from datetime import datetime
import logging

from qframe.api.models.responses import (
    ApiResponse, PositionResponse, PortfolioResponse, PaginatedResponse
)
from qframe.api.services.position_service import PositionService
from qframe.api.services.portfolio_service import PortfolioService
from qframe.core.container import get_container

logger = logging.getLogger(__name__)
router = APIRouter()

# Injection de dépendances
def get_position_service() -> PositionService:
    container = get_container()
    return container.resolve(PositionService)

def get_portfolio_service() -> PortfolioService:
    container = get_container()
    return container.resolve(PortfolioService)


@router.get("/", response_model=PaginatedResponse)
async def get_positions(
    page: int = Query(1, ge=1, description="Numéro de page"),
    per_page: int = Query(20, ge=1, le=100, description="Positions par page"),
    symbol: Optional[str] = Query(None, description="Filtrer par symbole"),
    side: Optional[str] = Query(None, description="Filtrer par côté (LONG/SHORT)"),
    status: Optional[str] = Query(None, description="Filtrer par statut"),
    position_service: PositionService = Depends(get_position_service)
):
    """Récupère la liste des positions avec pagination et filtres."""
    try:
        filters = {
            "symbol": symbol,
            "side": side,
            "status": status
        }
        # Supprimer les filtres vides
        filters = {k: v for k, v in filters.items() if v is not None}

        positions, total = await position_service.get_positions(
            page=page,
            per_page=per_page,
            filters=filters
        )

        position_responses = [PositionResponse(**position) for position in positions]

        return PaginatedResponse.create(
            data=position_responses,
            total=total,
            page=page,
            per_page=per_page
        )
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{position_id}", response_model=ApiResponse)
async def get_position(
    position_id: str = Path(..., description="ID de la position"),
    position_service: PositionService = Depends(get_position_service)
):
    """Récupère une position spécifique."""
    try:
        position = await position_service.get_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")

        response = PositionResponse(**position)
        return ApiResponse(
            success=True,
            data=response,
            message=f"Position {position_id} retrieved"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching position {position_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{position_id}/stop-loss", response_model=ApiResponse)
async def update_stop_loss(
    position_id: str = Path(..., description="ID de la position"),
    stop_loss_price: float = Query(..., gt=0, description="Nouveau prix de stop loss"),
    position_service: PositionService = Depends(get_position_service)
):
    """Met à jour le stop loss d'une position."""
    try:
        position = await position_service.get_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")

        # Validation du stop loss selon le côté de la position
        current_price = position["current_price"]
        if position["side"] == "LONG" and stop_loss_price >= current_price:
            raise HTTPException(
                status_code=400,
                detail="Stop loss price must be below current price for LONG positions"
            )
        elif position["side"] == "SHORT" and stop_loss_price <= current_price:
            raise HTTPException(
                status_code=400,
                detail="Stop loss price must be above current price for SHORT positions"
            )

        updated_position = await position_service.update_stop_loss(position_id, stop_loss_price)
        response = PositionResponse(**updated_position)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Stop loss updated for position {position_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating stop loss for position {position_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{position_id}/take-profit", response_model=ApiResponse)
async def update_take_profit(
    position_id: str = Path(..., description="ID de la position"),
    take_profit_price: float = Query(..., gt=0, description="Nouveau prix de take profit"),
    position_service: PositionService = Depends(get_position_service)
):
    """Met à jour le take profit d'une position."""
    try:
        position = await position_service.get_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")

        # Validation du take profit selon le côté de la position
        current_price = position["current_price"]
        if position["side"] == "LONG" and take_profit_price <= current_price:
            raise HTTPException(
                status_code=400,
                detail="Take profit price must be above current price for LONG positions"
            )
        elif position["side"] == "SHORT" and take_profit_price >= current_price:
            raise HTTPException(
                status_code=400,
                detail="Take profit price must be below current price for SHORT positions"
            )

        updated_position = await position_service.update_take_profit(position_id, take_profit_price)
        response = PositionResponse(**updated_position)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Take profit updated for position {position_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating take profit for position {position_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{position_id}", response_model=ApiResponse)
async def close_position(
    position_id: str = Path(..., description="ID de la position"),
    close_price: Optional[float] = Query(None, gt=0, description="Prix de fermeture (market si non spécifié)"),
    position_service: PositionService = Depends(get_position_service)
):
    """Ferme une position."""
    try:
        position = await position_service.get_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")

        # Fermer la position
        closed_position = await position_service.close_position(position_id, close_price)
        response = PositionResponse(**closed_position)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Position {position_id} closed successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position {position_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-all", response_model=ApiResponse)
async def close_all_positions(
    symbol: Optional[str] = Query(None, description="Symbole spécifique (sinon toutes)"),
    confirm: bool = Query(False, description="Confirmation required"),
    position_service: PositionService = Depends(get_position_service)
):
    """Ferme toutes les positions."""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirmation required to close all positions (set confirm=true)"
            )

        closed_positions = await position_service.close_all_positions(symbol)

        return ApiResponse(
            success=True,
            data={
                "closed_count": len(closed_positions),
                "closed_positions": [PositionResponse(**pos) for pos in closed_positions],
                "symbol": symbol
            },
            message=f"Closed {len(closed_positions)} positions"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/summary", response_model=ApiResponse)
async def get_portfolio_summary(
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    """Récupère le résumé du portefeuille."""
    try:
        summary = await portfolio_service.get_portfolio_summary()
        response = PortfolioResponse(**summary)

        return ApiResponse(
            success=True,
            data=response,
            message="Portfolio summary retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/allocation", response_model=ApiResponse)
async def get_portfolio_allocation(
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    """Récupère l'allocation du portefeuille par asset."""
    try:
        allocation = await portfolio_service.get_portfolio_allocation()

        return ApiResponse(
            success=True,
            data=allocation,
            message="Portfolio allocation retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching portfolio allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/performance", response_model=ApiResponse)
async def get_portfolio_performance(
    start_date: Optional[datetime] = Query(None, description="Date de début"),
    end_date: Optional[datetime] = Query(None, description="Date de fin"),
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    """Récupère les performances du portefeuille."""
    try:
        performance = await portfolio_service.get_portfolio_performance(
            start_date=start_date,
            end_date=end_date
        )

        return ApiResponse(
            success=True,
            data=performance,
            message="Portfolio performance retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/pnl", response_model=ApiResponse)
async def get_pnl_analytics(
    period: str = Query("24h", pattern="^(1h|4h|24h|7d|30d)$", description="Période d'analyse"),
    symbol: Optional[str] = Query(None, description="Symbole spécifique"),
    position_service: PositionService = Depends(get_position_service)
):
    """Récupère l'analyse PnL détaillée."""
    try:
        analytics = await position_service.get_pnl_analytics(period, symbol)

        return ApiResponse(
            success=True,
            data=analytics,
            message=f"PnL analytics for {period} retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching PnL analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{position_id}", response_model=ApiResponse)
async def get_position_history(
    position_id: str = Path(..., description="ID de la position"),
    position_service: PositionService = Depends(get_position_service)
):
    """Récupère l'historique d'une position."""
    try:
        history = await position_service.get_position_history(position_id)
        if not history:
            raise HTTPException(status_code=404, detail=f"Position {position_id} history not found")

        return ApiResponse(
            success=True,
            data=history,
            message=f"Position {position_id} history retrieved"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching position history for {position_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exposure/net", response_model=ApiResponse)
async def get_net_exposure(
    symbol: Optional[str] = Query(None, description="Symbole spécifique"),
    position_service: PositionService = Depends(get_position_service)
):
    """Récupère l'exposition nette."""
    try:
        exposure = await position_service.get_net_exposure(symbol)

        return ApiResponse(
            success=True,
            data=exposure,
            message="Net exposure retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching net exposure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=ApiResponse)
async def get_position_statistics(
    start_date: Optional[datetime] = Query(None, description="Date de début"),
    end_date: Optional[datetime] = Query(None, description="Date de fin"),
    symbol: Optional[str] = Query(None, description="Symbole spécifique"),
    position_service: PositionService = Depends(get_position_service)
):
    """Récupère les statistiques des positions."""
    try:
        stats = await position_service.get_position_statistics(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol
        )

        return ApiResponse(
            success=True,
            data=stats,
            message="Position statistics retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching position statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))