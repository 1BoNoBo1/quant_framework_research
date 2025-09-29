"""
🧠 Strategies Router
Endpoints pour la gestion des stratégies
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional
from datetime import datetime
import logging

from qframe.api.models.requests import CreateStrategyRequest, BacktestRequest, PaginationRequest
from qframe.api.models.responses import (
    ApiResponse, StrategyResponse, BacktestResponse, PaginatedResponse
)
from qframe.api.services.strategy_service import StrategyService
from qframe.api.services.backtest_service import BacktestService
from qframe.core.container import get_container

logger = logging.getLogger(__name__)
router = APIRouter()

# Injection de dépendances
def get_strategy_service() -> StrategyService:
    container = get_container()
    return container.resolve(StrategyService)

def get_backtest_service() -> BacktestService:
    container = get_container()
    return container.resolve(BacktestService)


@router.get("/", response_model=PaginatedResponse)
async def get_strategies(
    page: int = Query(1, ge=1, description="Numéro de page"),
    per_page: int = Query(20, ge=1, le=100, description="Stratégies par page"),
    type: Optional[str] = Query(None, description="Filtrer par type"),
    status: Optional[str] = Query(None, description="Filtrer par statut"),
    strategy_service: StrategyService = Depends(get_strategy_service)
):
    """Récupère la liste des stratégies avec pagination et filtres."""
    try:
        filters = {
            "type": type,
            "status": status
        }
        # Supprimer les filtres vides
        filters = {k: v for k, v in filters.items() if v is not None}

        strategies, total = await strategy_service.get_strategies(
            page=page,
            per_page=per_page,
            filters=filters
        )

        strategy_responses = [StrategyResponse(**strategy) for strategy in strategies]

        return PaginatedResponse.create(
            data=strategy_responses,
            total=total,
            page=page,
            per_page=per_page
        )
    except Exception as e:
        logger.error(f"Error fetching strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=ApiResponse)
async def create_strategy(
    request: CreateStrategyRequest,
    strategy_service: StrategyService = Depends(get_strategy_service)
):
    """Crée une nouvelle stratégie."""
    try:
        # Valider le type de stratégie
        valid_types = await strategy_service.get_supported_strategy_types()
        if request.type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy type. Supported types: {valid_types}"
            )

        # Créer la stratégie
        strategy = await strategy_service.create_strategy(request)
        response = StrategyResponse(**strategy)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Strategy '{request.name}' created successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}", response_model=ApiResponse)
async def get_strategy(
    strategy_id: str = Path(..., description="ID de la stratégie"),
    strategy_service: StrategyService = Depends(get_strategy_service)
):
    """Récupère une stratégie spécifique."""
    try:
        strategy = await strategy_service.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        response = StrategyResponse(**strategy)
        return ApiResponse(
            success=True,
            data=response,
            message=f"Strategy {strategy_id} retrieved"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{strategy_id}", response_model=ApiResponse)
async def update_strategy(
    strategy_id: str = Path(..., description="ID de la stratégie"),
    request: CreateStrategyRequest = ...,
    strategy_service: StrategyService = Depends(get_strategy_service)
):
    """Met à jour une stratégie existante."""
    try:
        # Vérifier que la stratégie existe
        existing_strategy = await strategy_service.get_strategy(strategy_id)
        if not existing_strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Mettre à jour la stratégie
        updated_strategy = await strategy_service.update_strategy(strategy_id, request)
        response = StrategyResponse(**updated_strategy)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Strategy {strategy_id} updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{strategy_id}", response_model=ApiResponse)
async def delete_strategy(
    strategy_id: str = Path(..., description="ID de la stratégie"),
    force: bool = Query(False, description="Forcer la suppression même si active"),
    strategy_service: StrategyService = Depends(get_strategy_service)
):
    """Supprime une stratégie."""
    try:
        # Vérifier que la stratégie existe
        existing_strategy = await strategy_service.get_strategy(strategy_id)
        if not existing_strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Vérifier si la stratégie est active
        if existing_strategy["status"] == "ACTIVE" and not force:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete active strategy. Use force=true to override."
            )

        # Supprimer la stratégie
        await strategy_service.delete_strategy(strategy_id)

        return ApiResponse(
            success=True,
            message=f"Strategy {strategy_id} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_id}/start", response_model=ApiResponse)
async def start_strategy(
    strategy_id: str = Path(..., description="ID de la stratégie"),
    strategy_service: StrategyService = Depends(get_strategy_service)
):
    """Démarre une stratégie."""
    try:
        # Vérifier que la stratégie existe
        strategy = await strategy_service.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Vérifier que la stratégie peut être démarrée
        if strategy["status"] == "ACTIVE":
            raise HTTPException(
                status_code=400,
                detail="Strategy is already active"
            )

        # Démarrer la stratégie
        started_strategy = await strategy_service.start_strategy(strategy_id)
        response = StrategyResponse(**started_strategy)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Strategy {strategy_id} started successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_id}/stop", response_model=ApiResponse)
async def stop_strategy(
    strategy_id: str = Path(..., description="ID de la stratégie"),
    strategy_service: StrategyService = Depends(get_strategy_service)
):
    """Arrête une stratégie."""
    try:
        # Vérifier que la stratégie existe
        strategy = await strategy_service.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Vérifier que la stratégie peut être arrêtée
        if strategy["status"] != "ACTIVE":
            raise HTTPException(
                status_code=400,
                detail="Strategy is not active"
            )

        # Arrêter la stratégie
        stopped_strategy = await strategy_service.stop_strategy(strategy_id)
        response = StrategyResponse(**stopped_strategy)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Strategy {strategy_id} stopped successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}/performance", response_model=ApiResponse)
async def get_strategy_performance(
    strategy_id: str = Path(..., description="ID de la stratégie"),
    start_date: Optional[datetime] = Query(None, description="Date de début"),
    end_date: Optional[datetime] = Query(None, description="Date de fin"),
    strategy_service: StrategyService = Depends(get_strategy_service)
):
    """Récupère les performances d'une stratégie."""
    try:
        performance = await strategy_service.get_strategy_performance(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        if not performance:
            raise HTTPException(status_code=404, detail=f"Performance data for strategy {strategy_id} not found")

        return ApiResponse(
            success=True,
            data=performance,
            message=f"Performance data for strategy {strategy_id} retrieved"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching strategy performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types", response_model=ApiResponse)
async def get_strategy_types(
    strategy_service: StrategyService = Depends(get_strategy_service)
):
    """Récupère les types de stratégies supportés."""
    try:
        types = await strategy_service.get_supported_strategy_types()
        return ApiResponse(
            success=True,
            data=types,
            message=f"Found {len(types)} supported strategy types"
        )
    except Exception as e:
        logger.error(f"Error fetching strategy types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints de backtesting
@router.post("/{strategy_id}/backtest", response_model=ApiResponse)
async def create_backtest(
    strategy_id: str = Path(..., description="ID de la stratégie"),
    request: BacktestRequest = ...,
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """Lance un backtest pour une stratégie."""
    try:
        # Vérifier que l'ID de stratégie correspond
        if request.strategy_id != strategy_id:
            raise HTTPException(
                status_code=400,
                detail="Strategy ID in path must match strategy ID in request"
            )

        # Créer le backtest
        backtest = await backtest_service.create_backtest(request)
        response = BacktestResponse(**backtest)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Backtest created for strategy {strategy_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}/backtests", response_model=PaginatedResponse)
async def get_strategy_backtests(
    strategy_id: str = Path(..., description="ID de la stratégie"),
    page: int = Query(1, ge=1, description="Numéro de page"),
    per_page: int = Query(20, ge=1, le=100, description="Backtests par page"),
    status: Optional[str] = Query(None, description="Filtrer par statut"),
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """Récupère les backtests d'une stratégie."""
    try:
        filters = {"strategy_id": strategy_id}
        if status:
            filters["status"] = status

        backtests, total = await backtest_service.get_backtests(
            page=page,
            per_page=per_page,
            filters=filters
        )

        backtest_responses = [BacktestResponse(**backtest) for backtest in backtests]

        return PaginatedResponse.create(
            data=backtest_responses,
            total=total,
            page=page,
            per_page=per_page
        )
    except Exception as e:
        logger.error(f"Error fetching strategy backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtests/{backtest_id}", response_model=ApiResponse)
async def get_backtest(
    backtest_id: str = Path(..., description="ID du backtest"),
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """Récupère un backtest spécifique."""
    try:
        backtest = await backtest_service.get_backtest(backtest_id)
        if not backtest:
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        response = BacktestResponse(**backtest)
        return ApiResponse(
            success=True,
            data=response,
            message=f"Backtest {backtest_id} retrieved"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching backtest {backtest_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/backtests/{backtest_id}", response_model=ApiResponse)
async def cancel_backtest(
    backtest_id: str = Path(..., description="ID du backtest"),
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """Annule un backtest en cours."""
    try:
        # Vérifier que le backtest existe
        backtest = await backtest_service.get_backtest(backtest_id)
        if not backtest:
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        # Vérifier que le backtest peut être annulé
        if backtest["status"] not in ["RUNNING"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel backtest with status {backtest['status']}"
            )

        # Annuler le backtest
        cancelled_backtest = await backtest_service.cancel_backtest(backtest_id)
        response = BacktestResponse(**cancelled_backtest)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Backtest {backtest_id} cancelled successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling backtest {backtest_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtests/{backtest_id}/results", response_model=ApiResponse)
async def get_backtest_results(
    backtest_id: str = Path(..., description="ID du backtest"),
    include_trades: bool = Query(False, description="Inclure les détails des trades"),
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """Récupère les résultats détaillés d'un backtest."""
    try:
        results = await backtest_service.get_backtest_results(
            backtest_id=backtest_id,
            include_trades=include_trades
        )
        if not results:
            raise HTTPException(status_code=404, detail=f"Results for backtest {backtest_id} not found")

        return ApiResponse(
            success=True,
            data=results,
            message=f"Results for backtest {backtest_id} retrieved"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))