"""
📋 Orders Router
Endpoints pour la gestion des ordres
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional
from datetime import datetime
import logging

from qframe.api.models.requests import (
    CreateOrderRequest, UpdateOrderRequest, BulkOrderRequest,
    PaginationRequest, FilterRequest
)
from qframe.api.models.responses import (
    ApiResponse, OrderResponse, PaginatedResponse
)
from qframe.api.services.order_service import OrderService
from qframe.api.services.risk_service import RiskService
from qframe.core.container import get_container

logger = logging.getLogger(__name__)
router = APIRouter()

# Injection de dépendances
def get_order_service() -> OrderService:
    container = get_container()
    return container.resolve(OrderService)

def get_risk_service() -> RiskService:
    container = get_container()
    return container.resolve(RiskService)


@router.post("/", response_model=ApiResponse)
async def create_order(
    request: CreateOrderRequest,
    order_service: OrderService = Depends(get_order_service),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Crée un nouvel ordre."""
    try:
        # Validation des risques avant création
        risk_check = await risk_service.validate_order_risk(request)
        if not risk_check.approved:
            raise HTTPException(
                status_code=400,
                detail=f"Order rejected by risk management: {risk_check.reason}"
            )

        # Créer l'ordre
        order = await order_service.create_order(request)
        response = OrderResponse(**order)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Order {order['id']} created successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=PaginatedResponse)
async def get_orders(
    page: int = Query(1, ge=1, description="Numéro de page"),
    per_page: int = Query(20, ge=1, le=100, description="Ordres par page"),
    symbol: Optional[str] = Query(None, description="Filtrer par symbole"),
    status: Optional[str] = Query(None, description="Filtrer par statut"),
    side: Optional[str] = Query(None, description="Filtrer par côté"),
    order_type: Optional[str] = Query(None, description="Filtrer par type"),
    start_date: Optional[datetime] = Query(None, description="Date de début"),
    end_date: Optional[datetime] = Query(None, description="Date de fin"),
    order_service: OrderService = Depends(get_order_service)
):
    """Récupère la liste des ordres avec pagination et filtres."""
    try:
        filters = {
            "symbol": symbol,
            "status": status,
            "side": side,
            "type": order_type,
            "start_date": start_date,
            "end_date": end_date
        }
        # Supprimer les filtres vides
        filters = {k: v for k, v in filters.items() if v is not None}

        orders, total = await order_service.get_orders(
            page=page,
            per_page=per_page,
            filters=filters
        )

        order_responses = [OrderResponse(**order) for order in orders]

        return PaginatedResponse.create(
            data=order_responses,
            total=total,
            page=page,
            per_page=per_page
        )
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{order_id}", response_model=ApiResponse)
async def get_order(
    order_id: str = Path(..., description="ID de l'ordre"),
    order_service: OrderService = Depends(get_order_service)
):
    """Récupère un ordre spécifique."""
    try:
        order = await order_service.get_order(order_id)
        if not order:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        response = OrderResponse(**order)
        return ApiResponse(
            success=True,
            data=response,
            message=f"Order {order_id} retrieved"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{order_id}", response_model=ApiResponse)
async def update_order(
    order_id: str = Path(..., description="ID de l'ordre"),
    request: UpdateOrderRequest = ...,
    order_service: OrderService = Depends(get_order_service),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Met à jour un ordre existant."""
    try:
        # Vérifier que l'ordre existe
        existing_order = await order_service.get_order(order_id)
        if not existing_order:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Vérifier que l'ordre peut être modifié
        if existing_order["status"] not in ["PENDING", "PARTIAL"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot update order with status {existing_order['status']}"
            )

        # Validation des risques pour la modification
        risk_check = await risk_service.validate_order_update_risk(order_id, request)
        if not risk_check.approved:
            raise HTTPException(
                status_code=400,
                detail=f"Order update rejected by risk management: {risk_check.reason}"
            )

        # Mettre à jour l'ordre
        updated_order = await order_service.update_order(order_id, request)
        response = OrderResponse(**updated_order)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Order {order_id} updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{order_id}", response_model=ApiResponse)
async def cancel_order(
    order_id: str = Path(..., description="ID de l'ordre"),
    order_service: OrderService = Depends(get_order_service)
):
    """Annule un ordre."""
    try:
        # Vérifier que l'ordre existe
        existing_order = await order_service.get_order(order_id)
        if not existing_order:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Vérifier que l'ordre peut être annulé
        if existing_order["status"] not in ["PENDING", "PARTIAL"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel order with status {existing_order['status']}"
            )

        # Annuler l'ordre
        cancelled_order = await order_service.cancel_order(order_id)
        response = OrderResponse(**cancelled_order)

        return ApiResponse(
            success=True,
            data=response,
            message=f"Order {order_id} cancelled successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk", response_model=ApiResponse)
async def create_bulk_orders(
    request: BulkOrderRequest,
    order_service: OrderService = Depends(get_order_service),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Crée plusieurs ordres en une seule requête."""
    try:
        if len(request.orders) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 orders per bulk request")

        # Validation des risques pour tous les ordres
        for i, order_request in enumerate(request.orders):
            risk_check = await risk_service.validate_order_risk(order_request)
            if not risk_check.approved:
                if request.fail_on_error:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Order {i+1} rejected by risk management: {risk_check.reason}"
                    )

        # Créer les ordres
        results = await order_service.create_bulk_orders(request.orders, request.fail_on_error)

        successful_orders = [OrderResponse(**order) for order in results if order.get("success")]
        failed_orders = [order for order in results if not order.get("success")]

        return ApiResponse(
            success=len(failed_orders) == 0,
            data={
                "successful": successful_orders,
                "failed": failed_orders,
                "total": len(request.orders),
                "success_count": len(successful_orders),
                "failure_count": len(failed_orders)
            },
            message=f"Bulk order creation completed: {len(successful_orders)} success, {len(failed_orders)} failed"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating bulk orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active/count", response_model=ApiResponse)
async def get_active_orders_count(
    symbol: Optional[str] = Query(None, description="Symbole spécifique"),
    order_service: OrderService = Depends(get_order_service)
):
    """Récupère le nombre d'ordres actifs."""
    try:
        count = await order_service.get_active_orders_count(symbol)
        return ApiResponse(
            success=True,
            data={"count": count, "symbol": symbol},
            message=f"Active orders count: {count}"
        )
    except Exception as e:
        logger.error(f"Error getting active orders count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel-all", response_model=ApiResponse)
async def cancel_all_orders(
    symbol: Optional[str] = Query(None, description="Symbole spécifique (sinon tous)"),
    confirm: bool = Query(False, description="Confirmation required"),
    order_service: OrderService = Depends(get_order_service)
):
    """Annule tous les ordres actifs."""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirmation required to cancel all orders (set confirm=true)"
            )

        cancelled_orders = await order_service.cancel_all_orders(symbol)

        return ApiResponse(
            success=True,
            data={
                "cancelled_count": len(cancelled_orders),
                "cancelled_orders": [OrderResponse(**order) for order in cancelled_orders],
                "symbol": symbol
            },
            message=f"Cancelled {len(cancelled_orders)} orders"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling all orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{order_id}", response_model=ApiResponse)
async def get_order_history(
    order_id: str = Path(..., description="ID de l'ordre"),
    order_service: OrderService = Depends(get_order_service)
):
    """Récupère l'historique des modifications d'un ordre."""
    try:
        history = await order_service.get_order_history(order_id)
        if not history:
            raise HTTPException(status_code=404, detail=f"Order {order_id} history not found")

        return ApiResponse(
            success=True,
            data=history,
            message=f"Order {order_id} history retrieved"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching order history for {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fills/{order_id}", response_model=ApiResponse)
async def get_order_fills(
    order_id: str = Path(..., description="ID de l'ordre"),
    order_service: OrderService = Depends(get_order_service)
):
    """Récupère les exécutions partielles d'un ordre."""
    try:
        fills = await order_service.get_order_fills(order_id)

        return ApiResponse(
            success=True,
            data=fills,
            message=f"Order {order_id} fills retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching order fills for {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=ApiResponse)
async def get_order_statistics(
    start_date: Optional[datetime] = Query(None, description="Date de début"),
    end_date: Optional[datetime] = Query(None, description="Date de fin"),
    symbol: Optional[str] = Query(None, description="Symbole spécifique"),
    order_service: OrderService = Depends(get_order_service)
):
    """Récupère les statistiques des ordres."""
    try:
        stats = await order_service.get_order_statistics(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol
        )

        return ApiResponse(
            success=True,
            data=stats,
            message="Order statistics retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching order statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))