"""
⚠️ Risk Router
Endpoints pour la gestion des risques
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from datetime import datetime
import logging

from qframe.api.models.requests import RiskConfigRequest
from qframe.api.models.responses import ApiResponse, RiskMetricsResponse
from qframe.api.services.risk_service import RiskService
from qframe.core.container import get_container

logger = logging.getLogger(__name__)
router = APIRouter()

# Injection de dépendances
def get_risk_service() -> RiskService:
    container = get_container()
    return container.resolve(RiskService)


@router.get("/metrics", response_model=ApiResponse)
async def get_risk_metrics(
    risk_service: RiskService = Depends(get_risk_service)
):
    """Récupère les métriques de risque actuelles."""
    try:
        metrics = await risk_service.get_current_risk_metrics()
        response = RiskMetricsResponse(**metrics)

        return ApiResponse(
            success=True,
            data=response,
            message="Risk metrics retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/var", response_model=ApiResponse)
async def get_var_calculation(
    confidence_level: float = Query(0.95, ge=0.9, le=0.99, description="Niveau de confiance"),
    time_horizon: int = Query(1, ge=1, le=30, description="Horizon temporel en jours"),
    method: str = Query("monte_carlo", pattern="^(monte_carlo|historical|parametric)$", description="Méthode de calcul"),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Calcule la VaR du portefeuille."""
    try:
        var_result = await risk_service.calculate_portfolio_var(
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=method
        )

        return ApiResponse(
            success=True,
            data=var_result,
            message=f"VaR calculation completed using {method} method"
        )
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stress-test", response_model=ApiResponse)
async def run_stress_test(
    scenario: str = Query(..., description="Scénario de stress test"),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Exécute un stress test sur le portefeuille."""
    try:
        # Valider le scénario
        valid_scenarios = ["market_crash", "volatility_spike", "liquidity_crisis", "correlation_breakdown", "custom"]
        if scenario not in valid_scenarios:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scenario. Valid options: {valid_scenarios}"
            )

        stress_result = await risk_service.run_stress_test(scenario)

        return ApiResponse(
            success=True,
            data=stress_result,
            message=f"Stress test '{scenario}' completed"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/limits", response_model=ApiResponse)
async def get_risk_limits(
    risk_service: RiskService = Depends(get_risk_service)
):
    """Récupère les limites de risque configurées."""
    try:
        limits = await risk_service.get_risk_limits()

        return ApiResponse(
            success=True,
            data=limits,
            message="Risk limits retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/limits", response_model=ApiResponse)
async def update_risk_limits(
    request: RiskConfigRequest,
    risk_service: RiskService = Depends(get_risk_service)
):
    """Met à jour les limites de risque."""
    try:
        updated_limits = await risk_service.update_risk_limits(request)

        return ApiResponse(
            success=True,
            data=updated_limits,
            message="Risk limits updated successfully"
        )
    except Exception as e:
        logger.error(f"Error updating risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=ApiResponse)
async def get_risk_alerts(
    active_only: bool = Query(True, description="Seulement les alertes actives"),
    severity: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$", description="Niveau de sévérité"),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Récupère les alertes de risque."""
    try:
        alerts = await risk_service.get_risk_alerts(
            active_only=active_only,
            severity=severity
        )

        return ApiResponse(
            success=True,
            data=alerts,
            message=f"Risk alerts retrieved ({len(alerts)} alerts)"
        )
    except Exception as e:
        logger.error(f"Error fetching risk alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge", response_model=ApiResponse)
async def acknowledge_alert(
    alert_id: str,
    risk_service: RiskService = Depends(get_risk_service)
):
    """Accuse réception d'une alerte de risque."""
    try:
        acknowledged_alert = await risk_service.acknowledge_alert(alert_id)
        if not acknowledged_alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return ApiResponse(
            success=True,
            data=acknowledged_alert,
            message=f"Alert {alert_id} acknowledged"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exposure", response_model=ApiResponse)
async def get_risk_exposure(
    by_asset: bool = Query(True, description="Exposition par asset"),
    by_sector: bool = Query(False, description="Exposition par secteur"),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Récupère l'exposition aux risques."""
    try:
        exposure = await risk_service.get_risk_exposure(
            by_asset=by_asset,
            by_sector=by_sector
        )

        return ApiResponse(
            success=True,
            data=exposure,
            message="Risk exposure retrieved"
        )
    except Exception as e:
        logger.error(f"Error fetching risk exposure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlation", response_model=ApiResponse)
async def get_correlation_matrix(
    lookback_days: int = Query(30, ge=7, le=365, description="Période de lookback en jours"),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Récupère la matrice de corrélation des positions."""
    try:
        correlation_matrix = await risk_service.get_correlation_matrix(lookback_days)

        return ApiResponse(
            success=True,
            data=correlation_matrix,
            message=f"Correlation matrix calculated for {lookback_days} days"
        )
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/concentration", response_model=ApiResponse)
async def get_concentration_risk(
    risk_service: RiskService = Depends(get_risk_service)
):
    """Récupère l'analyse du risque de concentration."""
    try:
        concentration = await risk_service.get_concentration_risk()

        return ApiResponse(
            success=True,
            data=concentration,
            message="Concentration risk analysis retrieved"
        )
    except Exception as e:
        logger.error(f"Error analyzing concentration risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/liquidity", response_model=ApiResponse)
async def get_liquidity_risk(
    time_horizon: str = Query("1d", pattern="^(1h|4h|1d|1w)$", description="Horizon de liquidation"),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Récupère l'analyse du risque de liquidité."""
    try:
        liquidity_risk = await risk_service.get_liquidity_risk(time_horizon)

        return ApiResponse(
            success=True,
            data=liquidity_risk,
            message=f"Liquidity risk analysis for {time_horizon} retrieved"
        )
    except Exception as e:
        logger.error(f"Error analyzing liquidity risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency-stop", response_model=ApiResponse)
async def emergency_stop(
    reason: str = Query(..., description="Raison de l'arrêt d'urgence"),
    confirm: bool = Query(False, description="Confirmation required"),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Déclenche un arrêt d'urgence du trading."""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirmation required for emergency stop (set confirm=true)"
            )

        emergency_result = await risk_service.trigger_emergency_stop(reason)

        return ApiResponse(
            success=True,
            data=emergency_result,
            message="Emergency stop triggered successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtesting/risk", response_model=ApiResponse)
async def get_backtesting_risk_metrics(
    backtest_id: str = Query(..., description="ID du backtest"),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Récupère les métriques de risque d'un backtest."""
    try:
        risk_metrics = await risk_service.get_backtesting_risk_metrics(backtest_id)
        if not risk_metrics:
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        return ApiResponse(
            success=True,
            data=risk_metrics,
            message=f"Risk metrics for backtest {backtest_id} retrieved"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching backtest risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scenario-analysis", response_model=ApiResponse)
async def run_scenario_analysis(
    scenarios: str = Query(..., description="Scénarios séparés par des virgules"),
    risk_service: RiskService = Depends(get_risk_service)
):
    """Exécute une analyse de scénarios multiples."""
    try:
        scenario_list = [s.strip() for s in scenarios.split(",")]
        if len(scenario_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 scenarios allowed")

        analysis_result = await risk_service.run_scenario_analysis(scenario_list)

        return ApiResponse(
            success=True,
            data=analysis_result,
            message=f"Scenario analysis completed for {len(scenario_list)} scenarios"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running scenario analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))