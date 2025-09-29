"""
🌐 QFrame FastAPI Backend
API Backend pour l'interface Streamlit et services externes
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging
from contextlib import asynccontextmanager

# QFrame imports
from qframe.core.config import FrameworkConfig
from qframe.core.container import get_container
from qframe.api.routers import market_data, orders, positions, risk, strategies
from qframe.api.services.real_time_service import RealTimeService
from qframe.api.services.market_data_service import MarketDataService
from qframe.api.middleware.auth import AuthMiddleware
from qframe.api.models.responses import HealthResponse
from qframe.api.services_registration import register_api_services

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration globale
config = FrameworkConfig()
container = get_container()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    # Startup
    logger.info("🚀 Starting QFrame API Backend...")

    # Enregistrer les services API dans le container DI
    register_api_services()

    # Initialiser les services
    real_time_service = container.resolve(RealTimeService)
    market_data_service = container.resolve(MarketDataService)

    # Démarrer les services en arrière-plan
    await real_time_service.start()
    await market_data_service.start()

    logger.info("✅ QFrame API Backend started successfully")

    yield

    # Shutdown
    logger.info("🛑 Shutting down QFrame API Backend...")
    await real_time_service.stop()
    await market_data_service.stop()
    logger.info("✅ QFrame API Backend stopped")


# Application FastAPI
app = FastAPI(
    title="QFrame API",
    description="Backend API pour le framework de trading quantitatif QFrame",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],  # Streamlit + React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware d'authentification
app.add_middleware(AuthMiddleware)

# Routes principales
app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["Market Data"])
app.include_router(orders.router, prefix="/api/v1/orders", tags=["Orders"])
app.include_router(positions.router, prefix="/api/v1/positions", tags=["Positions"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["Risk Management"])
app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["Strategies"])


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Point d'entrée principal de l'API."""
    return {
        "message": "🚀 QFrame API Backend",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow(),
        "environment": config.environment.value,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de vérification de santé."""
    try:
        # Vérifier les services critiques
        real_time_service = container.resolve(RealTimeService)
        market_data_service = container.resolve(MarketDataService)

        services_status = {
            "real_time_service": real_time_service.is_healthy(),
            "market_data_service": market_data_service.is_healthy(),
            "database": True,  # TODO: Vérifier la DB
            "redis": True      # TODO: Vérifier Redis
        }

        overall_status = all(services_status.values())

        return HealthResponse(
            status="healthy" if overall_status else "degraded",
            timestamp=datetime.utcnow(),
            services=services_status,
            version="1.0.0",
            uptime=_get_uptime()
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            services={},
            version="1.0.0",
            uptime=_get_uptime(),
            error=str(e)
        )


@app.get("/api/v1/status")
async def api_status():
    """Statut détaillé de l'API."""
    return {
        "api_version": "1.0.0",
        "framework_version": config.app_name,
        "environment": config.environment.value,
        "timestamp": datetime.utcnow(),
        "active_connections": 0,  # TODO: Compter les connexions WebSocket
        "memory_usage": _get_memory_usage(),
        "endpoints": [
            "/api/v1/market-data",
            "/api/v1/orders",
            "/api/v1/positions",
            "/api/v1/risk",
            "/api/v1/strategies"
        ]
    }


@app.get("/api/v1/config")
async def get_config():
    """Configuration publique de l'API."""
    return {
        "environment": config.environment.value,
        "features": {
            "real_time_data": True,
            "paper_trading": True,
            "live_trading": config.environment.value == "PRODUCTION",
            "backtesting": True,
            "risk_management": True
        },
        "limits": {
            "max_positions": 100,
            "max_orders_per_minute": 120,
            "max_websocket_connections": 50
        },
        "supported_exchanges": ["binance", "coinbase", "kraken"],
        "supported_symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT"]
    }


# Handlers d'erreurs globaux
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Gestionnaire d'erreurs HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gestionnaire d'erreurs général."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal Server Error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


def _get_uptime() -> str:
    """Calcule l'uptime de l'application."""
    # TODO: Implémenter le calcul réel d'uptime
    return "0d 0h 0m"


def _get_memory_usage() -> Dict[str, Any]:
    """Retourne l'utilisation mémoire."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss": memory_info.rss,  # Resident Set Size
        "vms": memory_info.vms,  # Virtual Memory Size
        "percent": process.memory_percent(),
        "available": psutil.virtual_memory().available
    }


# WebSocket pour données temps réel
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """Endpoint WebSocket pour données temps réel."""
    from qframe.api.websocket.real_time_handler import RealTimeWebSocketHandler

    handler = container.resolve(RealTimeWebSocketHandler)
    await handler.handle_connection(websocket)


# Point d'entrée pour le serveur
def create_app() -> FastAPI:
    """Factory pour créer l'application FastAPI."""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Lance le serveur API."""
    uvicorn.run(
        "qframe.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    run_server(reload=True)