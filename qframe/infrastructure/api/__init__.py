"""
Infrastructure Layer: API Services
==================================

Services API pour exposer les fonctionnalités du framework via REST, WebSocket et GraphQL.
Gestion de l'authentification, rate limiting, validation et documentation.
"""

from .rest import (
    FastAPIService,
    APIRouter,
    get_api_service
)

from .websocket import (
    WebSocketManager,
    ConnectionManager,
    get_websocket_manager
)

from .auth import (
    AuthenticationService,
    AuthorizationService,
    JWTManager,
    get_auth_service
)

from .middleware import (
    CORSMiddleware,
    RateLimitMiddleware,
    AuthMiddleware,
    LoggingMiddleware
)

__all__ = [
    # REST API
    'FastAPIService',
    'APIRouter',
    'get_api_service',

    # WebSocket
    'WebSocketManager',
    'ConnectionManager',
    'get_websocket_manager',

    # Authentication
    'AuthenticationService',
    'AuthorizationService',
    'JWTManager',
    'get_auth_service',

    # Middleware
    'CORSMiddleware',
    'RateLimitMiddleware',
    'AuthMiddleware',
    'LoggingMiddleware'
]