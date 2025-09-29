"""
Infrastructure Layer: API Middleware
====================================

Middleware pour l'API incluant CORS, rate limiting, authentification,
logging et gestion des erreurs.
"""

import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware de limitation de taux (rate limiting).
    Limite le nombre de requêtes par IP et par utilisateur.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size

        # Stockage des requêtes par IP
        self._requests_by_ip: Dict[str, List[datetime]] = defaultdict(list)
        self._burst_by_ip: Dict[str, List[datetime]] = defaultdict(list)

        # Stockage des requêtes par utilisateur
        self._requests_by_user: Dict[str, List[datetime]] = defaultdict(list)

        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traiter la requête avec rate limiting"""
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)

        # Nettoyer les anciennes entrées
        self._cleanup_old_requests()

        # Vérifier les limites
        if not self._check_rate_limits(client_ip, user_id):
            self.logger.warning(f"Rate limit exceeded for IP {client_ip}")
            self.metrics.collector.increment_counter(
                "api.rate_limit_exceeded",
                labels={"client_ip": client_ip}
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )

        # Enregistrer la requête
        now = datetime.utcnow()
        self._requests_by_ip[client_ip].append(now)
        if user_id:
            self._requests_by_user[user_id].append(now)

        # Continuer la requête
        response = await call_next(request)

        # Ajouter headers de rate limiting
        remaining = max(0, self.requests_per_minute - len(self._requests_by_ip[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int((now + timedelta(minutes=1)).timestamp()))

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Obtenir l'IP du client"""
        # Vérifier les headers de proxy
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _get_user_id(self, request: Request) -> Optional[str]:
        """Obtenir l'ID utilisateur si authentifié"""
        # Extraire du token JWT si présent
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from .auth import get_auth_service
                auth_service = get_auth_service()
                token = auth_header.split(" ")[1]
                user_data = auth_service.jwt_manager.get_user_from_token(token)
                return user_data.get("user_id")
            except Exception:
                pass
        return None

    def _check_rate_limits(self, client_ip: str, user_id: Optional[str]) -> bool:
        """Vérifier les limites de taux"""
        now = datetime.utcnow()

        # Limites par IP
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        ip_requests_minute = [
            req for req in self._requests_by_ip[client_ip]
            if req > minute_ago
        ]
        ip_requests_hour = [
            req for req in self._requests_by_ip[client_ip]
            if req > hour_ago
        ]

        # Vérifier burst
        burst_window = now - timedelta(seconds=10)
        burst_requests = [
            req for req in self._requests_by_ip[client_ip]
            if req > burst_window
        ]

        if len(burst_requests) >= self.burst_size:
            return False

        if len(ip_requests_minute) >= self.requests_per_minute:
            return False

        if len(ip_requests_hour) >= self.requests_per_hour:
            return False

        # Limites par utilisateur (plus généreuses)
        if user_id:
            user_requests_minute = [
                req for req in self._requests_by_user[user_id]
                if req > minute_ago
            ]
            if len(user_requests_minute) >= self.requests_per_minute * 2:
                return False

        return True

    def _cleanup_old_requests(self):
        """Nettoyer les anciennes requêtes"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)

        # Nettoyer par IP
        for ip in list(self._requests_by_ip.keys()):
            self._requests_by_ip[ip] = [
                req for req in self._requests_by_ip[ip]
                if req > cutoff_time
            ]
            if not self._requests_by_ip[ip]:
                del self._requests_by_ip[ip]

        # Nettoyer par utilisateur
        for user_id in list(self._requests_by_user.keys()):
            self._requests_by_user[user_id] = [
                req for req in self._requests_by_user[user_id]
                if req > cutoff_time
            ]
            if not self._requests_by_user[user_id]:
                del self._requests_by_user[user_id]


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware d'authentification.
    Vérifie les tokens JWT et injecte l'utilisateur dans le contexte.
    """

    def __init__(self, app, excluded_paths: List[str] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/docs", "/redoc", "/openapi.json", "/health", "/api/v1/auth/login"
        ]
        self.logger = LoggerFactory.get_logger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traiter la requête avec authentification"""
        # Vérifier si le path est exclu
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        # Vérifier le token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Missing or invalid authorization header"}
            )

        try:
            from .auth import get_auth_service
            auth_service = get_auth_service()
            token = auth_header.split(" ")[1]
            user_data = auth_service.jwt_manager.get_user_from_token(token)

            # Injecter l'utilisateur dans le state de la requête
            request.state.user_id = user_data["user_id"]
            request.state.username = user_data["username"]
            request.state.roles = user_data["roles"]
            request.state.permissions = user_data["permissions"]

        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail}
            )
        except Exception as e:
            self.logger.error(f"Auth middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Authentication failed"}
            )

        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware de logging avancé.
    Log toutes les requêtes avec détails complets.
    """

    def __init__(self, app, log_request_body: bool = False, log_response_body: bool = False):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traiter la requête avec logging complet"""
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Informations de la requête
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        request_size = int(request.headers.get("Content-Length", 0))

        # Logger la requête entrante
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "request_size": request_size,
            "headers": dict(request.headers) if self.log_request_body else None
        }

        # Log request body si activé
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    log_data["request_body"] = body.decode("utf-8")[:1000]  # Limiter la taille
            except Exception as e:
                log_data["request_body_error"] = str(e)

        self.logger.info("HTTP request received", **log_data)

        # Traiter la requête
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Logger la réponse
            response_log_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time_ms": process_time * 1000,
                "response_size": len(response.body) if hasattr(response, 'body') else 0
            }

            # Log response body si activé
            if self.log_response_body and hasattr(response, 'body'):
                try:
                    response_log_data["response_body"] = response.body.decode("utf-8")[:1000]
                except Exception:
                    pass

            self.logger.info("HTTP request completed", **response_log_data)

            # Métriques
            self.metrics.collector.increment_counter(
                "http.requests_total",
                labels={
                    "method": request.method,
                    "status": str(response.status_code),
                    "endpoint": request.url.path
                }
            )

            self.metrics.collector.record_histogram(
                "http.request_duration_seconds",
                process_time,
                labels={
                    "method": request.method,
                    "endpoint": request.url.path
                }
            )

            # Ajouter headers de debug
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"

            return response

        except Exception as e:
            process_time = time.time() - start_time

            # Logger l'erreur
            self.logger.error(
                "HTTP request failed",
                request_id=request_id,
                error=str(e),
                process_time_ms=process_time * 1000
            )

            # Métriques d'erreur
            self.metrics.collector.increment_counter(
                "http.requests_total",
                labels={
                    "method": request.method,
                    "status": "500",
                    "endpoint": request.url.path
                }
            )

            raise

    def _get_client_ip(self, request: Request) -> str:
        """Obtenir l'IP du client"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware de gestion d'erreurs globales.
    Capture et formate toutes les erreurs non gérées.
    """

    def __init__(self, app, include_details: bool = False):
        super().__init__(app)
        self.include_details = include_details
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traiter la requête avec gestion d'erreur"""
        try:
            return await call_next(request)

        except HTTPException as e:
            # HTTPException est géré par FastAPI, on la laisse passer
            raise

        except Exception as e:
            # Erreur non gérée
            request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

            self.logger.error(
                f"Unhandled exception in {request.method} {request.url.path}",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )

            # Métriques
            self.metrics.collector.increment_counter(
                "http.server_errors_total",
                labels={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "error_type": type(e).__name__
                }
            )

            # Réponse d'erreur
            error_response = {
                "error": "Internal server error",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            if self.include_details:
                error_response["details"] = str(e)
                error_response["error_type"] = type(e).__name__

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response
            )


class CORSMiddleware:
    """
    Configuration CORS pour l'API.
    Wrapper autour de FastAPI CORSMiddleware avec configuration par défaut.
    """

    @staticmethod
    def create_middleware(
        allow_origins: List[str] = None,
        allow_credentials: bool = True,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None
    ):
        """Créer le middleware CORS avec configuration par défaut"""
        return FastAPICORSMiddleware(
            allow_origins=allow_origins or ["*"],  # À restreindre en production
            allow_credentials=allow_credentials,
            allow_methods=allow_methods or ["*"],
            allow_headers=allow_headers or ["*"]
        )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware pour ajouter des headers de sécurité.
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Ajouter les headers de sécurité"""
        response = await call_next(request)

        # Headers de sécurité
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # CSP pour l'API (restrictif)
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"

        return response


# Fonctions utilitaires pour configurer tous les middleware
def setup_middleware(app, config: Dict[str, Any] = None):
    """Configurer tous les middleware sur l'app FastAPI"""
    config = config or {}

    # Sécurité
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS
    if config.get("enable_cors", True):
        app.add_middleware(
            FastAPICORSMiddleware,
            allow_origins=config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Rate limiting
    if config.get("enable_rate_limiting", True):
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=config.get("rate_limit_per_minute", 60),
            requests_per_hour=config.get("rate_limit_per_hour", 1000)
        )

    # Authentification (optionnel)
    if config.get("enable_auth_middleware", False):
        app.add_middleware(
            AuthMiddleware,
            excluded_paths=config.get("auth_excluded_paths", [])
        )

    # Logging
    if config.get("enable_logging_middleware", True):
        app.add_middleware(
            LoggingMiddleware,
            log_request_body=config.get("log_request_body", False),
            log_response_body=config.get("log_response_body", False)
        )

    # Gestion d'erreurs
    app.add_middleware(
        ErrorHandlingMiddleware,
        include_details=config.get("include_error_details", False)
    )