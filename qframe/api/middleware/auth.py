"""
🔐 Authentication Middleware
Middleware d'authentification pour l'API
"""

import logging
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware d'authentification basique."""

    def __init__(self, app, require_auth: bool = False):
        super().__init__(app)
        self.require_auth = require_auth
        self.public_paths = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traite les requêtes d'authentification."""

        # Vérifier si le chemin est public
        if request.url.path in self.public_paths:
            return await call_next(request)

        # Pour le moment, on accepte toutes les requêtes
        # En production, ajouter une vraie logique d'auth
        if self.require_auth:
            # Vérifier l'en-tête Authorization
            auth_header = request.headers.get("Authorization")

            if not auth_header:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authentication required"}
                )

            # Ici, on ajouterait la validation du token
            # Pour la démo, on accepte n'importe quel token
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid authentication format"}
                )

        # Continuer le traitement
        response = await call_next(request)

        # Ajouter des en-têtes de sécurité
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Content-Type-Options"] = "nosniff"

        return response