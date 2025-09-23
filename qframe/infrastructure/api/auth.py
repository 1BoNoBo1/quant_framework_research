"""
Infrastructure Layer: Authentication & Authorization
===================================================

Services d'authentification et d'autorisation pour l'API
avec JWT, RBAC et gestion des sessions.
"""

import hashlib
import jwt
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics


class UserRole(str, Enum):
    """Rôles utilisateur"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Permissions du système"""
    # Strategy permissions
    STRATEGY_CREATE = "strategy:create"
    STRATEGY_READ = "strategy:read"
    STRATEGY_UPDATE = "strategy:update"
    STRATEGY_DELETE = "strategy:delete"
    STRATEGY_EXECUTE = "strategy:execute"

    # Portfolio permissions
    PORTFOLIO_CREATE = "portfolio:create"
    PORTFOLIO_READ = "portfolio:read"
    PORTFOLIO_UPDATE = "portfolio:update"
    PORTFOLIO_DELETE = "portfolio:delete"

    # Order permissions
    ORDER_CREATE = "order:create"
    ORDER_READ = "order:read"
    ORDER_UPDATE = "order:update"
    ORDER_CANCEL = "order:cancel"

    # Market data permissions
    MARKET_DATA_READ = "market_data:read"
    MARKET_DATA_SUBSCRIBE = "market_data:subscribe"

    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"


@dataclass
class User:
    """Utilisateur du système"""
    user_id: str
    username: str
    email: str
    roles: Set[UserRole] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    # Métadonnées
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """Vérifier si l'utilisateur a une permission"""
        return permission in self.permissions or Permission.SYSTEM_ADMIN in self.permissions

    def has_role(self, role: UserRole) -> bool:
        """Vérifier si l'utilisateur a un rôle"""
        return role in self.roles

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": [role.value for role in self.roles],
            "permissions": [perm.value for perm in self.permissions],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.metadata
        }


@dataclass
class Session:
    """Session utilisateur"""
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        """Vérifier si la session a expiré"""
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active
        }


class JWTManager:
    """Gestionnaire des tokens JWT"""

    def __init__(
        self,
        secret_key: str = "your-secret-key",  # À configurer en production
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 60,
        refresh_token_expire_days: int = 30
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.logger = LoggerFactory.get_logger(__name__)

    def create_access_token(self, user: User) -> str:
        """Créer un token d'accès"""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "sub": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "iat": now,
            "exp": expire,
            "type": "access"
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def create_refresh_token(self, user: User) -> str:
        """Créer un token de rafraîchissement"""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "sub": user.user_id,
            "iat": now,
            "exp": expire,
            "type": "refresh"
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Vérifier et décoder un token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    def get_user_from_token(self, token: str) -> Dict[str, Any]:
        """Extraire les informations utilisateur du token"""
        payload = self.verify_token(token)

        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )

        return {
            "user_id": payload["sub"],
            "username": payload["username"],
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", [])
        }


class AuthenticationService:
    """Service d'authentification"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

        # Contexte de hachage des mots de passe
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Gestionnaire JWT
        self.jwt_manager = JWTManager()

        # Stockage en mémoire (à remplacer par une base de données)
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}

        # Créer des utilisateurs par défaut
        self._create_default_users()

    def _create_default_users(self):
        """Créer des utilisateurs par défaut pour les tests"""
        # Admin
        admin = User(
            user_id="admin",
            username="admin",
            email="admin@qframe.com",
            roles={UserRole.ADMIN},
            permissions=set(Permission)  # Toutes les permissions
        )
        self._users["admin"] = admin

        # Trader
        trader = User(
            user_id="trader",
            username="trader",
            email="trader@qframe.com",
            roles={UserRole.TRADER},
            permissions={
                Permission.STRATEGY_READ, Permission.STRATEGY_EXECUTE,
                Permission.PORTFOLIO_READ, Permission.PORTFOLIO_UPDATE,
                Permission.ORDER_CREATE, Permission.ORDER_READ, Permission.ORDER_UPDATE, Permission.ORDER_CANCEL,
                Permission.MARKET_DATA_READ, Permission.MARKET_DATA_SUBSCRIBE
            }
        )
        self._users["trader"] = trader

        # Analyst
        analyst = User(
            user_id="analyst",
            username="analyst",
            email="analyst@qframe.com",
            roles={UserRole.ANALYST},
            permissions={
                Permission.STRATEGY_READ, Permission.STRATEGY_CREATE, Permission.STRATEGY_UPDATE,
                Permission.PORTFOLIO_READ,
                Permission.ORDER_READ,
                Permission.MARKET_DATA_READ, Permission.MARKET_DATA_SUBSCRIBE,
                Permission.SYSTEM_MONITOR
            }
        )
        self._users["analyst"] = analyst

        # Viewer
        viewer = User(
            user_id="viewer",
            username="viewer",
            email="viewer@qframe.com",
            roles={UserRole.VIEWER},
            permissions={
                Permission.STRATEGY_READ,
                Permission.PORTFOLIO_READ,
                Permission.ORDER_READ,
                Permission.MARKET_DATA_READ
            }
        )
        self._users["viewer"] = viewer

    def hash_password(self, password: str) -> str:
        """Hacher un mot de passe"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Vérifier un mot de passe"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authentifier un utilisateur"""
        # Pour la démo, accepter le password = username
        if username in self._users and password == username:
            user = self._users[username]
            user.last_login = datetime.utcnow()

            self.logger.info(f"User authenticated: {username}")
            self.metrics.collector.increment_counter("auth.login_success", labels={"username": username})

            return user

        self.logger.warning(f"Authentication failed for user: {username}")
        self.metrics.collector.increment_counter("auth.login_failure", labels={"username": username})
        return None

    def create_session(self, user: User, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Session:
        """Créer une session"""
        session = Session(
            session_id=str(uuid.uuid4()),
            user_id=user.user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )

        self._sessions[session.session_id] = session

        self.logger.info(f"Session created for user {user.username}", session_id=session.session_id)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Obtenir une session"""
        session = self._sessions.get(session_id)
        if session and not session.is_expired and session.is_active:
            return session
        return None

    def invalidate_session(self, session_id: str):
        """Invalider une session"""
        if session_id in self._sessions:
            self._sessions[session_id].is_active = False
            self.logger.info(f"Session invalidated: {session_id}")

    def get_user(self, user_id: str) -> Optional[User]:
        """Obtenir un utilisateur par ID"""
        return self._users.get(user_id)

    def login(self, username: str, password: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Connexion complète"""
        user = self.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled"
            )

        # Créer session
        session = self.create_session(user, ip_address, user_agent)

        # Créer tokens
        access_token = self.jwt_manager.create_access_token(user)
        refresh_token = self.jwt_manager.create_refresh_token(user)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.jwt_manager.access_token_expire_minutes * 60,
            "user": user.to_dict(),
            "session_id": session.session_id
        }

    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Rafraîchir un token"""
        try:
            payload = self.jwt_manager.verify_token(refresh_token)
            if payload.get("type") != "refresh":
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

            user = self.get_user(payload["sub"])
            if not user or not user.is_active:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

            # Créer nouveau token d'accès
            access_token = self.jwt_manager.create_access_token(user)

            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.jwt_manager.access_token_expire_minutes * 60
            }

        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")


class AuthorizationService:
    """Service d'autorisation"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)

    def check_permission(self, user: User, permission: Permission) -> bool:
        """Vérifier une permission"""
        has_perm = user.has_permission(permission)

        self.logger.debug(
            f"Permission check: {permission.value}",
            user_id=user.user_id,
            has_permission=has_perm
        )

        return has_perm

    def require_permission(self, permission: Permission):
        """Décorateur pour exiger une permission"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Récupérer l'utilisateur depuis le contexte
                # (implémentation simplifiée)
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def check_resource_access(self, user: User, resource_type: str, resource_id: str, action: str) -> bool:
        """Vérifier l'accès à une ressource spécifique"""
        # Logique d'autorisation basée sur les ressources
        # Par exemple, un trader ne peut voir que ses propres portfolios

        if user.has_role(UserRole.ADMIN):
            return True

        # Logique métier spécifique
        if resource_type == "portfolio":
            # Vérifier si le portfolio appartient à l'utilisateur
            # (implémentation simplifiée)
            return True

        return False


# Dépendances FastAPI
security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Obtenir l'utilisateur actuel depuis le token JWT"""
    auth_service = get_auth_service()
    user_data = auth_service.jwt_manager.get_user_from_token(credentials.credentials)

    user = auth_service.get_user(user_data["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )

    return user


def require_permission(permission: Permission):
    """Dépendance pour exiger une permission"""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        auth_service = get_auth_service()
        if not auth_service.authorization_service.check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        return current_user
    return permission_checker


def require_role(role: UserRole):
    """Dépendance pour exiger un rôle"""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role.value}"
            )
        return current_user
    return role_checker


class AuthService:
    """Service d'authentification et d'autorisation unifié"""

    def __init__(self):
        self.authentication_service = AuthenticationService()
        self.authorization_service = AuthorizationService()
        self.jwt_manager = self.authentication_service.jwt_manager

    def login(self, username: str, password: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Connexion"""
        return self.authentication_service.login(username, password, ip_address, user_agent)

    def refresh_token(self, refresh_token: str):
        """Rafraîchir un token"""
        return self.authentication_service.refresh_token(refresh_token)

    def get_user(self, user_id: str) -> Optional[User]:
        """Obtenir un utilisateur"""
        return self.authentication_service.get_user(user_id)

    def check_permission(self, user: User, permission: Permission) -> bool:
        """Vérifier une permission"""
        return self.authorization_service.check_permission(user, permission)


# Instance globale
_global_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Obtenir l'instance globale du service d'authentification"""
    global _global_auth_service
    if _global_auth_service is None:
        _global_auth_service = AuthService()
    return _global_auth_service