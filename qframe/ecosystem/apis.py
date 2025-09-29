"""
QFrame Public API System
========================

RESTful API infrastructure for external access to framework capabilities,
with authentication, rate limiting, and comprehensive endpoint management.
"""

import asyncio
import hashlib
import hmac
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from qframe.core.container import injectable
from qframe.core.config import FrameworkConfig
from qframe.core.interfaces import Strategy, DataProvider


class APIScope(str, Enum):
    """API access scopes."""
    READ_MARKET_DATA = "read:market_data"
    READ_STRATEGIES = "read:strategies"
    READ_PORTFOLIO = "read:portfolio"
    READ_ANALYTICS = "read:analytics"
    WRITE_STRATEGIES = "write:strategies"
    WRITE_PORTFOLIO = "write:portfolio"
    EXECUTE_TRADES = "execute:trades"
    ADMIN = "admin"


class APIKeyType(str, Enum):
    """API key types with different access levels."""
    PUBLIC = "public"        # Read-only access to public data
    PRIVATE = "private"      # Full access to user data
    WEBHOOK = "webhook"      # Webhook endpoints only
    SYSTEM = "system"        # Internal system access


class RateLimitType(str, Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    limit_type: RateLimitType = RateLimitType.SLIDING_WINDOW


@dataclass
class APIKey:
    """API key with metadata and permissions."""
    key_id: str
    key_secret: str
    user_id: str
    key_type: APIKeyType
    scopes: List[APIScope]
    rate_limit: RateLimitRule

    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True

    # Usage statistics
    total_requests: int = 0
    failed_requests: int = 0

    # Restrictions
    allowed_ips: Optional[List[str]] = None
    allowed_origins: Optional[List[str]] = None


@dataclass
class APIEndpoint:
    """API endpoint configuration."""
    path: str
    method: str
    handler: Callable
    required_scopes: List[APIScope]
    rate_limit: Optional[RateLimitRule] = None
    requires_auth: bool = True

    # Documentation
    summary: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Validation
    request_model: Optional[type] = None
    response_model: Optional[type] = None

    # Caching
    cache_ttl: Optional[int] = None
    cache_key_func: Optional[Callable] = None


class APIAuthentication:
    """API authentication and authorization system."""

    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_api_key(self,
                      user_id: str,
                      key_type: APIKeyType,
                      scopes: List[APIScope],
                      rate_limit: RateLimitRule,
                      expires_in_days: Optional[int] = None) -> APIKey:
        """Create a new API key."""
        key_id = f"qf_{key_type.value}_{uuid4().hex[:16]}"
        key_secret = self._generate_secret()

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_secret=key_secret,
            user_id=user_id,
            key_type=key_type,
            scopes=scopes,
            rate_limit=rate_limit,
            expires_at=expires_at
        )

        self.api_keys[key_id] = api_key
        return api_key

    def validate_api_key(self, key_id: str, signature: str, timestamp: str, request_data: str) -> Optional[APIKey]:
        """Validate API key and signature."""
        api_key = self.api_keys.get(key_id)
        if not api_key or not api_key.is_active:
            return None

        # Check expiration
        if api_key.expires_at and datetime.now() > api_key.expires_at:
            return None

        # Validate timestamp (prevent replay attacks)
        request_time = datetime.fromtimestamp(float(timestamp))
        if abs((datetime.now() - request_time).total_seconds()) > 300:  # 5 minutes
            return None

        # Validate signature
        expected_signature = self._generate_signature(
            api_key.key_secret, timestamp, request_data
        )

        if not hmac.compare_digest(signature, expected_signature):
            return None

        # Update usage
        api_key.last_used_at = datetime.now()
        api_key.total_requests += 1

        return api_key

    def has_scope(self, api_key: APIKey, required_scope: APIScope) -> bool:
        """Check if API key has required scope."""
        return required_scope in api_key.scopes or APIScope.ADMIN in api_key.scopes

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False
            return True
        return False

    def _generate_secret(self) -> str:
        """Generate a secure API key secret."""
        return hashlib.sha256(uuid4().bytes).hexdigest()

    def _generate_signature(self, secret: str, timestamp: str, data: str) -> str:
        """Generate HMAC signature for request validation."""
        message = f"{timestamp}{data}"
        return hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()


class RateLimiter:
    """Advanced rate limiting system with multiple strategies."""

    def __init__(self):
        self.windows: Dict[str, deque] = defaultdict(deque)
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def is_allowed(self, key: str, rule: RateLimitRule, client_ip: str = "") -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limit."""
        now = time.time()

        if rule.limit_type == RateLimitType.SLIDING_WINDOW:
            return self._sliding_window_check(key, rule, now)
        elif rule.limit_type == RateLimitType.TOKEN_BUCKET:
            return self._token_bucket_check(key, rule, now)
        elif rule.limit_type == RateLimitType.ADAPTIVE:
            return self._adaptive_check(key, rule, now, client_ip)
        else:  # FIXED_WINDOW
            return self._fixed_window_check(key, rule, now)

    def _sliding_window_check(self, key: str, rule: RateLimitRule, now: float) -> tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting."""
        window = self.windows[key]

        # Remove old requests
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400

        while window and window[0] < minute_ago:
            window.popleft()

        # Count requests in different windows
        minute_requests = sum(1 for req_time in window if req_time > minute_ago)
        hour_requests = sum(1 for req_time in window if req_time > hour_ago)
        day_requests = sum(1 for req_time in window if req_time > day_ago)

        # Check limits
        if (minute_requests >= rule.requests_per_minute or
            hour_requests >= rule.requests_per_hour or
            day_requests >= rule.requests_per_day):

            return False, {
                'minute_requests': minute_requests,
                'hour_requests': hour_requests,
                'day_requests': day_requests,
                'reset_time': int(now + 60)
            }

        # Allow request
        window.append(now)
        return True, {
            'minute_requests': minute_requests + 1,
            'hour_requests': hour_requests + 1,
            'day_requests': day_requests + 1,
            'reset_time': int(now + 60)
        }

    def _token_bucket_check(self, key: str, rule: RateLimitRule, now: float) -> tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting."""
        bucket = self.buckets[key]

        if 'tokens' not in bucket:
            bucket['tokens'] = rule.burst_limit
            bucket['last_refill'] = now

        # Refill tokens
        time_passed = now - bucket['last_refill']
        tokens_to_add = time_passed * (rule.requests_per_minute / 60.0)
        bucket['tokens'] = min(rule.burst_limit, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now

        # Check if token available
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True, {
                'tokens_remaining': int(bucket['tokens']),
                'refill_rate': rule.requests_per_minute / 60.0
            }

        return False, {
            'tokens_remaining': 0,
            'retry_after': int((1 - bucket['tokens']) / (rule.requests_per_minute / 60.0))
        }

    def _fixed_window_check(self, key: str, rule: RateLimitRule, now: float) -> tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting."""
        window_start = int(now // 60) * 60  # 1-minute windows
        window_key = f"{key}:{window_start}"

        if window_key not in self.buckets:
            self.buckets[window_key] = {'count': 0}

        bucket = self.buckets[window_key]

        if bucket['count'] >= rule.requests_per_minute:
            return False, {
                'requests_in_window': bucket['count'],
                'window_start': window_start,
                'reset_time': window_start + 60
            }

        bucket['count'] += 1
        return True, {
            'requests_in_window': bucket['count'],
            'window_start': window_start,
            'reset_time': window_start + 60
        }

    def _adaptive_check(self, key: str, rule: RateLimitRule, now: float, client_ip: str) -> tuple[bool, Dict[str, Any]]:
        """Adaptive rate limiting based on system load and client behavior."""
        # Basic adaptive logic - in production, this would be more sophisticated
        base_limit = rule.requests_per_minute

        # Adjust based on client behavior (simplified)
        if client_ip:
            client_key = f"client:{client_ip}"
            client_window = self.windows[client_key]

            # Count recent requests from this client
            minute_ago = now - 60
            recent_requests = sum(1 for req_time in client_window if req_time > minute_ago)

            # Reduce limit for high-frequency clients
            if recent_requests > base_limit * 0.8:
                adjusted_rule = RateLimitRule(
                    requests_per_minute=int(base_limit * 0.5),
                    requests_per_hour=rule.requests_per_hour,
                    requests_per_day=rule.requests_per_day,
                    burst_limit=rule.burst_limit
                )
                return self._sliding_window_check(key, adjusted_rule, now)

        return self._sliding_window_check(key, rule, now)


@injectable
class PublicAPIManager:
    """Main API management system for external access."""

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.app = FastAPI(
            title="QFrame Quantitative Trading API",
            description="Professional API for quantitative trading framework",
            version="1.0.0"
        )

        self.auth = APIAuthentication()
        self.rate_limiter = RateLimiter()
        self.endpoints: Dict[str, APIEndpoint] = {}

        # Setup middleware
        self._setup_middleware()

        # Initialize core endpoints
        self._initialize_core_endpoints()

    def _setup_middleware(self) -> None:
        """Setup FastAPI middleware."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=getattr(self.config, 'api_allowed_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Rate limiting middleware
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            # Extract API key for rate limiting
            api_key = await self._extract_api_key(request)
            if api_key:
                rate_limit_key = f"api_key:{api_key.key_id}"
                allowed, info = self.rate_limiter.is_allowed(
                    rate_limit_key,
                    api_key.rate_limit,
                    request.client.host if request.client else ""
                )

                if not allowed:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "details": info
                        },
                        headers={
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(info.get('reset_time', 0)),
                            "Retry-After": str(info.get('retry_after', 60))
                        }
                    )

            response = await call_next(request)
            return response

    def _initialize_core_endpoints(self) -> None:
        """Initialize core API endpoints."""

        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        # API info
        @self.app.get("/api/info")
        async def api_info():
            return {
                "name": "QFrame API",
                "version": "1.0.0",
                "endpoints": len(self.endpoints),
                "authentication": "API Key + HMAC",
                "rate_limiting": "Sliding Window"
            }

        # Market data endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/market/ohlcv/{symbol}",
            method="GET",
            handler=self._get_market_data,
            required_scopes=[APIScope.READ_MARKET_DATA],
            summary="Get OHLCV market data",
            description="Retrieve historical OHLCV data for a symbol"
        ))

        # Strategy endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/strategies",
            method="GET",
            handler=self._list_strategies,
            required_scopes=[APIScope.READ_STRATEGIES],
            summary="List available strategies",
            description="Get list of all available trading strategies"
        ))

        self.register_endpoint(APIEndpoint(
            path="/api/v1/strategies/{strategy_id}/signals",
            method="GET",
            handler=self._get_strategy_signals,
            required_scopes=[APIScope.READ_STRATEGIES],
            summary="Get strategy signals",
            description="Retrieve latest signals from a specific strategy"
        ))

        # Portfolio endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/portfolio/positions",
            method="GET",
            handler=self._get_portfolio_positions,
            required_scopes=[APIScope.READ_PORTFOLIO],
            summary="Get portfolio positions",
            description="Retrieve current portfolio positions"
        ))

        # Analytics endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/analytics/performance",
            method="GET",
            handler=self._get_performance_analytics,
            required_scopes=[APIScope.READ_ANALYTICS],
            summary="Get performance analytics",
            description="Retrieve portfolio performance metrics"
        ))

    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """Register a new API endpoint."""
        endpoint_key = f"{endpoint.method}:{endpoint.path}"
        self.endpoints[endpoint_key] = endpoint

        # Register with FastAPI
        if endpoint.method == "GET":
            self.app.get(endpoint.path,
                         summary=endpoint.summary,
                         description=endpoint.description,
                         tags=endpoint.tags)(endpoint.handler)
        elif endpoint.method == "POST":
            self.app.post(endpoint.path,
                          summary=endpoint.summary,
                          description=endpoint.description,
                          tags=endpoint.tags)(endpoint.handler)
        elif endpoint.method == "PUT":
            self.app.put(endpoint.path,
                         summary=endpoint.summary,
                         description=endpoint.description,
                         tags=endpoint.tags)(endpoint.handler)
        elif endpoint.method == "DELETE":
            self.app.delete(endpoint.path,
                            summary=endpoint.summary,
                            description=endpoint.description,
                            tags=endpoint.tags)(endpoint.handler)

    async def _extract_api_key(self, request: Request) -> Optional[APIKey]:
        """Extract and validate API key from request."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        try:
            # Extract key components
            key_id = request.headers.get("X-API-Key")
            timestamp = request.headers.get("X-Timestamp")
            signature = request.headers.get("X-Signature")

            if not all([key_id, timestamp, signature]):
                return None

            # Get request body for signature validation
            body = await request.body()
            request_data = body.decode('utf-8') if body else ""

            return self.auth.validate_api_key(key_id, signature, timestamp, request_data)

        except Exception:
            return None

    def create_api_key(self,
                      user_id: str,
                      key_type: APIKeyType,
                      scopes: List[APIScope],
                      rate_limit: Optional[RateLimitRule] = None) -> APIKey:
        """Create a new API key for a user."""
        if not rate_limit:
            # Default rate limits by key type
            if key_type == APIKeyType.PUBLIC:
                rate_limit = RateLimitRule(60, 1000, 10000, 10)
            elif key_type == APIKeyType.PRIVATE:
                rate_limit = RateLimitRule(300, 5000, 50000, 50)
            elif key_type == APIKeyType.SYSTEM:
                rate_limit = RateLimitRule(1000, 20000, 200000, 100)
            else:
                rate_limit = RateLimitRule(100, 2000, 20000, 20)

        return self.auth.create_api_key(user_id, key_type, scopes, rate_limit)

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the API server."""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    # API endpoint handlers
    async def _get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        """Get market data endpoint handler."""
        # This would integrate with the actual data providers
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": "Market data would be here",
            "timestamp": datetime.now().isoformat()
        }

    async def _list_strategies(self):
        """List strategies endpoint handler."""
        # This would integrate with the strategy registry
        return {
            "strategies": [
                {"id": "dmn_lstm", "name": "DMN LSTM Strategy", "type": "ml"},
                {"id": "mean_reversion", "name": "Mean Reversion Strategy", "type": "statistical"},
                {"id": "funding_arbitrage", "name": "Funding Arbitrage Strategy", "type": "arbitrage"},
                {"id": "rl_alpha", "name": "RL Alpha Strategy", "type": "reinforcement_learning"}
            ]
        }

    async def _get_strategy_signals(self, strategy_id: str):
        """Get strategy signals endpoint handler."""
        # This would integrate with the actual strategy execution
        return {
            "strategy_id": strategy_id,
            "signals": [],
            "timestamp": datetime.now().isoformat()
        }

    async def _get_portfolio_positions(self):
        """Get portfolio positions endpoint handler."""
        # This would integrate with the portfolio manager
        return {
            "positions": [],
            "total_value": 0.0,
            "timestamp": datetime.now().isoformat()
        }

    async def _get_performance_analytics(self):
        """Get performance analytics endpoint handler."""
        # This would integrate with the analytics engine
        return {
            "metrics": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            },
            "timestamp": datetime.now().isoformat()
        }


# Pydantic models for request/response validation
class CreateAPIKeyRequest(BaseModel):
    user_id: str
    key_type: APIKeyType
    scopes: List[APIScope]
    expires_in_days: Optional[int] = None


class CreateAPIKeyResponse(BaseModel):
    key_id: str
    key_secret: str
    scopes: List[APIScope]
    rate_limit: Dict[str, Any]
    expires_at: Optional[datetime]


class MarketDataRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, le=1000)


class StrategySignalsResponse(BaseModel):
    strategy_id: str
    signals: List[Dict[str, Any]]
    timestamp: datetime


class PortfolioPositionsResponse(BaseModel):
    positions: List[Dict[str, Any]]
    total_value: float
    currency: str = "USD"
    timestamp: datetime


class PerformanceMetrics(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float


class PerformanceAnalyticsResponse(BaseModel):
    metrics: PerformanceMetrics
    period_start: datetime
    period_end: datetime
    timestamp: datetime