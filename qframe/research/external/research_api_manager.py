"""
Research API Manager
===================

Centralized management of research platform APIs.
Authentication, rate limiting, and request orchestration.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
import asyncio
import aiohttp
import json
import time
from pathlib import Path
import hashlib
import base64

from qframe.core.container import injectable
from qframe.core.config import FrameworkConfig

logger = logging.getLogger(__name__)


class APIAuthType(Enum):
    """Types d'authentification API"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"
    JWT = "jwt"


class RequestMethod(Enum):
    """Méthodes HTTP"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class RateLimitStrategy(Enum):
    """Stratégies de rate limiting"""
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


@dataclass
class APICredentials:
    """Identifiants API"""
    provider_id: str
    auth_type: APIAuthType

    # Différents types d'auth
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    bearer_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    # Headers personnalisés
    custom_headers: Dict[str, str] = field(default_factory=dict)

    # OAuth2
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None


@dataclass
class APIEndpoint:
    """Définition d'un endpoint API"""
    endpoint_id: str
    provider_id: str
    name: str
    description: str

    # Configuration URL
    base_url: str
    path: str
    method: RequestMethod = RequestMethod.GET

    # Paramètres
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)

    # Rate limiting spécifique
    rate_limit: Optional[int] = None
    rate_window: int = 60  # secondes

    # Response handling
    response_format: str = "json"  # json, xml, csv, text
    pagination_support: bool = False
    max_records_per_request: Optional[int] = None

    # Métadonnées
    documentation_url: Optional[str] = None
    cost_per_request: float = 0.0
    requires_subscription: bool = False


@dataclass
class APIProvider:
    """Fournisseur d'API de recherche"""
    provider_id: str
    name: str
    description: str
    base_url: str

    # Configuration générale
    default_auth_type: APIAuthType = APIAuthType.API_KEY
    global_rate_limit: int = 100  # requêtes par minute
    rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

    # Endpoints disponibles
    endpoints: Dict[str, APIEndpoint] = field(default_factory=dict)

    # Métadonnées
    documentation_url: Optional[str] = None
    status_page_url: Optional[str] = None
    support_email: Optional[str] = None
    pricing_url: Optional[str] = None

    # Configuration avancée
    retry_attempts: int = 3
    timeout_seconds: float = 30.0
    verify_ssl: bool = True


@dataclass
class RateLimiter:
    """Rate limiter pour API"""
    provider_id: str
    strategy: RateLimitStrategy
    limit: int
    window_seconds: int

    # État interne
    requests: List[datetime] = field(default_factory=list)
    tokens: float = 0.0
    last_refill: datetime = field(default_factory=datetime.utcnow)

    # Statistiques
    total_requests: int = 0
    rejected_requests: int = 0
    reset_time: Optional[datetime] = None


@dataclass
class APIRequest:
    """Requête API"""
    request_id: str
    provider_id: str
    endpoint_id: str
    method: RequestMethod

    # Paramètres
    url: str
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Union[str, Dict[str, Any]]] = None

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.utcnow)
    priority: int = 1  # 1=low, 5=high
    retry_count: int = 0
    timeout: float = 30.0


@dataclass
class APIResponse:
    """Réponse API"""
    request: APIRequest
    status_code: int
    headers: Dict[str, str]
    content: Union[str, bytes, Dict[str, Any]]

    # Timing
    response_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Métadonnées
    from_cache: bool = False
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None

    # Erreurs
    error: Optional[str] = None
    is_success: bool = True


@injectable
class ResearchAPIManager:
    """
    Gestionnaire centralisé des APIs de recherche.

    Fonctionnalités:
    - Gestion centralisée des identifiants
    - Rate limiting sophistiqué
    - Queue de requêtes avec priorité
    - Cache intelligent
    - Retry automatique avec backoff
    - Monitoring et métriques
    - Support multi-provider
    """

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.providers: Dict[str, APIProvider] = {}
        self.credentials: Dict[str, APICredentials] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}

        # Queue de requêtes
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_requests: Dict[str, APIRequest] = {}

        # Cache
        self.response_cache: Dict[str, APIResponse] = {}
        self.cache_ttl_seconds = 300  # 5 minutes par défaut

        # Session HTTP
        self.session: Optional[aiohttp.ClientSession] = None

        # Métriques
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "rate_limited_requests": 0,
            "by_provider": {},
            "by_endpoint": {},
            "response_times": []
        }

        # Worker pour traiter les requêtes
        self._request_worker_task: Optional[asyncio.Task] = None

        self._initialize_default_providers()
        logger.info("Research API Manager initialized")

    def _initialize_default_providers(self):
        """Initialise les providers par défaut"""

        # Semantic Scholar API
        semantic_scholar = APIProvider(
            provider_id="semantic_scholar",
            name="Semantic Scholar",
            description="AI-powered research tool for scientific literature",
            base_url="https://api.semanticscholar.org",
            default_auth_type=APIAuthType.NONE,
            global_rate_limit=100,
            documentation_url="https://api.semanticscholar.org/",
        )

        # Endpoints Semantic Scholar
        semantic_scholar.endpoints["paper_search"] = APIEndpoint(
            endpoint_id="paper_search",
            provider_id="semantic_scholar",
            name="Paper Search",
            description="Search for academic papers",
            base_url=semantic_scholar.base_url,
            path="/graph/v1/paper/search",
            method=RequestMethod.GET,
            required_params=["query"],
            optional_params=["limit", "offset", "fields"],
            default_params={"limit": 10, "fields": "title,authors,abstract,year,url"},
            pagination_support=True,
            max_records_per_request=100
        )

        semantic_scholar.endpoints["paper_details"] = APIEndpoint(
            endpoint_id="paper_details",
            provider_id="semantic_scholar",
            name="Paper Details",
            description="Get detailed information about a paper",
            base_url=semantic_scholar.base_url,
            path="/graph/v1/paper/{paper_id}",
            method=RequestMethod.GET,
            required_params=["paper_id"],
            optional_params=["fields"],
            default_params={"fields": "title,authors,abstract,year,citations,references"}
        )

        self.register_provider(semantic_scholar)

        # OpenAlex API
        openalex = APIProvider(
            provider_id="openalex",
            name="OpenAlex",
            description="Open catalog of scholarly papers, authors, institutions",
            base_url="https://api.openalex.org",
            default_auth_type=APIAuthType.NONE,
            global_rate_limit=10,  # 10 requests per second
            documentation_url="https://docs.openalex.org/",
        )

        openalex.endpoints["works_search"] = APIEndpoint(
            endpoint_id="works_search",
            provider_id="openalex",
            name="Works Search",
            description="Search for academic works",
            base_url=openalex.base_url,
            path="/works",
            method=RequestMethod.GET,
            optional_params=["search", "filter", "sort", "per-page", "page"],
            default_params={"per-page": 25},
            pagination_support=True,
            max_records_per_request=200
        )

        self.register_provider(openalex)

        # CrossRef API
        crossref = APIProvider(
            provider_id="crossref",
            name="CrossRef",
            description="Scholarly metadata API",
            base_url="https://api.crossref.org",
            default_auth_type=APIAuthType.NONE,
            global_rate_limit=50,
            documentation_url="https://github.com/CrossRef/rest-api-doc",
        )

        crossref.endpoints["works_search"] = APIEndpoint(
            endpoint_id="works_search",
            provider_id="crossref",
            name="Works Search",
            description="Search CrossRef works",
            base_url=crossref.base_url,
            path="/works",
            method=RequestMethod.GET,
            optional_params=["query", "filter", "sort", "order", "rows", "offset"],
            default_params={"rows": 20},
            pagination_support=True,
            max_records_per_request=1000
        )

        self.register_provider(crossref)

    def register_provider(self, provider: APIProvider):
        """Enregistre un nouveau provider"""
        self.providers[provider.provider_id] = provider

        # Initialiser rate limiter
        self.rate_limiters[provider.provider_id] = RateLimiter(
            provider_id=provider.provider_id,
            strategy=provider.rate_limit_strategy,
            limit=provider.global_rate_limit,
            window_seconds=60
        )

        # Initialiser métriques
        self.metrics["by_provider"][provider.provider_id] = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "avg_response_time": 0.0
        }

        logger.info(f"Registered API provider: {provider.provider_id}")

    def set_credentials(self, provider_id: str, credentials: APICredentials):
        """Configure les identifiants pour un provider"""
        credentials.provider_id = provider_id
        self.credentials[provider_id] = credentials
        logger.info(f"Credentials set for provider: {provider_id}")

    async def start(self):
        """Démarre le gestionnaire d'API"""
        # Créer session HTTP
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "QFrame Research API Manager"}
        )

        # Démarrer le worker de requêtes
        self._request_worker_task = asyncio.create_task(self._request_worker())

        logger.info("Research API Manager started")

    async def stop(self):
        """Arrête le gestionnaire d'API"""
        # Arrêter le worker
        if self._request_worker_task:
            self._request_worker_task.cancel()
            try:
                await self._request_worker_task
            except asyncio.CancelledError:
                pass

        # Fermer la session
        if self.session:
            await self.session.close()

        logger.info("Research API Manager stopped")

    async def request(
        self,
        provider_id: str,
        endpoint_id: str,
        params: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        use_cache: bool = True
    ) -> APIResponse:
        """
        Effectue une requête API asynchrone.

        Args:
            provider_id: ID du provider
            endpoint_id: ID de l'endpoint
            params: Paramètres de la requête
            priority: Priorité (1=low, 5=high)
            use_cache: Utiliser le cache si disponible

        Returns:
            APIResponse avec les données
        """

        # Vérifier provider et endpoint
        if provider_id not in self.providers:
            raise ValueError(f"Unknown provider: {provider_id}")

        provider = self.providers[provider_id]

        if endpoint_id not in provider.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint_id}")

        endpoint = provider.endpoints[endpoint_id]

        # Vérifier cache si demandé
        if use_cache:
            cache_key = self._generate_cache_key(provider_id, endpoint_id, params or {})
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                # Vérifier TTL
                if (datetime.utcnow() - cached_response.timestamp).total_seconds() < self.cache_ttl_seconds:
                    cached_response.from_cache = True
                    self.metrics["cache_hits"] += 1
                    return cached_response

        # Construire la requête
        request = await self._build_request(provider, endpoint, params or {})

        # Ajouter à la queue avec priorité
        await self.request_queue.put((10 - priority, time.time(), request))
        self.active_requests[request.request_id] = request

        # Attendre la réponse (timeout après 60 secondes)
        timeout_time = time.time() + 60.0
        while request.request_id in self.active_requests:
            if time.time() > timeout_time:
                raise asyncio.TimeoutError("Request timeout")
            await asyncio.sleep(0.1)

        # Récupérer la réponse du cache (elle y sera mise par le worker)
        cache_key = self._generate_cache_key(provider_id, endpoint_id, params or {})
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        else:
            raise Exception("Request failed - no response found")

    async def search_papers(
        self,
        query: str,
        providers: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Recherche de papiers sur plusieurs providers.

        Args:
            query: Requête de recherche
            providers: Liste des providers à utiliser
            limit: Limite de résultats par provider

        Returns:
            Liste unifiée de papiers
        """

        if providers is None:
            providers = ["semantic_scholar", "openalex", "crossref"]

        # Filtrer les providers disponibles
        available_providers = [p for p in providers if p in self.providers]

        if not available_providers:
            return []

        # Créer requêtes parallèles
        tasks = []
        for provider_id in available_providers:
            if provider_id == "semantic_scholar":
                task = self.request(
                    provider_id=provider_id,
                    endpoint_id="paper_search",
                    params={"query": query, "limit": limit}
                )
            elif provider_id == "openalex":
                task = self.request(
                    provider_id=provider_id,
                    endpoint_id="works_search",
                    params={"search": query, "per-page": min(limit, 200)}
                )
            elif provider_id == "crossref":
                task = self.request(
                    provider_id=provider_id,
                    endpoint_id="works_search",
                    params={"query": query, "rows": min(limit, 1000)}
                )
            else:
                continue

            tasks.append(task)

        # Exécuter en parallèle
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Traiter les réponses
        all_papers = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.warning(f"Search failed for {available_providers[i]}: {response}")
                continue

            papers = self._extract_papers_from_response(available_providers[i], response)
            all_papers.extend(papers)

        # Dédupliquer et normaliser
        return self._deduplicate_papers(all_papers)

    async def get_paper_details(
        self,
        paper_id: str,
        provider_id: str = "semantic_scholar"
    ) -> Optional[Dict[str, Any]]:
        """Récupère les détails d'un papier"""

        try:
            if provider_id == "semantic_scholar":
                response = await self.request(
                    provider_id=provider_id,
                    endpoint_id="paper_details",
                    params={"paper_id": paper_id}
                )

                if response.is_success and isinstance(response.content, dict):
                    return response.content
            else:
                logger.warning(f"Paper details not supported for provider: {provider_id}")

        except Exception as e:
            logger.error(f"Failed to get paper details: {e}")

        return None

    def get_provider_status(self) -> Dict[str, Any]:
        """Retourne le statut des providers"""
        status = {}

        for provider_id, provider in self.providers.items():
            rate_limiter = self.rate_limiters.get(provider_id)
            provider_metrics = self.metrics["by_provider"].get(provider_id, {})

            status[provider_id] = {
                "name": provider.name,
                "available": True,  # TODO: health check
                "rate_limit": {
                    "limit": rate_limiter.limit if rate_limiter else 0,
                    "remaining": max(0, rate_limiter.limit - len(rate_limiter.requests)) if rate_limiter else 0,
                    "reset_time": rate_limiter.reset_time.isoformat() if rate_limiter and rate_limiter.reset_time else None
                },
                "metrics": provider_metrics,
                "endpoints": list(provider.endpoints.keys())
            }

        return status

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques d'utilisation"""
        return {
            "global": {
                "total_requests": self.metrics["total_requests"],
                "successful_requests": self.metrics["successful_requests"],
                "failed_requests": self.metrics["failed_requests"],
                "cache_hits": self.metrics["cache_hits"],
                "success_rate": self.metrics["successful_requests"] / max(1, self.metrics["total_requests"]),
                "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["total_requests"]),
                "avg_response_time": np.mean(self.metrics["response_times"][-1000:]) if self.metrics["response_times"] else 0
            },
            "by_provider": self.metrics["by_provider"].copy(),
            "queue_size": self.request_queue.qsize(),
            "active_requests": len(self.active_requests),
            "cache_size": len(self.response_cache)
        }

    async def _request_worker(self):
        """Worker qui traite la queue de requêtes"""
        logger.info("API request worker started")

        try:
            while True:
                # Récupérer requête de la queue
                _, timestamp, request = await self.request_queue.get()

                try:
                    # Vérifier rate limiting
                    if not await self._check_rate_limit(request.provider_id):
                        # Remettre en queue avec délai
                        await asyncio.sleep(1)
                        await self.request_queue.put((5, time.time(), request))
                        continue

                    # Exécuter la requête
                    response = await self._execute_request(request)

                    # Mettre en cache
                    cache_key = self._generate_cache_key(
                        request.provider_id,
                        request.endpoint_id,
                        request.params
                    )
                    self.response_cache[cache_key] = response

                    # Mettre à jour métriques
                    self._update_metrics(request, response)

                except Exception as e:
                    logger.error(f"Request execution failed: {e}")

                    # Créer réponse d'erreur
                    error_response = APIResponse(
                        request=request,
                        status_code=500,
                        headers={},
                        content={"error": str(e)},
                        response_time=0.0,
                        is_success=False,
                        error=str(e)
                    )

                    # Mettre à jour métriques
                    self._update_metrics(request, error_response)

                finally:
                    # Retirer de la liste active
                    if request.request_id in self.active_requests:
                        del self.active_requests[request.request_id]

                    self.request_queue.task_done()

        except asyncio.CancelledError:
            logger.info("API request worker stopped")
            raise

    async def _build_request(
        self,
        provider: APIProvider,
        endpoint: APIEndpoint,
        params: Dict[str, Any]
    ) -> APIRequest:
        """Construit une requête API"""

        # Générer ID unique
        request_id = f"{provider.provider_id}_{endpoint.endpoint_id}_{int(time.time() * 1000000)}"

        # Combiner paramètres par défaut et fournis
        final_params = {**endpoint.default_params, **params}

        # Vérifier paramètres requis
        missing_params = [p for p in endpoint.required_params if p not in final_params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Construire URL
        url = endpoint.base_url + endpoint.path
        # Remplacer les paramètres dans le path (ex: {paper_id})
        for param_name, param_value in final_params.items():
            placeholder = f"{{{param_name}}}"
            if placeholder in url:
                url = url.replace(placeholder, str(param_value))
                # Retirer des params query
                del final_params[param_name]

        # Construire headers
        headers = await self._build_headers(provider)

        return APIRequest(
            request_id=request_id,
            provider_id=provider.provider_id,
            endpoint_id=endpoint.endpoint_id,
            method=endpoint.method,
            url=url,
            params=final_params,
            headers=headers,
            timeout=provider.timeout_seconds
        )

    async def _build_headers(self, provider: APIProvider) -> Dict[str, str]:
        """Construit les headers pour un provider"""
        headers = {
            "User-Agent": "QFrame Research API Manager",
            "Accept": "application/json"
        }

        # Ajouter authentification
        if provider.provider_id in self.credentials:
            credentials = self.credentials[provider.provider_id]

            if credentials.auth_type == APIAuthType.API_KEY:
                if credentials.api_key:
                    headers["X-API-Key"] = credentials.api_key

            elif credentials.auth_type == APIAuthType.BEARER_TOKEN:
                if credentials.bearer_token:
                    headers["Authorization"] = f"Bearer {credentials.bearer_token}"

            elif credentials.auth_type == APIAuthType.BASIC_AUTH:
                if credentials.username and credentials.password:
                    auth_string = f"{credentials.username}:{credentials.password}"
                    encoded_auth = base64.b64encode(auth_string.encode()).decode()
                    headers["Authorization"] = f"Basic {encoded_auth}"

            # Headers personnalisés
            headers.update(credentials.custom_headers)

        return headers

    async def _execute_request(self, request: APIRequest) -> APIResponse:
        """Exécute une requête HTTP"""
        start_time = time.time()

        try:
            async with self.session.request(
                method=request.method.value,
                url=request.url,
                params=request.params if request.method == RequestMethod.GET else None,
                json=request.params if request.method in [RequestMethod.POST, RequestMethod.PUT] else None,
                headers=request.headers,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:

                response_time = time.time() - start_time

                # Lire contenu
                content_type = response.headers.get('content-type', '').lower()
                if 'json' in content_type:
                    content = await response.json()
                else:
                    content = await response.text()

                # Extraire rate limit info
                rate_limit_remaining = None
                rate_limit_reset = None

                if 'x-ratelimit-remaining' in response.headers:
                    rate_limit_remaining = int(response.headers['x-ratelimit-remaining'])

                if 'x-ratelimit-reset' in response.headers:
                    reset_timestamp = int(response.headers['x-ratelimit-reset'])
                    rate_limit_reset = datetime.fromtimestamp(reset_timestamp)

                return APIResponse(
                    request=request,
                    status_code=response.status,
                    headers=dict(response.headers),
                    content=content,
                    response_time=response_time,
                    is_success=200 <= response.status < 300,
                    rate_limit_remaining=rate_limit_remaining,
                    rate_limit_reset=rate_limit_reset
                )

        except Exception as e:
            response_time = time.time() - start_time
            return APIResponse(
                request=request,
                status_code=500,
                headers={},
                content={"error": str(e)},
                response_time=response_time,
                is_success=False,
                error=str(e)
            )

    async def _check_rate_limit(self, provider_id: str) -> bool:
        """Vérifie les limites de taux"""
        if provider_id not in self.rate_limiters:
            return True

        rate_limiter = self.rate_limiters[provider_id]
        current_time = datetime.utcnow()

        # Nettoyer anciennes requêtes
        if rate_limiter.strategy == RateLimitStrategy.SLIDING_WINDOW:
            cutoff_time = current_time - timedelta(seconds=rate_limiter.window_seconds)
            rate_limiter.requests = [
                req_time for req_time in rate_limiter.requests
                if req_time > cutoff_time
            ]

            # Vérifier limite
            if len(rate_limiter.requests) >= rate_limiter.limit:
                return False

            # Ajouter cette requête
            rate_limiter.requests.append(current_time)
            return True

        return True  # Autres stratégies à implémenter

    def _update_metrics(self, request: APIRequest, response: APIResponse):
        """Met à jour les métriques"""
        self.metrics["total_requests"] += 1

        if response.is_success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

        # Métriques par provider
        provider_metrics = self.metrics["by_provider"].setdefault(request.provider_id, {
            "requests": 0, "successes": 0, "failures": 0, "avg_response_time": 0.0
        })

        provider_metrics["requests"] += 1
        if response.is_success:
            provider_metrics["successes"] += 1
        else:
            provider_metrics["failures"] += 1

        # Response time
        self.metrics["response_times"].append(response.response_time)
        if len(self.metrics["response_times"]) > 10000:
            self.metrics["response_times"] = self.metrics["response_times"][-5000:]

        # Average response time par provider
        provider_times = [rt for rt in self.metrics["response_times"][-1000:]]
        if provider_times:
            provider_metrics["avg_response_time"] = np.mean(provider_times)

    def _generate_cache_key(
        self,
        provider_id: str,
        endpoint_id: str,
        params: Dict[str, Any]
    ) -> str:
        """Génère une clé de cache"""
        key_data = {
            "provider": provider_id,
            "endpoint": endpoint_id,
            "params": params
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _extract_papers_from_response(
        self,
        provider_id: str,
        response: APIResponse
    ) -> List[Dict[str, Any]]:
        """Extrait les papiers d'une réponse API"""
        papers = []

        if not response.is_success or not isinstance(response.content, dict):
            return papers

        try:
            if provider_id == "semantic_scholar":
                papers_data = response.content.get("data", [])
                for paper in papers_data:
                    papers.append({
                        "id": paper.get("paperId"),
                        "title": paper.get("title"),
                        "authors": [a.get("name", "") for a in paper.get("authors", [])],
                        "abstract": paper.get("abstract"),
                        "year": paper.get("year"),
                        "url": paper.get("url"),
                        "source": "semantic_scholar"
                    })

            elif provider_id == "openalex":
                results = response.content.get("results", [])
                for work in results:
                    papers.append({
                        "id": work.get("id"),
                        "title": work.get("display_name"),
                        "authors": [a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])],
                        "abstract": work.get("abstract"),
                        "year": work.get("publication_year"),
                        "url": work.get("doi"),
                        "source": "openalex"
                    })

            elif provider_id == "crossref":
                items = response.content.get("message", {}).get("items", [])
                for item in items:
                    papers.append({
                        "id": item.get("DOI"),
                        "title": " ".join(item.get("title", [])),
                        "authors": [f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author", [])],
                        "abstract": item.get("abstract"),
                        "year": item.get("published-print", {}).get("date-parts", [[None]])[0][0],
                        "url": item.get("URL"),
                        "source": "crossref"
                    })

        except Exception as e:
            logger.error(f"Failed to extract papers from {provider_id} response: {e}")

        return papers

    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Déduplique les papiers par titre et auteurs"""
        seen_papers = set()
        unique_papers = []

        for paper in papers:
            # Créer clé unique basée sur titre et premier auteur
            title = paper.get("title", "").lower().strip()
            first_author = paper.get("authors", [""])[0].lower().strip() if paper.get("authors") else ""
            key = f"{title}|{first_author}"

            if key not in seen_papers and title:
                seen_papers.add(key)
                unique_papers.append(paper)

        return unique_papers