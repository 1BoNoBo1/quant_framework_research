"""
Infrastructure Layer: Health Monitoring System
=============================================

Système de health checks et monitoring de la santé des services.
Circuit breakers, readiness/liveness probes, et alerting.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from threading import Lock, Thread
import json
import logging


class HealthStatus(str, Enum):
    """Statuts de santé possibles"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types de composants à monitorer"""
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    API = "api"
    SERVICE = "service"
    BROKER = "broker"
    DATA_PROVIDER = "data_provider"
    RISK_ENGINE = "risk_engine"
    EXECUTION_ENGINE = "execution_engine"


class CircuitBreakerState(str, Enum):
    """États du circuit breaker"""
    CLOSED = "closed"    # Fonctionnement normal
    OPEN = "open"        # Circuit ouvert, toutes les requêtes échouent
    HALF_OPEN = "half_open"  # Test de récupération


@dataclass
class HealthCheckResult:
    """Résultat d'un health check"""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ComponentHealth:
    """État de santé d'un composant"""
    name: str
    type: ComponentType
    status: HealthStatus
    last_check: datetime
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    check_results: List[HealthCheckResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: HealthCheckResult):
        """Ajouter un résultat de health check"""
        self.check_results.append(result)
        self.last_check = result.timestamp
        self.status = result.status

        if result.status == HealthStatus.HEALTHY:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0

        # Limiter l'historique
        if len(self.check_results) > 100:
            self.check_results = self.check_results[-100:]

    def get_uptime_percentage(self, hours: int = 24) -> float:
        """Calculer le pourcentage d'uptime sur une période"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_results = [r for r in self.check_results if r.timestamp > cutoff_time]

        if not recent_results:
            return 0.0

        healthy_count = sum(1 for r in recent_results if r.status == HealthStatus.HEALTHY)
        return (healthy_count / len(recent_results)) * 100

    def get_average_response_time(self, count: int = 10) -> Optional[float]:
        """Calculer le temps de réponse moyen"""
        recent_results = self.check_results[-count:]
        response_times = [r.response_time_ms for r in recent_results if r.response_time_ms is not None]

        if not response_times:
            return None

        return sum(response_times) / len(response_times)


class HealthCheck(ABC):
    """Interface pour un health check"""

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Effectuer le health check"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du health check"""
        pass

    @property
    def component_type(self) -> ComponentType:
        """Type de composant"""
        return ComponentType.SERVICE


class DatabaseHealthCheck(HealthCheck):
    """Health check pour la base de données"""

    def __init__(self, connection_pool: Any, name: str = "database"):
        self._pool = connection_pool
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DATABASE

    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        try:
            # Tentative de requête simple
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")

            if result == 1:
                response_time = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    response_time_ms=response_time
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Unexpected database response",
                    response_time_ms=(time.time() - start_time) * 1000
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )


class MarketDataHealthCheck(HealthCheck):
    """Health check pour les données de marché"""

    def __init__(self, data_provider: Any, name: str = "market_data"):
        self._provider = data_provider
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DATA_PROVIDER

    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        try:
            # Vérifier la connexion et récupérer un prix test
            test_symbol = "BTC/USDT"
            price = await self._provider.get_current_price(test_symbol)

            if price and price > 0:
                response_time = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Market data available",
                    response_time_ms=response_time,
                    details={"test_symbol": test_symbol, "price": float(price)}
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="Market data unavailable or invalid",
                    response_time_ms=(time.time() - start_time) * 1000
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Market data provider error: {str(e)}",
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )


class BrokerHealthCheck(HealthCheck):
    """Health check pour la connexion au broker"""

    def __init__(self, broker_service: Any, name: str = "broker"):
        self._broker = broker_service
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.BROKER

    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        try:
            # Vérifier la connexion et le solde du compte
            balance = await self._broker.get_account_balance()

            if balance:
                response_time = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Broker connection successful",
                    response_time_ms=response_time,
                    details={"account_active": True}
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="Broker connected but account data unavailable",
                    response_time_ms=(time.time() - start_time) * 1000
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Broker connection failed: {str(e)}",
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )


class CircuitBreaker:
    """
    Circuit breaker pour protéger les services défaillants.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

        self._lock = Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Exécuter une fonction avec protection du circuit breaker"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    # Assez de tests, décider de l'état
                    if self.success_count > self.failure_count:
                        self._reset()
                    else:
                        self._trip()
                    raise Exception(f"Circuit breaker {self.name} is testing")
                self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Vérifier si on doit tenter une réinitialisation"""
        if self.last_failure_time is None:
            return False
        return (datetime.utcnow() - self.last_failure_time).total_seconds() > self.recovery_timeout

    def _on_success(self):
        """Gérer un appel réussi"""
        with self._lock:
            self.success_count += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count > self.failure_count:
                    self._reset()
            else:
                self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Gérer un appel échoué"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()

            if self.failure_count >= self.failure_threshold:
                self._trip()

    def _trip(self):
        """Ouvrir le circuit"""
        self.state = CircuitBreakerState.OPEN
        self.half_open_calls = 0
        logging.warning(f"Circuit breaker {self.name} tripped (OPEN)")

    def _reset(self):
        """Réinitialiser le circuit"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        logging.info(f"Circuit breaker {self.name} reset (CLOSED)")

    def get_state(self) -> Dict[str, Any]:
        """Obtenir l'état du circuit breaker"""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
            }


class HealthMonitor:
    """
    Moniteur principal de santé pour l'ensemble du système.
    """

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self._checks: Dict[str, HealthCheck] = {}
        self._components: Dict[str, ComponentHealth] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._alerts: List[Callable[[str, HealthStatus], None]] = []
        self._lock = Lock()
        self._running = False
        self._monitor_thread: Optional[Thread] = None

    def register_check(self, check: HealthCheck):
        """Enregistrer un health check"""
        with self._lock:
            self._checks[check.name] = check
            self._components[check.name] = ComponentHealth(
                name=check.name,
                type=check.component_type,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.utcnow()
            )

            # Créer un circuit breaker pour ce composant
            self._circuit_breakers[check.name] = CircuitBreaker(
                name=check.name,
                failure_threshold=3,
                recovery_timeout=60
            )

    def add_alert_handler(self, handler: Callable[[str, HealthStatus], None]):
        """Ajouter un handler d'alerte"""
        self._alerts.append(handler)

    def start(self):
        """Démarrer le monitoring"""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logging.info("Health monitor started")

    def stop(self):
        """Arrêter le monitoring"""
        with self._lock:
            self._running = False

        if self._monitor_thread:
            self._monitor_thread.join()
        logging.info("Health monitor stopped")

    def _monitor_loop(self):
        """Boucle principale de monitoring"""
        while self._running:
            asyncio.run(self._check_all_components())
            time.sleep(self.check_interval)

    async def _check_all_components(self):
        """Vérifier tous les composants"""
        checks = list(self._checks.values())

        for check in checks:
            try:
                result = await check.check()
                self._process_check_result(check.name, result)
            except Exception as e:
                logging.error(f"Error checking {check.name}: {e}")
                result = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    error=str(e)
                )
                self._process_check_result(check.name, result)

    def _process_check_result(self, component_name: str, result: HealthCheckResult):
        """Traiter le résultat d'un health check"""
        with self._lock:
            component = self._components.get(component_name)
            if not component:
                return

            old_status = component.status
            component.add_result(result)

            # Déclencher des alertes si nécessaire
            if old_status != result.status:
                self._trigger_alerts(component_name, result.status)

            # Mettre à jour le circuit breaker
            cb = self._circuit_breakers.get(component_name)
            if cb:
                if result.status == HealthStatus.HEALTHY:
                    cb._on_success()
                else:
                    cb._on_failure()

    def _trigger_alerts(self, component_name: str, new_status: HealthStatus):
        """Déclencher les alertes pour un changement de statut"""
        for handler in self._alerts:
            try:
                handler(component_name, new_status)
            except Exception as e:
                logging.error(f"Error in alert handler: {e}")

    async def check_component(self, component_name: str) -> Optional[HealthCheckResult]:
        """Vérifier un composant spécifique"""
        check = self._checks.get(component_name)
        if not check:
            return None

        result = await check.check()
        self._process_check_result(component_name, result)
        return result

    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Obtenir l'état de santé d'un composant"""
        with self._lock:
            return self._components.get(component_name)

    def get_system_health(self) -> Dict[str, Any]:
        """Obtenir l'état de santé global du système"""
        with self._lock:
            components_status = {}
            unhealthy_count = 0
            degraded_count = 0

            for name, component in self._components.items():
                components_status[name] = {
                    "type": component.type.value,
                    "status": component.status.value,
                    "last_check": component.last_check.isoformat(),
                    "uptime_24h": component.get_uptime_percentage(24),
                    "avg_response_time": component.get_average_response_time(),
                    "consecutive_failures": component.consecutive_failures
                }

                if component.status == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1
                elif component.status == HealthStatus.DEGRADED:
                    degraded_count += 1

            # Déterminer le statut global
            if unhealthy_count > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif degraded_count > 0:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY

            return {
                "status": overall_status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "components": components_status,
                "circuit_breakers": {
                    name: cb.get_state()
                    for name, cb in self._circuit_breakers.items()
                }
            }

    def is_healthy(self) -> bool:
        """Vérifier si le système est globalement sain"""
        health = self.get_system_health()
        return health["status"] == HealthStatus.HEALTHY.value


class ReadinessProbe:
    """Probe de readiness pour Kubernetes/conteneurs"""

    def __init__(self, monitor: HealthMonitor, required_components: List[str]):
        self.monitor = monitor
        self.required_components = required_components

    async def check(self) -> bool:
        """Vérifier si l'application est prête à recevoir du trafic"""
        for component_name in self.required_components:
            health = self.monitor.get_component_health(component_name)
            if not health or health.status != HealthStatus.HEALTHY:
                return False
        return True


class LivenessProbe:
    """Probe de liveness pour Kubernetes/conteneurs"""

    def __init__(self, monitor: HealthMonitor):
        self.monitor = monitor
        self.last_check = datetime.utcnow()

    async def check(self) -> bool:
        """Vérifier si l'application est vivante"""
        # Vérification simple: au moins un composant répond
        system_health = self.monitor.get_system_health()
        self.last_check = datetime.utcnow()
        return system_health["status"] != HealthStatus.UNHEALTHY.value


# Instance globale
_global_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Obtenir l'instance globale du moniteur de santé"""
    return _global_monitor