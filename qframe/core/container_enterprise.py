"""
Enterprise Dependency Injection Container
=========================================

Container DI de niveau enterprise avec type safety complète,
gestion avancée des scopes, et monitoring intégré.
"""

import inspect
import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
TInterface = TypeVar("TInterface")
TImplementation = TypeVar("TImplementation")


class LifetimeScope(str, Enum):
    """Scopes de vie des objets avec sémantique claire."""

    SINGLETON = "singleton"  # Une instance globale
    TRANSIENT = "transient"  # Nouvelle instance à chaque résolution
    SCOPED = "scoped"  # Une instance par scope
    REQUEST = "request"  # Une instance par requête (web)
    THREAD = "thread"  # Une instance par thread


class DependencyScope(str, Enum):
    """Portées des dépendances."""

    APPLICATION = "application"
    REQUEST = "request"
    OPERATION = "operation"


@dataclass
class InjectionMetrics:
    """Métriques d'injection pour monitoring."""

    total_resolutions: int = 0
    failed_resolutions: int = 0
    circular_dependencies: int = 0
    average_resolution_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    def record_resolution(self, success: bool, time_ms: float) -> None:
        """Enregistre une résolution."""
        self.total_resolutions += 1
        if not success:
            self.failed_resolutions += 1

        # Calcul moyenne mobile
        total_time = self.average_resolution_time_ms * (self.total_resolutions - 1)
        self.average_resolution_time_ms = (
            total_time + time_ms
        ) / self.total_resolutions


@dataclass
class ServiceDescriptor(Generic[T]):
    """Descripteur de service type-safe."""

    interface: Type[T]
    implementation: Optional[Type[T]] = None
    factory: Optional[Callable[[], T]] = None
    instance: Optional[T] = None
    lifetime: LifetimeScope = LifetimeScope.TRANSIENT

    # Métadonnées
    name: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[Type[Any]] = field(default_factory=set)

    # Monitoring
    creation_count: int = 0
    last_accessed: Optional[float] = None
    is_disposed: bool = False

    def __post_init__(self) -> None:
        """Validation post-initialisation."""
        if not self.implementation and not self.factory and not self.instance:
            raise ValueError(
                "Au moins implementation, factory ou instance doit être fourni"
            )

        if self.implementation and not inspect.isclass(self.implementation):
            raise ValueError("Implementation doit être une classe")


class Injectable(Protocol):
    """Protocol pour marquer les classes injectables."""

    pass


class ILifetimeScope(Protocol):
    """Interface pour les gestionnaires de scope."""

    def get_instance(self, descriptor: ServiceDescriptor[T]) -> T:
        """Obtient une instance selon le scope."""
        ...

    def dispose(self) -> None:
        """Libère les ressources du scope."""
        ...


class IServiceContainer(Protocol[T]):
    """Interface du container de services."""

    def register_singleton(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
    ) -> "EnterpriseContainer":
        """Enregistre un service singleton."""
        ...

    def register_transient(
        self, interface: Type[T], implementation: Optional[Type[T]] = None
    ) -> "EnterpriseContainer":
        """Enregistre un service transient."""
        ...

    def resolve(self, interface: Type[T]) -> T:
        """Résout un service."""
        ...


class CircularDependencyError(Exception):
    """Erreur de dépendance circulaire."""

    def __init__(self, resolution_path: List[Type[Any]]):
        self.resolution_path = resolution_path
        path_str = " -> ".join(cls.__name__ for cls in resolution_path)
        super().__init__(f"Dépendance circulaire détectée: {path_str}")


class ServiceNotFoundError(Exception):
    """Service non trouvé."""

    def __init__(self, interface: Type[Any]):
        super().__init__(f"Service non enregistré: {interface.__name__}")


class SingletonLifetimeManager:
    """Gestionnaire de lifetime pour singletons."""

    def __init__(self):
        self._instances: Dict[Type[Any], Any] = {}
        self._lock = threading.RLock()

    def get_instance(
        self, descriptor: ServiceDescriptor[T], factory: Callable[[], T]
    ) -> T:
        """Obtient ou crée l'instance singleton."""
        interface = descriptor.interface

        if interface in self._instances:
            descriptor.last_accessed = time.time()
            return self._instances[interface]

        with self._lock:
            # Double-check locking pattern
            if interface in self._instances:
                return self._instances[interface]

            instance = factory()
            self._instances[interface] = instance
            descriptor.creation_count += 1
            descriptor.last_accessed = time.time()

            logger.debug(f"Singleton créé: {interface.__name__}")
            return instance

    def dispose(self, interface: Type[Any]) -> None:
        """Dispose un singleton."""
        with self._lock:
            if interface in self._instances:
                instance = self._instances.pop(interface)
                if hasattr(instance, "dispose"):
                    instance.dispose()


class ScopeManager:
    """Gestionnaire de scope avancé."""

    def __init__(self, container: "EnterpriseContainer", scope_id: str):
        self.container = container
        self.scope_id = scope_id
        self._instances: Dict[Type[Any], Any] = {}
        self._disposed = False

    def __enter__(self) -> "ScopeManager":
        self.container._enter_scope(self.scope_id, self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()

    def get_instance(
        self, descriptor: ServiceDescriptor[T], factory: Callable[[], T]
    ) -> T:
        """Obtient instance dans ce scope."""
        if self._disposed:
            raise RuntimeError("Scope disposed")

        interface = descriptor.interface
        if interface in self._instances:
            return self._instances[interface]

        instance = factory()
        self._instances[interface] = instance
        descriptor.creation_count += 1

        return instance

    def dispose(self) -> None:
        """Dispose le scope et ses instances."""
        if self._disposed:
            return

        for instance in self._instances.values():
            if hasattr(instance, "dispose"):
                try:
                    instance.dispose()
                except Exception as e:
                    logger.error(f"Erreur lors du dispose: {e}")

        self._instances.clear()
        self._disposed = True
        self.container._exit_scope(self.scope_id)


class EnterpriseContainer:
    """Container DI enterprise avec fonctionnalités avancées."""

    def __init__(self, enable_metrics: bool = True):
        # Core storage
        self._services: Dict[Type[Any], ServiceDescriptor[Any]] = {}
        self._instances: Dict[Type[Any], Any] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Scope management
        self._scopes: Dict[str, ScopeManager] = {}
        self._current_scope: Optional[str] = None

        # Singleton management
        self._singleton_manager = SingletonLifetimeManager()

        # Resolution tracking
        self._resolution_stack: List[Type[Any]] = []

        # Metrics and monitoring
        self.enable_metrics = enable_metrics
        self.metrics = InjectionMetrics()

        # Auto-discovery
        self._auto_discovered: Set[Type[Any]] = set()

        logger.info("Enterprise DI Container initialisé")

    def register_singleton(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        name: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> "EnterpriseContainer":
        """Enregistre un service singleton avec métadonnées."""
        return self._register(
            interface, implementation, factory, LifetimeScope.SINGLETON, name, tags
        )

    def register_transient(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        name: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> "EnterpriseContainer":
        """Enregistre un service transient."""
        return self._register(
            interface, implementation, factory, LifetimeScope.TRANSIENT, name, tags
        )

    def register_scoped(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        name: Optional[str] = None,
    ) -> "EnterpriseContainer":
        """Enregistre un service scoped."""
        return self._register(
            interface, implementation, factory, LifetimeScope.SCOPED, name
        )

    def register_instance(
        self, interface: Type[T], instance: T, name: Optional[str] = None
    ) -> "EnterpriseContainer":
        """Enregistre une instance existante."""
        with self._lock:
            descriptor = ServiceDescriptor(
                interface=interface,
                instance=instance,
                lifetime=LifetimeScope.SINGLETON,
                name=name,
            )
            self._services[interface] = descriptor

        logger.debug(f"Instance enregistrée: {interface.__name__}")
        return self

    def _register(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]],
        factory: Optional[Callable[[], T]],
        lifetime: LifetimeScope,
        name: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> "EnterpriseContainer":
        """Enregistrement interne avec validation."""

        # Validation
        if implementation is None and factory is None:
            implementation = interface

        if implementation and not self._is_assignable(interface, implementation):
            raise ValueError(
                f"{implementation.__name__} n'implémente pas {interface.__name__}"
            )

        with self._lock:
            descriptor = ServiceDescriptor(
                interface=interface,
                implementation=implementation,
                factory=factory,
                lifetime=lifetime,
                name=name,
                tags=tags or set(),
            )

            # Analyse des dépendances
            if implementation:
                descriptor.dependencies = self._analyze_dependencies(implementation)

            self._services[interface] = descriptor

        logger.debug(f"Service enregistré: {interface.__name__} -> {lifetime.value}")
        return self

    def resolve(self, interface: Type[T]) -> T:
        """Résout un service avec monitoring complet."""
        start_time = time.perf_counter()
        success = False

        try:
            result = self._resolve_internal(interface)
            success = True

            if self.enable_metrics:
                self.metrics.cache_hits += 1

            return result

        except Exception as e:
            if self.enable_metrics:
                self.metrics.cache_misses += 1
            logger.error(f"Échec résolution {interface.__name__}: {e}")
            raise

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if self.enable_metrics:
                self.metrics.record_resolution(success, elapsed_ms)

    def _resolve_internal(self, interface: Type[T]) -> T:
        """Résolution interne avec détection de cycles."""

        # Détection de dépendance circulaire
        if interface in self._resolution_stack:
            circular_path = self._resolution_stack + [interface]
            if self.enable_metrics:
                self.metrics.circular_dependencies += 1
            raise CircularDependencyError(circular_path)

        # Auto-discovery si service non enregistré
        if interface not in self._services:
            if self._try_auto_discover(interface):
                logger.info(f"Service auto-découvert: {interface.__name__}")
            else:
                raise ServiceNotFoundError(interface)

        descriptor = self._services[interface]

        # Gestion par lifetime
        if descriptor.lifetime == LifetimeScope.SINGLETON:
            return self._resolve_singleton(descriptor)
        elif descriptor.lifetime == LifetimeScope.SCOPED:
            return self._resolve_scoped(descriptor)
        else:  # TRANSIENT
            return self._resolve_transient(descriptor)

    def _resolve_singleton(self, descriptor: ServiceDescriptor[T]) -> T:
        """Résolution singleton."""
        if descriptor.instance:
            descriptor.last_accessed = time.time()
            return descriptor.instance

        return self._singleton_manager.get_instance(
            descriptor, lambda: self._create_instance(descriptor)
        )

    def _resolve_scoped(self, descriptor: ServiceDescriptor[T]) -> T:
        """Résolution scoped."""
        if not self._current_scope:
            # Fallback à singleton si pas de scope
            return self._resolve_singleton(descriptor)

        scope_manager = self._scopes.get(self._current_scope)
        if not scope_manager:
            raise RuntimeError(f"Scope non trouvé: {self._current_scope}")

        return scope_manager.get_instance(
            descriptor, lambda: self._create_instance(descriptor)
        )

    def _resolve_transient(self, descriptor: ServiceDescriptor[T]) -> T:
        """Résolution transient (nouvelle instance)."""
        return self._create_instance(descriptor)

    def _create_instance(self, descriptor: ServiceDescriptor[T]) -> T:
        """Crée une nouvelle instance avec injection de dépendances."""
        interface = descriptor.interface

        try:
            self._resolution_stack.append(interface)

            if descriptor.factory:
                return descriptor.factory()

            if descriptor.instance:
                return descriptor.instance

            if not descriptor.implementation:
                raise ValueError(f"Pas d'implémentation pour {interface.__name__}")

            # Injection de dépendances via constructeur
            return self._create_with_injection(descriptor.implementation)

        finally:
            if interface in self._resolution_stack:
                self._resolution_stack.remove(interface)

    def _create_with_injection(self, implementation: Type[T]) -> T:
        """Crée instance avec injection automatique des dépendances."""

        # Obtenir signature du constructeur
        constructor = implementation.__init__
        sig = inspect.signature(constructor)

        # Préparer arguments d'injection
        inject_args = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Obtenir type annotation
            if param.annotation != inspect.Parameter.empty:
                dependency_type = param.annotation

                # Résoudre la dépendance
                try:
                    dependency = self._resolve_internal(dependency_type)
                    inject_args[param_name] = dependency
                except ServiceNotFoundError:
                    if param.default != inspect.Parameter.empty:
                        # Utiliser valeur par défaut si disponible
                        continue
                    else:
                        logger.warning(f"Dépendance non résolue: {dependency_type}")
                        raise

        # Créer l'instance
        instance = implementation(**inject_args)

        # Post-injection hook
        if hasattr(instance, "__post_inject__"):
            instance.__post_inject__()

        return instance

    def _try_auto_discover(self, interface: Type[T]) -> bool:
        """Tentative d'auto-découverte du service."""

        # Ne pas redécouvrir
        if interface in self._auto_discovered:
            return False

        self._auto_discovered.add(interface)

        # Si c'est une classe concrète, s'auto-enregistrer
        if inspect.isclass(interface) and not inspect.isabstract(interface):
            self.register_transient(interface, interface)
            return True

        return False

    def _analyze_dependencies(self, implementation: Type[Any]) -> Set[Type[Any]]:
        """Analyse les dépendances d'une implémentation."""
        dependencies = set()

        if hasattr(implementation, "__init__"):
            sig = inspect.signature(implementation.__init__)
            for param in sig.parameters.values():
                if param.name != "self" and param.annotation != inspect.Parameter.empty:
                    dependencies.add(param.annotation)

        return dependencies

    def _is_assignable(self, interface: Type[Any], implementation: Type[Any]) -> bool:
        """Vérifie si implementation peut être assignée à interface."""
        try:
            return issubclass(implementation, interface)
        except TypeError:
            # Pour les types génériques et protocols
            return True

    @contextmanager
    def create_scope(self, scope_id: Optional[str] = None):
        """Context manager pour création de scope."""
        if scope_id is None:
            import uuid

            scope_id = str(uuid.uuid4())

        scope_manager = ScopeManager(self, scope_id)
        try:
            with scope_manager:
                yield scope_manager
        finally:
            pass  # Cleanup handled by ScopeManager

    def _enter_scope(self, scope_id: str, scope_manager: ScopeManager) -> None:
        """Entre dans un scope."""
        self._scopes[scope_id] = scope_manager
        self._current_scope = scope_id

    def _exit_scope(self, scope_id: str) -> None:
        """Sort d'un scope."""
        if scope_id in self._scopes:
            del self._scopes[scope_id]

        if self._current_scope == scope_id:
            self._current_scope = None

    def get_metrics(self) -> InjectionMetrics:
        """Obtient les métriques du container."""
        return self.metrics

    def get_service_info(self, interface: Type[T]) -> Optional[Dict[str, Any]]:
        """Obtient informations sur un service."""
        if interface not in self._services:
            return None

        descriptor = self._services[interface]
        return {
            "interface": interface.__name__,
            "implementation": descriptor.implementation.__name__
            if descriptor.implementation
            else None,
            "lifetime": descriptor.lifetime.value,
            "creation_count": descriptor.creation_count,
            "last_accessed": descriptor.last_accessed,
            "dependencies": [dep.__name__ for dep in descriptor.dependencies],
            "tags": list(descriptor.tags),
            "name": descriptor.name,
        }

    def list_services(self) -> List[Dict[str, Any]]:
        """Liste tous les services enregistrés."""
        return [self.get_service_info(interface) for interface in self._services.keys()]

    def validate_container(self) -> List[str]:
        """Valide la configuration du container."""
        errors = []

        for interface, descriptor in self._services.items():
            # Vérifier que les dépendances sont satisfiables
            for dep in descriptor.dependencies:
                if dep not in self._services and not self._try_auto_discover(dep):
                    errors.append(
                        f"{interface.__name__} dépend de {dep.__name__} non enregistré"
                    )

        return errors

    def dispose(self) -> None:
        """Dispose le container et toutes ses ressources."""
        logger.info("Disposing Enterprise DI Container")

        # Dispose singletons
        for interface in list(self._services.keys()):
            descriptor = self._services[interface]
            if descriptor.lifetime == LifetimeScope.SINGLETON:
                self._singleton_manager.dispose(interface)

        # Dispose scopes
        for scope in list(self._scopes.values()):
            scope.dispose()

        # Clear everything
        self._services.clear()
        self._instances.clear()
        self._scopes.clear()


# Factory function pour usage global
_global_container: Optional[EnterpriseContainer] = None


def get_enterprise_container() -> EnterpriseContainer:
    """Obtient le container global enterprise."""
    global _global_container
    if _global_container is None:
        _global_container = EnterpriseContainer()
    return _global_container


def configure_enterprise_container(container: EnterpriseContainer) -> None:
    """Configure le container global."""
    global _global_container
    _global_container = container


# Décorateurs pour faciliter l'usage
def injectable(
    lifetime: LifetimeScope = LifetimeScope.TRANSIENT,
    name: Optional[str] = None,
    tags: Optional[Set[str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Décorateur pour marquer une classe comme injectable."""

    def decorator(cls: Type[T]) -> Type[T]:
        # Marquer la classe avec métadonnées
        cls.__injectable__ = True
        cls.__lifetime__ = lifetime
        cls.__service_name__ = name
        cls.__service_tags__ = tags or set()

        # Auto-enregistrement dans container global
        container = get_enterprise_container()
        if lifetime == LifetimeScope.SINGLETON:
            container.register_singleton(cls, cls, name=name, tags=tags)
        elif lifetime == LifetimeScope.SCOPED:
            container.register_scoped(cls, cls, name=name)
        else:
            container.register_transient(cls, cls, name=name, tags=tags)

        return cls

    return decorator


def inject(interface: Type[T]) -> T:
    """Helper function pour injection directe."""
    container = get_enterprise_container()
    return container.resolve(interface)
