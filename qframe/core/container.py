"""
Dependency Injection Container
=============================

Container IoC pour gestion des dépendances et inversion de contrôle.
Permet une architecture testable et découplée.
"""

from typing import Any, Dict, Type, TypeVar, Callable, Optional, Protocol
import inspect
import threading
from functools import wraps

T = TypeVar('T')


class Injectable(Protocol):
    """Marque une classe comme injectable"""
    pass


class LifetimeScope:
    """Scopes de vie des objets"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor:
    """Descripteur de service pour le container"""

    def __init__(
        self,
        interface: Type,
        implementation: Type,
        lifetime: str = LifetimeScope.TRANSIENT,
        factory: Optional[Callable] = None
    ):
        self.interface = interface
        self.implementation = implementation
        self.lifetime = lifetime
        self.factory = factory
        self.instance: Optional[Any] = None


class DIContainer:
    """
    Dependency Injection Container avec support pour:
    - Singleton, Transient, Scoped lifetimes
    - Factory methods
    - Constructor injection automatique
    - Circular dependency detection
    - Thread safety
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
        self._resolution_stack: list = []

    def register_singleton(
        self,
        interface: Type[T],
        implementation: Type[T] = None,
        factory: Callable[[], T] = None
    ) -> 'DIContainer':
        """Enregistre un service singleton"""
        return self._register(interface, implementation, LifetimeScope.SINGLETON, factory)

    def register_transient(
        self,
        interface: Type[T],
        implementation: Type[T] = None,
        factory: Callable[[], T] = None
    ) -> 'DIContainer':
        """Enregistre un service transient (nouvelle instance à chaque résolution)"""
        return self._register(interface, implementation, LifetimeScope.TRANSIENT, factory)

    def register_scoped(
        self,
        interface: Type[T],
        implementation: Type[T] = None,
        factory: Callable[[], T] = None
    ) -> 'DIContainer':
        """Enregistre un service scoped (une instance par scope)"""
        return self._register(interface, implementation, LifetimeScope.SCOPED, factory)

    def _register(
        self,
        interface: Type[T],
        implementation: Type[T] = None,
        lifetime: str = LifetimeScope.TRANSIENT,
        factory: Callable[[], T] = None
    ) -> 'DIContainer':
        """Enregistrement interne"""
        with self._lock:
            if implementation is None and factory is None:
                implementation = interface

            descriptor = ServiceDescriptor(
                interface=interface,
                implementation=implementation,
                lifetime=lifetime,
                factory=factory
            )

            self._services[interface] = descriptor

            # Clear cached singleton if re-registering
            if interface in self._instances:
                del self._instances[interface]

        return self

    def resolve(self, interface: Type[T]) -> T:
        """Résout un service avec ses dépendances"""
        with self._lock:
            return self._resolve_internal(interface)

    def _resolve_internal(self, interface: Type[T]) -> T:
        """Résolution interne avec détection de dépendances circulaires"""

        # Check for circular dependencies
        if interface in self._resolution_stack:
            stack_str = " -> ".join([cls.__name__ for cls in self._resolution_stack])
            raise ValueError(
                f"Circular dependency detected: {stack_str} -> {interface.__name__}"
            )

        # Add to resolution stack
        self._resolution_stack.append(interface)

        try:
            return self._create_instance(interface)
        finally:
            # Remove from resolution stack
            self._resolution_stack.pop()

    def _create_instance(self, interface: Type[T]) -> T:
        """Crée une instance selon la stratégie de lifetime"""

        if interface not in self._services:
            # Try to auto-register if it's a concrete class (not a Protocol or abstract class)
            if (inspect.isclass(interface) and
                not inspect.isabstract(interface) and
                not getattr(interface, '_is_protocol', False)):
                self.register_transient(interface, interface)
            else:
                # Handle string annotations and forward references
                interface_name = getattr(interface, '__name__', str(interface))
                raise ValueError(f"Service {interface_name} not registered")

        descriptor = self._services[interface]

        # Singleton handling
        if descriptor.lifetime == LifetimeScope.SINGLETON:
            if interface not in self._instances:
                self._instances[interface] = self._build_instance(descriptor)
            return self._instances[interface]

        # Scoped handling
        elif descriptor.lifetime == LifetimeScope.SCOPED:
            if self._current_scope is None:
                raise ValueError("No active scope for scoped service")

            if self._current_scope not in self._scoped_instances:
                self._scoped_instances[self._current_scope] = {}

            scope_instances = self._scoped_instances[self._current_scope]
            if interface not in scope_instances:
                scope_instances[interface] = self._build_instance(descriptor)

            return scope_instances[interface]

        # Transient handling
        else:
            return self._build_instance(descriptor)

    def _build_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Construit une instance avec injection de dépendances"""

        # Use factory if provided
        if descriptor.factory is not None:
            return descriptor.factory()

        # Constructor injection
        implementation = descriptor.implementation
        constructor = implementation.__init__
        sig = inspect.signature(constructor)

        # Get constructor parameters (skip 'self')
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # Skip *args and **kwargs
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                             inspect.Parameter.VAR_KEYWORD):
                continue

            if param.annotation == inspect.Parameter.empty:
                # Skip parameters without annotations or use default value
                if param.default != inspect.Parameter.empty:
                    params[param_name] = param.default
                continue

            # Handle string annotations and forward references
            annotation = param.annotation
            if isinstance(annotation, str):
                # For string annotations, try to resolve using registered services
                # This handles the case where ServiceB is defined locally and referenced as string
                for registered_type in self._services.keys():
                    if getattr(registered_type, '__name__', None) == annotation:
                        annotation = registered_type
                        break

                # If still a string, skip it
                if isinstance(annotation, str):
                    continue

            # Resolve dependency recursively
            dependency = self._resolve_internal(annotation)
            params[param_name] = dependency

        return implementation(**params)

    def create_scope(self, scope_id: str = None) -> 'ScopeManager':
        """Crée un nouveau scope"""
        if scope_id is None:
            import uuid
            scope_id = str(uuid.uuid4())

        return ScopeManager(self, scope_id)

    def _enter_scope(self, scope_id: str):
        """Interne: entre dans un scope"""
        self._current_scope = scope_id
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}

    def _exit_scope(self, scope_id: str):
        """Interne: sort d'un scope et nettoie"""
        if scope_id in self._scoped_instances:
            # Cleanup scoped instances
            for instance in self._scoped_instances[scope_id].values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception:
                        pass  # Best effort cleanup

            del self._scoped_instances[scope_id]

        if self._current_scope == scope_id:
            self._current_scope = None

    def get_registrations(self) -> Dict[Type, ServiceDescriptor]:
        """Retourne les enregistrements de services (pour debug)"""
        return self._services.copy()

    def clear(self):
        """Nettoie le container"""
        with self._lock:
            # Cleanup singletons
            for instance in self._instances.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception:
                        pass

            self._services.clear()
            self._instances.clear()
            self._scoped_instances.clear()
            self._current_scope = None
            self._resolution_stack.clear()


class ScopeManager:
    """Context manager pour gestion de scopes"""

    def __init__(self, container: DIContainer, scope_id: str):
        self.container = container
        self.scope_id = scope_id

    def __enter__(self):
        self.container._enter_scope(self.scope_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container._exit_scope(self.scope_id)

    def resolve(self, interface: Type[T]) -> T:
        """Résout un service dans ce scope"""
        return self.container.resolve(interface)


# ================================
# DECORATORS
# ================================

def injectable(cls: Type[T]) -> Type[T]:
    """Décorateur pour marquer une classe comme injectable"""
    if not hasattr(cls, '__annotations__'):
        cls.__annotations__ = {}

    # Add Injectable protocol marker
    cls.__injectable__ = True
    return cls


def singleton(cls: Type[T]) -> Type[T]:
    """Décorateur pour auto-enregistrer comme singleton"""
    cls.__lifetime__ = LifetimeScope.SINGLETON
    return injectable(cls)


def transient(cls: Type[T]) -> Type[T]:
    """Décorateur pour auto-enregistrer comme transient"""
    cls.__lifetime__ = LifetimeScope.TRANSIENT
    return injectable(cls)


def scoped(cls: Type[T]) -> Type[T]:
    """Décorateur pour auto-enregistrer comme scoped"""
    cls.__lifetime__ = LifetimeScope.SCOPED
    return injectable(cls)


# ================================
# AUTO-REGISTRATION
# ================================

def auto_register(container: DIContainer, module_or_class):
    """Auto-enregistre les classes décorées d'un module"""

    if inspect.ismodule(module_or_class):
        # Register all injectable classes in module
        for name in dir(module_or_class):
            obj = getattr(module_or_class, name)
            if inspect.isclass(obj) and hasattr(obj, '__injectable__'):
                _register_class(container, obj)

    elif inspect.isclass(module_or_class):
        # Register single class
        if hasattr(module_or_class, '__injectable__'):
            _register_class(container, module_or_class)


def _register_class(container: DIContainer, cls: Type):
    """Enregistre une classe selon son lifetime décoré"""
    lifetime = getattr(cls, '__lifetime__', LifetimeScope.TRANSIENT)

    if lifetime == LifetimeScope.SINGLETON:
        container.register_singleton(cls, cls)
    elif lifetime == LifetimeScope.SCOPED:
        container.register_scoped(cls, cls)
    else:
        container.register_transient(cls, cls)


# ================================
# FACTORY HELPERS
# ================================

def factory(func: Callable[[], T]) -> Callable[[], T]:
    """Décorateur pour marquer une fonction comme factory"""
    func.__is_factory__ = True
    return func


def configure_services(container: DIContainer):
    """Configuration de base des services framework"""

    # Import local pour éviter les dépendances circulaires
    from ..infrastructure.config.service_configuration import configure_production_services

    # Configuration complète via le nouveau système
    configure_production_services(container)


# ================================
# TESTING HELPERS
# ================================

class MockContainer(DIContainer):
    """Container pour tests avec mocks"""

    def register_mock(self, interface: Type[T], mock_instance: T) -> 'MockContainer':
        """Enregistre un mock"""
        self._instances[interface] = mock_instance
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=type(mock_instance),
            lifetime=LifetimeScope.SINGLETON
        )
        self._services[interface] = descriptor
        return self


# ================================
# GLOBAL CONTAINER
# ================================

# Container global pour l'application
# Dans une vraie app, ceci serait configuré au démarrage
_global_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Retourne le container global"""
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
        configure_services(_global_container)
    return _global_container


def set_container(container: DIContainer):
    """Définit le container global (pour tests principalement)"""
    global _global_container
    _global_container = container