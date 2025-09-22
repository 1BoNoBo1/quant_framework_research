"""
Tests unitaires pour le DI Container
===================================

Tests complets du système d'injection de dépendances.
"""

import pytest
from typing import Protocol
import threading
import time

from qframe.core.container import (
    DIContainer,
    LifetimeScope,
    ServiceDescriptor,
    injectable,
    singleton,
    transient,
    scoped,
    auto_register,
    MockContainer
)


# Classes de test
class ITestService(Protocol):
    def get_value(self) -> str: ...


@injectable
class SimpleService:
    def get_value(self) -> str:
        return "simple"


@injectable
class DependentService:
    def __init__(self, service: SimpleService):
        self.service = service

    def get_combined_value(self) -> str:
        return f"dependent-{self.service.get_value()}"


@singleton
class SingletonService:
    def __init__(self):
        self.created_at = time.time()

    def get_creation_time(self) -> float:
        return self.created_at


@transient
class TransientService:
    def __init__(self):
        self.created_at = time.time()

    def get_creation_time(self) -> float:
        return self.created_at


@scoped
class ScopedService:
    def __init__(self):
        self.created_at = time.time()

    def get_creation_time(self) -> float:
        return self.created_at


class DisposableService:
    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True


class TestDIContainer:
    """Tests du container DI principal"""

    def test_register_and_resolve_simple(self):
        """Test d'enregistrement et résolution simple"""
        container = DIContainer()
        container.register_transient(SimpleService)

        service = container.resolve(SimpleService)
        assert isinstance(service, SimpleService)
        assert service.get_value() == "simple"

    def test_register_with_interface(self):
        """Test d'enregistrement avec interface"""
        container = DIContainer()
        container.register_transient(ITestService, SimpleService)

        service = container.resolve(ITestService)
        assert isinstance(service, SimpleService)
        assert service.get_value() == "simple"

    def test_dependency_injection(self):
        """Test d'injection de dépendances automatique"""
        container = DIContainer()
        container.register_transient(SimpleService)
        container.register_transient(DependentService)

        service = container.resolve(DependentService)
        assert isinstance(service, DependentService)
        assert service.get_combined_value() == "dependent-simple"

    def test_singleton_lifetime(self):
        """Test du lifetime singleton"""
        container = DIContainer()
        container.register_singleton(SingletonService)

        service1 = container.resolve(SingletonService)
        time.sleep(0.01)  # Petit délai
        service2 = container.resolve(SingletonService)

        assert service1 is service2
        assert service1.get_creation_time() == service2.get_creation_time()

    def test_transient_lifetime(self):
        """Test du lifetime transient"""
        container = DIContainer()
        container.register_transient(TransientService)

        service1 = container.resolve(TransientService)
        time.sleep(0.01)  # Petit délai
        service2 = container.resolve(TransientService)

        assert service1 is not service2
        assert service1.get_creation_time() != service2.get_creation_time()

    def test_scoped_lifetime(self):
        """Test du lifetime scoped"""
        container = DIContainer()
        container.register_scoped(ScopedService)

        # Sans scope actif, devrait lever une erreur
        with pytest.raises(ValueError, match="No active scope"):
            container.resolve(ScopedService)

        # Avec scope
        with container.create_scope() as scope:
            service1 = scope.resolve(ScopedService)
            service2 = scope.resolve(ScopedService)

            assert service1 is service2
            assert service1.get_creation_time() == service2.get_creation_time()

        # Nouveau scope
        with container.create_scope() as scope:
            service3 = scope.resolve(ScopedService)
            assert service3 is not service1

    def test_factory_registration(self):
        """Test d'enregistrement avec factory"""
        container = DIContainer()

        def create_service():
            return SimpleService()

        container.register_transient(SimpleService, factory=create_service)

        service = container.resolve(SimpleService)
        assert isinstance(service, SimpleService)

    def test_circular_dependency_detection(self):
        """Test de détection de dépendances circulaires"""

        @injectable
        class ServiceA:
            def __init__(self, service_b: 'ServiceB'): ...

        @injectable
        class ServiceB:
            def __init__(self, service_a: ServiceA): ...

        container = DIContainer()
        container.register_transient(ServiceA)
        container.register_transient(ServiceB)

        with pytest.raises(ValueError, match="Circular dependency detected"):
            container.resolve(ServiceA)

    def test_thread_safety(self):
        """Test de la thread safety"""
        container = DIContainer()
        container.register_singleton(SingletonService)

        results = []

        def resolve_service():
            service = container.resolve(SingletonService)
            results.append(service)

        # Créer plusieurs threads qui résolvent en parallèle
        threads = [threading.Thread(target=resolve_service) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Tous les services doivent être la même instance
        assert len(set(id(service) for service in results)) == 1

    def test_auto_registration(self):
        """Test d'auto-enregistrement avec décorateurs"""
        container = DIContainer()

        auto_register(container, SingletonService)
        auto_register(container, TransientService)
        auto_register(container, ScopedService)

        # Vérifier les enregistrements
        registrations = container.get_registrations()

        assert SingletonService in registrations
        assert registrations[SingletonService].lifetime == LifetimeScope.SINGLETON

        assert TransientService in registrations
        assert registrations[TransientService].lifetime == LifetimeScope.TRANSIENT

        assert ScopedService in registrations
        assert registrations[ScopedService].lifetime == LifetimeScope.SCOPED

    def test_cleanup_on_clear(self):
        """Test du nettoyage lors du clear"""
        container = DIContainer()
        container.register_singleton(DisposableService)

        service = container.resolve(DisposableService)
        assert not service.disposed

        container.clear()
        assert service.disposed

    def test_scope_cleanup(self):
        """Test du nettoyage des scopes"""
        container = DIContainer()
        container.register_scoped(DisposableService)

        service = None
        with container.create_scope() as scope:
            service = scope.resolve(DisposableService)
            assert not service.disposed

        # Après sortie du scope, dispose doit être appelé
        assert service.disposed

    def test_unregistered_service(self):
        """Test de résolution de service non enregistré"""
        container = DIContainer()

        # Classe concrète non abstraite devrait être auto-enregistrée
        service = container.resolve(SimpleService)
        assert isinstance(service, SimpleService)

        # Interface sans implémentation devrait lever une erreur
        with pytest.raises(ValueError, match="Service .* not registered"):
            container.resolve(ITestService)


class TestMockContainer:
    """Tests du container de mocks"""

    def test_register_mock(self):
        """Test d'enregistrement de mocks"""
        container = MockContainer()
        mock_service = SimpleService()

        container.register_mock(SimpleService, mock_service)

        resolved = container.resolve(SimpleService)
        assert resolved is mock_service

    def test_mock_with_interface(self):
        """Test de mock avec interface"""
        container = MockContainer()
        mock_service = SimpleService()

        container.register_mock(ITestService, mock_service)

        resolved = container.resolve(ITestService)
        assert resolved is mock_service


class TestServiceDescriptor:
    """Tests du descripteur de service"""

    def test_creation(self):
        """Test de création du descripteur"""
        descriptor = ServiceDescriptor(
            interface=ITestService,
            implementation=SimpleService,
            lifetime=LifetimeScope.SINGLETON
        )

        assert descriptor.interface == ITestService
        assert descriptor.implementation == SimpleService
        assert descriptor.lifetime == LifetimeScope.SINGLETON
        assert descriptor.instance is None

    def test_with_factory(self):
        """Test avec factory"""
        def factory():
            return SimpleService()

        descriptor = ServiceDescriptor(
            interface=SimpleService,
            implementation=SimpleService,
            factory=factory
        )

        assert descriptor.factory is factory


class TestDecorators:
    """Tests des décorateurs"""

    def test_injectable_decorator(self):
        """Test du décorateur injectable"""
        @injectable
        class TestClass:
            pass

        assert hasattr(TestClass, '__injectable__')
        assert TestClass.__injectable__ is True

    def test_singleton_decorator(self):
        """Test du décorateur singleton"""
        @singleton
        class TestClass:
            pass

        assert hasattr(TestClass, '__injectable__')
        assert hasattr(TestClass, '__lifetime__')
        assert TestClass.__lifetime__ == LifetimeScope.SINGLETON

    def test_transient_decorator(self):
        """Test du décorateur transient"""
        @transient
        class TestClass:
            pass

        assert TestClass.__lifetime__ == LifetimeScope.TRANSIENT

    def test_scoped_decorator(self):
        """Test du décorateur scoped"""
        @scoped
        class TestClass:
            pass

        assert TestClass.__lifetime__ == LifetimeScope.SCOPED