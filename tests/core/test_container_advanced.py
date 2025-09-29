"""
Tests for DI Container (Advanced)
=================================

Tests avancés pour le container d'injection de dépendances.
"""

import pytest
from unittest.mock import Mock
import threading
import time
from typing import Protocol

from qframe.core.container import DIContainer, injectable, LifetimeScope


class TestAdvancedDIContainer:
    """Tests avancés pour DIContainer."""

    @pytest.fixture
    def container(self):
        return DIContainer()

    def test_singleton_lifecycle(self, container):
        """Test lifecycle singleton."""
        @injectable
        class SingletonService:
            def __init__(self):
                self.created_at = time.time()

        container.register(SingletonService, SingletonService, LifetimeScope.SINGLETON)

        # Deux résolutions devraient retourner la même instance
        instance1 = container.resolve(SingletonService)
        instance2 = container.resolve(SingletonService)

        assert instance1 is instance2
        assert instance1.created_at == instance2.created_at

    def test_transient_lifecycle(self, container):
        """Test lifecycle transient."""
        @injectable
        class TransientService:
            def __init__(self):
                self.created_at = time.time()

        container.register(TransientService, TransientService, LifetimeScope.TRANSIENT)

        # Petite pause pour s'assurer que les timestamps diffèrent
        instance1 = container.resolve(TransientService)
        time.sleep(0.001)
        instance2 = container.resolve(TransientService)

        assert instance1 is not instance2
        assert instance1.created_at != instance2.created_at

    def test_scoped_lifecycle(self, container):
        """Test lifecycle scoped."""
        @injectable
        class ScopedService:
            def __init__(self):
                self.created_at = time.time()

        container.register(ScopedService, ScopedService, LifetimeScope.SCOPED)

        # Dans le même scope
        with container.create_scope() as scope:
            instance1 = scope.resolve(ScopedService)
            instance2 = scope.resolve(ScopedService)
            assert instance1 is instance2

        # Dans un nouveau scope
        with container.create_scope() as scope:
            instance3 = scope.resolve(ScopedService)
            assert instance3 is not instance1

    def test_dependency_injection_chain(self, container):
        """Test chaîne d'injection de dépendances."""
        @injectable
        class DatabaseService:
            def connect(self):
                return "connected"

        @injectable
        class RepositoryService:
            def __init__(self, db: DatabaseService):
                self.db = db

        @injectable
        class BusinessService:
            def __init__(self, repo: RepositoryService):
                self.repo = repo

        container.register(DatabaseService, DatabaseService)
        container.register(RepositoryService, RepositoryService)
        container.register(BusinessService, BusinessService)

        business = container.resolve(BusinessService)

        assert business is not None
        assert business.repo is not None
        assert business.repo.db is not None
        assert business.repo.db.connect() == "connected"

    def test_circular_dependency_detection(self, container):
        """Test détection de dépendances circulaires."""
        @injectable
        class ServiceA:
            def __init__(self, service_b):
                self.service_b = service_b

        @injectable
        class ServiceB:
            def __init__(self, service_a):
                self.service_a = service_a

        container.register(ServiceA, ServiceA)
        container.register(ServiceB, ServiceB)

        # Devrait détecter la dépendance circulaire
        with pytest.raises(Exception, match="circular.*dependency|dependency.*cycle"):
            container.resolve(ServiceA)

    def test_interface_registration(self, container):
        """Test enregistrement avec interfaces."""
        class IDataProvider(Protocol):
            def get_data(self) -> str:
                ...

        @injectable
        class MockDataProvider:
            def get_data(self) -> str:
                return "mock data"

        @injectable
        class RealDataProvider:
            def get_data(self) -> str:
                return "real data"

        # Enregistrer avec interface
        container.register(IDataProvider, MockDataProvider)

        @injectable
        class ConsumerService:
            def __init__(self, provider: IDataProvider):
                self.provider = provider

        container.register(ConsumerService, ConsumerService)

        consumer = container.resolve(ConsumerService)
        assert consumer.provider.get_data() == "mock data"

        # Changer l'implémentation
        container.register(IDataProvider, RealDataProvider)
        new_consumer = container.resolve(ConsumerService)
        assert new_consumer.provider.get_data() == "real data"

    def test_factory_registration(self, container):
        """Test enregistrement avec factory."""
        class ConfigurableService:
            def __init__(self, config: dict):
                self.config = config

        def service_factory() -> ConfigurableService:
            return ConfigurableService(config={"mode": "test"})

        container.register_factory(ConfigurableService, service_factory)

        service = container.resolve(ConfigurableService)
        assert service.config["mode"] == "test"

    def test_conditional_registration(self, container):
        """Test enregistrement conditionnel."""
        class ILogger(Protocol):
            def log(self, message: str):
                ...

        @injectable
        class ConsoleLogger:
            def log(self, message: str):
                print(f"Console: {message}")

        @injectable
        class FileLogger:
            def log(self, message: str):
                pass  # Simuler écriture fichier

        # Enregistrement conditionnel basé sur environnement
        environment = "development"
        if environment == "development":
            container.register(ILogger, ConsoleLogger)
        else:
            container.register(ILogger, FileLogger)

        logger = container.resolve(ILogger)
        assert isinstance(logger, ConsoleLogger)

    def test_thread_safety(self, container):
        """Test thread safety du container."""
        @injectable
        class ThreadSafeService:
            def __init__(self):
                self.thread_id = threading.current_thread().ident

        container.register(ThreadSafeService, ThreadSafeService, LifetimeScope.SINGLETON)

        results = []

        def resolve_in_thread():
            service = container.resolve(ThreadSafeService)
            results.append(service)

        # Créer plusieurs threads qui résolvent simultanément
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=resolve_in_thread)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Tous devraient avoir la même instance (singleton)
        assert len(results) == 10
        assert all(service is results[0] for service in results)

    def test_container_hierarchy(self, container):
        """Test hiérarchie de containers."""
        # Container parent
        parent_container = container

        @injectable
        class ParentService:
            def get_value(self):
                return "parent"

        parent_container.register(ParentService, ParentService)

        # Container enfant
        child_container = DIContainer(parent=parent_container)

        @injectable
        class ChildService:
            def __init__(self, parent: ParentService):
                self.parent = parent

            def get_value(self):
                return f"child with {self.parent.get_value()}"

        child_container.register(ChildService, ChildService)

        # Le container enfant devrait résoudre depuis le parent
        child_service = child_container.resolve(ChildService)
        assert child_service.get_value() == "child with parent"

    def test_container_disposal(self, container):
        """Test disposal du container."""
        disposed_services = []

        @injectable
        class DisposableService:
            def __init__(self):
                self.disposed = False

            def dispose(self):
                self.disposed = True
                disposed_services.append(self)

        container.register(DisposableService, DisposableService, LifetimeScope.SINGLETON)

        # Résoudre le service
        service = container.resolve(DisposableService)
        assert not service.disposed

        # Disposer le container
        container.dispose()

        # Le service devrait être disposé
        assert service.disposed
        assert service in disposed_services

    def test_lazy_initialization(self, container):
        """Test initialisation paresseuse."""
        initialization_count = 0

        @injectable
        class LazyService:
            def __init__(self):
                nonlocal initialization_count
                initialization_count += 1

        container.register(LazyService, LazyService, LifetimeScope.SINGLETON)

        # L'enregistrement ne devrait pas créer l'instance
        assert initialization_count == 0

        # Première résolution devrait créer l'instance
        service1 = container.resolve(LazyService)
        assert initialization_count == 1

        # Deuxième résolution ne devrait pas créer d'instance (singleton)
        service2 = container.resolve(LazyService)
        assert initialization_count == 1
        assert service1 is service2

    def test_generic_types(self, container):
        """Test types génériques."""
        from typing import Generic, TypeVar, List

        T = TypeVar('T')

        class Repository(Generic[T]):
            def __init__(self):
                self.items: List[T] = []

            def add(self, item: T):
                self.items.append(item)

        class User:
            def __init__(self, name: str):
                self.name = name

        @injectable
        class UserRepository(Repository[User]):
            pass

        container.register(UserRepository, UserRepository)

        user_repo = container.resolve(UserRepository)
        user_repo.add(User("Test User"))

        assert len(user_repo.items) == 1
        assert user_repo.items[0].name == "Test User"

    def test_multiple_implementations(self, container):
        """Test multiples implémentations."""
        class INotificationService(Protocol):
            def send(self, message: str):
                ...

        @injectable
        class EmailNotificationService:
            def send(self, message: str):
                return f"Email: {message}"

        @injectable
        class SMSNotificationService:
            def send(self, message: str):
                return f"SMS: {message}"

        # Enregistrer plusieurs implémentations
        container.register_named(INotificationService, "email", EmailNotificationService)
        container.register_named(INotificationService, "sms", SMSNotificationService)

        # Résoudre par nom
        email_service = container.resolve_named(INotificationService, "email")
        sms_service = container.resolve_named(INotificationService, "sms")

        assert email_service.send("test") == "Email: test"
        assert sms_service.send("test") == "SMS: test"

    def test_decorator_injection(self, container):
        """Test injection avec décorateurs."""
        @injectable
        class LoggingService:
            def log(self, message: str):
                return f"Log: {message}"

        def with_logging(service_type):
            def decorator(cls):
                original_init = cls.__init__

                def new_init(self, logger: LoggingService, *args, **kwargs):
                    self.logger = logger
                    original_init(self, *args, **kwargs)

                cls.__init__ = new_init
                return cls
            return decorator

        @with_logging(LoggingService)
        @injectable
        class BusinessService:
            def __init__(self):
                pass

            def do_work(self):
                return self.logger.log("Work done")

        container.register(LoggingService, LoggingService)
        container.register(BusinessService, BusinessService)

        business = container.resolve(BusinessService)
        result = business.do_work()

        assert result == "Log: Work done"

    def test_container_performance(self, container):
        """Test performance du container."""
        @injectable
        class SimpleService:
            def __init__(self):
                self.value = 42

        container.register(SimpleService, SimpleService, LifetimeScope.SINGLETON)

        # Mesurer temps de résolution
        start_time = time.time()

        for _ in range(1000):
            service = container.resolve(SimpleService)

        resolution_time = time.time() - start_time

        # Devrait être rapide (moins de 100ms pour 1000 résolutions)
        assert resolution_time < 0.1
        assert service.value == 42

    def test_container_introspection(self, container):
        """Test introspection du container."""
        @injectable
        class ServiceA:
            pass

        @injectable
        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a

        container.register(ServiceA, ServiceA)
        container.register(ServiceB, ServiceB)

        # Vérifier enregistrements
        registrations = container.get_registrations()
        assert ServiceA in registrations
        assert ServiceB in registrations

        # Vérifier dépendances
        dependencies = container.get_dependencies(ServiceB)
        assert ServiceA in dependencies

        # Vérifier graphe de dépendances
        dependency_graph = container.build_dependency_graph()
        assert ServiceB in dependency_graph
        assert ServiceA in dependency_graph[ServiceB]