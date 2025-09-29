"""
Tests d'Exécution Réelle - Core Container
=========================================

Tests qui EXÉCUTENT vraiment le code qframe.core.container
"""

import pytest
import threading
import time
from typing import Protocol
from unittest.mock import Mock, patch

from qframe.core.container import (
    DIContainer, LifetimeScope, ServiceDescriptor, ScopeManager,
    injectable, singleton, transient, scoped, auto_register,
    factory, MockContainer, get_container, set_container,
    configure_services
)


class TestServiceDescriptor:
    """Tests d'exécution réelle pour ServiceDescriptor."""

    def test_service_descriptor_creation_execution(self):
        """Test création complète avec TOUTES les propriétés."""

        class TestInterface:
            pass

        class TestImplementation(TestInterface):
            pass

        def test_factory():
            return TestImplementation()

        # Exécuter création avec tous les paramètres
        descriptor = ServiceDescriptor(
            interface=TestInterface,
            implementation=TestImplementation,
            lifetime=LifetimeScope.SINGLETON,
            factory=test_factory
        )

        # Exécuter tous les accesseurs
        assert descriptor.interface == TestInterface
        assert descriptor.implementation == TestImplementation
        assert descriptor.lifetime == LifetimeScope.SINGLETON
        assert descriptor.factory == test_factory
        assert descriptor.instance is None

        # Tester modification instance
        test_instance = TestImplementation()
        descriptor.instance = test_instance
        assert descriptor.instance == test_instance


class TestDIContainerExecution:
    """Tests d'exécution réelle pour DIContainer."""

    @pytest.fixture
    def container(self):
        """Container propre pour chaque test."""
        return DIContainer()

    def test_container_creation_execution(self, container):
        """Test création et état initial."""
        # Exécuter tous les accesseurs
        assert isinstance(container._services, dict)
        assert isinstance(container._instances, dict)
        assert isinstance(container._lock, threading.RLock)
        assert isinstance(container._scoped_instances, dict)
        assert container._current_scope is None
        assert isinstance(container._resolution_stack, list)

        # État initial vide
        assert len(container._services) == 0
        assert len(container._instances) == 0
        assert len(container._scoped_instances) == 0
        assert len(container._resolution_stack) == 0

    def test_singleton_registration_execution(self, container):
        """Test enregistrement singleton complet."""

        @injectable
        class SingletonService:
            def __init__(self):
                self.created_at = time.time()
                self.value = "singleton"

        # Exécuter enregistrement
        result = container.register_singleton(SingletonService, SingletonService)

        # Vérifier fluent interface
        assert result is container

        # Vérifier enregistrement
        assert container.is_registered(SingletonService)
        registrations = container.get_registrations()
        assert SingletonService in registrations

        descriptor = registrations[SingletonService]
        assert descriptor.interface == SingletonService
        assert descriptor.implementation == SingletonService
        assert descriptor.lifetime == LifetimeScope.SINGLETON

    def test_transient_registration_execution(self, container):
        """Test enregistrement transient complet."""

        @injectable
        class TransientService:
            def __init__(self):
                self.id = id(self)

        # Exécuter enregistrement
        result = container.register_transient(TransientService, TransientService)

        # Vérifier enregistrement
        assert result is container
        assert container.is_registered(TransientService)

        descriptor = container.get_registrations()[TransientService]
        assert descriptor.lifetime == LifetimeScope.TRANSIENT

    def test_scoped_registration_execution(self, container):
        """Test enregistrement scoped complet."""

        @injectable
        class ScopedService:
            def __init__(self):
                self.scope_id = None

        # Exécuter enregistrement
        result = container.register_scoped(ScopedService, ScopedService)

        # Vérifier enregistrement
        assert result is container
        assert container.is_registered(ScopedService)

        descriptor = container.get_registrations()[ScopedService]
        assert descriptor.lifetime == LifetimeScope.SCOPED

    def test_factory_registration_execution(self, container):
        """Test enregistrement avec factory."""

        class FactoryService:
            def __init__(self, value: str):
                self.value = value

        @factory
        def create_factory_service():
            return FactoryService("factory_created")

        # Exécuter enregistrement factory
        container.register_singleton(FactoryService, factory=create_factory_service)

        # Vérifier enregistrement
        assert container.is_registered(FactoryService)
        descriptor = container.get_registrations()[FactoryService]
        assert descriptor.factory == create_factory_service

    def test_singleton_resolution_execution(self, container):
        """Test résolution singleton avec vraie instance."""

        @injectable
        class SingletonService:
            def __init__(self):
                self.created_at = time.time()
                self.call_count = 0

            def increment(self):
                self.call_count += 1
                return self.call_count

        # Enregistrer et résoudre
        container.register_singleton(SingletonService, SingletonService)

        # Exécuter première résolution
        instance1 = container.resolve(SingletonService)
        assert isinstance(instance1, SingletonService)
        assert instance1.increment() == 1

        # Exécuter deuxième résolution - doit être la même instance
        instance2 = container.resolve(SingletonService)
        assert instance2 is instance1
        assert instance2.increment() == 2

    def test_transient_resolution_execution(self, container):
        """Test résolution transient avec nouvelles instances."""

        @injectable
        class TransientService:
            def __init__(self):
                self.id = id(self)

        # Enregistrer et résoudre
        container.register_transient(TransientService, TransientService)

        # Exécuter résolutions multiples
        instance1 = container.resolve(TransientService)
        instance2 = container.resolve(TransientService)

        # Doivent être des instances différentes
        assert isinstance(instance1, TransientService)
        assert isinstance(instance2, TransientService)
        assert instance1 is not instance2
        assert instance1.id != instance2.id

    def test_dependency_injection_execution(self, container):
        """Test injection de dépendances réelle."""

        @injectable
        class ServiceA:
            def __init__(self):
                self.name = "ServiceA"

        @injectable
        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a
                self.name = "ServiceB"

        @injectable
        class ServiceC:
            def __init__(self, service_a: ServiceA, service_b: ServiceB):
                self.service_a = service_a
                self.service_b = service_b
                self.name = "ServiceC"

        # Enregistrer tous les services
        container.register_singleton(ServiceA, ServiceA)
        container.register_transient(ServiceB, ServiceB)
        container.register_transient(ServiceC, ServiceC)

        # Exécuter résolution avec injection automatique
        service_c = container.resolve(ServiceC)

        # Vérifier injection complète
        assert isinstance(service_c, ServiceC)
        assert isinstance(service_c.service_a, ServiceA)
        assert isinstance(service_c.service_b, ServiceB)
        assert service_c.service_b.service_a is service_c.service_a  # Singleton

        # Vérifier que les services sont opérationnels
        assert service_c.name == "ServiceC"
        assert service_c.service_a.name == "ServiceA"
        assert service_c.service_b.name == "ServiceB"

    def test_circular_dependency_detection_execution(self, container):
        """Test détection dépendances circulaires."""

        @injectable
        class ServiceX:
            def __init__(self, service_y: 'ServiceY'):
                self.service_y = service_y

        @injectable
        class ServiceY:
            def __init__(self, service_x: ServiceX):
                self.service_x = service_x

        # Enregistrer services circulaires
        container.register_transient(ServiceX, ServiceX)
        container.register_transient(ServiceY, ServiceY)

        # Exécuter résolution - doit détecter la circularité
        with pytest.raises(ValueError, match="Circular dependency detected"):
            container.resolve(ServiceX)

    def test_scoped_resolution_execution(self, container):
        """Test résolution scoped avec scopes."""

        @injectable
        class ScopedService:
            def __init__(self):
                self.id = id(self)
                self.value = 42

        container.register_scoped(ScopedService, ScopedService)

        # Test sans scope - doit échouer
        with pytest.raises(ValueError, match="No active scope"):
            container.resolve(ScopedService)

        # Exécuter avec scope
        with container.create_scope("scope1") as scope1:
            # Première résolution dans scope1
            instance1 = scope1.resolve(ScopedService)
            assert isinstance(instance1, ScopedService)

            # Deuxième résolution dans scope1 - même instance
            instance2 = scope1.resolve(ScopedService)
            assert instance2 is instance1

        # Nouveau scope - nouvelle instance
        with container.create_scope("scope2") as scope2:
            instance3 = scope2.resolve(ScopedService)
            assert isinstance(instance3, ScopedService)
            assert instance3 is not instance1

    def test_factory_resolution_execution(self, container):
        """Test résolution avec factory."""

        class FactoryService:
            def __init__(self, config_value: str):
                self.config = config_value

        def create_configured_service():
            return FactoryService("factory_config")

        # Enregistrer avec factory
        container.register_singleton(FactoryService, factory=create_configured_service)

        # Exécuter résolution
        instance = container.resolve(FactoryService)
        assert isinstance(instance, FactoryService)
        assert instance.config == "factory_config"

    def test_auto_registration_execution(self, container):
        """Test auto-enregistrement des classes non déclarées."""

        class AutoService:
            def __init__(self):
                self.auto_registered = True

        # Classe non enregistrée - doit être auto-enregistrée
        instance = container.resolve(AutoService)
        assert isinstance(instance, AutoService)
        assert instance.auto_registered is True

        # Vérifier qu'elle a été enregistrée
        assert container.is_registered(AutoService)

    def test_thread_safety_execution(self, container):
        """Test thread safety du container."""

        @injectable
        class ThreadSafeService:
            def __init__(self):
                self.thread_id = threading.current_thread().ident
                time.sleep(0.01)  # Simuler du travail

        container.register_singleton(ThreadSafeService, ThreadSafeService)

        results = []

        def resolve_service():
            instance = container.resolve(ThreadSafeService)
            results.append(instance)

        # Exécuter résolutions simultanées
        threads = []
        for i in range(5):
            thread = threading.Thread(target=resolve_service)
            threads.append(thread)
            thread.start()

        # Attendre completion
        for thread in threads:
            thread.join()

        # Vérifier singleton même avec threads
        assert len(results) == 5
        first_instance = results[0]
        for instance in results[1:]:
            assert instance is first_instance

    def test_container_clear_execution(self, container):
        """Test nettoyage du container."""

        class DisposableService:
            def __init__(self):
                self.disposed = False

            def dispose(self):
                self.disposed = True

        # Enregistrer et résoudre
        container.register_singleton(DisposableService, DisposableService)
        instance = container.resolve(DisposableService)

        assert not instance.disposed
        assert container.is_registered(DisposableService)

        # Exécuter nettoyage
        container.clear()

        # Vérifier nettoyage
        assert instance.disposed
        assert not container.is_registered(DisposableService)
        assert len(container.get_registrations()) == 0


class TestScopeManagerExecution:
    """Tests d'exécution réelle pour ScopeManager."""

    def test_scope_manager_context_execution(self):
        """Test context manager complet."""
        container = DIContainer()

        @injectable
        class ScopedService:
            def __init__(self):
                self.created = True

        container.register_scoped(ScopedService, ScopedService)

        # Exécuter avec context manager
        with container.create_scope("test_scope") as scope:
            assert isinstance(scope, ScopeManager)
            assert scope.scope_id == "test_scope"

            # Résoudre dans le scope
            instance = scope.resolve(ScopedService)
            assert isinstance(instance, ScopedService)
            assert instance.created

    def test_scope_cleanup_execution(self):
        """Test nettoyage automatique des scopes."""
        container = DIContainer()

        class DisposableScopedService:
            def __init__(self):
                self.disposed = False

            def dispose(self):
                self.disposed = True

        container.register_scoped(DisposableScopedService, DisposableScopedService)

        instance = None
        # Exécuter scope avec cleanup automatique
        with container.create_scope("cleanup_scope") as scope:
            instance = scope.resolve(DisposableScopedService)
            assert not instance.disposed

        # Vérifier cleanup après sortie du scope
        assert instance.disposed


class TestDecoratorsExecution:
    """Tests d'exécution réelle des décorateurs."""

    def test_injectable_decorator_execution(self):
        """Test décorateur injectable."""

        @injectable
        class InjectableClass:
            def __init__(self):
                self.injectable = True

        # Vérifier marquage
        assert hasattr(InjectableClass, '__injectable__')
        assert InjectableClass.__injectable__ is True

        # Vérifier fonctionnement
        instance = InjectableClass()
        assert instance.injectable is True

    def test_singleton_decorator_execution(self):
        """Test décorateur singleton."""

        @singleton
        class SingletonClass:
            def __init__(self):
                self.value = "singleton"

        # Vérifier marquages
        assert hasattr(SingletonClass, '__injectable__')
        assert hasattr(SingletonClass, '__lifetime__')
        assert SingletonClass.__lifetime__ == LifetimeScope.SINGLETON

        # Test avec container
        container = DIContainer()
        auto_register(container, SingletonClass)

        instance1 = container.resolve(SingletonClass)
        instance2 = container.resolve(SingletonClass)
        assert instance1 is instance2

    def test_transient_decorator_execution(self):
        """Test décorateur transient."""

        @transient
        class TransientClass:
            def __init__(self):
                self.id = id(self)

        # Vérifier marquages
        assert TransientClass.__lifetime__ == LifetimeScope.TRANSIENT

        # Test avec container
        container = DIContainer()
        auto_register(container, TransientClass)

        instance1 = container.resolve(TransientClass)
        instance2 = container.resolve(TransientClass)
        assert instance1 is not instance2

    def test_scoped_decorator_execution(self):
        """Test décorateur scoped."""

        @scoped
        class ScopedClass:
            def __init__(self):
                self.value = "scoped"

        # Vérifier marquages
        assert ScopedClass.__lifetime__ == LifetimeScope.SCOPED

        # Test avec container
        container = DIContainer()
        auto_register(container, ScopedClass)

        with container.create_scope() as scope:
            instance1 = scope.resolve(ScopedClass)
            instance2 = scope.resolve(ScopedClass)
            assert instance1 is instance2

    def test_factory_decorator_execution(self):
        """Test décorateur factory."""

        @factory
        def create_service():
            return {"created": "by_factory"}

        # Vérifier marquage
        assert hasattr(create_service, '__is_factory__')
        assert create_service.__is_factory__ is True

        # Vérifier fonctionnement
        result = create_service()
        assert result["created"] == "by_factory"


class TestAutoRegistrationExecution:
    """Tests d'exécution réelle de l'auto-enregistrement."""

    def test_auto_register_single_class_execution(self):
        """Test auto-enregistrement d'une classe."""
        container = DIContainer()

        @singleton
        class AutoSingletonService:
            def __init__(self):
                self.auto_registered = True

        # Exécuter auto-enregistrement
        auto_register(container, AutoSingletonService)

        # Vérifier enregistrement
        assert container.is_registered(AutoSingletonService)

        descriptor = container.get_registrations()[AutoSingletonService]
        assert descriptor.lifetime == LifetimeScope.SINGLETON

        # Vérifier résolution
        instance = container.resolve(AutoSingletonService)
        assert instance.auto_registered is True

    def test_auto_register_module_execution(self):
        """Test auto-enregistrement d'un module."""
        import types

        # Créer un module simulé
        mock_module = types.ModuleType("mock_module")

        @injectable
        class ModuleService1:
            pass

        @singleton
        class ModuleService2:
            pass

        class RegularClass:  # Pas injectable
            pass

        # Ajouter au module
        mock_module.ModuleService1 = ModuleService1
        mock_module.ModuleService2 = ModuleService2
        mock_module.RegularClass = RegularClass

        container = DIContainer()

        # Exécuter auto-enregistrement du module
        auto_register(container, mock_module)

        # Vérifier enregistrements
        assert container.is_registered(ModuleService1)
        assert container.is_registered(ModuleService2)
        assert not container.is_registered(RegularClass)  # Pas injectable


class TestMockContainerExecution:
    """Tests d'exécution réelle pour MockContainer."""

    def test_mock_container_execution(self):
        """Test container de mock complet."""
        mock_container = MockContainer()

        class ServiceInterface:
            def get_value(self):
                raise NotImplementedError

        # Créer mock
        mock_service = Mock(spec=ServiceInterface)
        mock_service.get_value.return_value = "mocked_value"

        # Exécuter enregistrement mock
        result = mock_container.register_mock(ServiceInterface, mock_service)
        assert result is mock_container

        # Vérifier résolution
        resolved = mock_container.resolve(ServiceInterface)
        assert resolved is mock_service
        assert resolved.get_value() == "mocked_value"


class TestGlobalContainerExecution:
    """Tests d'exécution réelle du container global."""

    def test_get_container_execution(self):
        """Test récupération container global."""
        # Reset global container
        import qframe.core.container as container_module
        container_module._global_container = None

        # Exécuter récupération
        container1 = get_container()
        assert isinstance(container1, DIContainer)

        # Deuxième appel - même instance
        container2 = get_container()
        assert container2 is container1

    def test_set_container_execution(self):
        """Test définition container global."""
        custom_container = DIContainer()

        # Exécuter définition
        set_container(custom_container)

        # Vérifier récupération
        retrieved = get_container()
        assert retrieved is custom_container

    @patch('qframe.core.container.configure_services')
    def test_configure_services_execution(self, mock_configure):
        """Test configuration des services."""
        container = DIContainer()

        # Exécuter configuration (mockée pour éviter dépendances)
        mock_configure.return_value = None
        configure_services(container)

        # Vérifier appel
        mock_configure.assert_called_once_with(container)


class TestIntegrationContainerExecution:
    """Tests d'intégration complète du container."""

    def test_complex_dependency_graph_execution(self):
        """Test graphe de dépendances complexe."""
        container = DIContainer()

        # Repository layer
        @injectable
        class DataRepository:
            def __init__(self):
                self.data = {"key": "value"}

            def get_data(self, key):
                return self.data.get(key)

        # Service layer
        @injectable
        class BusinessService:
            def __init__(self, repo: DataRepository):
                self.repo = repo

            def process_data(self, key):
                return f"processed_{self.repo.get_data(key)}"

        # Controller layer
        @injectable
        class ApiController:
            def __init__(self, service: BusinessService):
                self.service = service

            def handle_request(self, key):
                return {
                    "result": self.service.process_data(key),
                    "status": "success"
                }

        # Enregistrer toute la stack
        container.register_singleton(DataRepository, DataRepository)
        container.register_transient(BusinessService, BusinessService)
        container.register_transient(ApiController, ApiController)

        # Exécuter résolution complète
        controller = container.resolve(ApiController)

        # Vérifier fonctionnement end-to-end
        result = controller.handle_request("key")
        assert result["result"] == "processed_value"
        assert result["status"] == "success"

        # Vérifier partage singleton
        controller2 = container.resolve(ApiController)
        assert controller2.service.repo is controller.service.repo

    def test_factory_with_dependencies_execution(self):
        """Test factory avec dépendances."""
        container = DIContainer()

        @injectable
        class ConfigService:
            def __init__(self):
                self.config = {"db_url": "localhost:5432"}

        class DatabaseConnection:
            def __init__(self, url: str, config_service: ConfigService):
                self.url = url
                self.config_service = config_service
                self.connected = True

        def create_db_connection():
            # Factory peut résoudre ses propres dépendances via le container
            config_service = container.resolve(ConfigService)
            return DatabaseConnection(
                url=config_service.config["db_url"],
                config_service=config_service
            )

        # Enregistrer
        container.register_singleton(ConfigService, ConfigService)
        container.register_singleton(DatabaseConnection, factory=create_db_connection)

        # Exécuter résolution
        db_conn = container.resolve(DatabaseConnection)

        # Vérifier factory avec dépendances
        assert isinstance(db_conn, DatabaseConnection)
        assert db_conn.url == "localhost:5432"
        assert isinstance(db_conn.config_service, ConfigService)
        assert db_conn.connected is True

    def test_multi_scope_execution(self):
        """Test gestion de scopes multiples."""
        container = DIContainer()

        @injectable
        class ScopedCounter:
            def __init__(self):
                self.count = 0

            def increment(self):
                self.count += 1
                return self.count

        container.register_scoped(ScopedCounter, ScopedCounter)

        # Exécuter scopes parallèles
        results = {}

        with container.create_scope("scope_A") as scope_a:
            counter_a1 = scope_a.resolve(ScopedCounter)
            counter_a2 = scope_a.resolve(ScopedCounter)

            # Même instance dans le scope
            assert counter_a1 is counter_a2

            results["scope_a"] = [
                counter_a1.increment(),  # 1
                counter_a2.increment(),  # 2
            ]

        with container.create_scope("scope_B") as scope_b:
            counter_b = scope_b.resolve(ScopedCounter)

            # Nouvelle instance dans nouveau scope
            results["scope_b"] = [
                counter_b.increment(),  # 1 (nouveau compteur)
                counter_b.increment(),  # 2
            ]

        # Vérifier isolation des scopes
        assert results["scope_a"] == [1, 2]
        assert results["scope_b"] == [1, 2]  # Nouveau compteur, repart de 0