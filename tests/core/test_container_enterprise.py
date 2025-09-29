"""
Tests for Enterprise DI Container
=================================

Tests pour le container d'injection de dépendances entreprise.
"""

import pytest
from unittest.mock import Mock, patch
import threading
import time
from typing import Protocol, Optional
from decimal import Decimal

from qframe.core.container_enterprise import (
    EnterpriseContainer, ServiceDescriptor, Injectable,
    ILifetimeScope, LifetimeScope, DependencyScope,
    CircularDependencyError, ServiceNotFoundError
)


class TestEnterpriseContainer:
    """Tests pour EnterpriseContainer."""

    @pytest.fixture
    def enterprise_container(self):
        return EnterpriseContainer()

    def test_container_initialization(self, enterprise_container):
        """Test initialisation container entreprise."""
        assert enterprise_container is not None
        assert hasattr(enterprise_container, 'resolve')
        assert hasattr(enterprise_container, 'register_singleton')
        assert hasattr(enterprise_container, 'register_transient')

    def test_singleton_registration(self, enterprise_container):
        """Test enregistrement singleton."""
        class TestService:
            def __init__(self):
                self.created_at = time.time()

            def process(self):
                return "processed"

        enterprise_container.register_singleton(TestService, TestService)

        # Deux résolutions doivent retourner la même instance
        service1 = enterprise_container.resolve(TestService)
        service2 = enterprise_container.resolve(TestService)

        assert service1 is service2
        assert service1.created_at == service2.created_at

    def test_transient_registration(self, enterprise_container):
        """Test enregistrement transient."""
        class TransientService:
            def __init__(self):
                self.created_at = time.time()

        enterprise_container.register_transient(TransientService, TransientService)

        # Deux résolutions doivent retourner des instances différentes
        service1 = enterprise_container.resolve(TransientService)
        time.sleep(0.001)  # Petit délai pour différencier les timestamps
        service2 = enterprise_container.resolve(TransientService)

        assert service1 is not service2
        assert service1.created_at != service2.created_at

    def test_dependency_injection(self, enterprise_container):
        """Test injection de dépendances."""
        class DatabaseService:
            def connect(self):
                return "connected"

        class APIService:
            def __init__(self, db: DatabaseService):
                self.db = db

        enterprise_container.register_singleton(DatabaseService, DatabaseService)
        enterprise_container.register_transient(APIService, APIService)

        # Résoudre service avec dépendance
        api_service = enterprise_container.resolve(APIService)

        assert api_service is not None
        assert api_service.db is not None
        assert api_service.db.connect() == "connected"

    def test_circular_dependency_detection(self, enterprise_container):
        """Test détection de dépendances circulaires."""
        class ServiceA:
            def __init__(self, service_b):
                self.service_b = service_b

        class ServiceB:
            def __init__(self, service_a):
                self.service_a = service_a

        enterprise_container.register_transient(ServiceA, ServiceA)
        enterprise_container.register_transient(ServiceB, ServiceB)

        # Devrait lever une exception de dépendance circulaire
        with pytest.raises(CircularDependencyError):
            enterprise_container.resolve(ServiceA)
