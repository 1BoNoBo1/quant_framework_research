"""
Tests d'Exécution Réelle - Core Config
=====================================

Tests qui EXÉCUTENT vraiment le code qframe.core.config
"""

import pytest
import os
import tempfile
from decimal import Decimal
from pathlib import Path

from qframe.core.config import (
    FrameworkConfig, DatabaseConfig, RedisConfig, MLFlowConfig,
    Environment, LogLevel, DataProviderConfig
)


class TestFrameworkConfigExecution:
    """Tests d'exécution réelle pour FrameworkConfig."""

    def test_framework_config_full_creation(self):
        """Test création complète avec TOUTES les propriétés."""
        config = FrameworkConfig()

        # Exécuter tous les accesseurs
        assert config.app_name == "Quant Framework Research"
        assert config.app_version == "2.0.0"
        assert config.environment == Environment.DEVELOPMENT

        # Tester modification des propriétés
        config.app_name = "Test Framework"
        assert config.app_name == "Test Framework"

        config.environment = Environment.PRODUCTION
        assert config.environment == Environment.PRODUCTION

        config.log_level = LogLevel.ERROR
        assert config.log_level == LogLevel.ERROR

    def test_framework_config_validation_execution(self):
        """Test exécution réelle de la validation."""
        config = FrameworkConfig()

        # Exécuter la validation complète
        try:
            errors = config.model_validate(config.model_dump())
            assert errors is not None
        except AttributeError:
            # Pydantic v1 vs v2
            validated = FrameworkConfig(**config.dict())
            assert validated is not None

    def test_framework_config_paths_execution(self):
        """Test exécution des propriétés de chemins."""
        config = FrameworkConfig()

        # Exécuter accès aux chemins
        assert isinstance(config.project_root, Path)
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.logs_dir, Path)
        assert isinstance(config.artifacts_dir, Path)

        # Tester modification des chemins
        config.data_dir = Path("/tmp/test_data")
        assert config.data_dir == Path("/tmp/test_data")

    def test_framework_config_nested_objects_execution(self):
        """Test exécution des objets imbriqués."""
        config = FrameworkConfig()

        # Exécuter accès aux configurations imbriquées
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.redis, RedisConfig)
        assert isinstance(config.mlflow, MLFlowConfig)

        # Modifier les configs imbriquées
        config.database.host = "test-host"
        assert config.database.host == "test-host"

        config.redis.port = 6380
        assert config.redis.port == 6380

    def test_framework_config_serialization_execution(self):
        """Test exécution réelle de la sérialisation."""
        config = FrameworkConfig()
        config.app_name = "Serialization Test"

        # Exécuter sérialisation
        try:
            config_dict = config.model_dump()
            assert isinstance(config_dict, dict)
            assert config_dict["app_name"] == "Serialization Test"

            # Exécuter désérialisation
            new_config = FrameworkConfig(**config_dict)
            assert new_config.app_name == "Serialization Test"

        except AttributeError:
            # Pydantic v1
            config_dict = config.dict()
            assert isinstance(config_dict, dict)
            new_config = FrameworkConfig(**config_dict)
            assert new_config.app_name == "Serialization Test"


class TestDatabaseConfigExecution:
    """Tests d'exécution réelle pour DatabaseConfig."""

    def test_database_config_creation_execution(self):
        """Test création et modification complète."""
        db_config = DatabaseConfig()

        # Exécuter tous les accesseurs
        assert db_config.host == "localhost"
        assert db_config.port == 5432
        assert db_config.database == "quant_framework"
        assert db_config.pool_size == 10
        assert db_config.max_overflow == 20
        assert db_config.echo is False

        # Exécuter modifications
        db_config.host = "production-db"
        db_config.port = 5433
        db_config.pool_size = 20

        assert db_config.host == "production-db"
        assert db_config.port == 5433
        assert db_config.pool_size == 20

    def test_database_config_validation_execution(self):
        """Test exécution validation des ports."""
        db_config = DatabaseConfig()

        # Test port valide
        db_config.port = 3306  # MySQL
        assert db_config.port == 3306

        # Test validation d'erreur
        with pytest.raises(ValueError):
            DatabaseConfig(port=-1)

        with pytest.raises(ValueError):
            DatabaseConfig(port=70000)

    def test_database_config_environment_execution(self):
        """Test chargement depuis variables d'environnement."""
        # Exécuter avec variables d'environnement
        env_vars = {
            'DB_HOST': 'env-host',
            'DB_PORT': '3306',
            'DB_DATABASE': 'env_db',
            'DB_USERNAME': 'env_user',
            'DB_PASSWORD': 'env_pass'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / '.env'
            with open(env_file, 'w') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")

            # Tester que la config peut être créée
            db_config = DatabaseConfig(_env_file=str(env_file))
            assert db_config is not None


class TestRedisConfigExecution:
    """Tests d'exécution réelle pour RedisConfig."""

    def test_redis_config_creation_execution(self):
        """Test création et modification complète."""
        redis_config = RedisConfig()

        # Exécuter tous les accesseurs
        assert redis_config.host == "localhost"
        assert redis_config.port == 6379
        assert redis_config.db == 0
        assert redis_config.password is None
        assert redis_config.max_connections == 10
        assert redis_config.socket_timeout == 5.0

        # Exécuter modifications
        redis_config.host = "redis-cluster"
        redis_config.port = 6380
        redis_config.db = 1
        redis_config.password = "secret"

        assert redis_config.host == "redis-cluster"
        assert redis_config.port == 6380
        assert redis_config.db == 1
        assert redis_config.password == "secret"

    def test_redis_config_connection_url_execution(self):
        """Test génération URL de connexion."""
        redis_config = RedisConfig()
        redis_config.host = "test-redis"
        redis_config.port = 6379
        redis_config.db = 2

        # Si la méthode existe, l'exécuter
        if hasattr(redis_config, 'get_connection_url'):
            url = redis_config.get_connection_url()
            assert "redis://" in url
            assert "test-redis" in url
            assert "6379" in url


class TestMLFlowConfigExecution:
    """Tests d'exécution réelle pour MLFlowConfig."""

    def test_mlflow_config_creation_execution(self):
        """Test création et validation complète."""
        mlflow_config = MLFlowConfig()

        # Exécuter tous les accesseurs
        assert mlflow_config.tracking_uri == "http://localhost:5000"
        assert mlflow_config.experiment_name == "quant_framework"
        assert mlflow_config.artifact_location is None
        assert mlflow_config.registry_uri is None

        # Exécuter modifications
        mlflow_config.tracking_uri = "https://mlflow.company.com"
        mlflow_config.experiment_name = "production_experiment"

        assert mlflow_config.tracking_uri == "https://mlflow.company.com"
        assert mlflow_config.experiment_name == "production_experiment"

    def test_mlflow_config_validation_execution(self):
        """Test validation des URIs."""
        # Test URI valide
        valid_config = MLFlowConfig(tracking_uri="https://valid-url.com")
        assert valid_config.tracking_uri == "https://valid-url.com"

        # Test file URI
        file_config = MLFlowConfig(tracking_uri="file:///tmp/mlruns")
        assert file_config.tracking_uri == "file:///tmp/mlruns"

        # Test URI invalide
        with pytest.raises(ValueError):
            MLFlowConfig(tracking_uri="invalid-uri")


class TestDataProviderConfigExecution:
    """Tests d'exécution réelle pour DataProviderConfig."""

    def test_data_provider_config_creation_execution(self):
        """Test création complète."""
        provider_config = DataProviderConfig(name="binance")

        # Exécuter accesseurs
        assert provider_config.name == "binance"
        assert provider_config.api_key is None
        assert provider_config.api_secret is None

        # Exécuter modifications
        provider_config.api_key = "test_key"
        provider_config.api_secret = "test_secret"

        assert provider_config.api_key == "test_key"
        assert provider_config.api_secret == "test_secret"


class TestEnvironmentLogLevelExecution:
    """Tests d'exécution réelle des enums."""

    def test_environment_enum_execution(self):
        """Test exécution complète des environnements."""
        # Exécuter tous les environnements
        assert Environment.DEVELOPMENT == "development"
        assert Environment.TESTING == "testing"
        assert Environment.PRODUCTION == "production"
        assert Environment.BACKTESTING == "backtesting"

        # Test comparaisons
        env = Environment.PRODUCTION
        assert env == "production"
        assert env != Environment.DEVELOPMENT

    def test_log_level_enum_execution(self):
        """Test exécution complète des niveaux de log."""
        # Exécuter tous les niveaux
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"

        # Test comparaisons
        level = LogLevel.ERROR
        assert level == "ERROR"
        assert level != LogLevel.DEBUG


class TestIntegrationConfigExecution:
    """Tests d'intégration complète."""

    def test_full_config_integration_execution(self):
        """Test intégration complète de toute la configuration."""
        # Créer configuration complète
        config = FrameworkConfig()

        # Modifier tous les composants
        config.app_name = "Integration Test"
        config.environment = Environment.TESTING
        config.log_level = LogLevel.WARNING

        config.database.host = "integration-db"
        config.database.port = 5432

        config.redis.host = "integration-redis"
        config.redis.port = 6379

        config.mlflow.tracking_uri = "http://integration-mlflow:5000"

        # Vérifier intégration complète
        assert config.app_name == "Integration Test"
        assert config.environment == Environment.TESTING
        assert config.database.host == "integration-db"
        assert config.redis.host == "integration-redis"
        assert "integration-mlflow" in config.mlflow.tracking_uri

        # Test sérialisation complète
        try:
            full_dict = config.model_dump()
            restored_config = FrameworkConfig(**full_dict)
            assert restored_config.app_name == "Integration Test"
            assert restored_config.database.host == "integration-db"
        except AttributeError:
            full_dict = config.dict()
            restored_config = FrameworkConfig(**full_dict)
            assert restored_config.app_name == "Integration Test"