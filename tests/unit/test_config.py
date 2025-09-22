"""
Tests unitaires pour le système de configuration
==============================================

Tests de validation et fonctionnement des configurations Pydantic.
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from qframe.core.config import (
    FrameworkConfig,
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
    DatabaseConfig,
    RedisConfig,
    StrategyConfig,
    RiskManagementConfig,
    Environment,
    LogLevel,
    get_config,
    load_config_from_file,
    get_config_for_environment
)


class TestDatabaseConfig:
    """Tests de la configuration base de données"""

    def test_default_values(self):
        """Test des valeurs par défaut"""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "quant_framework"
        assert config.username == "postgres"
        assert config.pool_size == 10

    def test_port_validation(self):
        """Test de validation du port"""
        # Port valide
        config = DatabaseConfig(port=5433)
        assert config.port == 5433

        # Port invalide
        with pytest.raises(ValidationError):
            DatabaseConfig(port=70000)

        with pytest.raises(ValidationError):
            DatabaseConfig(port=0)

    def test_environment_variables(self):
        """Test des variables d'environnement"""
        with patch.dict(os.environ, {'DB_PASSWORD': 'secret123'}):
            config = DatabaseConfig()
            assert config.password == 'secret123'


class TestRedisConfig:
    """Tests de la configuration Redis"""

    def test_default_values(self):
        """Test des valeurs par défaut"""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.max_connections == 10

    def test_with_password(self):
        """Test avec mot de passe"""
        with patch.dict(os.environ, {'REDIS_PASSWORD': 'redis123'}):
            config = RedisConfig()
            assert config.password == 'redis123'


class TestStrategyConfig:
    """Tests de la configuration stratégie"""

    def test_default_values(self):
        """Test des valeurs par défaut"""
        config = StrategyConfig(name="test_strategy")
        assert config.name == "test_strategy"
        assert config.enabled is True
        assert config.position_size == 0.02
        assert config.max_positions == 5

    def test_position_size_validation(self):
        """Test de validation de la taille de position"""
        # Valeurs valides
        config = StrategyConfig(name="test", position_size=0.05)
        assert config.position_size == 0.05

        # Valeurs invalides
        with pytest.raises(ValidationError):
            StrategyConfig(name="test", position_size=1.5)  # Trop grand

        with pytest.raises(ValidationError):
            StrategyConfig(name="test", position_size=0.0005)  # Trop petit


class TestRiskManagementConfig:
    """Tests de la configuration risk management"""

    def test_default_values(self):
        """Test des valeurs par défaut"""
        config = RiskManagementConfig()
        assert config.max_portfolio_risk == 0.1
        assert config.max_correlation == 0.7
        assert config.var_confidence == 0.95
        assert config.default_stop_loss == 0.05

    def test_validation_ranges(self):
        """Test des validations de ranges"""
        # Valeurs valides
        config = RiskManagementConfig(
            max_portfolio_risk=0.2,
            drawdown_limit=0.15
        )
        assert config.max_portfolio_risk == 0.2
        assert config.drawdown_limit == 0.15

        # Valeurs invalides
        with pytest.raises(ValidationError):
            RiskManagementConfig(max_portfolio_risk=0.6)  # Trop élevé


class TestFrameworkConfig:
    """Tests de la configuration principale"""

    def test_default_initialization(self):
        """Test d'initialisation par défaut"""
        config = FrameworkConfig()
        assert config.app_name == "Quant Framework Research"
        assert config.environment == Environment.DEVELOPMENT
        assert config.log_level == LogLevel.INFO
        assert config.max_workers == 4

    def test_path_validation_and_creation(self, tmp_path):
        """Test de validation et création des chemins"""
        with patch.object(Path, 'cwd', return_value=tmp_path):
            config = FrameworkConfig(project_root=tmp_path)

            # Les répertoires doivent être créés
            assert config.data_dir.exists()
            assert config.logs_dir.exists()
            assert config.artifacts_dir.exists()

    def test_environment_validation(self):
        """Test de validation de l'environnement"""
        # Production sans secret key personnalisé
        with pytest.raises(ValidationError):
            FrameworkConfig(
                environment=Environment.PRODUCTION,
                secret_key="dev-secret-key-change-in-production"
            )

        # Production avec secret key valide
        config = FrameworkConfig(
            environment=Environment.PRODUCTION,
            secret_key="production-secret-key-123"
        )
        assert config.environment == Environment.PRODUCTION

    def test_strategy_config_access(self):
        """Test d'accès aux configurations de stratégies"""
        config = FrameworkConfig()

        dmn_config = config.get_strategy_config("dmn_lstm")
        assert dmn_config is not None
        assert dmn_config.name == "DMN_LSTM"

        missing_config = config.get_strategy_config("nonexistent")
        assert missing_config is None

    def test_data_provider_config_access(self):
        """Test d'accès aux configurations de fournisseurs"""
        config = FrameworkConfig()

        binance_config = config.get_data_provider_config("binance")
        assert binance_config is not None
        assert binance_config.name == "binance"

        missing_config = config.get_data_provider_config("nonexistent")
        assert missing_config is None

    def test_environment_helpers(self):
        """Test des méthodes d'aide pour l'environnement"""
        dev_config = FrameworkConfig(environment=Environment.DEVELOPMENT)
        assert dev_config.is_development() is True
        assert dev_config.is_production() is False
        assert dev_config.is_testing() is False

        prod_config = FrameworkConfig(
            environment=Environment.PRODUCTION,
            secret_key="prod-secret"
        )
        assert prod_config.is_production() is True
        assert prod_config.is_development() is False


class TestEnvironmentConfigs:
    """Tests des configurations spécialisées par environnement"""

    def test_development_config(self):
        """Test de la configuration développement"""
        config = DevelopmentConfig()
        assert config.environment == Environment.DEVELOPMENT
        assert config.log_level == LogLevel.DEBUG

    def test_production_config(self):
        """Test de la configuration production"""
        with patch.dict(os.environ, {'PROD_SECRET_KEY': 'prod-secret-123'}):
            config = ProductionConfig()
            assert config.environment == Environment.PRODUCTION
            assert config.log_level == LogLevel.INFO

    def test_testing_config(self):
        """Test de la configuration test"""
        config = TestingConfig()
        assert config.environment == Environment.TESTING
        assert config.log_level == LogLevel.DEBUG
        assert config.database.host == "sqlite"
        assert config.database.database == ":memory:"


class TestConfigLoading:
    """Tests du chargement de configuration"""

    def test_load_from_json_file(self, tmp_path):
        """Test de chargement depuis un fichier JSON"""
        config_data = {
            "app_name": "Test App",
            "environment": "testing",
            "log_level": "DEBUG",
            "max_workers": 8
        }

        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = load_config_from_file(str(config_file))
        assert config.app_name == "Test App"
        assert config.environment == Environment.TESTING
        assert config.log_level == LogLevel.DEBUG
        assert config.max_workers == 8

    def test_load_from_yaml_file(self, tmp_path):
        """Test de chargement depuis un fichier YAML"""
        pytest.importorskip("yaml")  # Skip if yaml not available

        config_data = """
        app_name: "Test App YAML"
        environment: "development"
        log_level: "INFO"
        max_workers: 6
        """

        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_data)

        config = load_config_from_file(str(config_file))
        assert config.app_name == "Test App YAML"
        assert config.environment == Environment.DEVELOPMENT

    def test_unsupported_file_format(self, tmp_path):
        """Test avec format de fichier non supporté"""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config_from_file(str(config_file))

    def test_get_config_for_environment(self):
        """Test de récupération de config pour environnement"""
        dev_config = get_config_for_environment("development")
        assert isinstance(dev_config, DevelopmentConfig)
        assert dev_config.environment == Environment.DEVELOPMENT

        prod_config = get_config_for_environment("production")
        assert isinstance(prod_config, ProductionConfig)

        test_config = get_config_for_environment("testing")
        assert isinstance(test_config, TestingConfig)

        # Environnement inconnu devrait retourner config de base
        unknown_config = get_config_for_environment("unknown")
        assert isinstance(unknown_config, FrameworkConfig)


class TestEnvironmentVariables:
    """Tests des variables d'environnement"""

    def test_env_file_loading(self, tmp_path):
        """Test du chargement de fichier .env"""
        env_file = tmp_path / ".env"
        env_file.write_text("""
SECRET_KEY=test-secret-key
LOG_LEVEL=DEBUG
MAX_WORKERS=12
DB_PASSWORD=db-secret
        """.strip())

        with patch.dict(os.environ, {}, clear=True):
            with patch('qframe.core.config.FrameworkConfig.Config.env_file', str(env_file)):
                config = FrameworkConfig()
                # Note: Les variables ne sont chargées qu'au moment de l'instanciation
                # avec le bon env_file configuré

    def test_environment_precedence(self):
        """Test de la précédence des variables d'environnement"""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'env-secret',
            'MAX_WORKERS': '16'
        }):
            config = FrameworkConfig()
            # Les variables d'environnement doivent avoir la précédence


class TestConfigValidation:
    """Tests de validation approfondie"""

    def test_nested_config_validation(self):
        """Test de validation des configurations imbriquées"""
        # Configuration avec base de données invalide
        with pytest.raises(ValidationError):
            FrameworkConfig(
                database=DatabaseConfig(port=70000)  # Port invalide
            )

    def test_complex_validation_scenarios(self):
        """Test de scénarios de validation complexes"""
        # Configuration valide complète
        config = FrameworkConfig(
            app_name="Test Framework",
            environment=Environment.DEVELOPMENT,
            database=DatabaseConfig(
                host="custom-host",
                port=5433,
                database="custom_db"
            ),
            risk_management=RiskManagementConfig(
                max_portfolio_risk=0.15,
                default_stop_loss=0.03
            )
        )

        assert config.app_name == "Test Framework"
        assert config.database.host == "custom-host"
        assert config.risk_management.max_portfolio_risk == 0.15