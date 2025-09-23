"""
Environment Configuration System
===============================

Système de configuration avancé pour multi-environnement avec:
- Support fichiers YAML/JSON par environnement
- Variables d'environnement sécurisées
- Feature flags dynamiques
- Configuration secrets
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class EnvironmentType(str, Enum):
    """Types d'environnements supportés"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    BACKTESTING = "backtesting"


@dataclass
class FeatureFlag:
    """Configuration d'un feature flag"""
    name: str
    enabled: bool
    description: str = ""
    environments: List[str] = field(default_factory=list)
    rollout_percentage: float = 100.0
    dependencies: List[str] = field(default_factory=list)

    def is_enabled_for_env(self, env: str) -> bool:
        """Vérifie si le flag est activé pour un environnement"""
        if not self.enabled:
            return False

        if self.environments and env not in self.environments:
            return False

        return True


@dataclass
class SecretConfig:
    """Configuration pour un secret"""
    key: str
    env_var: str
    required: bool = True
    default_value: Optional[str] = None
    description: str = ""

    def get_value(self) -> Optional[str]:
        """Récupère la valeur du secret depuis l'environnement"""
        value = os.getenv(self.env_var, self.default_value)

        if self.required and not value:
            raise ValueError(f"Required secret '{self.key}' not found in environment variable '{self.env_var}'")

        return value


class ConfigurationManager:
    """
    Gestionnaire de configuration multi-environnement.

    Gère le chargement et la fusion de configurations depuis:
    - Fichiers de configuration par environnement
    - Variables d'environnement
    - Secrets sécurisés
    - Feature flags
    """

    def __init__(self,
                 config_dir: Union[str, Path] = None,
                 environment: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.environment = environment or os.getenv("QFRAME_ENV", "development")
        self.env_type = EnvironmentType(self.environment)

        # Dictionnaires de configuration
        self._base_config: Dict[str, Any] = {}
        self._env_config: Dict[str, Any] = {}
        self._secrets: Dict[str, SecretConfig] = {}
        self._feature_flags: Dict[str, FeatureFlag] = {}
        self._merged_config: Optional[Dict[str, Any]] = None

        logger.info(f"🔧 ConfigurationManager initialisé pour environnement: {self.environment}")

    def load_configuration(self) -> Dict[str, Any]:
        """
        Charge et fusionne toute la configuration.

        Returns:
            Configuration complète fusionnée
        """
        logger.info("📖 Chargement de la configuration...")

        # 1. Charger la configuration de base
        self._load_base_config()

        # 2. Charger la configuration spécifique à l'environnement
        self._load_environment_config()

        # 3. Charger les secrets
        self._load_secrets()

        # 4. Charger les feature flags
        self._load_feature_flags()

        # 5. Fusionner toutes les configurations
        self._merged_config = self._merge_configurations()

        # 6. Appliquer les variables d'environnement
        self._apply_environment_variables()

        # 7. Validation finale
        self._validate_configuration()

        logger.info(f"✅ Configuration chargée pour {self.environment}")
        return self._merged_config.copy()

    def _load_base_config(self) -> None:
        """Charge la configuration de base"""
        base_files = [
            self.config_dir / "base.yml",
            self.config_dir / "base.yaml",
            self.config_dir / "base.json"
        ]

        for config_file in base_files:
            if config_file.exists():
                self._base_config = self._load_config_file(config_file)
                logger.debug(f"📄 Configuration de base chargée: {config_file}")
                break
        else:
            logger.warning("⚠️ Aucun fichier de configuration de base trouvé")
            self._base_config = self._get_default_base_config()

    def _load_environment_config(self) -> None:
        """Charge la configuration spécifique à l'environnement"""
        env_files = [
            self.config_dir / f"{self.environment}.yml",
            self.config_dir / f"{self.environment}.yaml",
            self.config_dir / f"{self.environment}.json"
        ]

        for config_file in env_files:
            if config_file.exists():
                self._env_config = self._load_config_file(config_file)
                logger.debug(f"📄 Configuration environnement chargée: {config_file}")
                break
        else:
            logger.warning(f"⚠️ Aucun fichier de configuration pour {self.environment}")
            self._env_config = {}

    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Charge un fichier de configuration"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:  # YAML
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"❌ Erreur chargement {file_path}: {e}")
            return {}

    def _load_secrets(self) -> None:
        """Charge la configuration des secrets"""
        secrets_file = self.config_dir / "secrets.yml"

        if secrets_file.exists():
            secrets_config = self._load_config_file(secrets_file)

            for secret_name, secret_data in secrets_config.get("secrets", {}).items():
                self._secrets[secret_name] = SecretConfig(
                    key=secret_name,
                    env_var=secret_data["env_var"],
                    required=secret_data.get("required", True),
                    default_value=secret_data.get("default"),
                    description=secret_data.get("description", "")
                )

        # Ajouter des secrets par défaut
        self._add_default_secrets()

        logger.debug(f"🔐 {len(self._secrets)} secrets configurés")

    def _load_feature_flags(self) -> None:
        """Charge les feature flags"""
        flags_file = self.config_dir / "feature_flags.yml"

        if flags_file.exists():
            flags_config = self._load_config_file(flags_file)

            for flag_name, flag_data in flags_config.get("flags", {}).items():
                self._feature_flags[flag_name] = FeatureFlag(
                    name=flag_name,
                    enabled=flag_data.get("enabled", False),
                    description=flag_data.get("description", ""),
                    environments=flag_data.get("environments", []),
                    rollout_percentage=flag_data.get("rollout_percentage", 100.0),
                    dependencies=flag_data.get("dependencies", [])
                )

        # Ajouter des flags par défaut
        self._add_default_feature_flags()

        logger.debug(f"🚩 {len(self._feature_flags)} feature flags configurés")

    def _merge_configurations(self) -> Dict[str, Any]:
        """Fusionne toutes les configurations"""
        merged = self._base_config.copy()

        # Fusionner la config environnement (priorité plus élevée)
        merged = self._deep_merge(merged, self._env_config)

        # Ajouter les secrets résolus
        secrets_dict = {}
        for secret_name, secret_config in self._secrets.items():
            try:
                value = secret_config.get_value()
                if value:
                    secrets_dict[secret_name] = value
            except ValueError as e:
                if self.env_type == EnvironmentType.PRODUCTION:
                    raise  # En production, les secrets requis doivent être présents
                else:
                    logger.warning(f"⚠️ Secret manquant en {self.environment}: {e}")

        if secrets_dict:
            merged["secrets"] = secrets_dict

        # Ajouter les feature flags actifs
        active_flags = {}
        for flag_name, flag in self._feature_flags.items():
            if flag.is_enabled_for_env(self.environment):
                active_flags[flag_name] = True

        if active_flags:
            merged["feature_flags"] = active_flags

        return merged

    def _apply_environment_variables(self) -> None:
        """Applique les overrides des variables d'environnement"""
        env_overrides = {
            "QFRAME_LOG_LEVEL": ("logging", "level"),
            "QFRAME_DB_HOST": ("database", "host"),
            "QFRAME_DB_PORT": ("database", "port"),
            "QFRAME_REDIS_HOST": ("redis", "host"),
            "QFRAME_REDIS_PORT": ("redis", "port"),
            "QFRAME_API_PORT": ("api", "port"),
            "QFRAME_DEBUG": ("debug",),
        }

        for env_var, config_path in env_overrides.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(self._merged_config, config_path, self._convert_env_value(value))
                logger.debug(f"🔄 Override depuis {env_var}: {config_path} = {value}")

    def _validate_configuration(self) -> None:
        """Valide la configuration finale"""
        required_sections = ["database", "logging"]

        for section in required_sections:
            if section not in self._merged_config:
                logger.warning(f"⚠️ Section manquante: {section}")

        # Validation spécifique à la production
        if self.env_type == EnvironmentType.PRODUCTION:
            self._validate_production_config()

    def _validate_production_config(self) -> None:
        """Validations spécifiques à la production"""
        # Vérifier que les secrets critiques sont présents
        critical_secrets = ["database_password", "api_secret_key", "jwt_secret"]

        secrets = self._merged_config.get("secrets", {})
        for secret in critical_secrets:
            if secret not in secrets:
                raise ValueError(f"Secret critique manquant en production: {secret}")

        # Vérifier que le debug est désactivé
        if self._merged_config.get("debug", False):
            logger.warning("⚠️ Debug activé en production!")

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Fusion profonde de deux dictionnaires"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _set_nested_value(self, config: Dict[str, Any], path: tuple, value: Any) -> None:
        """Définit une valeur dans un dictionnaire imbriqué"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convertit une valeur d'environnement dans le bon type"""
        # Boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # String
        return value

    def _add_default_secrets(self) -> None:
        """Ajoute les secrets par défaut"""
        default_secrets = {
            "database_password": SecretConfig(
                key="database_password",
                env_var="QFRAME_DB_PASSWORD",
                required=False,
                default_value="",
                description="Database password"
            ),
            "api_secret_key": SecretConfig(
                key="api_secret_key",
                env_var="QFRAME_API_SECRET",
                required=self.env_type == EnvironmentType.PRODUCTION,
                default_value="dev-secret-key" if self.env_type == EnvironmentType.DEVELOPMENT else None,
                description="API secret key"
            ),
            "jwt_secret": SecretConfig(
                key="jwt_secret",
                env_var="QFRAME_JWT_SECRET",
                required=self.env_type == EnvironmentType.PRODUCTION,
                default_value="dev-jwt-secret" if self.env_type == EnvironmentType.DEVELOPMENT else None,
                description="JWT signing secret"
            )
        }

        for name, secret in default_secrets.items():
            if name not in self._secrets:
                self._secrets[name] = secret

    def _add_default_feature_flags(self) -> None:
        """Ajoute les feature flags par défaut"""
        default_flags = {
            "advanced_analytics": FeatureFlag(
                name="advanced_analytics",
                enabled=True,
                description="Activer les analytics avancées",
                environments=["development", "staging", "production"]
            ),
            "real_time_monitoring": FeatureFlag(
                name="real_time_monitoring",
                enabled=self.env_type in [EnvironmentType.STAGING, EnvironmentType.PRODUCTION],
                description="Monitoring temps réel"
            ),
            "experimental_strategies": FeatureFlag(
                name="experimental_strategies",
                enabled=self.env_type == EnvironmentType.DEVELOPMENT,
                description="Stratégies expérimentales"
            ),
            "performance_optimization": FeatureFlag(
                name="performance_optimization",
                enabled=True,
                description="Optimisations de performance"
            )
        }

        for name, flag in default_flags.items():
            if name not in self._feature_flags:
                self._feature_flags[name] = flag

    def _get_default_base_config(self) -> Dict[str, Any]:
        """Configuration de base par défaut"""
        return {
            "app": {
                "name": "QFrame",
                "version": "2.0.0"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "qframe",
                "pool_size": 10
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }

    # Méthodes publiques pour accéder à la configuration

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration complète"""
        if self._merged_config is None:
            self.load_configuration()
        return self._merged_config.copy()

    def get_secret(self, secret_name: str) -> Optional[str]:
        """Récupère un secret spécifique"""
        if secret_name in self._secrets:
            return self._secrets[secret_name].get_value()
        return None

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Vérifie si un feature flag est activé"""
        if feature_name in self._feature_flags:
            return self._feature_flags[feature_name].is_enabled_for_env(self.environment)
        return False

    def get_environment_type(self) -> EnvironmentType:
        """Retourne le type d'environnement"""
        return self.env_type

    def reload_configuration(self) -> Dict[str, Any]:
        """Recharge la configuration"""
        self._merged_config = None
        return self.load_configuration()


# Instance globale du gestionnaire de configuration
_config_manager: Optional[ConfigurationManager] = None


def get_configuration_manager(
    config_dir: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None
) -> ConfigurationManager:
    """
    Retourne l'instance globale du gestionnaire de configuration.

    Args:
        config_dir: Répertoire des fichiers de configuration
        environment: Environnement à utiliser

    Returns:
        Instance du gestionnaire de configuration
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir, environment)
        _config_manager.load_configuration()

    return _config_manager


def reload_configuration() -> Dict[str, Any]:
    """Recharge la configuration globale"""
    global _config_manager

    if _config_manager:
        return _config_manager.reload_configuration()
    else:
        manager = get_configuration_manager()
        return manager.get_config()


def get_current_config() -> Dict[str, Any]:
    """Retourne la configuration actuelle"""
    manager = get_configuration_manager()
    return manager.get_config()


def is_feature_enabled(feature_name: str) -> bool:
    """Vérifie si un feature flag est activé"""
    manager = get_configuration_manager()
    return manager.is_feature_enabled(feature_name)