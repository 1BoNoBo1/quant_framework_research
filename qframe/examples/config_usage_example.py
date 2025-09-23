"""
Example: Utilisation du système de configuration multi-environnement
===================================================================

Exemple d'utilisation du nouveau système de configuration avec:
- Chargement par environnement
- Feature flags
- Secrets sécurisés
- Variables d'environnement
"""

import os
import asyncio
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports du framework
from ..infrastructure.config.environment_config import (
    ConfigurationManager,
    get_configuration_manager,
    get_current_config,
    is_feature_enabled,
    EnvironmentType
)


def example_basic_config_loading():
    """
    Exemple de base du chargement de configuration.
    """
    print("🔧 Exemple de chargement de configuration de base")
    print("=" * 50)

    # Définir l'environnement via variable d'environnement
    os.environ["QFRAME_ENV"] = "development"

    # Chemin vers les fichiers de configuration
    config_dir = Path(__file__).parent.parent.parent / "config"
    print(f"📁 Répertoire config: {config_dir}")

    # Créer le gestionnaire de configuration
    config_manager = ConfigurationManager(
        config_dir=config_dir,
        environment="development"
    )

    # Charger la configuration
    config = config_manager.load_configuration()

    print(f"✅ Configuration chargée pour: {config_manager.environment}")
    print(f"📊 Sections disponibles: {list(config.keys())}")

    # Afficher quelques valeurs
    print(f"🎯 App name: {config.get('app', {}).get('name')}")
    print(f"🎯 Debug mode: {config.get('app', {}).get('debug')}")
    print(f"🎯 Database name: {config.get('database', {}).get('name')}")
    print(f"🎯 API port: {config.get('api', {}).get('port')}")

    return config


def example_environment_specific_config():
    """
    Exemple de configuration spécifique par environnement.
    """
    print("\n🌍 Exemple de configuration par environnement")
    print("=" * 50)

    config_dir = Path(__file__).parent.parent.parent / "config"
    environments = ["development", "testing", "production"]

    for env in environments:
        print(f"\n--- Configuration {env.upper()} ---")

        config_manager = ConfigurationManager(
            config_dir=config_dir,
            environment=env
        )

        config = config_manager.load_configuration()

        # Comparer les différences
        print(f"Debug mode: {config.get('app', {}).get('debug', 'N/A')}")
        print(f"Log level: {config.get('logging', {}).get('level', 'N/A')}")
        print(f"Database name: {config.get('database', {}).get('name', 'N/A')}")
        print(f"API workers: {config.get('api', {}).get('workers', 'N/A')}")


def example_feature_flags():
    """
    Exemple d'utilisation des feature flags.
    """
    print("\n🚩 Exemple de feature flags")
    print("=" * 50)

    config_dir = Path(__file__).parent.parent.parent / "config"

    # Tester les flags dans différents environnements
    environments = ["development", "testing", "production"]

    for env in environments:
        print(f"\n--- Feature flags pour {env.upper()} ---")

        config_manager = ConfigurationManager(
            config_dir=config_dir,
            environment=env
        )

        config = config_manager.load_configuration()

        # Tester des flags spécifiques
        test_flags = [
            "experimental_strategies",
            "real_time_monitoring",
            "advanced_analytics",
            "high_frequency_trading",
            "debug_endpoints"
        ]

        for flag in test_flags:
            enabled = config_manager.is_feature_enabled(flag)
            status = "🟢 ON" if enabled else "🔴 OFF"
            print(f"  {flag}: {status}")


def example_secrets_handling():
    """
    Exemple de gestion des secrets.
    """
    print("\n🔐 Exemple de gestion des secrets")
    print("=" * 50)

    config_dir = Path(__file__).parent.parent.parent / "config"

    # Définir quelques variables d'environnement pour test
    test_secrets = {
        "QFRAME_API_SECRET": "test-api-secret-123",
        "QFRAME_JWT_SECRET": "test-jwt-secret-456",
        "QFRAME_DB_PASSWORD": "test-db-password-789"
    }

    # Sauvegarder les valeurs originales
    original_values = {}
    for key in test_secrets:
        original_values[key] = os.environ.get(key)
        os.environ[key] = test_secrets[key]

    try:
        config_manager = ConfigurationManager(
            config_dir=config_dir,
            environment="development"
        )

        config = config_manager.load_configuration()

        print("🔑 Secrets chargés:")
        secrets = config.get("secrets", {})
        for secret_name, value in secrets.items():
            masked_value = value[:4] + "*" * (len(value) - 4) if len(value) > 4 else "***"
            print(f"  {secret_name}: {masked_value}")

        # Test d'accès direct aux secrets
        api_secret = config_manager.get_secret("api_secret_key")
        if api_secret:
            print(f"\n✅ Secret API récupéré: {api_secret[:4]}***")

    finally:
        # Restaurer les valeurs originales
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def example_environment_variables_override():
    """
    Exemple d'override avec variables d'environnement.
    """
    print("\n🔄 Exemple d'override avec variables d'environnement")
    print("=" * 50)

    config_dir = Path(__file__).parent.parent.parent / "config"

    # Définir des overrides via variables d'environnement
    env_overrides = {
        "QFRAME_LOG_LEVEL": "ERROR",
        "QFRAME_DB_PORT": "5433",
        "QFRAME_API_PORT": "8001",
        "QFRAME_DEBUG": "true"
    }

    # Sauvegarder et définir
    original_values = {}
    for key, value in env_overrides.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        config_manager = ConfigurationManager(
            config_dir=config_dir,
            environment="development"
        )

        config = config_manager.load_configuration()

        print("🔄 Valeurs overridées par les variables d'environnement:")
        print(f"  Log level: {config.get('logging', {}).get('level')}")
        print(f"  DB port: {config.get('database', {}).get('port')}")
        print(f"  API port: {config.get('api', {}).get('port')}")
        print(f"  Debug: {config.get('debug')}")

    finally:
        # Restaurer les valeurs originales
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def example_global_configuration_manager():
    """
    Exemple d'utilisation du gestionnaire global.
    """
    print("\n🌐 Exemple du gestionnaire de configuration global")
    print("=" * 50)

    # Définir l'environnement
    os.environ["QFRAME_ENV"] = "development"

    config_dir = Path(__file__).parent.parent.parent / "config"

    # Utiliser le gestionnaire global
    manager = get_configuration_manager(config_dir, "development")
    config = get_current_config()

    print(f"✅ Configuration globale chargée")
    print(f"📊 Environment type: {manager.get_environment_type()}")

    # Tester les feature flags globaux
    print(f"\n🚩 Tests de feature flags:")
    print(f"  Advanced analytics: {is_feature_enabled('advanced_analytics')}")
    print(f"  Experimental strategies: {is_feature_enabled('experimental_strategies')}")
    print(f"  Real-time monitoring: {is_feature_enabled('real_time_monitoring')}")


def example_config_validation():
    """
    Exemple de validation de configuration.
    """
    print("\n✅ Exemple de validation de configuration")
    print("=" * 50)

    config_dir = Path(__file__).parent.parent.parent / "config"

    try:
        # Test configuration development (doit passer)
        print("🧪 Test configuration development...")
        config_manager = ConfigurationManager(
            config_dir=config_dir,
            environment="development"
        )
        config = config_manager.load_configuration()
        print("✅ Configuration development valide")

    except Exception as e:
        print(f"❌ Erreur configuration development: {e}")

    try:
        # Test configuration production sans secrets (doit échouer)
        print("\n🧪 Test configuration production sans secrets...")
        config_manager = ConfigurationManager(
            config_dir=config_dir,
            environment="production"
        )
        config = config_manager.load_configuration()
        print("⚠️ Configuration production chargée (warnings possibles)")

    except Exception as e:
        print(f"❌ Erreur attendue en production sans secrets: {e}")


def example_configuration_differences():
    """
    Exemple montrant les différences entre environnements.
    """
    print("\n📊 Comparaison des configurations par environnement")
    print("=" * 50)

    config_dir = Path(__file__).parent.parent.parent / "config"
    environments = ["development", "testing", "production"]

    configs = {}
    for env in environments:
        try:
            manager = ConfigurationManager(config_dir, env)
            configs[env] = manager.load_configuration()
        except Exception as e:
            print(f"⚠️ Erreur chargement {env}: {e}")
            configs[env] = None

    # Comparer des clés spécifiques
    comparison_keys = [
        ("app", "debug"),
        ("logging", "level"),
        ("database", "echo"),
        ("api", "workers"),
        ("trading", "testnet"),
        ("performance", "max_workers")
    ]

    print("\n📋 Comparaison des valeurs:")
    print(f"{'Paramètre':<25} {'Development':<15} {'Testing':<15} {'Production':<15}")
    print("-" * 70)

    for section, key in comparison_keys:
        values = []
        for env in environments:
            if configs[env]:
                value = configs[env].get(section, {}).get(key, "N/A")
                values.append(str(value))
            else:
                values.append("ERROR")

        param_name = f"{section}.{key}"
        print(f"{param_name:<25} {values[0]:<15} {values[1]:<15} {values[2]:<15}")


async def main():
    """Fonction principale de l'exemple."""
    print("🚀 DÉMONSTRATION DU SYSTÈME DE CONFIGURATION MULTI-ENVIRONNEMENT")
    print("=" * 70)

    try:
        # Exemple 1: Chargement de base
        example_basic_config_loading()

        # Exemple 2: Configuration par environnement
        example_environment_specific_config()

        # Exemple 3: Feature flags
        example_feature_flags()

        # Exemple 4: Gestion des secrets
        example_secrets_handling()

        # Exemple 5: Override avec variables d'environnement
        example_environment_variables_override()

        # Exemple 6: Gestionnaire global
        example_global_configuration_manager()

        # Exemple 7: Validation
        example_config_validation()

        # Exemple 8: Comparaison des configurations
        example_configuration_differences()

        print("\n🎉 TOUS LES EXEMPLES DE CONFIGURATION TERMINÉS AVEC SUCCÈS!")

    except Exception as e:
        print(f"\n❌ ERREUR DANS L'EXEMPLE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())