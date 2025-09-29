# 🔍 Centre de Diagnostic QFrame

Bienvenue dans le **Centre de Diagnostic Intelligent** de QFrame ! Cette page vous aide à identifier et résoudre rapidement tous les problèmes liés au framework.

## 🚀 Statut Rapide

{{ quick_health_check() }}

## 📋 Diagnostic Complet

{{ diagnostic_report() }}

## 🏗️ Architecture & Dépendances

{{ architecture_overview() }}

## 🔧 Outils de Debug Avancés

### 💻 Playground de Code

Testez vos configurations et stratégies directement dans le navigateur :

{{ interactive_code_playground("""
# Test rapide du framework
from qframe.core.container import get_container
from qframe.core.config import get_config

# Vérification des imports
print("✅ QFrame importé avec succès")

# Test du container DI
container = get_container()
print(f"✅ Container DI initialisé: {type(container)}")

# Test de la configuration
config = get_config()
print(f"✅ Configuration chargée: {config.app_name}")
""") }}

### 🧭 Explorateur d'Architecture

{{ interactive_architecture_explorer() }}

### 🔌 Interfaces & Protocols

{{ interfaces_documentation() }}

## 🎯 Composants Détaillés

### Core Framework

{{ component_detail("core") }}

### Domain Logic

{{ component_detail("domain") }}

### Infrastructure

{{ component_detail("infrastructure") }}

### Stratégies de Trading

{{ component_detail("strategies") }}

## 📈 Graphe de Dépendances

{{ dependency_graph() }}

## 🆘 Aide Contextuelle

{{ ai_contextual_help("diagnostic", "help with troubleshooting") }}

## 📞 Support & Assistance

### 🔍 Questions Fréquentes

??? question "❌ ModuleNotFoundError: No module named 'qframe'"

    **Solution** : Le framework n'est pas installé correctement.

    ```bash
    # Vérifiez votre environnement Python
    poetry install

    # Ou avec pip
    pip install -e .
    ```

??? question "⚠️ ValidationError dans la configuration"

    **Solution** : Vérifiez vos types de configuration.

    ```python
    from qframe.core.config import get_config

    # La configuration se valide automatiquement
    config = get_config()
    ```

??? question "🔧 Container DI ne résout pas les services"

    **Solution** : Vérifiez l'enregistrement des services.

    ```python
    container = get_container()
    container.register_singleton(Interface, Implementation)
    ```

### 🛠️ Outils de Debug

=== "Logs"

    ```python
    import logging
    from qframe.core.config import get_config

    # Configuration des logs
    logging.basicConfig(level=logging.DEBUG)
    config = get_config()
    ```

=== "Profiling"

    ```python
    import cProfile

    # Profile une stratégie
    cProfile.run('strategy.generate_signals(data)')
    ```

=== "Memory"

    ```python
    import tracemalloc

    # Surveillance mémoire
    tracemalloc.start()
    # ... votre code ...
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory: {current / 1024 / 1024:.1f} MB")
    ```

### 📊 Métriques Système

{{ framework_health_check() }}

### 🔗 Liens Utiles

- [📖 Documentation Complète](/getting-started/installation/)
- [🏗️ Guide Architecture](/architecture/overview/)
- [🧠 Guide Stratégies](/strategies/)
- [🔧 Configuration Avancée](/architecture/configuration/)
- [🚀 Guide Démarrage](/getting-started/quickstart/)

---

{{ mkdocs_build_info() }}