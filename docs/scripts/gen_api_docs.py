#!/usr/bin/env python3
"""
Générateur automatique de documentation API complète pour QFrame.
Ce script analyse le code Python et génère une documentation API exhaustive.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import inspect
import importlib.util
from textwrap import dedent

# Ajouter le chemin du projet pour les imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def get_project_root() -> Path:
    """Retourne le chemin racine du projet."""
    return Path(__file__).parent.parent.parent

def should_document_module(module_path: Path) -> bool:
    """Détermine si un module doit être documenté."""
    if any(part.startswith(('.', '_', 'test', '__pycache__')) for part in module_path.parts):
        return False
    if module_path.suffix != '.py':
        return False
    if module_path.name in ['__init__.py'] and module_path.stat().st_size < 100:
        return False
    return True

def get_module_docstring(module_path: Path) -> Optional[str]:
    """Extrait la docstring d'un module."""
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
                return tree.body[0].value.value
    except Exception:
        pass
    return None

def get_classes_and_functions(module_path: Path) -> Dict[str, List[str]]:
    """Extrait les classes et fonctions d'un module."""
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        classes = []
        functions = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                functions.append(node.name)

        return {'classes': classes, 'functions': functions}
    except Exception:
        return {'classes': [], 'functions': []}

def generate_api_reference_page(module_path: Path, relative_path: Path) -> str:
    """Génère une page de référence API pour un module."""
    module_name = str(relative_path).replace('/', '.').replace('.py', '')

    # Nettoyer le nom du module
    if module_name.startswith('qframe.'):
        clean_module_name = module_name
    else:
        clean_module_name = f"qframe.{module_name}"

    docstring = get_module_docstring(module_path)
    components = get_classes_and_functions(module_path)

    content = f"# {clean_module_name}\n\n"

    if docstring:
        content += f"{docstring}\n\n"
    else:
        content += f"Module de référence pour {clean_module_name}\n\n"

    # Ajouter l'auto-documentation mkdocstrings
    content += f"::: {clean_module_name}\n"
    content += "    options:\n"
    content += "      show_root_heading: true\n"
    content += "      show_source: true\n"
    content += "      heading_level: 2\n"
    content += "      members_order: alphabetical\n"
    content += "      filters:\n"
    content += "        - \"!^_\"\n"
    content += "        - \"!^__\"\n"
    content += "      group_by_category: true\n"
    content += "      show_category_heading: true\n\n"

    # Ajouter un résumé des composants
    if components['classes'] or components['functions']:
        content += "## Composants\n\n"

        if components['classes']:
            content += "### Classes\n\n"
            for cls in components['classes']:
                content += f"- `{cls}`\n"
            content += "\n"

        if components['functions']:
            content += "### Fonctions\n\n"
            for func in components['functions']:
                content += f"- `{func}`\n"
            content += "\n"

    # Ajouter des exemples d'usage si c'est un module principal
    if any(keyword in module_name.lower() for keyword in ['strategy', 'container', 'config']):
        content += generate_usage_examples(clean_module_name, components)

    return content

def generate_usage_examples(module_name: str, components: Dict[str, List[str]]) -> str:
    """Génère des exemples d'usage pour les modules principaux."""
    examples = "## Exemples d'usage\n\n"

    if 'strategy' in module_name.lower():
        examples += dedent("""
        ```python
        from qframe.core.container import get_container

        # Utilisation avec DI Container
        container = get_container()
        strategy = container.resolve(YourStrategy)

        # Génération de signaux
        signals = strategy.generate_signals(market_data)
        ```
        """)

    elif 'container' in module_name.lower():
        examples += dedent("""
        ```python
        from qframe.core.container import get_container, injectable

        # Container automatique
        container = get_container()

        # Enregistrement manuel
        container.register_singleton(Interface, Implementation)

        # Résolution avec injection
        instance = container.resolve(MyClass)
        ```
        """)

    elif 'config' in module_name.lower():
        examples += dedent("""
        ```python
        from qframe.core.config import get_config, FrameworkConfig

        # Configuration automatique
        config = get_config()

        # Configuration personnalisée
        config = FrameworkConfig(
            environment=Environment.DEVELOPMENT,
            database=DatabaseConfig(url="...")
        )
        ```
        """)

    return examples

def generate_api_index() -> str:
    """Génère la page d'index de l'API."""
    content = dedent("""
    # 📖 Référence API Complète

    Documentation automatique complète de toutes les classes, fonctions et modules du framework QFrame.

    ## Navigation par Module

    Cette section contient la documentation auto-générée depuis le code source avec tous les détails techniques :

    ### Core Framework
    - **Container** : Système d'injection de dépendances
    - **Config** : Configuration Pydantic type-safe
    - **Interfaces** : Protocols et contrats

    ### Stratégies de Trading
    - **DMN LSTM** : Deep Market Networks avec LSTM
    - **Mean Reversion** : Stratégie de retour à la moyenne
    - **Funding Arbitrage** : Arbitrage de taux de financement
    - **RL Alpha** : Génération d'alphas par RL

    ### Infrastructure
    - **Data Providers** : Sources de données (Binance, CCXT, etc.)
    - **Persistence** : Repositories et stockage
    - **Observability** : Logging et monitoring

    ### Research Platform
    - **Data Lake** : Stockage et catalogue de données
    - **Backtesting** : Moteurs de test distribués
    - **Analytics** : Métriques de performance avancées

    ## Recherche dans l'API

    Utilisez **Ctrl+K** pour rechercher rapidement dans toute la documentation API.

    ## Mise à jour

    Cette documentation est générée automatiquement depuis le code source à chaque build.
    Dernière mise à jour : {{ build_timestamp() }}
    """)

    return content

def main():
    """Fonction principale de génération."""
    project_root = get_project_root()
    qframe_dir = project_root / "qframe"
    docs_dir = project_root / "docs"
    api_dir = docs_dir / "reference"

    # Créer le répertoire de référence API
    api_dir.mkdir(exist_ok=True)

    # Générer la page d'index
    index_content = generate_api_index()
    with open(api_dir / "index.md", "w", encoding="utf-8") as f:
        f.write(index_content)

    print(f"📚 Génération documentation API pour QFrame...")
    generated_files = 0

    # Parcourir tous les modules Python
    for py_file in qframe_dir.rglob("*.py"):
        if should_document_module(py_file):
            # Calculer le chemin relatif
            relative_path = py_file.relative_to(qframe_dir)

            # Créer la structure de répertoires
            api_file_path = api_dir / f"{relative_path.with_suffix('.md')}"
            api_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Générer le contenu
            content = generate_api_reference_page(py_file, relative_path)

            # Écrire le fichier
            with open(api_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            generated_files += 1
            print(f"  ✅ {relative_path} -> {api_file_path.relative_to(docs_dir)}")

    print(f"🎉 Documentation API générée : {generated_files} fichiers")
    print(f"📁 Emplacement : {api_dir.relative_to(project_root)}")

if __name__ == "__main__":
    main()