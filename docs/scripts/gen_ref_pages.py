"""
Génération automatique des pages de référence API depuis le code Python.

Ce script utilise mkdocs-gen-files pour créer automatiquement la documentation
de l'API à partir du code source Python.
"""

from pathlib import Path
import mkdocs_gen_files

# Configuration
nav = mkdocs_gen_files.Nav()
src = Path(__file__).parent.parent.parent  # Répertoire racine du projet
module_dir = src / "qframe"

def should_document_module(module_path: Path) -> bool:
    """Détermine si un module doit être documenté."""
    # Ignorer les fichiers de test et les dossiers cachés
    if any(part.startswith(('.', '_', 'test')) for part in module_path.parts):
        return False

    # Ignorer certains dossiers spécifiques
    ignored_dirs = {'__pycache__', 'migrations', 'alembic'}
    if any(part in ignored_dirs for part in module_path.parts):
        return False

    return True

def get_module_docstring(module_path: Path) -> str:
    """Extrait la docstring d'un module."""
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Recherche simple de docstring en début de fichier
        lines = content.split('\n')
        in_docstring = False
        docstring_lines = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('"""') or line.startswith("'''"):
                if in_docstring:
                    break
                in_docstring = True
                if len(line) > 3:
                    docstring_lines.append(line[3:])
                continue
            if in_docstring:
                docstring_lines.append(line)
            else:
                break

        return ' '.join(docstring_lines).strip()
    except Exception:
        return ""

# Parcourir tous les fichiers Python dans qframe/
for path in sorted(module_dir.rglob("*.py")):
    if not should_document_module(path):
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    # Créer le chemin de navigation
    parts = list(module_path.parts)
    nav_parts = []

    for part in parts:
        if part == '__init__':
            continue
        nav_parts.append(part.replace('_', ' ').title())

    if nav_parts:
        nav[nav_parts] = str(full_doc_path)

    # Générer le contenu de la page de documentation
    module_name = ".".join(module_path.parts)

    # Obtenir la docstring du module
    module_docstring = get_module_docstring(path)
    description = f"\n\n{module_docstring}\n" if module_docstring else ""

    # Créer le contenu markdown
    content = f"""# {module_name}

::: {module_name}{description}

## Configuration

Ce module fait partie du framework QFrame et utilise le système de Dependency Injection.

```python
from qframe.core.container import get_container
from {module_name} import *

# Utilisation avec le container DI
container = get_container()
# ... configuration spécifique
```

## Voir aussi

- [Architecture du Framework](/architecture/overview/)
- [Guide de Configuration](/getting-started/configuration/)
- [Exemples d'utilisation](/examples/)
"""

    # Écrire le fichier de documentation
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(content)

    # Ajouter le fichier à l'édition Git pour MkDocs
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Générer le fichier de navigation
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# Générer la page d'index de référence
index_content = """# 📖 Référence API

Cette section contient la documentation complète de l'API QFrame, générée automatiquement à partir du code source.

## Organisation

La documentation de l'API est organisée selon l'architecture du framework :

### 🏗️ Core Framework
- **[qframe.core](qframe/core/)** - Coeur du framework (DI, configuration, interfaces)
- **[qframe.domain](qframe/domain/)** - Entités métier et logique de domaine
- **[qframe.application](qframe/application/)** - Services applicatifs et use cases

### 🧠 Stratégies & Features
- **[qframe.strategies](qframe/strategies/)** - Stratégies de trading (DMN LSTM, Mean Reversion, etc.)
- **[qframe.features](qframe/features/)** - Feature engineering et opérateurs symboliques

### 🛠️ Infrastructure
- **[qframe.infrastructure](qframe/infrastructure/)** - Data providers, persistance, observabilité
- **[qframe.presentation](qframe/presentation/)** - API REST, CLI, interfaces utilisateur

### 🔬 Research Platform
- **[qframe.research](qframe/research/)** - Plateforme de recherche distribuée
- **[qframe.ecosystem](qframe/ecosystem/)** - Intégrations et extensions

## Navigation

- Utilisez la navigation latérale pour explorer les modules
- Chaque page contient la documentation complète des classes et fonctions
- Les exemples d'usage sont inclus quand disponibles
- Les liens croisés permettent de naviguer entre concepts liés

## Conventions

- **Classes** : Documentées avec leurs méthodes publiques et privées importantes
- **Fonctions** : Signatures complètes avec types et exemples
- **Protocols** : Interfaces définies pour l'injection de dépendances
- **Configurations** : Schémas Pydantic avec validation

!!! tip "Astuce"
    Utilisez la recherche (Ctrl+K) pour trouver rapidement classes, méthodes ou concepts spécifiques.

!!! info "Auto-génération"
    Cette documentation est générée automatiquement lors du build.
    Pour contribuer, modifiez les docstrings dans le code source.
"""

with mkdocs_gen_files.open("reference/index.md", "w") as index_file:
    index_file.write(index_content)