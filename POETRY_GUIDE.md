# 📚 Guide Complet Poetry pour QFrame

## 🎯 Qu'est-ce que Poetry ?

Poetry est un **gestionnaire de dépendances moderne** pour Python qui remplace pip, virtualenv, setup.py, requirements.txt et plus encore. C'est comme npm/yarn pour JavaScript, mais pour Python.

## 🔄 Poetry vs Méthode Classique

### Méthode Classique (pip + venv)
```bash
# Créer environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer dépendances
pip install package1 package2
pip freeze > requirements.txt

# Problèmes:
# - requirements.txt ne distingue pas dépendances directes/indirectes
# - Pas de gestion des versions
# - Conflits de dépendances difficiles à résoudre
```

### Méthode Poetry
```bash
# Tout est géré automatiquement
poetry add package1 package2

# Avantages:
# ✅ Résolution automatique des conflits
# ✅ Lock file pour reproductibilité
# ✅ Distinction dev/prod/optional
# ✅ Un seul fichier de configuration
```

## 📁 Structure des Fichiers Poetry

### 1. **pyproject.toml** (Configuration)
```toml
[tool.poetry]
name = "qframe"
version = "0.1.0"
description = "Framework quantitatif professionnel"

[tool.poetry.dependencies]
python = "^3.11"        # Version minimum Python
numpy = "^1.24.0"       # ^ = Compatible avec 1.x.x
pandas = "~2.1.0"       # ~ = Compatible avec 2.1.x uniquement
torch = ">=2.0,<3.0"    # Version entre 2.0 et 3.0

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"       # Dépendances de développement
black = "^23.0.0"       # Non installées en production
```

### 2. **poetry.lock** (Verrouillage)
- Généré automatiquement
- Contient les versions EXACTES de TOUTES les dépendances
- Garantit que tout le monde a les mêmes versions
- **NE PAS MODIFIER MANUELLEMENT**

## 🚀 Commandes Essentielles

### Installation et Configuration

```bash
# Installer Poetry (une fois)
curl -sSL https://install.python-poetry.org | python3 -

# Ajouter au PATH (dans ~/.bashrc ou ~/.zshrc)
export PATH="/home/$USER/.local/bin:$PATH"

# Vérifier l'installation
poetry --version
```

### Gestion du Projet

```bash
# Créer nouveau projet
poetry new mon-projet

# Initialiser projet existant
poetry init

# Installer les dépendances (depuis pyproject.toml)
poetry install

# Mettre à jour le lock file
poetry lock
```

### Gestion des Dépendances

```bash
# Ajouter une dépendance
poetry add pandas               # Production
poetry add --group dev pytest   # Développement uniquement
poetry add --optional plotly    # Optionnelle

# Supprimer une dépendance
poetry remove pandas

# Mettre à jour les dépendances
poetry update              # Toutes
poetry update pandas      # Une seule

# Voir les dépendances installées
poetry show              # Liste complète
poetry show --tree       # Arbre de dépendances
poetry show pandas       # Détails d'un package
```

### Environnement Virtuel

```bash
# Informations sur l'environnement
poetry env info
poetry env info --path     # Chemin de l'environnement

# Poetry 2.0+ : Pas de 'poetry shell' par défaut !
# Option 1: Utiliser poetry run
poetry run python
poetry run pytest
poetry run qframe --help

# Option 2: Activer manuellement
source $(poetry env info --path)/bin/activate

# Option 3: Installer le plugin shell
poetry self add poetry-plugin-shell
poetry shell  # Maintenant disponible

# Supprimer l'environnement
poetry env remove python
```

### Exécution et Scripts

```bash
# Exécuter une commande dans l'environnement
poetry run python script.py
poetry run pytest tests/
poetry run black .

# Scripts personnalisés (définis dans pyproject.toml)
[tool.poetry.scripts]
qframe = "qframe.apps.cli:main"

# Puis utiliser:
poetry run qframe --help
```

## 🔧 Configuration Avancée

### Groupes de Dépendances

```toml
# Production (par défaut)
[tool.poetry.dependencies]
pandas = "^2.0.0"

# Développement
[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"

# Tests (groupe personnalisé)
[tool.poetry.group.test.dependencies]
pytest-cov = "^4.0.0"

# Documentation
[tool.poetry.group.docs.dependencies]
sphinx = "^7.0.0"
```

Installation sélective :
```bash
poetry install                    # Tout
poetry install --without dev      # Sans dev
poetry install --only main,docs   # Seulement main et docs
```

### Sources de Packages

```toml
[[tool.poetry.source]]
name = "private"
url = "https://my-private-repo.com/simple/"
priority = "supplemental"
```

## 📍 Où est mon environnement virtuel ?

Poetry stocke les environnements dans :
- **Linux/Mac** : `~/.cache/pypoetry/virtualenvs/`
- **Windows** : `%LOCALAPPDATA%\pypoetry\Cache\virtualenvs`

Nom du dossier : `{project-name}-{hash}-py{version}`
Exemple : `qframe-mfsRtyXw-py3.13`

## 🎯 Workflow Typique avec Poetry

```bash
# 1. Cloner un projet
git clone https://github.com/user/project.git
cd project

# 2. Installer Poetry si nécessaire
curl -sSL https://install.python-poetry.org | python3 -

# 3. Installer les dépendances
poetry install

# 4. Travailler
poetry run python       # Shell Python
poetry run pytest       # Tests
poetry add new-package  # Ajouter package

# 5. Commit (lock file inclus !)
git add pyproject.toml poetry.lock
git commit -m "Add new-package"
```

## ⚠️ Problèmes Courants et Solutions

### 1. "command not found: poetry"
```bash
# Ajouter au PATH
export PATH="/home/$USER/.local/bin:$PATH"
# Ajouter cette ligne à ~/.bashrc pour permanent
```

### 2. "poetry: command not found" après installation
```bash
# Recharger le shell
source ~/.bashrc
# ou
exec bash
```

### 3. Problème de keyring/DBus
```bash
# Désactiver le keyring
export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
```

### 4. "poetry shell" ne fonctionne pas
```bash
# Poetry 2.0+ : installer le plugin
poetry self add poetry-plugin-shell
# Ou utiliser poetry run
```

### 5. Conflits de versions
```bash
# Voir les conflits
poetry show --why package-name

# Forcer la mise à jour
poetry lock --no-update
poetry install
```

## 💡 Bonnes Pratiques

1. **Toujours commiter `poetry.lock`** : Garantit la reproductibilité
2. **Utiliser `poetry run`** : Plus simple que l'activation manuelle
3. **Groupes de dépendances** : Séparer dev/test/docs/prod
4. **Versions sémantiques** : Utiliser `^` pour flexibilité, `~` pour stabilité
5. **Mise à jour régulière** : `poetry update` hebdomadaire pour sécurité

## 🔍 Commandes de Diagnostic

```bash
# Version de Poetry
poetry --version

# Configuration Poetry
poetry config --list

# Environnement actuel
poetry env info

# Dépendances obsolètes
poetry show --outdated

# Vérifier le projet
poetry check

# Nettoyer le cache
poetry cache clear pypi --all
```

## 📊 Comparaison Rapide

| Tâche | pip/venv | Poetry |
|-------|----------|---------|
| Créer environnement | `python -m venv venv` | Automatique |
| Activer environnement | `source venv/bin/activate` | `poetry run` ou auto |
| Installer package | `pip install package` | `poetry add package` |
| Installer depuis fichier | `pip install -r requirements.txt` | `poetry install` |
| Geler dépendances | `pip freeze > requirements.txt` | Automatique (lock) |
| Mettre à jour | `pip install --upgrade` | `poetry update` |
| Publier package | `python setup.py upload` | `poetry publish` |

## 🎓 Pour QFrame Spécifiquement

```bash
# Installation complète
poetry install

# Lancer les tests
poetry run pytest tests/

# Utiliser le CLI
poetry run qframe version
poetry run qframe info
poetry run qframe strategies

# Développement
poetry run python  # Shell interactif
>>> import qframe
>>> qframe.__version__
'0.1.0'

# Ajouter une nouvelle dépendance
poetry add ta-lib   # Si bibliothèque C installée
poetry add --group dev mypy  # Pour développement
```

---

**Poetry simplifie TOUT** : Plus besoin de jongler avec pip, venv, requirements.txt. Un seul outil, une seule commande, tout fonctionne ! 🚀