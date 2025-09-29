# 📦 Installation

## Prérequis

- Python 3.11+
- Poetry (recommandé)

## Installation Rapide

### Avec Poetry (Recommandé)

```bash
# Cloner le repository
git clone https://github.com/1BoNoBo1/quant_framework_research.git
cd quant_framework_research

# Installation des dépendances
poetry install

# Vérification
poetry run python demo_framework.py
```

### Avec Pip

```bash
# Cloner et installer
git clone https://github.com/1BoNoBo1/quant_framework_research.git
cd quant_framework_research

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Installation
pip install -e .
```

## Vérification de l'Installation

```bash
# Framework de base
poetry run python demo_framework.py

# Tests unitaires
poetry run pytest tests/unit/ -v

# Interface web
cd qframe/ui && ./deploy-simple.sh test
```

Le framework est maintenant prêt à l'utilisation !