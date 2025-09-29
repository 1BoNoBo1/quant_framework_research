# 🤝 Guide du Contributeur

Contribuez au développement de QFrame en suivant nos standards et procédures.

## Setup Développement

```bash
# Fork et clone
git clone https://github.com/votre-username/quant_framework_research.git
cd quant_framework_research

# Installation développement
poetry install
poetry run pre-commit install
```

## Standards Code

### Qualité
- **Black** : Formatage automatique
- **Ruff** : Linting et optimisations
- **MyPy** : Vérification types
- **Pytest** : Tests avec coverage > 75%

### Architecture
- **DI Container** : Injection de dépendances obligatoire
- **Protocols** : Interfaces Python modernes
- **Pydantic** : Configuration type-safe
- **Docstrings** : Documentation complète

## Workflow

### 1. Créer Branch
```bash
git checkout -b feature/nouvelle-strategie
```

### 2. Développement
```bash
# Tests pendant développement
poetry run pytest tests/unit/ -v

# Qualité code
poetry run black qframe/
poetry run ruff check qframe/
poetry run mypy qframe/
```

### 3. Pull Request
- Tests passent (> 75% coverage)
- Documentation mise à jour
- Changelog mis à jour
- Review par mainteneur

## Types de Contributions

### Stratégies
- Hériter de `BaseStrategy`
- Implémenter `generate_signals()`
- Tests unitaires complets
- Documentation avec exemples

### Features
- Implémenter `FeatureProcessor`
- Opérateurs vectorisés (pandas/numpy)
- Validation entrées/sorties
- Benchmarks performance

## Voir aussi

- [Tests](testing.md) - Stratégie de test
- [Architecture](../architecture/overview.md) - Principes techniques