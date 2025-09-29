# 📈 Coverage Reports

## Test Coverage

La coverage des tests est automatiquement générée et intégrée dans cette documentation.

```bash
# Générer rapport coverage
poetry run pytest --cov=qframe --cov-report=html --cov-report=xml

# Voir rapport
open htmlcov/index.html
```

## Intégration Continue

Les métriques de coverage sont trackées dans le CI/CD :

- **Target** : > 75%
- **Branches** : > 70%
- **Fonctions** : > 80%

{{ coverage_badge() }}

## Coverage par Module

Les détails de coverage par module sont disponibles dans le rapport HTML généré automatiquement.
