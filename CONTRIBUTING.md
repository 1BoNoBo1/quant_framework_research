# Contributing to QFrame

Merci de votre intérêt pour contribuer au framework QFrame ! 🚀

## 🌟 Comment contribuer

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/quant_framework_research.git
cd quant_framework_research
```

### 2. Installation développement
```bash
# Installer Poetry si pas déjà fait
curl -sSL https://install.python-poetry.org | python3 -

# Installer les dépendances
poetry install

# Installer TA-Lib (voir INSTALL_TALIB.md)
# Activer l'environnement
poetry shell
```

### 3. Développement

#### Standards de code
- **Python 3.11+** obligatoire
- **Type hints** pour toutes les fonctions publiques
- **Docstrings** Google style pour documentation
- **Tests** pour toute nouvelle fonctionnalité

#### Workflow de développement
1. Créer une branche feature: `git checkout -b feature/ma-nouvelle-feature`
2. Développer avec tests
3. Vérifier qualité code:
   ```bash
   # Tests
   poetry run pytest tests/ -v

   # Formatage
   poetry run black qframe/
   poetry run ruff check qframe/

   # Type checking
   poetry run mypy qframe/
   ```
4. Commit avec messages clairs
5. Push et créer Pull Request

### 4. Types de contributions

#### 🧠 Stratégies de trading
- Nouvelles stratégies dans `qframe/strategies/research/`
- Améliorations des stratégies existantes
- Backtesting et validation

#### 🔧 Infrastructure
- Améliorations du DI Container
- Nouveaux data providers
- Optimisations performance

#### 📊 Features & Opérateurs
- Nouveaux opérateurs symboliques
- Feature engineering avancé
- Pipeline ML/AI

#### 🧪 Tests & Qualité
- Augmentation couverture tests
- Tests d'intégration
- Benchmarks performance

### 5. Guidelines

#### Commit Messages
```
type(scope): description

feat(strategies): add new mean reversion strategy
fix(container): resolve circular dependency issue
docs(readme): update installation instructions
test(core): add tests for config validation
```

#### Pull Requests
- **Titre clair** décrivant le changement
- **Description détaillée** avec contexte
- **Tests ajoutés** pour nouvelles features
- **Documentation mise à jour** si nécessaire

#### Code Review
- Respecter l'architecture hexagonale
- Suivre les patterns DI existants
- Maintenir compatibilité API
- Performance et sécurité en priorité

### 6. Architecture

#### Principes
- **Dependency Injection** pour découplage
- **Interfaces Protocol** pour contrats
- **Configuration Pydantic** type-safe
- **Tests complets** avec mocks

#### Structure
```
qframe/
├── core/           # Coeur framework (DI, config, interfaces)
├── strategies/     # Stratégies de trading
├── features/       # Feature engineering
├── infra/         # Infrastructure (data, execution, monitoring)
└── apps/          # Applications (CLI, web)
```

### 7. Questions & Support

- **Issues GitHub** pour bugs et features
- **Discussions** pour questions générales
- **Discord/Slack** (liens dans README)

### 8. Code de Conduite

- Respecter tous les contributeurs
- Feedback constructif et bienveillant
- Focus sur amélioration continue
- Maintenir professionnalisme

## 🏆 Reconnaissance

Tous les contributeurs sont reconnus dans :
- README.md (section Contributors)
- CHANGELOG.md pour releases
- Documentation du projet

Merci de faire avancer l'écosystème quantitatif open source ! 💪

---

**Note**: Pour questions spécifiques au trading ou stratégies, consultez d'abord CLAUDE.md qui contient le guide technique complet.