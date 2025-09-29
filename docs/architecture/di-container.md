# 🔌 Dependency Injection Container

Le container DI de QFrame offre une injection de dépendances moderne et thread-safe.

## Fonctionnalités

- **Thread-safe** : RLock pour accès concurrent
- **Lifecycles** : Singleton, Transient, Scoped
- **Auto-injection** : Via annotations de type
- **Détection circulaire** : Prévention des dépendances circulaires

## Usage

```python
from qframe.core.container import get_container, injectable

@injectable
class MyStrategy(BaseStrategy):
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider

# Container automatique
container = get_container()
strategy = container.resolve(MyStrategy)  # Injection automatique
```

## Enregistrement Manuel

```python
container.register_singleton(DataProvider, BinanceProvider)
container.register_transient(Strategy, MyStrategy)
```

## Voir aussi

- [Configuration](configuration.md) - Configuration Pydantic
- [Interfaces](interfaces.md) - Protocols et contrats