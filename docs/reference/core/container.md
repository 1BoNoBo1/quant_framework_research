# qframe.core.container


Dependency Injection Container
=============================

Container IoC pour gestion des dépendances et inversion de contrôle.
Permet une architecture testable et découplée.


::: qframe.core.container
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      members_order: alphabetical
      filters:
        - "!^_"
        - "!^__"
      group_by_category: true
      show_category_heading: true

## Composants

### Classes

- `Injectable`
- `LifetimeScope`
- `ServiceDescriptor`
- `DIContainer`
- `ScopeManager`
- `MockContainer`

### Fonctions

- `injectable`
- `singleton`
- `transient`
- `scoped`
- `auto_register`
- `factory`
- `configure_services`
- `get_container`
- `set_container`

## Exemples d'usage


```python
from qframe.core.container import get_container, injectable

# Container automatique
container = get_container()

# Enregistrement manuel
container.register_singleton(Interface, Implementation)

# Résolution avec injection
instance = container.resolve(MyClass)
```
