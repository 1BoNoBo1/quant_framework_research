# qframe.core.container_enterprise


Enterprise Dependency Injection Container
=========================================

Container DI de niveau enterprise avec type safety complète,
gestion avancée des scopes, et monitoring intégré.


::: qframe.core.container_enterprise
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

- `LifetimeScope`
- `DependencyScope`
- `InjectionMetrics`
- `ServiceDescriptor`
- `Injectable`
- `ILifetimeScope`
- `IServiceContainer`
- `CircularDependencyError`
- `ServiceNotFoundError`
- `SingletonLifetimeManager`
- `ScopeManager`
- `EnterpriseContainer`

### Fonctions

- `get_enterprise_container`
- `configure_enterprise_container`
- `injectable`
- `inject`

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
