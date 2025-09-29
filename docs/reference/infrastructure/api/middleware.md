# qframe.infrastructure.api.middleware


Infrastructure Layer: API Middleware
====================================

Middleware pour l'API incluant CORS, rate limiting, authentification,
logging et gestion des erreurs.


::: qframe.infrastructure.api.middleware
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

- `RateLimitMiddleware`
- `AuthMiddleware`
- `LoggingMiddleware`
- `ErrorHandlingMiddleware`
- `CORSMiddleware`
- `SecurityHeadersMiddleware`

### Fonctions

- `setup_middleware`

