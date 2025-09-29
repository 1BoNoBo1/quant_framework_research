# qframe.infrastructure.api.auth


Infrastructure Layer: Authentication & Authorization
===================================================

Services d'authentification et d'autorisation pour l'API
avec JWT, RBAC et gestion des sessions.


::: qframe.infrastructure.api.auth
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

- `UserRole`
- `Permission`
- `User`
- `Session`
- `JWTManager`
- `AuthenticationService`
- `AuthorizationService`
- `AuthService`

### Fonctions

- `get_current_user`
- `require_permission`
- `require_role`
- `get_auth_service`

