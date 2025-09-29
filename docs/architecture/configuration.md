# ⚙️ Configuration Architecture

QFrame utilise Pydantic pour une configuration type-safe avec validation automatique.

## Structure Configuration

```python
class FrameworkConfig(BaseSettings):
    # Métadonnées
    app_name: str = "Quant Framework Research"
    environment: Environment = Environment.DEVELOPMENT

    # Configurations spécialisées
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    risk_management: RiskManagementConfig = RiskManagementConfig()
```

## Environnements

- **Development** : Log DEBUG, base locale
- **Production** : Log INFO, sécurité renforcée
- **Testing** : SQLite memory, mocks

## Variables d'Environnement

```bash
QFRAME_ENVIRONMENT=development
DATABASE_URL=postgresql://user:pass@localhost/qframe
REDIS_HOST=localhost
```

## Voir aussi

- [Installation](../getting-started/installation.md) - Configuration initiale
- [DI Container](di-container.md) - Injection de dépendances