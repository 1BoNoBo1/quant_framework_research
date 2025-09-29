# ⚙️ Configuration

## Configuration Pydantic

QFrame utilise Pydantic pour une configuration type-safe :

```python
from qframe.core.config import get_config

# Configuration automatique avec environnement
config = get_config()

# Configuration personnalisée
config = FrameworkConfig(
    environment=Environment.DEVELOPMENT,
    database=DatabaseConfig(
        url="postgresql://user:pass@localhost/qframe"
    ),
    redis=RedisConfig(
        host="localhost",
        port=6379
    )
)
```

## Variables d'Environnement

Créez un fichier `.env` :

```bash
# Environnement
QFRAME_ENVIRONMENT=development

# Base de données
DATABASE_URL=postgresql://user:pass@localhost/qframe

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API Keys (optionnel)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

## Configuration par Environnement

### Development

```python
class DevelopmentConfig(FrameworkConfig):
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "DEBUG"
```

### Production

```python
class ProductionConfig(FrameworkConfig):
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = Field(..., min_length=32)
```

La configuration est validée automatiquement au démarrage.