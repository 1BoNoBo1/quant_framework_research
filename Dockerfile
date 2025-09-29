# 🐳 QFrame API Backend Dockerfile
# Multi-stage build pour optimiser la taille de l'image

# Stage 1: Builder
FROM python:3.13-slim as builder

# Variables d'environnement pour Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installation des dépendances système pour la compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry
RUN pip install poetry==1.8.3

# Configuration Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copie des fichiers de configuration
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Installation des dépendances
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Stage 2: Runtime
FROM python:3.13-slim as runtime

# Variables d'environnement pour l'application
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    QFRAME_ENVIRONMENT=production

# Installation des dépendances système runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Création d'un utilisateur non-root pour la sécurité
RUN groupadd -r qframe && useradd -r -g qframe qframe

# Création des répertoires de l'application
WORKDIR /app
RUN mkdir -p /app/logs /app/data && \
    chown -R qframe:qframe /app

# Copie de l'environnement virtuel depuis le builder
COPY --from=builder --chown=qframe:qframe /app/.venv /app/.venv

# Copie du code source
COPY --chown=qframe:qframe . .

# Installation du package QFrame
RUN .venv/bin/pip install -e .

# Exposition du port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch vers l'utilisateur non-root
USER qframe

# Point d'entrée
CMD ["python", "start_api.py", "--host", "0.0.0.0", "--port", "8000"]