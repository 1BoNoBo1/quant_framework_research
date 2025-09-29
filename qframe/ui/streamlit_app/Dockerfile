# Dockerfile pour QFrame Streamlit GUI
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_PRIMARY_COLOR="#00ff88" \
    STREAMLIT_THEME_BACKGROUND_COLOR="#0e1117" \
    STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#262730"

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer utilisateur non-root
RUN useradd --create-home --shell /bin/bash qframe
USER qframe
WORKDIR /home/qframe

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --user --no-cache-dir -r requirements.txt

# Ajouter les binaires Python au PATH
ENV PATH="/home/qframe/.local/bin:${PATH}"

# Créer la structure de l'application
RUN mkdir -p /home/qframe/app/qframe/ui/streamlit_app

# Copier l'application Streamlit
COPY --chown=qframe:qframe . /home/qframe/app/qframe/ui/streamlit_app/

# Copier le framework QFrame depuis le contexte parent
COPY --chown=qframe:qframe ../../../core /home/qframe/app/qframe/core
COPY --chown=qframe:qframe ../../../domain /home/qframe/app/qframe/domain
COPY --chown=qframe:qframe ../../../infrastructure /home/qframe/app/qframe/infrastructure

# Définir le répertoire de travail
WORKDIR /home/qframe/app

# Configuration Streamlit
RUN mkdir -p /home/qframe/.streamlit
COPY --chown=qframe:qframe streamlit_config.toml /home/qframe/.streamlit/config.toml

# Script de démarrage
COPY --chown=qframe:qframe start.sh /home/qframe/start.sh
RUN chmod +x /home/qframe/start.sh

# Exposer le port Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Point d'entrée
ENTRYPOINT ["/home/qframe/start.sh"]