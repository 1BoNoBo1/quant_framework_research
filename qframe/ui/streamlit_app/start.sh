#!/bin/bash

# Script de démarrage pour QFrame Streamlit GUI
set -e

echo "🚀 Démarrage de QFrame Streamlit GUI..."

# Variables d'environnement par défaut
export QFRAME_API_URL=${QFRAME_API_URL:-"http://localhost:8000"}
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-"0.0.0.0"}

echo "📍 Configuration:"
echo "  - API URL: $QFRAME_API_URL"
echo "  - Streamlit Port: $STREAMLIT_SERVER_PORT"
echo "  - Streamlit Address: $STREAMLIT_SERVER_ADDRESS"

# Attendre que l'API soit disponible
echo "⏳ Attente de l'API QFrame..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f -s "$QFRAME_API_URL/health" > /dev/null 2>&1; then
        echo "✅ API QFrame disponible"
        break
    else
        echo "🔄 Tentative $attempt/$max_attempts - API non disponible, nouvelle tentative dans 5s..."
        sleep 5
        attempt=$((attempt + 1))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "⚠️  API QFrame non disponible après $max_attempts tentatives"
    echo "🚀 Démarrage de Streamlit sans connexion API..."
fi

# Créer le répertoire .streamlit s'il n'existe pas
mkdir -p /home/qframe/.streamlit

# Vérifier que les fichiers nécessaires existent
if [ ! -f "/home/qframe/app/qframe/ui/streamlit_app/main.py" ]; then
    echo "❌ Fichier main.py non trouvé"
    exit 1
fi

# Démarrer Streamlit
echo "🎯 Démarrage de Streamlit..."

cd /home/qframe/app/qframe/ui/streamlit_app

exec streamlit run main.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true \
    --browser.gatherUsageStats=false \
    --theme.primaryColor="#00ff88" \
    --theme.backgroundColor="#0e1117" \
    --theme.secondaryBackgroundColor="#262730" \
    --theme.textColor="#fafafa" \
    --theme.font="monospace"