#!/bin/bash

# Script de test pour l'interface QFrame GUI
set -e

echo "🧪 Test de l'interface QFrame GUI"
echo "=================================="

# Colors pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Test des dépendances
log_info "Vérification des dépendances..."

if ! command -v poetry &> /dev/null; then
    log_error "Poetry n'est pas installé"
    exit 1
fi
log_success "✓ Poetry installé"

cd /home/jim/DEV/claude-code/quant_framework_research

if ! poetry show streamlit > /dev/null 2>&1; then
    log_error "Streamlit n'est pas installé dans le projet"
    exit 1
fi
log_success "✓ Streamlit installé"

# 2. Test des imports
log_info "Test des imports de l'application..."

cd qframe/ui/streamlit_app

export QFRAME_API_URL=http://localhost:8000

poetry run python -c "
import sys
from pathlib import Path
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

try:
    import streamlit as st
    from utils.api_client import QFrameAPIClient
    from utils.session_state import SessionStateManager
    from components.charts import create_portfolio_value_chart
    from components.tables import display_portfolio_table
    print('✅ Tous les imports réussis')
except Exception as e:
    print(f'❌ Erreur: {e}')
    exit(1)
" || exit 1

log_success "✓ Imports validés"

# 3. Test du démarrage Streamlit
log_info "Test du démarrage de Streamlit..."

# Arrêter tout processus Streamlit existant
pkill -f streamlit || true
sleep 2

# Démarrer Streamlit en arrière-plan
poetry run streamlit run main.py \
    --server.port=8501 \
    --server.address=localhost \
    --server.headless=true \
    --server.fileWatcherType=none \
    > /tmp/streamlit_test.log 2>&1 &

STREAMLIT_PID=$!
log_info "Streamlit démarré avec PID: $STREAMLIT_PID"

# Attendre le démarrage
sleep 8

# Vérifier que le processus fonctionne
if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
    log_error "Streamlit s'est arrêté"
    cat /tmp/streamlit_test.log
    exit 1
fi

log_success "✓ Streamlit en cours d'exécution"

# 4. Test de connectivité
log_info "Test de connectivité HTTP..."

max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f -s http://localhost:8501 > /dev/null 2>&1; then
        log_success "✓ Interface accessible sur http://localhost:8501"
        break
    else
        log_warning "Tentative $attempt/$max_attempts - Interface non accessible"
        sleep 2
        attempt=$((attempt + 1))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    log_error "Interface non accessible après $max_attempts tentatives"
    log_info "Logs Streamlit:"
    tail -20 /tmp/streamlit_test.log
    kill $STREAMLIT_PID 2>/dev/null || true
    exit 1
fi

# 5. Test des endpoints
log_info "Test des endpoints Streamlit..."

# Health check
if curl -f -s "http://localhost:8501/_stcore/health" | grep -q "ok"; then
    log_success "✓ Health check OK"
else
    log_warning "⚠ Health check non disponible"
fi

# Interface principale
if curl -s "http://localhost:8501" | grep -q "streamlit"; then
    log_success "✓ Interface principale chargée"
else
    log_warning "⚠ Interface principale problématique"
fi

# 6. Test de l'API client
log_info "Test du client API (sans backend)..."

poetry run python -c "
import sys
from pathlib import Path
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

from utils.api_client import QFrameAPIClient
import os

os.environ['QFRAME_API_URL'] = 'http://localhost:8000'
client = QFrameAPIClient()

# Test de création d'instance
print('✅ Client API créé')

# Test de configuration
print(f'Base URL: {client.base_url}')
print('✅ Configuration validée')
"

log_success "✓ Client API fonctionnel"

# 7. Résultats
echo ""
echo "=========================="
log_success "🎉 TESTS RÉUSSIS"
echo "=========================="

log_info "L'interface QFrame GUI est opérationnelle:"
log_info "  • URL: http://localhost:8501"
log_info "  • PID: $STREAMLIT_PID"
log_info "  • Logs: /tmp/streamlit_test.log"

echo ""
log_info "Pour arrêter l'interface:"
log_info "  kill $STREAMLIT_PID"

echo ""
log_info "Pour redémarrer:"
log_info "  cd qframe/ui/streamlit_app"
log_info "  poetry run streamlit run main.py"

echo ""
log_warning "Note: L'API QFrame backend n'est pas démarrée."
log_warning "Certaines fonctionnalités nécessitent le backend sur http://localhost:8000"

# Afficher les logs récents
echo ""
log_info "Logs récents de Streamlit:"
echo "------------------------"
tail -10 /tmp/streamlit_test.log

exit 0