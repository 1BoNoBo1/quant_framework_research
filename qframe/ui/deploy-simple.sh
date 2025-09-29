#!/bin/bash

# Script de déploiement simplifié pour QFrame GUI
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

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

COMMAND=${1:-help}

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  up      Démarrer l'interface GUI"
    echo "  down    Arrêter l'interface GUI"
    echo "  logs    Afficher les logs"
    echo "  status  Vérifier le statut"
    echo "  test    Tester l'interface en local"
    echo "  help    Afficher cette aide"
}

start_gui() {
    log_info "Démarrage de l'interface QFrame GUI..."

    # Vérifier Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker n'est pas installé"
        exit 1
    fi

    # Build et démarrage
    log_info "Construction de l'image Docker..."
    docker compose -f docker-compose.local.yml build

    log_info "Démarrage du conteneur..."
    docker compose -f docker-compose.local.yml up -d

    # Attendre le démarrage
    log_info "Attente du démarrage..."
    sleep 10

    # Vérifier le statut
    check_status
}

stop_gui() {
    log_info "Arrêt de l'interface QFrame GUI..."
    docker compose -f docker-compose.local.yml down
    log_success "Interface arrêtée"
}

show_logs() {
    log_info "Logs de l'interface QFrame GUI:"
    docker compose -f docker-compose.local.yml logs -f qframe-gui
}

check_status() {
    log_info "Vérification du statut..."

    if docker ps --filter "name=qframe-gui" --filter "status=running" | grep -q "qframe-gui"; then
        log_success "✅ Interface GUI en cours d'exécution"

        # Test de connectivité
        sleep 5
        if curl -f -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
            log_success "✅ Interface accessible sur http://localhost:8501"
            echo ""
            echo "╔══════════════════════════════════════════════════════════════╗"
            echo "║                    🌟 QFrame GUI Active 🌟                   ║"
            echo "╠══════════════════════════════════════════════════════════════╣"
            echo "║                                                              ║"
            echo "║  📱 Interface Web:  http://localhost:8501                    ║"
            echo "║  🎯 Status:         Running in Docker                       ║"
            echo "║  📊 Features:       Dashboard, Portfolios, Strategies       ║"
            echo "║                                                              ║"
            echo "╚══════════════════════════════════════════════════════════════╝"
        else
            log_warning "⚠️ Conteneur en cours d'exécution mais interface non accessible"
        fi
    else
        log_error "❌ Interface GUI non démarrée"
        echo ""
        log_info "Pour voir les logs d'erreur:"
        log_info "  $0 logs"
        exit 1
    fi
}

test_local() {
    log_info "Test de l'interface en local (Poetry)..."

    # Aller au répertoire streamlit
    cd streamlit_app

    # Vérifier Poetry
    if ! command -v poetry &> /dev/null; then
        log_error "Poetry n'est pas installé"
        exit 1
    fi

    # Démarrer Streamlit en local
    log_info "Démarrage de Streamlit avec Poetry..."
    export QFRAME_API_URL=http://localhost:8000

    poetry run streamlit run main.py \
        --server.port=8502 \
        --server.address=localhost \
        --server.headless=true &

    STREAMLIT_PID=$!

    # Attendre le démarrage
    sleep 8

    # Vérifier
    if curl -f -s http://localhost:8502/_stcore/health > /dev/null 2>&1; then
        log_success "✅ Interface locale démarrée sur http://localhost:8502"
        log_info "PID: $STREAMLIT_PID"
        log_info "Pour arrêter: kill $STREAMLIT_PID"
    else
        log_error "❌ Erreur lors du démarrage local"
        kill $STREAMLIT_PID 2>/dev/null || true
        exit 1
    fi

    cd ..
}

case $COMMAND in
    up)
        start_gui
        ;;
    down)
        stop_gui
        ;;
    logs)
        show_logs
        ;;
    status)
        check_status
        ;;
    test)
        test_local
        ;;
    help|*)
        show_help
        ;;
esac