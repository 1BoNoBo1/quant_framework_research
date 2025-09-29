#!/bin/bash

# Script de vérification du statut global QFrame GUI
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

echo "🔍 QFrame GUI - Vérification du statut global"
echo "=============================================="

# Vérifier les ports Streamlit
PORTS_TO_CHECK=(8501 8502)
ACTIVE_INTERFACES=()

for port in "${PORTS_TO_CHECK[@]}"; do
    if curl -f -s "http://localhost:$port/_stcore/health" > /dev/null 2>&1; then
        ACTIVE_INTERFACES+=($port)
        log_success "✅ Interface active sur port $port"
    fi
done

# Vérifier les conteneurs Docker
DOCKER_RUNNING=false
if docker ps --filter "name=qframe-gui" --filter "status=running" | grep -q "qframe-gui"; then
    DOCKER_RUNNING=true
    log_success "✅ Conteneur Docker qframe-gui en cours d'exécution"
fi

# Vérifier les processus Streamlit locaux
STREAMLIT_PROCESSES=$(pgrep -f "streamlit run" | wc -l)
if [ $STREAMLIT_PROCESSES -gt 0 ]; then
    log_success "✅ $STREAMLIT_PROCESSES processus Streamlit locaux actifs"

    # Afficher les PIDs
    PIDS=$(pgrep -f "streamlit run")
    log_info "PIDs: $PIDS"
fi

echo ""
echo "📊 Résumé du statut"
echo "==================="

if [ ${#ACTIVE_INTERFACES[@]} -gt 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🌟 QFrame GUI Active 🌟                   ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║                                                              ║"

    for port in "${ACTIVE_INTERFACES[@]}"; do
        echo "║  📱 Interface disponible:  http://localhost:$port           ║"
    done

    echo "║                                                              ║"
    echo "║  🎯 Fonctionnalités:                                        ║"
    echo "║     • Dashboard principal avec métriques                    ║"
    echo "║     • Gestion des portfolios                                ║"
    echo "║     • Configuration des stratégies                          ║"
    echo "║     • Monitoring des risques                                ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    echo ""
    log_info "🎛️ Commandes utiles:"

    if [ $STREAMLIT_PROCESSES -gt 0 ]; then
        echo "  • Arrêter processus locaux: pkill -f 'streamlit run'"
    fi

    if [ "$DOCKER_RUNNING" = true ]; then
        echo "  • Arrêter Docker: ./deploy-simple.sh down"
        echo "  • Voir logs Docker: ./deploy-simple.sh logs"
    fi

    echo "  • Test interface: ./deploy-simple.sh test"
    echo "  • Redémarrer: ./deploy-simple.sh up"

else
    log_warning "⚠️ Aucune interface QFrame GUI active"
    echo ""
    log_info "🚀 Pour démarrer l'interface:"
    echo "  • Test local: ./deploy-simple.sh test"
    echo "  • Docker: ./deploy-simple.sh up"
    echo "  • Manuel: cd streamlit_app && poetry run streamlit run main.py"
fi

echo ""
log_info "🔍 Pour vérifier l'état de santé:"
echo "  curl http://localhost:8501/_stcore/health"
echo "  curl http://localhost:8502/_stcore/health"

# Vérifier la connectivité API (optionnel)
echo ""
log_info "🔌 Connectivité API QFrame:"
if curl -f -s "http://localhost:8000/health" > /dev/null 2>&1; then
    log_success "✅ API QFrame accessible sur http://localhost:8000"
else
    log_warning "⚠️ API QFrame non accessible (optionnel pour GUI)"
fi