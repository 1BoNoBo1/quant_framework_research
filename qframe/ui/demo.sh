#!/bin/bash

# Script de démonstration de l'interface QFrame GUI
set -e

echo "🚀 Démonstration QFrame GUI - Interface Web"
echo "============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[NOTE]${NC} $1"
}

# Aller au répertoire du projet
cd /home/jim/DEV/claude-code/quant_framework_research

echo ""
log_info "🎯 QFrame - Framework Quantitatif avec Interface Web Moderne"
echo ""
log_info "Phase 1 - Interface Streamlit: ✅ COMPLÉTÉE"
echo ""

# Arrêter les anciens processus
pkill -f streamlit > /dev/null 2>&1 || true
sleep 1

# Démarrer l'interface
log_info "Démarrage de l'interface QFrame GUI..."

cd qframe/ui/streamlit_app
export QFRAME_API_URL=http://localhost:8000

poetry run streamlit run main.py \
    --server.port=8501 \
    --server.address=localhost \
    --server.headless=true \
    --server.fileWatcherType=none \
    > /tmp/qframe_demo.log 2>&1 &

STREAMLIT_PID=$!

# Attendre le démarrage
sleep 6

# Vérifier que ça fonctionne
if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo "❌ Erreur lors du démarrage"
    cat /tmp/qframe_demo.log
    exit 1
fi

# Vérifier la connectivité
max_attempts=15
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        break
    else
        sleep 1
        attempt=$((attempt + 1))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Interface non accessible"
    kill $STREAMLIT_PID 2>/dev/null || true
    exit 1
fi

echo ""
log_success "🎉 Interface QFrame GUI démarrée avec succès!"
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    🌟 QFrame GUI Active 🌟                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  📱 Interface Web:  http://localhost:8501                    ║"
echo "║                                                              ║"
echo "║  📊 Fonctionnalités disponibles:                            ║"
echo "║     • Dashboard principal avec métriques temps réel         ║"
echo "║     • Gestion complète des portfolios                       ║"
echo "║     • Configuration des stratégies de trading               ║"
echo "║     • Monitoring avancé des risques                         ║"
echo "║     • Visualisations interactives avec Plotly               ║"
echo "║                                                              ║"
echo "║  🎨 Interface moderne:                                       ║"
echo "║     • Thème sombre professionnel                            ║"
echo "║     • Navigation intuitive par onglets                      ║"
echo "║     • Auto-refresh configurable                             ║"
echo "║     • Session state persistant                              ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

log_warning "L'interface est maintenant accessible dans votre navigateur."
log_warning "Ouvrez http://localhost:8501 pour explorer QFrame GUI."
echo ""

log_info "📋 Pages disponibles:"
echo "  🏠 Dashboard      - Vue d'ensemble et métriques clés"
echo "  📁 Portfolios     - Gestion complète des portfolios"
echo "  🎯 Strategies     - Configuration des stratégies de trading"
echo "  ⚠️  Risk Management - Monitoring et limites de risque"
echo ""

log_info "🎛️ Contrôles:"
echo "  • Sidebar         - Configuration et navigation"
echo "  • Auto-refresh    - Données temps réel"
echo "  • Filtres         - Personnalisation de l'affichage"
echo "  • Actions         - Création, modification, suppression"
echo ""

log_warning "📝 Note importante:"
echo "  L'API QFrame backend n'est pas démarrée dans cette démo."
echo "  Certaines fonctionnalités afficheront des messages d'erreur."
echo "  L'interface elle-même est complètement fonctionnelle."
echo ""

log_info "🚀 Pour une démonstration complète avec backend:"
echo "  1. Démarrer l'API QFrame sur http://localhost:8000"
echo "  2. L'interface se connectera automatiquement"
echo "  3. Toutes les fonctionnalités seront opérationnelles"
echo ""

log_info "⚙️ Commandes utiles:"
echo "  • Arrêter: kill $STREAMLIT_PID"
echo "  • Logs:    tail -f /tmp/qframe_demo.log"
echo "  • Test:    ./test_gui.sh"
echo ""

# Surveillance en arrière-plan
log_info "Surveillance active (PID: $STREAMLIT_PID)..."
echo "Appuyez sur Ctrl+C pour arrêter la démonstration."
echo ""

# Trap pour nettoyer en sortie
trap "echo ''; log_info 'Arrêt de la démonstration...'; kill $STREAMLIT_PID 2>/dev/null || true; exit 0" INT TERM

# Boucle de surveillance
while true; do
    if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
        echo ""
        echo "❌ L'interface s'est arrêtée de manière inattendue"
        echo "Logs:"
        tail -10 /tmp/qframe_demo.log
        exit 1
    fi

    # Afficher un heartbeat toutes les 30 secondes
    sleep 30
    echo "💓 Interface active - $(date '+%H:%M:%S') - http://localhost:8501"
done