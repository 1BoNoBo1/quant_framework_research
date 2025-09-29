#!/bin/bash
# =============================================================================
# 📚 Setup Documentation Professionnelle QFrame
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Setup Documentation Professionnelle QFrame"
echo "============================================="

# Vérifier Poetry
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry n'est pas installé. Installez Poetry d'abord."
    exit 1
fi

echo "📦 Installation des dépendances documentation..."
poetry install

echo "🔧 Génération des scripts documentation avancés..."

# Rendre les scripts exécutables
chmod +x docs/scripts/gen_api_docs.py
chmod +x docs/scripts/gen_benchmarks.py

echo "📊 Génération benchmarks..."
python docs/scripts/gen_benchmarks.py

echo "📖 Génération documentation API..."
python docs/scripts/gen_api_docs.py

echo "🏗️ Build documentation avec configuration professionnelle..."

# Arrêter le serveur actuel si il existe
pkill -f "mkdocs serve" || true
sleep 2

echo "🌐 Démarrage serveur documentation professionnelle..."

# Utiliser la nouvelle configuration
mkdocs serve -f mkdocs.professional.yml --dev-addr=127.0.0.1:8082 &
SERVER_PID=$!

echo "⏳ Attente du serveur..."
sleep 5

# Tester l'accès
if curl -s http://127.0.0.1:8082 > /dev/null; then
    echo "✅ Documentation professionnelle disponible sur http://127.0.0.1:8082"
    echo ""
    echo "🎉 Features Professionnelles Activées:"
    echo "  📖 Auto-génération API complète"
    echo "  📊 Benchmarks de performance"
    echo "  📈 Rapports de coverage"
    echo "  🔄 Macros avancées"
    echo "  🎨 Theme Material Pro"
    echo "  ⚡ Navigation optimisée"
    echo ""
    echo "🔗 URLs Importantes:"
    echo "  🏠 Accueil: http://127.0.0.1:8082"
    echo "  📖 API Reference: http://127.0.0.1:8082/reference/"
    echo "  📊 Benchmarks: http://127.0.0.1:8082/performance/benchmarks/"
    echo "  📚 API Documentation: http://127.0.0.1:8082/API_DOCUMENTATION/"
    echo ""
    echo "💡 Pour arrêter: kill $SERVER_PID"
else
    echo "❌ Erreur: Serveur non accessible"
    kill $SERVER_PID || true
    exit 1
fi