#!/bin/bash
# =============================================================================
# 📚 QFrame Documentation Server
# Script pour lancer et gérer la documentation MkDocs
# =============================================================================

set -euo pipefail

# Configuration
DOCS_PORT=${DOCS_PORT:-8080}
DOCS_HOST=${DOCS_HOST:-127.0.0.1}
WATCH_FILES=${WATCH_FILES:-true}

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Fonctions utilitaires
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

# Vérification des prérequis
check_requirements() {
    log_info "Vérification des prérequis..."

    # Vérifier Poetry
    if ! command -v poetry &> /dev/null; then
        log_error "Poetry n'est pas installé. Installez-le depuis https://python-poetry.org/"
        exit 1
    fi

    # Vérifier l'environnement virtuel
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_warning "Aucun environnement virtuel détecté, utilisation de Poetry..."
        POETRY_PREFIX="poetry run"
    else
        log_info "Environnement virtuel détecté: $VIRTUAL_ENV"
        POETRY_PREFIX=""
    fi

    # Vérifier MkDocs
    if ! $POETRY_PREFIX mkdocs --version &> /dev/null; then
        log_error "MkDocs n'est pas installé. Exécutez: poetry install --only dev"
        exit 1
    fi

    log_success "Prérequis validés ✅"
}

# Construction de la documentation
build_docs() {
    log_info "Construction de la documentation..."

    if $POETRY_PREFIX mkdocs build --clean; then
        log_success "Documentation construite avec succès 📚"
        return 0
    else
        log_error "Échec de la construction de la documentation"
        return 1
    fi
}

# Serveur de développement
serve_docs() {
    local host="${1:-$DOCS_HOST}"
    local port="${2:-$DOCS_PORT}"

    log_info "Démarrage du serveur de documentation..."
    log_info "URL: http://${host}:${port}"
    log_info "Ctrl+C pour arrêter"

    echo -e "${PURPLE}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════════╗
    ║                  📚 QFrame Documentation                      ║
    ║                                                               ║
    ║  🌐 Interface Web: http://127.0.0.1:8080                     ║
    ║  🔄 Auto-reload: Activé                                      ║
    ║  🎨 Thème: Material Design                                   ║
    ║  🔍 Recherche: Activée                                       ║
    ║                                                               ║
    ║  💡 Navigation rapide: Ctrl+K                                ║
    ║  🌙 Mode sombre/clair: Bouton en haut à droite              ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    # Démarrer le serveur avec options
    local mkdocs_args=(
        "serve"
        "--dev-addr=${host}:${port}"
        "--livereload"
    )

    if [[ "$WATCH_FILES" == "true" ]]; then
        mkdocs_args+=("--watch" "qframe/")
    fi

    $POETRY_PREFIX mkdocs "${mkdocs_args[@]}"
}

# Déploiement
deploy_docs() {
    local target="${1:-gh-pages}"

    log_info "Déploiement de la documentation vers $target..."

    if [[ "$target" == "gh-pages" ]]; then
        if $POETRY_PREFIX mkdocs gh-deploy --clean; then
            log_success "Documentation déployée sur GitHub Pages 🚀"
        else
            log_error "Échec du déploiement GitHub Pages"
            return 1
        fi
    else
        log_error "Target de déploiement non supporté: $target"
        return 1
    fi
}

# Validation de la documentation
validate_docs() {
    log_info "Validation de la documentation..."

    # Construire avec mode strict
    if $POETRY_PREFIX mkdocs build --strict --clean; then
        log_success "Documentation valide ✅"
    else
        log_warning "La documentation contient des warnings ou erreurs"
        log_info "Exécutez './scripts/serve-docs.sh build' pour voir les détails"
    fi

    # Vérifier les liens (si linkchecker est disponible)
    if command -v linkchecker &> /dev/null; then
        log_info "Vérification des liens..."
        if linkchecker site/index.html; then
            log_success "Tous les liens sont valides 🔗"
        else
            log_warning "Certains liens sont cassés"
        fi
    fi
}

# Statistiques de la documentation
stats_docs() {
    log_info "Statistiques de la documentation..."

    if [[ -d "site" ]]; then
        local total_files=$(find site -name "*.html" | wc -l)
        local total_size=$(du -sh site | cut -f1)

        echo
        echo -e "${BLUE}📊 Statistiques Documentation${NC}"
        echo -e "├── 📄 Fichiers HTML: ${GREEN}$total_files${NC}"
        echo -e "├── 💾 Taille totale: ${GREEN}$total_size${NC}"
        echo -e "├── 🎨 Thème: ${GREEN}Material Design${NC}"
        echo -e "└── 🔌 Plugins: ${GREEN}$(grep -c "^  - " mkdocs.yml) activés${NC}"
        echo
    else
        log_warning "Documentation non construite. Exécutez: ./scripts/serve-docs.sh build"
    fi
}

# Nettoyage
clean_docs() {
    log_info "Nettoyage des fichiers de documentation..."

    if [[ -d "site" ]]; then
        rm -rf site/
        log_success "Dossier 'site' supprimé"
    fi

    if [[ -d ".mkdocs_cache" ]]; then
        rm -rf .mkdocs_cache/
        log_success "Cache MkDocs supprimé"
    fi

    log_success "Nettoyage terminé 🧹"
}

# Aide
show_help() {
    cat << EOF
📚 QFrame Documentation Manager

Usage: $0 [COMMAND] [OPTIONS]

COMMANDS:
    serve           Lancer le serveur de développement (défaut)
    build           Construire la documentation statique
    deploy          Déployer sur GitHub Pages
    validate        Valider la documentation
    stats           Afficher les statistiques
    clean           Nettoyer les fichiers générés
    help            Afficher cette aide

OPTIONS (pour serve):
    --host HOST     Host du serveur (défaut: 127.0.0.1)
    --port PORT     Port du serveur (défaut: 8080)
    --no-watch      Désactiver le watch des fichiers source

EXEMPLES:
    $0                                    # Serveur de développement
    $0 serve --host 0.0.0.0 --port 9000  # Serveur accessible externe
    $0 build                              # Construction statique
    $0 deploy                             # Déploiement GitHub Pages
    $0 validate                           # Validation complète

ENVIRONNEMENT:
    DOCS_PORT       Port par défaut (défaut: 8080)
    DOCS_HOST       Host par défaut (défaut: 127.0.0.1)
    WATCH_FILES     Watch fichiers sources (défaut: true)

📖 Documentation: https://qframe.readthedocs.io
🐛 Issues: https://github.com/1BoNoBo1/quant_framework_research/issues
EOF
}

# Parse des arguments
parse_args() {
    local command="${1:-serve}"
    shift || true

    case "$command" in
        "serve")
            local host="$DOCS_HOST"
            local port="$DOCS_PORT"

            while [[ $# -gt 0 ]]; do
                case $1 in
                    --host)
                        host="$2"
                        shift 2
                        ;;
                    --port)
                        port="$2"
                        shift 2
                        ;;
                    --no-watch)
                        WATCH_FILES="false"
                        shift
                        ;;
                    *)
                        log_error "Option inconnue: $1"
                        show_help
                        exit 1
                        ;;
                esac
            done

            check_requirements
            serve_docs "$host" "$port"
            ;;

        "build")
            check_requirements
            build_docs
            ;;

        "deploy")
            local target="${1:-gh-pages}"
            check_requirements
            build_docs && deploy_docs "$target"
            ;;

        "validate")
            check_requirements
            validate_docs
            ;;

        "stats")
            stats_docs
            ;;

        "clean")
            clean_docs
            ;;

        "help"|"--help"|"-h")
            show_help
            ;;

        *)
            log_error "Commande inconnue: $command"
            show_help
            exit 1
            ;;
    esac
}

# Point d'entrée principal
main() {
    # Banner
    echo -e "${PURPLE}"
    cat << 'EOF'
   ____  ____                          ____
  / __ \/ __/________ _____ ___  ___  / __ \____  __________
 / / / / /_/ ___/ __ `/ __ `__ \/ _ \/ / / / __ \/ ___/ ___/
/ /_/ / __/ /  / /_/ / / / / / /  __/ /_/ / /_/ / /__(__  )
\____/_/ /_/   \__,_/_/ /_/ /_/\___/_____/\____/\___/____/

EOF
    echo -e "${NC}"

    parse_args "$@"
}

# Exécution si script appelé directement
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi