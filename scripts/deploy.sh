#!/bin/bash
# 🚀 QFrame Deployment Script
# Script complet pour déployer QFrame en production

set -euo pipefail

# 🔧 Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"

# 🎨 Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 📝 Logging functions
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

# 🔍 Vérification des prérequis
check_prerequisites() {
    log_info "Vérification des prérequis..."

    # Vérifier Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker n'est pas installé"
        exit 1
    fi

    # Vérifier kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl n'est pas installé"
        exit 1
    fi

    # Vérifier kustomize
    if ! command -v kustomize &> /dev/null; then
        log_error "kustomize n'est pas installé"
        exit 1
    fi

    # Vérifier la connexion Kubernetes
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Impossible de se connecter au cluster Kubernetes"
        exit 1
    fi

    log_success "Tous les prérequis sont satisfaits"
}

# 🐳 Construction des images Docker
build_images() {
    log_info "Construction des images Docker..."

    cd "$PROJECT_ROOT"

    # Construction de l'image API
    log_info "Construction de l'image QFrame API..."
    docker build -t "qframe/api:$VERSION" -f Dockerfile .

    # Construction de l'image UI
    log_info "Construction de l'image QFrame UI..."
    docker build -t "qframe/ui:$VERSION" -f Dockerfile.ui .

    log_success "Images Docker construites avec succès"
}

# 📦 Push des images vers le registry
push_images() {
    log_info "Push des images vers le registry..."

    # Tag et push API
    docker tag "qframe/api:$VERSION" "ghcr.io/your-org/qframe-api:$VERSION"
    docker push "ghcr.io/your-org/qframe-api:$VERSION"

    # Tag et push UI
    docker tag "qframe/ui:$VERSION" "ghcr.io/your-org/qframe-ui:$VERSION"
    docker push "ghcr.io/your-org/qframe-ui:$VERSION"

    log_success "Images poussées vers le registry"
}

# 🏗️ Déploiement Kubernetes
deploy_kubernetes() {
    log_info "Déploiement sur Kubernetes ($ENVIRONMENT)..."

    cd "$PROJECT_ROOT/k8s"

    # Déterminer le namespace
    if [ "$ENVIRONMENT" = "production" ]; then
        NAMESPACE="qframe"
    else
        NAMESPACE="qframe-$ENVIRONMENT"
    fi

    # Créer le namespace s'il n'existe pas
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Mettre à jour les images dans kustomization
    kustomize edit set image "qframe/api=ghcr.io/your-org/qframe-api:$VERSION"
    kustomize edit set image "qframe/ui=ghcr.io/your-org/qframe-ui:$VERSION"

    # Appliquer la configuration
    kubectl apply -k . -n "$NAMESPACE"

    log_success "Configuration Kubernetes appliquée"
}

# ⏳ Attendre que les déploiements soient prêts
wait_for_deployment() {
    log_info "Attente de la disponibilité des déploiements..."

    local namespace
    if [ "$ENVIRONMENT" = "production" ]; then
        namespace="qframe"
    else
        namespace="qframe-$ENVIRONMENT"
    fi

    # Attendre l'API
    log_info "Attente du déploiement qframe-api..."
    kubectl rollout status deployment/qframe-api -n "$namespace" --timeout=600s

    # Attendre l'UI
    log_info "Attente du déploiement qframe-ui..."
    kubectl rollout status deployment/qframe-ui -n "$namespace" --timeout=600s

    # Attendre que les pods soient prêts
    kubectl wait --for=condition=ready pod -l app=qframe-api -n "$namespace" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=qframe-ui -n "$namespace" --timeout=300s

    log_success "Tous les déploiements sont prêts"
}

# 🧪 Tests de santé post-déploiement
health_checks() {
    log_info "Exécution des tests de santé..."

    local namespace
    if [ "$ENVIRONMENT" = "production" ]; then
        namespace="qframe"
    else
        namespace="qframe-$ENVIRONMENT"
    fi

    # Test de l'API
    log_info "Test de santé de l'API..."
    kubectl run health-test-api --image=curlimages/curl --rm -i --restart=Never -n "$namespace" -- \
        curl -f "http://qframe-api:8000/health" || {
        log_error "Test de santé de l'API échoué"
        return 1
    }

    # Test de l'UI
    log_info "Test de santé de l'UI..."
    kubectl run health-test-ui --image=curlimages/curl --rm -i --restart=Never -n "$namespace" -- \
        curl -f "http://qframe-ui:8501/_stcore/health" || {
        log_error "Test de santé de l'UI échoué"
        return 1
    }

    log_success "Tous les tests de santé ont réussi"
}

# 📊 Afficher le statut du déploiement
show_status() {
    log_info "Statut du déploiement:"

    local namespace
    if [ "$ENVIRONMENT" = "production" ]; then
        namespace="qframe"
    else
        namespace="qframe-$ENVIRONMENT"
    fi

    echo ""
    echo "📦 Pods:"
    kubectl get pods -n "$namespace" -o wide

    echo ""
    echo "🌐 Services:"
    kubectl get services -n "$namespace"

    echo ""
    echo "🔗 Ingress:"
    kubectl get ingress -n "$namespace" 2>/dev/null || echo "Aucun ingress configuré"

    echo ""
    echo "📊 Resource Usage:"
    kubectl top pods -n "$namespace" 2>/dev/null || echo "Metrics server non disponible"
}

# 🔄 Rollback en cas de problème
rollback() {
    log_warning "Rollback du déploiement..."

    local namespace
    if [ "$ENVIRONMENT" = "production" ]; then
        namespace="qframe"
    else
        namespace="qframe-$ENVIRONMENT"
    fi

    # Rollback API
    kubectl rollout undo deployment/qframe-api -n "$namespace"

    # Rollback UI
    kubectl rollout undo deployment/qframe-ui -n "$namespace"

    log_success "Rollback effectué"
}

# 🏠 Fonction principale
main() {
    log_info "🚀 Démarrage du déploiement QFrame"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"

    # Trap pour nettoyer en cas d'erreur
    trap 'log_error "Déploiement échoué! Voir les logs ci-dessus."' ERR

    check_prerequisites
    build_images

    # Push seulement si on n'est pas en développement local
    if [ "$ENVIRONMENT" != "local" ]; then
        push_images
    fi

    deploy_kubernetes
    wait_for_deployment

    # Tests de santé
    if ! health_checks; then
        log_error "Tests de santé échoués! Rollback recommandé."
        read -p "Voulez-vous effectuer un rollback? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback
        fi
        exit 1
    fi

    show_status

    log_success "🎉 Déploiement QFrame terminé avec succès!"

    if [ "$ENVIRONMENT" = "production" ]; then
        log_info "🌐 URLs de production:"
        log_info "  - UI: https://qframe.your-domain.com"
        log_info "  - API: https://api.qframe.your-domain.com"
        log_info "  - Monitoring: https://grafana.qframe.your-domain.com"
    fi
}

# 🆘 Aide
show_help() {
    cat << EOF
🚀 QFrame Deployment Script

Usage: $0 [ENVIRONMENT] [VERSION]

ENVIRONMENT:
  - local      : Déploiement local (défaut: staging)
  - staging    : Environnement de staging
  - production : Environnement de production

VERSION:
  - latest     : Dernière version (défaut)
  - v1.0.0     : Version spécifique
  - main-abc123: Version basée sur commit

Examples:
  $0                           # Déploie staging avec latest
  $0 production v1.0.0         # Déploie production avec v1.0.0
  $0 staging main-abc123       # Déploie staging avec version commit

Options:
  -h, --help   : Affiche cette aide
  --rollback   : Effectue un rollback du dernier déploiement

EOF
}

# 🎯 Point d'entrée
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --rollback)
        rollback
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac