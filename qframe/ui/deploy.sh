#!/bin/bash

# Script de déploiement QFrame avec GUI
set -e

# Colors pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Configuration par défaut
ENVIRONMENT=${ENVIRONMENT:-development}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-qframe123}
SECRET_KEY=${SECRET_KEY:-$(openssl rand -hex 32)}
COMPOSE_FILE="docker-compose.yml"

# Fonction d'aide
show_help() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  up          Démarrer tous les services"
    echo "  down        Arrêter tous les services"
    echo "  restart     Redémarrer tous les services"
    echo "  logs        Afficher les logs"
    echo "  status      Vérifier le statut des services"
    echo "  clean       Nettoyer les images et volumes inutilisés"
    echo "  backup      Sauvegarder les données"
    echo "  restore     Restaurer les données"
    echo ""
    echo "Options:"
    echo "  -e, --env     Environnement (development|production) [default: development]"
    echo "  -m, --monitor Inclure monitoring (Prometheus/Grafana)"
    echo "  -h, --help    Afficher cette aide"
    echo ""
    echo "Examples:"
    echo "  $0 up                    # Démarrer en mode développement"
    echo "  $0 -e production up      # Démarrer en production"
    echo "  $0 -m up                 # Démarrer avec monitoring"
    echo "  $0 logs qframe-gui       # Logs du service GUI"
}

# Parse arguments
INCLUDE_MONITORING=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -m|--monitor)
            INCLUDE_MONITORING=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            COMMAND="$1"
            shift
            break
            ;;
    esac
done

# Vérifier Docker
check_dependencies() {
    log_info "Vérification des dépendances..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker n'est pas installé"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose n'est pas installé"
        exit 1
    fi

    log_success "Dépendances vérifiées"
}

# Créer les fichiers d'environnement
setup_environment() {
    log_info "Configuration de l'environnement: $ENVIRONMENT"

    # Créer .env
    cat > .env << EOF
# QFrame Environment Configuration
ENVIRONMENT=$ENVIRONMENT
SECRET_KEY=$SECRET_KEY
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_USER=qframe
POSTGRES_DB=qframe
LOG_LEVEL=INFO
GRAFANA_PASSWORD=admin123

# URLs
DATABASE_URL=postgresql://qframe:$POSTGRES_PASSWORD@postgres:5432/qframe
REDIS_URL=redis://redis:6379/0

# Monitoring
PROMETHEUS_RETENTION=15d
EOF

    # Créer init-db.sql
    cat > init-db.sql << 'EOF'
-- Initialisation de la base de données QFrame
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Tables de base (à adapter selon votre schéma)
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    initial_capital DECIMAL(20, 8) NOT NULL,
    base_currency VARCHAR(10) DEFAULT 'USD',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    portfolio_id UUID REFERENCES portfolios(id),
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id),
    strategy_id UUID REFERENCES strategies(id),
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes pour performance
CREATE INDEX IF NOT EXISTS idx_portfolios_created_at ON portfolios(created_at);
CREATE INDEX IF NOT EXISTS idx_strategies_portfolio_id ON strategies(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_orders_portfolio_id ON orders(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
EOF

    log_success "Environnement configuré"
}

# Démarrer les services
start_services() {
    log_info "Démarrage des services QFrame..."

    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi

    local profiles=""
    if [ "$INCLUDE_MONITORING" = true ]; then
        profiles="--profile monitoring"
        log_info "Monitoring activé (Prometheus + Grafana)"
    fi

    $compose_cmd $profiles up -d --build

    log_info "Attente du démarrage des services..."
    sleep 10

    # Vérifier le statut
    check_services_health
}

# Arrêter les services
stop_services() {
    log_info "Arrêt des services QFrame..."

    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi

    $compose_cmd down

    log_success "Services arrêtés"
}

# Redémarrer les services
restart_services() {
    log_info "Redémarrage des services QFrame..."
    stop_services
    start_services
}

# Vérifier la santé des services
check_services_health() {
    log_info "Vérification de la santé des services..."

    local services=("qframe-api" "qframe-gui" "redis" "postgres")
    local healthy_count=0

    for service in "${services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            log_success "✓ $service: Running"
            ((healthy_count++))
        else
            log_error "✗ $service: Not running"
        fi
    done

    echo ""
    log_info "Services en cours d'exécution: $healthy_count/${#services[@]}"

    if [ $healthy_count -eq ${#services[@]} ]; then
        echo ""
        log_success "🎉 QFrame est maintenant accessible:"
        log_info "   • Interface GUI: http://localhost:8501"
        log_info "   • API Backend:   http://localhost:8000"
        log_info "   • API Docs:      http://localhost:8000/docs"

        if [ "$INCLUDE_MONITORING" = true ]; then
            log_info "   • Prometheus:    http://localhost:9090"
            log_info "   • Grafana:       http://localhost:3000"
        fi
    fi
}

# Afficher les logs
show_logs() {
    local service=${1:-}

    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi

    if [ -n "$service" ]; then
        log_info "Logs du service: $service"
        $compose_cmd logs -f "$service"
    else
        log_info "Logs de tous les services"
        $compose_cmd logs -f
    fi
}

# Nettoyer le système
clean_system() {
    log_info "Nettoyage du système Docker..."

    docker system prune -f
    docker volume prune -f
    docker image prune -f

    log_success "Nettoyage terminé"
}

# Sauvegarder les données
backup_data() {
    local backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"

    log_info "Sauvegarde des données vers: $backup_dir"

    mkdir -p "$backup_dir"

    # Backup PostgreSQL
    docker exec qframe-postgres pg_dump -U qframe qframe > "$backup_dir/postgres_backup.sql"

    # Backup Redis
    docker exec qframe-redis redis-cli BGSAVE
    docker cp qframe-redis:/data/dump.rdb "$backup_dir/redis_backup.rdb"

    # Backup volumes
    docker run --rm -v qframe_qframe_data:/data -v "$PWD/$backup_dir":/backup alpine tar czf /backup/qframe_data.tar.gz -C /data .

    log_success "Sauvegarde terminée: $backup_dir"
}

# Restaurer les données
restore_data() {
    local backup_dir=${1:-}

    if [ -z "$backup_dir" ]; then
        log_error "Usage: $0 restore <backup_directory>"
        exit 1
    fi

    if [ ! -d "$backup_dir" ]; then
        log_error "Répertoire de sauvegarde non trouvé: $backup_dir"
        exit 1
    fi

    log_warning "⚠️  La restauration va écraser les données existantes!"
    read -p "Continuer? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restauration annulée"
        exit 0
    fi

    log_info "Restauration des données depuis: $backup_dir"

    # Restaurer PostgreSQL
    if [ -f "$backup_dir/postgres_backup.sql" ]; then
        docker exec -i qframe-postgres psql -U qframe -d qframe < "$backup_dir/postgres_backup.sql"
        log_success "PostgreSQL restauré"
    fi

    # Restaurer Redis
    if [ -f "$backup_dir/redis_backup.rdb" ]; then
        docker cp "$backup_dir/redis_backup.rdb" qframe-redis:/data/dump.rdb
        docker restart qframe-redis
        log_success "Redis restauré"
    fi

    log_success "Restauration terminée"
}

# Main
main() {
    case $COMMAND in
        up)
            check_dependencies
            setup_environment
            start_services
            ;;
        down)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs "$1"
            ;;
        status)
            check_services_health
            ;;
        clean)
            clean_system
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$1"
            ;;
        *)
            log_error "Commande inconnue: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Exécuter la commande
if [ -z "$COMMAND" ]; then
    show_help
    exit 1
fi

main "$@"