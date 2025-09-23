# QFrame Deployment Guide

Complete guide for deploying QFrame to production.

## Quick Start - Local Development

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f qframe-api

# Access services:
# - API: http://localhost:8000
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9091
# - Jaeger: http://localhost:16686
```

### Manual Setup

```bash
# Install dependencies
poetry install

# Start PostgreSQL
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=qframe \
  -e POSTGRES_PASSWORD=qframe \
  -e POSTGRES_DB=qframe \
  postgres:15-alpine

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Start InfluxDB
docker run -d -p 8086:8086 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=qframe \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=qframepassword \
  influxdb:2.7-alpine

# Run migrations
poetry run python -m qframe.infrastructure.persistence.migrations

# Start API server
poetry run uvicorn qframe.infrastructure.api.rest:app --reload
```

## Production Deployment

### Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- Docker registry access
- PostgreSQL database (managed service recommended)
- Redis cluster
- InfluxDB instance

### Step 1: Build and Push Docker Image

```bash
# Build image
docker build -t qframe/qframe:v1.0.0 .

# Tag for registry
docker tag qframe/qframe:v1.0.0 your-registry.com/qframe:v1.0.0

# Push to registry
docker push your-registry.com/qframe:v1.0.0
```

### Step 2: Configure Secrets

```bash
# Create namespace
kubectl create namespace qframe-prod

# Create secrets
kubectl create secret generic qframe-secrets \
  --from-literal=database-url='postgresql://user:pass@db-host:5432/qframe' \
  --from-literal=redis-url='redis://redis-host:6379/0' \
  --from-literal=influxdb-url='http://influxdb-host:8086' \
  --from-literal=binance-api-key='your-api-key' \
  --from-literal=binance-api-secret='your-api-secret' \
  -n qframe-prod
```

### Step 3: Deploy to Kubernetes

```bash
# Apply base configuration
kubectl apply -f deployment/kubernetes/base/ -n qframe-prod

# Apply production-specific config
kubectl apply -f deployment/kubernetes/production/ -n qframe-prod

# Check deployment status
kubectl get pods -n qframe-prod
kubectl get svc -n qframe-prod

# View logs
kubectl logs -f deployment/qframe-api -n qframe-prod
```

### Step 4: Configure Monitoring

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace

# Apply ServiceMonitor for QFrame
kubectl apply -f monitoring/prometheus/servicemonitor.yaml

# Import Grafana dashboards
kubectl create configmap qframe-dashboard \
  --from-file=monitoring/grafana/qframe-dashboard.json \
  -n monitoring
```

### Step 5: Configure Ingress

```bash
# Install nginx-ingress
helm install nginx-ingress ingress-nginx/ingress-nginx

# Apply ingress rules
kubectl apply -f deployment/kubernetes/production/ingress.yaml
```

### Step 6: Setup SSL/TLS

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f deployment/kubernetes/production/cert-issuer.yaml
```

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `REDIS_URL` | Redis connection string | `redis://host:6379/0` |
| `INFLUXDB_URL` | InfluxDB URL | `http://host:8086` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_PORT` | API server port | `8000` |
| `METRICS_PORT` | Metrics port | `9090` |
| `MAX_WORKERS` | Worker processes | `4` |

## Scaling

### Horizontal Pod Autoscaling

```bash
# HPA is configured automatically based on CPU/Memory
# View current status
kubectl get hpa -n qframe-prod

# Manual scaling
kubectl scale deployment qframe-api --replicas=5 -n qframe-prod
```

### Database Scaling

- Use managed PostgreSQL service (AWS RDS, GCP Cloud SQL)
- Configure read replicas for queries
- Use connection pooling (PgBouncer)

### Redis Scaling

- Use Redis Cluster mode
- Configure sentinel for high availability
- Use separate instances for cache vs. pub/sub

## Backup & Recovery

### Database Backups

```bash
# Automated daily backups (add to cron)
pg_dump -h db-host -U qframe qframe | gzip > backup-$(date +%Y%m%d).sql.gz

# Restore from backup
gunzip < backup-20241225.sql.gz | psql -h db-host -U qframe qframe
```

### InfluxDB Backups

```bash
# Backup
influx backup /backup/path

# Restore
influx restore /backup/path
```

## Monitoring & Alerting

### Key Metrics to Monitor

1. **Trading Performance**
   - P&L (realized/unrealized)
   - Win rate
   - Sharpe ratio
   - Max drawdown

2. **System Health**
   - API response time (p95, p99)
   - Error rate
   - CPU/Memory usage
   - Database connections

3. **Business Metrics**
   - Trades executed
   - Active strategies
   - Order fill rate
   - Slippage

### Alerting Rules

```yaml
# Example Prometheus alert
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  annotations:
    summary: High error rate detected
```

## Troubleshooting

### Common Issues

**1. Pod Not Starting**
```bash
kubectl describe pod <pod-name> -n qframe-prod
kubectl logs <pod-name> -n qframe-prod
```

**2. Database Connection Issues**
```bash
# Test connection from pod
kubectl exec -it <pod-name> -n qframe-prod -- \
  psql $DATABASE_URL -c "SELECT version();"
```

**3. High Memory Usage**
```bash
# Check memory usage
kubectl top pods -n qframe-prod

# Adjust memory limits in deployment.yaml
```

### Performance Tuning

1. **Database Query Optimization**
   - Enable query logging
   - Add indexes for frequent queries
   - Use EXPLAIN ANALYZE

2. **API Performance**
   - Enable response caching
   - Use async endpoints
   - Implement request batching

3. **Worker Optimization**
   - Tune worker processes
   - Use process pools for CPU-bound tasks
   - Implement task queues (Celery/RQ)

## Security Best Practices

1. **Secrets Management**
   - Use Kubernetes secrets or external secret managers (Vault, AWS Secrets Manager)
   - Rotate credentials regularly
   - Never commit secrets to git

2. **Network Security**
   - Use Network Policies to restrict pod communication
   - Enable TLS for all external connections
   - Implement API rate limiting

3. **Access Control**
   - Use RBAC for Kubernetes access
   - Implement JWT authentication for API
   - Enable audit logging

## Disaster Recovery

### High Availability Setup

```yaml
# Multi-zone deployment
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - topologyKey: topology.kubernetes.io/zone
```

### Recovery Procedures

1. **Database Failure**
   - Promote read replica
   - Restore from backup
   - Update connection string

2. **Complete Region Failure**
   - Failover to secondary region
   - Update DNS records
   - Verify data consistency

## CI/CD Integration

See `.github/workflows/ci.yml` for automated deployment pipeline.

### Manual Deployment

```bash
# Tag release
git tag v1.0.0
git push origin v1.0.0

# Build and deploy
./scripts/deploy.sh production v1.0.0
```

## Support

- Documentation: `/docs`
- Issues: https://github.com/1BoNoBo1/quant_framework_research/issues
- Email: support@qframe.dev
