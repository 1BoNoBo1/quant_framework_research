# 🐳 QFrame Kubernetes Deployment

Configuration Kubernetes complète pour le framework QFrame en production.

## 📋 Vue d'ensemble

Cette configuration déploie un stack complet comprenant :

- **🚀 API Backend** : FastAPI avec haute disponibilité (2-3 réplicas)
- **🖥️ UI Frontend** : Streamlit pour l'interface utilisateur
- **🗄️ Base de données** : PostgreSQL avec persistence
- **🔴 Cache** : Redis pour les sessions et cache
- **📊 Monitoring** : Prometheus + Grafana
- **🌐 Ingress** : Exposition sécurisée avec TLS

## 🚀 Déploiement Rapide

### Prérequis

1. **Cluster Kubernetes** (v1.24+)
2. **kubectl** configuré
3. **Ingress Controller** (nginx recommandé)
4. **Storage Classes** : `fast-ssd`, `standard`

### Déploiement

```bash
# 1. Construction des images Docker
docker build -t qframe/api:latest -f Dockerfile .
docker build -t qframe/ui:latest -f Dockerfile.ui .

# 2. Push vers registry (adapter selon votre registry)
docker tag qframe/api:latest your-registry/qframe/api:latest
docker push your-registry/qframe/api:latest
docker tag qframe/ui:latest your-registry/qframe/ui:latest
docker push your-registry/qframe/ui:latest

# 3. Déploiement avec Kustomize
kubectl apply -k k8s/

# 4. Vérification
kubectl get pods -n qframe
kubectl get services -n qframe
kubectl get ingress -n qframe
```

## 📁 Structure des fichiers

```
k8s/
├── namespace.yaml           # Namespace et policies
├── configmap.yaml          # Configuration non-sensible
├── storage.yaml            # PVC pour persistence
├── postgres.yaml           # Base de données PostgreSQL
├── redis.yaml              # Cache Redis
├── qframe-api.yaml         # API Backend + HPA
├── qframe-ui.yaml          # Interface Streamlit
├── monitoring.yaml         # Prometheus + Grafana
├── ingress.yaml            # Exposition externe + TLS
├── kustomization.yaml      # Configuration Kustomize
├── production-patches.yaml # Patches production
└── README.md               # Cette documentation
```

## 🔧 Configuration

### Variables d'environnement

Principales configurations dans `configmap.yaml` :

```yaml
QFRAME_ENVIRONMENT: "production"
QFRAME_LOG_LEVEL: "INFO"
POSTGRES_DB: "qframe"
STREAMLIT_SERVER_HEADLESS: "true"
```

### Secrets

Configurez les secrets sensibles dans `configmap.yaml` :

```bash
# Modifier les secrets avant déploiement
kubectl edit secret qframe-secrets -n qframe
```

**⚠️ IMPORTANT** : Changez tous les mots de passe par défaut !

### Domaines

Modifiez les domaines dans `ingress.yaml` :

```yaml
rules:
- host: qframe.your-domain.com      # Interface UI
- host: api.qframe.your-domain.com  # API Backend
- host: grafana.qframe.your-domain.com  # Monitoring
```

## 📊 Monitoring

### Prometheus

- **URL** : `http://grafana.your-domain.com`
- **Métriques** : API, UI, PostgreSQL, Redis
- **Alertes** : Configurées pour panne API, mémoire haute

### Grafana

- **URL** : `http://grafana.your-domain.com`
- **Login** : admin / admin (à changer !)
- **Dashboards** : Pré-configurés pour QFrame

### Vérifications de santé

```bash
# Status des pods
kubectl get pods -n qframe

# Logs de l'API
kubectl logs -f deployment/qframe-api -n qframe

# Status des services
kubectl get svc -n qframe

# Métriques Prometheus
kubectl port-forward svc/qframe-prometheus 9090:9090 -n qframe
# Ouvrir http://localhost:9090
```

## 🔐 Sécurité

### TLS/SSL

1. **Certificats** : Configurez cert-manager ou certificats manuels
2. **Ingress** : Force redirect HTTPS
3. **Network Policies** : Isolation réseau entre composants

### Accès

1. **API Keys** : Stockées dans secrets Kubernetes
2. **Database** : Accès restreint au namespace
3. **Monitoring** : Authentification Grafana obligatoire

## 📈 Scaling

### Horizontal Pod Autoscaler

L'API scale automatiquement :

```yaml
minReplicas: 2
maxReplicas: 10
targetCPUUtilizationPercentage: 70
```

### Manuel

```bash
# Scaler l'API
kubectl scale deployment qframe-api --replicas=5 -n qframe

# Scaler PostgreSQL (attention aux données !)
kubectl scale deployment qframe-postgres --replicas=1 -n qframe
```

## 🔧 Maintenance

### Mises à jour

```bash
# Mise à jour de l'image API
kubectl set image deployment/qframe-api qframe-api=qframe/api:v1.1.0 -n qframe

# Rollback si problème
kubectl rollout undo deployment/qframe-api -n qframe

# Historique des déploiements
kubectl rollout history deployment/qframe-api -n qframe
```

### Backup

```bash
# Backup PostgreSQL
kubectl exec -it deployment/qframe-postgres -n qframe -- pg_dump -U qframe qframe > backup.sql

# Backup volumes (selon votre storage)
kubectl get pvc -n qframe
```

### Logs

```bash
# Logs de tous les pods
kubectl logs -f -l app.kubernetes.io/name=qframe -n qframe

# Logs spécifiques
kubectl logs -f deployment/qframe-api -n qframe
kubectl logs -f deployment/qframe-ui -n qframe
```

## 🐛 Troubleshooting

### Problèmes courants

1. **Pods en Pending** : Vérifier les ressources et storage classes
2. **ImagePullBackOff** : Vérifier les URLs d'images et registry access
3. **CrashLoopBackOff** : Vérifier les logs et configuration

### Commandes utiles

```bash
# Debug d'un pod
kubectl describe pod <pod-name> -n qframe

# Shell dans un pod
kubectl exec -it <pod-name> -n qframe -- /bin/bash

# Port forwarding pour debug
kubectl port-forward svc/qframe-api 8000:8000 -n qframe
kubectl port-forward svc/qframe-ui 8501:8501 -n qframe

# Events du namespace
kubectl get events -n qframe --sort-by='.firstTimestamp'
```

## 🔄 Environments

### Development

```bash
# Utiliser le cluster local (minikube/kind)
kubectl apply -k k8s/ --dry-run=client

# Réduire les ressources
kubectl patch deployment qframe-api -n qframe -p '{"spec":{"replicas":1}}'
```

### Production

```bash
# Utiliser les patches de production
kubectl apply -k k8s/

# Vérifier les ressources allouées
kubectl top pods -n qframe
kubectl top nodes
```

## 📞 Support

- **Documentation** : [QFrame Docs](../README.md)
- **Issues** : Utiliser les logs Kubernetes et Grafana
- **Monitoring** : Prometheus alertes configurées

---

**🎯 Le framework QFrame est maintenant prêt pour la production avec Kubernetes !**