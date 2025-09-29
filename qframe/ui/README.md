# QFrame GUI - Interface Utilisateur Web

Interface web moderne pour le framework quantitatif QFrame, construite avec Streamlit et déployée via Docker.

## 🎯 Fonctionnalités

### ✅ Phase 1 - Streamlit (COMPLÉTÉE)

#### 📊 **Dashboard Principal**
- Vue d'ensemble des métriques clés (valeur totale, P&L, stratégies actives)
- Métriques de risque (VaR, Drawdown, Sharpe Ratio)
- Graphiques interactifs en temps réel avec Plotly
- Navigation rapide vers les sections spécialisées
- Auto-refresh configurable

#### 📁 **Gestion des Portfolios**
- **Vue d'ensemble** : Tableau de tous les portfolios avec métriques
- **Création** : Interface complète de création avec configuration avancée
- **Gestion** : Modification, recalcul et suppression des portfolios
- **Analyse** : Comparaison multi-portfolios et analyse de performance
- **Visualisations** : Allocation, évolution de valeur, P&L historique

#### 🎯 **Gestion des Stratégies**
- **Vue d'ensemble** : État et performance de toutes les stratégies
- **Création** : Support pour 6 types de stratégies (Mean Reversion, Momentum, Grid Trading, DMN LSTM, RL Alpha, Arbitrage)
- **Configuration** : Interface intuitive pour paramètres de trading et risque
- **Performance** : Analyse comparative avec graphiques et métriques détaillées
- **Contrôles** : Démarrage/arrêt des stratégies en un clic

#### ⚠️ **Gestion des Risques**
- **Vue d'ensemble** : Métriques de risque globales et par portfolio
- **Évaluation** : Calcul détaillé des risques avec paramètres personnalisables
- **Limites** : Configuration des seuils de risque et alertes
- **Alertes** : Monitoring en temps réel avec historique

#### 🎨 **Composants Réutilisables**
- **Charts** : 10+ types de graphiques avec Plotly (portfolio, stratégies, risque, ordres)
- **Tables** : Tableaux interactifs avec formatage et actions
- **API Client** : Client robuste avec gestion d'erreur et cache
- **Session Management** : État persistant avec TTL et préférences utilisateur

## 🚀 Démarrage Rapide

### Prérequis
- Docker & Docker Compose
- Git

### Installation

```bash
# Cloner le repository
git clone <repository_url>
cd quant_framework_research/qframe/ui

# Démarrer tous les services
chmod +x deploy.sh
./deploy.sh up

# Ou en mode développement avec monitoring
./deploy.sh -m up
```

### Accès aux Services

Une fois démarré, QFrame est accessible via :

- **🖥️ Interface GUI** : http://localhost:8501
- **🔌 API Backend** : http://localhost:8000
- **📚 Documentation API** : http://localhost:8000/docs
- **📊 Prometheus** (optionnel) : http://localhost:9090
- **📈 Grafana** (optionnel) : http://localhost:3000

## 🏗️ Architecture

### Services Docker

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  Streamlit GUI  │    │   QFrame API    │
│   Port 80/443   │◄──►│   Port 8501     │◄──►│   Port 8000     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │      Redis      │    │   Monitoring    │
│   Port 5432     │    │   Port 6379     │    │ Prometheus/Graf │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Structure de l'Application Streamlit

```
streamlit_app/
├── main.py                     # Point d'entrée principal
├── pages/                      # Pages Streamlit
│   ├── 01_🏠_Dashboard.py      # Dashboard principal
│   ├── 02_📁_Portfolios.py     # Gestion portfolios
│   ├── 03_🎯_Strategies.py     # Gestion stratégies
│   └── 05_⚠️_Risk_Management.py # Gestion risques
├── components/                 # Composants réutilisables
│   ├── charts.py              # Graphiques Plotly
│   └── tables.py              # Tableaux interactifs
├── utils/                     # Utilitaires
│   ├── api_client.py          # Client API QFrame
│   └── session_state.py       # Gestion de session
└── requirements.txt           # Dépendances Python
```

## 🎛️ Configuration

### Variables d'Environnement

```bash
# API Configuration
QFRAME_API_URL=http://localhost:8000

# Database
POSTGRES_USER=qframe
POSTGRES_PASSWORD=qframe123
POSTGRES_DB=qframe

# Security
SECRET_KEY=your-secret-key-here

# Monitoring (optionnel)
GRAFANA_PASSWORD=admin123
```

### Streamlit Configuration

Le fichier `streamlit_config.toml` configure :
- Thème sombre avec couleurs QFrame
- Performance et caching
- Sécurité et CORS
- WebSocket pour temps réel

## 📊 Fonctionnalités Détaillées

### Dashboard
- **Métriques temps réel** : Valeur portfolio, P&L, stratégies actives
- **Visualisations** : Graphiques de performance, allocation, timeline ordres
- **Alertes** : Système d'alertes avec sévérité et accusé de réception
- **Navigation** : Accès rapide aux sections spécialisées

### Portfolios
- **Création guidée** : Interface step-by-step avec validation
- **Monitoring** : Positions, performance, allocation en temps réel
- **Analyse comparative** : Comparaison multi-portfolios
- **Calcul de risque** : Intégration avec le moteur de risque

### Stratégies
- **Types supportés** :
  - Mean Reversion : Stratégie de retour à la moyenne adaptative
  - Momentum : Stratégie de momentum avec détection de tendance
  - Grid Trading : Grilles de trading automatiques
  - DMN LSTM : Deep Learning avec réseaux de neurones
  - RL Alpha : Génération d'alphas par Reinforcement Learning
  - Arbitrage : Arbitrage de taux de financement

- **Configuration avancée** : Paramètres de trading, gestion du risque, horaires
- **Monitoring** : Performance en temps réel, P&L, métriques de risque

### Risk Management
- **Évaluation complète** : VaR, CVaR, Drawdown, Volatilité
- **Limites configurables** : Seuils de risque avec alertes automatiques
- **Monitoring temps réel** : Détection de dépassement de limites
- **Historique** : Évolution du risque dans le temps

## 🔧 Commandes Utiles

### Gestion des Services

```bash
# Démarrer
./deploy.sh up

# Arrêter
./deploy.sh down

# Redémarrer
./deploy.sh restart

# Voir les logs
./deploy.sh logs                 # Tous les services
./deploy.sh logs qframe-gui      # Service spécifique

# Vérifier le statut
./deploy.sh status

# Nettoyer
./deploy.sh clean
```

### Sauvegarde/Restauration

```bash
# Sauvegarder
./deploy.sh backup

# Restaurer
./deploy.sh restore ./backups/20231201_120000
```

## 🛠️ Développement

### Mode Développement

```bash
# Démarrer en mode développement
ENVIRONMENT=development ./deploy.sh up

# Avec rechargement automatique
cd streamlit_app
streamlit run main.py --server.fileWatcherType=poll
```

### Structure de Développement

```bash
# Installation des dépendances
cd streamlit_app
pip install -r requirements.txt

# Variables d'environnement
export QFRAME_API_URL=http://localhost:8000

# Démarrage local
streamlit run main.py
```

### Tests

```bash
# Tests de l'API client
python -m pytest utils/test_api_client.py

# Tests des composants
python -m pytest components/test_charts.py
```

## 🔍 Monitoring & Observabilité

### Logs

```bash
# Logs en temps réel
docker-compose logs -f qframe-gui

# Logs avec filtrage
docker-compose logs qframe-gui | grep ERROR
```

### Métriques (avec profil monitoring)

- **Prometheus** : Collecte de métriques système et application
- **Grafana** : Dashboards de monitoring avec alertes
- **Health Checks** : Vérification automatique de santé des services

### Debugging

- **Session State** : Mode debug dans la sidebar pour inspecter l'état
- **API Client** : Gestion d'erreur avec logs détaillés
- **Performance** : Cache TTL configurable pour optimiser les performances

## 🔐 Sécurité

### Production

```bash
# Démarrer en mode production
ENVIRONMENT=production ./deploy.sh up
```

### Mesures de Sécurité

- **HTTPS** : Configuration SSL/TLS avec Nginx
- **Rate Limiting** : Protection contre les abus
- **Headers de sécurité** : CORS, XSS, CSRF protection
- **Secrets** : Variables d'environnement pour clés sensibles
- **User isolation** : Container utilisateur non-root

## 📋 Prochaines Étapes

### Phase 2 - React Application (PLANIFIÉE)
- Interface React moderne avec TypeScript
- Real-time updates avec WebSockets
- Advanced charting avec D3.js
- Mobile responsive design

### Phase 3 - Mobile PWA (PLANIFIÉE)
- Progressive Web App
- Notifications push
- Mode hors-ligne
- Interface tactile optimisée

## 🐛 Résolution de Problèmes

### Problèmes Courants

1. **API non disponible**
   ```bash
   # Vérifier les logs
   ./deploy.sh logs qframe-api

   # Redémarrer l'API
   docker-compose restart qframe-api
   ```

2. **Interface lente**
   ```bash
   # Vider le cache
   # Dans l'interface > Sidebar > Vider cache

   # Ou redémarrer
   ./deploy.sh restart qframe-gui
   ```

3. **Données manquantes**
   ```bash
   # Vérifier la base de données
   docker exec -it qframe-postgres psql -U qframe -d qframe

   # Recalculer les métriques
   # Dans l'interface > Actions > Recalculer
   ```

### Support

- **Documentation** : Consulter la documentation API à `/docs`
- **Logs** : Utiliser `./deploy.sh logs` pour diagnostiquer
- **Issues** : Créer une issue sur le repository GitHub

---

## 🎉 Conclusion

L'interface QFrame Streamlit offre une solution complète et moderne pour la gestion de portfolios quantitatifs. Avec son architecture containerisée, ses composants réutilisables et son interface intuitive, elle constitue une base solide pour le trading algorithmique professionnel.

**Status : Phase 1 COMPLÉTÉE** ✅
**Prochaine étape : Phase 2 React** 🚀