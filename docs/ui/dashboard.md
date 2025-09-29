# 🖥️ Dashboard Web

Interface web moderne Streamlit avec gestion complète des portfolios et monitoring temps réel.

## Architecture Interface

```
qframe/ui/
├── streamlit_app/              # Application Streamlit principale
│   ├── main.py                 # Entry point avec configuration
│   ├── pages/                  # Pages multi-onglets
│   │   ├── 01_🏠_Dashboard.py  # Dashboard principal
│   │   ├── 02_📁_Portfolios.py # Gestion portfolios
│   │   ├── 03_🎯_Strategies.py # Configuration stratégies
│   │   └── 05_⚠️_Risk_Management.py # Monitoring risques
│   ├── components/             # Composants réutilisables
│   └── api_client.py          # Client API QFrame
├── deploy-simple.sh           # Script déploiement
└── docker-compose.local.yml   # Configuration Docker
```

## Fonctionnalités

### 🏠 Dashboard Principal
- **Métriques temps réel** : P&L, Sharpe ratio, drawdown
- **Graphiques performance** : Courbes equity, rolling metrics
- **Alertes** : Notifications risk management
- **Vue d'ensemble** : Status global des stratégies

### 📁 Gestion Portfolios
- **Création/édition** : Nouveaux portfolios avec paramètres
- **Analytics** : Performance détaillée par portfolio
- **Comparaison** : Analyse comparative multi-portfolios
- **Historique** : Évolution temporelle des positions

### 🎯 Configuration Stratégies
Support des 4 stratégies principales :
- **Mean Reversion** : Paramètres seuils et régimes
- **DMN LSTM** : Configuration réseau neuronal
- **Funding Arbitrage** : Setup multi-exchanges
- **RL Alpha** : Paramètres agent et environment

### ⚠️ Risk Management
- **VaR/CVaR** : Value at Risk en temps réel
- **Exposure** : Monitoring positions par asset
- **Limites** : Configuration et alertes
- **Stress Testing** : Scénarios de stress

## Déploiement

### Test Local Rapide
```bash
cd qframe/ui && ./deploy-simple.sh test
# → Interface sur http://localhost:8502
```

### Docker Complet
```bash
cd qframe/ui && ./deploy-simple.sh up
# → Interface sur http://localhost:8501
```

### Vérification Status
```bash
cd qframe/ui && ./check-status.sh
```

## Architecture Technique

- **Frontend** : Streamlit avec session state management
- **Backend** : API client avec fallback mode
- **Cache** : TTL intelligent pour performance
- **Visualisations** : Plotly pour graphiques interactifs
- **Docker** : Configuration simplifiée single-service

## Fonctionnalités Avancées

### Auto-refresh
- **Données** : Mise à jour automatique toutes les 30s
- **Graphiques** : Refresh conditionnel sur nouveaux data
- **State** : Persistance session utilisateur

### Responsive Design
- **Mobile** : Adaptation layout pour mobiles
- **Dark Mode** : Thème sombre automatique
- **Performance** : Lazy loading des graphiques

## Voir aussi

- [API REST](api.md) - Backend API
- [CLI](cli.md) - Interface ligne de commande
- [Portfolio Management](../portfolio/management.md) - Logique métier