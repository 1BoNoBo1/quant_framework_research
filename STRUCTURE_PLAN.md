# 🎯 QUANT STACK MINIMAL - STRUCTURE ESSENTIELLE

## 📁 Arborescence Épurée (SEULEMENT l'essentiel)

```
quant-stack-minimal/
├── 📋 Makefile                    # Interface unique (obligatoire)
├── 📋 requirements.txt            # Dépendances strictes
├── 📋 .env.example               # Configuration
├── 📋 README.md                  # Guide utilisateur simple
├── 📋 CLAUDE.md                  # Règles de sécurité
│
├── 🧠 mlpipeline/                # CORE FRAMEWORK
│   ├── data_sources/             # Collecte données réelles
│   ├── features/                 # Feature engineering
│   ├── alphas/                   # 3 stratégies principales
│   ├── portfolio/                # Optimisation Kelly-Markowitz
│   ├── backtesting/              # Tests rigoureux
│   ├── monitoring/               # Alertes temps réel
│   └── utils/                    # Validation anti-fake
│
├── 📊 data/                      # Données de marché
│   ├── raw/                      # OHLCV brut
│   └── processed/                # Features calculées
│
├── 🧪 tests/                     # Tests essentiels
│   ├── test_integration.py       # Test complet
│   └── test_real_data.py         # Validation données
│
└── 📈 scripts/                   # Scripts utilitaires
    ├── night_trader_real.py      # Trading nocturne
    └── validate_framework.py     # Validation complète
```

## ❌ SUPPRIMÉ (Code mort/obsolète)

- ❌ `deploy_package/` - Duplication inutile
- ❌ `freqtrade-*` - Framework externe non utilisé
- ❌ `exports/` multiples - Encombrement
- ❌ `infrastructure/` - Complexité inutile
- ❌ `mlops/` - Non essentiel pour le trading
- ❌ Tous les `.md` de rapports - Documentation excessive
- ❌ Tests symboliques redondants
- ❌ Scripts de démo multiples

## ✅ CONSERVÉ (Code essentiel)

- ✅ Pipeline de données Binance
- ✅ 3 alphas fonctionnels (DMN, Mean Rev, Funding)
- ✅ Optimisateur portfolio Kelly-Markowitz
- ✅ Backtesting avec walk-forward
- ✅ Validation anti-données-simulées
- ✅ Monitoring temps réel
- ✅ Interface Makefile unifiée

## 🎯 RÉSULTAT ATTENDU

- **📉 Complexité** : 16,963 → ~200 fichiers Python
- **📁 Dossiers** : 70+ → 15 dossiers essentiels
- **🧹 Clarté** : Structure évidente pour tout dev
- **⚡ Performance** : Tests rapides et fiables
- **🛡️ Sécurité** : Aucune donnée simulée possible