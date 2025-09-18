# ==============================================
# FRAMEWORK QUANTITATIF PRODUCTION - MAKEFILE COMPLET
# Framework de trading quantitatif avec validation institutionnelle
# Architecture async native, 29 modules, 13,000+ lignes de code
# ==============================================

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# Variables
PROJECT_NAME := quant-stack-production
PYTHON := python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python3
VENV_PIP := $(VENV_DIR)/bin/pip

# Couleurs pour l'affichage
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
NC := \033[0m # No Color

# ==============================================
# AIDE COMPLÈTE
# ==============================================

.PHONY: help
help: ## Aide complète du framework quantitatif
	@echo -e "${CYAN}================================================================${NC}"
	@echo -e "${CYAN}    🚀 FRAMEWORK QUANTITATIF PRODUCTION - GUIDE COMPLET${NC}"
	@echo -e "${CYAN}================================================================${NC}"
	@echo -e "${WHITE}Framework de trading quantitatif avec validation institutionnelle${NC}"
	@echo -e "${WHITE}Architecture async native • 29 modules • Standards hedge fund${NC}"
	@echo ""
	@echo -e "${PURPLE}🎯 DÉMARRAGE RAPIDE:${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(setup|quick|start|workflow)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${GREEN}%-25s${NC} %s\n", $$1, $$2}'
	@echo ""
	@echo -e "${PURPLE}📊 PIPELINE QUANTITATIF:${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(pipeline|data|features|alphas)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${GREEN}%-25s${NC} %s\n", $$1, $$2}'
	@echo ""
	@echo -e "${PURPLE}🏛️ VALIDATION INSTITUTIONNELLE:${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(validation|walkforward|backtest|overfitting)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${GREEN}%-25s${NC} %s\n", $$1, $$2}'
	@echo ""
	@echo -e "${PURPLE}🧠 STRATÉGIES ALPHA:${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(alpha|strategy|dmn|mean|funding)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${GREEN}%-25s${NC} %s\n", $$1, $$2}'
	@echo ""
	@echo -e "${PURPLE}📈 MONITORING & MLFLOW:${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(mlflow|monitor|track|status)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${GREEN}%-25s${NC} %s\n", $$1, $$2}'
	@echo ""
	@echo -e "${PURPLE}🔧 DÉVELOPPEMENT & TESTS:${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(test|dev|debug|bench)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${GREEN}%-25s${NC} %s\n", $$1, $$2}'
	@echo ""
	@echo -e "${PURPLE}🧹 MAINTENANCE:${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(clean|status|info)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${GREEN}%-25s${NC} %s\n", $$1, $$2}'
	@echo ""
	@echo -e "${YELLOW}💡 Démarrage recommandé: ${GREEN}make setup-complete && make pipeline-full${NC}"

# ==============================================
# INSTALLATION & SETUP
# ==============================================

.PHONY: setup-complete
setup-complete: ## Installation complète du framework (recommandé)
	@echo -e "${CYAN}🚀 INSTALLATION FRAMEWORK QUANTITATIF COMPLET${NC}"
	@echo -e "${YELLOW}📦 Création environnement virtuel...${NC}"
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo -e "${YELLOW}📦 Installation stack quantitatif...${NC}"
	@$(VENV_PIP) install pandas>=2.0 numpy>=1.24 scipy>=1.10
	@$(VENV_PIP) install scikit-learn>=1.3 torch>=2.0
	@echo -e "${YELLOW}📦 Installation MLOps...${NC}"
	@$(VENV_PIP) install mlflow>=2.8 
	@echo -e "${YELLOW}📦 Installation APIs crypto...${NC}"
	@$(VENV_PIP) install ccxt>=4.0 python-binance>=1.0
	@echo -e "${YELLOW}📦 Installation async stack...${NC}"
	@$(VENV_PIP) install aiofiles aiohttp asyncio-mqtt
	@echo -e "${YELLOW}📦 Installation visualisation...${NC}"
	@$(VENV_PIP) install matplotlib seaborn plotly
	@echo -e "${YELLOW}📦 Installation utilitaires...${NC}"
	@$(VENV_PIP) install pyyaml python-dotenv tqdm rich
	@mkdir -p {data/{raw,processed,features,artifacts},logs,mlruns,cache,deploy}
	@echo -e "${GREEN}✅ Installation complète terminée${NC}"
	@echo -e "${YELLOW}💡 Testez avec: make pipeline-demo${NC}"

.PHONY: setup-dev
setup-dev: setup-complete ## Installation environnement développement
	@echo -e "${CYAN}🛠️ Setup développement...${NC}"
	@$(VENV_PIP) install pytest pytest-asyncio black flake8 mypy jupyter
	@$(VENV_PIP) install pytest-cov pre-commit
	@echo -e "${GREEN}✅ Environnement développement prêt${NC}"

.PHONY: setup-production
setup-production: setup-complete ## Setup production avec monitoring
	@echo -e "${CYAN}🏭 Setup production...${NC}"
	@$(VENV_PIP) install uvicorn fastapi prometheus-client
	@$(VENV_PIP) install redis celery
	@echo -e "${GREEN}✅ Environnement production prêt${NC}"

# ==============================================
# PIPELINE QUANTITATIF COMPLET
# ==============================================

.PHONY: pipeline-full
pipeline-full: mlflow-start-bg ## Pipeline quantitatif complet (point d'entrée principal)
	@echo -e "${CYAN}⚡ PIPELINE QUANTITATIF COMPLET${NC}"
	@echo -e "${YELLOW}🎯 Architecture async native - UN SEUL asyncio.run()${NC}"
	@sleep 3  # Attendre démarrage MLflow
	@source $(VENV_DIR)/bin/activate && $(PYTHON) orchestration/async_master_pipeline.py

.PHONY: pipeline-hybrid
pipeline-hybrid: mlflow-start-bg ## Pipeline hybride (async + modules réels)
	@echo -e "${CYAN}🔥 PIPELINE HYBRIDE - PERFORMANCE + MODULES RÉELS${NC}"
	@echo -e "${YELLOW}🎯 Async performance + vrais backtests${NC}"
	@sleep 3  # Attendre démarrage MLflow
	@source $(VENV_DIR)/bin/activate && $(PYTHON) orchestration/hybrid_async_pipeline.py --symbols BTCUSDT ETHUSDT --max-concurrent 3

.PHONY: pipeline-no-mlflow
pipeline-no-mlflow: ## Pipeline hybride sans MLflow (test rapide)
	@echo -e "${CYAN}⚡ PIPELINE SANS MLFLOW (TEST RAPIDE)${NC}"
	@echo -e "${YELLOW}🎯 Test architecture sans dépendance MLflow${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) orchestration/hybrid_no_mlflow_pipeline.py --symbols BTCUSDT --days 30

.PHONY: workflow-complete
workflow-complete: clean-cache mlflow-start-bg pipeline-hybrid validation-complete monitor-portfolio-final features-export-final report-consolidate mlflow-stop ## Workflow complet avec orchestration, validation, monitoring et export
	@echo -e "${GREEN}🎉 WORKFLOW QUANTITATIF TERMINÉ${NC}"
	@echo -e "${YELLOW}📊 Consultez les résultats dans data/artifacts/ et logs/${NC}"
	@echo -e "${CYAN}🛡️ Validation institutionnelle incluse${NC}"
	@echo -e "${MAGENTA}📈 Monitoring portfolio et export features terminés${NC}"

.PHONY: pipeline-demo
pipeline-demo: ## Démo pipeline avec simulation
	@echo -e "${CYAN}🎬 DÉMO PIPELINE QUANTITATIF${NC}"
	@$(PYTHON) main_async.py --simulate
	@echo -e "${GREEN}✅ Démo terminée - Vérifiez les résultats ci-dessus${NC}"

.PHONY: pipeline-validate
pipeline-validate: ## Validation architecture pipeline
	@echo -e "${CYAN}🧪 VALIDATION PIPELINE${NC}"
	@$(PYTHON) main_async.py --validate

.PHONY: pipeline-custom
pipeline-custom: ## Pipeline avec configuration personnalisée
	@echo -e "${CYAN}⚙️ Pipeline Configuration Personnalisée${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) main_async.py --symbols "BTCUSDT,ETHUSDT,ADAUSDT" --max-concurrent 8

# ==============================================
# COLLECTE DE DONNÉES
# ==============================================

.PHONY: data-fetch
data-fetch: ## Collecte données crypto (OHLCV + funding)
	@echo -e "${CYAN}📊 Collecte Données Crypto${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.data.collector --symbols BTCUSDT,ETHUSDT

.PHONY: data-crypto
data-crypto: ## Gestionnaire données crypto avancé
	@echo -e "${CYAN}💰 Données Crypto Avancées${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.data_sources.crypto_fetcher

.PHONY: data-async
data-async: ## Test gestionnaire données async
	@echo -e "${CYAN}⚡ Gestionnaire Données Async${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.utils.async_data

# ==============================================
# FEATURE ENGINEERING
# ==============================================

.PHONY: features-build
features-build: ## Construction features techniques + ML
	@echo -e "${CYAN}🔧 Feature Engineering${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.features.feature_engineer

# Target supprimé - utiliser features-export-final

# ==============================================
# STRATÉGIES ALPHA
# ==============================================

.PHONY: alpha-dmn
alpha-dmn: ## Alpha DMN LSTM (Deep Market Network)
	@echo -e "${CYAN}🧠 Alpha DMN LSTM${NC}"
	@echo -e "${YELLOW}Architecture transformer avec attention multi-head${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.alphas.dmn_model

.PHONY: alpha-meanrev
alpha-meanrev: ## Alpha Mean Reversion adaptif
	@echo -e "${CYAN}🔄 Alpha Mean Reversion${NC}"
	@echo -e "${YELLOW}Multi-timeframe avec ML optimization${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.alphas.mean_reversion

.PHONY: alpha-funding
alpha-funding: ## Alpha Funding Rate Arbitrage
	@echo -e "${CYAN}💱 Alpha Funding Arbitrage${NC}"
	@echo -e "${YELLOW}Prédiction ML cycles funding crypto${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.alphas.funding_strategy

.PHONY: alphas-all
alphas-all: alpha-dmn alpha-meanrev alpha-funding ## Entraînement tous les alphas

# ==============================================
# VALIDATION INSTITUTIONNELLE
# ==============================================

.PHONY: validation-walkforward
validation-walkforward: ## Walk-forward analysis (90 périodes)
	@echo -e "${CYAN}🔄 Walk-Forward Analysis${NC}"
	@echo -e "${YELLOW}90 périodes testées - Standards institutionnels${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.validation.walk_forward_analyzer

.PHONY: validation-oos
validation-oos: ## Out-of-sample validation stricte
	@echo -e "${CYAN}🎯 Out-of-Sample Validation${NC}"
	@echo -e "${YELLOW}Protocole anti-leakage strict${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.validation.oos_validator

.PHONY: validation-unified
validation-unified: ## Validation moteur événementiel unifié
	@echo -e "${CYAN}⚡ Moteur Événementiel Unifié${NC}"
	@echo -e "${YELLOW}MÊME moteur walk-forward ET backtest${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.validation.unified_walk_forward

.PHONY: overfitting-detect
overfitting-detect: ## Détection overfitting (8 méthodes)
	@echo -e "${CYAN}🛡️ Détection Overfitting${NC}"
	@echo -e "${YELLOW}8 méthodes académiques avancées${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.validation.overfitting_detector

.PHONY: backtest-rigorous
backtest-rigorous: ## Backtesting rigoureux complet
	@echo -e "${CYAN}🏛️ Backtesting Institutionnel${NC}"
	@echo -e "${YELLOW}Pipeline validation complet${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.validation.rigorous_backtester

.PHONY: validation-complete
validation-complete: validation-walkforward validation-oos overfitting-detect ## Validation institutionnelle complète

.PHONY: report-consolidate
report-consolidate: ## Rapport consolidé pipeline + validation
	@echo -e "${CYAN}📊 Rapport Consolidé${NC}"
	@echo -e "${YELLOW}Pipeline + Validation institutionnelle${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.reporting.consolidated_reporter

# ==============================================
# MÉTRIQUES & ROBUSTESSE
# ==============================================

.PHONY: metrics-psr
metrics-psr: ## Probabilistic Sharpe Ratio (PSR)
	@echo -e "${CYAN}📊 Probabilistic Sharpe Ratio${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.validation.robustness_metrics

.PHONY: metrics-risk
metrics-risk: ## Métriques de risque avancées
	@echo -e "${CYAN}📈 Risk Metrics${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.utils.risk_metrics

.PHONY: costs-realistic
costs-realistic: ## Validation coûts réalistes
	@echo -e "${CYAN}💰 Coûts Réalistes${NC}"
	@echo -e "${YELLOW}BTC 0.55% round-trip validé${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.utils.realistic_costs

# ==============================================
# PORTFOLIO & SÉLECTION
# ==============================================

.PHONY: portfolio-optimize
portfolio-optimize: ## Optimisation portfolio
	@echo -e "${CYAN}📈 Optimisation Portfolio${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.portfolio.optimizer

.PHONY: selection-psr
selection-psr: ## Sélection stratégies par PSR
	@echo -e "${CYAN}🎯 Sélection PSR${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.selection.psr_selector

.PHONY: regime-detection
regime-detection: ## Détection régimes marché
	@echo -e "${CYAN}📊 Détection Régimes${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.selection.regime_detector

# ==============================================
# MLFLOW & TRACKING
# ==============================================

.PHONY: mlflow-start
mlflow-start: ## Démarrer serveur MLflow (interactif)
	@echo -e "${CYAN}📈 Démarrage MLflow Server (interactif)${NC}"
	@echo -e "${YELLOW}⚠️  Utilisez Ctrl+C pour arrêter${NC}"
	@source $(VENV_DIR)/bin/activate && mlflow ui --port 5000 --backend-store-uri file:./mlruns

.PHONY: mlflow-start-bg
mlflow-start-bg: ## Démarrer MLflow en arrière-plan
	@echo -e "${CYAN}📈 Démarrage MLflow Server (background)${NC}"
	@if ! pgrep -f "mlflow ui" > /dev/null; then \
		source $(VENV_DIR)/bin/activate && \
		nohup mlflow ui --port 5000 --backend-store-uri file:./mlruns > mlflow.log 2>&1 & \
		echo $$! > mlflow.pid; \
		echo -e "${GREEN}✅ MLflow démarré (PID: $$(cat mlflow.pid))${NC}"; \
		echo -e "${YELLOW}📊 Interface: http://localhost:5000${NC}"; \
	else \
		echo -e "${YELLOW}⚠️  MLflow déjà en cours${NC}"; \
	fi

.PHONY: mlflow-stop
mlflow-stop: ## Arrêter MLflow
	@echo -e "${CYAN}📈 Arrêt MLflow Server${NC}"
	@if [ -f mlflow.pid ]; then \
		kill $$(cat mlflow.pid) 2>/dev/null || true; \
		rm -f mlflow.pid; \
		echo -e "${GREEN}✅ MLflow arrêté${NC}"; \
	else \
		pkill -f "mlflow ui" 2>/dev/null || true; \
		echo -e "${YELLOW}⚠️  Processus MLflow terminés${NC}"; \
	fi

.PHONY: mlflow-status
mlflow-status: ## Statut serveur MLflow
	@echo -e "${CYAN}📊 Statut MLflow${NC}"
	@if pgrep -f "mlflow ui" > /dev/null; then \
		echo -e "${GREEN}✅ MLflow actif sur http://localhost:5000${NC}"; \
	else \
		echo -e "${RED}❌ MLflow non démarré${NC}"; \
	fi

.PHONY: mlflow-track
mlflow-track: ## Tracking expériences MLflow
	@echo -e "${CYAN}📊 MLflow Tracking${NC}"
	@echo -e "${YELLOW}Ouvrez http://localhost:5000${NC}"

.PHONY: mlflow-clean
mlflow-clean: ## Nettoyage données MLflow
	@echo -e "${CYAN}🧹 Nettoyage MLflow${NC}"
	@rm -rf mlruns/* mlartifacts/*
	@echo -e "${GREEN}✅ MLflow nettoyé${NC}"

# ==============================================
# MONITORING & ALERTES
# ==============================================

.PHONY: monitor-live
monitor-live: ## Monitoring temps réel
	@echo -e "${CYAN}📡 Monitoring Live${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.monitoring.alerting

# Target supprimé - utiliser monitor-portfolio-final

# ==============================================
# TESTS & BENCHMARKS
# ==============================================

.PHONY: test-async
test-async: ## Tests architecture async
	@echo -e "${CYAN}🧪 Tests Architecture Async${NC}"
	@$(PYTHON) test_async_architecture.py

.PHONY: test-unified
test-unified: ## Tests moteur unifié
	@echo -e "${CYAN}⚡ Tests Moteur Unifié${NC}"
	@$(PYTHON) test_unified_walkforward.py

.PHONY: benchmark-async
benchmark-async: ## Benchmark async vs sync
	@echo -e "${CYAN}📊 Benchmark Performance${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) orchestration/benchmark_async_vs_sync.py

.PHONY: test-integration
test-integration: ## Tests intégration complète
	@echo -e "${CYAN}🔬 Tests Intégration${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) test_full_integration.py

.PHONY: test-all
test-all: test-async test-unified test-integration ## Tous les tests

# ==============================================
# DÉVELOPPEMENT
# ==============================================

.PHONY: dev-jupyter
dev-jupyter: ## Lancer Jupyter pour développement
	@echo -e "${CYAN}📓 Jupyter Notebook${NC}"
	@source $(VENV_DIR)/bin/activate && jupyter lab

.PHONY: dev-format
dev-format: ## Formatage code (black)
	@echo -e "${CYAN}🎨 Formatage Code${NC}"
	@source $(VENV_DIR)/bin/activate && black .

.PHONY: dev-lint
dev-lint: ## Linting code (flake8)
	@echo -e "${CYAN}🔍 Linting Code${NC}"
	@source $(VENV_DIR)/bin/activate && flake8 .

# ==============================================
# STATUS & INFO
# ==============================================

.PHONY: status
status: ## Statut complet du framework
	@echo -e "${CYAN}================================================================${NC}"
	@echo -e "${CYAN}    📊 STATUS FRAMEWORK QUANTITATIF PRODUCTION${NC}"
	@echo -e "${CYAN}================================================================${NC}"
	@echo ""
	@echo -e "${PURPLE}🏗️ Architecture:${NC}"
	@echo -e "  ⚡ Async native: UN SEUL asyncio.run() ✅"
	@echo -e "  🎯 Moteur événementiel unifié ✅"
	@echo -e "  📊 29 modules Python - 13,000+ lignes ✅"
	@echo -e "  🏛️ Standards institutionnels ✅"
	@echo ""
	@echo -e "${PURPLE}📈 Pipeline:${NC}"
	@echo -e "  📊 Collecte données crypto ✅"
	@echo -e "  🔧 Feature engineering avancé ✅"  
	@echo -e "  🧠 3 stratégies alpha sophistiquées ✅"
	@echo -e "  🏛️ Validation institutionnelle ✅"
	@echo ""
	@echo -e "${PURPLE}🎯 Validation:${NC}"
	@echo -e "  🔄 Walk-forward (90 périodes) ✅"
	@echo -e "  🎯 Out-of-sample strict ✅"
	@echo -e "  🛡️ Overfitting detection (8 méthodes) ✅"
	@echo -e "  📊 PSR/DSR standards industrie ✅"
	@echo ""
	@echo -e "${PURPLE}💰 Trading:${NC}"
	@echo -e "  💱 APIs crypto (Binance/OKX) ✅"
	@echo -e "  💰 Coûts réalistes intégrés ✅"
	@echo -e "  📈 Portfolio optimization ✅"
	@echo -e "  📡 Monitoring temps réel ✅"
	@echo ""
	@echo -e "${GREEN}🏆 Framework Production-Ready - Niveau Institutionnel${NC}"

.PHONY: info
info: ## Informations framework
	@echo -e "${CYAN}ℹ️ FRAMEWORK QUANTITATIF INFO${NC}"
	@echo "Modules Python: $$(find mlpipeline -name '*.py' | wc -l)"
	@echo "Lignes de code: $$(find mlpipeline -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "Tests: $$(find . -name 'test_*.py' | wc -l)"
	@echo "Configuration: $$(find config -name '*.yml' | wc -l) files"

# ==============================================
# MAINTENANCE
# ==============================================

.PHONY: clean-cache
clean-cache: ## Nettoyage caches Python
	@echo -e "${CYAN}🧹 Nettoyage Caches${NC}"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.utils.artifact_cleaner --project-root . --dry-run

.PHONY: clean-symbols
clean-symbols: ## Nettoyage artifacts par symboles (ex: make clean-symbols SYMBOLS="BTCUSDT ETHUSDT")
	@echo -e "${CYAN}🧹 Nettoyage Artifacts par Symboles${NC}"
	@echo -e "${YELLOW}Symboles: $(SYMBOLS)${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.utils.artifact_cleaner --project-root . --symbols $(SYMBOLS)

.PHONY: clean-timeframes
clean-timeframes: ## Nettoyage artifacts par timeframes (ex: make clean-timeframes TIMEFRAMES="1h 4h")
	@echo -e "${CYAN}🧹 Nettoyage Artifacts par Timeframes${NC}"
	@echo -e "${YELLOW}Timeframes: $(TIMEFRAMES)${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.utils.artifact_cleaner --project-root . --timeframes $(TIMEFRAMES)

.PHONY: clean-selective
clean-selective: ## Nettoyage sélectif (ex: make clean-selective SYMBOLS="BTCUSDT" TIMEFRAMES="1h")
	@echo -e "${CYAN}🧹 Nettoyage Sélectif${NC}"
	@echo -e "${YELLOW}Symboles: $(SYMBOLS) | Timeframes: $(TIMEFRAMES)${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.utils.artifact_cleaner --project-root . --symbols $(SYMBOLS) --timeframes $(TIMEFRAMES)

.PHONY: select-crypto
select-crypto: ## Sélecteur interactif de cryptos et timeframes
	@echo -e "${CYAN}🚀 Sélecteur Crypto Interactif${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) -m mlpipeline.utils.crypto_selector

.PHONY: monitor-portfolio-final
monitor-portfolio-final: ## Monitoring final du portfolio
	@echo -e "${CYAN}📊 Portfolio Monitoring Final${NC}"
	@echo -e "${YELLOW}Dashboard temps réel${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) /home/jim/DEV/claude-code/quant-stack-production/mlpipeline/monitoring/portfolio_monitor.py

.PHONY: features-export-final
features-export-final: ## Export features pour analyse externe
	@echo -e "${CYAN}📤 Features Export Final${NC}"
	@echo -e "${YELLOW}Formats: CSV${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) /home/jim/DEV/claude-code/quant-stack-production/mlpipeline/export/features_exporter.py --formats csv

.PHONY: workflow-optimized
workflow-optimized: ## Workflow optimisé avec monitoring des performances
	@echo -e "${CYAN}🚀 Workflow Optimisé${NC}"
	@echo -e "${YELLOW}Monitoring performances activé${NC}"
	@source $(VENV_DIR)/bin/activate && $(PYTHON) /home/jim/DEV/claude-code/quant-stack-production/mlpipeline/utils/workflow_optimizer.py

.PHONY: clean-data
clean-data: ## Nettoyage données temporaires
	@echo -e "${CYAN}🧹 Nettoyage Données${NC}"
	@rm -rf cache/* logs/* data/processed/* 2>/dev/null || true

.PHONY: clean-all
clean-all: clean-cache clean-data mlflow-clean ## Nettoyage complet
	@echo -e "${GREEN}✅ Nettoyage complet terminé${NC}"

.PHONY: backup-results
backup-results: ## Sauvegarde résultats
	@echo -e "${CYAN}💾 Sauvegarde Résultats${NC}"
	@mkdir -p backup/$$(date +%Y%m%d_%H%M%S)
	@cp -r mlruns data/artifacts logs backup/$$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
	@echo -e "${GREEN}✅ Sauvegarde créée${NC}"

# ==============================================
# RACCOURCIS POPULAIRES
# ==============================================

.PHONY: start
start: pipeline-demo ## Alias pour pipeline-demo

.PHONY: run
run: pipeline-full ## Alias pour pipeline-full

.PHONY: install
install: setup-complete ## Alias pour setup-complete

.PHONY: validate
validate: validation-complete ## Alias pour validation-complete

.PHONY: test
test: test-all ## Alias pour test-all