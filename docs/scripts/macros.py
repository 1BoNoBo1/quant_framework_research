"""
Macros personnalisées pour la documentation MkDocs.

Ces macros permettent d'injecter du contenu dynamique dans la documentation,
comme les informations du projet, les métriques en temps réel, etc.
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path


def define_env(env):
    """
    Définit les macros disponibles dans la documentation.

    Args:
        env: Environnement MkDocs macros
    """

    @env.macro
    def qframe_version():
        """Retourne la version actuelle du framework."""
        try:
            # Lire depuis pyproject.toml
            import toml
            pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
            data = toml.load(pyproject_path)
            return data["tool"]["poetry"]["version"]
        except Exception:
            return "0.1.0"

    @env.macro
    def git_info():
        """Retourne les informations Git du repository."""
        try:
            # Commit hash
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent.parent.parent,
                text=True
            ).strip()

            # Dernière date de commit
            commit_date = subprocess.check_output(
                ["git", "log", "-1", "--format=%ci"],
                cwd=Path(__file__).parent.parent.parent,
                text=True
            ).strip()

            # Auteur du dernier commit
            commit_author = subprocess.check_output(
                ["git", "log", "-1", "--format=%an"],
                cwd=Path(__file__).parent.parent.parent,
                text=True
            ).strip()

            return {
                "hash": commit_hash,
                "date": commit_date,
                "author": commit_author
            }
        except Exception:
            return {
                "hash": "unknown",
                "date": datetime.now().isoformat(),
                "author": "QFrame Team"
            }

    @env.macro
    def project_stats():
        """Retourne les statistiques du projet."""
        project_root = Path(__file__).parent.parent.parent

        try:
            # Compter les fichiers Python
            python_files = list(project_root.glob("qframe/**/*.py"))
            python_count = len([f for f in python_files if "__pycache__" not in str(f)])

            # Compter les lignes de code
            total_lines = 0
            for py_file in python_files:
                if "__pycache__" in str(py_file):
                    continue
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except Exception:
                    continue

            # Compter les tests
            test_files = list(project_root.glob("tests/**/*.py"))
            test_count = len([f for f in test_files if "__pycache__" not in str(f)])

            # Compter les stratégies
            strategy_files = list(project_root.glob("qframe/strategies/**/*strategy.py"))
            strategy_count = len(strategy_files)

            return {
                "python_files": python_count,
                "lines_of_code": total_lines,
                "test_files": test_count,
                "strategies": strategy_count,
                "last_update": datetime.now().strftime("%d/%m/%Y")
            }

        except Exception as e:
            return {
                "python_files": "N/A",
                "lines_of_code": "N/A",
                "test_files": "N/A",
                "strategies": "N/A",
                "last_update": datetime.now().strftime("%d/%m/%Y"),
                "error": str(e)
            }

    @env.macro
    def strategy_list():
        """Retourne la liste des stratégies disponibles."""
        strategies_dir = Path(__file__).parent.parent.parent / "qframe" / "strategies"

        strategies = []

        # Parcourir les dossiers de stratégies
        for category_dir in ["research", "production", "hybrid"]:
            category_path = strategies_dir / category_dir
            if not category_path.exists():
                continue

            for strategy_file in category_path.glob("*strategy.py"):
                if strategy_file.name.startswith("_"):
                    continue

                strategy_name = strategy_file.stem.replace("_strategy", "").replace("_", " ").title()

                # Essayer de lire la docstring
                description = "Stratégie de trading avancée"
                try:
                    with open(strategy_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extraction simple de la première docstring
                        if '"""' in content:
                            start = content.find('"""') + 3
                            end = content.find('"""', start)
                            if end > start:
                                description = content[start:end].strip().split('\n')[0]
                except Exception:
                    pass

                strategies.append({
                    "name": strategy_name,
                    "file": strategy_file.name,
                    "category": category_dir.title(),
                    "description": description
                })

        return strategies

    @env.macro
    def feature_matrix():
        """Retourne la matrice des fonctionnalités QFrame."""
        return {
            "Core": {
                "Dependency Injection": "✅ Complet",
                "Configuration Type-Safe": "✅ Pydantic",
                "Interfaces Protocol": "✅ Modern Python",
                "Architecture Hexagonale": "✅ Domain/Infrastructure"
            },
            "Strategies": {
                "DMN LSTM": "✅ PyTorch + Attention",
                "Mean Reversion": "✅ Régimes adaptatifs",
                "Funding Arbitrage": "✅ Multi-exchanges",
                "RL Alpha Generation": "✅ PPO Agent"
            },
            "Infrastructure": {
                "Data Providers": "✅ Binance, YFinance, CCXT",
                "Portfolio Management": "✅ Repository pattern",
                "Risk Management": "✅ VaR/CVaR",
                "Observability": "✅ Structured logging"
            },
            "Research Platform": {
                "Data Lake": "✅ Multi-backend",
                "Distributed Computing": "✅ Dask/Ray",
                "MLOps Pipeline": "✅ MLflow",
                "Feature Store": "✅ Centralized"
            },
            "Interface": {
                "Web Dashboard": "✅ Streamlit",
                "REST API": "✅ FastAPI",
                "CLI": "✅ Typer",
                "Docker": "✅ Multi-service"
            }
        }

    @env.macro
    def performance_metrics():
        """Retourne les métriques de performance simulées."""
        # En production, ces métriques viendraient d'une vraie base de données
        return {
            "framework": {
                "uptime": "99.9%",
                "response_time": "< 100ms",
                "memory_usage": "< 500MB",
                "cpu_usage": "< 30%"
            },
            "strategies": {
                "dmn_lstm": {
                    "sharpe_ratio": "1.85",
                    "max_drawdown": "-8.5%",
                    "win_rate": "67%",
                    "information_coefficient": "0.12"
                },
                "mean_reversion": {
                    "sharpe_ratio": "2.1",
                    "max_drawdown": "-5.2%",
                    "win_rate": "72%",
                    "information_coefficient": "0.08"
                }
            }
        }

    @env.macro
    def code_example(strategy="basic", language="python"):
        """Génère des exemples de code contextuels."""
        examples = {
            "basic": '''
from qframe.core.container import get_container
from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy

# Configuration automatique via DI
container = get_container()
strategy = container.resolve(AdaptiveMeanReversionStrategy)

# Génération de signaux
signals = strategy.generate_signals(market_data)
print(f"Generated {len(signals)} signals")
''',
            "advanced": '''
from qframe.core.container import get_container
from qframe.strategies.research.rl_alpha_strategy import RLAlphaStrategy
from qframe.features.symbolic_operators import SymbolicFeatureProcessor

# Setup avancé avec features
container = get_container()
strategy = container.resolve(RLAlphaStrategy)
feature_processor = container.resolve(SymbolicFeatureProcessor)

# Pipeline complet
features = feature_processor.process(market_data)
signals = strategy.generate_signals(market_data, features)

# Métriques de performance
ic_score = strategy.calculate_information_coefficient(signals, returns)
print(f"Information Coefficient: {ic_score:.4f}")
''',
            "research": '''
from qframe.research.integration_layer import create_research_integration
from qframe.research.backtesting import DistributedBacktestEngine

# Recherche distribuée
integration = create_research_integration()
engine = DistributedBacktestEngine(compute_backend="dask")

# Backtesting multi-stratégies
results = await engine.run_distributed_backtest(
    strategies=["adaptive_mean_reversion", "dmn_lstm"],
    datasets=[data1, data2, data3],
    parameter_grids={"lookback": [10, 20, 30]},
    n_splits=5
)

# Analyse des résultats
best_config = results.get_best_configuration()
print(f"Best Sharpe: {best_config.sharpe_ratio:.2f}")
'''
        }

        return examples.get(strategy, examples["basic"])

    @env.macro
    def build_timestamp():
        """Retourne le timestamp de build de la documentation."""
        return datetime.now().strftime("%d %B %Y à %H:%M")

    @env.macro
    def environment_info():
        """Retourne les informations d'environnement."""
        return {
            "python_version": "3.11+",
            "platform": "Linux, macOS, Windows",
            "dependencies": ["Poetry", "Docker"],
            "recommended_memory": "4GB+",
            "recommended_cpu": "4 cores+"
        }