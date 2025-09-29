#!/usr/bin/env python3
"""
Système de diagnostic intelligent pour QFrame.
Détecte automatiquement les problèmes et propose des solutions.
"""

import sys
import subprocess
import importlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import inspect

@dataclass
class DiagnosticResult:
    """Résultat d'un diagnostic."""
    category: str
    status: str  # "ok", "warning", "error"
    message: str
    solution: Optional[str] = None
    docs_link: Optional[str] = None
    code_example: Optional[str] = None

class QFrameDiagnostics:
    """Système de diagnostic intelligent pour QFrame."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.results: List[DiagnosticResult] = []

    def run_full_diagnostic(self) -> List[DiagnosticResult]:
        """Exécute un diagnostic complet du framework."""
        self.results.clear()

        # Diagnostics par catégorie
        self._check_installation()
        self._check_dependencies()
        self._check_configuration()
        self._check_di_container()
        self._check_strategies()
        self._check_data_providers()
        self._check_common_issues()

        return self.results

    def _add_result(self, category: str, status: str, message: str,
                   solution: str = None, docs_link: str = None, code_example: str = None):
        """Ajoute un résultat de diagnostic."""
        self.results.append(DiagnosticResult(
            category=category,
            status=status,
            message=message,
            solution=solution,
            docs_link=docs_link,
            code_example=code_example
        ))

    def _check_installation(self):
        """Vérifie l'installation du framework."""
        try:
            import qframe
            self._add_result(
                "Installation", "ok",
                "✅ QFrame correctement installé",
                docs_link="/getting-started/installation/"
            )

            # Vérifier les imports critiques
            critical_modules = [
                "qframe.core.container",
                "qframe.core.config",
                "qframe.core.interfaces"
            ]

            for module in critical_modules:
                try:
                    importlib.import_module(module)
                except ImportError as e:
                    self._add_result(
                        "Installation", "error",
                        f"❌ Module critique manquant: {module}",
                        f"Réinstallez QFrame: poetry install",
                        "/getting-started/installation/",
                        "poetry install --no-dev"
                    )

        except ImportError:
            self._add_result(
                "Installation", "error",
                "❌ QFrame non installé ou non accessible",
                "Installez QFrame avec Poetry",
                "/getting-started/installation/",
                "poetry install"
            )

    def _check_dependencies(self):
        """Vérifie les dépendances critiques."""
        critical_deps = {
            "pandas": "Traitement des données",
            "numpy": "Calculs numériques",
            "pydantic": "Validation de configuration",
            "fastapi": "API REST",
            "torch": "Deep Learning (optionnel)"
        }

        for dep, description in critical_deps.items():
            try:
                importlib.import_module(dep)
                self._add_result(
                    "Dépendances", "ok",
                    f"✅ {dep} disponible ({description})"
                )
            except ImportError:
                status = "warning" if dep == "torch" else "error"
                self._add_result(
                    "Dépendances", status,
                    f"⚠️ {dep} manquant ({description})",
                    f"Installez avec: pip install {dep}",
                    code_example=f"poetry add {dep}"
                )

    def _check_configuration(self):
        """Vérifie la configuration du framework."""
        try:
            from qframe.core.config import get_config
            config = get_config()

            self._add_result(
                "Configuration", "ok",
                "✅ Configuration chargée correctement",
                docs_link="/architecture/configuration/"
            )

            # Vérifier les configs critiques
            if hasattr(config, 'database') and config.database.url == "sqlite:///memory":
                self._add_result(
                    "Configuration", "warning",
                    "⚠️ Base de données en mémoire (development only)",
                    "Configurez une base persistante pour production",
                    "/architecture/configuration/",
                    '''
# Dans votre .env
QFRAME_DATABASE__URL=postgresql://user:pass@localhost/qframe
'''
                )

        except Exception as e:
            self._add_result(
                "Configuration", "error",
                f"❌ Erreur de configuration: {str(e)}",
                "Vérifiez votre fichier de configuration",
                "/architecture/configuration/"
            )

    def _check_di_container(self):
        """Vérifie le container DI."""
        try:
            from qframe.core.container import get_container
            container = get_container()

            self._add_result(
                "DI Container", "ok",
                "✅ Container DI opérationnel",
                docs_link="/architecture/di-container/"
            )

            # Test de résolution de services critiques
            critical_services = [
                "qframe.infrastructure.data.binance_provider.BinanceDataProvider",
                "qframe.domain.services.portfolio_service.PortfolioService"
            ]

            for service_path in critical_services:
                try:
                    module_path, class_name = service_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    service_class = getattr(module, class_name)

                    # Tenter la résolution
                    container.resolve(service_class)
                    self._add_result(
                        "DI Container", "ok",
                        f"✅ Service {class_name} résolu"
                    )
                except Exception as e:
                    self._add_result(
                        "DI Container", "warning",
                        f"⚠️ Service {class_name} non résolvable: {str(e)}",
                        f"Vérifiez l'enregistrement du service",
                        "/architecture/di-container/",
                        f"container.register_singleton({class_name}, {class_name})"
                    )

        except Exception as e:
            self._add_result(
                "DI Container", "error",
                f"❌ Container DI non fonctionnel: {str(e)}",
                "Vérifiez l'initialisation du container",
                "/architecture/di-container/"
            )

    def _check_strategies(self):
        """Vérifie les stratégies disponibles."""
        try:
            strategies_path = self.project_root / "qframe" / "strategies"

            if not strategies_path.exists():
                self._add_result(
                    "Stratégies", "error",
                    "❌ Dossier des stratégies non trouvé"
                )
                return

            strategy_files = list(strategies_path.rglob("*strategy*.py"))

            if not strategy_files:
                self._add_result(
                    "Stratégies", "warning",
                    "⚠️ Aucune stratégie trouvée"
                )
                return

            self._add_result(
                "Stratégies", "ok",
                f"✅ {len(strategy_files)} stratégies disponibles",
                docs_link="/strategies/"
            )

            # Test d'import des stratégies principales
            main_strategies = [
                "qframe.strategies.research.adaptive_mean_reversion_strategy.AdaptiveMeanReversionStrategy",
                "qframe.strategies.research.dmn_lstm_strategy.DMNLSTMStrategy"
            ]

            for strategy_path in main_strategies:
                try:
                    module_path, class_name = strategy_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    strategy_class = getattr(module, class_name)

                    self._add_result(
                        "Stratégies", "ok",
                        f"✅ {class_name} importable"
                    )
                except Exception as e:
                    self._add_result(
                        "Stratégies", "warning",
                        f"⚠️ {class_name} non importable: {str(e)}",
                        f"Vérifiez les dépendances de la stratégie"
                    )

        except Exception as e:
            self._add_result(
                "Stratégies", "error",
                f"❌ Erreur lors de la vérification des stratégies: {str(e)}"
            )

    def _check_data_providers(self):
        """Vérifie les fournisseurs de données."""
        providers = [
            ("qframe.infrastructure.data.binance_provider.BinanceDataProvider", "Binance"),
            ("qframe.data.providers.yfinance_provider.YFinanceDataProvider", "Yahoo Finance")
        ]

        for provider_path, name in providers:
            try:
                module_path, class_name = provider_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                provider_class = getattr(module, class_name)

                self._add_result(
                    "Data Providers", "ok",
                    f"✅ Provider {name} disponible"
                )
            except ImportError:
                self._add_result(
                    "Data Providers", "warning",
                    f"⚠️ Provider {name} non disponible",
                    f"Vérifiez les dépendances du provider {name}"
                )

    def _check_common_issues(self):
        """Vérifie les problèmes courants."""
        # Vérifier les variables d'environnement importantes
        import os

        env_vars = [
            ("QFRAME_SECRET_KEY", "Clé secrète pour production"),
            ("QFRAME_DATABASE__URL", "URL de base de données"),
            ("QFRAME_BINANCE__API_KEY", "Clé API Binance (optionnel)")
        ]

        for var, description in env_vars:
            if os.getenv(var):
                self._add_result(
                    "Configuration", "ok",
                    f"✅ {var} configuré ({description})"
                )
            else:
                status = "warning" if "BINANCE" in var else "info"
                self._add_result(
                    "Configuration", status,
                    f"ℹ️ {var} non configuré ({description})",
                    f"Ajoutez {var} à votre .env si nécessaire",
                    code_example=f"{var}=your_value_here"
                )

        # Vérifier les permissions de fichiers
        important_files = [
            self.project_root / "pyproject.toml",
            self.project_root / "qframe" / "core" / "config.py"
        ]

        for file_path in important_files:
            if file_path.exists() and file_path.is_file():
                self._add_result(
                    "Fichiers", "ok",
                    f"✅ {file_path.name} accessible"
                )
            else:
                self._add_result(
                    "Fichiers", "error",
                    f"❌ {file_path.name} non trouvé",
                    "Vérifiez l'intégrité de votre installation"
                )

    def generate_diagnostic_report_html(self) -> str:
        """Génère un rapport HTML du diagnostic."""
        if not self.results:
            self.run_full_diagnostic()

        # Compter par status
        status_counts = {}
        for result in self.results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        html = '<div class="diagnostic-report">'
        html += '<h3>🔍 Rapport de Diagnostic QFrame</h3>'

        # Résumé
        html += '<div class="diagnostic-summary">'
        html += f'<span class="status-ok">✅ {status_counts.get("ok", 0)} OK</span> '
        html += f'<span class="status-warning">⚠️ {status_counts.get("warning", 0)} Warnings</span> '
        html += f'<span class="status-error">❌ {status_counts.get("error", 0)} Errors</span>'
        html += '</div>'

        # Résultats par catégorie
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        for category, results in categories.items():
            html += f'<div class="diagnostic-category">'
            html += f'<h4>{category}</h4>'
            html += '<ul>'

            for result in results:
                status_class = f"status-{result.status}"
                html += f'<li class="{status_class}">'
                html += f'<strong>{result.message}</strong>'

                if result.solution:
                    html += f'<br><em>💡 Solution: {result.solution}</em>'

                if result.docs_link:
                    html += f'<br><a href="{result.docs_link}">📖 Documentation</a>'

                if result.code_example:
                    html += f'<details><summary>Code</summary><pre><code>{result.code_example}</code></pre></details>'

                html += '</li>'

            html += '</ul></div>'

        html += '</div>'
        return html

def define_env(env):
    """Intégration avec MkDocs."""
    diagnostics = QFrameDiagnostics()

    @env.macro
    def diagnostic_report():
        """Macro pour afficher le rapport de diagnostic."""
        return diagnostics.generate_diagnostic_report_html()

    @env.macro
    def quick_health_check():
        """Check de santé rapide."""
        try:
            from qframe.core.container import get_container
            from qframe.core.config import get_config

            container = get_container()
            config = get_config()

            return '''
            <div class="health-status healthy">
                <h4>🚀 QFrame Status: Opérationnel</h4>
                <ul>
                    <li>✅ Framework chargé</li>
                    <li>✅ Configuration valide</li>
                    <li>✅ Container DI actif</li>
                </ul>
                <button onclick="window.open('/diagnostic/', '_blank')">🔍 Diagnostic Complet</button>
            </div>
            '''
        except Exception as e:
            return f'''
            <div class="health-status error">
                <h4>⚠️ QFrame Status: Problème Détecté</h4>
                <p><strong>Erreur:</strong> {str(e)}</p>
                <button onclick="window.open('/diagnostic/', '_blank')">🔍 Diagnostic Complet</button>
            </div>
            '''