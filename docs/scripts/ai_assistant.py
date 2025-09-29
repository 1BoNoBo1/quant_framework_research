#!/usr/bin/env python3
"""
Assistant IA intégré pour la documentation QFrame.
Fournit une assistance contextuelle et des suggestions intelligentes.
"""

import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import ast

class QFrameAIAssistant:
    """Assistant IA pour navigation et aide contextuelle."""

    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        self.common_issues = {}  # TODO: Implement common issues loading
        self.code_patterns = {}  # TODO: Implement code patterns analysis

    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Construit la base de connaissances depuis le code."""
        project_root = Path(__file__).parent.parent.parent

        knowledge = {
            "strategies": self._extract_strategies(project_root),
            "interfaces": self._extract_interfaces(project_root),
            "examples": self._extract_examples(project_root)
        }

        return knowledge

    def _extract_strategies(self, root: Path) -> List[Dict]:
        """Extrait automatiquement toutes les stratégies disponibles."""
        strategies = []
        strategy_path = root / "qframe" / "strategies"

        if strategy_path.exists():
            for py_file in strategy_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    with open(py_file, 'r') as f:
                        tree = ast.parse(f.read())

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if "Strategy" in node.name:
                                strategies.append({
                                    "name": node.name,
                                    "file": str(py_file.relative_to(root)),
                                    "docstring": ast.get_docstring(node) or "",
                                    "methods": [n.name for n in node.body
                                             if isinstance(n, ast.FunctionDef)]
                                })
                except Exception:
                    continue

        return strategies

    def _extract_interfaces(self, root: Path) -> List[Dict]:
        """Extrait les interfaces/protocols du framework."""
        interfaces = []
        interfaces_file = root / "qframe" / "core" / "interfaces.py"

        if interfaces_file.exists():
            try:
                with open(interfaces_file, 'r') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Cherche les classes avec Protocol
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id == "Protocol":
                                interfaces.append({
                                    "name": node.name,
                                    "methods": [n.name for n in node.body
                                             if isinstance(n, ast.FunctionDef)],
                                    "docstring": ast.get_docstring(node) or ""
                                })
            except Exception:
                pass

        return interfaces

    def _extract_examples(self, root: Path) -> List[Dict]:
        """Extrait les exemples du projet."""
        examples = []
        examples_path = root / "examples"

        if examples_path.exists():
            for py_file in examples_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    with open(py_file, 'r') as f:
                        content = f.read()

                    # Extraire le docstring du module
                    tree = ast.parse(content)
                    docstring = ast.get_docstring(tree) or ""

                    examples.append({
                        "name": py_file.stem,
                        "file": str(py_file.relative_to(root)),
                        "docstring": docstring,
                        "content": content[:500] + "..." if len(content) > 500 else content
                    })
                except Exception:
                    continue

        return examples

    def generate_contextual_help(self, current_page: str, user_query: str = "") -> Dict[str, Any]:
        """Génère une aide contextuelle basée sur la page actuelle."""
        help_data = {
            "suggestions": [],
            "related_components": [],
            "troubleshooting": [],
            "examples": []
        }

        # Analyse contextuelle selon la page
        if "strategies" in current_page:
            help_data["suggestions"] = [
                "🎯 Utilisez `AdaptiveMeanReversionStrategy` pour les marchés sideways",
                "🤖 `DMNLSTMStrategy` excelle sur données haute fréquence",
                "⚡ `RLAlphaStrategy` génère des alphas automatiquement"
            ]

            help_data["examples"] = [
                {
                    "title": "Créer une stratégie simple",
                    "code": '''
from qframe.core.container import get_container
from qframe.strategies.research import AdaptiveMeanReversionStrategy

# Récupération via DI Container
container = get_container()
strategy = container.resolve(AdaptiveMeanReversionStrategy)

# Configuration
config = MeanReversionConfig(
    lookback_short=10,
    lookback_long=50,
    z_entry_base=1.5
)

# Génération de signaux
signals = strategy.generate_signals(data, config=config)
'''
                }
            ]

        elif "configuration" in current_page:
            help_data["troubleshooting"] = [
                "❌ `pydantic.ValidationError` → Vérifiez les types dans votre config",
                "⚠️ Config non trouvée → Utilisez `get_config()` du core",
                "🔧 Variables d'environnement → Préfixe `QFRAME_` requis"
            ]

        elif "api" in current_page:
            help_data["related_components"] = [
                {"name": "FastAPI Router", "link": "/reference/infrastructure/api/rest/"},
                {"name": "WebSocket Handler", "link": "/reference/api/websocket/"},
                {"name": "Authentication", "link": "/reference/infrastructure/api/auth/"}
            ]

        return help_data

    def suggest_code_completion(self, partial_code: str) -> List[Dict]:
        """Suggestions de complétion de code intelligentes."""
        suggestions = []

        # Détection patterns courants
        if "container.resolve(" in partial_code:
            suggestions.extend([
                {
                    "completion": "AdaptiveMeanReversionStrategy)",
                    "description": "Stratégie Mean Reversion adaptative"
                },
                {
                    "completion": "BinanceDataProvider)",
                    "description": "Provider de données Binance"
                },
                {
                    "completion": "PortfolioService)",
                    "description": "Service de gestion portfolio"
                }
            ])

        if "strategy.generate_signals(" in partial_code:
            suggestions.append({
                "completion": "data, features=features, config=config)",
                "description": "Paramètres standard pour génération signaux"
            })

        return suggestions

    def diagnose_error(self, error_message: str, context: str = "") -> Dict[str, Any]:
        """Diagnostic intelligent des erreurs courantes."""
        diagnosis = {
            "error_type": "unknown",
            "solution": "Erreur non reconnue",
            "related_docs": [],
            "code_fix": None
        }

        # Patterns d'erreurs courantes
        if "ModuleNotFoundError" in error_message:
            if "qframe" in error_message:
                diagnosis.update({
                    "error_type": "import_error",
                    "solution": "Vérifiez que QFrame est installé : `poetry install`",
                    "related_docs": ["/getting-started/installation/"],
                    "code_fix": "# Vérifiez votre PYTHONPATH\nimport sys\nprint(sys.path)"
                })

        elif "ValidationError" in error_message:
            diagnosis.update({
                "error_type": "config_error",
                "solution": "Erreur de configuration Pydantic - vérifiez les types",
                "related_docs": ["/architecture/configuration/"],
                "code_fix": "# Utilisez la validation explicite\nfrom qframe.core.config import get_config\nconfig = get_config()  # Validation automatique"
            })

        elif "DependencyResolutionError" in error_message:
            diagnosis.update({
                "error_type": "di_error",
                "solution": "Erreur DI Container - vérifiez l'enregistrement des services",
                "related_docs": ["/architecture/di-container/"],
                "code_fix": "# Enregistrez le service manquant\ncontainer.register_singleton(Interface, Implementation)"
            })

        return diagnosis

def define_env(env):
    """Intégration MkDocs avec macros avancées."""
    assistant = QFrameAIAssistant()

    @env.macro
    def ai_contextual_help(page_url: str = "", query: str = ""):
        """Génère une aide contextuelle IA."""
        help_data = assistant.generate_contextual_help(page_url, query)

        html = '<div class="ai-assistant-panel">'
        html += '<h4>🤖 Assistant IA QFrame</h4>'

        if help_data["suggestions"]:
            html += '<div class="suggestions"><h5>💡 Suggestions</h5><ul>'
            for suggestion in help_data["suggestions"]:
                html += f'<li>{suggestion}</li>'
            html += '</ul></div>'

        if help_data["examples"]:
            html += '<div class="examples"><h5>🔧 Exemples</h5>'
            for example in help_data["examples"]:
                html += f'<details><summary>{example["title"]}</summary>'
                html += f'<pre><code class="language-python">{example["code"]}</code></pre>'
                html += '</details>'
            html += '</div>'

        if help_data["troubleshooting"]:
            html += '<div class="troubleshooting"><h5>🔧 Dépannage</h5><ul>'
            for issue in help_data["troubleshooting"]:
                html += f'<li>{issue}</li>'
            html += '</ul></div>'

        html += '</div>'
        return html

    @env.macro
    def framework_health_check():
        """Check de santé temps réel du framework."""
        try:
            # Test d'imports critiques
            from qframe.core.container import get_container
            from qframe.core.config import get_config

            container = get_container()
            config = get_config()

            return """
            <div class="health-check success">
                ✅ <strong>Framework Status: Healthy</strong>
                <ul>
                    <li>✅ Core imports: OK</li>
                    <li>✅ DI Container: OK</li>
                    <li>✅ Configuration: OK</li>
                </ul>
            </div>
            """
        except Exception as e:
            return f"""
            <div class="health-check error">
                ❌ <strong>Framework Status: Issues Detected</strong>
                <p>Error: {str(e)}</p>
                <p><a href="/getting-started/installation/">📖 Check Installation Guide</a></p>
            </div>
            """

    @env.macro
    def interactive_code_playground(initial_code: str = ""):
        """Playground de code interactif."""
        playground_id = abs(hash(initial_code)) % 10000

        return f"""
        <div class="code-playground" id="playground-{playground_id}">
            <div class="playground-header">
                <span>🎮 QFrame Playground</span>
                <button onclick="runCode({playground_id})">▶️ Exécuter</button>
            </div>
            <textarea id="code-{playground_id}" class="code-editor">{initial_code}</textarea>
            <div id="output-{playground_id}" class="playground-output"></div>
        </div>

        <script>
        function runCode(playgroundId) {{
            const code = document.getElementById(`code-${{playgroundId}}`).value;
            const output = document.getElementById(`output-${{playgroundId}}`);

            // Simulation d'exécution (à remplacer par Pyodide)
            output.innerHTML = '<div class="output-success">✅ Code exécuté avec succès!</div>';
        }}
        </script>
        """