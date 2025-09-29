#!/usr/bin/env python3
"""
Visualiseur d'architecture interactive pour QFrame.
Génère des diagrammes dynamiques et navigables.
"""

import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import importlib.util
import inspect

class QFrameArchitectureVisualizer:
    """Visualiseur d'architecture QFrame avec diagrammes interactifs."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.components = {}
        self.dependencies = {}
        self.interfaces = {}
        self._analyze_codebase()

    def _analyze_codebase(self):
        """Analyse la base de code pour extraire l'architecture."""
        qframe_path = self.project_root / "qframe"

        if qframe_path.exists():
            self._extract_components(qframe_path)
            self._extract_dependencies(qframe_path)
            self._extract_interfaces()

    def _extract_components(self, qframe_path: Path):
        """Extrait les composants principaux."""
        self.components = {
            "core": self._analyze_directory(qframe_path / "core"),
            "domain": self._analyze_directory(qframe_path / "domain"),
            "infrastructure": self._analyze_directory(qframe_path / "infrastructure"),
            "strategies": self._analyze_directory(qframe_path / "strategies"),
            "api": self._analyze_directory(qframe_path / "api") if (qframe_path / "api").exists() else {}
        }

    def _analyze_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Analyse un répertoire et extrait ses composants."""
        if not dir_path.exists():
            return {}

        components = {
            "classes": [],
            "functions": [],
            "submodules": {}
        }

        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                file_components = {
                    "classes": [],
                    "functions": [],
                    "imports": []
                }

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        file_components["classes"].append({
                            "name": node.name,
                            "methods": [n.name for n in node.body
                                      if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')],
                            "bases": [self._get_name(base) for base in node.bases],
                            "docstring": ast.get_docstring(node) or ""
                        })
                    elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                        file_components["functions"].append({
                            "name": node.name,
                            "docstring": ast.get_docstring(node) or ""
                        })
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            file_components["imports"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                file_components["imports"].append(f"{node.module}.{alias.name}")

                components["submodules"][py_file.stem] = file_components

            except Exception as e:
                continue

        # Analyser les sous-répertoires
        for subdir in dir_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('_'):
                components["submodules"][subdir.name] = self._analyze_directory(subdir)

        return components

    def _get_name(self, node):
        """Extrait le nom d'un nœud AST."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)

    def _extract_dependencies(self, qframe_path: Path):
        """Extrait les dépendances entre composants."""
        # Analyser les imports pour construire le graphe de dépendances
        for component_name, component_data in self.components.items():
            self.dependencies[component_name] = set()

            if isinstance(component_data, dict) and "submodules" in component_data:
                for module_name, module_data in component_data["submodules"].items():
                    if "imports" in module_data:
                        for import_name in module_data["imports"]:
                            if "qframe" in import_name:
                                # Extraire le composant principal de l'import
                                parts = import_name.split(".")
                                if len(parts) >= 2 and parts[0] == "qframe":
                                    dep_component = parts[1]
                                    if dep_component != component_name:
                                        self.dependencies[component_name].add(dep_component)

    def _extract_interfaces(self):
        """Extrait les interfaces/protocols."""
        interfaces_file = self.project_root / "qframe" / "core" / "interfaces.py"

        if interfaces_file.exists():
            try:
                with open(interfaces_file, 'r') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Cherche les classes Protocol
                        is_protocol = any(
                            isinstance(base, ast.Name) and base.id == "Protocol"
                            for base in node.bases
                        )

                        if is_protocol:
                            methods = []
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef):
                                    methods.append({
                                        "name": item.name,
                                        "args": [arg.arg for arg in item.args.args[1:]],  # Skip self
                                        "docstring": ast.get_docstring(item) or ""
                                    })

                            self.interfaces[node.name] = {
                                "methods": methods,
                                "docstring": ast.get_docstring(node) or ""
                            }

            except Exception as e:
                pass

    def generate_architecture_diagram_mermaid(self) -> str:
        """Génère un diagramme d'architecture Mermaid."""
        mermaid = "```mermaid\nflowchart TD\n"

        # Nœuds principaux
        colors = {
            "core": "#FF6B6B",
            "domain": "#4ECDC4",
            "infrastructure": "#45B7D1",
            "strategies": "#96CEB4",
            "api": "#FECA57"
        }

        for component, color in colors.items():
            if component in self.components:
                mermaid += f'    {component}["{component.title()}"]:::component{component}\n'

        # Dépendances
        for component, deps in self.dependencies.items():
            for dep in deps:
                if dep in colors:
                    mermaid += f'    {component} --> {dep}\n'

        # Styles
        for component, color in colors.items():
            mermaid += f'    classDef component{component} fill:{color},stroke:#333,stroke-width:2px,color:#fff\n'

        mermaid += "```"
        return mermaid

    def generate_detailed_component_view(self, component_name: str) -> str:
        """Génère une vue détaillée d'un composant."""
        if component_name not in self.components:
            return f"Composant '{component_name}' non trouvé"

        component = self.components[component_name]
        html = f'<div class="component-detail" id="{component_name}-detail">'
        html += f'<h3>🏗️ Composant: {component_name.title()}</h3>'

        if isinstance(component, dict) and "submodules" in component:
            for module_name, module_data in component["submodules"].items():
                if isinstance(module_data, dict):
                    html += f'<div class="module-section">'
                    html += f'<h4>📦 Module: {module_name}</h4>'

                    # Classes
                    if "classes" in module_data and module_data["classes"]:
                        html += '<div class="classes-section">'
                        html += '<h5>🏛️ Classes</h5>'
                        html += '<ul>'

                        for class_info in module_data["classes"]:
                            html += f'<li>'
                            html += f'<strong>{class_info["name"]}</strong>'

                            if class_info.get("bases"):
                                html += f' <em>extends {", ".join(class_info["bases"])}</em>'

                            if class_info.get("docstring"):
                                html += f'<br><small>{class_info["docstring"][:100]}...</small>'

                            if class_info.get("methods"):
                                html += '<details><summary>Méthodes</summary><ul>'
                                for method in class_info["methods"]:
                                    html += f'<li><code>{method}()</code></li>'
                                html += '</ul></details>'

                            html += '</li>'

                        html += '</ul></div>'

                    # Functions
                    if "functions" in module_data and module_data["functions"]:
                        html += '<div class="functions-section">'
                        html += '<h5>⚡ Fonctions</h5>'
                        html += '<ul>'

                        for func_info in module_data["functions"]:
                            html += f'<li>'
                            html += f'<code>{func_info["name"]}()</code>'
                            if func_info.get("docstring"):
                                html += f'<br><small>{func_info["docstring"][:100]}...</small>'
                            html += '</li>'

                        html += '</ul></div>'

                    html += '</div>'

        # Dépendances
        if component_name in self.dependencies and self.dependencies[component_name]:
            html += '<div class="dependencies-section">'
            html += '<h4>🔗 Dépendances</h4>'
            html += '<ul>'
            for dep in self.dependencies[component_name]:
                html += f'<li><a href="#{dep}-detail">{dep}</a></li>'
            html += '</ul></div>'

        html += '</div>'
        return html

    def generate_interfaces_documentation(self) -> str:
        """Génère la documentation des interfaces."""
        if not self.interfaces:
            return "Aucune interface trouvée"

        html = '<div class="interfaces-documentation">'
        html += '<h3>🔌 Interfaces & Protocols</h3>'

        for interface_name, interface_data in self.interfaces.items():
            html += f'<div class="interface-section">'
            html += f'<h4>{interface_name}</h4>'

            if interface_data.get("docstring"):
                html += f'<p><em>{interface_data["docstring"]}</em></p>'

            if interface_data.get("methods"):
                html += '<div class="methods-section">'
                html += '<h5>Méthodes requises:</h5>'
                html += '<ul>'

                for method in interface_data["methods"]:
                    html += f'<li>'
                    html += f'<code>{method["name"]}('
                    if method.get("args"):
                        html += ", ".join(method["args"])
                    html += ')</code>'

                    if method.get("docstring"):
                        html += f'<br><small>{method["docstring"]}</small>'

                    html += '</li>'

                html += '</ul></div>'

            html += '</div>'

        html += '</div>'
        return html

    def generate_dependency_graph_mermaid(self) -> str:
        """Génère un graphe de dépendances détaillé."""
        mermaid = "```mermaid\ngraph LR\n"

        # Nœuds avec descriptions
        descriptions = {
            "core": "Configuration & DI",
            "domain": "Business Logic",
            "infrastructure": "External Systems",
            "strategies": "Trading Strategies",
            "api": "REST & WebSocket"
        }

        for component in self.components.keys():
            desc = descriptions.get(component, component.title())
            mermaid += f'    {component}["{component.title()}<br/><small>{desc}</small>"]\n'

        # Dépendances avec labels
        for component, deps in self.dependencies.items():
            for dep in deps:
                if dep in self.components:
                    mermaid += f'    {component} -->|uses| {dep}\n'

        mermaid += "```"
        return mermaid

def define_env(env):
    """Intégration MkDocs."""
    visualizer = QFrameArchitectureVisualizer()

    @env.macro
    def architecture_overview():
        """Vue d'ensemble de l'architecture."""
        return visualizer.generate_architecture_diagram_mermaid()

    @env.macro
    def component_detail(component_name: str):
        """Vue détaillée d'un composant."""
        return visualizer.generate_detailed_component_view(component_name)

    @env.macro
    def interfaces_documentation():
        """Documentation des interfaces."""
        return visualizer.generate_interfaces_documentation()

    @env.macro
    def dependency_graph():
        """Graphe de dépendances."""
        return visualizer.generate_dependency_graph_mermaid()

    @env.macro
    def interactive_architecture_explorer():
        """Explorateur d'architecture interactif."""
        components = list(visualizer.components.keys())

        html = '<div class="architecture-explorer">'
        html += '<h4>🧭 Explorateur d\'Architecture Interactive</h4>'
        html += '<div class="explorer-controls">'
        html += '<label for="component-select">Sélectionner un composant:</label>'
        html += '<select id="component-select" onchange="showComponentDetail(this.value)">'
        html += '<option value="">-- Choisir un composant --</option>'

        for component in components:
            html += f'<option value="{component}">{component.title()}</option>'

        html += '</select></div>'

        html += '<div id="component-display">'
        html += '<p>Sélectionnez un composant pour voir ses détails.</p>'
        html += '</div>'

        # JavaScript pour l'interactivité
        html += '''
        <script>
        function showComponentDetail(componentName) {
            if (!componentName) {
                document.getElementById('component-display').innerHTML =
                    '<p>Sélectionnez un composant pour voir ses détails.</p>';
                return;
            }

            // Ici on chargerait dynamiquement les détails du composant
            // Pour l'instant, simulation
            document.getElementById('component-display').innerHTML =
                `<div class="loading">🔄 Chargement des détails de ${componentName}...</div>`;

            // Simulation d'un appel async
            setTimeout(() => {
                document.getElementById('component-display').innerHTML =
                    `<h5>📦 ${componentName.charAt(0).toUpperCase() + componentName.slice(1)}</h5>
                     <p>Détails du composant ${componentName}...</p>
                     <p><a href="/reference/${componentName}/">📖 Documentation complète</a></p>`;
            }, 500);
        }
        </script>
        '''

        html += '</div>'
        return html