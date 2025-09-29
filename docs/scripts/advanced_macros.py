#!/usr/bin/env python3
"""
Macros avancées pour la documentation MkDocs professionnelle.
Fournit des macros sophistiquées pour enrichir automatiquement la documentation.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

def define_env(env):
    """Définit les macros avancées pour MkDocs."""

    @env.macro
    def coverage_badge():
        """Génère un badge de coverage des tests."""
        try:
            # Essayer de lire le rapport coverage XML
            project_root = Path(env.variables.get('config', {}).get('docs_dir', '.')).parent
            coverage_file = project_root / "coverage.xml"

            if coverage_file.exists():
                # Parser le XML pour extraire le pourcentage
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                coverage_pct = root.get('line-rate', '0')
                coverage_pct = float(coverage_pct) * 100

                color = "brightgreen" if coverage_pct >= 80 else "yellow" if coverage_pct >= 60 else "red"

                return f"![Coverage](https://img.shields.io/badge/coverage-{coverage_pct:.1f}%25-{color})"
            else:
                return "![Coverage](https://img.shields.io/badge/coverage-unknown-lightgrey)"
        except Exception:
            return "![Coverage](https://img.shields.io/badge/coverage-error-red)"

    @env.macro
    def performance_metrics():
        """Affiche les métriques de performance du framework."""
        try:
            project_root = Path(env.variables.get('config', {}).get('docs_dir', '.')).parent
            perf_file = project_root / "docs" / "performance" / "benchmarks.json"

            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    data = json.load(f)

                metrics = data.get('performance', {}).get('benchmarks', {})
                html = "<div class='performance-metrics'>"
                html += "<h4>⚡ Métriques Performance</h4>"
                html += "<table>"
                html += "<tr><th>Métrique</th><th>Valeur</th><th>Statut</th></tr>"

                for name, result in metrics.items():
                    time_ms = round(result.get('time_seconds', 0) * 1000, 2)
                    status = "✅" if result.get('status') == 'success' else "❌"
                    html += f"<tr><td>{name}</td><td>{time_ms}ms</td><td>{status}</td></tr>"

                html += "</table></div>"
                return html
            else:
                return "_Métriques de performance non disponibles_"
        except Exception as e:
            return f"_Erreur chargement métriques: {str(e)}_"

    @env.macro
    def api_endpoints_count():
        """Compte le nombre d'endpoints API disponibles."""
        try:
            project_root = Path(env.variables.get('config', {}).get('docs_dir', '.')).parent
            api_file = project_root / "docs" / "API_DOCUMENTATION.md"

            if api_file.exists():
                with open(api_file, 'r') as f:
                    content = f.read()

                # Compter les endpoints (lignes qui commencent par ####)
                endpoints = content.count('#### GET') + content.count('#### POST') + \
                           content.count('#### PUT') + content.count('#### DELETE')

                return f"**{endpoints} endpoints** disponibles"
            else:
                return "Endpoints: _Non documentés_"
        except Exception:
            return "Endpoints: _Erreur comptage_"

    @env.macro
    def test_statistics():
        """Affiche les statistiques des tests."""
        try:
            project_root = Path(env.variables.get('config', {}).get('docs_dir', '.')).parent

            # Compter les fichiers de test
            test_files = list(project_root.glob("tests/**/*test*.py"))
            test_count = len(test_files)

            # Essayer d'exécuter pytest pour compter les tests
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "--collect-only", "-q"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if "collected" in result.stdout:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "collected" in line and "items" in line:
                            collected = line.split("collected")[1].split("items")[0].strip()
                            return f"**{collected} tests** dans **{test_count} fichiers**"
            except:
                pass

            return f"**{test_count} fichiers de test** trouvés"

        except Exception:
            return "_Statistiques tests non disponibles_"

    @env.macro
    def recent_commits(limit=5):
        """Affiche les commits récents."""
        try:
            project_root = Path(env.variables.get('config', {}).get('docs_dir', '.')).parent

            result = subprocess.run(
                ["git", "log", f"--max-count={limit}", "--pretty=format:%h|%s|%an|%ar"],
                cwd=project_root,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                html = "<div class='recent-commits'>"
                html += "<h4>📝 Commits Récents</h4>"
                html += "<ul>"

                for commit in commits:
                    if '|' in commit:
                        hash_commit, message, author, date = commit.split('|', 3)
                        html += f"<li><code>{hash_commit}</code> {message} <em>par {author}, {date}</em></li>"

                html += "</ul></div>"
                return html
            else:
                return "_Historique Git non disponible_"

        except Exception:
            return "_Erreur lecture commits_"

    @env.macro
    def dependency_status():
        """Affiche le statut des dépendances."""
        try:
            project_root = Path(env.variables.get('config', {}).get('docs_dir', '.')).parent
            pyproject_file = project_root / "pyproject.toml"

            if pyproject_file.exists():
                with open(pyproject_file, 'r') as f:
                    content = f.read()

                # Compter grossièrement les dépendances
                lines = content.split('\n')
                deps_count = 0
                in_deps = False

                for line in lines:
                    if '[tool.poetry.dependencies]' in line:
                        in_deps = True
                        continue
                    elif line.startswith('[') and in_deps:
                        break
                    elif in_deps and '=' in line and not line.startswith('#'):
                        deps_count += 1

                # Essayer de détecter si les deps sont à jour
                try:
                    result = subprocess.run(
                        ["poetry", "show", "--outdated"],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    outdated = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                except:
                    outdated = "?"

                status = "✅" if outdated == 0 else "⚠️" if isinstance(outdated, int) else "❓"
                return f"{status} **{deps_count} dépendances** ({outdated} obsolètes)"

            return "_Dépendances: Non analysées_"

        except Exception as e:
            return f"_Erreur analyse dépendances: {str(e)}_"

    @env.macro
    def framework_health():
        """Affiche un indicateur de santé global du framework."""
        try:
            metrics = []

            # Test import rapide
            try:
                import sys
                old_path = sys.path[:]
                project_root = Path(env.variables.get('config', {}).get('docs_dir', '.')).parent
                sys.path.insert(0, str(project_root))

                from qframe.core.container import get_container
                from qframe.core.config import get_config

                metrics.append(("Import Core", "✅"))

                # Test container
                container = get_container()
                metrics.append(("Container DI", "✅"))

                # Test config
                config = get_config()
                metrics.append(("Configuration", "✅"))

                sys.path[:] = old_path

            except Exception as e:
                metrics.append(("Framework Core", f"❌ {str(e)[:50]}"))

            # Affichage
            html = "<div class='framework-health'>"
            html += "<h4>🏥 Santé du Framework</h4>"
            html += "<ul>"
            for name, status in metrics:
                html += f"<li><strong>{name}</strong>: {status}</li>"
            html += "</ul></div>"

            return html

        except Exception as e:
            return f"<div class='framework-health'>❌ Erreur: {str(e)}</div>"

    @env.macro
    def documentation_stats():
        """Affiche les statistiques de la documentation."""
        try:
            project_root = Path(env.variables.get('config', {}).get('docs_dir', '.')).parent
            docs_dir = project_root / "docs"

            # Compter les fichiers markdown
            md_files = list(docs_dir.glob("**/*.md"))
            total_pages = len(md_files)

            # Compter les mots approximativement
            total_words = 0
            for md_file in md_files:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Approximation simple du nombre de mots
                        words = len(content.split())
                        total_words += words
                except:
                    pass

            # Estimer temps de lecture (250 mots/minute)
            reading_time = total_words // 250

            return f"📚 **{total_pages} pages** • **{total_words:,} mots** • **~{reading_time} min lecture**"

        except Exception:
            return "_Statistiques documentation non disponibles_"

    @env.macro
    def mkdocs_build_info():
        """Informations sur le build MkDocs."""
        build_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Version MkDocs
            import mkdocs
            mkdocs_version = mkdocs.__version__
        except:
            mkdocs_version = "unknown"

        return f"🔧 **Build** {build_time} • **MkDocs** v{mkdocs_version}"