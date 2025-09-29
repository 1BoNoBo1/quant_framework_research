#!/usr/bin/env python3
"""
Générateur automatique de documentation benchmarks pour QFrame.
Génère des rapports de performance et des métriques depuis les tests.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import subprocess
import statistics

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_performance_tests() -> Dict[str, Any]:
    """Exécute les tests de performance et collecte les métriques."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "framework_version": "0.1.0",
        "python_version": sys.version,
        "benchmarks": {}
    }

    # Test d'import du framework
    start_time = time.time()
    try:
        from qframe.core.container import get_container
        from qframe.core.config import get_config
        import_time = time.time() - start_time
        results["benchmarks"]["framework_import"] = {
            "time_seconds": import_time,
            "status": "success",
            "description": "Temps d'import du framework core"
        }
    except Exception as e:
        results["benchmarks"]["framework_import"] = {
            "time_seconds": time.time() - start_time,
            "status": "error",
            "error": str(e),
            "description": "Temps d'import du framework core"
        }

    # Test de création du container DI
    start_time = time.time()
    try:
        container = get_container()
        container_time = time.time() - start_time
        results["benchmarks"]["di_container_creation"] = {
            "time_seconds": container_time,
            "status": "success",
            "description": "Temps de création du container DI"
        }
    except Exception as e:
        results["benchmarks"]["di_container_creation"] = {
            "time_seconds": time.time() - start_time,
            "status": "error",
            "error": str(e),
            "description": "Temps de création du container DI"
        }

    # Test de chargement configuration
    start_time = time.time()
    try:
        config = get_config()
        config_time = time.time() - start_time
        results["benchmarks"]["config_loading"] = {
            "time_seconds": config_time,
            "status": "success",
            "description": "Temps de chargement configuration"
        }
    except Exception as e:
        results["benchmarks"]["config_loading"] = {
            "time_seconds": time.time() - start_time,
            "status": "error",
            "error": str(e),
            "description": "Temps de chargement configuration"
        }

    # Test des stratégies (si disponibles)
    try:
        from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy

        start_time = time.time()
        strategy = container.resolve(AdaptiveMeanReversionStrategy)
        strategy_time = time.time() - start_time

        results["benchmarks"]["strategy_resolution"] = {
            "time_seconds": strategy_time,
            "status": "success",
            "description": "Temps de résolution stratégie Mean Reversion"
        }
    except Exception as e:
        results["benchmarks"]["strategy_resolution"] = {
            "time_seconds": 0,
            "status": "error",
            "error": str(e),
            "description": "Temps de résolution stratégie Mean Reversion"
        }

    return results

def run_memory_benchmarks() -> Dict[str, Any]:
    """Mesure l'utilisation mémoire du framework."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    memory_results = {
        "initial_memory_mb": initial_memory / (1024 * 1024),
        "benchmarks": {}
    }

    # Mesurer mémoire après import
    try:
        from qframe.core.container import get_container
        from qframe.core.config import get_config

        after_import_memory = process.memory_info().rss
        memory_results["benchmarks"]["after_imports"] = {
            "memory_mb": after_import_memory / (1024 * 1024),
            "delta_mb": (after_import_memory - initial_memory) / (1024 * 1024),
            "description": "Mémoire après imports core"
        }

        # Mesurer mémoire après création container
        container = get_container()
        after_container_memory = process.memory_info().rss
        memory_results["benchmarks"]["after_container"] = {
            "memory_mb": after_container_memory / (1024 * 1024),
            "delta_mb": (after_container_memory - after_import_memory) / (1024 * 1024),
            "description": "Mémoire après création container"
        }

    except Exception as e:
        memory_results["error"] = str(e)

    return memory_results

def generate_benchmark_report(perf_results: Dict, memory_results: Dict) -> str:
    """Génère le rapport de benchmarks en Markdown."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# 📊 Benchmarks Performance QFrame

**Dernière mise à jour** : {timestamp}

## Métriques de Performance

### Temps d'Exécution

| Benchmark | Temps (ms) | Statut | Description |
|-----------|------------|--------|-------------|
"""

    for name, result in perf_results.get("benchmarks", {}).items():
        time_ms = round(result.get("time_seconds", 0) * 1000, 2)
        status = "✅" if result.get("status") == "success" else "❌"
        description = result.get("description", "")
        content += f"| {name} | {time_ms} | {status} | {description} |\n"

    content += "\n### Métriques Mémoire\n\n"

    if "error" not in memory_results:
        content += f"**Mémoire initiale** : {memory_results['initial_memory_mb']:.2f} MB\n\n"
        content += "| Étape | Mémoire (MB) | Delta (MB) | Description |\n"
        content += "|-------|--------------|------------|-------------|\n"

        for name, result in memory_results.get("benchmarks", {}).items():
            memory_mb = round(result.get("memory_mb", 0), 2)
            delta_mb = round(result.get("delta_mb", 0), 2)
            delta_sign = "+" if delta_mb > 0 else ""
            description = result.get("description", "")
            content += f"| {name} | {memory_mb} | {delta_sign}{delta_mb} | {description} |\n"
    else:
        content += f"❌ Erreur mesure mémoire : {memory_results['error']}\n"

    content += "\n## Informations Système\n\n"
    content += f"- **Framework** : QFrame v{perf_results.get('framework_version', 'unknown')}\n"
    content += f"- **Python** : {sys.version.split()[0]}\n"
    content += f"- **Platform** : {sys.platform}\n"

    # Ajouter recommandations performance
    content += "\n## Recommandations Performance\n\n"

    import_time = perf_results.get("benchmarks", {}).get("framework_import", {}).get("time_seconds", 0)
    if import_time > 1.0:
        content += "⚠️ **Import Framework** : Temps d'import élevé (> 1s). Considérer lazy loading.\n"
    elif import_time < 0.5:
        content += "✅ **Import Framework** : Temps d'import optimal (< 0.5s).\n"

    container_time = perf_results.get("benchmarks", {}).get("di_container_creation", {}).get("time_seconds", 0)
    if container_time > 0.1:
        content += "⚠️ **Container DI** : Création lente (> 100ms). Vérifier enregistrements.\n"
    elif container_time < 0.05:
        content += "✅ **Container DI** : Création rapide (< 50ms).\n"

    total_memory = memory_results.get("benchmarks", {}).get("after_container", {}).get("memory_mb", 0)
    if total_memory > 100:
        content += "⚠️ **Mémoire** : Usage élevé (> 100MB). Optimiser imports.\n"
    elif total_memory < 50:
        content += "✅ **Mémoire** : Usage optimal (< 50MB).\n"

    content += "\n## Graphiques\n\n"
    content += "```mermaid\n"
    content += "graph TD\n"
    content += "    A[Import Framework] --> B[Création Container]\n"
    content += "    B --> C[Chargement Config]\n"
    content += "    C --> D[Résolution Stratégies]\n"
    content += f"    A -.-> A1[{import_time*1000:.1f}ms]\n"
    content += f"    B -.-> B1[{container_time*1000:.1f}ms]\n"
    content += "```\n"

    return content

def generate_coverage_integration() -> str:
    """Génère l'intégration avec les rapports de coverage."""
    content = """# 📈 Coverage Reports

## Test Coverage

La coverage des tests est automatiquement générée et intégrée dans cette documentation.

```bash
# Générer rapport coverage
poetry run pytest --cov=qframe --cov-report=html --cov-report=xml

# Voir rapport
open htmlcov/index.html
```

## Intégration Continue

Les métriques de coverage sont trackées dans le CI/CD :

- **Target** : > 75%
- **Branches** : > 70%
- **Fonctions** : > 80%

{{ coverage_badge() }}

## Coverage par Module

Les détails de coverage par module sont disponibles dans le rapport HTML généré automatiquement.
"""
    return content

def main():
    """Fonction principale."""
    print("🚀 Génération benchmarks QFrame...")

    # Créer répertoire performance
    docs_dir = project_root / "docs"
    perf_dir = docs_dir / "performance"
    perf_dir.mkdir(exist_ok=True)

    # Exécuter benchmarks
    print("📊 Exécution benchmarks performance...")
    perf_results = run_performance_tests()

    print("💾 Mesure utilisation mémoire...")
    memory_results = run_memory_benchmarks()

    # Générer rapport
    print("📝 Génération rapport...")
    report_content = generate_benchmark_report(perf_results, memory_results)

    # Sauvegarder rapport
    with open(perf_dir / "benchmarks.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    # Sauvegarder données JSON pour autres usages
    with open(perf_dir / "benchmarks.json", "w", encoding="utf-8") as f:
        json.dump({
            "performance": perf_results,
            "memory": memory_results
        }, f, indent=2)

    # Générer page coverage
    coverage_content = generate_coverage_integration()
    coverage_dir = docs_dir / "coverage"
    coverage_dir.mkdir(exist_ok=True)

    with open(coverage_dir / "index.md", "w", encoding="utf-8") as f:
        f.write(coverage_content)

    print(f"✅ Benchmarks générés dans {perf_dir}")
    print(f"✅ Coverage setup dans {coverage_dir}")

if __name__ == "__main__":
    main()