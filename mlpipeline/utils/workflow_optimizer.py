#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow Optimizer - Optimise le workflow pour de meilleures performances
Détecte les goulots d'étranglement et propose des améliorations
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import psutil
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkflowOptimizer:
    """Optimiseur de workflow avec monitoring des performances"""

    def __init__(self):
        self.start_time = None
        self.metrics = {
            "workflow_start": None,
            "workflow_end": None,
            "total_duration": 0,
            "phases": {},
            "system_metrics": {},
            "recommendations": []
        }

    def start_monitoring(self):
        """Démarre le monitoring du workflow"""
        self.start_time = time.time()
        self.metrics["workflow_start"] = datetime.now().isoformat()

        # Métriques système initiales
        self.metrics["system_metrics"]["start"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }

        logger.info("🚀 Monitoring workflow démarré")
        return self.start_time

    def measure_phase(self, phase_name: str, command: str = None):
        """Mesure les performances d'une phase"""
        phase_start = time.time()

        logger.info(f"📊 Début phase: {phase_name}")

        if command:
            try:
                # Exécution de la commande avec monitoring
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Monitoring pendant l'exécution
                cpu_samples = []
                memory_samples = []

                while process.poll() is None:
                    cpu_samples.append(psutil.cpu_percent())
                    memory_samples.append(psutil.virtual_memory().percent)
                    time.sleep(1)

                stdout, stderr = process.communicate()
                phase_duration = time.time() - phase_start

                # Stockage des métriques
                self.metrics["phases"][phase_name] = {
                    "duration": phase_duration,
                    "success": process.returncode == 0,
                    "avg_cpu": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
                    "max_cpu": max(cpu_samples) if cpu_samples else 0,
                    "avg_memory": sum(memory_samples) / len(memory_samples) if memory_samples else 0,
                    "max_memory": max(memory_samples) if memory_samples else 0,
                    "return_code": process.returncode,
                    "command": command
                }

                if process.returncode != 0:
                    logger.error(f"❌ Échec phase {phase_name}: {stderr}")
                else:
                    logger.info(f"✅ Phase {phase_name} terminée en {phase_duration:.1f}s")

                return process.returncode == 0

            except Exception as e:
                logger.error(f"❌ Erreur phase {phase_name}: {e}")
                return False

        return True

    def end_monitoring(self):
        """Termine le monitoring et génère le rapport"""
        if self.start_time:
            self.metrics["workflow_end"] = datetime.now().isoformat()
            self.metrics["total_duration"] = time.time() - self.start_time

            # Métriques système finales
            self.metrics["system_metrics"]["end"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }

            # Génération des recommandations
            self._generate_recommendations()

            logger.info(f"⏱️ Workflow terminé en {self.metrics['total_duration']:.1f}s")

    def _generate_recommendations(self):
        """Génère des recommandations d'optimisation"""
        recommendations = []

        # Analyse des phases les plus lentes
        if self.metrics["phases"]:
            phase_durations = {name: data["duration"] for name, data in self.metrics["phases"].items()}
            slowest_phase = max(phase_durations, key=phase_durations.get)
            slowest_duration = phase_durations[slowest_phase]

            if slowest_duration > 60:  # Plus d'une minute
                recommendations.append(f"🐌 Phase la plus lente: {slowest_phase} ({slowest_duration:.1f}s)")
                recommendations.append(f"💡 Considérer la parallélisation de {slowest_phase}")

        # Analyse CPU
        high_cpu_phases = [
            name for name, data in self.metrics["phases"].items()
            if data.get("max_cpu", 0) > 80
        ]
        if high_cpu_phases:
            recommendations.append(f"🔥 Phases intensives CPU: {', '.join(high_cpu_phases)}")
            recommendations.append("💡 Réduire la concurrence ou optimiser les algorithmes")

        # Analyse mémoire
        high_memory_phases = [
            name for name, data in self.metrics["phases"].items()
            if data.get("max_memory", 0) > 85
        ]
        if high_memory_phases:
            recommendations.append(f"🧠 Phases intensives mémoire: {', '.join(high_memory_phases)}")
            recommendations.append("💡 Traitement par batch ou optimisation mémoire")

        # Analyse des échecs
        failed_phases = [
            name for name, data in self.metrics["phases"].items()
            if not data.get("success", True)
        ]
        if failed_phases:
            recommendations.append(f"❌ Phases échouées: {', '.join(failed_phases)}")
            recommendations.append("🔧 Vérifier les dépendances et configurations")

        # Recommandations générales
        total_duration = self.metrics["total_duration"]
        if total_duration > 300:  # Plus de 5 minutes
            recommendations.append(f"⏱️ Workflow long ({total_duration:.1f}s)")
            recommendations.append("🚀 Envisager l'async ou la parallélisation globale")

        self.metrics["recommendations"] = recommendations

    def display_performance_report(self):
        """Affiche le rapport de performance"""
        print("\n" + "="*80)
        print("🚀 WORKFLOW PERFORMANCE REPORT")
        print("="*80)

        print(f"⏱️ Durée totale: {self.metrics['total_duration']:.1f}s")
        print(f"🕐 Début: {self.metrics['workflow_start']}")
        print(f"🏁 Fin: {self.metrics['workflow_end']}")

        # Phases
        if self.metrics["phases"]:
            print(f"\n📊 PERFORMANCES PAR PHASE:")
            print("-" * 60)
            for phase_name, data in self.metrics["phases"].items():
                status = "✅" if data.get("success", True) else "❌"
                duration = data.get("duration", 0)
                cpu = data.get("avg_cpu", 0)
                memory = data.get("avg_memory", 0)
                print(f"{status} {phase_name:20} | {duration:6.1f}s | CPU: {cpu:4.1f}% | MEM: {memory:4.1f}%")

        # Métriques système
        start_sys = self.metrics["system_metrics"].get("start", {})
        end_sys = self.metrics["system_metrics"].get("end", {})

        print(f"\n🖥️ MÉTRIQUES SYSTÈME:")
        print("-" * 60)
        print(f"CPU début/fin:    {start_sys.get('cpu_percent', 0):4.1f}% → {end_sys.get('cpu_percent', 0):4.1f}%")
        print(f"Mémoire début/fin: {start_sys.get('memory_percent', 0):4.1f}% → {end_sys.get('memory_percent', 0):4.1f}%")
        print(f"Mémoire libre:     {end_sys.get('available_memory_gb', 0):4.1f}GB")

        # Recommandations
        if self.metrics["recommendations"]:
            print(f"\n💡 RECOMMANDATIONS D'OPTIMISATION:")
            print("-" * 60)
            for i, rec in enumerate(self.metrics["recommendations"], 1):
                print(f"{i}. {rec}")

        print("\n" + "="*80)

    def save_report(self, output_file: str = None):
        """Sauvegarde le rapport de performance"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"logs/workflow_performance_{timestamp}.json"

        Path(output_file).parent.mkdir(exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Rapport sauvé: {output_file}")
        return output_file

def optimize_workflow():
    """Exécute le workflow optimisé avec monitoring"""
    optimizer = WorkflowOptimizer()

    try:
        # Démarrage monitoring
        optimizer.start_monitoring()

        # Phases du workflow optimisé
        phases = [
            ("clean-cache", "make clean-cache"),
            ("mlflow-start", "make mlflow-start-bg"),
            ("pipeline-hybrid", "make pipeline-hybrid"),
            ("validation", "make validation-complete"),
            ("monitoring", "make monitor-portfolio"),
            ("export", "make features-export"),
            ("report", "make report-consolidate"),
            ("mlflow-stop", "make mlflow-stop")
        ]

        print("🚀 DÉMARRAGE WORKFLOW OPTIMISÉ")
        print("="*50)

        success_count = 0
        for phase_name, command in phases:
            success = optimizer.measure_phase(phase_name, command)
            if success:
                success_count += 1
            else:
                logger.warning(f"⚠️ Phase {phase_name} échouée, continuation...")

        # Fin du monitoring
        optimizer.end_monitoring()

        # Affichage du rapport
        optimizer.display_performance_report()

        # Sauvegarde
        report_file = optimizer.save_report()

        print(f"\n✅ Workflow terminé: {success_count}/{len(phases)} phases réussies")
        print(f"📊 Rapport détaillé: {report_file}")

        return success_count == len(phases)

    except KeyboardInterrupt:
        logger.info("⏹️ Workflow interrompu par l'utilisateur")
        optimizer.end_monitoring()
        return False
    except Exception as e:
        logger.error(f"❌ Erreur workflow: {e}")
        optimizer.end_monitoring()
        return False

if __name__ == "__main__":
    import sys
    success = optimize_workflow()
    sys.exit(0 if success else 1)