#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated Reporter - Rapport unifié pipeline + validation
Consolide tous les résultats (pipeline, validation, overfitting) en un rapport final
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConsolidatedReporter:
    """Génère un rapport consolidé pipeline + validation institutionnelle"""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.report_data = {
            "metadata": {
                "report_type": "consolidated",
                "generation_timestamp": datetime.now().isoformat(),
                "framework_version": "1.0.0"
            },
            "pipeline": {},
            "validation": {},
            "recommendations": [],
            "summary": {}
        }

    def load_latest_pipeline_report(self) -> Optional[Dict[str, Any]]:
        """Charge le dernier rapport de pipeline"""
        try:
            pattern = str(self.output_dir / "hybrid_pipeline_report_*.json")
            files = glob.glob(pattern)
            if not files:
                logger.warning("Aucun rapport pipeline trouvé")
                return None

            latest_file = max(files, key=os.path.getctime)
            logger.info(f"📊 Chargement pipeline: {latest_file}")

            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Erreur chargement pipeline: {e}")
            return None

    def load_validation_results(self) -> Dict[str, Any]:
        """Charge les résultats de validation"""
        validation_data = {
            "oos_validation": None,
            "walk_forward": None,
            "overfitting": None,
            "psr_selection": None
        }

        # OOS Validation
        oos_files = ["test_oos_results.json", "oos_validation_results.json"]
        for oos_file in oos_files:
            if os.path.exists(oos_file):
                try:
                    with open(oos_file, 'r') as f:
                        validation_data["oos_validation"] = json.load(f)
                        logger.info(f"📈 OOS validation chargée: {oos_file}")
                        break
                except Exception as e:
                    logger.warning(f"⚠️ Erreur OOS: {e}")

        # PSR Selection
        psr_file = "data/artifacts/psr_selection_results.json"
        if os.path.exists(psr_file):
            try:
                with open(psr_file, 'r') as f:
                    validation_data["psr_selection"] = json.load(f)
                    logger.info(f"🎯 PSR selection chargée")
            except Exception as e:
                logger.warning(f"⚠️ Erreur PSR: {e}")

        return validation_data

    def analyze_pipeline_health(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse la santé du pipeline"""
        if not pipeline_data:
            return {"status": "MISSING", "issues": ["Pipeline data not found"]}

        summary = pipeline_data.get("summary", {})
        success_rate = summary.get("success_rate", 0)
        execution_time = pipeline_data.get("execution_time", 0)

        issues = []
        if success_rate < 1.0:
            issues.append(f"Taux de succès: {success_rate*100:.1f}% (cible: 100%)")
        if execution_time > 300:  # 5 minutes
            issues.append(f"Temps d'exécution élevé: {execution_time:.1f}s")

        status = "HEALTHY" if not issues else "DEGRADED" if success_rate > 0.8 else "CRITICAL"

        return {
            "status": status,
            "success_rate": success_rate,
            "execution_time": execution_time,
            "total_tasks": summary.get("total_tasks", 0),
            "issues": issues
        }

    def analyze_validation_health(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse la santé de la validation"""
        issues = []
        scores = {}

        # OOS Analysis
        oos = validation_data.get("oos_validation")
        if oos:
            oos_sharpe = oos.get("performance", {}).get("out_of_sample", {}).get("sharpe_ratio", 0)
            scores["oos_sharpe"] = oos_sharpe
            if oos_sharpe <= 0:
                issues.append("Sharpe OOS ≤ 0 (stratégies non viables)")
        else:
            issues.append("Validation OOS manquante")

        # Determine overall validation status
        if not validation_data.get("oos_validation"):
            status = "MISSING"
        elif scores.get("oos_sharpe", 0) > 0.5:
            status = "GOOD"
        elif scores.get("oos_sharpe", 0) > 0:
            status = "WEAK"
        else:
            status = "FAILED"

        return {
            "status": status,
            "scores": scores,
            "issues": issues
        }

    def generate_recommendations(self, pipeline_health: Dict, validation_health: Dict) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []

        # Pipeline recommendations
        if pipeline_health["status"] != "HEALTHY":
            recommendations.append("🔧 Optimiser la stabilité du pipeline")
            if pipeline_health.get("execution_time", 0) > 300:
                recommendations.append("⚡ Améliorer les performances d'exécution")

        # Validation recommendations
        val_status = validation_health["status"]
        if val_status == "FAILED":
            recommendations.extend([
                "📈 Revoir complètement les stratégies alpha (performance OOS nulle)",
                "🎯 Réduire l'overfitting (optimisation excessive)",
                "📊 Augmenter la période d'entraînement"
            ])
        elif val_status == "WEAK":
            recommendations.extend([
                "🔧 Optimiser les hyperparamètres des stratégies",
                "📈 Améliorer la robustesse out-of-sample",
                "🎲 Réduire le multiple testing bias"
            ])

        # Cross-cutting recommendations
        if pipeline_health["success_rate"] == 1.0 and val_status in ["WEAK", "FAILED"]:
            recommendations.append("🏗️ Pipeline stable mais stratégies à améliorer")

        return recommendations

    def generate_executive_summary(self, pipeline_health: Dict, validation_health: Dict) -> Dict[str, Any]:
        """Génère un résumé exécutif"""
        pipeline_status = pipeline_health["status"]
        validation_status = validation_health["status"]

        # Overall status logic
        if pipeline_status == "HEALTHY" and validation_status == "GOOD":
            overall_status = "PRODUCTION_READY"
            overall_message = "✅ Système prêt pour la production"
        elif pipeline_status == "HEALTHY" and validation_status in ["WEAK", "FAILED"]:
            overall_status = "NEEDS_OPTIMIZATION"
            overall_message = "🟡 Infrastructure stable, stratégies à optimiser"
        elif pipeline_status != "HEALTHY":
            overall_status = "INFRASTRUCTURE_ISSUES"
            overall_message = "🔧 Problèmes d'infrastructure à résoudre"
        else:
            overall_status = "CRITICAL"
            overall_message = "❌ Système non viable - révision complète nécessaire"

        return {
            "overall_status": overall_status,
            "message": overall_message,
            "pipeline_health": pipeline_status,
            "validation_health": validation_status,
            "readiness_score": self._calculate_readiness_score(pipeline_health, validation_health)
        }

    def _calculate_readiness_score(self, pipeline_health: Dict, validation_health: Dict) -> float:
        """Calcule un score de maturité globale (0-1)"""
        pipeline_score = pipeline_health.get("success_rate", 0) * 0.4

        val_scores = validation_health.get("scores", {})
        validation_score = max(0, val_scores.get("oos_sharpe", 0)) * 0.6

        return min(1.0, pipeline_score + validation_score)

    def consolidate_report(self) -> str:
        """Génère le rapport consolidé complet"""
        logger.info("🏗️ Génération rapport consolidé...")

        # Chargement des données
        pipeline_data = self.load_latest_pipeline_report()
        validation_data = self.load_validation_results()

        # Analyse
        pipeline_health = self.analyze_pipeline_health(pipeline_data)
        validation_health = self.analyze_validation_health(validation_data)

        # Construction du rapport
        self.report_data["pipeline"] = {
            "data": pipeline_data,
            "health": pipeline_health
        }

        self.report_data["validation"] = {
            "data": validation_data,
            "health": validation_health
        }

        self.report_data["recommendations"] = self.generate_recommendations(
            pipeline_health, validation_health
        )

        self.report_data["summary"] = self.generate_executive_summary(
            pipeline_health, validation_health
        )

        # Sauvegarde
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"consolidated_report_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)

        # Affichage résumé
        self._display_summary()

        logger.info(f"📊 Rapport consolidé sauvé: {output_file}")
        return str(output_file)

    def _display_summary(self):
        """Affiche le résumé du rapport"""
        summary = self.report_data["summary"]
        pipeline_health = self.report_data["pipeline"]["health"]
        validation_health = self.report_data["validation"]["health"]

        print(f"\n{'-'*60}")
        print("📊 RAPPORT CONSOLIDÉ QUANTITATIF")
        print(f"{'-'*60}")

        print(f"\n🎯 STATUT GLOBAL: {summary['message']}")
        print(f"📈 Score de maturité: {summary['readiness_score']:.1%}")

        print(f"\n🏗️ INFRASTRUCTURE PIPELINE:")
        print(f"   Status: {pipeline_health['status']}")
        print(f"   Taux succès: {pipeline_health.get('success_rate', 0):.1%}")
        print(f"   Temps exécution: {pipeline_health.get('execution_time', 0):.1f}s")

        print(f"\n🛡️ VALIDATION INSTITUTIONNELLE:")
        print(f"   Status: {validation_health['status']}")
        scores = validation_health.get('scores', {})
        if 'oos_sharpe' in scores:
            print(f"   Sharpe OOS: {scores['oos_sharpe']:.3f}")

        print(f"\n🔧 RECOMMANDATIONS:")
        for i, rec in enumerate(self.report_data["recommendations"][:5], 1):
            print(f"   {i}. {rec}")

        print(f"\n{'-'*60}")

def main():
    """Point d'entrée principal"""
    try:
        reporter = ConsolidatedReporter()
        report_file = reporter.consolidate_report()

        print(f"\n✅ Rapport consolidé généré: {report_file}")
        return 0

    except Exception as e:
        logger.error(f"❌ Erreur génération rapport: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())