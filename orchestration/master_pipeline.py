#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Pipeline - Orchestrateur Principal
Version Production avec workflow idempotent et gestion d'erreurs robuste
"""

import logging
import os
import sys
import json
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import traceback

import pandas as pd
import mlflow
from mlflow import MlflowClient

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MasterPipeline:
    """
    Orchestrateur principal du pipeline quantitatif
    
    Fonctionnalités :
    - Exécution séquentielle et parallèle des étapes
    - Gestion des dépendances entre modules
    - Recovery automatique en cas d'erreur
    - Validation complète à chaque étape
    - Integration MLflow pour tracking
    """
    
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.execution_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # État du pipeline
        self.step_status = {}
        self.step_results = {}
        self.start_time = None
        self.total_duration = 0
        
        # MLflow client
        self.mlflow_client = MlflowClient()
        
        # Création des dossiers nécessaires
        self._setup_directories()
        
        logger.info(f"✅ Master Pipeline initialisé (ID: {self.execution_id})")
    
    def _load_config(self) -> Dict:
        """Charge la configuration du pipeline"""
        try:
            if os.path.exists(self.config_path):
                import yaml
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Configuration par défaut
                return {
                    "data": {
                        "symbols": ["BTCUSDT", "ETHUSDT"],
                        "timeframes": ["1h"],
                        "days_lookback": 365
                    },
                    "alphas": {
                        "dmn": {"enabled": True, "config": {}},
                        "mean_reversion": {"enabled": True, "config": {}},
                        "funding": {"enabled": True, "config": {}}
                    },
                    "selection": {
                        "psr_threshold": 0.65,
                        "sharpe_threshold": 1.0,
                        "maxdd_threshold": 0.25
                    },
                    "regime_detection": {
                        "n_regimes": 3,
                        "optimize": True
                    },
                    "execution": {
                        "parallel_alphas": True,
                        "cleanup_before": True,
                        "validate_strict": True
                    }
                }
        except Exception as e:
            logger.error(f"❌ Erreur chargement config: {e}")
            return {}
    
    def _setup_directories(self):
        """Création des dossiers nécessaires"""
        directories = [
            "data/raw",
            "data/processed", 
            "data/artifacts",
            "logs",
            "freqtrade-prod/user_data/data/ml_scores"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _run_step(self, step_name: str, command: List[str], 
                  dependencies: List[str] = None,
                  timeout: int = 3600) -> Tuple[bool, Dict]:
        """
        Exécution d'une étape du pipeline avec gestion d'erreurs
        
        Args:
            step_name: Nom de l'étape
            command: Commande à exécuter
            dependencies: Étapes prerequises
            timeout: Timeout en secondes
        """
        
        logger.info(f"🚀 Début étape: {step_name}")
        step_start = time.time()
        
        # Vérification des dépendances
        if dependencies:
            for dep in dependencies:
                if dep not in self.step_status:
                    error_msg = f"Dépendance {dep} non trouvée"
                    logger.error(f"❌ {step_name}: {error_msg}")
                    return False, {"error": error_msg, "duration": 0}
                
                # Gestion des formats de step_status (tuple ou dict)
                dep_result = self.step_status[dep]
                if isinstance(dep_result, tuple):
                    # Format (success, result_dict)
                    success, _ = dep_result
                    if not success:
                        error_msg = f"Dépendance {dep} échouée"
                        logger.error(f"❌ {step_name}: {error_msg}")
                        return False, {"error": error_msg, "duration": 0}
                elif isinstance(dep_result, dict):
                    # Format dict direct
                    if not dep_result.get('success', False):
                        error_msg = f"Dépendance {dep} échouée"
                        logger.error(f"❌ {step_name}: {error_msg}")
                        return False, {"error": error_msg, "duration": 0}
                else:
                    error_msg = f"Format dépendance {dep} invalide"
                    logger.error(f"❌ {step_name}: {error_msg}")
                    return False, {"error": error_msg, "duration": 0}
        
        try:
            # Exécution avec timeout
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            duration = time.time() - step_start
            
            if result.returncode == 0:
                logger.info(f"✅ {step_name} terminé ({duration:.1f}s)")
                
                return True, {
                    "success": True,
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                logger.error(f"❌ {step_name} échoué (code {result.returncode})")
                logger.error(f"STDERR: {result.stderr}")
                
                return False, {
                    "success": False,
                    "duration": duration,
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        
        except subprocess.TimeoutExpired:
            duration = time.time() - step_start
            error_msg = f"Timeout ({timeout}s) dépassé"
            logger.error(f"❌ {step_name}: {error_msg}")
            
            return False, {
                "success": False,
                "duration": duration,
                "error": error_msg
            }
        
        except Exception as e:
            duration = time.time() - step_start
            error_msg = f"Erreur inattendue: {str(e)}"
            logger.error(f"❌ {step_name}: {error_msg}")
            logger.error(traceback.format_exc())
            
            return False, {
                "success": False,
                "duration": duration,
                "error": error_msg,
                "traceback": traceback.format_exc()
            }
    
    async def run_parallel_steps(self, steps: List[Tuple]) -> Dict[str, Tuple]:
        """Exécution parallèle d'étapes indépendantes"""
        
        logger.info(f"🔄 Exécution parallèle de {len(steps)} étapes...")
        
        async def run_step_async(step_info):
            step_name, command, dependencies, timeout = step_info
            return step_name, self._run_step(step_name, command, dependencies, timeout)
        
        # Exécution en parallèle
        tasks = [run_step_async(step_info) for step_info in steps]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    def execute_pipeline(self) -> Dict:
        """
        Exécution complète du pipeline
        """
        
        logger.info("🚀 DÉBUT PIPELINE QUANTITATIF COMPLET")
        self.start_time = time.time()
        
        # MLflow experiment
        mlflow.set_experiment("QuantPipeline_Master")
        
        with mlflow.start_run(run_name=self.execution_id):
            
            try:
                # Log configuration
                mlflow.log_params({
                    "execution_id": self.execution_id,
                    "symbols": self.config["data"]["symbols"],
                    "timeframes": self.config["data"]["timeframes"]
                })
                
                pipeline_success = True
                
                # ÉTAPE 1: Nettoyage des anciens artifacts (si configuré)
                if self.config["execution"]["cleanup_before"]:
                    success, result = self._run_step(
                        "artifact_cleanup",
[".venv/bin/python", "mlpipeline/utils/artifact_cleaner.py", "--project-root", "."]
                    )
                    self.step_status["artifact_cleanup"] = result
                    if not success:
                        logger.warning("⚠️  Nettoyage échoué, continuation...")
                
                # ÉTAPE 2: Récupération des données
                data_steps = []
                for symbol in self.config["data"]["symbols"]:
                    for timeframe in self.config["data"]["timeframes"]:
                        step_name = f"data_fetch_{symbol}_{timeframe}"
                        command = [
".venv/bin/python", "mlpipeline/data_sources/crypto_fetcher.py",
                            "--symbols", symbol,
                            "--timeframes", timeframe,
                            "--days", str(self.config["data"]["days_lookback"])
                        ]
                        data_steps.append((step_name, command, None, 1800))  # 30min timeout
                
                # Exécution parallèle récupération données
                if data_steps:
                    data_results = asyncio.run(self.run_parallel_steps(data_steps))
                    self.step_status.update(data_results)
                    
                    # Vérification succès global
                    data_success = all(result[0] for result in data_results.values())
                    if not data_success:
                        raise RuntimeError("❌ Échec récupération données")
                
                # ÉTAPE 3: Construction des features
                feature_steps = []
                for symbol in self.config["data"]["symbols"]:
                    for timeframe in self.config["data"]["timeframes"]:
                        step_name = f"features_{symbol}_{timeframe}"
                        command = [
".venv/bin/python", "mlpipeline/features/feature_engineer.py",
                            "--input", f"data/raw/ohlcv_{symbol}_{timeframe}.parquet",
                            "--output", f"data/processed/features_{symbol}_{timeframe}.parquet"
                        ]
                        # Dépend de la récupération des données correspondantes
                        dependencies = [f"data_fetch_{symbol}_{timeframe}"]
                        feature_steps.append((step_name, command, dependencies, 600))
                
                if feature_steps:
                    feature_results = asyncio.run(self.run_parallel_steps(feature_steps))
                    self.step_status.update(feature_results)
                    
                    feature_success = all(result[0] for result in feature_results.values())
                    if not feature_success:
                        raise RuntimeError("❌ Échec construction features")
                
                # ÉTAPE 4: Entraînement des alphas
                alpha_steps = []
                
                for symbol in self.config["data"]["symbols"]:
                    for timeframe in self.config["data"]["timeframes"]:
                        feature_file = f"data/processed/features_{symbol}_{timeframe}.parquet"
                        
                        # Alpha DMN
                        if self.config["alphas"]["dmn"]["enabled"]:
                            step_name = f"alpha_dmn_{symbol}_{timeframe}"
                            command = [
".venv/bin/python", "mlpipeline/alphas/dmn_model.py",
                                "--data-path", feature_file
                            ]
                            dependencies = [f"features_{symbol}_{timeframe}"]
                            alpha_steps.append((step_name, command, dependencies, 3600))
                        
                        # Alpha Mean Reversion
                        if self.config["alphas"]["mean_reversion"]["enabled"]:
                            step_name = f"alpha_mr_{symbol}_{timeframe}"
                            command = [
".venv/bin/python", "mlpipeline/alphas/mean_reversion.py",
                                "--data-1h", feature_file
                            ]
                            dependencies = [f"features_{symbol}_{timeframe}"]
                            alpha_steps.append((step_name, command, dependencies, 1800))
                        
                        # Alpha Funding
                        if self.config["alphas"]["funding"]["enabled"]:
                            step_name = f"alpha_funding_{symbol}_{timeframe}"
                            command = [
".venv/bin/python", "mlpipeline/alphas/funding_strategy.py",
                                "--spot-data", feature_file
                            ]
                            dependencies = [f"features_{symbol}_{timeframe}"]
                            alpha_steps.append((step_name, command, dependencies, 1800))
                
                # Exécution alphas
                if alpha_steps:
                    if self.config["execution"]["parallel_alphas"]:
                        alpha_results = asyncio.run(self.run_parallel_steps(alpha_steps))
                    else:
                        # Séquentiel
                        alpha_results = {}
                        for step_info in alpha_steps:
                            step_name, command, dependencies, timeout = step_info
                            success, result = self._run_step(step_name, command, dependencies, timeout)
                            alpha_results[step_name] = (success, result)
                    
                    self.step_status.update(alpha_results)
                    
                    alpha_success = all(result[0] for result in alpha_results.values())
                    if not alpha_success:
                        logger.warning("⚠️  Certains alphas ont échoué, continuation...")
                
                # ÉTAPE 5: Sélection PSR
                success, result = self._run_step(
                    "psr_selection",
                    [
".venv/bin/python", "scripts/simple_psr_fallback.py",
                        "--psr-threshold", str(self.config["selection"]["psr_threshold"]),
                        "--sharpe-threshold", str(self.config["selection"]["sharpe_threshold"]),
                        "--maxdd-threshold", str(self.config["selection"]["maxdd_threshold"])
                    ],
                    dependencies=None  # Pas de dépendances strictes - marche même si alphas échouent
                )
                self.step_status["psr_selection"] = result
                
                # ÉTAPE 6: Détection des régimes
                for symbol in self.config["data"]["symbols"]:
                    for timeframe in self.config["data"]["timeframes"]:
                        step_name = f"regime_detection_{symbol}_{timeframe}"
                        success, result = self._run_step(
                            step_name,
                            [
".venv/bin/python", "mlpipeline/selection/regime_detector.py",
                                "--data-path", f"data/processed/features_{symbol}_{timeframe}.parquet",
                                "--n-regimes", str(self.config["regime_detection"]["n_regimes"])
                            ],
                            dependencies=[f"features_{symbol}_{timeframe}"]
                        )
                        self.step_status[step_name] = result
                
                # ÉTAPE 7: Export scores pour Freqtrade
                export_steps = []
                for symbol in self.config["data"]["symbols"]:
                    for timeframe in self.config["data"]["timeframes"]:
                        step_name = f"score_export_{symbol}_{timeframe}"
                        command = [
".venv/bin/python", "mlpipeline/scoring/ml_exporter.py",
                            "--symbol", symbol,
                            "--timeframe", timeframe
                        ]
                        # Dépend des alphas et de la sélection
                        dependencies = ["psr_selection"]
                        export_steps.append((step_name, command, dependencies, 300))
                
                if export_steps:
                    export_results = asyncio.run(self.run_parallel_steps(export_steps))
                    self.step_status.update(export_results)
                
                # Calcul durée totale
                self.total_duration = time.time() - self.start_time
                
                # Rapport final
                successful_steps = sum(1 for result in self.step_status.values() 
                                     if (isinstance(result, dict) and result.get('success', False)) or
                                        (isinstance(result, tuple) and len(result) > 0 and result[0]))
                total_steps = len(self.step_status)
                
                pipeline_success = successful_steps == total_steps
                
                # Log métriques finales MLflow
                mlflow.log_metrics({
                    "total_duration_seconds": self.total_duration,
                    "successful_steps": successful_steps,
                    "total_steps": total_steps,
                    "success_rate": successful_steps / max(total_steps, 1),
                    "pipeline_success": 1 if pipeline_success else 0
                })
                
                # Sauvegarde rapport d'exécution
                self._save_execution_report()
                
                if pipeline_success:
                    logger.info(f"✅ PIPELINE TERMINÉ AVEC SUCCÈS ({self.total_duration:.1f}s)")
                else:
                    logger.error(f"❌ PIPELINE TERMINÉ AVEC ERREURS ({self.total_duration:.1f}s)")
                
                return {
                    "success": pipeline_success,
                    "execution_id": self.execution_id,
                    "duration": self.total_duration,
                    "successful_steps": successful_steps,
                    "total_steps": total_steps,
                    "step_status": self.step_status
                }
                
            except Exception as e:
                self.total_duration = time.time() - self.start_time if self.start_time else 0
                logger.error(f"❌ ERREUR FATALE PIPELINE: {e}")
                logger.error(traceback.format_exc())
                
                # Log erreur MLflow
                mlflow.log_metrics({
                    "pipeline_success": 0,
                    "fatal_error": 1,
                    "duration": self.total_duration
                })
                
                return {
                    "success": False,
                    "execution_id": self.execution_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "duration": self.total_duration,
                    "step_status": self.step_status
                }
    
    def _save_execution_report(self):
        """Sauvegarde le rapport d'exécution"""
        
        report = {
            "execution_id": self.execution_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": self.total_duration,
            "config": self.config,
            "step_status": self.step_status
        }
        
        report_file = f"logs/pipeline_report_{self.execution_id}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Rapport sauvegardé: {report_file}")

# ==============================================
# SCRIPT PRINCIPAL
# ==============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Master Pipeline Quantitatif")
    parser.add_argument("--config", default="config/pipeline.yaml",
                       help="Fichier de configuration")
    parser.add_argument("--dry-run", action="store_true",
                       help="Simulation sans exécution")
    
    args = parser.parse_args()
    
    try:
        pipeline = MasterPipeline(args.config)
        
        if args.dry_run:
            logger.info("🎲 MODE DRY RUN - Simulation uniquement")
            # TODO: Implémenter dry run
        else:
            results = pipeline.execute_pipeline()
            
            # Code de sortie selon succès
            sys.exit(0 if results["success"] else 1)
            
    except Exception as e:
        logger.error(f"❌ ERREUR FATALE: {e}")
        logger.error(traceback.format_exc())
        sys.exit(2)