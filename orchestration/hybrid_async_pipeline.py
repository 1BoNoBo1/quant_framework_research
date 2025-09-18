#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Async Pipeline - Combine performance async + modules réels
Solution qui garde l'architecture async mais lance les vrais modules pour résultats complets
"""

import asyncio
import logging
import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import traceback
from dataclasses import dataclass

import pandas as pd
import mlflow
from mlflow import MlflowClient

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration du pipeline hybride"""
    symbols: List[str] = None
    timeframes: List[str] = None
    max_concurrent: int = 4
    timeout: int = 1800  # 30 min par défaut
    retry_attempts: int = 2
    retry_delay: float = 5.0
    days_lookback: int = 365

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTCUSDT", "ETHUSDT"]
        if self.timeframes is None:
            self.timeframes = ["1h"]

@dataclass
class TaskResult:
    """Résultat d'une tâche async"""
    task_name: str
    success: bool
    result: Union[Dict, str, None] = None
    error: Optional[str] = None
    duration: float = 0.0
    timestamp: datetime = None
    returncode: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class HybridAsyncPipeline:
    """
    Pipeline hybride : Architecture async + Modules réels

    Combine :
    - Performance et concurrence de l'architecture async
    - Exécution des vrais modules Python pour résultats complets
    - Backtests et exports vers Freqtrade
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.execution_id = f"hybrid_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # État du pipeline
        self.tasks_results: Dict[str, TaskResult] = {}
        self.start_time: Optional[datetime] = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # MLflow client
        self.mlflow_client = MlflowClient()

        # Setup directories
        self._setup_directories()

        logger.info(f"✅ Hybrid Async Pipeline initialisé (ID: {self.execution_id})")
        logger.info(f"   - Max concurrent: {self.config.max_concurrent}")
        logger.info(f"   - Symbols: {self.config.symbols}")
        logger.info(f"   - Timeout: {self.config.timeout}s")

    def _setup_directories(self):
        """Configuration des répertoires nécessaires"""
        directories = [
            "logs", "data/raw", "data/processed", "data/artifacts",
            "data/ml_exports", "models", "cache",
            "freqtrade-prod/user_data/data/ml_scores"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def execute_pipeline_async(self) -> Dict:
        """
        Point d'entrée principal du pipeline hybride
        Architecture async + modules réels
        """
        logger.info("🚀 Début exécution pipeline hybride (async + modules réels)")
        self.start_time = datetime.now()

        try:
            # MLflow experiment
            await self._log_pipeline_start()

            # PHASE 1: Nettoyage artifacts (optionnel)
            cleanup_result = await self._run_cleanup_phase()

            # PHASE 2: Récupération des données (parallèle)
            data_tasks = await self._run_real_data_fetching_phase()

            # PHASE 3: Feature engineering (parallèle)
            feature_tasks = await self._run_real_feature_engineering_phase(data_tasks)

            # PHASE 4: Alphas (parallèle avec vrais modules)
            alpha_tasks = await self._run_real_alphas_phase(feature_tasks)

            # PHASE 5: Sélection PSR
            psr_result = await self._run_real_psr_selection_phase()

            # PHASE 6: Détection régimes
            regime_tasks = await self._run_real_regime_detection_phase(feature_tasks)

            # PHASE 7: Export ML pour Freqtrade
            export_tasks = await self._run_real_ml_export_phase()

            # Compilation finale
            execution_time = (datetime.now() - self.start_time).total_seconds()
            pipeline_results = await self._compile_final_results(execution_time)

            await self._log_pipeline_completion(pipeline_results)

            success_rate = pipeline_results["summary"]["success_rate"]
            status = pipeline_results["summary"]["overall_status"]

            if status == "SUCCESS":
                logger.info(f"✅ Pipeline hybride RÉUSSI en {execution_time:.1f}s (taux: {success_rate:.1%})")
            elif status == "PARTIAL":
                logger.warning(f"⚠️ Pipeline hybride PARTIEL en {execution_time:.1f}s (taux: {success_rate:.1%})")
            else:
                logger.error(f"❌ Pipeline hybride ÉCHOUÉ en {execution_time:.1f}s (taux: {success_rate:.1%})")

            return pipeline_results

        except Exception as e:
            logger.error(f"❌ Erreur fatale pipeline hybride: {e}")
            logger.error(traceback.format_exc())
            return await self._handle_pipeline_error(e)

    async def _log_pipeline_start(self):
        """Log du début d'exécution"""
        try:
            mlflow.set_experiment("HybridQuantPipeline")
            mlflow.start_run(run_name=self.execution_id)

            mlflow.log_params({
                "execution_id": self.execution_id,
                "symbols": ",".join(self.config.symbols),
                "timeframes": ",".join(self.config.timeframes),
                "max_concurrent": self.config.max_concurrent,
                "architecture": "hybrid_async",
                "timeout": self.config.timeout,
                "days_lookback": self.config.days_lookback
            })
        except Exception as e:
            logger.warning(f"⚠️ Échec logging MLflow: {e}")

    async def _run_cleanup_phase(self) -> TaskResult:
        """Phase de nettoyage des artifacts"""
        logger.info("🧹 Phase: Nettoyage artifacts")

        task_name = "artifact_cleanup"
        command = [
            ".venv/bin/python",
            "mlpipeline/utils/artifact_cleaner.py",
            "--project-root", "."
        ]

        result = await self._run_subprocess_async(task_name, command)
        self.tasks_results[task_name] = result

        return result

    async def _run_real_data_fetching_phase(self) -> Dict[str, TaskResult]:
        """Phase de récupération RÉELLE des données crypto"""
        logger.info("📡 Phase: Récupération données réelles (crypto_fetcher)")

        tasks = []
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                task_name = f"data_fetch_{symbol}_{timeframe}"
                command = [
                    ".venv/bin/python",
                    "mlpipeline/data_sources/crypto_fetcher.py",
                    "--symbols", symbol,
                    "--timeframes", timeframe,
                    "--days", str(self.config.days_lookback)
                ]

                task = self._create_safe_subprocess_task(task_name, command)
                tasks.append(task)

        # Exécution parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compilation des résultats
        task_results = {}
        task_index = 0
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                task_name = f"data_fetch_{symbol}_{timeframe}"

                if task_index < len(results):
                    result = results[task_index]
                    if isinstance(result, TaskResult):
                        task_results[task_name] = result
                        self.tasks_results[task_name] = result
                    else:
                        error_result = TaskResult(task_name, False, error=str(result))
                        task_results[task_name] = error_result
                        self.tasks_results[task_name] = error_result

                task_index += 1

        successful_data = sum(1 for r in task_results.values() if r.success)
        logger.info(f"📊 Data fetching: {successful_data}/{len(task_results)} réussis")

        return task_results

    async def _run_real_feature_engineering_phase(self, data_tasks: Dict[str, TaskResult]) -> Dict[str, TaskResult]:
        """Phase de feature engineering RÉELLE"""
        logger.info("🔧 Phase: Feature Engineering réel (feature_engineer)")

        # Filtrer les tâches de données réussies
        successful_data = {k: v for k, v in data_tasks.items() if v.success}

        tasks = []
        for data_task_name, data_result in successful_data.items():
            # Extraire symbol et timeframe
            parts = data_task_name.split('_')
            symbol = parts[2]
            timeframe = parts[3]

            task_name = f"features_{symbol}_{timeframe}"
            command = [
                ".venv/bin/python",
                "mlpipeline/features/feature_engineer.py",
                "--input", f"data/raw/ohlcv_{symbol}_{timeframe}.parquet",
                "--output", f"data/processed/features_{symbol}_{timeframe}.parquet"
            ]

            task = self._create_safe_subprocess_task(task_name, command)
            tasks.append(task)

        # Exécution parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compilation
        task_results = {}
        task_index = 0
        for data_task_name in successful_data.keys():
            parts = data_task_name.split('_')
            symbol = parts[2]
            timeframe = parts[3]
            task_name = f"features_{symbol}_{timeframe}"

            if task_index < len(results):
                result = results[task_index]
                if isinstance(result, TaskResult):
                    task_results[task_name] = result
                    self.tasks_results[task_name] = result
                else:
                    error_result = TaskResult(task_name, False, error=str(result))
                    task_results[task_name] = error_result
                    self.tasks_results[task_name] = error_result

            task_index += 1

        successful_features = sum(1 for r in task_results.values() if r.success)
        logger.info(f"🔧 Feature engineering: {successful_features}/{len(task_results)} réussis")

        return task_results

    async def _run_real_alphas_phase(self, feature_tasks: Dict[str, TaskResult]) -> Dict[str, TaskResult]:
        """Phase d'exécution des VRAIS alphas"""
        logger.info("🧠 Phase: Alphas réels (dmn, mean_reversion, funding)")

        successful_features = {k: v for k, v in feature_tasks.items() if v.success}

        # Définir les alphas disponibles
        alpha_configs = [
            {
                "name": "dmn",
                "script": "mlpipeline/alphas/dmn_model.py",
                "args": ["--data-path"]
            },
            {
                "name": "mean_reversion",
                "script": "mlpipeline/alphas/mean_reversion.py",
                "args": ["--data-1h"]
            },
            {
                "name": "funding",
                "script": "mlpipeline/alphas/funding_strategy.py",
                "args": ["--spot-data"]
            }
        ]

        tasks = []
        for feature_name, feature_result in successful_features.items():
            # Extraire symbol et timeframe
            parts = feature_name.split('_')
            symbol = parts[1]
            timeframe = parts[2]

            feature_file = f"data/processed/features_{symbol}_{timeframe}.parquet"

            for alpha_config in alpha_configs:
                alpha_name = alpha_config["name"]
                task_name = f"alpha_{alpha_name}_{symbol}_{timeframe}"

                command = [
                    ".venv/bin/python",
                    alpha_config["script"],
                    alpha_config["args"][0], feature_file
                ]

                task = self._create_safe_subprocess_task(task_name, command)
                tasks.append(task)

        # Exécution parallèle avec limite de concurrence
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compilation
        task_results = {}
        task_index = 0
        for feature_name in successful_features.keys():
            parts = feature_name.split('_')
            symbol = parts[1]
            timeframe = parts[2]

            for alpha_config in alpha_configs:
                alpha_name = alpha_config["name"]
                task_name = f"alpha_{alpha_name}_{symbol}_{timeframe}"

                if task_index < len(results):
                    result = results[task_index]
                    if isinstance(result, TaskResult):
                        task_results[task_name] = result
                        self.tasks_results[task_name] = result
                    else:
                        error_result = TaskResult(task_name, False, error=str(result))
                        task_results[task_name] = error_result
                        self.tasks_results[task_name] = error_result

                task_index += 1

        successful_alphas = sum(1 for r in task_results.values() if r.success)
        logger.info(f"🧠 Alphas: {successful_alphas}/{len(task_results)} réussis")

        return task_results

    async def _run_real_psr_selection_phase(self) -> TaskResult:
        """Phase de sélection PSR RÉELLE"""
        logger.info("📊 Phase: Sélection PSR réelle")

        task_name = "psr_selection"
        command = [
            ".venv/bin/python",
            "scripts/simple_psr_fallback.py",
            "--psr-threshold", "0.65",
            "--sharpe-threshold", "1.0",
            "--maxdd-threshold", "0.25"
        ]

        result = await self._run_subprocess_async(task_name, command)
        self.tasks_results[task_name] = result

        return result

    async def _run_real_regime_detection_phase(self, feature_tasks: Dict[str, TaskResult]) -> Dict[str, TaskResult]:
        """Phase de détection des régimes RÉELLE"""
        logger.info("📈 Phase: Détection régimes réelle")

        successful_features = {k: v for k, v in feature_tasks.items() if v.success}

        tasks = []
        for feature_name, feature_result in successful_features.items():
            parts = feature_name.split('_')
            symbol = parts[1]
            timeframe = parts[2]

            task_name = f"regime_detection_{symbol}_{timeframe}"
            command = [
                ".venv/bin/python",
                "mlpipeline/selection/regime_detector.py",
                "--data-path", f"data/processed/features_{symbol}_{timeframe}.parquet",
                "--n-regimes", "3"
            ]

            task = self._create_safe_subprocess_task(task_name, command)
            tasks.append(task)

        # Exécution parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compilation
        task_results = {}
        task_index = 0
        for feature_name in successful_features.keys():
            parts = feature_name.split('_')
            symbol = parts[1]
            timeframe = parts[2]
            task_name = f"regime_detection_{symbol}_{timeframe}"

            if task_index < len(results):
                result = results[task_index]
                if isinstance(result, TaskResult):
                    task_results[task_name] = result
                    self.tasks_results[task_name] = result
                else:
                    error_result = TaskResult(task_name, False, error=str(result))
                    task_results[task_name] = error_result
                    self.tasks_results[task_name] = error_result

            task_index += 1

        successful_regimes = sum(1 for r in task_results.values() if r.success)
        logger.info(f"📈 Regime detection: {successful_regimes}/{len(task_results)} réussis")

        return task_results

    async def _run_real_ml_export_phase(self) -> Dict[str, TaskResult]:
        """Phase d'export ML pour Freqtrade RÉELLE"""
        logger.info("📤 Phase: Export ML Freqtrade réel")

        tasks = []
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                task_name = f"ml_export_{symbol}_{timeframe}"
                command = [
                    ".venv/bin/python",
                    "mlpipeline/scoring/ml_exporter.py",
                    "--symbol", symbol,
                    "--timeframe", timeframe
                ]

                task = self._create_safe_subprocess_task(task_name, command)
                tasks.append(task)

        # Exécution parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compilation
        task_results = {}
        task_index = 0
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                task_name = f"ml_export_{symbol}_{timeframe}"

                if task_index < len(results):
                    result = results[task_index]
                    if isinstance(result, TaskResult):
                        task_results[task_name] = result
                        self.tasks_results[task_name] = result
                    else:
                        error_result = TaskResult(task_name, False, error=str(result))
                        task_results[task_name] = error_result
                        self.tasks_results[task_name] = error_result

                task_index += 1

        successful_exports = sum(1 for r in task_results.values() if r.success)
        logger.info(f"📤 ML Export: {successful_exports}/{len(task_results)} réussis")

        return task_results

    async def _create_safe_subprocess_task(self, task_name: str, command: List[str]) -> TaskResult:
        """Wrapper sécurisé pour subprocess async avec retry"""

        async with self.semaphore:  # Limite de concurrence
            start_time = datetime.now()

            for attempt in range(self.config.retry_attempts):
                try:
                    result = await self._run_subprocess_async(task_name, command)
                    return result

                except Exception as e:
                    error_msg = f"Tentative {attempt + 1}/{self.config.retry_attempts}: {str(e)}"
                    logger.warning(f"⚠️ {task_name}: {error_msg}")

                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        duration = (datetime.now() - start_time).total_seconds()
                        return TaskResult(
                            task_name=task_name,
                            success=False,
                            error=error_msg,
                            duration=duration
                        )

    async def _run_subprocess_async(self, task_name: str, command: List[str]) -> TaskResult:
        """Exécution async d'un subprocess avec timeout"""

        start_time = datetime.now()
        logger.info(f"🚀 {task_name}: {' '.join(command[-3:])}")  # Log commande courte

        try:
            # Créer subprocess async
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )

            # Attendre avec timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Décoder les outputs
            stdout_str = stdout.decode('utf-8') if stdout else ""
            stderr_str = stderr.decode('utf-8') if stderr else ""

            if process.returncode == 0:
                logger.info(f"✅ {task_name} terminé ({duration:.1f}s)")
                return TaskResult(
                    task_name=task_name,
                    success=True,
                    result={"status": "completed"},
                    duration=duration,
                    returncode=process.returncode,
                    stdout=stdout_str,
                    stderr=stderr_str
                )
            else:
                logger.error(f"❌ {task_name} échoué (code {process.returncode})")
                if stderr_str:
                    logger.error(f"STDERR: {stderr_str[:500]}...")  # Tronquer pour lisibilité

                return TaskResult(
                    task_name=task_name,
                    success=False,
                    result={"status": "failed"},
                    error=f"Exit code {process.returncode}",
                    duration=duration,
                    returncode=process.returncode,
                    stdout=stdout_str,
                    stderr=stderr_str
                )

        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Timeout après {self.config.timeout}s"
            logger.error(f"⏰ {task_name}: {error_msg}")

            # Tuer le process si toujours en cours
            if 'process' in locals():
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass

            return TaskResult(
                task_name=task_name,
                success=False,
                error=error_msg,
                duration=duration
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Erreur subprocess: {str(e)}"
            logger.error(f"❌ {task_name}: {error_msg}")

            return TaskResult(
                task_name=task_name,
                success=False,
                error=error_msg,
                duration=duration
            )

    async def _compile_final_results(self, execution_time: float) -> Dict:
        """Compilation finale des résultats"""

        total_tasks = len(self.tasks_results)
        successful_tasks = sum(1 for r in self.tasks_results.values() if r.success)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0

        # Statistiques par phase
        phase_stats = {}
        phases = ["data_fetch", "features", "alpha", "psr_selection", "regime_detection", "ml_export"]

        for phase in phases:
            phase_tasks = {k: v for k, v in self.tasks_results.items() if k.startswith(phase)}
            if phase_tasks:
                phase_successful = sum(1 for r in phase_tasks.values() if r.success)
                phase_stats[phase] = {
                    "total": len(phase_tasks),
                    "successful": phase_successful,
                    "success_rate": phase_successful / len(phase_tasks),
                    "avg_duration": sum(r.duration for r in phase_tasks.values()) / len(phase_tasks)
                }

        # Déterminer statut global
        if success_rate >= 0.8:
            overall_status = "SUCCESS"
        elif success_rate >= 0.5:
            overall_status = "PARTIAL"
        else:
            overall_status = "FAILED"

        return {
            "execution_id": self.execution_id,
            "timestamp": self.start_time.isoformat(),
            "execution_time": execution_time,
            "architecture": "hybrid_async",
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": success_rate,
                "overall_status": overall_status
            },
            "phase_statistics": phase_stats,
            "task_details": {k: {
                "success": v.success,
                "duration": v.duration,
                "error": v.error,
                "returncode": v.returncode
            } for k, v in self.tasks_results.items()},
            "performance_metrics": {
                "total_duration": execution_time,
                "avg_task_duration": sum(r.duration for r in self.tasks_results.values()) / total_tasks if total_tasks > 0 else 0,
                "parallelization_efficiency": (sum(r.duration for r in self.tasks_results.values()) / execution_time) if execution_time > 0 else 0
            }
        }

    async def _log_pipeline_completion(self, results: Dict):
        """Log de completion async"""
        try:
            # Log métriques finales vers MLflow
            mlflow.log_metrics({
                "execution_time": results["execution_time"],
                "success_rate": results["summary"]["success_rate"],
                "total_tasks": results["summary"]["total_tasks"],
                "parallelization_efficiency": results["performance_metrics"]["parallelization_efficiency"]
            })

            # Log phase statistics
            for phase, stats in results["phase_statistics"].items():
                mlflow.log_metrics({
                    f"{phase}_success_rate": stats["success_rate"],
                    f"{phase}_avg_duration": stats["avg_duration"]
                })

            # Sauvegarde rapport JSON
            report_file = f"logs/hybrid_pipeline_report_{self.execution_id}.json"
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"📊 Rapport sauvegardé: {report_file}")

            mlflow.end_run()

        except Exception as e:
            logger.warning(f"⚠️ Échec logging completion: {e}")

    async def _handle_pipeline_error(self, error: Exception) -> Dict:
        """Gestion centralisée des erreurs"""
        error_details = {
            "execution_id": self.execution_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "architecture": "hybrid_async",
            "status": "FATAL_ERROR"
        }

        try:
            mlflow.log_params({"fatal_error": str(error)})
            mlflow.end_run(status="FAILED")
        except:
            pass

        return error_details


# Point d'entrée principal
async def main_hybrid():
    """
    Point d'entrée principal du pipeline hybride
    """
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Async Pipeline - Perf async + modules réels")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1h"])
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--days", type=int, default=365)

    args = parser.parse_args()

    # Configuration
    config = PipelineConfig(
        symbols=args.symbols,
        timeframes=args.timeframes,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        days_lookback=args.days
    )

    # Exécution pipeline
    pipeline = HybridAsyncPipeline(config)
    results = await pipeline.execute_pipeline_async()

    # Affichage résultats
    if "summary" in results:
        summary = results["summary"]
        print(f"\n=== RÉSULTATS PIPELINE HYBRIDE ===")
        print(f"Execution ID: {results['execution_id']}")
        print(f"Temps total: {results['execution_time']:.1f}s")
        print(f"Tâches: {summary['successful_tasks']}/{summary['total_tasks']}")
        print(f"Taux de succès: {summary['success_rate']:.1%}")
        print(f"Statut: {summary['overall_status']}")
        print(f"Efficacité parallélisation: {results['performance_metrics']['parallelization_efficiency']:.1f}x")

        # Détail par phase
        print(f"\n--- DÉTAIL PAR PHASE ---")
        for phase, stats in results.get("phase_statistics", {}).items():
            print(f"{phase}: {stats['successful']}/{stats['total']} ({stats['success_rate']:.1%}) - {stats['avg_duration']:.1f}s moy")

    return results


# Point d'entrée unique
if __name__ == "__main__":
    try:
        results = asyncio.run(main_hybrid())
        success = results.get("summary", {}).get("overall_status") == "SUCCESS"
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrompu par utilisateur")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(2)