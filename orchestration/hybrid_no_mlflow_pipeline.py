#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid No-MLflow Pipeline - Version test sans dépendance MLflow
Pour tester que les vrais modules s'exécutent correctement
"""

import asyncio
import logging
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import traceback
from dataclasses import dataclass

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
    timeout: int = 1800
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

class HybridNoMLflowPipeline:
    """
    Pipeline hybride sans MLflow pour test
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.execution_id = f"no_mlflow_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # État du pipeline
        self.tasks_results: Dict[str, TaskResult] = {}
        self.start_time: Optional[datetime] = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Setup directories
        self._setup_directories()

        logger.info(f"✅ Hybrid No-MLflow Pipeline initialisé (ID: {self.execution_id})")
        logger.info(f"   - Max concurrent: {self.config.max_concurrent}")
        logger.info(f"   - Symbols: {self.config.symbols}")

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
        Point d'entrée principal du pipeline
        """
        logger.info("🚀 Début pipeline hybride NO-MLFLOW")
        self.start_time = datetime.now()

        try:
            # PHASE 1: Récupération des données
            data_tasks = await self._run_real_data_fetching_phase()

            # PHASE 2: Feature engineering
            feature_tasks = await self._run_real_feature_engineering_phase(data_tasks)

            # PHASE 3: Alphas SANS MLFLOW
            alpha_tasks = await self._run_alphas_no_mlflow_phase(feature_tasks)

            # PHASE 4: Sélection PSR
            psr_result = await self._run_real_psr_selection_phase()

            # PHASE 5: Export ML
            export_tasks = await self._run_real_ml_export_phase()

            # Compilation finale
            execution_time = (datetime.now() - self.start_time).total_seconds()
            pipeline_results = await self._compile_final_results(execution_time)

            success_rate = pipeline_results["summary"]["success_rate"]
            status = pipeline_results["summary"]["overall_status"]

            if status == "SUCCESS":
                logger.info(f"✅ Pipeline NO-MLFLOW RÉUSSI en {execution_time:.1f}s (taux: {success_rate:.1%})")
            else:
                logger.warning(f"⚠️ Pipeline NO-MLFLOW PARTIEL en {execution_time:.1f}s (taux: {success_rate:.1%})")

            return pipeline_results

        except Exception as e:
            logger.error(f"❌ Erreur fatale pipeline: {e}")
            logger.error(traceback.format_exc())
            return await self._handle_pipeline_error(e)

    async def _run_real_data_fetching_phase(self) -> Dict[str, TaskResult]:
        """Phase de récupération données"""
        logger.info("📡 Phase: Récupération données")

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

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compilation
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
        """Phase feature engineering"""
        logger.info("🔧 Phase: Feature Engineering")

        successful_data = {k: v for k, v in data_tasks.items() if v.success}

        tasks = []
        for data_task_name in successful_data.keys():
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

    async def _run_alphas_no_mlflow_phase(self, feature_tasks: Dict[str, TaskResult]) -> Dict[str, TaskResult]:
        """Phase alphas SANS MLflow"""
        logger.info("🧠 Phase: Alphas NO-MLFLOW")

        successful_features = {k: v for k, v in feature_tasks.items() if v.success}

        # Version simplifiée sans MLflow - juste Mean Reversion qui fonctionne le mieux
        alpha_configs = [
            {
                "name": "mean_reversion",
                "script": "mlpipeline/alphas/mean_reversion.py",
                "args": ["--data-1h"]
            }
        ]

        tasks = []
        for feature_name in successful_features.keys():
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
        """Phase sélection PSR"""
        logger.info("📊 Phase: Sélection PSR")

        task_name = "psr_selection"
        command = [
            ".venv/bin/python",
            "scripts/simple_psr_fallback.py",
            "--psr-threshold", "0.3",  # Plus permissif pour test
            "--sharpe-threshold", "0.5",
            "--maxdd-threshold", "0.5"
        ]

        result = await self._run_subprocess_async(task_name, command)
        self.tasks_results[task_name] = result

        return result

    async def _run_real_ml_export_phase(self) -> Dict[str, TaskResult]:
        """Phase export ML"""
        logger.info("📤 Phase: Export ML")

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
        """Wrapper sécurisé pour subprocess async"""

        async with self.semaphore:
            start_time = datetime.now()

            for attempt in range(self.config.retry_attempts):
                try:
                    result = await self._run_subprocess_async(task_name, command)
                    return result

                except Exception as e:
                    error_msg = f"Tentative {attempt + 1}/{self.config.retry_attempts}: {str(e)}"
                    logger.warning(f"⚠️ {task_name}: {error_msg}")

                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay)
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
        """Exécution async subprocess"""

        start_time = datetime.now()
        logger.info(f"🚀 {task_name}: {' '.join(command[-3:])}")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout
            )

            duration = (datetime.now() - start_time).total_seconds()

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
                    logger.error(f"STDERR: {stderr_str[:300]}...")

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

        if success_rate >= 0.7:
            overall_status = "SUCCESS"
        elif success_rate >= 0.4:
            overall_status = "PARTIAL"
        else:
            overall_status = "FAILED"

        results = {
            "execution_id": self.execution_id,
            "timestamp": self.start_time.isoformat(),
            "execution_time": execution_time,
            "architecture": "hybrid_no_mlflow",
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": success_rate,
                "overall_status": overall_status
            },
            "task_details": {k: {
                "success": v.success,
                "duration": v.duration,
                "error": v.error,
                "returncode": v.returncode
            } for k, v in self.tasks_results.items()}
        }

        # Sauvegarde rapport
        report_file = f"logs/no_mlflow_pipeline_report_{self.execution_id}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Rapport sauvegardé: {report_file}")

        return results

    async def _handle_pipeline_error(self, error: Exception) -> Dict:
        """Gestion erreurs"""
        return {
            "execution_id": self.execution_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "error_type": type(error).__name__,
            "status": "FATAL_ERROR"
        }


async def main_no_mlflow():
    """Point d'entrée principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid No-MLflow Pipeline")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1h"])
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--days", type=int, default=30)

    args = parser.parse_args()

    config = PipelineConfig(
        symbols=args.symbols,
        timeframes=args.timeframes,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        days_lookback=args.days
    )

    pipeline = HybridNoMLflowPipeline(config)
    results = await pipeline.execute_pipeline_async()

    # Affichage résultats
    if "summary" in results:
        summary = results["summary"]
        print(f"\n=== RÉSULTATS PIPELINE NO-MLFLOW ===")
        print(f"Execution ID: {results['execution_id']}")
        print(f"Temps total: {results['execution_time']:.1f}s")
        print(f"Tâches: {summary['successful_tasks']}/{summary['total_tasks']}")
        print(f"Taux de succès: {summary['success_rate']:.1%}")
        print(f"Statut: {summary['overall_status']}")

    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(main_no_mlflow())
        success = results.get("summary", {}).get("overall_status") == "SUCCESS"
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrompu")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(2)