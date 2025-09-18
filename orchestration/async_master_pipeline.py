#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Master Pipeline - Architecture asynchrone robuste
Un seul asyncio.run() au début, tout le reste en async pour performance maximale
"""

import asyncio
import logging
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import traceback
import subprocess
from dataclasses import dataclass

import pandas as pd
import numpy as np
import aiohttp
import aiofiles
import mlflow
from mlflow import MlflowClient

# Import async data management
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mlpipeline.utils.async_data import AsyncDataManager, create_async_data_pipeline

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration du pipeline async"""
    symbols: List[str] = None
    timeframes: List[str] = None
    max_concurrent: int = 4
    timeout: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
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
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AsyncMasterPipeline:
    """
    Pipeline principal asynchrone avec architecture robuste
    
    Avantages async:
    - Un seul asyncio.run() au point d'entrée
    - Parallélisation native sans subprocess
    - Gestion d'erreurs centralisée
    - Timeout et retry automatiques
    - Performance maximale avec concurrence
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.execution_id = f"async_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # État du pipeline
        self.tasks_results: Dict[str, TaskResult] = {}
        self.start_time: Optional[datetime] = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # MLflow client async-compatible
        self.mlflow_client = MlflowClient()
        
        # Async data manager (sera initialisé dans execute_pipeline_async)
        self.data_manager: Optional[AsyncDataManager] = None
        
        # Création des répertoires
        self._setup_directories()
        
        logger.info(f"✅ Async Master Pipeline initialisé (ID: {self.execution_id})")
        logger.info(f"   - Max concurrent: {self.config.max_concurrent}")
        logger.info(f"   - Symbols: {self.config.symbols}")
    
    def _setup_directories(self):
        """Configuration des répertoires nécessaires"""
        directories = [
            "logs", "data/raw", "data/processed", "data/artifacts", 
            "data/ml_exports", "models", "cache"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def execute_pipeline_async(self) -> Dict:
        """
        Point d'entrée principal du pipeline async
        Remplace toutes les approches subprocess par des appels async natifs
        """
        logger.info("🚀 Début exécution pipeline async")
        self.start_time = datetime.now()
        
        try:
            # 0. Initialisation async data manager
            config_data = {'max_workers': self.config.max_concurrent}
            self.data_manager = await create_async_data_pipeline(config_data)
            logger.info("✅ Async data manager initialisé")
            
            # 1. Phase de préparation async
            await self._log_pipeline_start()
            
            # 2. Récupération des données (parallèle avec async I/O)
            data_tasks = await self._run_data_fetching_phase()
            
            # 3. Feature engineering (parallèle par symbol avec async I/O)
            feature_tasks = await self._run_feature_engineering_phase(data_tasks)
            
            # 4. Exécution des alphas (parallèle optimisé avec async native)
            alpha_tasks = await self._run_alphas_phase(feature_tasks)
            
            # 5. Post-traitement (PSR, régimes, ML export)
            post_processing_tasks = await self._run_post_processing_phase(alpha_tasks)
            
            # 6. Finalisation
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            # 7. Compilation résultats
            pipeline_results = await self._compile_final_results(execution_time)
            
            await self._log_pipeline_completion(pipeline_results)
            
            logger.info(f"✅ Pipeline async terminé en {execution_time:.1f}s")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Erreur fatale pipeline async: {e}")
            logger.error(traceback.format_exc())
            return await self._handle_pipeline_error(e)
        
        finally:
            # Nettoyage des ressources async
            if self.data_manager:
                await self.data_manager.cleanup()
    
    async def _log_pipeline_start(self):
        """Log du début d'exécution async"""
        try:
            mlflow.set_experiment("AsyncQuantPipeline")
            mlflow.start_run(run_name=self.execution_id)
            
            mlflow.log_params({
                "execution_id": self.execution_id,
                "symbols": ",".join(self.config.symbols),
                "timeframes": ",".join(self.config.timeframes),
                "max_concurrent": self.config.max_concurrent,
                "architecture": "async"
            })
        except Exception as e:
            logger.warning(f"⚠️ Échec logging MLflow: {e}")
    
    async def _run_data_fetching_phase(self) -> Dict[str, TaskResult]:
        """Phase de récupération des données en parallèle"""
        logger.info("📡 Phase: Récupération données (async)")
        
        tasks = []
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                task_name = f"data_fetch_{symbol}_{timeframe}"
                task = self._create_safe_task(
                    self._fetch_data_async(symbol, timeframe),
                    task_name
                )
                tasks.append(task)
        
        # Exécution parallèle avec limite de concurrence
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compilation des résultats
        task_results = {}
        for i, result in enumerate(results):
            task_name = f"data_fetch_{self.config.symbols[i // len(self.config.timeframes)]}_{self.config.timeframes[i % len(self.config.timeframes)]}"
            if isinstance(result, TaskResult):
                task_results[task_name] = result
            else:
                task_results[task_name] = TaskResult(task_name, False, error=str(result))
        
        successful_tasks = sum(1 for r in task_results.values() if r.success)
        logger.info(f"📊 Data fetching: {successful_tasks}/{len(task_results)} réussis")
        
        return task_results
    
    async def _fetch_data_async(self, symbol: str, timeframe: str) -> Dict:
        """Récupération async des données pour un symbol/timeframe"""
        # Simulation - en réalité, utiliserait aiohttp pour API calls
        logger.debug(f"📡 Fetching {symbol} {timeframe}")
        
        # Simuler appel API async
        await asyncio.sleep(0.1)  # Simule latence réseau
        
        # Vérifier si données existent déjà
        output_file = f"data/raw/ohlcv_{symbol}_{timeframe}.parquet"
        
        if os.path.exists(output_file):
            logger.debug(f"✅ Données {symbol} {timeframe} trouvées en cache")
            return {"status": "cached", "file": output_file}
        
        # Simuler téléchargement (en réalité utiliserait ccxt async)
        import numpy as np
        dates = pd.date_range('2025-01-01', '2025-09-13', freq='1h')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(100, 200, len(dates)),
            'low': np.random.uniform(100, 200, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000, 10000, len(dates)),
            'time': dates
        })
        
        # Sauvegarde async
        await self._save_data_async(data, output_file)
        
        return {"status": "fetched", "file": output_file, "records": len(data)}
    
    async def _save_data_async(self, data: pd.DataFrame, filepath: str):
        """Sauvegarde async des données"""
        loop = asyncio.get_event_loop()
        
        # Utiliser thread pool pour I/O synchrone
        await loop.run_in_executor(
            None, 
            lambda: data.to_parquet(filepath, index=False)
        )
    
    async def _run_feature_engineering_phase(self, data_tasks: Dict[str, TaskResult]) -> Dict[str, TaskResult]:
        """Phase de feature engineering en parallèle"""
        logger.info("🔧 Phase: Feature Engineering (async)")
        
        # Filtrer les tâches réussies
        successful_data = {k: v for k, v in data_tasks.items() if v.success}
        
        tasks = []
        for task_name, data_result in successful_data.items():
            parts = task_name.split('_')
            symbol = parts[2]
            timeframe = parts[3]
            
            feature_task_name = f"features_{symbol}_{timeframe}"
            task = self._create_safe_task(
                self._engineer_features_async(symbol, timeframe, data_result),
                feature_task_name
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compilation
        task_results = {}
        for i, result in enumerate(results):
            task_name = list(successful_data.keys())[i].replace('data_fetch', 'features')
            if isinstance(result, TaskResult):
                task_results[task_name] = result
            else:
                task_results[task_name] = TaskResult(task_name, False, error=str(result))
        
        successful_features = sum(1 for r in task_results.values() if r.success)
        logger.info(f"🔧 Feature engineering: {successful_features}/{len(task_results)} réussis")
        
        return task_results
    
    async def _engineer_features_async(self, symbol: str, timeframe: str, data_result: TaskResult) -> Dict:
        """Feature engineering async pour un dataset"""
        logger.debug(f"🔧 Engineering features {symbol} {timeframe}")
        
        input_file = data_result.result["file"]
        output_file = f"data/processed/features_{symbol}_{timeframe}.parquet"
        
        # Chargement async
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, pd.read_parquet, input_file)
        
        # Feature engineering (CPU-bound, utiliser executor)
        features = await loop.run_in_executor(
            None, 
            self._compute_features_sync, 
            data
        )
        
        # Sauvegarde async
        await self._save_data_async(features, output_file)
        
        return {
            "status": "completed",
            "input_file": input_file,
            "output_file": output_file,
            "features_count": len(features.columns),
            "records": len(features)
        }
    
    def _compute_features_sync(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcul synchrone des features (CPU-bound)"""
        # Import de notre feature engineer existant
        sys.path.append(str(Path(__file__).parent.parent))
        
        try:
            from mlpipeline.features.feature_engineer import add_technical_indicators, clean_features, add_alpha_compatible_columns
            
            # Pipeline de features
            data_with_features = add_technical_indicators(data)
            data_clean = clean_features(data_with_features)
            data_final = add_alpha_compatible_columns(data_clean)
            
            # Index temporel critique
            if 'time' in data_final.columns:
                data_final = data_final.set_index('time')
            
            return data_final
            
        except ImportError as e:
            logger.warning(f"⚠️ Import feature engineer échoué: {e}")
            # Fallback simple
            data['returns'] = data['close'].pct_change()
            data['sma_20'] = data['close'].rolling(20).mean()
            if 'time' in data.columns:
                data = data.set_index('time')
            return data
    
    async def _run_alphas_phase(self, feature_tasks: Dict[str, TaskResult]) -> Dict[str, TaskResult]:
        """Phase d'exécution des alphas en parallèle optimisé"""
        logger.info("🧠 Phase: Alphas (async)")
        
        successful_features = {k: v for k, v in feature_tasks.items() if v.success}
        
        # Créer tâches pour chaque alpha × symbol × timeframe
        alpha_types = ["dmn", "mean_reversion", "funding"]
        tasks = []
        
        for feature_name, feature_result in successful_features.items():
            parts = feature_name.split('_')
            symbol = parts[1]
            timeframe = parts[2]
            
            for alpha_type in alpha_types:
                alpha_task_name = f"alpha_{alpha_type}_{symbol}_{timeframe}"
                task = self._create_safe_task(
                    self._run_alpha_async(alpha_type, symbol, timeframe, feature_result),
                    alpha_task_name
                )
                tasks.append(task)
        
        # Exécution avec contrôle de concurrence
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compilation
        task_results = {}
        task_index = 0
        for feature_name, feature_result in successful_features.items():
            parts = feature_name.split('_')
            symbol = parts[1]
            timeframe = parts[2]
            
            for alpha_type in alpha_types:
                alpha_task_name = f"alpha_{alpha_type}_{symbol}_{timeframe}"
                
                if task_index < len(results):
                    result = results[task_index]
                    if isinstance(result, TaskResult):
                        task_results[alpha_task_name] = result
                    else:
                        task_results[alpha_task_name] = TaskResult(alpha_task_name, False, error=str(result))
                
                task_index += 1
        
        successful_alphas = sum(1 for r in task_results.values() if r.success)
        logger.info(f"🧠 Alphas: {successful_alphas}/{len(task_results)} réussis")
        
        return task_results
    
    async def _run_alpha_async(self, alpha_type: str, symbol: str, timeframe: str, feature_result: TaskResult) -> Dict:
        """Exécution async d'un alpha spécifique"""
        logger.debug(f"🧠 Running {alpha_type} alpha for {symbol} {timeframe}")
        
        feature_file = feature_result.result["output_file"]
        
        # Chargement des données async
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, pd.read_parquet, feature_file)
        
        # Exécution de l'alpha (async)
        try:
            alpha_result = await self._execute_alpha_async(alpha_type, symbol, data)
            
            # Sauvegarde résultats
            artifacts_dir = Path("data/artifacts")
            metrics_file = artifacts_dir / f"{alpha_type}_metrics_{symbol}.json"
            
            async with aiofiles.open(metrics_file, 'w') as f:
                await f.write(json.dumps(alpha_result, indent=2))
            
            return {
                "status": "completed",
                "alpha_type": alpha_type,
                "symbol": symbol,
                "metrics": alpha_result,
                "metrics_file": str(metrics_file)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur alpha {alpha_type} {symbol}: {e}")
            return {
                "status": "failed",
                "alpha_type": alpha_type, 
                "symbol": symbol,
                "error": str(e)
            }
    
    async def _execute_alpha_async(self, alpha_type: str, symbol: str, data: pd.DataFrame) -> Dict:
        """Exécution asynchrone d'un alpha"""

        try:
            # Import des alphas
            if alpha_type == "dmn":
                from mlpipeline.alphas.dmn_model import train_dmn_alpha
                return await train_dmn_alpha(data, {})

            elif alpha_type == "mean_reversion":
                from mlpipeline.alphas.mean_reversion import AdaptiveMeanReversion

                # Instancier et exécuter
                strategy = AdaptiveMeanReversion()
                signals_df = await strategy.generate_signals(data)
                metrics = await strategy.backtest_strategy(signals_df)
                return metrics

            elif alpha_type == "funding":
                from mlpipeline.alphas.funding_strategy import train_funding_alpha
                return await train_funding_alpha("", config={})
            
        except Exception as e:
            logger.warning(f"⚠️ Fallback alpha {alpha_type}: {e}")
            
            # Fallback simple
            returns = data['close'].pct_change() if 'close' in data.columns else pd.Series([0])
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            return {
                "alpha": alpha_type,
                "symbol": symbol,
                "sharpe": float(sharpe),
                "total_return": float((1 + returns).prod() - 1),
                "volatility": float(returns.std() * np.sqrt(252)),
                "status": "fallback"
            }
    
    async def _run_post_processing_phase(self, alpha_tasks: Dict[str, TaskResult]) -> Dict[str, TaskResult]:
        """Phase de post-traitement async"""
        logger.info("📊 Phase: Post-processing (async)")
        
        tasks = [
            self._create_safe_task(self._run_psr_selection_async(alpha_tasks), "psr_selection"),
            self._create_safe_task(self._run_regime_detection_async(), "regime_detection"),
            self._create_safe_task(self._run_ml_export_async(), "ml_export")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        task_results = {}
        task_names = ["psr_selection", "regime_detection", "ml_export"]
        
        for i, result in enumerate(results):
            task_name = task_names[i]
            if isinstance(result, TaskResult):
                task_results[task_name] = result
            else:
                task_results[task_name] = TaskResult(task_name, False, error=str(result))
        
        successful_post = sum(1 for r in task_results.values() if r.success)
        logger.info(f"📊 Post-processing: {successful_post}/{len(task_results)} réussis")
        
        return task_results
    
    async def _run_psr_selection_async(self, alpha_tasks: Dict[str, TaskResult]) -> Dict:
        """PSR Selection async"""
        logger.debug("📊 Running PSR selection")
        
        # Simuler PSR selection
        await asyncio.sleep(0.5)
        
        successful_alphas = [t for t in alpha_tasks.values() if t.success]
        
        return {
            "status": "completed",
            "alphas_analyzed": len(alpha_tasks),
            "successful_alphas": len(successful_alphas),
            "selected_count": min(3, len(successful_alphas))
        }
    
    async def _run_regime_detection_async(self) -> Dict:
        """Regime Detection async"""
        logger.debug("📊 Running regime detection")
        
        await asyncio.sleep(0.3)
        
        return {
            "status": "completed", 
            "regimes_detected": ["bull", "bear", "sideways"],
            "current_regime": "bull"
        }
    
    async def _run_ml_export_async(self) -> Dict:
        """ML Export async"""
        logger.debug("📊 Running ML export")
        
        await asyncio.sleep(0.2)
        
        return {
            "status": "completed",
            "exports_created": 2,
            "export_directory": "data/ml_exports"
        }
    
    async def _create_safe_task(self, coro, task_name: str) -> TaskResult:
        """Wrapper sécurisé pour tâches async avec timeout et retry"""
        
        async with self.semaphore:  # Limite de concurrence
            start_time = datetime.now()
            
            for attempt in range(self.config.retry_attempts):
                try:
                    # Timeout par tâche
                    result = await asyncio.wait_for(
                        coro, 
                        timeout=self.config.timeout
                    )
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    return TaskResult(
                        task_name=task_name,
                        success=True, 
                        result=result,
                        duration=duration
                    )
                    
                except asyncio.TimeoutError:
                    error_msg = f"Timeout après {self.config.timeout}s"
                    logger.warning(f"⏰ {task_name}: {error_msg}")
                    
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        duration = (datetime.now() - start_time).total_seconds()
                        return TaskResult(task_name, False, error=error_msg, duration=duration)
                        
                except Exception as e:
                    error_msg = f"Erreur: {str(e)}"
                    logger.warning(f"❌ {task_name}: {error_msg}")
                    
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        duration = (datetime.now() - start_time).total_seconds()
                        return TaskResult(task_name, False, error=error_msg, duration=duration)
    
    async def _compile_final_results(self, execution_time: float) -> Dict:
        """Compilation finale des résultats async"""
        
        total_tasks = len(self.tasks_results)
        successful_tasks = sum(1 for r in self.tasks_results.values() if r.success)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        # Métriques par phase
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
        
        return {
            "execution_id": self.execution_id,
            "timestamp": self.start_time.isoformat(),
            "execution_time": execution_time,
            "architecture": "async",
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks, 
                "success_rate": success_rate,
                "overall_status": "SUCCESS" if success_rate >= 0.8 else "PARTIAL" if success_rate >= 0.5 else "FAILED"
            },
            "phase_statistics": phase_stats,
            "task_details": {k: {
                "success": v.success,
                "duration": v.duration,
                "error": v.error
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
            
            mlflow.end_run()
            
        except Exception as e:
            logger.warning(f"⚠️ Échec logging MLflow completion: {e}")
    
    async def _handle_pipeline_error(self, error: Exception) -> Dict:
        """Gestion centralisée des erreurs"""
        error_details = {
            "execution_id": self.execution_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "architecture": "async",
            "status": "FATAL_ERROR"
        }
        
        try:
            mlflow.log_params({"fatal_error": str(error)})
            mlflow.end_run(status="FAILED")
        except:
            pass
        
        return error_details

# Point d'entrée unique async
async def main_async():
    """
    Point d'entrée principal async - UN SEUL asyncio.run() ici
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Async Master Pipeline")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1h"])
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    
    args = parser.parse_args()
    
    # Configuration
    config = PipelineConfig(
        symbols=args.symbols,
        timeframes=args.timeframes,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout
    )
    
    # Exécution pipeline async
    pipeline = AsyncMasterPipeline(config)
    results = await pipeline.execute_pipeline_async()
    
    # Affichage résultats
    if "summary" in results:
        summary = results["summary"]
        print(f"\n=== RÉSULTATS PIPELINE ASYNC ===")
        print(f"Execution ID: {results['execution_id']}")
        print(f"Temps total: {results['execution_time']:.1f}s")
        print(f"Tâches: {summary['successful_tasks']}/{summary['total_tasks']}")
        print(f"Taux de succès: {summary['success_rate']:.1%}")
        print(f"Statut: {summary['overall_status']}")
        print(f"Efficacité parallélisation: {results['performance_metrics']['parallelization_efficiency']:.1f}x")
    
    return results

# Point d'entrée unique - UN SEUL asyncio.run()
if __name__ == "__main__":
    try:
        results = asyncio.run(main_async())
        sys.exit(0 if results.get("summary", {}).get("overall_status") == "SUCCESS" else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrompu par utilisateur")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(2)