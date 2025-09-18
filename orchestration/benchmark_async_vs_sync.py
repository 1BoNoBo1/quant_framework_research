#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Async vs Sync - Test de performance du pipeline
Compare les performances entre l'architecture async et sync
"""

import asyncio
import time
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
import json

# Setup paths
sys.path.append(str(Path(__file__).parent.parent))

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """
    Benchmark pour comparer les performances async vs sync
    """
    
    def __init__(self):
        self.results = {
            "async": {},
            "sync": {},
            "comparison": {}
        }
    
    async def benchmark_async_pipeline(self) -> Dict:
        """
        Test performance du pipeline async
        """
        logger.info("🚀 Benchmark pipeline ASYNC")
        start_time = time.time()
        
        try:
            from orchestration.async_master_pipeline import AsyncMasterPipeline, PipelineConfig
            
            # Configuration pour le test
            config = PipelineConfig(
                symbols=["BTCUSDT", "ETHUSDT"],
                timeframes=["1h"],
                max_concurrent=4,
                timeout=60
            )
            
            # Création du pipeline
            pipeline = AsyncMasterPipeline(config)
            
            # Exécution async
            results = await pipeline.execute_pipeline_async()
            
            execution_time = time.time() - start_time
            
            # Calcul des métriques de performance
            async_metrics = {
                "execution_time": execution_time,
                "success_rate": self._calculate_success_rate(results),
                "tasks_completed": len([r for r in results.values() if isinstance(r, dict) and r.get('status') == 'success']),
                "memory_efficiency": "async_optimized",
                "concurrent_tasks": config.max_concurrent,
                "architecture": "async_native"
            }
            
            logger.info(f"✅ Async pipeline terminé en {execution_time:.2f}s")
            return async_metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur benchmark async: {e}")
            return {
                "execution_time": time.time() - start_time,
                "error": str(e),
                "success": False
            }
    
    def benchmark_sync_pipeline(self) -> Dict:
        """
        Test performance du pipeline sync (simulation)
        """
        logger.info("🔄 Benchmark pipeline SYNC (simulation)")
        start_time = time.time()
        
        try:
            # Simulation du pipeline sync avec les mêmes tâches
            symbols = ["BTCUSDT", "ETHUSDT"]
            
            # Phase 1: Collecte des données (séquentielle)
            for symbol in symbols:
                time.sleep(0.5)  # Simulation I/O blocking
                logger.debug(f"Sync data collection: {symbol}")
            
            # Phase 2: Feature engineering (séquentielle)
            for symbol in symbols:
                time.sleep(1.0)  # Simulation processing
                logger.debug(f"Sync feature engineering: {symbol}")
            
            # Phase 3: Entraînement alphas (séquentiel)
            alphas = ["dmn", "mean_reversion", "funding"]
            for alpha in alphas:
                time.sleep(2.0)  # Simulation training
                logger.debug(f"Sync alpha training: {alpha}")
            
            execution_time = time.time() - start_time
            
            sync_metrics = {
                "execution_time": execution_time,
                "success_rate": 1.0,  # Simulation parfaite
                "tasks_completed": len(symbols) * 2 + len(alphas),
                "memory_efficiency": "sync_sequential",
                "concurrent_tasks": 1,
                "architecture": "sync_sequential"
            }
            
            logger.info(f"✅ Sync pipeline simulé terminé en {execution_time:.2f}s")
            return sync_metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur benchmark sync: {e}")
            return {
                "execution_time": time.time() - start_time,
                "error": str(e),
                "success": False
            }
    
    async def benchmark_async_data_operations(self) -> Dict:
        """
        Benchmark spécifique des opérations I/O async
        """
        logger.info("📊 Benchmark I/O operations ASYNC")
        
        try:
            from mlpipeline.utils.async_data import AsyncDataManager
            
            data_manager = AsyncDataManager(max_workers=4)
            
            # Test 1: Chargement parallèle de fichiers
            test_files = [
                "data/processed/features_BTCUSDT_1h.parquet",
                "data/processed/features_ETHUSDT_1h.parquet"
            ]
            
            # Filtrer les fichiers existants
            existing_files = [f for f in test_files if Path(f).exists()]
            
            if not existing_files:
                logger.warning("⚠️ Pas de fichiers de test trouvés")
                return {"status": "no_test_files"}
            
            # Benchmark chargement parallèle
            start_time = time.time()
            dfs = await data_manager.load_multiple_parquets(existing_files)
            parallel_load_time = time.time() - start_time
            
            # Benchmark chargement séquentiel (pour comparaison)
            start_time = time.time()
            sequential_dfs = []
            for file_path in existing_files:
                df = await data_manager.read_parquet_async(file_path)
                sequential_dfs.append(df)
            sequential_load_time = time.time() - start_time
            
            await data_manager.cleanup()
            
            io_metrics = {
                "parallel_load_time": parallel_load_time,
                "sequential_load_time": sequential_load_time,
                "speedup": sequential_load_time / parallel_load_time if parallel_load_time > 0 else 1,
                "files_loaded": len(dfs),
                "total_rows": sum(len(df) for df in dfs)
            }
            
            logger.info(f"✅ I/O async: {io_metrics['speedup']:.2f}x speedup")
            return io_metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur benchmark I/O async: {e}")
            return {"error": str(e)}
    
    def _calculate_success_rate(self, results: Dict) -> float:
        """
        Calcule le taux de succès des tâches
        """
        if not results:
            return 0.0
        
        successful = sum(1 for r in results.values() 
                        if isinstance(r, dict) and r.get('status') != 'error')
        total = len(results)
        
        return successful / total if total > 0 else 0.0
    
    async def run_full_benchmark(self) -> Dict:
        """
        Exécute le benchmark complet
        """
        logger.info("🏁 Début benchmark complet Async vs Sync")
        
        # Test 1: Pipeline async
        async_results = await self.benchmark_async_pipeline()
        self.results["async"]["pipeline"] = async_results
        
        # Test 2: Pipeline sync (simulation)
        sync_results = self.benchmark_sync_pipeline()
        self.results["sync"]["pipeline"] = sync_results
        
        # Test 3: Opérations I/O async
        io_results = await self.benchmark_async_data_operations()
        self.results["async"]["io_operations"] = io_results
        
        # Calcul de la comparaison
        self._calculate_comparison()
        
        # Sauvegarde des résultats
        await self._save_results()
        
        return self.results
    
    def _calculate_comparison(self):
        """
        Calcule les métriques de comparaison
        """
        async_time = self.results["async"]["pipeline"].get("execution_time", 0)
        sync_time = self.results["sync"]["pipeline"].get("execution_time", 0)
        
        if async_time > 0 and sync_time > 0:
            speedup = sync_time / async_time
            efficiency = (sync_time - async_time) / sync_time * 100
        else:
            speedup = 1.0
            efficiency = 0.0
        
        self.results["comparison"] = {
            "speedup": speedup,
            "efficiency_improvement": efficiency,
            "async_time": async_time,
            "sync_time": sync_time,
            "winner": "async" if speedup > 1.0 else "sync",
            "recommendation": self._get_recommendation(speedup, efficiency)
        }
    
    def _get_recommendation(self, speedup: float, efficiency: float) -> str:
        """
        Génère une recommandation basée sur les résultats
        """
        if speedup > 2.0:
            return "✅ ASYNC HAUTEMENT RECOMMANDÉ - Gain de performance significatif"
        elif speedup > 1.5:
            return "✅ ASYNC RECOMMANDÉ - Bon gain de performance"
        elif speedup > 1.1:
            return "⚖️ ASYNC LÉGÈREMENT AVANTAGEUX - Gain modéré"
        elif speedup > 0.9:
            return "🟡 PERFORMANCES ÉQUIVALENTES - Choisir selon architecture"
        else:
            return "⚠️ SYNC PLUS PERFORMANT - Revoir l'implémentation async"
    
    async def _save_results(self):
        """
        Sauvegarde les résultats du benchmark
        """
        try:
            results_dir = Path("benchmark_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"benchmark_async_vs_sync_{timestamp}.json"
            
            from mlpipeline.utils.async_data import AsyncDataManager
            data_manager = AsyncDataManager()
            
            await data_manager.write_json_async(self.results, str(results_file))
            await data_manager.cleanup()
            
            logger.info(f"💾 Résultats sauvegardés: {results_file}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
    
    def print_summary(self):
        """
        Affiche un résumé des résultats
        """
        print("\n" + "="*60)
        print("🏁 RÉSULTATS BENCHMARK ASYNC vs SYNC")
        print("="*60)
        
        comparison = self.results.get("comparison", {})
        async_pipeline = self.results.get("async", {}).get("pipeline", {})
        sync_pipeline = self.results.get("sync", {}).get("pipeline", {})
        io_ops = self.results.get("async", {}).get("io_operations", {})
        
        print(f"⏱️  Temps d'exécution:")
        print(f"   - Async: {async_pipeline.get('execution_time', 0):.2f}s")
        print(f"   - Sync:  {sync_pipeline.get('execution_time', 0):.2f}s")
        
        print(f"\n📈 Performance:")
        print(f"   - Speedup: {comparison.get('speedup', 1):.2f}x")
        print(f"   - Efficacité: {comparison.get('efficiency_improvement', 0):.1f}%")
        print(f"   - Gagnant: {comparison.get('winner', 'inconnu').upper()}")
        
        if io_ops.get('speedup'):
            print(f"\n💾 I/O Operations:")
            print(f"   - Speedup I/O: {io_ops['speedup']:.2f}x")
            print(f"   - Fichiers: {io_ops.get('files_loaded', 0)}")
        
        print(f"\n🎯 Recommandation:")
        print(f"   {comparison.get('recommendation', 'Pas de recommandation')}")
        
        print("="*60)


async def main():
    """
    Point d'entrée principal du benchmark
    """
    logger.info("🚀 Démarrage benchmark Async vs Sync")
    
    benchmark = PerformanceBenchmark()
    
    try:
        results = await benchmark.run_full_benchmark()
        benchmark.print_summary()
        
        logger.info("✅ Benchmark terminé avec succès")
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur benchmark: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # Exécution avec un seul asyncio.run()
    results = asyncio.run(main())
    
    if results:
        print("\n✅ Benchmark terminé. Vérifiez les logs pour les détails.")
    else:
        print("\n❌ Benchmark échoué. Vérifiez les logs pour les erreurs.")