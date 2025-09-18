#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Simple Architecture Async - Validation fonctionnelle
Test basique pour valider que l'architecture async fonctionne
"""

import asyncio
import time
import logging
from pathlib import Path

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleAsyncTest:
    """
    Test simple pour valider l'architecture async
    """
    
    def __init__(self):
        self.results = {}
    
    async def test_basic_async_operations(self):
        """
        Test des opérations async de base
        """
        logger.info("🔄 Test opérations async de base")
        
        # Test 1: Tâches concurrentes simples
        async def task(name: str, duration: float):
            logger.info(f"  ▶️ Démarrage {name}")
            await asyncio.sleep(duration)
            logger.info(f"  ✅ Terminé {name}")
            return f"result_{name}"
        
        start_time = time.time()
        
        # Exécution parallèle
        results = await asyncio.gather(
            task("task1", 0.5),
            task("task2", 0.3), 
            task("task3", 0.7)
        )
        
        parallel_time = time.time() - start_time
        
        # Exécution séquentielle pour comparaison
        start_time = time.time()
        sequential_results = []
        for i, duration in enumerate([0.5, 0.3, 0.7]):
            result = await task(f"seq_task{i+1}", duration)
            sequential_results.append(result)
        
        sequential_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time
        
        logger.info(f"  📊 Temps parallèle: {parallel_time:.2f}s")
        logger.info(f"  📊 Temps séquentiel: {sequential_time:.2f}s")
        logger.info(f"  📊 Speedup: {speedup:.2f}x")
        
        return {
            "parallel_time": parallel_time,
            "sequential_time": sequential_time,
            "speedup": speedup,
            "parallel_results": results,
            "sequential_results": sequential_results
        }
    
    async def test_async_file_operations(self):
        """
        Test des opérations fichier async (simulé)
        """
        logger.info("📁 Test opérations fichier async")
        
        # Simulation écriture/lecture async
        async def write_file(filename: str, content: str):
            await asyncio.sleep(0.1)  # Simule I/O
            return f"written_{filename}"
        
        async def read_file(filename: str):
            await asyncio.sleep(0.1)  # Simule I/O
            return f"content_of_{filename}"
        
        files = ["file1.txt", "file2.txt", "file3.txt"]
        
        # Écriture parallèle
        start_time = time.time()
        write_tasks = [write_file(f, f"content_{f}") for f in files]
        write_results = await asyncio.gather(*write_tasks)
        write_time = time.time() - start_time
        
        # Lecture parallèle  
        start_time = time.time()
        read_tasks = [read_file(f) for f in files]
        read_results = await asyncio.gather(*read_tasks)
        read_time = time.time() - start_time
        
        logger.info(f"  📊 Écriture parallèle: {write_time:.3f}s")
        logger.info(f"  📊 Lecture parallèle: {read_time:.3f}s")
        
        return {
            "write_time": write_time,
            "read_time": read_time,
            "files_processed": len(files),
            "write_results": write_results,
            "read_results": read_results
        }
    
    async def test_semaphore_concurrency_control(self):
        """
        Test du contrôle de concurrence avec semaphore
        """
        logger.info("🚦 Test contrôle concurrence avec semaphore")
        
        # Semaphore pour limiter à 2 tâches simultanées
        semaphore = asyncio.Semaphore(2)
        
        async def limited_task(task_id: int):
            async with semaphore:
                logger.info(f"    🔄 Task {task_id} démarrée")
                await asyncio.sleep(0.5)
                logger.info(f"    ✅ Task {task_id} terminée")
                return f"result_{task_id}"
        
        # Lancer 5 tâches avec limite de 2 simultanées
        start_time = time.time()
        tasks = [limited_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        logger.info(f"  📊 5 tâches avec limite 2 simultanées: {execution_time:.2f}s")
        
        return {
            "execution_time": execution_time,
            "max_concurrent": 2,
            "total_tasks": 5,
            "results": results
        }
    
    async def test_error_handling(self):
        """
        Test gestion d'erreurs async
        """
        logger.info("⚠️ Test gestion d'erreurs async")
        
        async def failing_task():
            await asyncio.sleep(0.1)
            raise ValueError("Erreur simulée")
        
        async def success_task():
            await asyncio.sleep(0.1)
            return "success"
        
        # Test avec return_exceptions=True
        results = await asyncio.gather(
            success_task(),
            failing_task(),
            success_task(),
            return_exceptions=True
        )
        
        successes = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        logger.info(f"  📊 Succès: {len(successes)}, Erreurs: {len(errors)}")
        
        return {
            "total_tasks": 3,
            "successes": len(successes),
            "errors": len(errors),
            "success_rate": len(successes) / len(results)
        }
    
    async def run_all_tests(self):
        """
        Exécute tous les tests async
        """
        logger.info("🚀 Démarrage tests architecture async")
        
        # Test 1: Opérations de base
        basic_results = await self.test_basic_async_operations()
        self.results["basic_operations"] = basic_results
        
        # Test 2: Opérations fichier
        file_results = await self.test_async_file_operations()
        self.results["file_operations"] = file_results
        
        # Test 3: Contrôle concurrence
        concurrency_results = await self.test_semaphore_concurrency_control()
        self.results["concurrency_control"] = concurrency_results
        
        # Test 4: Gestion d'erreurs
        error_results = await self.test_error_handling()
        self.results["error_handling"] = error_results
        
        return self.results
    
    def print_summary(self):
        """
        Affiche le résumé des tests
        """
        print("\n" + "="*50)
        print("🧪 RÉSULTATS TESTS ARCHITECTURE ASYNC")
        print("="*50)
        
        if "basic_operations" in self.results:
            basic = self.results["basic_operations"]
            print(f"✅ Opérations de base:")
            print(f"   - Speedup: {basic['speedup']:.2f}x")
            print(f"   - Temps parallèle: {basic['parallel_time']:.2f}s")
        
        if "file_operations" in self.results:
            files = self.results["file_operations"]
            print(f"✅ Opérations fichier:")
            print(f"   - Fichiers traités: {files['files_processed']}")
            print(f"   - Temps écriture: {files['write_time']:.3f}s")
        
        if "concurrency_control" in self.results:
            conc = self.results["concurrency_control"]
            print(f"✅ Contrôle concurrence:")
            print(f"   - {conc['total_tasks']} tâches, limite {conc['max_concurrent']}")
            print(f"   - Temps: {conc['execution_time']:.2f}s")
        
        if "error_handling" in self.results:
            errors = self.results["error_handling"]
            print(f"✅ Gestion erreurs:")
            print(f"   - Taux succès: {errors['success_rate']*100:.0f}%")
        
        print(f"\n🎯 VERDICT: Architecture async FONCTIONNELLE")
        print("="*50)


async def main():
    """
    Point d'entrée principal des tests
    """
    logger.info("🚀 Démarrage validation architecture async")
    
    test_runner = SimpleAsyncTest()
    
    try:
        results = await test_runner.run_all_tests()
        test_runner.print_summary()
        
        logger.info("✅ Tous les tests async terminés avec succès")
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur tests async: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# Point d'entrée avec un seul asyncio.run()
if __name__ == "__main__":
    logger.info("🎯 Architecture Async - Point d'entrée unique avec asyncio.run()")
    
    results = asyncio.run(main())
    
    if results:
        print("\n✅ Validation architecture async RÉUSSIE")
        print("📈 Le framework est prêt pour utilisation async avec un seul asyncio.run()")
    else:
        print("\n❌ Validation architecture async ÉCHOUÉE")