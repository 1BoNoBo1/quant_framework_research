#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Data Management - Gestion asynchrone des données
Optimise les opérations I/O pour le pipeline ML
"""

import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

logger = logging.getLogger(__name__)

class AsyncDataManager:
    """
    Gestionnaire de données asynchrone pour optimiser les I/O
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
        self._cache = {}
    
    async def read_parquet_async(self, file_path: str) -> pd.DataFrame:
        """
        Lecture async d'un fichier parquet
        """
        if file_path in self._cache:
            logger.debug(f"📋 Cache hit: {file_path}")
            return self._cache[file_path].copy()
        
        logger.info(f"📊 Loading parquet async: {file_path}")
        
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            self.thread_executor,
            pd.read_parquet,
            file_path
        )
        
        # Cache avec limite de taille
        if len(self._cache) < 10:  # Limite à 10 fichiers en cache
            self._cache[file_path] = df.copy()
        
        logger.info(f"✅ Loaded: {len(df)} rows from {Path(file_path).name}")
        return df
    
    async def write_parquet_async(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Écriture async d'un fichier parquet
        """
        logger.info(f"💾 Saving parquet async: {file_path}")
        
        # Créer le répertoire si nécessaire
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.thread_executor,
            lambda: df.to_parquet(file_path, compression='snappy')
        )
        
        logger.info(f"✅ Saved: {len(df)} rows to {Path(file_path).name}")
    
    async def read_json_async(self, file_path: str) -> Dict:
        """
        Lecture async d'un fichier JSON
        """
        logger.debug(f"📄 Loading JSON async: {file_path}")
        
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            return json.loads(content)
    
    async def write_json_async(self, data: Dict, file_path: str) -> None:
        """
        Écriture async d'un fichier JSON
        """
        logger.debug(f"💾 Saving JSON async: {file_path}")
        
        # Créer le répertoire si nécessaire
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))
    
    async def load_multiple_parquets(self, file_paths: List[str]) -> List[pd.DataFrame]:
        """
        Chargement parallèle de plusieurs fichiers parquet
        """
        logger.info(f"📊 Loading {len(file_paths)} parquet files in parallel")
        
        tasks = [self.read_parquet_async(path) for path in file_paths]
        dfs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les erreurs
        valid_dfs = []
        for i, result in enumerate(dfs):
            if isinstance(result, Exception):
                logger.error(f"❌ Failed to load {file_paths[i]}: {result}")
            else:
                valid_dfs.append(result)
        
        logger.info(f"✅ Successfully loaded {len(valid_dfs)}/{len(file_paths)} files")
        return valid_dfs
    
    async def save_multiple_parquets(self, df_path_pairs: List[Tuple[pd.DataFrame, str]]) -> None:
        """
        Sauvegarde parallèle de plusieurs fichiers parquet
        """
        logger.info(f"💾 Saving {len(df_path_pairs)} parquet files in parallel")
        
        tasks = [
            self.write_parquet_async(df, path) 
            for df, path in df_path_pairs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compter les succès
        successes = sum(1 for r in results if not isinstance(r, Exception))
        errors = sum(1 for r in results if isinstance(r, Exception))
        
        if errors > 0:
            logger.warning(f"⚠️ {errors} files failed to save")
        
        logger.info(f"✅ Successfully saved {successes}/{len(df_path_pairs)} files")
    
    async def process_dataframe_async(self, 
                                    df: pd.DataFrame, 
                                    processing_func,
                                    *args, **kwargs) -> pd.DataFrame:
        """
        Traitement async d'un DataFrame avec fonction CPU-intensive
        """
        logger.debug(f"⚙️ Processing dataframe async: {len(df)} rows")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.process_executor,
            processing_func,
            df,
            *args
        )
        
        logger.debug(f"✅ Processing completed: {len(result)} rows")
        return result
    
    async def validate_data_async(self, df: pd.DataFrame, source: str = "unknown") -> bool:
        """
        Validation asynchrone des données
        """
        logger.debug(f"🔍 Validating data async: {source}")
        
        try:
            # Import validation function
            from mlpipeline.utils.artifact_cleaner import validate_real_data_only
            
            loop = asyncio.get_event_loop()
            is_valid = await loop.run_in_executor(
                self.thread_executor,
                validate_real_data_only,
                df,
                source
            )
            
            if is_valid:
                logger.debug(f"✅ Data validation passed: {source}")
            else:
                logger.warning(f"⚠️ Data validation failed: {source}")
            
            return is_valid
            
        except ImportError:
            logger.debug("📋 Validation function not available, skipping")
            return True
        except Exception as e:
            logger.error(f"❌ Validation error for {source}: {e}")
            return False
    
    async def cleanup(self):
        """
        Nettoyage des ressources async
        """
        logger.info("🧹 Cleaning up async resources")
        
        self.thread_executor.shutdown(wait=False)
        self.process_executor.shutdown(wait=False)
        self._cache.clear()
        
        logger.info("✅ Async cleanup completed")


class AsyncFeatureProcessor:
    """
    Processeur de features asynchrone
    """
    
    def __init__(self, data_manager: AsyncDataManager):
        self.data_manager = data_manager
    
    async def process_features_batch(self, 
                                   input_files: List[str],
                                   output_dir: str,
                                   processing_func) -> List[str]:
        """
        Traitement par batch des features en parallèle
        """
        logger.info(f"🔄 Processing {len(input_files)} feature files in batch")
        
        # Chargement parallèle
        dfs = await self.data_manager.load_multiple_parquets(input_files)
        
        # Traitement parallèle
        tasks = []
        output_files = []
        
        for i, df in enumerate(dfs):
            input_name = Path(input_files[i]).stem
            output_path = f"{output_dir}/processed_{input_name}.parquet"
            output_files.append(output_path)
            
            # Traitement async
            task = self.data_manager.process_dataframe_async(df, processing_func)
            tasks.append(task)
        
        processed_dfs = await asyncio.gather(*tasks)
        
        # Sauvegarde parallèle
        save_pairs = list(zip(processed_dfs, output_files))
        await self.data_manager.save_multiple_parquets(save_pairs)
        
        logger.info(f"✅ Batch processing completed: {len(output_files)} files")
        return output_files


async def create_async_data_pipeline(config: Dict) -> AsyncDataManager:
    """
    Factory pour créer un pipeline de données async
    """
    max_workers = config.get('max_workers', 4)
    
    logger.info(f"🚀 Creating async data pipeline (workers={max_workers})")
    
    data_manager = AsyncDataManager(max_workers=max_workers)
    
    return data_manager


async def example_usage():
    """
    Exemple d'utilisation du système async
    """
    logger.info("🔄 Demo async data management")
    
    # Configuration
    config = {'max_workers': 4}
    
    # Création du manager
    data_manager = await create_async_data_pipeline(config)
    
    try:
        # Exemple: chargement parallèle
        files = [
            "data/processed/features_BTCUSDT_1h.parquet",
            "data/processed/features_ETHUSDT_1h.parquet"
        ]
        
        # Filtrer les fichiers existants
        existing_files = [f for f in files if Path(f).exists()]
        
        if existing_files:
            dfs = await data_manager.load_multiple_parquets(existing_files)
            logger.info(f"📊 Loaded {len(dfs)} dataframes")
            
            # Validation parallèle
            validation_tasks = [
                data_manager.validate_data_async(df, f"file_{i}") 
                for i, df in enumerate(dfs)
            ]
            validations = await asyncio.gather(*validation_tasks)
            
            valid_count = sum(validations)
            logger.info(f"✅ {valid_count}/{len(dfs)} files passed validation")
        
        else:
            logger.info("📋 No files found for demo")
    
    finally:
        # Nettoyage
        await data_manager.cleanup()


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Lancer la démo
    asyncio.run(example_usage())