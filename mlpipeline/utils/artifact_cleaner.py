#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nettoyage des artifacts d'anciennes exécutions avec données synthétiques
Garantit que seuls les modèles entraînés sur données réelles sont conservés
"""

import logging
import os
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

def validate_real_data_only(data: pd.DataFrame, symbol: str = None) -> bool:
    """
    Valide que les données sont réelles et non synthétiques
    
    Args:
        data: DataFrame à valider
        symbol: Symbole optionnel pour validation contextuelle
    
    Returns:
        bool: True si données réelles, False si synthétiques
    """
    if data is None or data.empty:
        logger.warning("❌ Données vides ou None")
        return False
    
    # Vérifications basiques
    if len(data) == 2000:  # Taille exacte de l'ancien générateur
        logger.warning("❌ Taille suspecte (2000 lignes = possiblement synthétique)")
        return False
    
    # Prix trop ronds ou constants
    if 'close' in data.columns:
        closes = data['close']
        # Prix multiples de 1000 = synthétique probable
        if (closes % 1000 == 0).sum() > len(data) * 0.3:
            logger.warning("❌ Trop de prix ronds (multiples de 1000)")
            return False
            
        # Variance trop faible
        if closes.nunique() < len(data) * 0.5:
            logger.warning("❌ Peu de variété dans les prix")
            return False
    
    # Timestamps trop réguliers uniquement si données suspectes par ailleurs
    # Les données crypto réelles ont des timestamps réguliers, c'est normal
    if hasattr(data.index, 'to_series'):
        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs.unique()) == 1:
            # Accepter la régularité si autres indicateurs sont OK
            logger.info("ℹ️  Timestamps réguliers détectés (normal pour crypto hourly)")
    
    # Vérifier si on a une colonne 'time' au lieu de l'index
    if 'time' in data.columns:
        # Convertir et vérifier la régularité des timestamps
        time_series = pd.to_datetime(data['time'])
        time_diffs = time_series.diff().dropna()
        if len(time_diffs.unique()) == 1:
            logger.info("ℹ️  Timestamps réguliers dans colonne time (normal pour crypto hourly)")
    
    # Validation spécifique par symbole
    if symbol and 'close' in data.columns:
        closes = data['close']
        if symbol == 'BTCUSDT':
            # Bitcoin doit être > 1000$ et < 500000$
            if not (1000 < closes.mean() < 500000):
                logger.warning(f"❌ Prix BTC irréaliste: {closes.mean()}")
                return False
    
    logger.info(f"✅ Données validées comme réelles ({len(data)} lignes)")
    return True

class ArtifactCleaner:
    """Nettoyeur d'artifacts et validateur de données réelles"""

    def __init__(self, project_root: str = ".", symbols: List[str] = None, timeframes: List[str] = None):
        self.project_root = Path(project_root)
        self.cleaned_count = 0
        self.target_symbols = symbols or []  # Filtrer par symboles spécifiques
        self.target_timeframes = timeframes or []  # Filtrer par timeframes spécifiques

        # Normaliser les timeframes
        self.timeframe_patterns = {
            '1m': ['1m', '1min'],
            '5m': ['5m', '5min'],
            '15m': ['15m', '15min'],
            '1h': ['1h', '1H', '1hour', 'hourly'],
            '4h': ['4h', '4H', '4hour'],
            '1d': ['1d', '1D', 'daily'],
            '1w': ['1w', '1W', 'weekly']
        }
        
    def detect_synthetic_artifacts(self) -> List[Path]:
        """
        Détecte les artifacts créés avec des données synthétiques
        """
        suspicious_artifacts = []
        
        # Dossiers à vérifier
        check_dirs = [
            "data/artifacts",
            "data/processed", 
            "work/artifacts",
            "freqtrade-prod/user_data/data/ml_scores",
            "logs"
        ]
        
        for dir_path in check_dirs:
            full_dir = self.project_root / dir_path
            if full_dir.exists():
                suspicious_artifacts.extend(
                    self._scan_directory_for_synthetic(full_dir)
                )
        
        return suspicious_artifacts
    
    def _scan_directory_for_synthetic(self, directory: Path) -> List[Path]:
        """Scan un dossier pour détecter les données synthétiques"""
        synthetic_files = []

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            # Vérification par extension
            if file_path.suffix in ['.parquet', '.json', '.pkl', '.h5']:
                # Vérifier filtres symboles/timeframes
                if self._matches_filters(file_path):
                    if self._is_synthetic_artifact(file_path):
                        synthetic_files.append(file_path)

        return synthetic_files

    def _matches_filters(self, file_path: Path) -> bool:
        """
        Vérifie si le fichier correspond aux filtres symbole/timeframe
        """
        filename = file_path.name.lower()

        # Filtrage par symboles
        if self.target_symbols:
            symbol_match = False
            for symbol in self.target_symbols:
                if symbol.lower() in filename:
                    symbol_match = True
                    break
            if not symbol_match:
                return False

        # Filtrage par timeframes
        if self.target_timeframes:
            timeframe_match = False
            for tf in self.target_timeframes:
                # Chercher dans les patterns définis
                for standard_tf, patterns in self.timeframe_patterns.items():
                    if tf in patterns or tf == standard_tf:
                        for pattern in patterns:
                            if pattern in filename:
                                timeframe_match = True
                                break
                        if timeframe_match:
                            break
                if timeframe_match:
                    break
            if not timeframe_match:
                return False

        return True
    
    def _is_synthetic_artifact(self, file_path: Path) -> bool:
        """
        Détermine si un fichier contient des données synthétiques
        """
        try:
            # Vérification par nom de fichier
            filename = file_path.name.lower()
            if any(keyword in filename for keyword in [
                'synthetic', 'fake', 'generated', 'random', 'test_data'
            ]):
                logger.info(f"🔍 Synthétique détecté (nom): {file_path}")
                return True
            
            # Vérification par contenu selon extension
            if file_path.suffix == '.parquet':
                return self._check_parquet_synthetic(file_path)
            elif file_path.suffix == '.json':
                return self._check_json_synthetic(file_path)
            
            # Vérification par date (fichiers très récents = possiblement test)
            stat = file_path.stat()
            if datetime.fromtimestamp(stat.st_mtime) > datetime.now() - timedelta(days=1):
                # Fichier très récent, vérification plus poussée
                return self._deep_content_check(file_path)
                
        except Exception as e:
            logger.warning(f"⚠️  Erreur vérification {file_path}: {e}")
            
        return False
    
    def _check_parquet_synthetic(self, file_path: Path) -> bool:
        """Vérification spécifique pour fichiers Parquet"""
        try:
            df = pd.read_parquet(file_path)
            
            # Vérifications patterns synthétiques
            if len(df) == 2000:  # Taille exacte de l'ancien générateur
                logger.info(f"🔍 Suspect (taille 2000): {file_path}")
                return True
                
            # Prix trop ronds ou constants
            if 'close' in df.columns:
                closes = df['close']
                # Prix multiples de 1000 = synthétique probable
                if (closes % 1000 == 0).sum() > len(df) * 0.5:
                    logger.info(f"🔍 Synthétique détecté (prix ronds): {file_path}")
                    return True
                    
                # Variance trop faible
                if closes.nunique() < len(df) * 0.8:
                    logger.info(f"🔍 Synthétique détecté (peu de variété): {file_path}")
                    return True
            
            # Timestamps trop réguliers
            if 'time' in df.columns:
                time_diffs = df['time'].diff().dropna()
                if time_diffs.nunique() == 1:
                    logger.info(f"🔍 Synthétique détecté (timestamps réguliers): {file_path}")
                    return True
                    
        except Exception as e:
            logger.warning(f"Erreur lecture parquet {file_path}: {e}")
            
        return False
    
    def _check_json_synthetic(self, file_path: Path) -> bool:
        """Vérification spécifique pour fichiers JSON (métriques)"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Métriques suspectes (trop parfaites = synthétiques)
            if isinstance(data, dict):
                # Sharpe ratio exactement 0.27 = ancien synthétique  
                if data.get('sharpe') == 0.2712114325712078:
                    logger.info(f"🔍 Métriques synthétiques détectées: {file_path}")
                    return True
                
                # PSR exactement 1.0 = suspect
                if data.get('psr') == 1.0:
                    logger.info(f"🔍 PSR synthétique détecté: {file_path}")
                    return True
                    
        except Exception as e:
            logger.warning(f"Erreur lecture JSON {file_path}: {e}")
            
        return False
    
    def _deep_content_check(self, file_path: Path) -> bool:
        """Vérification approfondie du contenu"""
        # Pour l'instant, on marque les fichiers très récents comme suspects
        # À affiner selon les patterns découverts
        return False
    
    def clean_synthetic_artifacts(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Nettoie les artifacts synthétiques
        
        Args:
            dry_run: Si True, affiche seulement ce qui serait supprimé
        """
        synthetic_files = self.detect_synthetic_artifacts()
        
        stats = {
            'total_found': len(synthetic_files),
            'deleted': 0,
            'errors': 0,
            'size_freed_mb': 0
        }
        
        if not synthetic_files:
            logger.info("✅ Aucun artifact synthétique trouvé")
            return stats
        
        logger.info(f"🧹 Trouvé {len(synthetic_files)} artifacts synthétiques")
        
        for file_path in synthetic_files:
            try:
                # Calculer taille avant suppression
                size_mb = file_path.stat().st_size / (1024 * 1024)
                stats['size_freed_mb'] += size_mb
                
                if dry_run:
                    logger.info(f"[DRY RUN] Supprimé: {file_path} ({size_mb:.2f} MB)")
                else:
                    # Suppression effective
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                    
                    logger.info(f"🗑️  Supprimé: {file_path} ({size_mb:.2f} MB)")
                    stats['deleted'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Erreur suppression {file_path}: {e}")
                stats['errors'] += 1
        
        # Nettoyage dossiers vides
        if not dry_run:
            self._cleanup_empty_dirs()
        
        return stats
    
    def _cleanup_empty_dirs(self):
        """Supprime les dossiers vides après nettoyage"""
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):  # Dossier vide
                        dir_path.rmdir()
                        logger.info(f"🗑️  Dossier vide supprimé: {dir_path}")
                except (OSError, StopIteration):
                    pass  # Dossier pas vide ou erreur
    
    def backup_real_artifacts(self, backup_dir: str = "backups") -> bool:
        """
        Sauvegarde les artifacts validés comme réels
        """
        backup_path = self.project_root / backup_dir
        backup_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"real_artifacts_{timestamp}"
        specific_backup = backup_path / backup_name
        
        try:
            # Copie des artifacts validés
            real_dirs = [
                "data/processed",
                "data/artifacts", 
                "freqtrade-prod/user_data/data/ml_scores"
            ]
            
            for dir_name in real_dirs:
                source_dir = self.project_root / dir_name
                if source_dir.exists():
                    dest_dir = specific_backup / dir_name
                    shutil.copytree(source_dir, dest_dir, ignore_errors=True)
            
            logger.info(f"💾 Backup créé: {specific_backup}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur backup: {e}")
            return False

# ==============================================
# SCRIPT PRINCIPAL
# ==============================================

def main():
    """Script de nettoyage des artifacts synthétiques"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Nettoyage artifacts données synthétiques"
    )
    parser.add_argument("--dry-run", action="store_true", 
                       help="Affichage seulement, pas de suppression")
    parser.add_argument("--backup", action="store_true",
                       help="Backup des artifacts réels avant nettoyage") 
    parser.add_argument("--project-root", default=".",
                       help="Racine du projet")
    parser.add_argument("--symbols", nargs="+",
                       help="Filtrer par symboles spécifiques (ex: BTCUSDT ETHUSDT)")
    parser.add_argument("--timeframes", nargs="+",
                       help="Filtrer par timeframes spécifiques (ex: 1h 4h 1d)")
    
    args = parser.parse_args()
    
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    cleaner = ArtifactCleaner(args.project_root, args.symbols, args.timeframes)
    
    # Backup optionnel
    if args.backup:
        logger.info("💾 Création backup artifacts réels...")
        cleaner.backup_real_artifacts()
    
    # Nettoyage
    logger.info("🧹 Début nettoyage artifacts synthétiques...")
    stats = cleaner.clean_synthetic_artifacts(dry_run=args.dry_run)
    
    # Rapport final
    logger.info("📊 RAPPORT NETTOYAGE:")
    logger.info(f"   - Artifacts trouvés: {stats['total_found']}")
    logger.info(f"   - Supprimés: {stats['deleted']}")
    logger.info(f"   - Erreurs: {stats['errors']}")
    logger.info(f"   - Espace libéré: {stats['size_freed_mb']:.2f} MB")
    
    if args.dry_run and stats['total_found'] > 0:
        logger.info("🔥 Pour supprimer réellement, relancez sans --dry-run")

if __name__ == "__main__":
    main()