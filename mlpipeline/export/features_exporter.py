#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features Exporter - Export features pour analyse externe
Exporte les features dans différents formats pour outils externes (Excel, CSV, Parquet, JSON)
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeaturesExporter:
    """Exporteur de features pour analyse externe"""

    def __init__(self, data_dir: str = "data", output_dir: str = "exports"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Répertoires de données
        self.processed_dir = self.data_dir / "processed"
        self.artifacts_dir = self.data_dir / "artifacts"
        self.raw_dir = self.data_dir / "raw"

        # Formats supportés
        self.supported_formats = ['csv', 'excel', 'parquet', 'json', 'feather']

        # Statistiques d'export
        self.export_stats = {
            "files_exported": 0,
            "total_rows": 0,
            "formats": [],
            "symbols": [],
            "timeframes": []
        }

    def discover_feature_files(self) -> Dict[str, List[Path]]:
        """Découvre tous les fichiers de features disponibles"""
        feature_files = {
            "processed_features": [],
            "raw_ohlcv": [],
            "ml_scores": [],
            "signals": [],
            "regimes": [],
            "metrics": []
        }

        # Features processées
        if self.processed_dir.exists():
            feature_files["processed_features"] = list(self.processed_dir.glob("features_*.parquet"))
            logger.info(f"📊 {len(feature_files['processed_features'])} fichiers features processées trouvés")

        # Données OHLCV brutes
        if self.raw_dir.exists():
            feature_files["raw_ohlcv"] = list(self.raw_dir.glob("ohlcv_*.parquet"))
            logger.info(f"📈 {len(feature_files['raw_ohlcv'])} fichiers OHLCV trouvés")

        # Artifacts (signaux, régimes, métriques)
        if self.artifacts_dir.exists():
            feature_files["signals"] = list(self.artifacts_dir.glob("*_signals_*.parquet"))
            feature_files["regimes"] = list(self.artifacts_dir.glob("regime_states_*.parquet"))
            feature_files["metrics"] = list(self.artifacts_dir.glob("*_metrics_*.json"))

            logger.info(f"📡 {len(feature_files['signals'])} fichiers signaux trouvés")
            logger.info(f"🎯 {len(feature_files['regimes'])} fichiers régimes trouvés")
            logger.info(f"📊 {len(feature_files['metrics'])} fichiers métriques trouvés")

        # Scores ML (si disponibles)
        ml_scores_dir = Path("freqtrade-prod/user_data/data/ml_scores")
        if ml_scores_dir.exists():
            feature_files["ml_scores"] = list(ml_scores_dir.glob("*.parquet"))
            logger.info(f"🤖 {len(feature_files['ml_scores'])} fichiers scores ML trouvés")

        return feature_files

    def load_and_enrich_features(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Charge et enrichit un fichier de features"""
        try:
            # Chargement selon le format
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            else:
                logger.warning(f"⚠️ Format non supporté: {file_path}")
                return None

            # Enrichissement avec métadonnées
            df_enriched = df.copy()

            # Ajouter informations de source
            df_enriched['_source_file'] = file_path.name
            df_enriched['_export_timestamp'] = datetime.now().isoformat()

            # Extraire symbole et timeframe du nom de fichier
            filename = file_path.stem
            symbol_match = self._extract_symbol_from_filename(filename)
            timeframe_match = self._extract_timeframe_from_filename(filename)

            if symbol_match:
                df_enriched['_symbol'] = symbol_match
                self.export_stats["symbols"].append(symbol_match)

            if timeframe_match:
                df_enriched['_timeframe'] = timeframe_match
                self.export_stats["timeframes"].append(timeframe_match)

            # Ajout de statistiques de base
            if len(df_enriched) > 0:
                numeric_cols = df_enriched.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df_enriched['_row_count'] = len(df_enriched)
                    df_enriched['_numeric_features'] = len(numeric_cols)

            logger.info(f"✅ Chargé: {file_path.name} ({len(df_enriched)} lignes, {len(df_enriched.columns)} colonnes)")
            return df_enriched

        except Exception as e:
            logger.error(f"❌ Erreur chargement {file_path}: {e}")
            return None

    def _extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """Extrait le symbole du nom de fichier"""
        # Patterns communs
        patterns = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
                   'SOLUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'AVAXUSDT']

        filename_upper = filename.upper()
        for pattern in patterns:
            if pattern in filename_upper:
                return pattern

        # Pattern générique *USDT
        import re
        match = re.search(r'([A-Z]+USDT)', filename_upper)
        if match:
            return match.group(1)

        return None

    def _extract_timeframe_from_filename(self, filename: str) -> Optional[str]:
        """Extrait le timeframe du nom de fichier"""
        filename_lower = filename.lower()

        # Patterns timeframe
        timeframe_patterns = {
            '1m': ['1m', '1min'],
            '5m': ['5m', '5min'],
            '15m': ['15m', '15min'],
            '1h': ['1h', 'hourly', 'hour'],
            '4h': ['4h', '4hour'],
            '1d': ['1d', 'daily', 'day'],
            '1w': ['1w', 'weekly']
        }

        for standard_tf, patterns in timeframe_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return standard_tf

        return None

    def export_to_format(self, df: pd.DataFrame, output_path: Path, format_type: str) -> bool:
        """Exporte le DataFrame dans le format spécifié"""
        try:
            if format_type == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif format_type == 'excel':
                df.to_excel(output_path, index=False, engine='openpyxl')
            elif format_type == 'parquet':
                df.to_parquet(output_path)
            elif format_type == 'json':
                df.to_json(output_path, orient='records', indent=2)
            elif format_type == 'feather':
                df.to_feather(output_path)
            else:
                logger.error(f"❌ Format non supporté: {format_type}")
                return False

            logger.info(f"✅ Export {format_type}: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur export {format_type} vers {output_path}: {e}")
            return False

    def create_summary_report(self, feature_files: Dict[str, List[Path]]) -> pd.DataFrame:
        """Crée un rapport de synthèse des features disponibles"""
        summary_data = []

        for category, files in feature_files.items():
            for file_path in files:
                try:
                    # Info de base
                    file_info = {
                        'category': category,
                        'filename': file_path.name,
                        'file_path': str(file_path),
                        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                        'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }

                    # Analyse du contenu pour parquet/csv
                    if file_path.suffix in ['.parquet', '.csv']:
                        try:
                            if file_path.suffix == '.parquet':
                                df = pd.read_parquet(file_path)
                            else:
                                df = pd.read_csv(file_path, nrows=1000)  # Sample pour les gros fichiers

                            file_info.update({
                                'rows': len(df),
                                'columns': len(df.columns),
                                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                                'date_columns': len(df.select_dtypes(include=['datetime64']).columns),
                                'has_ohlcv': all(col in df.columns for col in ['open', 'high', 'low', 'close']),
                                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                            })

                        except Exception as e:
                            file_info['analysis_error'] = str(e)

                    # Extraction métadonnées
                    symbol = self._extract_symbol_from_filename(file_path.stem)
                    timeframe = self._extract_timeframe_from_filename(file_path.stem)

                    if symbol:
                        file_info['symbol'] = symbol
                    if timeframe:
                        file_info['timeframe'] = timeframe

                    summary_data.append(file_info)

                except Exception as e:
                    logger.warning(f"⚠️ Erreur analyse {file_path}: {e}")
                    summary_data.append({
                        'category': category,
                        'filename': file_path.name,
                        'error': str(e)
                    })

        return pd.DataFrame(summary_data)

    def export_features(self, formats: List[str] = None,
                       symbols: List[str] = None,
                       categories: List[str] = None,
                       include_summary: bool = True) -> Dict[str, Any]:
        """Exporte les features selon les critères spécifiés"""

        if formats is None:
            formats = ['csv', 'excel']

        # Validation des formats
        invalid_formats = [f for f in formats if f not in self.supported_formats]
        if invalid_formats:
            logger.error(f"❌ Formats non supportés: {invalid_formats}")
            return {"success": False, "error": f"Formats invalides: {invalid_formats}"}

        logger.info(f"🚀 Début export features...")
        logger.info(f"📋 Formats: {formats}")
        if symbols:
            logger.info(f"🎯 Symboles filtrés: {symbols}")
        if categories:
            logger.info(f"📂 Catégories filtrées: {categories}")

        # Découverte des fichiers
        feature_files = self.discover_feature_files()

        # Filtrage par catégories
        if categories:
            feature_files = {k: v for k, v in feature_files.items() if k in categories}

        # Création du répertoire de sortie avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.output_dir / f"features_export_{timestamp}"
        export_dir.mkdir(exist_ok=True)

        export_results = {
            "success": True,
            "export_dir": str(export_dir),
            "timestamp": timestamp,
            "exported_files": [],
            "errors": []
        }

        # Export par catégorie
        for category, files in feature_files.items():
            if not files:
                continue

            category_dir = export_dir / category
            category_dir.mkdir(exist_ok=True)

            logger.info(f"📂 Traitement catégorie: {category} ({len(files)} fichiers)")

            for file_path in files:
                # Filtrage par symbole
                if symbols:
                    file_symbol = self._extract_symbol_from_filename(file_path.stem)
                    if file_symbol and file_symbol not in symbols:
                        continue

                # Chargement et enrichissement
                df = self.load_and_enrich_features(file_path)
                if df is None:
                    export_results["errors"].append(f"Échec chargement: {file_path}")
                    continue

                # Export dans les formats demandés
                base_name = file_path.stem
                for format_type in formats:
                    if format_type == 'excel':
                        output_path = category_dir / f"{base_name}.xlsx"
                    else:
                        output_path = category_dir / f"{base_name}.{format_type}"

                    if self.export_to_format(df, output_path, format_type):
                        export_results["exported_files"].append(str(output_path))
                        self.export_stats["files_exported"] += 1
                        self.export_stats["total_rows"] += len(df)

                        if format_type not in self.export_stats["formats"]:
                            self.export_stats["formats"].append(format_type)

        # Génération du rapport de synthèse
        if include_summary:
            logger.info("📊 Génération rapport de synthèse...")
            summary_df = self.create_summary_report(feature_files)

            for format_type in formats:
                if format_type == 'excel':
                    summary_path = export_dir / "features_summary.xlsx"
                else:
                    summary_path = export_dir / f"features_summary.{format_type}"

                if self.export_to_format(summary_df, summary_path, format_type):
                    export_results["exported_files"].append(str(summary_path))

        # Sauvegarde des statistiques d'export
        stats_path = export_dir / "export_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.export_stats, f, indent=2)

        export_results["statistics"] = self.export_stats
        logger.info(f"✅ Export terminé: {len(export_results['exported_files'])} fichiers")

        return export_results

    def display_export_summary(self, export_results: Dict[str, Any]):
        """Affiche le résumé de l'export"""
        print("\n" + "="*70)
        print("📤 FEATURES EXPORT SUMMARY")
        print("="*70)

        print(f"📁 Répertoire: {export_results.get('export_dir', 'N/A')}")
        print(f"🕐 Timestamp: {export_results.get('timestamp', 'N/A')}")
        print(f"✅ Succès: {export_results.get('success', False)}")

        stats = export_results.get("statistics", {})
        print(f"\n📊 STATISTIQUES:")
        print(f"  Fichiers exportés: {stats.get('files_exported', 0)}")
        print(f"  Total lignes: {stats.get('total_rows', 0):,}")
        print(f"  Formats: {', '.join(stats.get('formats', []))}")
        print(f"  Symboles: {', '.join(set(stats.get('symbols', [])))}")
        print(f"  Timeframes: {', '.join(set(stats.get('timeframes', [])))}")

        if export_results.get("errors"):
            print(f"\n❌ ERREURS ({len(export_results['errors'])}):")
            for error in export_results['errors'][:5]:  # Limiter l'affichage
                print(f"  - {error}")
            if len(export_results['errors']) > 5:
                print(f"  ... et {len(export_results['errors']) - 5} autres")

        print("\n" + "="*70)

def main():
    """Point d'entrée principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Export features pour analyse externe")
    parser.add_argument("--formats", nargs="+", default=["csv", "excel"],
                       choices=["csv", "excel", "parquet", "json", "feather"],
                       help="Formats d'export")
    parser.add_argument("--symbols", nargs="+",
                       help="Filtrer par symboles (ex: BTCUSDT ETHUSDT)")
    parser.add_argument("--categories", nargs="+",
                       choices=["processed_features", "raw_ohlcv", "signals", "regimes", "metrics", "ml_scores"],
                       help="Filtrer par catégories")
    parser.add_argument("--no-summary", action="store_true",
                       help="Ne pas générer le rapport de synthèse")

    args = parser.parse_args()

    try:
        exporter = FeaturesExporter()

        # Export
        results = exporter.export_features(
            formats=args.formats,
            symbols=args.symbols,
            categories=args.categories,
            include_summary=not args.no_summary
        )

        # Affichage résumé
        exporter.display_export_summary(results)

        if results["success"]:
            print(f"\n✅ Export réussi vers: {results['export_dir']}")
            return 0
        else:
            print(f"\n❌ Échec export: {results.get('error', 'Erreur inconnue')}")
            return 1

    except Exception as e:
        logger.error(f"❌ Erreur export: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())