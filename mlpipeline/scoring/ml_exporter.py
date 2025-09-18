#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Exporter - Score and Signal Export for ML Pipeline
Exports alpha signals, metrics, and predictions for downstream ML consumption
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc

# Configuration paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLExporter:
    """
    Exports alpha signals and metrics for ML pipeline consumption
    """
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.artifacts_dir = Path("data/artifacts")
        self.export_dir = Path("data/ml_exports")
        self.export_dir.mkdir(exist_ok=True)
        
    def export_alpha_signals(self) -> Dict[str, Any]:
        """
        Export all alpha signals for the symbol/timeframe
        """
        logger.info(f"📊 Exporting alpha signals for {self.symbol}_{self.timeframe}")
        
        exported_signals = {}
        
        # Alpha types to export
        alpha_types = ['dmn', 'mr', 'funding']
        
        for alpha_type in alpha_types:
            signals_file = self.artifacts_dir / f"{alpha_type}_signals_{self.symbol}.parquet"
            metrics_file = self.artifacts_dir / f"{alpha_type}_metrics_{self.symbol}.json"
            
            alpha_data = {}
            
            # Load signals if available
            if signals_file.exists():
                try:
                    signals_df = pd.read_parquet(signals_file)
                    alpha_data['signals'] = {
                        'shape': signals_df.shape,
                        'columns': signals_df.columns.tolist(),
                        'date_range': [str(signals_df.index.min()), str(signals_df.index.max())],
                        'signal_stats': {
                            'mean_signal': float(signals_df['signal'].mean()) if 'signal' in signals_df.columns else 0.0,
                            'signal_std': float(signals_df['signal'].std()) if 'signal' in signals_df.columns else 0.0,
                            'positive_signals': int((signals_df['signal'] > 0).sum()) if 'signal' in signals_df.columns else 0,
                            'negative_signals': int((signals_df['signal'] < 0).sum()) if 'signal' in signals_df.columns else 0
                        }
                    }
                    logger.info(f"✅ {alpha_type} signals exported: {signals_df.shape[0]} rows")
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {alpha_type} signals: {e}")
                    alpha_data['signals'] = None
            else:
                alpha_data['signals'] = None
                
            # Load metrics if available
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    alpha_data['metrics'] = metrics
                    logger.info(f"✅ {alpha_type} metrics exported: {len(metrics)} metrics")
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {alpha_type} metrics: {e}")
                    alpha_data['metrics'] = None
            else:
                alpha_data['metrics'] = None
                
            exported_signals[alpha_type] = alpha_data
            
        return exported_signals
    
    def export_regime_predictions(self) -> Optional[Dict[str, Any]]:
        """
        Export regime detection predictions
        """
        logger.info(f"🎯 Exporting regime predictions for {self.symbol}_{self.timeframe}")
        
        regime_file = self.artifacts_dir / f"regime_predictions_{self.symbol}.parquet"
        regime_metrics_file = self.artifacts_dir / f"regime_metrics_{self.symbol}.json"
        
        regime_data = {}
        
        # Load regime predictions
        if regime_file.exists():
            try:
                regime_df = pd.read_parquet(regime_file)
                regime_data['predictions'] = {
                    'shape': regime_df.shape,
                    'columns': regime_df.columns.tolist(),
                    'date_range': [str(regime_df.index.min()), str(regime_df.index.max())],
                    'regime_distribution': regime_df['predicted_regime'].value_counts().to_dict() if 'predicted_regime' in regime_df.columns else {}
                }
                logger.info(f"✅ Regime predictions exported: {regime_df.shape[0]} rows")
            except Exception as e:
                logger.warning(f"⚠️ Error loading regime predictions: {e}")
                regime_data['predictions'] = None
        else:
            regime_data['predictions'] = None
            
        # Load regime metrics
        if regime_metrics_file.exists():
            try:
                with open(regime_metrics_file, 'r') as f:
                    metrics = json.load(f)
                regime_data['metrics'] = metrics
                logger.info(f"✅ Regime metrics exported: {len(metrics)} metrics")
            except Exception as e:
                logger.warning(f"⚠️ Error loading regime metrics: {e}")
                regime_data['metrics'] = None
        else:
            regime_data['metrics'] = None
            
        return regime_data if any(regime_data.values()) else None
    
    def export_psr_selection(self) -> Optional[Dict[str, Any]]:
        """
        Export PSR selection results
        """
        logger.info("🏆 Exporting PSR selection results")
        
        psr_file = self.artifacts_dir / "psr_selection_results.json"
        
        if psr_file.exists():
            try:
                with open(psr_file, 'r') as f:
                    psr_data = json.load(f)
                logger.info(f"✅ PSR selection exported: {len(psr_data.get('selected_alphas', []))} selected alphas")
                return psr_data
            except Exception as e:
                logger.warning(f"⚠️ Error loading PSR selection: {e}")
                
        return None
    
    def create_ml_features(self, signals_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Create ML-ready feature matrix from alpha signals
        """
        logger.info("🤖 Creating ML feature matrix")
        
        features_list = []
        
        for alpha_type, alpha_data in signals_data.items():
            if alpha_data['signals'] is not None:
                try:
                    # Load the actual signals dataframe
                    signals_file = self.artifacts_dir / f"{alpha_type}_signals_{self.symbol}.parquet"
                    if signals_file.exists():
                        signals_df = pd.read_parquet(signals_file)
                        
                        # Create features from signals
                        features = pd.DataFrame(index=signals_df.index)
                        
                        if 'signal' in signals_df.columns:
                            features[f'{alpha_type}_signal'] = signals_df['signal']
                            features[f'{alpha_type}_signal_ma5'] = signals_df['signal'].rolling(5).mean()
                            features[f'{alpha_type}_signal_std5'] = signals_df['signal'].rolling(5).std()
                            
                        if 'confidence' in signals_df.columns:
                            features[f'{alpha_type}_confidence'] = signals_df['confidence']
                            
                        if 'position_size' in signals_df.columns:
                            features[f'{alpha_type}_position_size'] = signals_df['position_size']
                            
                        features_list.append(features)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error creating features for {alpha_type}: {e}")
        
        if features_list:
            # Combine all features
            ml_features = pd.concat(features_list, axis=1)
            ml_features = ml_features.fillna(0)  # Fill NaN with 0
            
            logger.info(f"✅ ML features created: {ml_features.shape}")
            return ml_features
        else:
            logger.warning("⚠️ No features could be created")
            return pd.DataFrame()
    
    def export_to_mlflow(self, export_data: Dict[str, Any]) -> None:
        """
        Export results to MLflow for tracking
        """
        logger.info("📈 Exporting to MLflow")
        
        try:
            mlflow.set_experiment("ML_Export_Pipeline")
            
            with mlflow.start_run():
                # Log basic info
                mlflow.log_params({
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "export_timestamp": datetime.now().isoformat(),
                    "total_alphas": len(export_data.get('alpha_signals', {})),
                    "has_regime_data": export_data.get('regime_predictions') is not None,
                    "has_psr_data": export_data.get('psr_selection') is not None
                })
                
                # Log metrics from each alpha
                for alpha_type, alpha_data in export_data.get('alpha_signals', {}).items():
                    if alpha_data.get('metrics'):
                        # Flatten metrics with alpha prefix
                        for key, value in alpha_data['metrics'].items():
                            if isinstance(value, (int, float)) and not pd.isna(value):
                                mlflow.log_metric(f"{alpha_type}_{key}", float(value))
                
                logger.info("✅ MLflow export completed")
                
        except Exception as e:
            logger.warning(f"⚠️ MLflow export failed: {e}")
    
    def run_export(self) -> Dict[str, Any]:
        """
        Run complete export process
        """
        logger.info(f"🚀 Starting ML export for {self.symbol}_{self.timeframe}")
        
        export_data = {
            "metadata": {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "export_timestamp": datetime.now().isoformat(),
                "export_version": "1.0.0"
            },
            "alpha_signals": self.export_alpha_signals(),
            "regime_predictions": self.export_regime_predictions(),
            "psr_selection": self.export_psr_selection()
        }
        
        # Create ML features
        ml_features = self.create_ml_features(export_data["alpha_signals"])
        if not ml_features.empty:
            # Reset index and convert timestamp to string to avoid conversion issues
            ml_features_reset = ml_features.reset_index()
            
            # Convert any timestamp columns to string
            for col in ml_features_reset.columns:
                if pd.api.types.is_datetime64_any_dtype(ml_features_reset[col]):
                    ml_features_reset[col] = ml_features_reset[col].astype(str)
            
            # Save ML features as CSV to avoid timestamp conversion issues
            features_file = self.export_dir / f"ml_features_{self.symbol}_{self.timeframe}.csv"
            ml_features_reset.to_csv(features_file, index=False)
            export_data["ml_features"] = {
                "file_path": str(features_file),
                "shape": ml_features.shape,
                "columns": ml_features.columns.tolist()
            }
            logger.info(f"💾 ML features saved: {features_file}")
        
        # Save export summary
        summary_file = self.export_dir / f"export_summary_{self.symbol}_{self.timeframe}.json"
        with open(summary_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Export to MLflow
        self.export_to_mlflow(export_data)
        
        logger.info(f"✅ ML export completed for {self.symbol}_{self.timeframe}")
        logger.info(f"📊 Summary: {len(export_data['alpha_signals'])} alphas, "
                   f"{'✅' if export_data['regime_predictions'] else '❌'} regime, "
                   f"{'✅' if export_data['psr_selection'] else '❌'} PSR")
        
        return export_data

def main():
    """
    Main export function
    """
    parser = argparse.ArgumentParser(description="ML Exporter for Alpha Signals")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("--timeframe", required=True, help="Timeframe (e.g., 1h)")
    
    args = parser.parse_args()
    
    try:
        # Run export
        exporter = MLExporter(args.symbol, args.timeframe)
        result = exporter.run_export()
        
        # Print summary
        print(f"\n{'='*60}")
        print("ML EXPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Symbol: {args.symbol}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Alphas Exported: {len(result['alpha_signals'])}")
        
        for alpha_type, alpha_data in result['alpha_signals'].items():
            status = "✅" if alpha_data['signals'] is not None else "❌"
            print(f"  {status} {alpha_type.upper()}: {'signals + metrics' if alpha_data['signals'] and alpha_data['metrics'] else 'partial/missing'}")
        
        print(f"Regime Data: {'✅' if result['regime_predictions'] else '❌'}")
        print(f"PSR Selection: {'✅' if result['psr_selection'] else '❌'}")
        
        if 'ml_features' in result:
            print(f"ML Features: ✅ {result['ml_features']['shape'][0]} rows x {result['ml_features']['shape'][1]} features")
        else:
            print("ML Features: ❌ No features created")
            
        print(f"Export completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()