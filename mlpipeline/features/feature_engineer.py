#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineer - Construction des features pour ML
Conversion des données OHLCV brutes en features pour les modèles
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import talib

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des indicateurs techniques au DataFrame OHLCV
    
    Args:
        df: DataFrame avec colonnes OHLCV
        
    Returns:
        DataFrame enrichi avec indicateurs techniques
    """
    if df.empty or len(df) < 50:
        logger.warning("Pas assez de données pour calcul indicateurs")
        return df
        
    try:
        # Faire une copie
        result_df = df.copy()
        
        # Prix moyens
        result_df['hl2'] = (df['high'] + df['low']) / 2
        result_df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        result_df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Returns
        result_df['returns'] = df['close'].pct_change()
        result_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            result_df[f'sma_{period}'] = df['close'].rolling(period).mean()
            result_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Volatilité
        for period in [5, 10, 20]:
            result_df[f'volatility_{period}'] = df['close'].rolling(period).std()
            result_df[f'atr_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        # RSI
        result_df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        result_df['rsi_30'] = talib.RSI(df['close'], timeperiod=30)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(df['close'])
        result_df['macd'] = macd
        result_df['macd_signal'] = macdsignal
        result_df['macd_hist'] = macdhist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        result_df['bb_upper'] = upper
        result_df['bb_middle'] = middle
        result_df['bb_lower'] = lower
        result_df['bb_width'] = (upper - lower) / middle
        result_df['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        result_df['stoch_k'] = slowk
        result_df['stoch_d'] = slowd
        
        # Volume indicators
        if 'volume' in df.columns:
            result_df['volume_sma_10'] = df['volume'].rolling(10).mean()
            result_df['volume_ratio'] = df['volume'] / result_df['volume_sma_10']
            
            # On Balance Volume
            result_df['obv'] = talib.OBV(df['close'], df['volume'])
            
        # Price patterns
        result_df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        result_df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Support/Resistance levels
        result_df['resistance_20'] = df['high'].rolling(20).max()
        result_df['support_20'] = df['low'].rolling(20).min()
        result_df['resistance_distance'] = (result_df['resistance_20'] - df['close']) / df['close']
        result_df['support_distance'] = (df['close'] - result_df['support_20']) / df['close']
        
        # Lags pour séries temporelles
        for lag in [1, 2, 3, 5]:
            result_df[f'close_lag_{lag}'] = df['close'].shift(lag)
            result_df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
            result_df[f'returns_lag_{lag}'] = result_df['returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            result_df[f'close_rolling_max_{window}'] = df['close'].rolling(window).max()
            result_df[f'close_rolling_min_{window}'] = df['close'].rolling(window).min()
            result_df[f'close_rolling_mean_{window}'] = df['close'].rolling(window).mean()
            result_df[f'close_rolling_std_{window}'] = df['close'].rolling(window).std()
        
        logger.info(f"✅ Features ajoutées: {len(result_df.columns)} colonnes")
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Erreur calcul features: {e}")
        return df

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage et validation des features
    """
    if df.empty:
        return df
        
    # Suppression des NaN en début (warm-up period)
    df_clean = df.dropna()
    
    # Remplacement des infinis
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill pour les quelques NaN restants
    df_clean = df_clean.ffill()
    df_clean = df_clean.bfill()
    
    # Suppression des colonnes avec trop de NaN
    nan_threshold = 0.5
    for col in df_clean.columns:
        nan_pct = df_clean[col].isna().sum() / len(df_clean)
        if nan_pct > nan_threshold:
            df_clean = df_clean.drop(col, axis=1)
            logger.warning(f"⚠️  Colonne {col} supprimée ({nan_pct:.1%} NaN)")
    
    logger.info(f"✅ Features nettoyées: {len(df_clean)} lignes, {len(df_clean.columns)} colonnes")
    return df_clean

def add_alpha_compatible_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des alias de colonnes pour compatibilité avec les alphas existants
    """
    df_compat = df.copy()
    
    # Mapping des colonnes pour compatibilité alphas
    column_mapping = {
        # Pour DMN model
        'ret': 'returns',           # returns existe déjà
        'rsi': 'rsi_14',           # rsi_14 existe
        'atr': 'atr_20',           # atr_20 existe
        'mom': 'macd_hist',        # momentum proxy avec MACD histogram
        
        # Pour Mean Reversion
        'price': 'close',          # close existe
        'volume_ma': 'volume_sma_10',  # volume_sma_10 existe
        
        # Colonnes communes utiles
        'ma_short': 'sma_5',       # moyenne courte
        'ma_long': 'sma_20',       # moyenne longue
        'volatility': 'volatility_20',  # volatilité par défaut
        'momentum': 'macd',        # momentum MACD
    }
    
    # Ajouter les alias sans écraser les colonnes existantes
    for alias, source in column_mapping.items():
        if source in df_compat.columns and alias not in df_compat.columns:
            df_compat[alias] = df_compat[source]
            logger.info(f"   ✅ Alias ajouté: {alias} -> {source}")
    
    return df_compat

def main():
    """Script principal de feature engineering"""
    parser = argparse.ArgumentParser(description="Feature Engineering pour données crypto")
    parser.add_argument("--input", required=True, help="Fichier parquet d'entrée")
    parser.add_argument("--output", required=True, help="Fichier parquet de sortie")
    
    args = parser.parse_args()
    
    try:
        # Vérification fichier d'entrée
        if not os.path.exists(args.input):
            logger.error(f"❌ Fichier d'entrée non trouvé: {args.input}")
            sys.exit(1)
        
        logger.info(f"📊 Chargement données: {args.input}")
        
        # Chargement
        df = pd.read_parquet(args.input)
        logger.info(f"✅ {len(df)} lignes chargées")
        
        if df.empty:
            logger.error("❌ DataFrame vide")
            sys.exit(1)
        
        # Vérification colonnes OHLCV requises
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Colonnes manquantes: {missing_cols}")
            sys.exit(1)
        
        # Feature engineering
        logger.info("🔧 Calcul des features techniques...")
        df_features = add_technical_indicators(df)
        
        # Nettoyage
        logger.info("🧹 Nettoyage des features...")
        df_clean = clean_features(df_features)
        
        if df_clean.empty:
            logger.error("❌ Aucune donnée après nettoyage")
            sys.exit(1)
        
        # Création dossier de sortie
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ajout d'alias pour compatibilité avec les alphas existants
        logger.info("🔗 Ajout d'alias de colonnes pour compatibilité alphas...")
        df_clean = add_alpha_compatible_columns(df_clean)
        
        # CORRECTION CRITIQUE: Définir l'index temporel avant sauvegarde
        if 'time' in df_clean.columns:
            logger.info("🕒 Définition de l'index temporel (correction critique)")
            df_clean = df_clean.set_index('time')
            logger.info(f"✅ Index temporel défini: {type(df_clean.index)}")
        else:
            logger.warning("⚠️ Colonne 'time' manquante - impossible de définir index temporel")
        
        # Sauvegarde avec index temporel correct
        df_clean.to_parquet(args.output, compression='snappy')  # index=True par défaut maintenant
        logger.info(f"💾 Features sauvegardées: {args.output}")
        logger.info(f"📊 Shape finale: {df_clean.shape}")
        logger.info(f"🕒 Index type: {type(df_clean.index)}")
        
        # Statistiques finales
        logger.info("📈 Résumé des features:")
        logger.info(f"   - Lignes: {len(df_clean)}")
        logger.info(f"   - Colonnes: {len(df_clean.columns)}")
        logger.info(f"   - Période: {df_clean.index[0] if hasattr(df_clean.index, 'min') else 'N/A'} → {df_clean.index[-1] if hasattr(df_clean.index, 'max') else 'N/A'}")
        
        logger.info("✅ Feature engineering terminé avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale feature engineering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()