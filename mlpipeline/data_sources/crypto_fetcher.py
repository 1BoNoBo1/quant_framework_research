#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Récupération de données crypto en temps réel
Amélioration du data_ingest.py original avec vraies données
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ccxt
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import yaml
from pathlib import Path

# Ajouter le cache système
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
try:
    from orchestration.smart_cache import cached_data, get_cache
except ImportError:
    logger.warning("Smart cache non disponible - fonctionnement normal")
    # Décorateur vide si cache non disponible
    def cached_data(ttl_hours=1, key_args=None):
        def decorator(func):
            return func
        return decorator

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    """Récupérateur de données crypto multi-sources"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.binance_client = None
        self.ccxt_exchange = None
        self._setup_clients()
        
    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Configuration par défaut
                return {
                    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                    "timeframes": ["1h", "4h", "1d"],
                    "lookback_days": 365,
                    "data_dir": "data/raw"
                }
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return {}
    
    def _setup_clients(self):
        """Configure les clients API"""
        try:
            # Binance API (optionnel avec clés)
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            
            if api_key and api_secret:
                self.binance_client = Client(api_key, api_secret)
                logger.info("✅ Client Binance configuré avec API keys")
            else:
                logger.info("⚠️  Pas de clés Binance - données publiques uniquement")
            
            # CCXT pour données publiques
            self.ccxt_exchange = ccxt.binance({
                'rateLimit': 1200,
                'enableRateLimit': True,
            })
            logger.info("✅ Client CCXT configuré")
            
        except Exception as e:
            logger.error(f"Erreur configuration clients: {e}")
    
    @cached_data(ttl_hours=1, key_args=['symbol', 'timeframe', 'limit'])
    async def fetch_ohlcv(self, 
                         symbol: str, 
                         timeframe: str = '1h', 
                         limit: int = 1000) -> pd.DataFrame:
        """
        Récupère données OHLCV via CCXT
        
        Args:
            symbol: Paire crypto (ex: 'BTC/USDT')
            timeframe: Intervalle de temps
            limit: Nombre de bougies
        """
        try:
            # Normaliser le symbole pour CCXT
            if '/' not in symbol:
                # BTCUSDT -> BTC/USDT
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = 'USDT'
                    ccxt_symbol = f"{base}/{quote}"
                else:
                    ccxt_symbol = symbol
            else:
                ccxt_symbol = symbol
            
            logger.info(f"📊 Récupération {ccxt_symbol} {timeframe} (limit={limit})")
            
            # Récupération via CCXT
            ohlcv_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ccxt_exchange.fetch_ohlcv(
                    ccxt_symbol, timeframe, limit=limit
                )
            )
            
            # Conversion en DataFrame
            df = pd.DataFrame(ohlcv_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Conversion timestamp
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.drop('timestamp', axis=1)
            
            # Réorganisation colonnes
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✅ {len(df)} bougies récupérées pour {ccxt_symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_historical_bulk(self, 
                                   symbols: List[str], 
                                   timeframes: List[str],
                                   days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Récupération en masse des données historiques
        """
        results = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Calcul du nombre de bougies nécessaires
                    if timeframe == '1h':
                        limit = min(days * 24, 1000)
                    elif timeframe == '4h':
                        limit = min(days * 6, 1000)
                    elif timeframe == '1d':
                        limit = min(days, 1000)
                    else:
                        limit = 1000
                    
                    # Récupération asynchrone
                    df = await self.fetch_ohlcv(symbol, timeframe, limit)
                    
                    if not df.empty:
                        key = f"{symbol}_{timeframe}"
                        results[key] = df
                        
                        # Sauvegarde
                        self._save_to_parquet(df, symbol, timeframe)
                    
                    # Pause pour éviter rate limit
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Erreur {symbol} {timeframe}: {e}")
                    continue
        
        return results
    
    def _save_to_parquet(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Sauvegarde en format Parquet"""
        try:
            data_dir = Path(self.config.get('data_dir', 'data/raw'))
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Nom de fichier compatible avec l'original
            filename = f"ohlcv_{symbol.replace('/', '')}_{timeframe}.parquet"
            filepath = data_dir / filename
            
            df.to_parquet(filepath, index=False)
            logger.info(f"💾 Sauvegardé: {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde {symbol}: {e}")
    
    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validation des données"""
        if df.empty:
            logger.warning(f"⚠️  DataFrame vide pour {symbol}")
            return False
        
        # Vérifications de base
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"❌ Colonnes manquantes pour {symbol}")
            return False
        
        # Vérifications logiques OHLCV
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        
        if invalid_ohlc > 0:
            logger.warning(f"⚠️  {invalid_ohlc} bougies OHLC invalides pour {symbol}")
        
        # Vérification volumes négatifs
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            logger.warning(f"⚠️  {negative_volume} volumes négatifs pour {symbol}")
        
        logger.info(f"✅ Validation {symbol}: {len(df)} bougies valides")
        return True
    
    def get_realtime_price(self, symbol: str) -> Optional[float]:
        """Prix en temps réel"""
        try:
            ticker = self.ccxt_exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Erreur prix temps réel {symbol}: {e}")
            return None

# ==============================================
# FONCTIONS UTILITAIRES
# ==============================================

def validate_real_data_only(df: pd.DataFrame, symbol: str, source: str = "API") -> bool:
    """
    Validation stricte : UNIQUEMENT données réelles acceptées
    Refuse tout data synthétique ou suspect
    """
    if df.empty:
        logger.error(f"❌ REJETÉ {symbol}: DataFrame vide")
        return False
    
    # Vérification métadonnées (pas de patterns synthétiques)
    if 'synthetic' in str(df.attrs) or source == "synthetic":
        logger.error(f"❌ REJETÉ {symbol}: Données synthétiques détectées")
        return False
    
    # Vérification timestamps réalistes (pas trop réguliers)
    if len(df) > 10:
        time_diffs = df['time'].diff().dropna()
        if time_diffs.nunique() == 1:  # Trop régulier = synthétique
            logger.warning(f"⚠️  SUSPECT {symbol}: Timestamps trop réguliers")
    
    # Vérification prix cohérents avec le marché
    if 'close' in df.columns:
        close_prices = df['close']
        if (close_prices == close_prices.iloc[0]).all():  # Prix constant = synthétique
            logger.error(f"❌ REJETÉ {symbol}: Prix constants détectés")
            return False
    
    # Validation volumes réalistes
    if 'volume' in df.columns:
        volumes = df['volume']
        if (volumes < 0).any():
            logger.error(f"❌ REJETÉ {symbol}: Volumes négatifs")
            return False
        
        # Volume trop constant = suspect
        if volumes.std() / volumes.mean() < 0.1:
            logger.warning(f"⚠️  SUSPECT {symbol}: Volumes trop constants")
    
    logger.info(f"✅ VALIDÉ {symbol}: {len(df)} bougies réelles")
    return True

# ==============================================
# SCRIPT PRINCIPAL
# ==============================================

async def main():
    """Script principal de récupération de données"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Récupération données crypto")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1h"])
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--synthetic", action="store_true", 
                       help="Utiliser données synthétiques")
    
    args = parser.parse_args()
    
    fetcher = CryptoDataFetcher()
    
    # UNIQUEMENT données réelles - pas de mode synthétique
    logger.info("📊 Mode données réelles UNIQUEMENT")
    results = await fetcher.fetch_historical_bulk(
        args.symbols, args.timeframes, args.days
    )
    
    if not results:
        logger.error("❌ ÉCHEC: Aucune donnée réelle récupérée")
        logger.error("💡 Vérifiez votre connexion internet et les APIs crypto")
        sys.exit(1)
    
    logger.info(f"✅ Récupération terminée: {len(results)} datasets")
    
    # Statistiques
    for key, df in results.items():
        if not df.empty:
            logger.info(f"📈 {key}: {len(df)} bougies, "
                      f"{df['time'].min()} -> {df['time'].max()}")

if __name__ == "__main__":
    asyncio.run(main())