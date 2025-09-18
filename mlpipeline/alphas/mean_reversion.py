#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Mean Reversion - Version Production Async
Migration et amélioration majeure de l'alpha_mr.py original
- Régimes adaptatifs
- Multiple timeframes 
- Machine Learning pour optimisation
- Position sizing Kelly
UNIQUEMENT données réelles - Validation stricte
Converti en async pour performance optimale
"""

import asyncio
import logging
import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Configuration path pour imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import des utilitaires
try:
    from mlpipeline.utils.risk_metrics import (
        ratio_sharpe, drawdown_max, probabilistic_sharpe_ratio,
        comprehensive_metrics, validate_metrics
    )
    from mlpipeline.utils.artifact_cleaner import validate_real_data_only
except ImportError:
    # Fallback pour exécution comme script
    logger.warning("Utilitaires non disponibles - fonctions simulées")
    
    def ratio_sharpe(returns):
        return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    def drawdown_max(equity_curve):
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
    
    def probabilistic_sharpe_ratio(returns, benchmark=0):
        n = len(returns)
        sharpe = ratio_sharpe(returns)
        return 1 - stats.norm.cdf((benchmark - sharpe) * np.sqrt(n-1))
    
    def comprehensive_metrics(returns, frequence_annuelle=252):
        if len(returns) == 0:
            return {"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0}
        
        sharpe = ratio_sharpe(returns)
        equity_curve = (1+returns).cumprod()
        max_dd = drawdown_max(equity_curve)
        total_ret = equity_curve.iloc[-1] - 1 if len(equity_curve) > 0 else 0.0
        
        return {
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd), 
            "total_return": float(total_ret),
            "volatility": float(returns.std() * np.sqrt(frequence_annuelle)),
            "return_annualized": float(returns.mean() * frequence_annuelle)
        }
    
    def validate_metrics(metrics):
        return True
    
    def validate_real_data_only(df, source="API"):
        return not df.empty

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class AdaptiveMeanReversion:
    """
    Stratégie Mean Reversion adaptative avec régimes dynamiques
    Amélioration majeure de l'original avec ML et multi-timeframes
    """
    
    def __init__(self, 
                 lookback_short: int = 10,
                 lookback_long: int = 50, 
                 z_entry_base: float = 1.0,  # Plus permissif (1.5 → 1.0)
                 z_exit_base: float = 0.2,  # Plus permissif (0.3 → 0.2)
                 regime_window: int = 252,
                 use_ml_optimization: bool = True):
        
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.z_entry_base = z_entry_base
        self.z_exit_base = z_exit_base
        self.regime_window = regime_window
        self.use_ml_optimization = use_ml_optimization
        
        # Modèles ML pour optimisation dynamique
        self.ml_models = {}
        self.scalers = {}
        
        # État de la stratégie
        self.current_position = 0  # -1, 0, 1
        self.entry_price = None
        self.regime_state = "neutral"  # low_vol, normal, high_vol
        
        logger.info(f"✅ Mean Reversion initialisé (ML={use_ml_optimization})")
    
    def detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Détection de régimes de marché basée sur la volatilité
        - low_vol: Volatilité < 25ème percentile  
        - high_vol: Volatilité > 75ème percentile
        - normal: Entre les deux
        """
        
        # Calcul volatilité réalisée (window=20)
        returns = df['ret'] if 'ret' in df.columns else df['close'].pct_change()
        rolling_vol = returns.rolling(20).std() * np.sqrt(365)  # Annualisée
        
        # Volatilité de long terme pour régimes
        long_vol = rolling_vol.rolling(self.regime_window).mean()
        
        # Percentiles dynamiques
        vol_25 = rolling_vol.rolling(self.regime_window).quantile(0.25)
        vol_75 = rolling_vol.rolling(self.regime_window).quantile(0.75)
        
        # Classification des régimes
        regimes = pd.Series('normal', index=df.index)
        regimes[rolling_vol <= vol_25] = 'low_vol'
        regimes[rolling_vol >= vol_75] = 'high_vol'
        
        logger.info(f"📊 Régimes détectés: {regimes.value_counts().to_dict()}")
        return regimes
    
    def calculate_adaptive_zscore(self, df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """
        Z-score adaptatif selon les régimes de marché
        """
        results = pd.DataFrame(index=df.index)
        
        # Prix de référence (moyenne mobile)
        results['price'] = df['close']
        results['sma_short'] = df['close'].rolling(self.lookback_short).mean()
        results['sma_long'] = df['close'].rolling(self.lookback_long).mean()
        
        # Spread principal : Prix - SMA
        results['spread'] = results['price'] - results['sma_short']
        
        # Z-score de base
        results['spread_mean'] = results['spread'].rolling(self.lookback_long).mean()
        results['spread_std'] = results['spread'].rolling(self.lookback_long).std()
        results['zscore_base'] = ((results['spread'] - results['spread_mean']) / 
                                 (results['spread_std'] + 1e-8))
        
        # Adaptation selon régimes
        regime_multipliers = {
            'low_vol': 0.8,    # Plus sensible en faible volatilité
            'normal': 1.0,     # Standard
            'high_vol': 1.3    # Moins sensible en haute volatilité
        }
        
        results['regime'] = regimes
        results['regime_multiplier'] = regimes.map(regime_multipliers).fillna(1.0)
        results['zscore_adaptive'] = results['zscore_base'] * results['regime_multiplier']
        
        return results
    
    def calculate_multi_timeframe_signals(self, df_1h: pd.DataFrame, 
                                        df_4h: Optional[pd.DataFrame] = None,
                                        df_1d: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Signaux multi-timeframes avec pondération
        """
        
        # Signal principal 1H
        regimes_1h = self.detect_market_regime(df_1h)
        signals_1h = self.calculate_adaptive_zscore(df_1h, regimes_1h)
        
        # Poids par défaut
        signals_1h['weight_1h'] = 1.0
        signals_1h['weight_4h'] = 0.0
        signals_1h['weight_1d'] = 0.0
        
        # Signal 4H si disponible
        if df_4h is not None and len(df_4h) > 50:
            regimes_4h = self.detect_market_regime(df_4h)
            signals_4h = self.calculate_adaptive_zscore(df_4h, regimes_4h)
            
            # Alignement temporel (resample 4H vers 1H)
            signals_4h_resampled = signals_4h['zscore_adaptive'].resample('1H').ffill()
            signals_1h['zscore_4h'] = signals_4h_resampled.reindex(signals_1h.index).fillna(0)
            signals_1h['weight_4h'] = 0.3
        
        # Signal 1D si disponible  
        if df_1d is not None and len(df_1d) > 20:
            regimes_1d = self.detect_market_regime(df_1d)
            signals_1d = self.calculate_adaptive_zscore(df_1d, regimes_1d)
            
            # Alignement temporel (resample 1D vers 1H)
            signals_1d_resampled = signals_1d['zscore_adaptive'].resample('1H').ffill()
            signals_1h['zscore_1d'] = signals_1d_resampled.reindex(signals_1h.index).fillna(0)
            signals_1h['weight_1d'] = 0.2
        
        # Signal composite pondéré
        zscore_components = []
        weights = []
        
        if 'zscore_adaptive' in signals_1h.columns:
            zscore_components.append(signals_1h['zscore_adaptive'] * signals_1h['weight_1h'])
            weights.append(signals_1h['weight_1h'])
        
        if 'zscore_4h' in signals_1h.columns:
            zscore_components.append(signals_1h['zscore_4h'] * signals_1h['weight_4h'])
            weights.append(signals_1h['weight_4h'])
            
        if 'zscore_1d' in signals_1h.columns:
            zscore_components.append(signals_1h['zscore_1d'] * signals_1h['weight_1d'])
            weights.append(signals_1h['weight_1d'])
        
        # Z-score final pondéré
        total_weight = sum(weights) if weights else [1.0]
        if isinstance(total_weight, list):
            total_weight = total_weight[0]
            
        if zscore_components:
            # Correction: utiliser np.maximum pour opération élément par élément
            divisor = np.maximum(total_weight, 0.1)
            signals_1h['zscore_final'] = sum(zscore_components) / divisor
        else:
            signals_1h['zscore_final'] = signals_1h['zscore_adaptive']
        
        return signals_1h
    
    def optimize_parameters_ml(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Optimisation ML des paramètres de la stratégie
        """
        if not self.use_ml_optimization:
            return {
                'z_entry': self.z_entry_base,
                'z_exit': self.z_exit_base,
                'lookback_factor': 1.0
            }
        
        logger.info("🧠 Optimisation ML des paramètres...")
        
        # Features pour ML
        returns = df['ret'] if 'ret' in df.columns else df['close'].pct_change()
        features_df = pd.DataFrame()
        
        # Features techniques traditionnelles
        features_df['volatility'] = returns.rolling(20).std()
        features_df['momentum'] = df['close'].pct_change(5)
        features_df['volume_ma'] = df['volume'].rolling(20).mean() if 'volume' in df.columns else 1
        features_df['rsi'] = self._calculate_rsi(df['close'])
        features_df['bb_position'] = self._calculate_bb_position(df['close'])

        # Features symboliques avancées (inspirées du papier RL)
        try:
            from mlpipeline.features.symbolic_operators import AlphaFormulaGenerator
            alpha_gen = AlphaFormulaGenerator()
            symbolic_features = alpha_gen.generate_enhanced_features(df)

            # Sélection des features les plus pertinentes pour mean reversion
            key_symbolic_features = [
                'sign_returns', 'ts_rank_close_10', 'delta_close_5',
                'skew_returns_20', 'kurt_returns_20', 'mad_close_15',
                'alpha_006'  # Corrélation open-volume
            ]

            for feat in key_symbolic_features:
                if feat in symbolic_features.columns:
                    features_df[f'symbolic_{feat}'] = symbolic_features[feat]

            logger.info(f"✅ Ajout de {len(key_symbolic_features)} features symboliques")
        except Exception as e:
            logger.warning(f"⚠️ Erreur features symboliques: {e}")
        
        # Target : Rendement futur N périodes (VRAI TARGET PRÉDICTIF)
        features_df['target_future_return'] = self._calculate_future_return_target(df)
        
        # Nettoyage
        features_df = features_df.dropna()
        
        if len(features_df) < 100:
            logger.warning("⚠️  Pas assez de données pour ML, paramètres par défaut")
            return {
                'z_entry': self.z_entry_base,
                'z_exit': self.z_exit_base, 
                'lookback_factor': 1.0
            }
        
        # Préparation données
        feature_cols = ['volatility', 'momentum', 'volume_ma', 'rsi', 'bb_position']
        X = features_df[feature_cols].values
        y = features_df['target_future_return'].values
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split temporel
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Random Forest avec GridSearch
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [10, 20]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error')
        
        try:
            grid_search.fit(X_scaled, y)
            best_model = grid_search.best_estimator_
            
            # Prédiction pour état actuel
            if len(X_scaled) > 0:
                current_features = X_scaled[-1:] 
                predicted_return_signal = best_model.predict(current_features)[0]
                
                # Convertir signal de rendement en ajustement de z_entry
                # Si signal positif fort -> diminuer z_entry (être moins agressif)
                # Si signal négatif fort -> augmenter z_entry (être plus sélectif)
                z_adjustment = 1.0 - (predicted_return_signal * 0.5)  # Facteur d'ajustement
                predicted_z_entry = self.z_entry_base * np.clip(z_adjustment, 0.5, 2.0)
            else:
                predicted_z_entry = self.z_entry_base
                predicted_return_signal = 0.0
            
            # Sauvegarde modèle
            self.ml_models['return_predictor'] = best_model
            self.scalers['features'] = scaler
            
            optimized_params = {
                'z_entry': float(predicted_z_entry),
                'z_exit': float(predicted_z_entry * 0.3),  # Ratio fixe
                'lookback_factor': 1.0,
                'predicted_return_signal': float(predicted_return_signal)
            }
            
            logger.info(f"✅ Paramètres optimisés ML: {optimized_params}")
            # Sauvegarder les paramètres optimisés dans l'instance
            self.optimized_params = optimized_params
            return optimized_params
            
        except Exception as e:
            logger.error(f"❌ Erreur optimisation ML: {e}")
            return {
                'z_entry': self.z_entry_base,
                'z_exit': self.z_exit_base,
                'lookback_factor': 1.0
            }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcul RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.Series:
        """Position dans les Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_position = (prices - lower_band) / (upper_band - lower_band + 1e-10)
        return bb_position
    
    def _calculate_future_return_target(self, df: pd.DataFrame, horizon: int = 12) -> pd.Series:
        """
        Calcule le target basé sur les rendements futurs (SANS DATA LEAKAGE)
        
        Args:
            df: DataFrame avec OHLC data
            horizon: Nombre de périodes futures à regarder (12h par défaut)
            
        Returns:
            Series avec target de rendements futurs
        """
        
        # Calculer les rendements futurs
        future_returns = df['close'].pct_change(horizon).shift(-horizon)
        
        # Calculer le zscore actuel (disponible au moment t)
        current_returns = df['close'].pct_change()
        lookback = getattr(self, 'lookback_window', getattr(self, 'lookback_short', 20))
        rolling_mean = current_returns.rolling(lookback).mean()
        rolling_std = current_returns.rolling(lookback).std()
        current_zscore = (current_returns - rolling_mean) / (rolling_std + 1e-8)
        
        # Target : Rendement futur ajusté par le signal mean reversion
        # Si zscore élevé (surachat) -> on s'attend à rendement négatif
        # Si zscore faible (survente) -> on s'attend à rendement positif
        target = future_returns * np.sign(-current_zscore)  # Inverser le signe pour mean reversion
        
        return target
    
    def _find_best_z_entry_window(self, df_window: pd.DataFrame) -> float:
        """Trouve le meilleur z_entry pour une fenêtre donnée"""
        best_sharpe = -10
        best_z = self.z_entry_base
        
        # Test différents z_entry
        for z_test in [1.0, 1.5, 2.0, 2.5]:
            try:
                signals = self._calculate_simple_signals(df_window, z_test)
                returns = self._backtest_window(df_window, signals)
                
                if len(returns) > 10:
                    sharpe = ratio_sharpe(365, returns)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_z = z_test
            except:
                continue
                
        return best_z
    
    def _calculate_simple_signals(self, df: pd.DataFrame, z_entry: float) -> pd.Series:
        """Signaux simples pour optimisation"""
        spread = df['close'] - df['close'].rolling(10).mean()
        z_score = ((spread - spread.rolling(50).mean()) / 
                  (spread.rolling(50).std() + 1e-8))
        
        signals = pd.Series(0, index=df.index)
        signals[z_score < -z_entry] = 1   # Long
        signals[z_score > z_entry] = -1   # Short
        
        return signals
    
    def _backtest_window(self, df: pd.DataFrame, signals: pd.Series) -> np.ndarray:
        """Backtest rapide sur fenêtre"""
        returns = df['close'].pct_change().fillna(0)
        strategy_returns = signals.shift(1) * returns
        return strategy_returns.dropna().values
    
    async def generate_signals(self, 
                              df_1h: pd.DataFrame,
                              df_4h: Optional[pd.DataFrame] = None,
                              df_1d: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Génération des signaux de trading avec toutes les améliorations
        """
        
        # Validation données réelles
        if not validate_real_data_only(df_1h, "MeanReversion"):
            raise ValueError("❌ REJETÉ: Données synthétiques détectées")
        
        logger.info("🎯 Génération signaux Mean Reversion...")
        
        # Optimisation ML des paramètres  
        optimal_params = self.optimize_parameters_ml(df_1h)
        z_entry_dynamic = optimal_params['z_entry']
        z_exit_dynamic = optimal_params['z_exit']
        
        # Signaux multi-timeframes
        signals_df = self.calculate_multi_timeframe_signals(df_1h, df_4h, df_1d)
        
        # Signaux finaux
        signals_df['signal_raw'] = 0
        signals_df['signal_final'] = 0
        signals_df['position_size'] = 0.0
        
        # Logique de signaux
        z_final = signals_df['zscore_final'].fillna(0)
        
        # Entrées
        long_entry = z_final < -z_entry_dynamic
        short_entry = z_final > z_entry_dynamic
        
        # Sorties
        long_exit = z_final > -z_exit_dynamic
        short_exit = z_final < z_exit_dynamic
        
        # Signaux bruts - CORRECTION: Ne pas effacer tous les signaux !
        signals_df.loc[long_entry, 'signal_raw'] = 1
        signals_df.loc[short_entry, 'signal_raw'] = -1
        # NOTE: Les sorties sont gérées dans la machine à états, pas ici !
        
        # Machine à états pour signaux finaux
        current_position = 0
        for i in range(len(signals_df)):
            signal = signals_df['signal_raw'].iloc[i]
            
            if signal != 0:  # Nouveau signal
                current_position = signal
            elif current_position != 0:  # Position existante, vérifier sortie
                if (current_position == 1 and long_exit.iloc[i]) or \
                   (current_position == -1 and short_exit.iloc[i]):
                    current_position = 0
            
            signals_df.loc[signals_df.index[i], 'signal_final'] = current_position
        
        # Position sizing avec Kelly approximé
        signals_df['position_size'] = self._calculate_kelly_sizing(signals_df)
        
        # Méta-informations
        signals_df['z_entry_used'] = z_entry_dynamic
        signals_df['z_exit_used'] = z_exit_dynamic
        signals_df['regime'] = signals_df.get('regime', 'normal')
        
        return signals_df
    
    def _calculate_kelly_sizing(self, signals_df: pd.DataFrame, 
                               lookback: int = 100) -> pd.Series:
        """
        Position sizing approximé selon Kelly Criterion
        """
        position_sizes = pd.Series(0.0, index=signals_df.index)
        
        if 'signal_final' not in signals_df.columns:
            return position_sizes
        
        # Calcul Kelly sur fenêtre glissante
        for i in range(lookback, len(signals_df)):
            window_signals = signals_df['signal_final'].iloc[i-lookback:i]
            
            if 'ret' in signals_df.columns:
                window_returns = signals_df['ret'].iloc[i-lookback:i]
            else:
                prices = signals_df['price'] if 'price' in signals_df.columns else signals_df.index
                window_returns = pd.Series(prices).pct_change().iloc[i-lookback:i]
            
            # Returns de la stratégie sur la fenêtre
            strategy_rets = window_signals.shift(1) * window_returns
            strategy_rets = strategy_rets.dropna()
            
            if len(strategy_rets) < 20:
                kelly_fraction = 0.1  # Par défaut
            else:
                # Kelly approximé : (mean_return / variance) 
                mean_ret = strategy_rets.mean()
                var_ret = strategy_rets.var()
                
                if var_ret > 0:
                    kelly_raw = mean_ret / var_ret
                    # Limitation du Kelly (max 25%)
                    kelly_fraction = np.clip(abs(kelly_raw), 0.05, 0.25)
                else:
                    kelly_fraction = 0.1
            
            # Application du sizing
            current_signal = signals_df['signal_final'].iloc[i]
            if current_signal != 0:
                position_sizes.iloc[i] = np.sign(current_signal) * kelly_fraction
        
        return position_sizes
    
    async def backtest_strategy(self, signals_df: pd.DataFrame, 
                               transaction_cost: float = 0.001) -> Dict:
        """
        Backtest complet de la stratégie avec coûts de transaction
        """
        
        if 'signal_final' not in signals_df.columns:
            raise ValueError("Signaux manquants dans le DataFrame")
        
        # Returns de base
        if 'ret' in signals_df.columns:
            returns = signals_df['ret'].fillna(0)
        else:
            returns = signals_df['price'].pct_change().fillna(0) if 'price' in signals_df.columns else pd.Series(0, index=signals_df.index)
        
        # Position sizing
        positions = signals_df['position_size'].fillna(0) if 'position_size' in signals_df.columns else signals_df['signal_final'].fillna(0)
        
        # Returns de stratégie
        strategy_returns = positions.shift(1) * returns
        
        # Coûts de transaction
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * transaction_cost
        net_returns = strategy_returns - transaction_costs
        
        # Equity curve
        equity = (1 + net_returns).cumprod()
        
        # Métriques complètes
        metrics = comprehensive_metrics(net_returns.dropna(), frequence_annuelle=365*24)  # Hourly
        
        # Métriques spécifiques à la stratégie
        metrics.update({
            "total_trades": int(position_changes[position_changes > 0].sum()),
            "avg_trade_duration": self._calculate_avg_trade_duration(positions),
            "max_position_size": float(positions.abs().max()),
            "transaction_cost_impact": float(transaction_costs.sum()),
            "equity_final": float(equity.iloc[-1]) if len(equity) > 0 else 1.0
        })
        
        return metrics
    
    def _calculate_avg_trade_duration(self, positions: pd.Series) -> float:
        """Calcule la durée moyenne des trades"""
        if len(positions) == 0:
            return 0.0
        
        trades = []
        current_trade_start = None
        
        for i, pos in enumerate(positions):
            if pos != 0 and current_trade_start is None:
                current_trade_start = i
            elif pos == 0 and current_trade_start is not None:
                trades.append(i - current_trade_start)
                current_trade_start = None
        
        return np.mean(trades) if trades else 0.0

# ==============================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ==============================================

async def train_mean_reversion_alpha(data_path_1h: str = "data/processed/features_BTCUSDT_1h.parquet",
                                    data_path_4h: Optional[str] = None,
                                    data_path_1d: Optional[str] = None,
                                    config: Dict = None) -> Dict:
    """
    Fonction principale d'entraînement Mean Reversion avancé
    
    Args:
        data_path_1h: Chemin données 1H (obligatoire)
        data_path_4h: Chemin données 4H (optionnel)
        data_path_1d: Chemin données 1D (optionnel)
        config: Configuration de la stratégie
    """
    
    # Configuration par défaut - MISE À JOUR RÉALISTE CRYPTO
    default_config = {
        "lookback_short": 10,           # Garde réactif
        "lookback_long": 30,            # Plus court pour crypto (était 50)
        "z_entry_base": 1.0,            # Plus agressif pour crypto (était 1.5)
        "z_exit_base": 0.2,             # Sortie plus rapide (était 0.3)
        "regime_window": 168,           # 1 semaine crypto (était 252)
        "use_ml_optimization": True,    # Garde ML mais avec nouveaux params
        "transaction_cost": 0.002,      # Réaliste (était 0.001)
        "max_position_hold": 24,        # 24h max
        "min_trade_size": 0.005,        # 0.5% minimum
        "volatility_scaling": True      # Nouveau: adapter selon volatilité
    }
    
    # Merger avec la configuration passée
    if config is None:
        config = default_config
    else:
        # Ajouter les valeurs par défaut pour les clés manquantes
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    
    logger.info("🚀 Début entraînement Mean Reversion Alpha Avancé")
    
    # 1. Chargement données
    if not os.path.exists(data_path_1h):
        raise FileNotFoundError(f"❌ Données 1H introuvables: {data_path_1h}")
    
    df_1h = pd.read_parquet(data_path_1h)
    logger.info(f"📊 Données 1H: {len(df_1h)} lignes")
    
    # Validation stricte 1H
    if not validate_real_data_only(df_1h, "MR_Training_1H"):
        raise ValueError("❌ ENTRAÎNEMENT INTERROMPU: Données synthétiques 1H détectées")
    
    # Données optionnelles
    df_4h, df_1d = None, None
    
    if data_path_4h and os.path.exists(data_path_4h):
        df_4h = pd.read_parquet(data_path_4h)
        if not validate_real_data_only(df_4h, "MR_Training_4H"):
            logger.warning("⚠️  Données 4H synthétiques détectées, ignorées")
            df_4h = None
        else:
            logger.info(f"📊 Données 4H: {len(df_4h)} lignes")
    
    if data_path_1d and os.path.exists(data_path_1d):
        df_1d = pd.read_parquet(data_path_1d)
        if not validate_real_data_only(df_1d, "MR_Training_1D"):
            logger.warning("⚠️  Données 1D synthétiques détectées, ignorées") 
            df_1d = None
        else:
            logger.info(f"📊 Données 1D: {len(df_1d)} lignes")
    
    # 2. Initialisation stratégie
    strategy = AdaptiveMeanReversion(
        lookback_short=config["lookback_short"],
        lookback_long=config["lookback_long"],
        z_entry_base=config["z_entry_base"],
        z_exit_base=config["z_exit_base"],
        regime_window=config["regime_window"],
        use_ml_optimization=config["use_ml_optimization"]
    )
    
    # 3. Génération des signaux avec MLflow
    mlflow.set_experiment("MeanReversion_Production")
    
    with mlflow.start_run():
        # Log hyperparamètres
        mlflow.log_params({
            "strategy_type": "Mean_Reversion_Advanced",
            "lookback_short": config["lookback_short"],
            "lookback_long": config["lookback_long"],
            "z_entry_base": config["z_entry_base"],
            "use_ml_optimization": config["use_ml_optimization"],
            "has_4h_data": df_4h is not None,
            "has_1d_data": df_1d is not None
        })
        
        # Génération signaux (async)
        signals_df = await strategy.generate_signals(df_1h, df_4h, df_1d)
        
        # 4. Backtest (async)
        metrics = await strategy.backtest_strategy(signals_df, config["transaction_cost"])
        
        # Log métriques MLflow
        mlflow.log_metrics(metrics)
        
        # Sauvegarde modèle ML si utilisé
        if strategy.ml_models and 'return_predictor' in strategy.ml_models:
            mlflow.sklearn.log_model(
                strategy.ml_models['return_predictor'], 
                "mr_return_predictor"
            )
    
    # 5. Sauvegarde artifacts
    artifacts_dir = Path("data/artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Symbole depuis le nom de fichier
    symbol = Path(data_path_1h).stem.split('_')[1] if '_' in Path(data_path_1h).stem else "UNKNOWN"
    
    # Sauvegarde métriques
    metrics_file = artifacts_dir / f"mr_metrics_{symbol}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Sauvegarde signaux pour analyse
    signals_file = artifacts_dir / f"mr_signals_{symbol}.parquet"
    signals_df.to_parquet(signals_file)
    
    logger.info("📊 MÉTRIQUES MEAN REVERSION:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.4f}")
        else:
            logger.info(f"   {key}: {value}")
    
    # Validation des performances
    validation = validate_metrics(metrics, min_sharpe=0.8, max_drawdown=0.25)
    
    if not validation["is_valid"]:
        for error in validation["errors"]:
            logger.error(f"❌ {error}")
    
    for warning in validation["warnings"]:
        logger.warning(f"⚠️  {warning}")
    
    # Ajouter les paramètres optimisés au résultat
    if hasattr(strategy, 'optimized_params') and strategy.optimized_params:
        metrics.update({
            f"ml_{key}": value for key, value in strategy.optimized_params.items()
            if isinstance(value, (int, float))
        })
    
    return metrics

# ==============================================
# SCRIPT PRINCIPAL
# ==============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement Mean Reversion Alpha Avancé")
    parser.add_argument("--data-1h", required=True,
                       help="Chemin données 1H")
    parser.add_argument("--data-4h", 
                       help="Chemin données 4H (optionnel)")
    parser.add_argument("--data-1d",
                       help="Chemin données 1D (optionnel)")
    parser.add_argument("--z-entry", type=float, default=1.5)
    parser.add_argument("--z-exit", type=float, default=0.3)
    parser.add_argument("--no-ml", action="store_true",
                       help="Désactiver optimisation ML")
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--lookback-short", type=int, default=10)
    parser.add_argument("--lookback-long", type=int, default=50)
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "lookback_short": args.lookback_short,
        "lookback_long": args.lookback_long,
        "z_entry_base": args.z_entry,
        "z_exit_base": args.z_exit,
        "use_ml_optimization": not args.no_ml,
        "transaction_cost": args.transaction_cost
    }
    
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        metrics = asyncio.run(train_mean_reversion_alpha(
            args.data_1h, args.data_4h, args.data_1d, config
        ))
        logger.info("✅ Entraînement Mean Reversion terminé avec succès")
        
    except Exception as e:
        logger.error(f"❌ ERREUR entraînement Mean Reversion: {e}")
        sys.exit(1)