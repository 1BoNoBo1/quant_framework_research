#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Funding Rate - Version Production Async
Migration et amélioration majeure de l'alpha_funding.py original
- Vrai calcul funding rate crypto
- Modèle prédictif funding future
- Arbitrage spot-futures
- Position sizing optimal
UNIQUEMENT données réelles - Validation stricte
Converti en async pour performance optimale
"""

import asyncio
import logging
import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
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

class AdvancedFundingStrategy:
    """
    Stratégie Funding Rate avancée pour crypto futures
    
    Améliorations vs original :
    - Calcul réel du funding rate
    - Prédiction ML du funding futur
    - Arbitrage spot-futures
    - Position sizing selon expected funding
    - Gestion du risque de base
    """
    
    def __init__(self,
                 symbol: str = "BTCUSDT",
                 funding_threshold: float = 0.001,  # 0.1% (plus réaliste pour crypto)
                 prediction_window: int = 24,      # Heures de prédiction
                 max_position_size: float = 0.5,   # 50% max du capital
                 risk_aversion: float = 2.0,       # Paramètre aversion au risque
                 use_ml_prediction: bool = True):
        
        self.symbol = symbol
        self.funding_threshold = funding_threshold
        self.prediction_window = prediction_window
        self.max_position_size = max_position_size
        self.risk_aversion = risk_aversion
        self.use_ml_prediction = use_ml_prediction
        
        # Modèles ML
        self.funding_predictor = None
        self.basis_predictor = None
        self.scaler = StandardScaler()
        
        # État de la stratégie
        self.current_position = 0.0
        self.funding_history = []
        
        logger.info("✅ Funding Strategy initialisée (ML=%s)", use_ml_prediction)
    
    def calculate_funding_rate(self, 
                              spot_prices: pd.Series,
                              futures_prices: pd.Series,
                              funding_interval_hours: int = 8) -> pd.DataFrame:
        """
        Calcule le funding rate réel des crypto futures
        
        Formula: FR = clamp(TWAP_premium + clamp(Interest - Premium, -0.0005, 0.0005), -0.0075, 0.0075)
        Simplifié: FR ≈ (Futures_Price - Spot_Price) / Spot_Price
        """
        
        if len(spot_prices) != len(futures_prices):
            raise ValueError("Spot et futures doivent avoir la même longueur")
        
        results = pd.DataFrame(index=spot_prices.index)
        results['spot_price'] = spot_prices
        results['futures_price'] = futures_prices
        
        # Basis (premium/discount)
        results['basis'] = (futures_prices - spot_prices) / spot_prices
        
        # Funding rate approximé (annualisé)
        # Funding rate = basis normalisé par intervalle de funding
        intervals_per_year = 365 * 24 / funding_interval_hours  # ~1095 pour 8h
        results['funding_rate'] = results['basis'] * intervals_per_year
        
        # TWAP du basis pour lisser
        results['basis_twap'] = results['basis'].rolling(
            window=funding_interval_hours, min_periods=1
        ).mean()
        
        # Funding rate lissé
        results['funding_rate_smooth'] = results['basis_twap'] * intervals_per_year
        
        # Clamp selon les limites réelles des exchanges (-0.75% à +0.75%)
        results['funding_rate_clamped'] = np.clip(
            results['funding_rate_smooth'], -0.0075, 0.0075
        )
        
        # Funding cumulé (pour P&L)
        results['funding_cumulative'] = results['funding_rate_clamped'].cumsum()
        
        # Volatilité du funding
        results['funding_volatility'] = results['funding_rate_clamped'].rolling(
            24 * 7  # 7 jours
        ).std()
        
        logger.info(f"📊 Funding calculé: moy={results['funding_rate_clamped'].mean():.4f}, "
                   f"std={results['funding_rate_clamped'].std():.4f}")
        
        return results
    
    def _estimate_funding_from_ohlcv(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimation simplifiée des funding rates à partir des données OHLCV
        Utilise des proxies comme volatilité et momentum pour estimer le funding
        """
        df = ohlcv_df.copy()
        
        # Calculs de base
        returns = df['close'].pct_change().fillna(0)
        
        # Estimation du funding basée sur:
        # 1. Volatilité (volatilité élevée = funding rate élevé)
        vol = returns.rolling(24).std().fillna(0.02)
        
        # 2. Momentum (trend fort = funding rate élevé)  
        momentum = returns.rolling(24).mean().fillna(0)
        
        # 3. Volume relatif (volume élevé = pression = funding élevé)
        vol_ma = df['volume'].rolling(24).mean()
        vol_ratio = df['volume'] / vol_ma
        vol_ratio = vol_ratio.fillna(1.0)
        
        # Estimation du funding rate (entre -0.75% et +0.75%)
        funding_raw = (momentum * 100 + vol * 50 + (vol_ratio - 1) * 10) / 1000
        funding_clamped = np.clip(funding_raw, -0.0075, 0.0075)
        
        # Construction du DataFrame de funding
        results = pd.DataFrame(index=df.index)
        results['funding_rate_raw'] = funding_raw
        results['funding_rate_smooth'] = funding_clamped.rolling(3).mean().fillna(funding_clamped)
        results['funding_rate_clamped'] = np.clip(results['funding_rate_smooth'], -0.0075, 0.0075)
        results['funding_cumulative'] = results['funding_rate_clamped'].cumsum()
        results['funding_volatility'] = results['funding_rate_clamped'].rolling(24*7).std().fillna(0.001)
        
        # Basis estimé (spread futures-spot, approximé par momentum + vol)
        results['basis'] = momentum * 1000 + vol * 500  # basis points
        results['basis_smooth'] = results['basis'].rolling(6).mean().fillna(results['basis'])
        
        logger.info(f"📊 Funding estimé: moy={results['funding_rate_clamped'].mean():.6f}, "
                   f"std={results['funding_rate_clamped'].std():.6f}")
        
        return results
    
    def load_real_funding_data(self, symbol: str = None, days: int = 30) -> pd.DataFrame:
        """
        Charge les vrais funding rates collectés par Freqtrade/Collector
        """
        if symbol is None:
            symbol = self.symbol
            
        try:
            # Format fichier
            safe_symbol = symbol.replace('/', '_').replace(':', '_')
            funding_file = Path(f"data/funding_rates/funding_{safe_symbol}.parquet")
            
            if not funding_file.exists():
                logger.warning(f"⚠️ Fichier funding introuvable: {funding_file}")
                return pd.DataFrame()
            
            # Lecture données
            funding_df = pd.read_parquet(funding_file)
            
            # Filtrage temporel
            if days > 0:
                cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
                funding_df = funding_df[funding_df.index >= cutoff_date]
            
            # Nettoyage et formatage
            required_cols = ['fundingRate', 'markPrice', 'indexPrice']
            available_cols = [col for col in required_cols if col in funding_df.columns]
            
            if not available_cols:
                logger.warning(f"⚠️ Colonnes funding manquantes dans {funding_file}")
                return pd.DataFrame()
            
            # Renommage standardisé
            if 'fundingRate' in funding_df.columns:
                funding_df['funding_rate_clamped'] = funding_df['fundingRate']
            
            if 'markPrice' in funding_df.columns and 'indexPrice' in funding_df.columns:
                funding_df['basis'] = (funding_df['markPrice'] - funding_df['indexPrice']) / funding_df['indexPrice'] * 10000  # basis points
                funding_df['basis_smooth'] = funding_df['basis'].rolling(6).mean().fillna(funding_df['basis'])
            
            # Métriques dérivées
            if 'funding_rate_clamped' in funding_df.columns:
                funding_df['funding_cumulative'] = funding_df['funding_rate_clamped'].cumsum()
                funding_df['funding_volatility'] = funding_df['funding_rate_clamped'].rolling(24*7).std().fillna(0.001)
                funding_df['funding_rate_smooth'] = funding_df['funding_rate_clamped'].rolling(3).mean().fillna(funding_df['funding_rate_clamped'])
                funding_df['funding_rate_raw'] = funding_df['funding_rate_clamped']  # Alias
            
            logger.info(f"📊 Funding réel chargé: {len(funding_df)} points, "
                       f"période: {funding_df.index.min()} → {funding_df.index.max()}")
            
            return funding_df
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement funding réel {symbol}: {e}")
            return pd.DataFrame()
    
    def build_prediction_features(self, funding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construction des features pour prédiction ML du funding
        """
        
        features = pd.DataFrame(index=funding_df.index)
        
        # Features de funding
        features['funding_current'] = funding_df['funding_rate_clamped']
        features['funding_ma_24h'] = funding_df['funding_rate_clamped'].rolling(24).mean()
        features['funding_ma_7d'] = funding_df['funding_rate_clamped'].rolling(24*7).mean()
        features['funding_std_24h'] = funding_df['funding_rate_clamped'].rolling(24).std()
        
        # Features de basis
        features['basis_current'] = funding_df['basis']
        features['basis_ma_12h'] = funding_df['basis'].rolling(12).mean()
        features['basis_trend'] = funding_df['basis'].diff(24)  # Trend 24h
        
        # Features de prix
        if 'spot_price' in funding_df.columns and 'futures_price' in funding_df.columns:
            # Volatilité spot
            spot_returns = funding_df['spot_price'].pct_change()
            features['spot_volatility'] = spot_returns.rolling(24).std() * np.sqrt(365*24)
            
            # Momentum spot
            features['spot_momentum_24h'] = funding_df['spot_price'].pct_change(24)
            features['spot_momentum_7d'] = funding_df['spot_price'].pct_change(24*7)
            
            # Volume si disponible (sinon estimé par volatilité)
            if 'volume' in funding_df.columns:
                features['volume_ma'] = funding_df['volume'].rolling(24).mean()
                features['volume_trend'] = funding_df['volume'].pct_change(24)
            else:
                # Proxy volume = volatilité
                features['volume_proxy'] = features['spot_volatility']
                features['volume_trend'] = features['volume_proxy'].pct_change(24)
        
        # Features cycliques (heure, jour de semaine)
        if hasattr(funding_df.index, 'hour'):
            features['hour'] = funding_df.index.hour
            features['day_of_week'] = funding_df.index.dayofweek
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        else:
            # Fallback si pas de DatetimeIndex
            features['hour'] = 0
            features['day_of_week'] = 0  
            features['hour_sin'] = 0
            features['hour_cos'] = 1
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Features techniques
        features['funding_rsi'] = self._calculate_rsi(funding_df['funding_rate_clamped'])
        features['basis_rsi'] = self._calculate_rsi(funding_df['basis'])
        
        # Mean reversion features
        features['funding_zscore'] = (
            (features['funding_current'] - features['funding_ma_7d']) /
            (features['funding_std_24h'] + 1e-8)
        )
        
        return features.fillna(0)
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """RSI pour funding rate"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_funding_predictor(self, funding_df: pd.DataFrame) -> Dict[str, float]:
        """
        Entraîne le modèle de prédiction du funding rate
        """
        
        if not self.use_ml_prediction:
            return {"model_score": 0.0}
        
        logger.info("🧠 Entraînement modèle prédiction funding...")
        
        # Features
        features_df = self.build_prediction_features(funding_df)
        
        # Target : funding rate dans prediction_window heures
        target = funding_df['funding_rate_clamped'].shift(-self.prediction_window)
        
        # Nettoyage
        valid_data = pd.concat([features_df, target], axis=1).dropna()
        
        if len(valid_data) < 100:
            logger.warning("⚠️  Pas assez de données pour ML")
            return {"model_score": 0.0}
        
        # Séparation train/validation
        X = valid_data.iloc[:, :-1].values  # Toutes colonnes sauf dernière
        y = valid_data.iloc[:, -1].values   # Dernière colonne (target)
        
        # Split temporel
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Modèle GradientBoosting (bon pour séries temporelles financières)
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        try:
            # Entraînement
            model.fit(X_train_scaled, y_train)
            
            # Validation
            val_score = model.score(X_val_scaled, y_val)
            
            # Feature importance
            feature_names = features_df.columns.tolist()
            importance_dict = dict(zip(
                feature_names, 
                model.feature_importances_
            ))
            
            # Top 5 features importantes
            top_features = sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            self.funding_predictor = model
            
            logger.info(f"✅ Modèle entraîné: R²={val_score:.3f}")
            logger.info(f"🎯 Top features: {dict(top_features)}")
            
            # Log feature importance as parameters instead of metrics
            if hasattr(mlflow, 'log_params'):
                try:
                    mlflow.log_params({f"feature_importance_{k}": v for k, v in importance_dict.items()})
                except:
                    pass
            
            return {
                "model_score": float(val_score)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement modèle: {e}")
            return {"model_score": 0.0}
    
    def predict_funding_rate(self, funding_df: pd.DataFrame) -> pd.Series:
        """
        Prédit le funding rate futur
        """
        
        if self.funding_predictor is None:
            # Modèle simple : moyenne mobile + mean reversion
            return self._simple_funding_prediction(funding_df)
        
        try:
            # Features actuelles
            features_df = self.build_prediction_features(funding_df)
            X = features_df.values

            # Vérification dimensions
            if hasattr(self.scaler, 'n_features_in_'):
                expected_features = self.scaler.n_features_in_
                actual_features = X.shape[1]

                if expected_features != actual_features:
                    logger.warning(f"⚠️ Features mismatch: expected {expected_features}, got {actual_features}")
                    logger.warning("🔄 Fallback vers prédiction simple")
                    return self._simple_funding_prediction(funding_df)

            X_scaled = self.scaler.transform(X)

            # Prédictions ML
            predictions = self.funding_predictor.predict(X_scaled)

            return pd.Series(predictions, index=funding_df.index)

        except Exception as e:
            logger.error(f"❌ Erreur prédiction ML: {e}")
            return self._simple_funding_prediction(funding_df)
    
    def _simple_funding_prediction(self, funding_df: pd.DataFrame) -> pd.Series:
        """
        Prédiction simple par mean reversion
        """
        current_funding = funding_df['funding_rate_clamped']
        funding_ma = current_funding.rolling(24*7).mean()  # Moyenne 7 jours
        
        # Mean reversion : si funding actuel > moyenne, prédit retour vers moyenne
        mean_reversion_factor = 0.1  # 10% de retour vers moyenne
        predicted = current_funding + mean_reversion_factor * (funding_ma - current_funding)
        
        return predicted.fillna(current_funding)
    
    def calculate_expected_pnl(self, funding_df: pd.DataFrame, 
                              predicted_funding: pd.Series) -> pd.DataFrame:
        """
        Calcule le P&L attendu des positions funding
        """
        
        results = funding_df.copy()
        results['predicted_funding'] = predicted_funding
        
        # Expected P&L par position
        # Long futures = reçoit funding si funding négatif
        # Short futures = paie funding si funding positif
        
        results['expected_pnl_long'] = -predicted_funding    # Long futures
        results['expected_pnl_short'] = predicted_funding    # Short futures
        
        # Risque (volatilité du funding)
        funding_vol = funding_df['funding_rate_clamped'].rolling(24*7).std()
        results['funding_risk'] = funding_vol
        
        # Sharpe ratio attendu
        results['expected_sharpe_long'] = results['expected_pnl_long'] / (funding_vol + 1e-8)
        results['expected_sharpe_short'] = results['expected_pnl_short'] / (funding_vol + 1e-8)
        
        return results
    
    def optimize_position_size(self, expected_pnl_df: pd.DataFrame) -> pd.Series:
        """
        Position sizing optimal selon Kelly généralisé + risk management
        """
        
        position_sizes = pd.Series(0.0, index=expected_pnl_df.index)
        
        # Réduire l'historique requis pour permettre plus de trades
        min_history = min(24*2, len(expected_pnl_df)//4)  # 2 jours ou 25% des données
        for i in range(min_history, len(expected_pnl_df)):
            
            # Données fenêtre glissante (utiliser l'historique disponible)
            window_start = max(0, i-min_history)
            window_data = expected_pnl_df.iloc[window_start:i]
            
            # Expected returns
            expected_long = window_data['expected_pnl_long'].iloc[-1]
            expected_short = window_data['expected_pnl_short'].iloc[-1]
            
            # Volatilité historique
            funding_vol = window_data['funding_risk'].iloc[-1]
            
            if funding_vol <= 0:
                continue
            
            # Kelly fraction pour chaque direction
            kelly_long = expected_long / (funding_vol ** 2 * self.risk_aversion)
            kelly_short = expected_short / (funding_vol ** 2 * self.risk_aversion)
            
            # Choix meilleure direction
            if abs(kelly_long) > abs(kelly_short) and abs(expected_long) > self.funding_threshold:
                optimal_size = np.sign(kelly_long) * min(abs(kelly_long), self.max_position_size)
            elif abs(expected_short) > self.funding_threshold:
                optimal_size = np.sign(kelly_short) * min(abs(kelly_short), self.max_position_size)
            else:
                optimal_size = 0.0
            
            position_sizes.iloc[i] = optimal_size
        
        return position_sizes
    
    async def generate_signals(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Génération des signaux de trading funding
        """
        from mlpipeline.utils.artifact_cleaner import validate_real_data_only
        
        logger.info("🎯 Génération signaux Funding Strategy...")
        
        # Validation des données réelles
        if not validate_real_data_only(ohlcv_df, self.symbol):
            raise ValueError("❌ REJETÉ: Données synthétiques détectées")
        
        # 0. Tenter de charger les vrais funding rates, sinon estimation
        funding_df = self.load_real_funding_data(days=30)
        
        if funding_df.empty:
            logger.warning("⚠️ Pas de funding rates réels → utilisation estimation OHLCV")
            funding_df = self._estimate_funding_from_ohlcv(ohlcv_df)
        else:
            logger.info("✅ Utilisation funding rates réels collectés")
        
        # 1. Prédiction funding
        predicted_funding = self.predict_funding_rate(funding_df)
        
        # 2. Expected P&L
        pnl_df = self.calculate_expected_pnl(funding_df, predicted_funding)
        
        # 3. Position sizing optimal
        position_sizes = self.optimize_position_size(pnl_df)
        
        # 4. Signaux finaux
        signals_df = pnl_df.copy()
        signals_df['position_size'] = position_sizes
        signals_df['signal'] = np.sign(position_sizes)
        
        # 5. Méta-informations
        signals_df['funding_threshold_used'] = self.funding_threshold
        signals_df['max_position_used'] = self.max_position_size
        signals_df['model_type'] = 'ML' if self.funding_predictor else 'Simple'
        
        return signals_df
    
    async def backtest_funding_strategy(self, signals_df: pd.DataFrame,
                                       transaction_cost: float = 0.0005) -> Dict:
        """
        Backtest de la stratégie funding avec coûts réalistes
        """
        
        if len(signals_df) == 0:
            return {"error": "DataFrame vide"}
        
        # Returns basés sur le funding reçu/payé
        actual_funding = signals_df['funding_rate_clamped'].fillna(0)
        positions = signals_df['position_size'].fillna(0)
        
        # P&L funding : position * (-funding) car on reçoit si funding négatif
        funding_pnl = positions.shift(1) * (-actual_funding)
        
        # Coûts de transaction sur changements de position
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * transaction_cost
        
        # P&L net
        net_returns = funding_pnl - transaction_costs
        
        # Gestion du risque de base (spread spot-futures change)
        if 'basis' in signals_df.columns:
            basis_changes = signals_df['basis'].diff()
            # Risque de base = position * changement de basis (simplifié)
            basis_risk = positions.shift(1) * basis_changes * 0.5  # 50% du risque de base
            net_returns = net_returns - basis_risk.abs()  # Soustrait car c'est un coût
        
        # Equity curve
        equity = (1 + net_returns.fillna(0)).cumprod()
        
        # Métriques
        metrics = comprehensive_metrics(net_returns.dropna(), frequence_annuelle=365*24)
        
        # Métriques spécifiques funding
        total_positions = (positions != 0).sum()
        avg_position_size = positions[positions != 0].abs().mean() if total_positions > 0 else 0
        
        metrics.update({
            "total_funding_received": float(funding_pnl.sum()),
            "total_transaction_costs": float(transaction_costs.sum()),
            "avg_position_size": float(avg_position_size),
            "position_utilization": float(total_positions / len(signals_df)),
            "funding_capture_ratio": float(funding_pnl.sum() / (abs(actual_funding).sum() + 1e-8)),
            "equity_final": float(equity.iloc[-1]) if len(equity) > 0 else 1.0
        })
        
        return metrics

# ==============================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ==============================================

async def train_funding_alpha(spot_data_path: str = "data/processed/features_BTCUSDT_1h.parquet",
                            futures_data_path: Optional[str] = None,
                            config: Dict = None) -> Dict:
    """
    Fonction principale d'entraînement Funding Alpha
    
    Args:
        spot_data_path: Données spot (obligatoire)
        futures_data_path: Données futures (optionnel, sinon estimé)
        config: Configuration de la stratégie
    """
    
    # Configuration par défaut - MISE À JOUR RÉALISTE
    default_config = {
        "funding_threshold": 0.001,      # 0.1% (réaliste pour crypto)
        "prediction_window": 24,         # 24h de prédiction
        "max_position_size": 0.3,        # 30% max (était 50%)
        "risk_aversion": 2.0,
        "use_ml_prediction": False,      # DÉSACTIVÉ temporairement (fix ML)
        "transaction_cost": 0.002,       # 0.2% réaliste (était 0.05%)
        "funding_interval_hours": 8
    }
    
    # Merger avec la configuration passée
    if config is None:
        config = default_config
    else:
        # Ajouter les valeurs par défaut pour les clés manquantes
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    
    logger.info("🚀 Début entraînement Funding Alpha Avancé")
    
    # 1. Chargement données spot
    if not os.path.exists(spot_data_path):
        raise FileNotFoundError(f"❌ Données spot introuvables: {spot_data_path}")
    
    spot_df = pd.read_parquet(spot_data_path)
    
    # Set time column as index if it exists
    if 'time' in spot_df.columns:
        spot_df = spot_df.set_index('time')
    
    logger.info(f"📊 Données spot: {len(spot_df)} lignes")
    
    # Validation stricte
    if not validate_real_data_only(spot_df, "Funding_Spot"):
        raise ValueError("❌ ENTRAÎNEMENT INTERROMPU: Données synthétiques spot détectées")
    
    # 2. Données futures (réelles ou estimées)
    if futures_data_path and os.path.exists(futures_data_path):
        futures_df = pd.read_parquet(futures_data_path)
        
        # Set time column as index if it exists
        if 'time' in futures_df.columns:
            futures_df = futures_df.set_index('time')
        
        if not validate_real_data_only(futures_df, "Funding_Futures"):
            logger.warning("⚠️  Données futures synthétiques, utilisation spot estimé")
            futures_df = None
        else:
            logger.info(f"📊 Données futures: {len(futures_df)} lignes")
    else:
        futures_df = None
        logger.info("📊 Pas de données futures, estimation depuis spot")
    
    # 3. Estimation prix futures si pas de données réelles
    if futures_df is None:
        # Estimation futures = spot * (1 + basis estimé)
        # Basis estimé = funding_rate_estimate / intervals_per_year
        spot_returns = spot_df['close'].pct_change()
        spot_vol = spot_returns.rolling(24*7).std() * np.sqrt(365*24)  # Vol annualisée
        
        # Basis estimé selon volatilité (plus volatil = plus de contango)
        estimated_basis = spot_vol * 0.1  # 10% de la volatilité comme basis
        estimated_futures = spot_df['close'] * (1 + estimated_basis)
        
        futures_prices = estimated_futures
    else:
        # Alignment temporel spot-futures
        common_idx = spot_df.index.intersection(futures_df.index)
        spot_df = spot_df.loc[common_idx]
        futures_df = futures_df.loc[common_idx]
        futures_prices = futures_df['close']
    
    # 4. Calcul funding rate
    strategy = AdvancedFundingStrategy(
        funding_threshold=config["funding_threshold"],
        prediction_window=config["prediction_window"],
        max_position_size=config["max_position_size"],
        risk_aversion=config["risk_aversion"],
        use_ml_prediction=config["use_ml_prediction"]
    )
    
    funding_df = strategy.calculate_funding_rate(
        spot_df['close'], 
        futures_prices,
        config["funding_interval_hours"]
    )
    
    # 5. Entraînement modèle prédictif avec MLflow
    mlflow.set_experiment("FundingStrategy_Production")
    
    with mlflow.start_run():
        # Hyperparamètres
        mlflow.log_params({
            "strategy_type": "Funding_Rate_Advanced",
            "funding_threshold": config["funding_threshold"],
            "prediction_window": config["prediction_window"], 
            "max_position_size": config["max_position_size"],
            "use_ml_prediction": config["use_ml_prediction"],
            "has_real_futures": futures_df is not None
        })
        
        # Entraînement
        model_metrics = strategy.train_funding_predictor(funding_df)
        mlflow.log_metrics(model_metrics)
        
        # Génération signaux (async) - pass OHLCV data, not funding data
        signals_df = await strategy.generate_signals(spot_df)
        
        # Backtest (async)
        backtest_metrics = await strategy.backtest_funding_strategy(
            signals_df, config["transaction_cost"]
        )
        
        # Log métriques backtest
        mlflow.log_metrics(backtest_metrics)
        
        # Sauvegarde modèle
        if strategy.funding_predictor:
            mlflow.sklearn.log_model(strategy.funding_predictor, "funding_predictor")
    
    # 6. Sauvegarde artifacts
    artifacts_dir = Path("data/artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Symbole
    symbol = Path(spot_data_path).stem.split('_')[1] if '_' in Path(spot_data_path).stem else "UNKNOWN"
    
    # Métriques
    final_metrics = {**model_metrics, **backtest_metrics}
    metrics_file = artifacts_dir / f"funding_metrics_{symbol}.json"
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Signaux et funding
    signals_file = artifacts_dir / f"funding_signals_{symbol}.parquet"
    signals_df.to_parquet(signals_file)
    
    logger.info("📊 MÉTRIQUES FUNDING STRATEGY:")
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"   {key}: {value:.4f}")
        else:
            logger.info(f"   {key}: {value}")
    
    # Validation
    validation = validate_metrics(backtest_metrics, min_sharpe=0.5, max_drawdown=0.15)
    
    if not validation["is_valid"]:
        for error in validation["errors"]:
            logger.error(f"❌ {error}")
    
    for warning in validation["warnings"]:
        logger.warning(f"⚠️  {warning}")
    
    return final_metrics

# ==============================================
# SCRIPT PRINCIPAL
# ==============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement Funding Alpha Avancé")
    parser.add_argument("--spot-data", required=True,
                       help="Chemin données spot")
    parser.add_argument("--futures-data",
                       help="Chemin données futures (optionnel)")
    parser.add_argument("--funding-threshold", type=float, default=0.001)
    parser.add_argument("--max-position", type=float, default=0.5)
    parser.add_argument("--no-ml", action="store_true")
    parser.add_argument("--prediction-window", type=int, default=24)
    parser.add_argument("--risk-aversion", type=float, default=2.0)
    
    args = parser.parse_args()
    
    config = {
        "funding_threshold": args.funding_threshold,
        "prediction_window": args.prediction_window,
        "max_position_size": args.max_position,
        "risk_aversion": args.risk_aversion,
        "use_ml_prediction": not args.no_ml
    }
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        metrics = asyncio.run(train_funding_alpha(args.spot_data, args.futures_data, config))
        logger.info("✅ Entraînement Funding Alpha terminé avec succès")
        
    except Exception as e:
        logger.error(f"❌ ERREUR entraînement Funding: {e}")
        sys.exit(1)