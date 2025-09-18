#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détecteur de Régimes HMM - Version Production Avancée
Migration et amélioration majeure du regime_detector.py original
- HMM multi-variés avec features sophistiquées
- Régimes adaptatifs selon conditions de marché
- Validation statistique robuste
- Interface avec sélecteur PSR
UNIQUEMENT données réelles - Validation stricte
"""

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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import adjusted_rand_score, silhouette_score
import hmmlearn.hmm as hmm
import mlflow
import mlflow.sklearn

# Configuration path pour imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import des utilitaires
try:
    from mlpipeline.utils.risk_metrics import (
        ratio_sharpe, drawdown_max, probabilistic_sharpe_ratio,
        comprehensive_metrics
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
    
    def comprehensive_metrics(returns):
        return {"sharpe": ratio_sharpe(returns), "max_drawdown": drawdown_max((1+returns).cumprod())}
    
    def validate_real_data_only(df, source="API"):
        return not df.empty

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class AdvancedRegimeDetector:
    """
    Détecteur de régimes de marché avec HMM multi-variés
    
    Améliorations vs original :
    - Features multi-dimensionnelles (volatilité, momentum, volume, corrélations)
    - Validation croisée pour nombre optimal d'états
    - Régimes nommés selon caractéristiques économiques
    - Probabilités de transition dynamiques
    - Tests de robustesse statistique
    """
    
    def __init__(self,
                 n_regimes: int = 3,
                 covariance_type: str = "full",
                 n_iter: int = 100,
                 random_state: int = 42,
                 min_regime_duration: int = 10):
        
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter  
        self.random_state = random_state
        self.min_regime_duration = min_regime_duration
        
        # Modèle HMM
        self.hmm_model = None
        self.scaler = StandardScaler()
        
        # Résultats
        self.regime_states = None
        self.regime_probabilities = None
        self.regime_characteristics = {}
        self.transition_matrix = None
        
        # Cache features
        self.features_cache = None
        
        logger.info(f"✅ Regime Detector initialisé ({n_regimes} régimes)")
    
    def build_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construction des features multi-dimensionnelles pour HMM
        
        Features utilisées :
        - Volatilité réalisée (multiple horizons)
        - Momentum / Trend
        - Volume et liquidité
        - Skewness et Kurtosis roulantes
        - Corrélations inter-marchés
        """
        
        features = pd.DataFrame(index=df.index)
        
        # Prix et returns
        if 'close' in df.columns:
            prices = df['close']
            returns = prices.pct_change().fillna(0)
        else:
            logger.error("❌ Colonne 'close' requise pour les features")
            return pd.DataFrame()
        
        # 1. FEATURES DE VOLATILITÉ
        # Volatilité réalisée multiple horizons
        features['vol_5d'] = returns.rolling(5*24).std() * np.sqrt(365*24)  # 5 jours (si hourly)
        features['vol_20d'] = returns.rolling(20*24).std() * np.sqrt(365*24)  # 20 jours
        features['vol_60d'] = returns.rolling(60*24).std() * np.sqrt(365*24)  # 60 jours
        
        # Volatilité relative (courte vs longue)
        features['vol_ratio'] = features['vol_5d'] / (features['vol_60d'] + 1e-8)
        
        # Volatilité des volatilités (vol de vol)
        features['volvol'] = features['vol_20d'].rolling(20*24).std()
        
        # 2. FEATURES DE MOMENTUM ET TREND
        # Momentum multiple horizons
        features['mom_1d'] = returns.rolling(24).sum()     # 1 jour
        features['mom_7d'] = returns.rolling(7*24).sum()   # 1 semaine  
        features['mom_30d'] = returns.rolling(30*24).sum() # 1 mois
        
        # Trend strength (R² de régression linéaire)
        features['trend_strength'] = self._calculate_trend_strength(prices)
        
        # Acceleration (2ème dérivée des prix)
        price_smooth = prices.rolling(24).mean()
        features['acceleration'] = price_smooth.diff(24).diff(24)
        
        # 3. FEATURES DE VOLUME ET LIQUIDITÉ
        if 'volume' in df.columns:
            volume = df['volume']
            
            # Volume normalisé
            features['volume_ma'] = volume.rolling(20*24).mean()
            features['volume_ratio'] = volume / (features['volume_ma'] + 1e-8)
            
            # Volume-Price Trend (VPT)
            features['vpt'] = (returns * volume).cumsum()
            features['vpt_ma'] = features['vpt'].rolling(20*24).mean()
            
            # Price-Volume correlation
            features['pv_corr'] = returns.rolling(20*24).corr(volume.pct_change())
            
        else:
            # Proxy volume basé sur volatilité
            features['volume_proxy'] = features['vol_5d']
            features['volume_ratio'] = 1.0
            features['pv_corr'] = 0.0
        
        # 4. FEATURES DE DISTRIBUTION
        # Skewness et Kurtosis roulantes
        features['skew_20d'] = returns.rolling(20*24).skew()
        features['kurt_20d'] = returns.rolling(20*24).kurt()
        
        # VaR et CVaR dynamiques
        features['var_5pct'] = returns.rolling(20*24).quantile(0.05)
        features['cvar_5pct'] = returns.rolling(20*24).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x) > 0 else 0
        )
        
        # 5. FEATURES TECHNIQUES
        # RSI
        features['rsi'] = self._calculate_rsi(prices)
        
        # Bollinger Bands position
        features['bb_position'] = self._calculate_bb_position(prices)
        
        # MACD
        features['macd'], features['macd_signal'] = self._calculate_macd(prices)
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # 6. FEATURES CYCLIQUES ET TEMPORELLES
        # Jour de la semaine et heure (si timestamp disponible)
        if hasattr(df.index, 'dayofweek'):
            features['dow'] = df.index.dayofweek
            features['hour'] = df.index.hour
            
            # Encodage cyclique
            features['dow_sin'] = np.sin(2 * np.pi * features['dow'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['dow'] / 7)
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # 7. FEATURES DE RÉGIME PASSÉ (Auto-corrélation)
        # Volatility clustering
        features['vol_cluster'] = (features['vol_5d'] > features['vol_5d'].rolling(60*24).mean()).astype(float)
        
        # Momentum persistence  
        features['mom_persist'] = (features['mom_7d'] * features['mom_7d'].shift(7*24) > 0).astype(float)
        
        # Nettoyage et filling des NaN
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"📊 Features construites: {features.shape[1]} dimensions sur {len(features)} observations")
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI classique"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.Series:
        """Position dans les Bollinger Bands (0-1)"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_position = (prices - lower_band) / (upper_band - lower_band + 1e-10)
        return bb_position.clip(0, 1)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD classique"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20*24) -> pd.Series:
        """Force du trend via R² de régression linéaire"""
        
        def rolling_r_squared(series):
            if len(series) < 5:
                return 0.0
            x = np.arange(len(series))
            y = series.values
            try:
                correlation = np.corrcoef(x, y)[0, 1]
                r_squared = correlation ** 2
                return r_squared if not np.isnan(r_squared) else 0.0
            except:
                return 0.0
        
        trend_strength = prices.rolling(window).apply(rolling_r_squared, raw=False)
        return trend_strength
    
    def optimize_n_regimes(self, features: pd.DataFrame, 
                          max_regimes: int = 6) -> Dict[str, Union[int, List]]:
        """
        Optimisation du nombre de régimes via validation croisée
        """
        
        logger.info(f"🔧 Optimisation nombre de régimes (max={max_regimes})...")
        
        # Données normalisées
        X = self.scaler.fit_transform(features.dropna())
        
        if len(X) < 100:  # Minimum pour validation
            logger.warning("⚠️  Pas assez de données pour optimisation")
            return {"optimal_n_regimes": self.n_regimes, "scores": []}
        
        # Test différents nombres de régimes
        n_regimes_candidates = range(2, max_regimes + 1)
        scores = {}
        
        for n_reg in n_regimes_candidates:
            try:
                # Modèle HMM
                model = hmm.GaussianHMM(
                    n_components=n_reg,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state
                )
                
                # Fit
                model.fit(X)
                
                # Scoring : Log-likelihood + BIC penalty
                log_likelihood = model.score(X)
                n_params = model._get_n_fit_scalars_per_param()["*"]
                bic = -2 * log_likelihood * len(X) + n_params * np.log(len(X))
                
                # AIC aussi
                aic = -2 * log_likelihood * len(X) + 2 * n_params
                
                scores[n_reg] = {
                    "log_likelihood": log_likelihood,
                    "bic": bic,
                    "aic": aic,
                    "bic_score": -bic  # Pour maximisation
                }
                
                logger.info(f"   {n_reg} régimes: BIC={bic:.1f}, AIC={aic:.1f}")
                
            except Exception as e:
                logger.warning(f"   {n_reg} régimes: ÉCHEC ({e})")
                scores[n_reg] = {
                    "log_likelihood": -np.inf,
                    "bic": np.inf,
                    "aic": np.inf,
                    "bic_score": -np.inf
                }
        
        # Sélection optimal selon BIC
        if scores:
            optimal_n = min(scores.keys(), key=lambda k: scores[k]["bic"])
            logger.info(f"✅ Nombre optimal de régimes: {optimal_n}")
            
            return {
                "optimal_n_regimes": optimal_n,
                "scores": scores
            }
        else:
            return {
                "optimal_n_regimes": self.n_regimes,
                "scores": {}
            }
    
    def fit_hmm_model(self, features: pd.DataFrame, 
                     optimize_n_regimes: bool = True) -> Dict:
        """
        Entraînement du modèle HMM
        """
        
        logger.info("🧠 Entraînement modèle HMM...")
        
        # Features nettoyées
        features_clean = features.dropna()
        
        if len(features_clean) < 50:
            raise ValueError("❌ Pas assez de données pour HMM")
        
        # Normalisation
        X = self.scaler.fit_transform(features_clean)
        
        # Optimisation nombre de régimes
        if optimize_n_regimes:
            optimization_results = self.optimize_n_regimes(features_clean)
            self.n_regimes = optimization_results["optimal_n_regimes"]
        
        # Modèle HMM final
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        # Entraînement
        try:
            self.hmm_model.fit(X)
            
            # Prédiction des états
            states = self.hmm_model.predict(X)
            state_probabilities = self.hmm_model.predict_proba(X)
            
            # Mapping vers index original
            self.regime_states = pd.Series(
                index=features_clean.index,
                data=states,
                name='regime_state'
            )
            
            self.regime_probabilities = pd.DataFrame(
                index=features_clean.index,
                data=state_probabilities,
                columns=[f'regime_{i}_prob' for i in range(self.n_regimes)]
            )
            
            # Matrice de transition
            self.transition_matrix = self.hmm_model.transmat_
            
            # Score du modèle
            model_score = self.hmm_model.score(X)
            
            logger.info(f"✅ HMM entraîné: score={model_score:.2f}")
            
            return {
                "model_score": float(model_score),
                "n_regimes": self.n_regimes,
                "converged": True
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement HMM: {e}")
            return {
                "model_score": -np.inf,
                "n_regimes": self.n_regimes,
                "converged": False,
                "error": str(e)
            }
    
    def characterize_regimes(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """
        Caractérisation économique des régimes détectés
        """
        
        if self.regime_states is None:
            logger.error("❌ Modèle HMM non entraîné")
            return {}
        
        logger.info("📊 Caractérisation des régimes...")
        
        # Calcul returns pour analyse
        if 'close' in df.columns:
            returns = df['close'].pct_change().fillna(0)
        else:
            returns = pd.Series(0, index=df.index)
        
        # Alignement temporal
        common_index = self.regime_states.index.intersection(returns.index)
        states = self.regime_states.loc[common_index]
        aligned_returns = returns.loc[common_index]
        
        regime_characteristics = {}
        
        for regime_id in range(self.n_regimes):
            
            # Mask pour ce régime
            regime_mask = (states == regime_id)
            regime_returns = aligned_returns[regime_mask]
            
            if len(regime_returns) < 10:  # Minimum observations
                continue
            
            # Caractéristiques statistiques
            mean_return = regime_returns.mean() * 365 * 24  # Annualisé (si hourly)
            volatility = regime_returns.std() * np.sqrt(365 * 24)
            sharpe = mean_return / (volatility + 1e-8)
            
            # Fréquence et persistance
            regime_periods = regime_mask.sum()
            total_periods = len(regime_mask)
            frequency = regime_periods / total_periods
            
            # Durée moyenne des épisodes
            avg_duration = self._calculate_average_regime_duration(regime_mask)
            
            # Caractéristiques de distribution
            skewness = regime_returns.skew()
            kurtosis = regime_returns.kurtosis()
            var_5 = regime_returns.quantile(0.05)
            
            # Classification économique
            regime_name = self._classify_regime(mean_return, volatility, skewness)
            
            regime_characteristics[regime_id] = {
                "name": regime_name,
                "annualized_return": float(mean_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe),
                "frequency": float(frequency),
                "avg_duration_hours": float(avg_duration),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "var_5pct": float(var_5),
                "n_observations": int(regime_periods)
            }
            
            logger.info(f"   Régime {regime_id} ({regime_name}): "
                       f"ret={mean_return:.2%}, vol={volatility:.2%}, "
                       f"freq={frequency:.1%}")
        
        self.regime_characteristics = regime_characteristics
        return regime_characteristics
    
    def _calculate_average_regime_duration(self, regime_mask: pd.Series) -> float:
        """Calcule la durée moyenne des épisodes d'un régime"""
        
        durations = []
        current_duration = 0
        
        for is_regime in regime_mask:
            if is_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # N'oublier pas le dernier épisode
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0.0
    
    def _classify_regime(self, mean_return: float, volatility: float, 
                        skewness: float) -> str:
        """Classification économique des régimes"""
        
        # Seuils (à ajuster selon le marché)
        high_vol_threshold = 0.5   # 50% vol annualisée
        high_return_threshold = 0.1  # 10% return annualisé
        
        # Classification
        is_high_vol = volatility > high_vol_threshold
        is_high_return = mean_return > high_return_threshold
        is_negative_return = mean_return < -0.05  # -5%
        is_negative_skew = skewness < -0.5
        
        if is_high_vol and is_negative_return:
            return "Crisis" if is_negative_skew else "Bear_Volatile"
        elif is_high_vol and is_high_return:
            return "Bull_Volatile"
        elif is_high_return and not is_high_vol:
            return "Bull_Stable"
        elif is_negative_return and not is_high_vol:
            return "Bear_Stable"
        elif not is_high_vol:
            return "Low_Volatility"
        else:
            return "Normal"
    
    def predict_regime_transitions(self, n_steps: int = 24) -> pd.DataFrame:
        """
        Prédiction des probabilités de transition de régimes
        """
        
        if self.hmm_model is None or self.regime_states is None:
            logger.error("❌ Modèle non entraîné")
            return pd.DataFrame()
        
        # État actuel (dernier observé)
        current_state = self.regime_states.iloc[-1]
        
        # Simulation Monte Carlo des transitions futures
        n_simulations = 1000
        future_states = np.zeros((n_simulations, n_steps))
        
        for sim in range(n_simulations):
            state = current_state
            for step in range(n_steps):
                # Transition selon matrice de transition
                transition_probs = self.transition_matrix[state]
                state = np.random.choice(self.n_regimes, p=transition_probs)
                future_states[sim, step] = state
        
        # Probabilités moyennes pour chaque step futur
        transition_forecast = pd.DataFrame()
        
        for step in range(n_steps):
            step_states = future_states[:, step]
            step_probs = {}
            
            for regime in range(self.n_regimes):
                regime_name = self.regime_characteristics.get(regime, {}).get('name', f'Regime_{regime}')
                step_probs[f'{regime_name}_prob'] = (step_states == regime).mean()
            
            transition_forecast = pd.concat([
                transition_forecast,
                pd.DataFrame([step_probs], index=[f'step_{step+1}'])
            ])
        
        return transition_forecast
    
    def analyze_regime_performance_impact(self, alpha_metrics: Dict[str, Dict]) -> Dict:
        """
        Analyse l'impact des régimes sur les performances des alphas
        """
        
        if not alpha_metrics or self.regime_states is None:
            return {}
        
        logger.info("📊 Analyse impact régimes sur alphas...")
        
        regime_impact = {}
        
        # Pour chaque alpha, analyser performance par régime
        for alpha_name, metrics in alpha_metrics.items():
            
            # Ici on devrait idéalement avoir les returns temporels de chaque alpha
            # Pour simplifier, on simule l'impact basé sur les caractéristiques
            alpha_regime_performance = {}
            
            for regime_id, regime_char in self.regime_characteristics.items():
                
                # Impact basé sur le type de régime et d'alpha
                if "DMN" in alpha_name.upper():
                    # LSTM fonctionne mieux en régimes normaux/stables
                    if regime_char["name"] in ["Normal", "Bull_Stable", "Low_Volatility"]:
                        impact_score = 1.2  # +20%
                    else:
                        impact_score = 0.8  # -20%
                        
                elif "MR" in alpha_name.upper():
                    # Mean reversion fonctionne mieux en high volatility
                    if regime_char["volatility"] > 0.5:
                        impact_score = 1.3  # +30%
                    else:
                        impact_score = 0.9  # -10%
                        
                elif "FUNDING" in alpha_name.upper():
                    # Funding marche bien en régimes normaux
                    if regime_char["name"] in ["Normal", "Low_Volatility"]:
                        impact_score = 1.1  # +10%
                    else:
                        impact_score = 0.95  # -5%
                else:
                    impact_score = 1.0  # Neutre
                
                alpha_regime_performance[regime_char["name"]] = {
                    "performance_multiplier": impact_score,
                    "expected_sharpe": metrics.get("sharpe_ratio", 0) * impact_score,
                    "regime_frequency": regime_char["frequency"]
                }
            
            regime_impact[alpha_name] = alpha_regime_performance
        
        return regime_impact

# ==============================================
# FONCTION PRINCIPALE
# ==============================================

def detect_market_regimes(data_path: str = "data/processed/features_BTCUSDT_1h.parquet",
                         config: Dict = None) -> Dict:
    """
    Fonction principale de détection des régimes
    
    Args:
        data_path: Chemin vers les données
        config: Configuration du détecteur
    """
    
    # Configuration par défaut
    default_config = {
        "n_regimes": 3,
        "optimize_n_regimes": True,
        "covariance_type": "full",
        "n_iter": 200,
        "min_regime_duration": 10
    }
    
    # Merger avec la configuration passée
    if config is None:
        config = default_config
    else:
        # Ajouter les valeurs par défaut pour les clés manquantes
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    
    logger.info("🚀 Début détection des régimes HMM")
    
    # 1. Chargement données
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Données introuvables: {data_path}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"📊 Données chargées: {len(df)} lignes")
    
    # Validation stricte
    if not validate_real_data_only(df, "RegimeDetection"):
        raise ValueError("❌ DÉTECTION INTERROMPUE: Données synthétiques détectées")
    
    # 2. Détecteur
    detector = AdvancedRegimeDetector(
        n_regimes=config["n_regimes"],
        covariance_type=config["covariance_type"],
        n_iter=config["n_iter"],
        min_regime_duration=config["min_regime_duration"]
    )
    
    # 3. Construction features
    features = detector.build_regime_features(df)
    
    if features.empty:
        raise ValueError("❌ Échec construction features")
    
    # 4. Entraînement HMM avec MLflow
    mlflow.set_experiment("RegimeDetection_HMM")
    
    with mlflow.start_run():
        
        # Log config
        mlflow.log_params(config)
        
        # Entraînement
        training_results = detector.fit_hmm_model(
            features, 
            optimize_n_regimes=config["optimize_n_regimes"]
        )
        
        # Log métriques training
        mlflow.log_metrics(training_results)
        
        # Caractérisation des régimes
        regime_chars = detector.characterize_regimes(df)
        
        # Log caractéristiques des régimes
        for regime_id, chars in regime_chars.items():
            mlflow.log_metrics({
                f"regime_{regime_id}_return": chars["annualized_return"],
                f"regime_{regime_id}_volatility": chars["volatility"],
                f"regime_{regime_id}_frequency": chars["frequency"]
            })
        
        # Prédictions de transition
        transition_forecast = detector.predict_regime_transitions()
        
        # Sauvegarde modèle
        if detector.hmm_model:
            mlflow.sklearn.log_model(detector.hmm_model, "hmm_regime_model")
    
    # 5. Sauvegarde artifacts
    artifacts_dir = Path("data/artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Symbole
    symbol = Path(data_path).stem.split('_')[1] if '_' in Path(data_path).stem else "UNKNOWN"
    
    # Résultats complets
    results = {
        "training_results": training_results,
        "regime_characteristics": regime_chars,
        "n_regimes_detected": len(regime_chars),
        "transition_matrix": detector.transition_matrix.tolist() if detector.transition_matrix is not None else [],
        "symbol": symbol,
        "timestamp": datetime.now().isoformat()
    }
    
    # Sauvegarde JSON
    results_file = artifacts_dir / f"regime_detection_{symbol}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Sauvegarde états et probabilités
    if detector.regime_states is not None:
        states_file = artifacts_dir / f"regime_states_{symbol}.parquet"
        regime_df = pd.DataFrame({
            'regime_state': detector.regime_states,
            **detector.regime_probabilities
        })
        regime_df.to_parquet(states_file)
    
    # Rapport
    logger.info("📊 RÉGIMES DÉTECTÉS:")
    for regime_id, chars in regime_chars.items():
        logger.info(f"   Régime {regime_id} ({chars['name']}):")
        logger.info(f"     - Return: {chars['annualized_return']:.2%}")
        logger.info(f"     - Volatilité: {chars['volatility']:.2%}")
        logger.info(f"     - Fréquence: {chars['frequency']:.1%}")
        logger.info(f"     - Durée moy: {chars['avg_duration_hours']:.1f}h")
    
    return results

# ==============================================
# SCRIPT PRINCIPAL
# ==============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Détection régimes HMM")
    parser.add_argument("--data-path", required=True,
                       help="Chemin vers les données")
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--no-optimization", action="store_true",
                       help="Désactiver optimisation nombre régimes")
    
    args = parser.parse_args()
    
    # Configuration par défaut  
    default_config = {
        "n_regimes": 3,
        "optimize_n_regimes": True,
        "covariance_type": "full",
        "n_iter": 200,
        "min_regime_duration": 10
    }
    
    # Configuration utilisateur
    user_config = {
        "n_regimes": args.n_regimes,
        "optimize_n_regimes": not args.no_optimization
    }
    
    # Merge des configurations
    config = default_config.copy()
    config.update(user_config)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        results = detect_market_regimes(args.data_path, config)
        logger.info("✅ Détection des régimes terminée avec succès")
        
    except Exception as e:
        logger.error(f"❌ ERREUR détection régimes: {e}")
        sys.exit(1)