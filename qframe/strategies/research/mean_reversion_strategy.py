"""
Adaptive Mean Reversion Strategy
===============================

Migration de votre stratégie Mean Reversion avec améliorations ML
et détection de régimes. Préserve toute la logique sophistiquée.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from qframe.core.interfaces import (
    BaseStrategy,
    Signal,
    SignalAction,
    TimeFrame,
    MetricsCollector
)
from qframe.core.container import injectable

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionConfig:
    """Configuration pour stratégie Mean Reversion"""
    lookback_short: int = 10
    lookback_long: int = 50
    z_entry_base: float = 1.0  # Seuil d'entrée en z-score
    z_exit_base: float = 0.2   # Seuil de sortie en z-score
    regime_window: int = 252
    use_ml_optimization: bool = True
    position_size: float = 0.02
    max_position_days: int = 5
    risk_per_trade: float = 0.02


class RegimeDetector:
    """Détecteur de régimes de marché basé sur la volatilité"""

    def __init__(self, window: int = 252):
        self.window = window

    def detect_regimes(self, df: pd.DataFrame) -> pd.Series:
        """
        Détecte les régimes de marché:
        - low_vol: Volatilité < 25ème percentile
        - high_vol: Volatilité > 75ème percentile
        - normal: Entre les deux
        """
        # Calcul volatilité réalisée
        if "returns" in df.columns:
            returns = df["returns"]
        elif "close" in df.columns:
            returns = df["close"].pct_change().fillna(0)
        else:
            return pd.Series("normal", index=df.index)

        # Volatilité rolling
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)  # Annualisée

        # Percentiles dynamiques
        vol_25 = rolling_vol.rolling(self.window).quantile(0.25)
        vol_75 = rolling_vol.rolling(self.window).quantile(0.75)

        # Classification
        regimes = pd.Series("normal", index=df.index)
        regimes[rolling_vol <= vol_25] = "low_vol"
        regimes[rolling_vol >= vol_75] = "high_vol"

        return regimes


class MLOptimizer:
    """Optimiseur ML pour paramètres adaptatifs"""

    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare features pour ML"""
        features = pd.DataFrame(index=df.index)

        if "close" in df.columns:
            # Features de base
            returns = df["close"].pct_change().fillna(0)
            features["returns"] = returns
            features["volatility_20"] = returns.rolling(20).std()
            features["momentum_10"] = df["close"] / df["close"].shift(10) - 1
            features["momentum_5"] = df["close"] / df["close"].shift(5) - 1

            # RSI approché
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

            # Bollinger position
            sma_20 = df["close"].rolling(20).mean()
            std_20 = df["close"].rolling(20).std()
            features["bb_position"] = (df["close"] - sma_20) / (2 * std_20)

        return features.fillna(0)

    def train(self, df: pd.DataFrame, target_col: str = "future_returns"):
        """Entraîne le modèle ML pour optimisation"""
        try:
            features_df = self.prepare_features(df)

            # Target : returns futurs
            if "returns" in features_df.columns:
                target = features_df["returns"].shift(-1).fillna(0)
            else:
                return False

            # Suppression des NaN
            mask = ~(features_df.isna().any(axis=1) | target.isna())
            X = features_df[mask]
            y = target[mask]

            if len(X) < 100:
                logger.warning("Pas assez de données pour entraînement ML")
                return False

            # Normalisation
            X_scaled = self.scaler.fit_transform(X)

            # Modèle Random Forest
            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            # Entraînement avec validation temporelle
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                self.model.fit(X_train, y_train)
                pred = self.model.predict(X_val)
                score = mean_squared_error(y_val, pred)
                scores.append(score)

            # Entraînement final
            self.model.fit(X_scaled, y)
            self.is_trained = True

            logger.info(f"ML optimizer entraîné, MSE moyen: {np.mean(scores):.6f}")
            return True

        except Exception as e:
            logger.error(f"Erreur entraînement ML: {e}")
            return False

    def predict_optimal_thresholds(self, current_features: pd.Series) -> Dict[str, float]:
        """Prédit les seuils optimaux pour la situation actuelle"""
        if not self.is_trained or self.model is None:
            return {"z_entry": 1.0, "z_exit": 0.2}

        try:
            # Préparer features
            features_array = current_features.values.reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)

            # Prédiction (simplifié pour la démo)
            prediction = self.model.predict(features_scaled)[0]

            # Adapter seuils selon prédiction
            base_entry = 1.0
            base_exit = 0.2

            # Ajustement basé sur la prédiction
            volatility_factor = abs(prediction) * 2  # Plus de vol = seuils plus élevés

            z_entry = base_entry + volatility_factor * 0.5
            z_exit = base_exit + volatility_factor * 0.1

            return {
                "z_entry": min(z_entry, 3.0),  # Cap maximum
                "z_exit": min(z_exit, 1.0)
            }

        except Exception as e:
            logger.error(f"Erreur prédiction seuils: {e}")
            return {"z_entry": 1.0, "z_exit": 0.2}


@injectable
class MeanReversionStrategy(BaseStrategy):
    """
    Stratégie Mean Reversion adaptative migrée

    Features:
    - Détection de régimes de volatilité
    - Optimisation ML des paramètres
    - Multi-timeframes
    - Position sizing Kelly
    """

    def __init__(
        self,
        config: MeanReversionConfig = None,
        metrics_collector: MetricsCollector = None
    ):
        self.config = config or MeanReversionConfig()
        self.metrics_collector = metrics_collector

        super().__init__(
            name="Adaptive_Mean_Reversion",
            parameters=self.config.__dict__
        )

        # Composants
        self.regime_detector = RegimeDetector(self.config.regime_window)
        self.ml_optimizer = MLOptimizer() if self.config.use_ml_optimization else None

        # État de la stratégie
        self.current_position = 0  # -1, 0, 1
        self.entry_price = None
        self.entry_time = None
        self.current_regime = "normal"

        logger.info(f"Mean Reversion Strategy initialisée: {self.config}")

    def generate_signals(
        self,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """Génère signaux Mean Reversion avec adaptation de régime"""

        if len(data) < max(self.config.lookback_long, 50):
            logger.warning("Pas assez de données pour Mean Reversion")
            return []

        try:
            # Préparer données
            if features is not None:
                work_data = features.copy()
            else:
                work_data = self._prepare_features(data)

            # Détecter régime actuel
            regimes = self.regime_detector.detect_regimes(work_data)
            self.current_regime = regimes.iloc[-1] if len(regimes) > 0 else "normal"

            # Entraîner ML si nécessaire
            if self.ml_optimizer and not self.ml_optimizer.is_trained:
                self.ml_optimizer.train(work_data)

            # Calculer signaux
            signals = self._calculate_mean_reversion_signals(work_data, data.iloc[-1])

            # Métriques
            if self.metrics_collector and signals:
                self.metrics_collector.record_metric(
                    "mean_reversion_signals",
                    len(signals),
                    {"regime": self.current_regime}
                )

            return signals

        except Exception as e:
            logger.error(f"Erreur génération signaux Mean Reversion: {e}")
            return []

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare features si pas déjà fait"""
        if self._feature_processor is not None:
            return self._feature_processor.process(data)

        # Features de base
        features = pd.DataFrame(index=data.index)

        if "close" in data.columns:
            features["close"] = data["close"]
            features["returns"] = data["close"].pct_change().fillna(0)
            features["volume"] = data.get("volume", 1.0)

            # SMA court et long
            features["sma_short"] = data["close"].rolling(self.config.lookback_short).mean()
            features["sma_long"] = data["close"].rolling(self.config.lookback_long).mean()

            # Volatilité
            features["volatility"] = features["returns"].rolling(20).std()

        return features.fillna(method='ffill').fillna(0)

    def _calculate_mean_reversion_signals(
        self,
        data: pd.DataFrame,
        last_candle: pd.Series
    ) -> List[Signal]:
        """Calcule signaux mean reversion avec z-score adaptatif"""

        signals = []

        try:
            if len(data) < self.config.lookback_long:
                return signals

            # Z-score calculation
            close_prices = data["close"].tail(self.config.lookback_long)
            current_price = close_prices.iloc[-1]
            mean_price = close_prices.mean()
            std_price = close_prices.std()

            if std_price == 0:
                return signals

            z_score = (current_price - mean_price) / std_price

            # Ajuster seuils selon régime et ML
            thresholds = self._get_adaptive_thresholds(data)
            z_entry = thresholds["z_entry"]
            z_exit = thresholds["z_exit"]

            # Ajustement selon régime
            regime_multiplier = self._get_regime_multiplier(self.current_regime)
            z_entry *= regime_multiplier
            z_exit *= regime_multiplier

            # Logique de signal
            signal = None

            # Conditions d'entrée
            if self.current_position == 0:
                if z_score > z_entry:  # Prix trop haut -> vendre (mean reversion)
                    signal = self._create_signal(
                        SignalAction.SELL,
                        last_candle,
                        abs(z_score) / z_entry,  # Strength basée sur l'excès
                        {"z_score": z_score, "threshold": z_entry, "regime": self.current_regime}
                    )
                    self.current_position = -1

                elif z_score < -z_entry:  # Prix trop bas -> acheter
                    signal = self._create_signal(
                        SignalAction.BUY,
                        last_candle,
                        abs(z_score) / z_entry,
                        {"z_score": z_score, "threshold": z_entry, "regime": self.current_regime}
                    )
                    self.current_position = 1

            # Conditions de sortie
            elif self.current_position != 0:
                should_exit = False

                # Sortie par reversion
                if abs(z_score) < z_exit:
                    should_exit = True

                # Sortie par temps maximum
                if self.entry_time and len(data) - self.entry_time > self.config.max_position_days:
                    should_exit = True

                if should_exit:
                    action = SignalAction.CLOSE_SHORT if self.current_position == -1 else SignalAction.CLOSE_LONG
                    signal = self._create_signal(
                        action,
                        last_candle,
                        0.8,  # Force de sortie
                        {"z_score": z_score, "exit_reason": "reversion", "regime": self.current_regime}
                    )
                    self.current_position = 0

            if signal:
                signals.append(signal)

        except Exception as e:
            logger.error(f"Erreur calcul signaux mean reversion: {e}")

        return signals

    def _get_adaptive_thresholds(self, data: pd.DataFrame) -> Dict[str, float]:
        """Récupère seuils adaptatifs via ML ou valeurs par défaut"""
        if self.ml_optimizer and self.ml_optimizer.is_trained:
            try:
                current_features = self.ml_optimizer.prepare_features(data).iloc[-1]
                return self.ml_optimizer.predict_optimal_thresholds(current_features)
            except Exception as e:
                logger.warning(f"Erreur ML thresholds: {e}")

        return {
            "z_entry": self.config.z_entry_base,
            "z_exit": self.config.z_exit_base
        }

    def _get_regime_multiplier(self, regime: str) -> float:
        """Multiplicateur selon le régime de volatilité"""
        multipliers = {
            "low_vol": 0.7,   # Seuils plus sensibles en basse volatilité
            "normal": 1.0,    # Seuils normaux
            "high_vol": 1.5   # Seuils plus élevés en haute volatilité
        }
        return multipliers.get(regime, 1.0)

    def _create_signal(
        self,
        action: SignalAction,
        candle: pd.Series,
        strength: float,
        metadata: Dict[str, Any]
    ) -> Signal:
        """Crée un signal avec métadonnées"""
        return Signal(
            timestamp=datetime.now(),
            symbol=candle.get("symbol", "UNKNOWN"),
            action=action,
            strength=min(strength, 1.0),
            price=candle.get("close"),
            size=self.config.position_size,
            metadata={
                "strategy": self.name,
                **metadata
            }
        )

    def get_strategy_state(self) -> Dict[str, Any]:
        """État actuel de la stratégie"""
        return {
            "current_position": self.current_position,
            "current_regime": self.current_regime,
            "entry_price": self.entry_price,
            "ml_trained": self.ml_optimizer.is_trained if self.ml_optimizer else False,
            "config": self.config.__dict__
        }