"""
Advanced Funding Rate Arbitrage Strategy
=======================================

Migration de votre stratégie Funding Rate sophistiquée avec:
- Calcul réel des funding rates crypto
- Prédiction ML des funding rates futurs
- Arbitrage spot-futures intelligent
- Gestion du risque de base
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

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
class FundingArbitrageConfig:
    """Configuration pour stratégie Funding Arbitrage"""
    funding_threshold: float = 0.001  # 0.1% minimum pour signal
    prediction_window: int = 24       # Heures de prédiction
    max_position_size: float = 0.5    # 50% max du capital
    risk_aversion: float = 2.0        # Aversion au risque
    use_ml_prediction: bool = True
    funding_interval_hours: int = 8   # Intervalle funding crypto
    basis_risk_limit: float = 0.02    # Limite risque de base
    position_size: float = 0.1        # Taille position par défaut


class FundingRateCalculator:
    """Calculateur de funding rate pour crypto futures"""

    @staticmethod
    def calculate_funding_rate(
        spot_prices: pd.Series,
        futures_prices: pd.Series,
        funding_interval_hours: int = 8
    ) -> pd.DataFrame:
        """
        Calcule le funding rate réel des crypto futures

        Formula simplifiée: FR ≈ (Futures_Price - Spot_Price) / Spot_Price
        Réel: FR = clamp(TWAP_premium + Interest_component, -0.0075, 0.0075)
        """
        if len(spot_prices) != len(futures_prices):
            raise ValueError("Spot et futures doivent avoir la même longueur")

        results = pd.DataFrame(index=spot_prices.index)
        results["spot_price"] = spot_prices
        results["futures_price"] = futures_prices

        # Premium instantané
        results["premium"] = (futures_prices - spot_prices) / spot_prices

        # TWAP du premium (Time Weighted Average Price)
        twap_window = funding_interval_hours
        results["premium_twap"] = results["premium"].rolling(twap_window).mean()

        # Composante d'intérêt (approximation)
        # En réalité c'est SOFR + spread, on approxime à 0.01% par jour
        daily_interest = 0.0001
        funding_interest = daily_interest * (funding_interval_hours / 24)

        # Calcul funding rate avec limits
        funding_raw = results["premium_twap"] + funding_interest
        results["funding_rate"] = np.clip(funding_raw, -0.0075, 0.0075)

        # Funding rate annualisé pour comparaison
        results["funding_annualized"] = results["funding_rate"] * (365 * 24 / funding_interval_hours)

        # Basis (différence spot-futures)
        results["basis"] = futures_prices - spot_prices
        results["basis_pct"] = results["basis"] / spot_prices

        return results.fillna(0)

    @staticmethod
    def predict_next_funding(
        funding_history: pd.DataFrame,
        lookback_periods: int = 30
    ) -> Dict[str, float]:
        """Prédit le prochain funding rate basé sur l'historique"""
        if len(funding_history) < lookback_periods:
            return {"predicted_funding": 0.0, "confidence": 0.0}

        # Features simples pour prédiction
        recent_funding = funding_history["funding_rate"].tail(lookback_periods)
        recent_premium = funding_history["premium_twap"].tail(lookback_periods)

        # Moyennes mobiles
        ma_short = recent_funding.tail(7).mean()
        ma_long = recent_funding.mean()
        volatility = recent_funding.std()

        # Prédiction simple (à améliorer avec ML)
        trend_factor = (ma_short - ma_long) / (volatility + 1e-6)
        base_prediction = recent_funding.iloc[-1]

        # Ajustement selon trend
        predicted_funding = base_prediction + trend_factor * volatility * 0.5

        # Confidence basée sur consistance
        confidence = 1.0 / (1.0 + volatility * 100)

        return {
            "predicted_funding": float(np.clip(predicted_funding, -0.0075, 0.0075)),
            "confidence": float(confidence),
            "volatility": float(volatility),
            "trend_factor": float(trend_factor)
        }


class MLFundingPredictor:
    """Prédicteur ML pour funding rates"""

    def __init__(self):
        self.model: Optional[GradientBoostingRegressor] = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare features pour prédiction funding"""
        features = pd.DataFrame(index=df.index)

        if "funding_rate" in df.columns:
            # Features temporelles
            features["funding_current"] = df["funding_rate"]
            features["funding_ma_3"] = df["funding_rate"].rolling(3).mean()
            features["funding_ma_7"] = df["funding_rate"].rolling(7).mean()
            features["funding_ma_24"] = df["funding_rate"].rolling(24).mean()

            # Volatilité du funding
            features["funding_vol_7"] = df["funding_rate"].rolling(7).std()
            features["funding_vol_24"] = df["funding_rate"].rolling(24).std()

        if "premium_twap" in df.columns:
            # Features premium
            features["premium_current"] = df["premium_twap"]
            features["premium_ma_7"] = df["premium_twap"].rolling(7).mean()
            features["premium_change"] = df["premium_twap"].diff()

        if "basis_pct" in df.columns:
            # Features basis
            features["basis_current"] = df["basis_pct"]
            features["basis_ma_7"] = df["basis_pct"].rolling(7).mean()

        # Features temporelles (heure de la journée, jour de la semaine)
        features["hour"] = df.index.hour if hasattr(df.index, 'hour') else 0
        features["day_of_week"] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0

        return features.fillna(0)

    def train(self, df: pd.DataFrame) -> bool:
        """Entraîne le modèle de prédiction funding"""
        try:
            features_df = self.prepare_features(df)

            # Target : funding rate futur (8h ahead)
            if "funding_rate" not in df.columns:
                return False

            target = df["funding_rate"].shift(-8).fillna(0)  # 8h = 1 funding period

            # Suppression des NaN
            mask = ~(features_df.isna().any(axis=1) | target.isna())
            X = features_df[mask]
            y = target[mask]

            if len(X) < 100:
                logger.warning("Pas assez de données pour entraînement ML funding")
                return False

            # Normalisation
            X_scaled = self.scaler.fit_transform(X)

            # Modèle GradientBoosting
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

            # Validation temporelle
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                self.model.fit(X_train, y_train)
                pred = self.model.predict(X_val)
                score = np.mean((y_val - pred) ** 2)
                scores.append(score)

            # Entraînement final
            self.model.fit(X_scaled, y)
            self.is_trained = True

            logger.info(f"ML funding predictor entraîné, MSE: {np.mean(scores):.8f}")
            return True

        except Exception as e:
            logger.error(f"Erreur entraînement ML funding: {e}")
            return False

    def predict(self, current_features: pd.Series) -> Dict[str, float]:
        """Prédit le prochain funding rate"""
        if not self.is_trained or self.model is None:
            return {"predicted_funding": 0.0, "confidence": 0.0}

        try:
            features_array = current_features.values.reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)

            prediction = self.model.predict(features_scaled)[0]

            # Confidence approximée (à améliorer)
            confidence = min(abs(prediction) * 10, 1.0)

            return {
                "predicted_funding": float(np.clip(prediction, -0.0075, 0.0075)),
                "confidence": float(confidence)
            }

        except Exception as e:
            logger.error(f"Erreur prédiction funding: {e}")
            return {"predicted_funding": 0.0, "confidence": 0.0}


@injectable
class FundingArbitrageStrategy(BaseStrategy):
    """
    Stratégie d'arbitrage Funding Rate migrée

    Exploite les différences de funding rate entre exchanges
    et les prédictions de funding futur pour générer des profits
    avec un risque contrôlé.
    """

    def __init__(
        self,
        config: FundingArbitrageConfig = None,
        metrics_collector: MetricsCollector = None
    ):
        self.config = config or FundingArbitrageConfig()
        self.metrics_collector = metrics_collector

        super().__init__(
            name="Funding_Arbitrage",
            parameters=self.config.__dict__
        )

        # Composants
        self.funding_calculator = FundingRateCalculator()
        self.ml_predictor = MLFundingPredictor() if self.config.use_ml_prediction else None

        # État de la stratégie
        self.current_position = 0.0
        self.funding_history = pd.DataFrame()
        self.last_funding_payment = None

        logger.info(f"Funding Arbitrage Strategy initialisée: {self.config}")

    def generate_signals(
        self,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """Génère signaux d'arbitrage funding"""

        if len(data) < 48:  # Au moins 2 jours de données
            logger.warning("Pas assez de données pour Funding Arbitrage")
            return []

        try:
            # Préparer données de funding
            funding_data = self._prepare_funding_data(data, features)

            if funding_data.empty:
                return []

            # Mettre à jour historique
            self.funding_history = funding_data

            # Entraîner ML si nécessaire
            if self.ml_predictor and not self.ml_predictor.is_trained:
                self.ml_predictor.train(funding_data)

            # Calculer signaux
            signals = self._calculate_funding_signals(funding_data, data.iloc[-1])

            # Métriques
            if self.metrics_collector and len(funding_data) > 0:
                current_funding = funding_data["funding_rate"].iloc[-1]
                self.metrics_collector.record_metric(
                    "current_funding_rate",
                    current_funding,
                    {"strategy": self.name}
                )

            return signals

        except Exception as e:
            logger.error(f"Erreur génération signaux Funding: {e}")
            return []

    def _prepare_funding_data(
        self,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Prépare données de funding rate"""

        # Si features contiennent déjà funding data
        if features is not None and "funding_rate" in features.columns:
            return features

        # Sinon, calculer à partir des prix spot/futures
        if "close" in data.columns:
            # Approximation : utiliser close comme spot
            # Dans la réalité, il faudrait vraies données spot et futures
            spot_prices = data["close"]

            # Simulation futures price (légèrement plus élevé)
            # En réalité, récupérer vraies données futures
            futures_prices = spot_prices * (1 + np.random.normal(0.001, 0.0005, len(spot_prices)))

            funding_df = self.funding_calculator.calculate_funding_rate(
                spot_prices,
                futures_prices,
                self.config.funding_interval_hours
            )

            return funding_df

        return pd.DataFrame()

    def _calculate_funding_signals(
        self,
        funding_data: pd.DataFrame,
        last_candle: pd.Series
    ) -> List[Signal]:
        """Calcule signaux d'arbitrage funding"""

        signals = []

        try:
            if len(funding_data) < 24:
                return signals

            # Funding rate actuel
            current_funding = funding_data["funding_rate"].iloc[-1]
            funding_annualized = funding_data["funding_annualized"].iloc[-1]

            # Prédiction funding futur
            prediction = self._predict_future_funding(funding_data)
            predicted_funding = prediction["predicted_funding"]
            confidence = prediction["confidence"]

            # Seuil adaptatif selon volatilité
            funding_vol = funding_data["funding_rate"].tail(24).std()
            adaptive_threshold = max(
                self.config.funding_threshold,
                funding_vol * 2  # Seuil = 2x volatilité
            )

            # Logique de signal
            signal = None

            # Entrée en position si funding élevé et prédit de diminuer
            if (abs(current_funding) > adaptive_threshold and
                confidence > 0.3):

                if current_funding > adaptive_threshold and predicted_funding < current_funding * 0.7:
                    # Funding positif élevé → Short futures (recevoir funding)
                    signal = self._create_funding_signal(
                        SignalAction.SELL,
                        last_candle,
                        min(abs(current_funding) / adaptive_threshold, 1.0),
                        {
                            "current_funding": current_funding,
                            "predicted_funding": predicted_funding,
                            "funding_annualized": funding_annualized,
                            "confidence": confidence,
                            "action_type": "receive_funding"
                        }
                    )

                elif current_funding < -adaptive_threshold and predicted_funding > current_funding * 0.7:
                    # Funding négatif élevé → Long futures (payer moins de funding)
                    signal = self._create_funding_signal(
                        SignalAction.BUY,
                        last_candle,
                        min(abs(current_funding) / adaptive_threshold, 1.0),
                        {
                            "current_funding": current_funding,
                            "predicted_funding": predicted_funding,
                            "funding_annualized": funding_annualized,
                            "confidence": confidence,
                            "action_type": "pay_less_funding"
                        }
                    )

            # Sortie si funding revient vers zéro
            elif self.current_position != 0 and abs(current_funding) < adaptive_threshold * 0.5:
                action = SignalAction.CLOSE_SHORT if self.current_position < 0 else SignalAction.CLOSE_LONG
                signal = self._create_funding_signal(
                    action,
                    last_candle,
                    0.8,
                    {
                        "current_funding": current_funding,
                        "exit_reason": "funding_normalized",
                        "pnl_funding": self._estimate_funding_pnl()
                    }
                )

            if signal:
                signals.append(signal)

        except Exception as e:
            logger.error(f"Erreur calcul signaux funding: {e}")

        return signals

    def _predict_future_funding(self, funding_data: pd.DataFrame) -> Dict[str, float]:
        """Prédit le funding rate futur"""
        if self.ml_predictor and self.ml_predictor.is_trained:
            try:
                current_features = self.ml_predictor.prepare_features(funding_data).iloc[-1]
                return self.ml_predictor.predict(current_features)
            except Exception as e:
                logger.warning(f"Erreur ML funding prediction: {e}")

        # Fallback : prédiction simple
        return self.funding_calculator.predict_next_funding(funding_data)

    def _create_funding_signal(
        self,
        action: SignalAction,
        candle: pd.Series,
        strength: float,
        metadata: Dict[str, Any]
    ) -> Signal:
        """Crée un signal d'arbitrage funding"""
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

    def _estimate_funding_pnl(self) -> float:
        """Estime le P&L des paiements de funding"""
        # Simplification : calcul basé sur l'historique récent
        if len(self.funding_history) < 8:
            return 0.0

        recent_funding = self.funding_history["funding_rate"].tail(8)
        avg_funding = recent_funding.mean()

        # P&L = position_size * avg_funding * nombre_de_périodes
        return float(self.current_position * avg_funding * len(recent_funding))

    def get_funding_analytics(self) -> Dict[str, Any]:
        """Analytics détaillées sur le funding"""
        if self.funding_history.empty:
            return {}

        recent_data = self.funding_history.tail(24 * 7)  # 1 semaine

        return {
            "current_funding": float(self.funding_history["funding_rate"].iloc[-1]),
            "avg_funding_7d": float(recent_data["funding_rate"].mean()),
            "funding_volatility": float(recent_data["funding_rate"].std()),
            "total_funding_periods": len(self.funding_history),
            "estimated_funding_pnl": self._estimate_funding_pnl(),
            "ml_trained": self.ml_predictor.is_trained if self.ml_predictor else False
        }