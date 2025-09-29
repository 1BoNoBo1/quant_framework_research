"""
Deep Market Network LSTM Strategy
=================================

Migration propre de votre alpha DMN existant vers l'architecture moderne.
Préserve toute la logique métier tout en utilisant les nouvelles interfaces.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from qframe.core.interfaces import (
    BaseStrategy,
    TimeFrame,
    FeatureProcessor,
    MetricsCollector
)
from qframe.core.container import injectable
from qframe.domain.value_objects.signal import (
    Signal,
    SignalAction,
    SignalConfidence
)

logger = logging.getLogger(__name__)


@dataclass
class DMNConfig:
    """Configuration pour DMN LSTM Strategy"""
    window_size: int = 64
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    use_attention: bool = False
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    signal_threshold: float = 0.1
    position_size: float = 0.02
    validation_split: float = 0.2


class MarketDataset(Dataset):
    """Dataset optimisé pour données de marché"""

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 64,
        horizon: int = 1,
        feature_cols: List[str] = None
    ):
        self.window_size = window_size
        self.horizon = horizon

        # Colonnes de features par défaut
        if feature_cols is None:
            feature_cols = ["returns", "rsi_14", "atr_14", "momentum_10"]

        # Validation des colonnes
        available_cols = [col for col in feature_cols if col in df.columns]
        if not available_cols:
            raise ValueError(f"Aucune feature trouvée dans {df.columns.tolist()}")

        logger.info(f"Features utilisées: {available_cols}")

        # Préparation données
        self.features = df[available_cols].fillna(0).values.astype(np.float32)

        # Target : returns futurs
        if "returns" in df.columns:
            self.targets = df["returns"].shift(-horizon).fillna(0).values.astype(np.float32)
        elif "close" in df.columns:
            returns = df["close"].pct_change().fillna(0)
            self.targets = returns.shift(-horizon).fillna(0).values.astype(np.float32)
        else:
            raise ValueError("Impossible de calculer les returns cibles")

        # Normalisation
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # Création des séquences
        self._create_sequences()

    def _create_sequences(self):
        """Crée les séquences temporelles"""
        self.sequences_X = []
        self.sequences_y = []

        for i in range(len(self.features) - self.window_size - self.horizon):
            self.sequences_X.append(self.features[i:i + self.window_size])
            self.sequences_y.append(self.targets[i + self.window_size])

        self.X = torch.tensor(np.array(self.sequences_X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.sequences_y), dtype=torch.float32).unsqueeze(1)

        logger.info(f"Dataset créé: {len(self.X)} séquences")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class DMNModel(nn.Module):
    """Deep Market Network avec LSTM et attention optionnelle"""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = False
    ):
        super(DMNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM principal
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention optionnelle
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_size, 4, batch_first=True)

        # Tête de prédiction
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # Output entre -1 et 1
        )

        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier des poids (déterministe si seed fixé)"""
        # Sauvegarder l'état du générateur aléatoire
        rng_state = torch.get_rng_state()

        # Fixer le seed pour une initialisation déterministe
        torch.manual_seed(42)

        for name, param in self.named_parameters():
            if "weight" in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        # Restaurer l'état du générateur aléatoire
        torch.set_rng_state(rng_state)

    def forward(self, x):
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)

        # Attention si activée
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            final_hidden = attn_out[:, -1, :]
        else:
            final_hidden = lstm_out[:, -1, :]

        # Prédiction
        output = self.head(final_hidden)
        return output


@injectable
class DMNLSTMStrategy(BaseStrategy):
    """
    Stratégie DMN LSTM migrée vers l'architecture moderne

    Préserve votre logique métier existante avec les nouvelles interfaces.
    """

    def __init__(
        self,
        config: DMNConfig = None,
        feature_processor: FeatureProcessor = None,
        metrics_collector: MetricsCollector = None
    ):
        self.config = config or DMNConfig()
        self.feature_processor = feature_processor
        self.metrics_collector = metrics_collector

        super().__init__(
            name="DMN_LSTM",
            parameters=self.config.__dict__
        )

        # État du modèle
        self.model: Optional[DMNModel] = None
        self.is_trained = False
        self.feature_scaler: Optional[StandardScaler] = None
        self.scaler: Optional[StandardScaler] = None  # Alias for test compatibility
        self.last_prediction = None
        self.training_feature_columns: Optional[List[str]] = None  # Colonnes utilisées lors de l'entraînement

        # Device configuration
        self.device = None  # Will be set by _setup_device
        self._setup_device()

        logger.info(f"DMN LSTM Strategy initialisée avec config: {self.config}")

    def generate_signals(
        self,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """
        Génère des signaux de trading basés sur les prédictions DMN
        """
        # Vérification des colonnes requises d'abord
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Vérification de données insuffisantes
        if len(data) < self.config.window_size:
            raise ValueError(f"Insufficient data: {len(data)} < {self.config.window_size}")

        try:
            # Utiliser features préparées ou préparer à partir des données
            if features is not None:
                model_data = features
            else:
                # Préparer les features à partir des données OHLCV
                preprocessed_data = self._preprocess_data(data)
                model_data = self._engineer_features(preprocessed_data)

            # Entraîner le modèle si nécessaire
            if not self.is_trained:
                self._train_model(model_data)

            # Générer prédiction
            prediction = self._predict(model_data)
            self.last_prediction = prediction

            # Convertir en signal
            signals = self._prediction_to_signals(prediction, data.iloc[-1])

            # Métriques
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    "dmn_prediction",
                    float(prediction),
                    {"strategy": self.name}
                )

            return signals

        except Exception as e:
            logger.error(f"Erreur génération signaux DMN: {e}")
            raise

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features nécessaires si pas déjà fait"""
        if self.feature_processor is not None:
            return self.feature_processor.process(data)

        # Fallback : features basiques
        features = pd.DataFrame(index=data.index)

        if "close" in data.columns:
            features["returns"] = data["close"].pct_change().fillna(0)

            # RSI simple
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features["rsi_14"] = 100 - (100 / (1 + rs))

            # ATR approximatif
            features["atr_14"] = (data["high"] - data["low"]).rolling(14).mean()

            # Momentum
            features["momentum_10"] = data["close"] / data["close"].shift(10) - 1

        features = features.fillna(0)
        return features

    def _train_model(self, data: pd.DataFrame) -> None:
        """Entraîne le modèle DMN"""
        logger.info("Démarrage entraînement modèle DMN...")

        try:
            # Création dataset
            dataset = MarketDataset(
                data,
                window_size=self.config.window_size,
                feature_cols=["returns", "rsi_14", "atr_14", "momentum_10"]
            )

            if len(dataset) < 100:
                logger.warning("Dataset trop petit pour entraînement robuste")
                return

            # Split train/validation
            train_size = int(len(dataset) * (1 - self.config.validation_split))
            train_dataset = torch.utils.data.Subset(dataset, range(train_size))
            val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

            # DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False  # Important pour séries temporelles
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

            # Modèle
            input_size = dataset.features.shape[1]
            self.model = DMNModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                use_attention=self.config.use_attention
            )

            # Optimisation
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

            # Entraînement
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.epochs):
                # Train
                self.model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 15:
                    logger.info(f"Early stopping à l'epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.6f}, val_loss={val_loss:.6f}")

            self.is_trained = True
            self.feature_scaler = dataset.scaler
            logger.info(f"✅ Modèle DMN entraîné (best_val_loss: {best_val_loss:.6f})")

        except Exception as e:
            logger.error(f"Erreur entraînement DMN: {e}")
            raise

    def _predict(self, data: pd.DataFrame) -> float:
        """Génère une prédiction avec le modèle entraîné"""
        if self.model is None or not self.is_trained:
            return 0.0

        try:
            # Utiliser les colonnes d'entraînement si disponibles, sinon fallback
            if self.training_feature_columns:
                available_cols = [col for col in self.training_feature_columns if col in data.columns]
            else:
                # Fallback pour les anciens modèles ou tests
                feature_cols = ["returns", "rsi_14", "atr_14", "momentum_10"]
                available_cols = [col for col in feature_cols if col in data.columns]

            if not available_cols:
                logger.warning(f"Aucune feature disponible dans les colonnes: {data.columns.tolist()}")
                logger.warning(f"Colonnes d'entraînement attendues: {self.training_feature_columns}")
                return 0.0

            # Dernières observations
            last_sequence = data[available_cols].tail(self.config.window_size).fillna(0).values

            if len(last_sequence) < self.config.window_size:
                # Pad avec zéros si nécessaire
                padding = np.zeros((self.config.window_size - len(last_sequence), len(available_cols)))
                last_sequence = np.vstack([padding, last_sequence])

            # Normalisation
            if self.feature_scaler is not None:
                last_sequence = self.feature_scaler.transform(last_sequence)

            # Prédiction (support des modèles mockés avec méthode predict)
            if hasattr(self.model, 'predict') and callable(self.model.predict):
                # Modèle mocké avec méthode predict
                prediction_array = self.model.predict(last_sequence)
                if isinstance(prediction_array, np.ndarray) and len(prediction_array) > 0:
                    return float(prediction_array[0])
                else:
                    return float(prediction_array) if prediction_array is not None else 0.0
            else:
                # Modèle PyTorch standard
                self.model.eval()
                with torch.no_grad():
                    sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
                    prediction = self.model(sequence_tensor)
                    return float(prediction.item())

        except Exception as e:
            logger.error(f"Erreur prédiction DMN: {e}")
            return 0.0

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Préprocesse les données de marché pour l'entraînement DMN

        Args:
            data: DataFrame avec colonnes OHLCV

        Returns:
            DataFrame préprocessé avec features normalisées
        """
        try:
            # Copier les données pour ne pas modifier l'original
            processed = data.copy()

            # Vérifier les colonnes requises
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in processed.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes: {missing_cols}")

            # Supprimer les valeurs nulles
            processed = processed.dropna()

            # Calculer les features de base
            if 'returns' not in processed.columns:
                processed['returns'] = processed['close'].pct_change()

            # Calculer RSI
            if 'rsi_14' not in processed.columns:
                delta = processed['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                processed['rsi_14'] = 100 - (100 / (1 + rs))

            # Calculer ATR (Average True Range)
            if 'atr_14' not in processed.columns:
                high_low = processed['high'] - processed['low']
                high_close = (processed['high'] - processed['close'].shift()).abs()
                low_close = (processed['low'] - processed['close'].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                processed['atr_14'] = true_range.rolling(window=14).mean()

            # Calculer momentum
            if 'momentum_10' not in processed.columns:
                processed['momentum_10'] = processed['close'] / processed['close'].shift(10) - 1

            # Supprimer les NaN créés par les calculs
            processed = processed.dropna()

            # Normaliser les features numériques (mais pas les prix de base)
            feature_cols = ['returns', 'rsi_14', 'atr_14', 'momentum_10']
            for col in feature_cols:
                if col in processed.columns:
                    # Normalisation robuste
                    q25 = processed[col].quantile(0.25)
                    q75 = processed[col].quantile(0.75)
                    iqr = q75 - q25
                    if iqr > 0:
                        processed[col] = (processed[col] - processed[col].median()) / iqr

            logger.info(f"Données préprocessées: {len(processed)} lignes, colonnes: {list(processed.columns)}")

            return processed

        except Exception as e:
            logger.error(f"Erreur préprocessing DMN: {e}")
            # Retourner au minimum les données originales nettoyées
            return data.dropna()

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des features techniques avancées pour le modèle DMN

        Args:
            data: DataFrame avec données de marché

        Returns:
            DataFrame avec features techniques ajoutées
        """
        try:
            # Copier les données
            features = data.copy()

            # Returns
            if 'returns' not in features.columns:
                features['returns'] = features['close'].pct_change()

            # Volatility (rolling std des returns)
            if 'volatility' not in features.columns:
                features['volatility'] = features['returns'].rolling(window=20).std()

            # RSI
            if 'rsi' not in features.columns:
                delta = features['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            if 'macd' not in features.columns:
                ema_12 = features['close'].ewm(span=12).mean()
                ema_26 = features['close'].ewm(span=26).mean()
                features['macd'] = ema_12 - ema_26
                features['macd_signal'] = features['macd'].ewm(span=9).mean()
                features['macd_histogram'] = features['macd'] - features['macd_signal']

            # Bollinger Bands
            if 'bb_upper' not in features.columns:
                sma_20 = features['close'].rolling(window=20).mean()
                std_20 = features['close'].rolling(window=20).std()
                features['bb_upper'] = sma_20 + (2 * std_20)
                features['bb_lower'] = sma_20 - (2 * std_20)
                features['bb_width'] = features['bb_upper'] - features['bb_lower']

            # Momentum indicators
            if 'momentum_5' not in features.columns:
                features['momentum_5'] = features['close'] / features['close'].shift(5) - 1
                features['momentum_10'] = features['close'] / features['close'].shift(10) - 1
                features['momentum_20'] = features['close'] / features['close'].shift(20) - 1

            # Volume features
            if 'volume_sma' not in features.columns:
                features['volume_sma'] = features['volume'].rolling(window=20).mean()
                features['volume_ratio'] = features['volume'] / features['volume_sma']

            # Price position dans la range high-low
            if 'price_position' not in features.columns:
                features['price_position'] = (features['close'] - features['low']) / (features['high'] - features['low'])

            # True Range et ATR
            if 'atr' not in features.columns:
                high_low = features['high'] - features['low']
                high_close = (features['high'] - features['close'].shift()).abs()
                low_close = (features['low'] - features['close'].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                features['atr'] = true_range.rolling(window=14).mean()

            # Stochastic Oscillator
            if 'stoch_k' not in features.columns:
                low_14 = features['low'].rolling(window=14).min()
                high_14 = features['high'].rolling(window=14).max()
                features['stoch_k'] = 100 * ((features['close'] - low_14) / (high_14 - low_14))
                features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()

            logger.info(f"Features engineering: {len(features.columns)} colonnes générées")

            # Ne pas supprimer les NaN ici pour garder la même longueur que l'input
            return features

        except Exception as e:
            logger.error(f"Erreur feature engineering DMN: {e}")
            # Retourner au minimum les données avec les features de base
            return data

    def _setup_device(self):
        """
        Configure le device PyTorch (CPU/GPU) pour l'entraînement
        """
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
                torch.cuda.empty_cache()  # Nettoyer la mémoire GPU
                logger.info("GPU CUDA disponible et configuré")
            else:
                self.device = "cpu"
                logger.info("GPU non disponible, utilisation du CPU")

            # Vérifier que le device fonctionne
            test_tensor = torch.tensor([1.0], device=self.device)
            logger.info(f"Device configuré: {self.device}")

        except Exception as e:
            logger.warning(f"Erreur configuration device: {e}, fallback vers CPU")
            self.device = "cpu"

    def _prediction_to_signals(self, prediction: float, last_candle: pd.Series) -> List[Signal]:
        """Convertit une prédiction en signaux de trading"""
        signals = []

        try:
            # Seuillage pour générer signaux
            if abs(prediction) > self.config.signal_threshold:
                action = SignalAction.BUY if prediction > 0 else SignalAction.SELL

                # Déterminer le niveau de confiance
                abs_pred = abs(prediction)
                if abs_pred > 0.8:
                    confidence = SignalConfidence.VERY_HIGH
                elif abs_pred > 0.6:
                    confidence = SignalConfidence.HIGH
                elif abs_pred > 0.4:
                    confidence = SignalConfidence.MEDIUM
                else:
                    confidence = SignalConfidence.LOW

                from decimal import Decimal
                signal = Signal(
                    symbol=last_candle.get("symbol", "BTCUSDT"),
                    action=action,
                    timestamp=datetime.now(),
                    strength=Decimal(str(min(abs(prediction), 1.0))),
                    confidence=confidence,
                    price=Decimal(str(last_candle.get("close", 0))),
                    strategy_id=self.__class__.__name__,
                    metadata={
                        "prediction": float(prediction),
                        "raw_confidence": float(abs_pred),
                        "features_used": ["returns", "rsi_14", "atr_14", "momentum_10"]
                    }
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"Erreur conversion signal DMN: {e}")

        return signals

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Préprocesse les données de marché pour l'entraînement

        Args:
            data: DataFrame avec colonnes OHLCV

        Returns:
            DataFrame préprocessé et nettoyé
        """
        # Vérifier les colonnes requises
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Vérifier la taille minimale seulement si on a assez de données pour une prédiction
        # Le nettoyage peut être fait même sur de petits échantillons
        will_validate_length = len(data) >= self.config.window_size

        # Créer une copie pour éviter de modifier l'original
        processed = data[required_columns].copy()

        # Nettoyer les valeurs invalides
        processed = processed.replace([np.inf, -np.inf], np.nan)
        processed = processed.ffill().bfill()

        # Assurer que le volume est positif
        processed['volume'] = np.maximum(processed['volume'], 0.01)

        # Vérifier la cohérence OHLC
        processed['high'] = np.maximum(processed['high'], processed[['open', 'close']].max(axis=1))
        processed['low'] = np.minimum(processed['low'], processed[['open', 'close']].min(axis=1))

        return processed

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineering des features techniques à partir des données OHLCV

        Args:
            data: DataFrame préprocessé avec OHLCV

        Returns:
            DataFrame avec features techniques ajoutées
        """
        features = data.copy()

        # Returns
        features['returns'] = features['close'].pct_change()

        # Volatilité (rolling std des returns)
        features['volatility'] = features['returns'].rolling(window=20).std()

        # RSI (Relative Strength Index)
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = features['close'].ewm(span=12).mean()
        exp2 = features['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # Moving averages
        features['sma_20'] = features['close'].rolling(window=20).mean()
        features['sma_50'] = features['close'].rolling(window=50).mean()

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = features['close'].rolling(window=bb_period).mean()
        std = features['close'].rolling(window=bb_period).std()
        features['bb_upper'] = sma + (std * bb_std)
        features['bb_lower'] = sma - (std * bb_std)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma

        # Average True Range (ATR)
        high_low = features['high'] - features['low']
        high_close = np.abs(features['high'] - features['close'].shift())
        low_close = np.abs(features['low'] - features['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        features['atr'] = true_range.rolling(window=14).mean()

        # Volume indicators
        features['volume_sma'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma']

        # Price position within the day's range
        features['price_position'] = (features['close'] - features['low']) / (features['high'] - features['low'])

        # Momentum indicators
        features['momentum_5'] = features['close'] / features['close'].shift(5) - 1
        features['momentum_10'] = features['close'] / features['close'].shift(10) - 1
        features['momentum_20'] = features['close'] / features['close'].shift(20) - 1

        # Remplir les NaN avec 0
        features = features.fillna(0)

        return features

    def _setup_device(self) -> None:
        """
        Configure le device PyTorch (GPU ou CPU)
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            logger.info("Using CPU for training")

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Entraîne le modèle DMN LSTM sur les données fournies

        Args:
            data: DataFrame avec données de marché

        Returns:
            Dictionnaire avec l'historique d'entraînement
        """
        logger.info("Starting DMN LSTM training...")

        # Préprocessing et feature engineering
        processed_data = self._preprocess_data(data)
        features_data = self._engineer_features(processed_data)

        # Sélectionner les features pour l'entraînement
        feature_cols = ['returns', 'volatility', 'rsi', 'macd', 'atr', 'momentum_10']
        available_cols = [col for col in feature_cols if col in features_data.columns]

        if len(available_cols) < 3:
            raise ValueError(f"Not enough features available: {available_cols}")

        # Créer le dataset
        dataset = MarketDataset(
            features_data,
            window_size=self.config.window_size,
            feature_cols=available_cols
        )

        if len(dataset) < 50:
            logger.warning(f"Small dataset size: {len(dataset)}. Training may be unstable.")

        # Split train/validation
        train_size = int(len(dataset) * (1 - self.config.validation_split))
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False  # Important pour les séries temporelles
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Initialiser le modèle
        input_size = len(available_cols)
        self.model = DMNModel(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            use_attention=self.config.use_attention
        ).to(self.device)

        # Optimiseur et loss
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        # Historique d'entraînement
        history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'lr_schedule': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Entraînement
            train_metrics = self._train_epoch(train_loader, optimizer, criterion)

            # Validation
            val_metrics = self._validate_epoch(val_loader, criterion)

            # Mise à jour du scheduler
            scheduler.step(val_metrics['loss'])

            # Sauvegarde des métriques
            history['train_losses'].append(train_metrics['loss'])
            history['val_losses'].append(val_metrics['loss'])
            history['epochs'].append(epoch)
            history['lr_schedule'].append(optimizer.param_groups[0]['lr'])

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 15:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.6f}, "
                    f"val_loss={val_metrics['loss']:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        self.is_trained = True
        self.feature_scaler = dataset.scaler
        self.scaler = dataset.scaler  # Alias pour compatibilité tests
        self.training_feature_columns = available_cols  # Sauvegarder les colonnes utilisées

        logger.info(f"✅ Training completed. Best val loss: {best_val_loss:.6f}")

        return history

    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> Dict[str, float]:
        """
        Entraîne le modèle pour une époque

        Args:
            train_loader: DataLoader pour les données d'entraînement
            optimizer: Optimiseur PyTorch
            criterion: Fonction de loss

        Returns:
            Dictionnaire avec les métriques d'entraînement
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping pour éviter l'explosion des gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }

    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """
        Valide le modèle pour une époque

        Args:
            val_loader: DataLoader pour les données de validation
            criterion: Fonction de loss

        Returns:
            Dictionnaire avec les métriques de validation
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }

    def _get_time_series_split(self, n_splits: int = 5) -> TimeSeriesSplit:
        """
        Retourne un objet TimeSeriesSplit pour la validation croisée

        Args:
            n_splits: Nombre de splits pour la validation croisée

        Returns:
            Objet TimeSeriesSplit configuré
        """
        return TimeSeriesSplit(n_splits=n_splits)

    def _predictions_to_signals(self, predictions: np.ndarray, timestamps: pd.Index,
                               prices: pd.Series, symbol: str = "BTCUSDT") -> List[Signal]:
        """
        Convertit les prédictions en signaux de trading

        Args:
            predictions: Array des prédictions du modèle
            timestamps: Index temporel correspondant
            prices: Séries des prix correspondants
            symbol: Symbole de l'actif tradé

        Returns:
            Liste des signaux générés
        """
        from decimal import Decimal

        signals = []
        confidences = self._calculate_prediction_confidence(predictions)

        for i, (pred, timestamp, price, confidence) in enumerate(zip(predictions, timestamps, prices, confidences)):
            # Déterminer l'action en fonction de la prédiction
            if abs(pred) > self.config.signal_threshold:
                action = SignalAction.BUY if pred > 0 else SignalAction.SELL

                # Mappage de la force de prédiction vers le niveau de confiance
                if confidence > 0.8:
                    signal_confidence = SignalConfidence.VERY_HIGH
                elif confidence > 0.6:
                    signal_confidence = SignalConfidence.HIGH
                elif confidence > 0.4:
                    signal_confidence = SignalConfidence.MEDIUM
                else:
                    signal_confidence = SignalConfidence.LOW

                # Créer le signal avec les paramètres corrects
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    timestamp=timestamp if hasattr(timestamp, 'to_pydatetime') else datetime.now(),
                    strength=Decimal(str(min(abs(pred), 1.0))),
                    confidence=signal_confidence,
                    price=Decimal(str(price)) if pd.notna(price) else Decimal("0"),
                    strategy_id=self.__class__.__name__,
                    metadata={
                        "prediction": float(pred),
                        "raw_confidence": float(confidence),
                        "signal_index": i
                    }
                )
                signals.append(signal)

        return signals

    def save_model(self, path: str) -> None:
        """
        Sauvegarde le modèle entraîné

        Args:
            path: Chemin de sauvegarde
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modèle et les métadonnées
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'feature_scaler': self.feature_scaler,
            'is_trained': self.is_trained,
            'model_class': self.model.__class__.__name__,
            'model_params': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'use_attention': self.model.use_attention
            }
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Charge un modèle sauvegardé

        Args:
            path: Chemin du modèle à charger
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Recréer le modèle avec les bons paramètres
        model_params = checkpoint['model_params']
        self.model = DMNModel(
            input_size=model_params['input_size'],
            hidden_size=model_params['hidden_size'],
            num_layers=model_params['num_layers'],
            use_attention=model_params.get('use_attention', False)
        ).to(self.device)

        # Charger les poids
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restaurer les autres attributs
        self.feature_scaler = checkpoint.get('feature_scaler')
        self.scaler = self.feature_scaler  # Alias
        self.is_trained = checkpoint.get('is_trained', True)

        logger.info(f"Model loaded from {path}")

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcule les métriques d'évaluation du modèle

        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions du modèle

        Returns:
            Dictionnaire avec les métriques calculées
        """
        # Métriques de base
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Accuracy directionnelle
        y_true_direction = np.sign(y_true)
        y_pred_direction = np.sign(y_pred)
        directional_accuracy = np.mean(y_true_direction == y_pred_direction)

        # Sharpe ratio approximatif (en supposant que y_true sont des returns)
        if np.std(y_true) > 0:
            sharpe_ratio = np.mean(y_true) / np.std(y_true) * np.sqrt(252)  # Annualisé
        else:
            sharpe_ratio = 0.0

        # Corrélation
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy),
            'sharpe_ratio': float(sharpe_ratio),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'r2_score': float(r2_score)
        }

    def _calculate_prediction_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calcule la confiance des prédictions

        Args:
            predictions: Array des prédictions

        Returns:
            Array des scores de confiance (0-1)
        """
        # Utiliser la valeur absolue comme proxy de confiance
        # Plus la prédiction est extrême, plus on est confiant
        abs_predictions = np.abs(predictions)

        # Normaliser entre 0 et 1
        if np.max(abs_predictions) > 0:
            confidences = abs_predictions / np.max(abs_predictions)
        else:
            confidences = np.zeros_like(predictions)

        # Appliquer une transformation sigmoid pour lisser
        confidences = 1 / (1 + np.exp(-5 * (confidences - 0.5)))

        return confidences

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur la stratégie

        Returns:
            Dictionnaire avec les informations de la stratégie
        """
        model_info = {
            'parameters': 0,
            'input_size': 0,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'use_attention': self.config.use_attention
        }

        if self.model is not None:
            model_info['parameters'] = sum(p.numel() for p in self.model.parameters())
            model_info['input_size'] = self.model.input_size

        return {
            'name': 'dmn_lstm',
            'type': 'deep_learning',
            'config': self.config.__dict__,
            'model_info': model_info,
            'training_status': {
                'is_trained': self.is_trained,
                'device': self.device,
                'last_prediction': self.last_prediction
            },
            'features': {
                'feature_processor': self.feature_processor is not None,
                'metrics_collector': self.metrics_collector is not None
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Informations sur l'état du modèle (compatibilité avec l'ancien code)"""
        return {
            "is_trained": self.is_trained,
            "last_prediction": self.last_prediction,
            "config": self.config.__dict__,
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions sur de nouvelles données

        Args:
            data: DataFrame avec les données de marché

        Returns:
            Array des prédictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Préprocessing et feature engineering
        processed_data = self._preprocess_data(data)
        features_data = self._engineer_features(processed_data)

        # Sélectionner les features pour la prédiction
        feature_cols = ['returns', 'volatility', 'rsi', 'macd', 'atr', 'momentum_10']
        available_cols = [col for col in feature_cols if col in features_data.columns]

        if len(available_cols) < 3:
            logger.warning(f"Not enough features available. Only found: {available_cols}")
            # Utiliser toutes les colonnes numériques disponibles
            numeric_cols = features_data.select_dtypes(include=[np.number]).columns
            X = features_data[numeric_cols].values
        else:
            X = features_data[available_cols].values

        # Normalisation
        if self.scaler is not None:
            X = self.scaler.transform(X)
        else:
            # Fallback: normalisation simple
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        # Créer des séquences pour la prédiction
        sequences = []
        for i in range(self.config.window_size, len(X)):
            sequences.append(X[i-self.config.window_size:i])

        if not sequences:
            raise ValueError(f"Not enough data to create sequences. Need at least {self.config.window_size} rows")

        X_tensor = torch.FloatTensor(np.array(sequences)).to(self.device)

        # Prédiction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().squeeze()

        self.last_prediction = datetime.now()
        return predictions

    def evaluate(self, data: pd.DataFrame, test_split: float = 0.2) -> Dict[str, Any]:
        """
        Évalue le modèle sur des données

        Args:
            data: DataFrame avec les données de marché
            test_split: Proportion des données à utiliser pour le test

        Returns:
            Dictionnaire avec les métriques d'évaluation et les prédictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Séparer les données train/test
        split_idx = int(len(data) * (1 - test_split))
        test_data = data.iloc[split_idx:]

        # Faire les prédictions
        predictions = self.predict(test_data)

        # Calculer les vraies valeurs (returns futurs)
        y_true = test_data['close'].pct_change().shift(-1).dropna().values

        # Ajuster la taille si nécessaire
        min_len = min(len(predictions), len(y_true))
        predictions = predictions[:min_len]
        y_true = y_true[:min_len]

        # Calculer les métriques
        metrics = self._calculate_metrics(y_true, predictions)

        # Générer des signaux
        timestamps = test_data.index[-len(predictions):]
        prices = test_data['close'].iloc[-len(predictions):]
        signals = self._predictions_to_signals(predictions, timestamps, prices)

        # Calculer des métriques supplémentaires sur les signaux
        buy_signals = [s for s in signals if s.action == SignalAction.BUY]
        sell_signals = [s for s in signals if s.action == SignalAction.SELL]

        return {
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'y_true': y_true.tolist(),
            'signals': {
                'total': len(signals),
                'buy': len(buy_signals),
                'sell': len(sell_signals),
                'buy_ratio': len(buy_signals) / len(signals) if signals else 0,
                'sell_ratio': len(sell_signals) / len(signals) if signals else 0
            },
            'test_period': {
                'start': str(test_data.index[0]),
                'end': str(test_data.index[-1]),
                'num_samples': len(test_data)
            }
        }

    def _make_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Méthode interne pour faire des prédictions avec le modèle

        Args:
            X: Features d'entrée normalisées

        Returns:
            Array des prédictions
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train the model first.")

        # Créer des séquences
        sequences = []
        for i in range(self.config.window_size, len(X)):
            sequences.append(X[i-self.config.window_size:i])

        if not sequences:
            return np.array([])

        X_tensor = torch.FloatTensor(np.array(sequences)).to(self.device)

        # Prédiction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().squeeze()