"""
Deep Market Network LSTM Strategy
=================================

Migration propre de votre alpha DMN existant vers l'architecture moderne.
Préserve toute la logique métier tout en utilisant les nouvelles interfaces.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from qframe.core.interfaces import (
    BaseStrategy,
    Signal,
    SignalAction,
    TimeFrame,
    FeatureProcessor,
    MetricsCollector
)
from qframe.core.container import injectable

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
        """Initialisation Xavier des poids"""
        for name, param in self.named_parameters():
            if "weight" in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

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
        metrics_collector: MetricsCollector = None
    ):
        self.config = config or DMNConfig()
        self.metrics_collector = metrics_collector

        super().__init__(
            name="DMN_LSTM",
            parameters=self.config.__dict__
        )

        # État du modèle
        self.model: Optional[DMNModel] = None
        self.is_trained = False
        self.feature_scaler: Optional[StandardScaler] = None
        self.last_prediction = None

        logger.info(f"DMN LSTM Strategy initialisée avec config: {self.config}")

    def generate_signals(
        self,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """
        Génère des signaux de trading basés sur les prédictions DMN
        """
        if len(data) < self.config.window_size:
            logger.warning(f"Pas assez de données: {len(data)} < {self.config.window_size}")
            return []

        try:
            # Utiliser features préparées ou préparer à partir des données
            if features is not None:
                model_data = features
            else:
                model_data = self._prepare_features(data)

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
            return []

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features nécessaires si pas déjà fait"""
        if self._feature_processor is not None:
            return self._feature_processor.process(data)

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
            # Préparer dernière séquence
            feature_cols = ["returns", "rsi_14", "atr_14", "momentum_10"]
            available_cols = [col for col in feature_cols if col in data.columns]

            if not available_cols:
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

            # Prédiction
            self.model.eval()
            with torch.no_grad():
                sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
                prediction = self.model(sequence_tensor)
                return float(prediction.item())

        except Exception as e:
            logger.error(f"Erreur prédiction DMN: {e}")
            return 0.0

    def _prediction_to_signals(self, prediction: float, last_candle: pd.Series) -> List[Signal]:
        """Convertit une prédiction en signaux de trading"""
        signals = []

        try:
            # Seuillage pour générer signaux
            if abs(prediction) > self.config.signal_threshold:
                action = SignalAction.BUY if prediction > 0 else SignalAction.SELL

                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=last_candle.get("symbol", "UNKNOWN"),
                    action=action,
                    strength=min(abs(prediction), 1.0),
                    price=last_candle.get("close"),
                    size=self.config.position_size,
                    metadata={
                        "strategy": self.name,
                        "prediction": prediction,
                        "model_confidence": abs(prediction),
                        "features_used": ["returns", "rsi_14", "atr_14", "momentum_10"]
                    }
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"Erreur conversion signal DMN: {e}")

        return signals

    def get_model_info(self) -> Dict[str, Any]:
        """Informations sur l'état du modèle"""
        return {
            "is_trained": self.is_trained,
            "last_prediction": self.last_prediction,
            "config": self.config.__dict__,
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }