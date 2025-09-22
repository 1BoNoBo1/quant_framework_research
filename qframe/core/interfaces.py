"""
Core interfaces - Contrats fondamentaux du framework
====================================================

Définit les interfaces principales pour l'architecture hexagonale.
Utilise Protocol pour typing duck typing moderne.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np


# ================================
# ENUMS & VALUE OBJECTS
# ================================

class SignalAction(Enum):
    """Actions possibles pour un signal de trading"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class TimeFrame(Enum):
    """Timeframes supportés"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass(frozen=True)
class MarketData:
    """Données de marché OHLCV + métadonnées"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: TimeFrame
    source: str = "unknown"


@dataclass(frozen=True)
class Signal:
    """Signal de trading avec métadonnées"""
    timestamp: datetime
    symbol: str
    action: SignalAction
    strength: float  # 0.0 to 1.0
    price: Optional[float] = None
    size: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass(frozen=True)
class Position:
    """Position de trading"""
    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    current_price: float
    timestamp: datetime
    unrealized_pnl: float = 0.0
    metadata: Dict[str, Any] = None


# ================================
# CORE PROTOCOLS
# ================================

class DataProvider(Protocol):
    """Interface pour fournisseurs de données"""

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Récupère données OHLCV pour un symbole et timeframe"""
        ...

    async def fetch_latest_price(self, symbol: str) -> float:
        """Prix actuel d'un symbole"""
        ...

    async def get_available_symbols(self) -> List[str]:
        """Liste des symboles disponibles"""
        ...


class FeatureProcessor(Protocol):
    """Interface pour processeurs de features"""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforme données brutes en features"""
        ...

    def get_feature_names(self) -> List[str]:
        """Noms des features générées"""
        ...


class Strategy(Protocol):
    """Interface pour stratégies de trading"""

    def generate_signals(
        self,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """Génère signaux de trading"""
        ...

    def get_name(self) -> str:
        """Nom de la stratégie"""
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """Paramètres de la stratégie"""
        ...


class RiskManager(Protocol):
    """Interface pour gestion des risques"""

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_positions: List[Position]
    ) -> float:
        """Calcule taille de position"""
        ...

    def check_risk_limits(
        self,
        signal: Signal,
        current_positions: List[Position]
    ) -> bool:
        """Vérifie limites de risque"""
        ...


class Portfolio(Protocol):
    """Interface pour gestion de portfolio"""

    def get_positions(self) -> List[Position]:
        """Positions actuelles"""
        ...

    def get_total_value(self) -> float:
        """Valeur totale du portfolio"""
        ...

    def get_pnl(self) -> float:
        """P&L total"""
        ...

    def add_position(self, position: Position) -> None:
        """Ajoute une position"""
        ...

    def close_position(self, symbol: str) -> Optional[Position]:
        """Ferme une position"""
        ...


class OrderExecutor(Protocol):
    """Interface pour exécution d'ordres"""

    async def execute_signal(
        self,
        signal: Signal,
        position_size: float
    ) -> Dict[str, Any]:
        """Exécute un signal de trading"""
        ...

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Statut d'un ordre"""
        ...


class CacheService(Protocol):
    """Interface pour service de cache"""

    async def get(self, key: str) -> Optional[Any]:
        """Récupère valeur du cache"""
        ...

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Stocke valeur dans le cache"""
        ...

    async def delete(self, key: str) -> None:
        """Supprime clé du cache"""
        ...


# ================================
# ABSTRACT BASE CLASSES
# ================================

class BaseStrategy(ABC):
    """Classe de base pour stratégies avec implémentation commune"""

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self._feature_processor: Optional[FeatureProcessor] = None

    def get_name(self) -> str:
        return self.name

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def set_feature_processor(self, processor: FeatureProcessor) -> None:
        """Injection du processeur de features"""
        self._feature_processor = processor

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """Implémentation spécifique à chaque stratégie"""
        pass

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features si processeur disponible"""
        if self._feature_processor is not None:
            return self._feature_processor.process(data)
        return data


class BaseDataProvider(ABC):
    """Classe de base pour data providers"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self._cache: Optional[CacheService] = None

    def set_cache(self, cache: CacheService) -> None:
        """Injection du service de cache"""
        self._cache = cache

    async def _get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """Récupère données du cache si disponible"""
        if self._cache is not None:
            return await self._cache.get(key)
        return None

    async def _cache_data(self, key: str, data: pd.DataFrame, ttl: int = 3600) -> None:
        """Met en cache les données"""
        if self._cache is not None:
            await self._cache.set(key, data, ttl)


# ================================
# REPOSITORY PATTERNS
# ================================

class StrategyRepository(Protocol):
    """Repository pour persistance des stratégies"""

    async def save_strategy(self, strategy: Strategy) -> str:
        """Sauvegarde une stratégie"""
        ...

    async def load_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Charge une stratégie"""
        ...

    async def list_strategies(self) -> List[Dict[str, Any]]:
        """Liste toutes les stratégies"""
        ...


class BacktestRepository(Protocol):
    """Repository pour résultats de backtests"""

    async def save_backtest_result(
        self,
        strategy_name: str,
        result: Dict[str, Any]
    ) -> str:
        """Sauvegarde résultat de backtest"""
        ...

    async def get_backtest_results(
        self,
        strategy_name: str
    ) -> List[Dict[str, Any]]:
        """Récupère historique des backtests"""
        ...


# ================================
# EVENT SYSTEM
# ================================

class EventHandler(Protocol):
    """Interface pour gestionnaires d'événements"""

    async def handle(self, event: Dict[str, Any]) -> None:
        """Traite un événement"""
        ...


class EventBus(Protocol):
    """Interface pour bus d'événements"""

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """S'abonne à un type d'événement"""
        ...

    async def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publie un événement"""
        ...


# ================================
# METRICS & MONITORING
# ================================

class MetricsCollector(Protocol):
    """Interface pour collecte de métriques"""

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Enregistre une métrique"""
        ...

    def increment_counter(self, name: str, tags: Dict[str, str] = None) -> None:
        """Incrémente un compteur"""
        ...

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Enregistre dans un histogramme"""
        ...


class AlertManager(Protocol):
    """Interface pour gestion des alertes"""

    async def send_alert(
        self,
        level: str,
        message: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Envoie une alerte"""
        ...

    def set_threshold(self, metric: str, threshold: float, operator: str = "gt") -> None:
        """Définit un seuil d'alerte"""
        ...