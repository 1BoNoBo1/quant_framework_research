"""
Application Queries: Signal Queries
==================================

Queries CQRS pour la lecture des signaux de trading.
Encapsulent les critères de recherche et filtrage.
"""

from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)  # Immutable query
class GetSignalsByStrategyQuery:
    """
    Query pour récupérer les signaux d'une stratégie spécifique.
    """
    strategy_id: str

    # Filtres optionnels
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    action: Optional[str] = None  # "buy", "sell", "hold", "close"
    min_strength: Optional[Decimal] = None
    limit: Optional[int] = None  # Nombre max de résultats

    def __post_init__(self):
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")

        if self.action and self.action not in ["buy", "sell", "hold", "close"]:
            raise ValueError("action must be one of: buy, sell, hold, close")

        if self.min_strength is not None and (self.min_strength < 0 or self.min_strength > 1):
            raise ValueError("min_strength must be between 0 and 1")

        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")

        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")


@dataclass(frozen=True)
class GetSignalsBySymbolQuery:
    """
    Query pour récupérer tous les signaux d'un symbole.
    """
    symbol: str

    # Filtres optionnels
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_strength: Optional[Decimal] = None
    limit: Optional[int] = None

    def __post_init__(self):
        if not self.symbol:
            raise ValueError("symbol cannot be empty")

        if self.min_strength is not None and (self.min_strength < 0 or self.min_strength > 1):
            raise ValueError("min_strength must be between 0 and 1")

        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")

        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")


@dataclass(frozen=True)
class GetActiveSignalsQuery:
    """
    Query pour récupérer les signaux actifs et valides.
    """
    max_age_minutes: int = 60  # Age maximum des signaux
    min_priority: Optional[Decimal] = None  # Priorité minimale
    limit: Optional[int] = None

    def __post_init__(self):
        if self.max_age_minutes <= 0:
            raise ValueError("max_age_minutes must be positive")

        if self.min_priority is not None and (self.min_priority < 0 or self.min_priority > 100):
            raise ValueError("min_priority must be between 0 and 100")

        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")


@dataclass(frozen=True)
class GetSignalStatisticsQuery:
    """
    Query pour récupérer les statistiques des signaux.
    """
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    strategy_ids: Optional[List[str]] = None  # Limiter à certaines stratégies

    def __post_init__(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")

        if self.strategy_ids is not None and len(self.strategy_ids) == 0:
            raise ValueError("strategy_ids cannot be empty if provided")


@dataclass(frozen=True)
class SearchSignalsQuery:
    """
    Query pour recherche complexe de signaux.
    """
    # Critères de recherche
    symbols: Optional[List[str]] = None
    actions: Optional[List[str]] = None  # ["buy", "sell", "hold", "close"]
    confidence_levels: Optional[List[str]] = None  # ["low", "medium", "high", "very_high"]
    strategy_types: Optional[List[str]] = None
    strategy_ids: Optional[List[str]] = None

    # Filtres numériques
    min_strength: Optional[Decimal] = None
    max_strength: Optional[Decimal] = None
    min_signal_score: Optional[Decimal] = None

    # Filtres temporels
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Filtres booléens
    valid_only: bool = True
    actionable_only: bool = False

    # Tri et pagination
    sort_by: str = "timestamp"  # "timestamp", "strength", "signal_score"
    sort_desc: bool = True  # True = DESC, False = ASC
    limit: Optional[int] = None
    offset: int = 0

    def __post_init__(self):
        # Validation des actions
        if self.actions:
            valid_actions = ["buy", "sell", "hold", "close"]
            for action in self.actions:
                if action not in valid_actions:
                    raise ValueError(f"Invalid action: {action}. Must be one of: {valid_actions}")

        # Validation des niveaux de confiance
        if self.confidence_levels:
            valid_confidence = ["low", "medium", "high", "very_high"]
            for conf in self.confidence_levels:
                if conf not in valid_confidence:
                    raise ValueError(f"Invalid confidence: {conf}. Must be one of: {valid_confidence}")

        # Validation des forces
        if self.min_strength is not None and (self.min_strength < 0 or self.min_strength > 1):
            raise ValueError("min_strength must be between 0 and 1")

        if self.max_strength is not None and (self.max_strength < 0 or self.max_strength > 1):
            raise ValueError("max_strength must be between 0 and 1")

        if (self.min_strength is not None and self.max_strength is not None and
            self.min_strength > self.max_strength):
            raise ValueError("min_strength must be <= max_strength")

        # Validation du score
        if self.min_signal_score is not None and (self.min_signal_score < 0 or self.min_signal_score > 100):
            raise ValueError("min_signal_score must be between 0 and 100")

        # Validation des dates
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")

        # Validation du tri
        valid_sort_fields = ["timestamp", "strength", "signal_score"]
        if self.sort_by not in valid_sort_fields:
            raise ValueError(f"sort_by must be one of: {valid_sort_fields}")

        # Validation pagination
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")

        if self.offset < 0:
            raise ValueError("offset cannot be negative")


@dataclass(frozen=True)
class GetSignalConflictsQuery:
    """
    Query pour détecter les conflits entre signaux.
    """
    symbols: Optional[List[str]] = None  # Limiter à certains symboles
    max_age_hours: int = 24  # Age maximum des signaux à considérer
    include_resolved: bool = False  # Inclure les conflits résolus

    def __post_init__(self):
        if self.max_age_hours <= 0:
            raise ValueError("max_age_hours must be positive")


@dataclass(frozen=True)
class GetSignalPerformanceQuery:
    """
    Query pour analyser la performance des signaux.
    """
    strategy_id: Optional[str] = None  # Analyser une stratégie spécifique
    symbol: Optional[str] = None  # Analyser un symbole spécifique
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_holding_hours: int = 1  # Durée minimale de holding pour l'analyse

    def __post_init__(self):
        if self.min_holding_hours <= 0:
            raise ValueError("min_holding_hours must be positive")

        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")


@dataclass(frozen=True)
class GetSignalAggregationQuery:
    """
    Query pour l'agrégation de signaux multiples.
    """
    symbol: str
    max_age_minutes: int = 30  # Age maximum pour agrégation
    min_signals: int = 2  # Nombre minimum de signaux pour agrégation
    aggregation_method: str = "weighted_average"  # "weighted_average", "consensus", "strongest"

    def __post_init__(self):
        if not self.symbol:
            raise ValueError("symbol cannot be empty")

        if self.max_age_minutes <= 0:
            raise ValueError("max_age_minutes must be positive")

        if self.min_signals < 2:
            raise ValueError("min_signals must be at least 2")

        valid_methods = ["weighted_average", "consensus", "strongest"]
        if self.aggregation_method not in valid_methods:
            raise ValueError(f"aggregation_method must be one of: {valid_methods}")


# Factory functions pour création simplifiée des queries

def signals_by_strategy_query(
    strategy_id: str,
    last_n_hours: Optional[int] = None,
    action: Optional[str] = None,
    limit: int = 100
) -> GetSignalsByStrategyQuery:
    """Factory pour query des signaux par stratégie."""
    end_date = datetime.utcnow()
    start_date = None
    if last_n_hours:
        start_date = end_date - datetime.timedelta(hours=last_n_hours)

    return GetSignalsByStrategyQuery(
        strategy_id=strategy_id,
        start_date=start_date,
        end_date=end_date,
        action=action,
        limit=limit
    )


def active_signals_query(
    max_age_minutes: int = 60,
    min_priority: Optional[float] = None,
    limit: int = 50
) -> GetActiveSignalsQuery:
    """Factory pour query des signaux actifs."""
    return GetActiveSignalsQuery(
        max_age_minutes=max_age_minutes,
        min_priority=Decimal(str(min_priority)) if min_priority else None,
        limit=limit
    )


def signal_stats_query(
    last_n_days: Optional[int] = None
) -> GetSignalStatisticsQuery:
    """Factory pour query des statistiques de signaux."""
    end_date = datetime.utcnow()
    start_date = None
    if last_n_days:
        start_date = end_date - datetime.timedelta(days=last_n_days)

    return GetSignalStatisticsQuery(
        start_date=start_date,
        end_date=end_date
    )


def search_signals_query(
    symbols: Optional[List[str]] = None,
    actions: Optional[List[str]] = None,
    last_n_hours: Optional[int] = None,
    min_strength: Optional[float] = None,
    limit: int = 100
) -> SearchSignalsQuery:
    """Factory pour query de recherche de signaux."""
    end_date = datetime.utcnow()
    start_date = None
    if last_n_hours:
        start_date = end_date - datetime.timedelta(hours=last_n_hours)

    return SearchSignalsQuery(
        symbols=symbols,
        actions=actions,
        start_date=start_date,
        end_date=end_date,
        min_strength=Decimal(str(min_strength)) if min_strength else None,
        limit=limit
    )