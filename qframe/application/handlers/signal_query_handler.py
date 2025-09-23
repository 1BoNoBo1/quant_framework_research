"""
Application Handler: SignalQueryHandler
=====================================

Handler pour les queries liées aux signaux de trading.
Gère la lecture et l'agrégation des signaux sans modification d'état.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from ...domain.repositories.strategy_repository import StrategyRepository
from ...domain.services.signal_service import SignalService
from ...domain.value_objects.signal import Signal, SignalAction, SignalConfidence
from ..queries.signal_queries import (
    GetSignalsByStrategyQuery,
    GetSignalsBySymbolQuery,
    GetActiveSignalsQuery,
    GetSignalStatisticsQuery,
    SearchSignalsQuery
)

logger = logging.getLogger(__name__)


class SignalQueryHandler:
    """
    Handler pour les queries de signaux.

    Implémente les cas d'usage de lecture et d'analyse
    des signaux sans modifier l'état du système.
    """

    def __init__(
        self,
        strategy_repository: StrategyRepository,
        signal_service: SignalService
    ):
        self.strategy_repository = strategy_repository
        self.signal_service = signal_service

    async def handle_get_signals_by_strategy(
        self,
        query: GetSignalsByStrategyQuery
    ) -> List[Dict[str, Any]]:
        """
        Récupère les signaux d'une stratégie spécifique.

        Args:
            query: Query avec les critères de recherche

        Returns:
            Liste des signaux sérialisés
        """
        logger.info(f"🔍 Recherche signaux pour stratégie: {query.strategy_id}")

        strategy = await self.strategy_repository.find_by_id(query.strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {query.strategy_id} not found")

        signals = strategy.signal_history

        # Filtrage par date si spécifié
        if query.start_date or query.end_date:
            signals = self._filter_signals_by_date(
                signals,
                query.start_date,
                query.end_date
            )

        # Filtrage par action si spécifié
        if query.action:
            action_enum = SignalAction(query.action)
            signals = [s for s in signals if s.action == action_enum]

        # Filtrage par force minimale
        if query.min_strength:
            signals = [s for s in signals if s.strength >= query.min_strength]

        # Limitation du nombre de résultats
        if query.limit:
            signals = signals[-query.limit:]  # Les plus récents

        # Sérialisation
        result = [signal.to_dict() for signal in signals]

        logger.info(f"✅ Trouvé {len(result)} signaux")
        return result

    async def handle_get_signals_by_symbol(
        self,
        query: GetSignalsBySymbolQuery
    ) -> List[Dict[str, Any]]:
        """
        Récupère tous les signaux pour un symbole donné.

        Args:
            query: Query avec le symbole et critères

        Returns:
            Liste des signaux agrégés par stratégie
        """
        logger.info(f"🔍 Recherche signaux pour symbole: {query.symbol}")

        # Récupérer toutes les stratégies qui tradent ce symbole
        strategies = await self.strategy_repository.find_by_universe(query.symbol)

        all_signals = []
        for strategy in strategies:
            # Filtrer les signaux de cette stratégie pour le symbole
            strategy_signals = [
                s for s in strategy.signal_history
                if s.symbol == query.symbol
            ]

            # Appliquer les filtres
            if query.start_date or query.end_date:
                strategy_signals = self._filter_signals_by_date(
                    strategy_signals,
                    query.start_date,
                    query.end_date
                )

            if query.min_strength:
                strategy_signals = [
                    s for s in strategy_signals
                    if s.strength >= query.min_strength
                ]

            all_signals.extend(strategy_signals)

        # Tri par timestamp (plus récents en premier)
        all_signals.sort(key=lambda s: s.timestamp, reverse=True)

        # Limitation
        if query.limit:
            all_signals = all_signals[:query.limit]

        # Sérialisation avec informations de stratégie
        result = []
        for signal in all_signals:
            signal_dict = signal.to_dict()
            signal_dict["strategy_name"] = next(
                (s.name for s in strategies if s.id == signal.strategy_id),
                "Unknown"
            )
            result.append(signal_dict)

        logger.info(f"✅ Trouvé {len(result)} signaux pour {query.symbol}")
        return result

    async def handle_get_active_signals(
        self,
        query: GetActiveSignalsQuery
    ) -> List[Dict[str, Any]]:
        """
        Récupère tous les signaux actifs et valides.

        Args:
            query: Query avec critères de filtrage

        Returns:
            Liste des signaux actifs avec priorités calculées
        """
        logger.info("🔍 Recherche signaux actifs")

        # Récupérer les stratégies actives
        active_strategies = await self.strategy_repository.find_active_strategies()

        active_signals = []
        for strategy in active_strategies:
            # Récupérer les signaux récents et valides
            recent_signals = [
                s for s in strategy.signal_history
                if s.is_valid and s.is_actionable and
                (datetime.utcnow() - s.timestamp).total_seconds() / 60 <= query.max_age_minutes
            ]

            for signal in recent_signals:
                # Calculer la priorité
                priority = self.signal_service.calculate_signal_priority(
                    signal,
                    strategy,
                    strategy.get_current_exposure()
                )

                # Filtrer par priorité minimale si spécifiée
                if query.min_priority and priority < query.min_priority:
                    continue

                signal_dict = signal.to_dict()
                signal_dict.update({
                    "strategy_id": strategy.id,
                    "strategy_name": strategy.name,
                    "priority": float(priority),
                    "current_exposure": float(strategy.get_current_exposure())
                })

                active_signals.append(signal_dict)

        # Tri par priorité (plus élevée en premier)
        active_signals.sort(key=lambda s: s["priority"], reverse=True)

        # Limitation
        if query.limit:
            active_signals = active_signals[:query.limit]

        logger.info(f"✅ Trouvé {len(active_signals)} signaux actifs")
        return active_signals

    async def handle_get_signal_statistics(
        self,
        query: GetSignalStatisticsQuery
    ) -> Dict[str, Any]:
        """
        Calcule les statistiques des signaux.

        Args:
            query: Query avec période et filtres

        Returns:
            Dictionnaire avec les statistiques
        """
        logger.info("📊 Calcul statistiques des signaux")

        strategies = await self.strategy_repository.find_all()

        # Collecter tous les signaux dans la période
        all_signals = []
        for strategy in strategies:
            strategy_signals = strategy.signal_history

            if query.start_date or query.end_date:
                strategy_signals = self._filter_signals_by_date(
                    strategy_signals,
                    query.start_date,
                    query.end_date
                )

            all_signals.extend(strategy_signals)

        if not all_signals:
            return {
                "total_signals": 0,
                "period_start": query.start_date.isoformat() if query.start_date else None,
                "period_end": query.end_date.isoformat() if query.end_date else None
            }

        # Calcul des statistiques
        stats = self._calculate_signal_statistics(all_signals)

        # Ajouter les métadonnées de la query
        stats.update({
            "period_start": query.start_date.isoformat() if query.start_date else None,
            "period_end": query.end_date.isoformat() if query.end_date else None,
            "total_strategies": len(strategies),
            "active_strategies": len([s for s in strategies if s.is_active()])
        })

        logger.info(f"✅ Statistiques calculées: {stats['total_signals']} signaux")
        return stats

    async def handle_search_signals(self, query: SearchSignalsQuery) -> List[Dict[str, Any]]:
        """
        Recherche de signaux avec critères complexes.

        Args:
            query: Query avec critères de recherche

        Returns:
            Liste des signaux correspondants
        """
        logger.info(f"🔍 Recherche complexe de signaux")

        strategies = await self.strategy_repository.find_all()
        matching_signals = []

        for strategy in strategies:
            strategy_signals = strategy.signal_history

            # Appliquer tous les filtres
            filtered_signals = self._apply_search_filters(strategy_signals, query)

            # Ajouter les métadonnées de stratégie
            for signal in filtered_signals:
                signal_dict = signal.to_dict()
                signal_dict.update({
                    "strategy_id": strategy.id,
                    "strategy_name": strategy.name,
                    "strategy_type": strategy.strategy_type.value
                })
                matching_signals.append(signal_dict)

        # Tri selon les critères
        if query.sort_by == "timestamp":
            matching_signals.sort(
                key=lambda s: s["timestamp"],
                reverse=query.sort_desc
            )
        elif query.sort_by == "strength":
            matching_signals.sort(
                key=lambda s: s["strength"],
                reverse=query.sort_desc
            )
        elif query.sort_by == "signal_score":
            matching_signals.sort(
                key=lambda s: s["signal_score"],
                reverse=query.sort_desc
            )

        # Limitation
        if query.limit:
            matching_signals = matching_signals[:query.limit]

        logger.info(f"✅ Recherche terminée: {len(matching_signals)} signaux trouvés")
        return matching_signals

    def _filter_signals_by_date(
        self,
        signals: List[Signal],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Signal]:
        """Filtre les signaux par période."""
        filtered = signals

        if start_date:
            filtered = [s for s in filtered if s.timestamp >= start_date]

        if end_date:
            filtered = [s for s in filtered if s.timestamp <= end_date]

        return filtered

    def _calculate_signal_statistics(self, signals: List[Signal]) -> Dict[str, Any]:
        """Calcule les statistiques détaillées des signaux."""
        if not signals:
            return {"total_signals": 0}

        # Statistiques de base
        total_signals = len(signals)
        buy_signals = len([s for s in signals if s.action == SignalAction.BUY])
        sell_signals = len([s for s in signals if s.action == SignalAction.SELL])
        hold_signals = len([s for s in signals if s.action == SignalAction.HOLD])
        close_signals = len([s for s in signals if s.action == SignalAction.CLOSE])

        # Statistiques de force
        strengths = [float(s.strength) for s in signals]
        avg_strength = sum(strengths) / len(strengths)
        max_strength = max(strengths)
        min_strength = min(strengths)

        # Statistiques par confiance
        confidence_stats = {}
        for conf in SignalConfidence:
            count = len([s for s in signals if s.confidence == conf])
            confidence_stats[conf.value] = {
                "count": count,
                "percentage": (count / total_signals) * 100
            }

        # Statistiques temporelles
        timestamps = [s.timestamp for s in signals]
        time_span = max(timestamps) - min(timestamps)

        # Signaux par jour
        signals_per_day = total_signals / max(1, time_span.days)

        # Top symboles
        symbol_counts = {}
        for signal in signals:
            symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1

        top_symbols = sorted(
            symbol_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "total_signals": total_signals,
            "signals_by_action": {
                "buy": buy_signals,
                "sell": sell_signals,
                "hold": hold_signals,
                "close": close_signals
            },
            "action_percentages": {
                "buy": (buy_signals / total_signals) * 100,
                "sell": (sell_signals / total_signals) * 100,
                "hold": (hold_signals / total_signals) * 100,
                "close": (close_signals / total_signals) * 100
            },
            "strength_statistics": {
                "average": avg_strength,
                "maximum": max_strength,
                "minimum": min_strength
            },
            "confidence_distribution": confidence_stats,
            "temporal_statistics": {
                "time_span_days": time_span.days,
                "signals_per_day": signals_per_day,
                "first_signal": min(timestamps).isoformat(),
                "last_signal": max(timestamps).isoformat()
            },
            "top_symbols": [
                {"symbol": symbol, "count": count, "percentage": (count/total_signals)*100}
                for symbol, count in top_symbols
            ]
        }

    def _apply_search_filters(
        self,
        signals: List[Signal],
        query: SearchSignalsQuery
    ) -> List[Signal]:
        """Applique tous les filtres de recherche."""
        filtered = signals

        # Filtre par symboles
        if query.symbols:
            filtered = [s for s in filtered if s.symbol in query.symbols]

        # Filtre par actions
        if query.actions:
            action_enums = [SignalAction(a) for a in query.actions]
            filtered = [s for s in filtered if s.action in action_enums]

        # Filtre par plage de force
        if query.min_strength is not None:
            filtered = [s for s in filtered if s.strength >= query.min_strength]
        if query.max_strength is not None:
            filtered = [s for s in filtered if s.strength <= query.max_strength]

        # Filtre par confiance
        if query.confidence_levels:
            conf_enums = [SignalConfidence(c) for c in query.confidence_levels]
            filtered = [s for s in filtered if s.confidence in conf_enums]

        # Filtre par période
        if query.start_date or query.end_date:
            filtered = self._filter_signals_by_date(
                filtered,
                query.start_date,
                query.end_date
            )

        # Filtre par validité
        if query.valid_only:
            filtered = [s for s in filtered if s.is_valid]

        if query.actionable_only:
            filtered = [s for s in filtered if s.is_actionable]

        return filtered