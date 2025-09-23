"""
Domain Service: SignalService
===========================

Service du domaine pour la logique métier complexe des signaux de trading.
Contient la logique qui ne s'applique pas naturellement à une seule entité.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import statistics

from ..entities.strategy import Strategy
from ..value_objects.signal import Signal, SignalAction, SignalConfidence
from ..value_objects.position import Position, PositionSide


class SignalService:
    """
    Service du domaine pour la gestion des signaux de trading.

    Encapsule la logique métier complexe pour:
    - Validation et filtrage des signaux
    - Agrégation de signaux multiples
    - Calcul de priorités
    - Détection de conflits
    """

    def __init__(self):
        self.min_signal_strength = Decimal("0.2")
        self.max_signal_age_minutes = 60
        self.max_conflicting_signals = 2

    def validate_signal(self, signal: Signal, current_positions: Dict[str, Position]) -> bool:
        """
        Valide un signal selon les règles métier.

        Args:
            signal: Signal à valider
            current_positions: Positions actuelles pour détecter les conflits

        Returns:
            True si le signal est valide, False sinon
        """
        # Validation de base du signal
        if not signal.is_valid or not signal.is_actionable:
            return False

        # Vérification de la force minimale
        if signal.strength < self.min_signal_strength:
            return False

        # Vérification de l'âge du signal
        if self._is_signal_too_old(signal):
            return False

        # Vérification des conflits avec positions existantes
        if self._has_position_conflict(signal, current_positions):
            return False

        return True

    def aggregate_signals(
        self,
        signals: List[Signal],
        aggregation_method: str = "weighted_average"
    ) -> Optional[Signal]:
        """
        Agrège plusieurs signaux pour le même symbole.

        Args:
            signals: Liste de signaux pour le même symbole
            aggregation_method: Méthode d'agrégation ("weighted_average", "consensus", "strongest")

        Returns:
            Signal agrégé ou None si pas possible d'agréger
        """
        if not signals:
            return None

        # Filtrer les signaux valides et récents
        valid_signals = [s for s in signals if s.is_valid and not self._is_signal_too_old(s)]

        if not valid_signals:
            return None

        symbol = valid_signals[0].symbol

        # Vérifier que tous les signaux sont pour le même symbole
        if not all(s.symbol == symbol for s in valid_signals):
            raise ValueError("All signals must be for the same symbol")

        if aggregation_method == "weighted_average":
            return self._aggregate_weighted_average(valid_signals)
        elif aggregation_method == "consensus":
            return self._aggregate_consensus(valid_signals)
        elif aggregation_method == "strongest":
            return self._aggregate_strongest(valid_signals)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    def calculate_signal_priority(
        self,
        signal: Signal,
        strategy: Strategy,
        current_exposure: Decimal
    ) -> Decimal:
        """
        Calcule la priorité d'exécution d'un signal.

        Args:
            signal: Signal à évaluer
            strategy: Stratégie qui a généré le signal
            current_exposure: Exposition actuelle du portfolio

        Returns:
            Score de priorité (0-100, plus élevé = plus prioritaire)
        """
        priority = signal.get_signal_score()

        # Bonus pour les stratégies performantes
        if strategy.sharpe_ratio:
            sharpe_bonus = min(Decimal("20"), strategy.sharpe_ratio * 10)
            priority += sharpe_bonus

        # Bonus pour bon ratio risque/récompense
        rr_ratio = signal.risk_reward_ratio
        if rr_ratio and rr_ratio >= 2:
            priority += Decimal("15")
        elif rr_ratio and rr_ratio >= 1.5:
            priority += Decimal("10")

        # Pénalité pour exposition élevée
        if current_exposure > Decimal("0.7"):
            exposure_penalty = (current_exposure - Decimal("0.7")) * 50
            priority -= exposure_penalty

        # Bonus pour signaux récents
        signal_age_minutes = (datetime.utcnow() - signal.timestamp).total_seconds() / 60
        if signal_age_minutes < 5:
            priority += Decimal("10")
        elif signal_age_minutes < 15:
            priority += Decimal("5")

        return max(Decimal("0"), min(priority, Decimal("100")))

    def detect_signal_conflicts(
        self,
        signals: List[Signal],
        positions: Dict[str, Position]
    ) -> List[Dict[str, Any]]:
        """
        Détecte les conflits entre signaux et positions.

        Args:
            signals: Liste de signaux à analyser
            positions: Positions actuelles

        Returns:
            Liste des conflits détectés avec détails
        """
        conflicts = []

        # Groupe les signaux par symbole
        signals_by_symbol = {}
        for signal in signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)

        # Détection de conflits par symbole
        for symbol, symbol_signals in signals_by_symbol.items():
            # Conflit: signaux opposés pour le même symbole
            buy_signals = [s for s in symbol_signals if s.action == SignalAction.BUY]
            sell_signals = [s for s in symbol_signals if s.action == SignalAction.SELL]

            if buy_signals and sell_signals:
                conflicts.append({
                    "type": "opposing_signals",
                    "symbol": symbol,
                    "buy_signals": len(buy_signals),
                    "sell_signals": len(sell_signals),
                    "severity": "high"
                })

            # Conflit: signal opposé à position existante
            if symbol in positions:
                position = positions[symbol]
                conflicting_signals = self._find_conflicting_signals(symbol_signals, position)

                if conflicting_signals:
                    conflicts.append({
                        "type": "signal_position_conflict",
                        "symbol": symbol,
                        "position_side": position.side.value,
                        "conflicting_signals": len(conflicting_signals),
                        "severity": "medium"
                    })

        return conflicts

    def filter_signals_by_risk(
        self,
        signals: List[Signal],
        max_risk_per_trade: Decimal,
        current_portfolio_risk: Decimal,
        max_portfolio_risk: Decimal
    ) -> List[Signal]:
        """
        Filtre les signaux selon les contraintes de risque.

        Args:
            signals: Signaux à filtrer
            max_risk_per_trade: Risque maximum par trade
            current_portfolio_risk: Risque actuel du portfolio
            max_portfolio_risk: Risque maximum du portfolio

        Returns:
            Liste des signaux respectant les contraintes de risque
        """
        filtered_signals = []

        for signal in signals:
            # Calcul du risque du signal
            signal_risk = self._calculate_signal_risk(signal)

            # Vérification risque par trade
            if signal_risk > max_risk_per_trade:
                continue

            # Vérification risque portfolio
            if current_portfolio_risk + signal_risk > max_portfolio_risk:
                continue

            filtered_signals.append(signal)

        return filtered_signals

    def optimize_signal_timing(
        self,
        signal: Signal,
        market_hours: bool = True,
        volatility_threshold: Optional[Decimal] = None
    ) -> Tuple[bool, Optional[datetime]]:
        """
        Optimise le timing d'exécution d'un signal.

        Args:
            signal: Signal à analyser
            market_hours: Si True, vérifie les heures de marché
            volatility_threshold: Seuil de volatilité pour reporter l'exécution

        Returns:
            Tuple (should_execute_now, optimal_execution_time)
        """
        current_time = datetime.utcnow()

        # Vérification des heures de marché
        if market_hours and not self._is_market_open(current_time):
            next_open = self._get_next_market_open(current_time)
            return False, next_open

        # Vérification de la volatilité
        if volatility_threshold:
            # Cette logique nécessiterait des données de marché
            # Pour maintenant, on suppose que c'est OK
            pass

        # Le signal peut être exécuté maintenant
        return True, current_time

    def _is_signal_too_old(self, signal: Signal) -> bool:
        """Vérifie si un signal est trop ancien."""
        age_minutes = (datetime.utcnow() - signal.timestamp).total_seconds() / 60
        return age_minutes > self.max_signal_age_minutes

    def _has_position_conflict(self, signal: Signal, positions: Dict[str, Position]) -> bool:
        """Vérifie s'il y a conflit avec les positions existantes."""
        if signal.symbol not in positions:
            return False

        position = positions[signal.symbol]

        # Conflit si signal opposé à la position existante
        if position.side == PositionSide.LONG and signal.action == SignalAction.SELL:
            return True
        if position.side == PositionSide.SHORT and signal.action == SignalAction.BUY:
            return True

        return False

    def _find_conflicting_signals(self, signals: List[Signal], position: Position) -> List[Signal]:
        """Trouve les signaux en conflit avec une position."""
        conflicting = []

        for signal in signals:
            if self._has_position_conflict(signal, {position.symbol: position}):
                conflicting.append(signal)

        return conflicting

    def _aggregate_weighted_average(self, signals: List[Signal]) -> Signal:
        """Agrégation par moyenne pondérée."""
        symbol = signals[0].symbol
        total_weight = sum(s.strength for s in signals)

        if total_weight == 0:
            return signals[0]  # Fallback

        # Moyenne pondérée des forces
        weighted_strength = sum(s.strength * s.strength for s in signals) / total_weight

        # Moyenne pondérée des prix
        weighted_price = sum(s.price * s.strength for s in signals) / total_weight

        # Action majoritaire pondérée
        action_weights = {}
        for signal in signals:
            if signal.action not in action_weights:
                action_weights[signal.action] = Decimal("0")
            action_weights[signal.action] += signal.strength

        majority_action = max(action_weights, key=action_weights.get)

        # Confiance moyenne
        confidence_scores = {
            SignalConfidence.LOW: 1,
            SignalConfidence.MEDIUM: 2,
            SignalConfidence.HIGH: 3,
            SignalConfidence.VERY_HIGH: 4
        }
        avg_confidence_score = statistics.mean(confidence_scores[s.confidence] for s in signals)

        if avg_confidence_score >= 3.5:
            avg_confidence = SignalConfidence.VERY_HIGH
        elif avg_confidence_score >= 2.5:
            avg_confidence = SignalConfidence.HIGH
        elif avg_confidence_score >= 1.5:
            avg_confidence = SignalConfidence.MEDIUM
        else:
            avg_confidence = SignalConfidence.LOW

        return Signal(
            symbol=symbol,
            action=majority_action,
            timestamp=datetime.utcnow(),
            strength=weighted_strength,
            confidence=avg_confidence,
            price=weighted_price,
            metadata={"aggregation_method": "weighted_average", "signal_count": len(signals)}
        )

    def _aggregate_consensus(self, signals: List[Signal]) -> Optional[Signal]:
        """Agrégation par consensus (au moins 60% d'accord)."""
        if len(signals) < 2:
            return signals[0] if signals else None

        symbol = signals[0].symbol

        # Compter les votes par action
        action_votes = {}
        for signal in signals:
            if signal.action not in action_votes:
                action_votes[signal.action] = []
            action_votes[signal.action].append(signal)

        # Vérifier s'il y a consensus (60% minimum)
        majority_action = max(action_votes, key=lambda k: len(action_votes[k]))
        majority_count = len(action_votes[majority_action])
        consensus_ratio = majority_count / len(signals)

        if consensus_ratio < 0.6:
            return None  # Pas de consensus

        # Créer signal de consensus
        consensus_signals = action_votes[majority_action]
        avg_strength = statistics.mean(s.strength for s in consensus_signals)
        avg_price = statistics.mean(s.price for s in consensus_signals)

        return Signal(
            symbol=symbol,
            action=majority_action,
            timestamp=datetime.utcnow(),
            strength=Decimal(str(avg_strength)),
            confidence=SignalConfidence.HIGH,  # Consensus = haute confiance
            price=Decimal(str(avg_price)),
            metadata={"aggregation_method": "consensus", "consensus_ratio": consensus_ratio}
        )

    def _aggregate_strongest(self, signals: List[Signal]) -> Signal:
        """Agrégation en prenant le signal le plus fort."""
        return max(signals, key=lambda s: s.get_signal_score())

    def _calculate_signal_risk(self, signal: Signal) -> Decimal:
        """Calcule le risque d'un signal."""
        if not (signal.stop_loss and signal.price):
            return Decimal("0.02")  # Risque par défaut

        # Risque = distance au stop loss / prix d'entrée
        if signal.action == SignalAction.BUY:
            risk = abs(signal.price - signal.stop_loss) / signal.price
        else:  # SELL
            risk = abs(signal.stop_loss - signal.price) / signal.price

        return risk

    def _is_market_open(self, time: datetime) -> bool:
        """Vérifie si le marché est ouvert (simplifié - crypto 24/7)."""
        # Pour les crypto: toujours ouvert
        # Pour les marchés traditionnels: logique plus complexe nécessaire
        return True

    def _get_next_market_open(self, time: datetime) -> datetime:
        """Retourne la prochaine ouverture du marché."""
        # Pour les crypto: immédiatement
        # Pour les marchés traditionnels: calcul plus complexe
        return time