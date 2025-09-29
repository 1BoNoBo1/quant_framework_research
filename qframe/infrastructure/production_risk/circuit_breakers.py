"""
Circuit Breakers
================

Intelligent circuit breakers for automatic trading halts based on
market conditions, portfolio performance, and risk metrics.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import statistics

from ...core.container import injectable
from .risk_metrics import RiskMetrics


logger = logging.getLogger(__name__)


class BreakCondition(str, Enum):
    """Conditions de déclenchement des circuit breakers"""
    PORTFOLIO_DRAWDOWN = "portfolio_drawdown"
    RAPID_LOSS = "rapid_loss"
    MARKET_VOLATILITY = "market_volatility"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    EXECUTION_DEGRADATION = "execution_degradation"
    EXTERNAL_SIGNAL = "external_signal"
    MANUAL_TRIGGER = "manual_trigger"


class BreakStatus(str, Enum):
    """États du circuit breaker"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    COOLING_DOWN = "cooling_down"
    DISABLED = "disabled"


@dataclass
class CircuitBreakerConfig:
    """Configuration d'un circuit breaker"""
    condition: BreakCondition
    threshold_value: Decimal
    lookback_period: timedelta = timedelta(minutes=15)
    cooldown_period: timedelta = timedelta(minutes=30)
    auto_reset: bool = True
    severity_level: int = 1  # 1-5, 5 étant le plus sévère
    enabled: bool = True


@dataclass
class BreakEvent:
    """Événement de déclenchement"""
    condition: BreakCondition
    trigger_time: datetime
    trigger_value: Decimal
    threshold_value: Decimal
    description: str
    reset_time: Optional[datetime] = None
    manual_reset: bool = False


@injectable
class CircuitBreaker:
    """
    Système de circuit breakers intelligent.

    Implémente des conditions sophistiquées pour arrêts automatiques:
    - Drawdown rapide du portfolio
    - Volatilité de marché extrême
    - Dégradation de l'exécution
    - Breakdown des corrélations
    - Crises de liquidité

    Avec logique de cooldown et auto-reset intelligents.
    """

    def __init__(
        self,
        configs: Optional[List[CircuitBreakerConfig]] = None,
        global_cooldown: timedelta = timedelta(minutes=60)
    ):
        self.configs = configs or self._create_default_configs()
        self.global_cooldown = global_cooldown

        # État des circuit breakers
        self.status = BreakStatus.ACTIVE
        self.triggered_events: List[BreakEvent] = []
        self.last_trigger_time: Optional[datetime] = None

        # Historique des métriques pour analyse tendances
        self.metrics_history: List[RiskMetrics] = []
        self.max_history_length = 1000

        # Conditions de marché détectées
        self.market_regime = "normal"  # "normal", "volatile", "trending", "crisis"

        # Callbacks
        self.on_trigger: Optional[Callable] = None
        self.on_reset: Optional[Callable] = None

    async def should_trigger(self, risk_metrics: RiskMetrics) -> Tuple[bool, str]:
        """
        Détermine si le circuit breaker doit se déclencher.

        Args:
            risk_metrics: Métriques de risque actuelles

        Returns:
            Tuple (should_trigger, reason)
        """
        if self.status != BreakStatus.ACTIVE:
            return False, "Circuit breaker not active"

        # Enregistrer métriques
        self._record_metrics(risk_metrics)

        # Vérifier chaque condition configurée
        for config in self.configs:
            if not config.enabled:
                continue

            should_break, reason = await self._check_condition(config, risk_metrics)
            if should_break:
                return True, f"{config.condition.value}: {reason}"

        # Vérifications composites (combinaisons de conditions)
        composite_trigger, composite_reason = await self._check_composite_conditions(risk_metrics)
        if composite_trigger:
            return True, f"Composite condition: {composite_reason}"

        return False, "No trigger conditions met"

    async def trigger(self, reason: str, condition: BreakCondition = BreakCondition.MANUAL_TRIGGER) -> None:
        """
        Déclenche manuellement le circuit breaker.

        Args:
            reason: Raison du déclenchement
            condition: Condition déclenchée
        """
        if self.status == BreakStatus.TRIGGERED:
            logger.warning("Circuit breaker already triggered")
            return

        # Créer événement
        event = BreakEvent(
            condition=condition,
            trigger_time=datetime.utcnow(),
            trigger_value=Decimal("0"),  # Placeholder pour trigger manuel
            threshold_value=Decimal("0"),
            description=reason
        )

        self.triggered_events.append(event)
        self.status = BreakStatus.TRIGGERED
        self.last_trigger_time = datetime.utcnow()

        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")

        # Callback
        if self.on_trigger:
            await self.on_trigger(event)

        # Programmer reset automatique si configuré
        if any(config.auto_reset for config in self.configs):
            asyncio.create_task(self._schedule_auto_reset())

    async def reset(self, manual: bool = False) -> bool:
        """
        Remet le circuit breaker en service.

        Args:
            manual: Si le reset est manuel (ignore cooldown)

        Returns:
            True si reset réussi
        """
        if self.status == BreakStatus.ACTIVE:
            return True  # Déjà actif

        # Vérifier cooldown pour reset automatique
        if not manual and self.last_trigger_time:
            time_since_trigger = datetime.utcnow() - self.last_trigger_time
            if time_since_trigger < self.global_cooldown:
                logger.warning(f"Circuit breaker still in cooldown for {self.global_cooldown - time_since_trigger}")
                return False

        # Reset
        self.status = BreakStatus.ACTIVE

        # Marquer derniers événements comme resetés
        if self.triggered_events:
            self.triggered_events[-1].reset_time = datetime.utcnow()
            self.triggered_events[-1].manual_reset = manual

        logger.info(f"Circuit breaker reset ({'manual' if manual else 'automatic'})")

        # Callback
        if self.on_reset:
            await self.on_reset(manual)

        return True

    async def get_status(self) -> Dict[str, Any]:
        """Retourne le statut détaillé du circuit breaker"""
        active_configs = [c for c in self.configs if c.enabled]
        recent_events = [e for e in self.triggered_events if
                        (datetime.utcnow() - e.trigger_time).days < 1]

        status_info = {
            "status": self.status.value,
            "last_trigger_time": self.last_trigger_time.isoformat() if self.last_trigger_time else None,
            "active_configs_count": len(active_configs),
            "total_configs": len(self.configs),
            "recent_triggers_24h": len(recent_events),
            "market_regime": self.market_regime,
            "cooldown_remaining": None
        }

        # Calculer temps de cooldown restant
        if self.status == BreakStatus.TRIGGERED and self.last_trigger_time:
            time_since_trigger = datetime.utcnow() - self.last_trigger_time
            if time_since_trigger < self.global_cooldown:
                status_info["cooldown_remaining"] = (self.global_cooldown - time_since_trigger).total_seconds()

        return status_info

    async def update_config(self, condition: BreakCondition, **kwargs) -> bool:
        """Met à jour la configuration d'une condition"""
        for config in self.configs:
            if config.condition == condition:
                for key, value in kwargs.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        logger.info(f"Updated {condition.value} config: {key}={value}")
                return True
        return False

    # === Vérifications des conditions ===

    async def _check_condition(
        self,
        config: CircuitBreakerConfig,
        risk_metrics: RiskMetrics
    ) -> Tuple[bool, str]:
        """Vérifie une condition spécifique"""

        if config.condition == BreakCondition.PORTFOLIO_DRAWDOWN:
            return await self._check_portfolio_drawdown(config, risk_metrics)

        elif config.condition == BreakCondition.RAPID_LOSS:
            return await self._check_rapid_loss(config, risk_metrics)

        elif config.condition == BreakCondition.MARKET_VOLATILITY:
            return await self._check_market_volatility(config, risk_metrics)

        elif config.condition == BreakCondition.LIQUIDITY_CRISIS:
            return await self._check_liquidity_crisis(config, risk_metrics)

        elif config.condition == BreakCondition.CORRELATION_BREAKDOWN:
            return await self._check_correlation_breakdown(config, risk_metrics)

        elif config.condition == BreakCondition.EXECUTION_DEGRADATION:
            return await self._check_execution_degradation(config, risk_metrics)

        return False, "Unknown condition"

    async def _check_portfolio_drawdown(
        self,
        config: CircuitBreakerConfig,
        risk_metrics: RiskMetrics
    ) -> Tuple[bool, str]:
        """Vérifie le drawdown du portfolio"""
        current_drawdown = risk_metrics.max_drawdown

        if current_drawdown > config.threshold_value:
            return True, f"Portfolio drawdown {current_drawdown:.2%} exceeds threshold {config.threshold_value:.2%}"

        return False, ""

    async def _check_rapid_loss(
        self,
        config: CircuitBreakerConfig,
        risk_metrics: RiskMetrics
    ) -> Tuple[bool, str]:
        """Vérifie les pertes rapides sur la période de lookback"""
        if len(self.metrics_history) < 2:
            return False, "Insufficient history"

        # Calculer perte sur période de lookback
        lookback_time = datetime.utcnow() - config.lookback_period
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= lookback_time
        ]

        if len(recent_metrics) < 2:
            return False, "Insufficient recent data"

        # Calculer perte relative
        initial_value = recent_metrics[0].portfolio_value
        current_value = recent_metrics[-1].portfolio_value

        if initial_value > 0:
            loss_rate = (initial_value - current_value) / initial_value

            if loss_rate > config.threshold_value:
                return True, f"Rapid loss {loss_rate:.2%} over {config.lookback_period} exceeds threshold {config.threshold_value:.2%}"

        return False, ""

    async def _check_market_volatility(
        self,
        config: CircuitBreakerConfig,
        risk_metrics: RiskMetrics
    ) -> Tuple[bool, str]:
        """Vérifie la volatilité de marché excessive"""
        current_volatility = risk_metrics.portfolio_volatility

        if current_volatility > config.threshold_value:
            # Vérifier si c'est un spike récent
            recent_volatilities = [
                m.portfolio_volatility for m in self.metrics_history[-10:]
                if m.portfolio_volatility is not None
            ]

            if recent_volatilities:
                avg_recent_vol = statistics.mean(recent_volatilities)
                vol_spike = current_volatility / avg_recent_vol if avg_recent_vol > 0 else 1

                if vol_spike > 3:  # Spike de 3x la volatilité récente
                    self.market_regime = "volatile"
                    return True, f"Extreme volatility spike: {current_volatility:.2%} (3x recent average)"

        return False, ""

    async def _check_liquidity_crisis(
        self,
        config: CircuitBreakerConfig,
        risk_metrics: RiskMetrics
    ) -> Tuple[bool, str]:
        """Vérifie les conditions de crise de liquidité"""
        # Approximation basée sur bid-ask spreads et volume
        if hasattr(risk_metrics, 'liquidity_score') and risk_metrics.liquidity_score is not None:
            if risk_metrics.liquidity_score < config.threshold_value:
                return True, f"Liquidity crisis detected: score {risk_metrics.liquidity_score} below threshold {config.threshold_value}"

        return False, ""

    async def _check_correlation_breakdown(
        self,
        config: CircuitBreakerConfig,
        risk_metrics: RiskMetrics
    ) -> Tuple[bool, str]:
        """Vérifie la breakdown des corrélations attendues"""
        if len(self.metrics_history) < 20:  # Besoin d'historique pour corrélations
            return False, "Insufficient history for correlation analysis"

        # Analyser les corrélations récentes vs historiques
        # (Implémentation simplifiée - dans la réalité, analyserait les corrélations entre assets)

        recent_returns = [
            m.portfolio_return for m in self.metrics_history[-10:]
            if m.portfolio_return is not None
        ]

        if len(recent_returns) >= 5:
            recent_volatility = statistics.stdev(recent_returns) if len(recent_returns) > 1 else 0

            # Si volatilité récente >> volatilité historique = possible breakdown
            historical_returns = [
                m.portfolio_return for m in self.metrics_history[-50:-10]
                if m.portfolio_return is not None
            ]

            if len(historical_returns) > 1:
                historical_volatility = statistics.stdev(historical_returns)

                if historical_volatility > 0:
                    vol_ratio = recent_volatility / historical_volatility

                    if vol_ratio > 5:  # 5x l'augmentation de volatilité
                        return True, f"Correlation breakdown detected: volatility increased {vol_ratio:.1f}x"

        return False, ""

    async def _check_execution_degradation(
        self,
        config: CircuitBreakerConfig,
        risk_metrics: RiskMetrics
    ) -> Tuple[bool, str]:
        """Vérifie la dégradation de l'exécution des ordres"""
        # Cette métrique viendrait normalement du OrderManager
        if hasattr(risk_metrics, 'execution_quality_score') and risk_metrics.execution_quality_score is not None:
            if risk_metrics.execution_quality_score < config.threshold_value:
                return True, f"Execution degradation: quality score {risk_metrics.execution_quality_score} below threshold {config.threshold_value}"

        return False, ""

    async def _check_composite_conditions(self, risk_metrics: RiskMetrics) -> Tuple[bool, str]:
        """Vérifie les conditions composites (combinaisons intelligentes)"""

        # Condition composite 1: Drawdown modéré + Volatilité élevée
        if (risk_metrics.max_drawdown > Decimal("0.03") and  # 3% drawdown
            risk_metrics.portfolio_volatility and risk_metrics.portfolio_volatility > Decimal("0.05")):  # 5% volatilité

            # Vérifier si c'est une tendance récente
            if len(self.metrics_history) >= 5:
                recent_drawdowns = [m.max_drawdown for m in self.metrics_history[-5:]]
                if all(dd > Decimal("0.02") for dd in recent_drawdowns):  # Drawdown persistant
                    return True, "Persistent drawdown + high volatility combination"

        # Condition composite 2: Regime de marché crisis + perte récente
        if self.market_regime == "crisis":
            recent_loss = await self._calculate_recent_loss(timedelta(minutes=30))
            if recent_loss > Decimal("0.02"):  # 2% perte en 30 min pendant crise
                return True, "Crisis market regime + rapid recent loss"

        # Condition composite 3: Multiple conditions moyennes simultanées
        moderate_conditions = 0

        if risk_metrics.max_drawdown > Decimal("0.02"):
            moderate_conditions += 1
        if risk_metrics.portfolio_volatility and risk_metrics.portfolio_volatility > Decimal("0.03"):
            moderate_conditions += 1
        if risk_metrics.concentration_risk > Decimal("0.4"):
            moderate_conditions += 1

        if moderate_conditions >= 3:
            return True, f"Multiple moderate risk conditions active ({moderate_conditions})"

        return False, ""

    # === Méthodes utilitaires ===

    async def _calculate_recent_loss(self, period: timedelta) -> Decimal:
        """Calcule la perte sur une période récente"""
        cutoff_time = datetime.utcnow() - period
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if len(recent_metrics) >= 2:
            initial_value = recent_metrics[0].portfolio_value
            current_value = recent_metrics[-1].portfolio_value

            if initial_value > 0:
                return (initial_value - current_value) / initial_value

        return Decimal("0")

    async def _schedule_auto_reset(self) -> None:
        """Programme un reset automatique après cooldown"""
        await asyncio.sleep(self.global_cooldown.total_seconds())

        if self.status == BreakStatus.TRIGGERED:
            success = await self.reset(manual=False)
            if success:
                logger.info("Circuit breaker auto-reset completed")

    def _record_metrics(self, metrics: RiskMetrics) -> None:
        """Enregistre les métriques dans l'historique"""
        self.metrics_history.append(metrics)

        # Limiter taille historique
        if len(self.metrics_history) > self.max_history_length:
            self.metrics_history = self.metrics_history[-self.max_history_length:]

        # Détecter régime de marché
        self._detect_market_regime()

    def _detect_market_regime(self) -> None:
        """Détecte le régime de marché actuel"""
        if len(self.metrics_history) < 10:
            return

        recent_volatilities = [
            m.portfolio_volatility for m in self.metrics_history[-10:]
            if m.portfolio_volatility is not None
        ]

        if recent_volatilities:
            avg_volatility = statistics.mean(recent_volatilities)

            if avg_volatility > Decimal("0.08"):  # 8% volatilité
                self.market_regime = "crisis"
            elif avg_volatility > Decimal("0.04"):  # 4% volatilité
                self.market_regime = "volatile"
            else:
                self.market_regime = "normal"

    def _create_default_configs(self) -> List[CircuitBreakerConfig]:
        """Crée les configurations par défaut"""
        return [
            # Drawdown portfolio
            CircuitBreakerConfig(
                condition=BreakCondition.PORTFOLIO_DRAWDOWN,
                threshold_value=Decimal("0.15"),  # 15%
                cooldown_period=timedelta(hours=1),
                severity_level=5
            ),

            # Perte rapide
            CircuitBreakerConfig(
                condition=BreakCondition.RAPID_LOSS,
                threshold_value=Decimal("0.05"),  # 5% en 15 min
                lookback_period=timedelta(minutes=15),
                cooldown_period=timedelta(minutes=30),
                severity_level=4
            ),

            # Volatilité de marché
            CircuitBreakerConfig(
                condition=BreakCondition.MARKET_VOLATILITY,
                threshold_value=Decimal("0.10"),  # 10% volatilité
                cooldown_period=timedelta(minutes=45),
                severity_level=3
            ),

            # Crise de liquidité
            CircuitBreakerConfig(
                condition=BreakCondition.LIQUIDITY_CRISIS,
                threshold_value=Decimal("0.2"),  # Score liquidité < 0.2
                cooldown_period=timedelta(hours=2),
                severity_level=4
            ),

            # Breakdown corrélations
            CircuitBreakerConfig(
                condition=BreakCondition.CORRELATION_BREAKDOWN,
                threshold_value=Decimal("0.3"),  # Seuil corrélation
                lookback_period=timedelta(hours=1),
                cooldown_period=timedelta(hours=3),
                severity_level=3
            ),

            # Dégradation exécution
            CircuitBreakerConfig(
                condition=BreakCondition.EXECUTION_DEGRADATION,
                threshold_value=Decimal("0.5"),  # Score qualité < 0.5
                lookback_period=timedelta(minutes=30),
                cooldown_period=timedelta(minutes=20),
                severity_level=2
            )
        ]