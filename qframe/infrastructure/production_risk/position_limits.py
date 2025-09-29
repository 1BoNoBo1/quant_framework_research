"""
Position Limits Manager
======================

Dynamic position limits with adaptive thresholds based on volatility,
liquidity, and market conditions.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import logging

from ...core.container import injectable
from ...domain.entities.position import Position


logger = logging.getLogger(__name__)


class LimitType(str, Enum):
    """Types de limites de position"""
    ABSOLUTE_POSITION = "absolute_position"
    PORTFOLIO_PERCENTAGE = "portfolio_percentage"
    NOTIONAL_VALUE = "notional_value"
    RISK_WEIGHTED = "risk_weighted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    LIQUIDITY_ADJUSTED = "liquidity_adjusted"


class LimitPeriod(str, Enum):
    """Périodes pour les limites"""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class PositionLimit:
    """Limite de position configurée"""
    symbol: str
    limit_type: LimitType
    limit_value: Decimal
    period: LimitPeriod = LimitPeriod.INTRADAY
    enabled: bool = True
    created_at: datetime = datetime.utcnow()
    last_updated: datetime = datetime.utcnow()

    # Paramètres adaptatifs
    volatility_multiplier: Optional[Decimal] = None
    liquidity_multiplier: Optional[Decimal] = None
    base_limit: Optional[Decimal] = None  # Limite de base avant adjustements


@dataclass
class LimitViolation:
    """Violation de limite détectée"""
    symbol: str
    limit_type: LimitType
    current_value: Decimal
    limit_value: Decimal
    violation_percentage: Decimal
    severity: str  # "warning", "breach", "critical"
    message: str
    timestamp: datetime = datetime.utcnow()


@injectable
class PositionLimitManager:
    """
    Gestionnaire de limites de positions avec adaptation dynamique.

    Fonctionnalités:
    - Limites statiques et dynamiques
    - Adaptation basée sur volatilité
    - Adjustment pour liquidité
    - Limites composites (portfolio-wide)
    - Escalade des violations
    """

    def __init__(
        self,
        default_portfolio_limit: Decimal = Decimal("0.20"),  # 20% max par position
        volatility_lookback: timedelta = timedelta(days=30),
        liquidity_lookback: timedelta = timedelta(days=7)
    ):
        self.default_portfolio_limit = default_portfolio_limit
        self.volatility_lookback = volatility_lookback
        self.liquidity_lookback = liquidity_lookback

        # Stockage des limites
        self.position_limits: Dict[str, List[PositionLimit]] = {}
        self.global_limits: List[PositionLimit] = []

        # Historique des violations
        self.violation_history: List[LimitViolation] = []

        # Cache pour données de marché
        self._volatility_cache: Dict[str, Decimal] = {}
        self._liquidity_cache: Dict[str, Decimal] = {}
        self._cache_expiry: Dict[str, datetime] = {}

        # Configuration d'adaptation
        self.volatility_adaptation_enabled = True
        self.liquidity_adaptation_enabled = True

        self._setup_default_limits()

    async def set_position_limit(
        self,
        symbol: str,
        limit_type: LimitType,
        limit_value: Decimal,
        period: LimitPeriod = LimitPeriod.INTRADAY,
        adaptive: bool = True
    ) -> None:
        """
        Définit une limite de position.

        Args:
            symbol: Symbole de l'asset
            limit_type: Type de limite
            limit_value: Valeur de la limite
            period: Période d'application
            adaptive: Si la limite doit s'adapter automatiquement
        """
        if symbol not in self.position_limits:
            self.position_limits[symbol] = []

        # Créer limite
        limit = PositionLimit(
            symbol=symbol,
            limit_type=limit_type,
            limit_value=limit_value,
            period=period,
            base_limit=limit_value if adaptive else None
        )

        # Configurer multiplicateurs adaptatifs
        if adaptive:
            if self.volatility_adaptation_enabled:
                limit.volatility_multiplier = Decimal("1.0")
            if self.liquidity_adaptation_enabled:
                limit.liquidity_multiplier = Decimal("1.0")

        # Remplacer limite existante du même type ou ajouter
        existing_idx = None
        for i, existing_limit in enumerate(self.position_limits[symbol]):
            if existing_limit.limit_type == limit_type and existing_limit.period == period:
                existing_idx = i
                break

        if existing_idx is not None:
            self.position_limits[symbol][existing_idx] = limit
            logger.info(f"Updated limit for {symbol}: {limit_type.value} = {limit_value}")
        else:
            self.position_limits[symbol].append(limit)
            logger.info(f"Added limit for {symbol}: {limit_type.value} = {limit_value}")

    async def remove_position_limit(
        self,
        symbol: str,
        limit_type: LimitType,
        period: LimitPeriod = LimitPeriod.INTRADAY
    ) -> bool:
        """Supprime une limite de position"""
        if symbol not in self.position_limits:
            return False

        original_count = len(self.position_limits[symbol])

        self.position_limits[symbol] = [
            limit for limit in self.position_limits[symbol]
            if not (limit.limit_type == limit_type and limit.period == period)
        ]

        removed = len(self.position_limits[symbol]) < original_count

        if removed:
            logger.info(f"Removed limit for {symbol}: {limit_type.value}")

        return removed

    async def check_position_limits(self, position: Position) -> List[Dict[str, Any]]:
        """
        Vérifie toutes les limites pour une position.

        Args:
            position: Position à vérifier

        Returns:
            Liste des violations détectées
        """
        violations = []

        # Vérifier limites spécifiques au symbole
        symbol_limits = self.position_limits.get(position.symbol, [])

        for limit in symbol_limits:
            if not limit.enabled:
                continue

            # Adapter la limite si nécessaire
            adapted_limit = await self._adapt_limit(limit, position)

            # Vérifier la limite
            violation = await self._check_single_limit(position, adapted_limit)
            if violation:
                violations.append({
                    'limit_type': violation.limit_type.value,
                    'current_value': violation.current_value,
                    'limit_value': violation.limit_value,
                    'violation_percentage': violation.violation_percentage,
                    'severity': violation.severity,
                    'message': violation.message
                })

                # Enregistrer dans historique
                self.violation_history.append(violation)

        # Vérifier limites globales
        for global_limit in self.global_limits:
            if not global_limit.enabled:
                continue

            violation = await self._check_single_limit(position, global_limit)
            if violation:
                violations.append({
                    'limit_type': violation.limit_type.value,
                    'current_value': violation.current_value,
                    'limit_value': violation.limit_value,
                    'violation_percentage': violation.violation_percentage,
                    'severity': violation.severity,
                    'message': violation.message
                })

                self.violation_history.append(violation)

        return violations

    async def check_portfolio_limits(
        self,
        positions: List[Position],
        portfolio_value: Decimal
    ) -> List[Dict[str, Any]]:
        """
        Vérifie les limites au niveau du portfolio.

        Args:
            positions: Toutes les positions du portfolio
            portfolio_value: Valeur totale du portfolio

        Returns:
            Liste des violations portfolio-wide
        """
        violations = []

        # Vérifier concentration par asset
        for position in positions:
            if portfolio_value > 0:
                concentration = abs(position.market_value or Decimal("0")) / portfolio_value

                if concentration > self.default_portfolio_limit:
                    violation = LimitViolation(
                        symbol=position.symbol,
                        limit_type=LimitType.PORTFOLIO_PERCENTAGE,
                        current_value=concentration,
                        limit_value=self.default_portfolio_limit,
                        violation_percentage=(concentration - self.default_portfolio_limit) / self.default_portfolio_limit * 100,
                        severity="warning" if concentration < self.default_portfolio_limit * Decimal("1.2") else "critical",
                        message=f"Position concentration {concentration:.2%} exceeds limit {self.default_portfolio_limit:.2%}"
                    )

                    violations.append({
                        'limit_type': violation.limit_type.value,
                        'current_value': violation.current_value,
                        'limit_value': violation.limit_value,
                        'violation_percentage': violation.violation_percentage,
                        'severity': violation.severity,
                        'message': violation.message
                    })

                    self.violation_history.append(violation)

        # Vérifier exposition totale
        total_long_exposure = sum(
            pos.market_value for pos in positions
            if pos.quantity > 0 and pos.market_value
        )

        total_short_exposure = sum(
            abs(pos.market_value) for pos in positions
            if pos.quantity < 0 and pos.market_value
        )

        # Limite d'exposition brute (long + short)
        gross_exposure = total_long_exposure + total_short_exposure
        max_gross_exposure = portfolio_value * Decimal("2.0")  # 200% leverage max

        if gross_exposure > max_gross_exposure:
            violation = LimitViolation(
                symbol="PORTFOLIO",
                limit_type=LimitType.NOTIONAL_VALUE,
                current_value=gross_exposure,
                limit_value=max_gross_exposure,
                violation_percentage=(gross_exposure - max_gross_exposure) / max_gross_exposure * 100,
                severity="critical",
                message=f"Gross exposure {gross_exposure} exceeds limit {max_gross_exposure}"
            )

            violations.append({
                'limit_type': violation.limit_type.value,
                'current_value': violation.current_value,
                'limit_value': violation.limit_value,
                'violation_percentage': violation.violation_percentage,
                'severity': violation.severity,
                'message': violation.message
            })

            self.violation_history.append(violation)

        return violations

    async def get_limits_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des limites configurées"""
        total_limits = sum(len(limits) for limits in self.position_limits.values())
        total_limits += len(self.global_limits)

        # Violations récentes
        recent_violations = [
            v for v in self.violation_history
            if (datetime.utcnow() - v.timestamp).hours < 24
        ]

        violation_stats = {}
        for severity in ["warning", "breach", "critical"]:
            violation_stats[severity] = len([v for v in recent_violations if v.severity == severity])

        return {
            "total_limits_configured": total_limits,
            "symbols_with_limits": len(self.position_limits),
            "global_limits": len(self.global_limits),
            "recent_violations_24h": len(recent_violations),
            "violations_by_severity": violation_stats,
            "adaptive_features": {
                "volatility_adaptation": self.volatility_adaptation_enabled,
                "liquidity_adaptation": self.liquidity_adaptation_enabled
            }
        }

    # === Adaptation dynamique ===

    async def _adapt_limit(self, limit: PositionLimit, position: Position) -> PositionLimit:
        """Adapte une limite basée sur les conditions de marché"""
        if not limit.base_limit:
            return limit  # Pas d'adaptation configurée

        adapted_value = limit.base_limit

        # Adaptation à la volatilité
        if limit.volatility_multiplier is not None and self.volatility_adaptation_enabled:
            volatility = await self._get_volatility(position.symbol)
            if volatility > 0:
                # Plus volatil = limite plus restrictive
                vol_adjustment = max(Decimal("0.5"), Decimal("2.0") - volatility * 10)
                adapted_value *= vol_adjustment

        # Adaptation à la liquidité
        if limit.liquidity_multiplier is not None and self.liquidity_adaptation_enabled:
            liquidity = await self._get_liquidity(position.symbol)
            if liquidity > 0:
                # Plus liquide = limite plus permissive
                liq_adjustment = min(Decimal("2.0"), Decimal("0.5") + liquidity)
                adapted_value *= liq_adjustment

        # Créer limite adaptée
        adapted_limit = PositionLimit(
            symbol=limit.symbol,
            limit_type=limit.limit_type,
            limit_value=adapted_value,
            period=limit.period,
            enabled=limit.enabled,
            created_at=limit.created_at,
            last_updated=datetime.utcnow(),
            volatility_multiplier=limit.volatility_multiplier,
            liquidity_multiplier=limit.liquidity_multiplier,
            base_limit=limit.base_limit
        )

        return adapted_limit

    async def _get_volatility(self, symbol: str) -> Decimal:
        """Récupère ou calcule la volatilité d'un asset"""
        cache_key = f"vol_{symbol}"

        # Vérifier cache
        if (cache_key in self._volatility_cache and
            cache_key in self._cache_expiry and
            self._cache_expiry[cache_key] > datetime.utcnow()):
            return self._volatility_cache[cache_key]

        # Calculer volatilité (simulation - dans la réalité, utiliserait des données historiques)
        # Valeurs typiques de volatilité annualisée
        volatility_map = {
            "BTC": Decimal("0.80"),   # 80% volatilité crypto
            "ETH": Decimal("0.90"),   # 90% volatilité crypto
            "AAPL": Decimal("0.25"),  # 25% volatilité actions
            "TSLA": Decimal("0.60"),  # 60% volatilité actions volatiles
            "SPY": Decimal("0.15"),   # 15% volatilité index
        }

        volatility = volatility_map.get(symbol, Decimal("0.30"))  # 30% par défaut

        # Mettre en cache
        self._volatility_cache[cache_key] = volatility
        self._cache_expiry[cache_key] = datetime.utcnow() + timedelta(hours=1)

        return volatility

    async def _get_liquidity(self, symbol: str) -> Decimal:
        """Récupère ou calcule la liquidité d'un asset"""
        cache_key = f"liq_{symbol}"

        # Vérifier cache
        if (cache_key in self._liquidity_cache and
            cache_key in self._cache_expiry and
            self._cache_expiry[cache_key] > datetime.utcnow()):
            return self._liquidity_cache[cache_key]

        # Calculer liquidité (simulation - dans la réalité, utiliserait volume, bid-ask spreads)
        liquidity_map = {
            "BTC": Decimal("0.85"),   # Très liquide
            "ETH": Decimal("0.80"),   # Très liquide
            "AAPL": Decimal("0.90"),  # Extrêmement liquide
            "TSLA": Decimal("0.75"),  # Assez liquide
            "SPY": Decimal("0.95"),   # Extrêmement liquide
        }

        liquidity = liquidity_map.get(symbol, Decimal("0.50"))  # 50% par défaut

        # Mettre en cache
        self._liquidity_cache[cache_key] = liquidity
        self._cache_expiry[cache_key] = datetime.utcnow() + timedelta(hours=1)

        return liquidity

    # === Vérification des limites ===

    async def _check_single_limit(
        self,
        position: Position,
        limit: PositionLimit
    ) -> Optional[LimitViolation]:
        """Vérifie une limite spécifique contre une position"""

        current_value = await self._get_limit_value(position, limit.limit_type)

        if current_value is None:
            return None

        # Déterminer si limite violée
        if current_value > limit.limit_value:
            violation_percentage = (current_value - limit.limit_value) / limit.limit_value * 100

            # Déterminer sévérité
            if violation_percentage > 50:
                severity = "critical"
            elif violation_percentage > 20:
                severity = "breach"
            else:
                severity = "warning"

            return LimitViolation(
                symbol=position.symbol,
                limit_type=limit.limit_type,
                current_value=current_value,
                limit_value=limit.limit_value,
                violation_percentage=violation_percentage,
                severity=severity,
                message=f"{limit.limit_type.value} limit exceeded: {current_value} > {limit.limit_value}"
            )

        return None

    async def _get_limit_value(
        self,
        position: Position,
        limit_type: LimitType
    ) -> Optional[Decimal]:
        """Extrait la valeur à comparer selon le type de limite"""

        if limit_type == LimitType.ABSOLUTE_POSITION:
            return abs(position.quantity)

        elif limit_type == LimitType.NOTIONAL_VALUE:
            return abs(position.market_value or Decimal("0"))

        elif limit_type == LimitType.PORTFOLIO_PERCENTAGE:
            # Nécessiterait la valeur du portfolio - placeholder
            return Decimal("0.1")  # 10% placeholder

        elif limit_type == LimitType.RISK_WEIGHTED:
            # Calcul risk-weighted basé sur volatilité
            volatility = await self._get_volatility(position.symbol)
            notional = abs(position.market_value or Decimal("0"))
            return notional * volatility

        elif limit_type == LimitType.VOLATILITY_ADJUSTED:
            # Position size ajustée par volatilité
            volatility = await self._get_volatility(position.symbol)
            base_size = abs(position.quantity)
            return base_size * volatility

        elif limit_type == LimitType.LIQUIDITY_ADJUSTED:
            # Position size ajustée par liquidité
            liquidity = await self._get_liquidity(position.symbol)
            base_size = abs(position.quantity)
            return base_size / liquidity if liquidity > 0 else base_size

        return None

    def _setup_default_limits(self) -> None:
        """Configure les limites par défaut"""
        # Limite globale portfolio
        self.global_limits.append(PositionLimit(
            symbol="*",  # Wildcard pour tous les symboles
            limit_type=LimitType.PORTFOLIO_PERCENTAGE,
            limit_value=self.default_portfolio_limit,
            period=LimitPeriod.INTRADAY
        ))

        logger.info("Default position limits configured")