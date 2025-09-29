"""
Strategy Marketplace
===================

Decentralized marketplace for trading strategies with advanced validation,
peer review, and incentive mechanisms.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Set
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
import asyncio
import json
import hashlib
from pathlib import Path
import uuid

from qframe.core.container import injectable
from qframe.core.interfaces import Strategy
from qframe.core.config import FrameworkConfig
from qframe.research.backtesting.advanced_backtesting_engine import AdvancedBacktestingEngine

logger = logging.getLogger(__name__)


class StrategyCategory(Enum):
    """Catégories de stratégies"""
    ARBITRAGE = "arbitrage"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    MACHINE_LEARNING = "machine_learning"
    HIGH_FREQUENCY = "high_frequency"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    OPTIONS = "options"
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    COMMODITIES = "commodities"
    PAIRS_TRADING = "pairs_trading"
    MARKET_MAKING = "market_making"


class ListingType(Enum):
    """Type de listing de stratégie"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ListingStatus(Enum):
    """Statuts des listings"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"


class TransactionType(Enum):
    """Types de transactions"""
    PURCHASE = "purchase"
    SUBSCRIPTION = "subscription"
    REVENUE_SHARE = "revenue_share"
    FREE_TRIAL = "free_trial"


class TransactionStatus(Enum):
    """Statuts des transactions"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class PricingModel(Enum):
    """Modèles de prix"""
    FREE = "free"
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    REVENUE_SHARE = "revenue_share"
    PERFORMANCE_FEE = "performance_fee"
    HYBRID = "hybrid"


@dataclass
class StrategyMetrics:
    """Métriques de performance d'une stratégie"""
    # Performance
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float

    # Risk
    volatility: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float

    # Trading
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    max_consecutive_losses: int

    # Time periods
    backtest_start: datetime
    backtest_end: datetime
    out_of_sample_start: Optional[datetime] = None
    out_of_sample_end: Optional[datetime] = None

    # Validation
    walk_forward_validated: bool = False
    monte_carlo_validated: bool = False
    peer_reviewed: bool = False
    live_validated: bool = False

    # Market conditions
    market_regimes_tested: List[str] = field(default_factory=list)
    stress_tested: bool = False
    correlation_analysis: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyReview:
    """Review d'une stratégie par un pair"""
    review_id: str
    strategy_id: str
    reviewer_id: str
    reviewer_reputation: float

    # Notation
    overall_rating: float  # 0-10
    code_quality: float
    documentation_quality: float
    methodology_soundness: float
    risk_management: float
    originality: float

    # Commentaires
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    detailed_analysis: str

    # Validation
    reproduced_results: bool = False
    independent_validation: bool = False
    validation_notes: str = ""

    # Métadonnées
    review_date: datetime = field(default_factory=datetime.utcnow)
    time_spent_hours: float = 0.0
    reviewer_expertise_tags: List[str] = field(default_factory=list)

    # État
    is_verified: bool = False
    helpful_votes: int = 0
    total_votes: int = 0


@dataclass
class StrategyListing:
    """Listing d'une stratégie sur le marketplace"""
    listing_id: str
    strategy_id: str
    owner_id: str

    # Métadonnées de base
    name: str
    description: str
    category: StrategyCategory
    tags: List[str]

    # Contenu
    strategy_code: str
    documentation: str
    example_usage: str

    # Pricing
    pricing_model: PricingModel

    # Default fields
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    price: Decimal = Decimal("0")
    revenue_share_percentage: Optional[float] = None
    performance_fee_percentage: Optional[float] = None

    # Métriques
    performance_metrics: Optional[StrategyMetrics] = None

    # État
    status: ListingStatus = ListingStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None

    # Engagement
    downloads: int = 0
    views: int = 0
    favorites: int = 0
    reviews: List[StrategyReview] = field(default_factory=list)
    average_rating: float = 0.0

    # Validation
    validation_status: str = "pending"
    validation_reports: List[Dict[str, Any]] = field(default_factory=list)
    compliance_checks: Dict[str, bool] = field(default_factory=dict)

    # Revenue
    total_revenue: Decimal = Decimal("0")
    active_subscriptions: int = 0
    trial_conversions: float = 0.0


@dataclass
class MarketplaceTransaction:
    """Transaction sur le marketplace"""
    transaction_id: str
    strategy_id: str
    buyer_id: str
    seller_id: str

    # Détails
    transaction_type: TransactionType
    amount: Decimal
    currency: str = "USD"

    # Timestamps
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # État
    status: TransactionStatus = TransactionStatus.PENDING
    payment_method: str = "credit_card"

    # Métadonnées
    platform_fee: Decimal = Decimal("0")
    seller_revenue: Decimal = Decimal("0")
    tax_amount: Decimal = Decimal("0")

    # Références
    payment_reference: Optional[str] = None
    invoice_id: Optional[str] = None


@injectable
class StrategyMarketplace:
    """
    Marketplace décentralisée pour stratégies de trading.

    Fonctionnalités:
    - Publication et découverte de stratégies
    - Système de review et validation par pairs
    - Modèles de pricing flexibles
    - Analytics et métriques de performance
    - Système de réputation
    - Protection IP et licensing
    - Integration avec backtesting
    """

    def __init__(self, config: FrameworkConfig):
        self.config = config

        # Storage
        self.listings: Dict[str, StrategyListing] = {}
        self.transactions: Dict[str, MarketplaceTransaction] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}

        # Validation
        self.backtest_engine = AdvancedBacktestingEngine(config)
        self.validators: List[callable] = []

        # Pricing
        self.platform_fee_rate = Decimal("0.05")  # 5%
        self.review_incentive_pool = Decimal("10000")  # Pool pour récompenser les reviewers

        # Métriques
        self.marketplace_metrics = {
            "total_listings": 0,
            "active_listings": 0,
            "total_transactions": 0,
            "total_volume": Decimal("0"),
            "average_rating": 0.0,
            "user_satisfaction": 0.0
        }

        logger.info("Strategy Marketplace initialized")

    async def create_listing(
        self,
        owner_id: str,
        strategy: Strategy,
        listing_data: Dict[str, Any]
    ) -> StrategyListing:
        """
        Crée un nouveau listing de stratégie.

        Args:
            owner_id: ID du propriétaire
            strategy: Instance de la stratégie
            listing_data: Données du listing

        Returns:
            StrategyListing créé
        """

        listing_id = str(uuid.uuid4())
        strategy_id = self._generate_strategy_id(strategy, owner_id)

        # Extraire le code de la stratégie
        strategy_code = await self._extract_strategy_code(strategy)

        # Créer listing
        listing = StrategyListing(
            listing_id=listing_id,
            strategy_id=strategy_id,
            owner_id=owner_id,
            name=listing_data["name"],
            description=listing_data["description"],
            category=StrategyCategory(listing_data["category"]),
            tags=listing_data.get("tags", []),
            strategy_code=strategy_code,
            documentation=listing_data.get("documentation", ""),
            example_usage=listing_data.get("example_usage", ""),
            pricing_model=PricingModel(listing_data["pricing_model"]),
            price=Decimal(str(listing_data.get("price", 0)))
        )

        # Validation initiale
        validation_result = await self._validate_strategy_listing(listing, strategy)
        listing.validation_status = validation_result["status"]
        listing.validation_reports = validation_result["reports"]
        listing.compliance_checks = validation_result["compliance"]

        # Générer métriques de performance
        if "backtest_data" in listing_data:
            listing.performance_metrics = await self._generate_performance_metrics(
                strategy, listing_data["backtest_data"]
            )

        # Stocker
        self.listings[listing_id] = listing
        self.marketplace_metrics["total_listings"] += 1

        logger.info(f"Strategy listing created: {listing_id}")
        return listing

    async def submit_for_review(self, listing_id: str) -> bool:
        """Soumet une stratégie pour review par les pairs"""

        if listing_id not in self.listings:
            raise ValueError(f"Listing not found: {listing_id}")

        listing = self.listings[listing_id]

        # Vérifications pré-review
        if not listing.performance_metrics:
            raise ValueError("Performance metrics required for review")

        if not listing.documentation:
            raise ValueError("Documentation required for review")

        # Changer statut
        listing.status = ListingStatus.PENDING_REVIEW
        listing.updated_at = datetime.utcnow()

        # Notifier reviewers potentiels
        await self._notify_potential_reviewers(listing)

        logger.info(f"Strategy submitted for review: {listing_id}")
        return True

    async def submit_review(
        self,
        strategy_id: str,
        reviewer_id: str,
        review_data: Dict[str, Any]
    ) -> StrategyReview:
        """Soumet une review pour une stratégie"""

        review_id = str(uuid.uuid4())

        review = StrategyReview(
            review_id=review_id,
            strategy_id=strategy_id,
            reviewer_id=reviewer_id,
            reviewer_reputation=self._get_reviewer_reputation(reviewer_id),
            overall_rating=review_data["overall_rating"],
            code_quality=review_data["code_quality"],
            documentation_quality=review_data["documentation_quality"],
            methodology_soundness=review_data["methodology_soundness"],
            risk_management=review_data["risk_management"],
            originality=review_data["originality"],
            summary=review_data["summary"],
            strengths=review_data.get("strengths", []),
            weaknesses=review_data.get("weaknesses", []),
            suggestions=review_data.get("suggestions", []),
            detailed_analysis=review_data.get("detailed_analysis", ""),
            time_spent_hours=review_data.get("time_spent_hours", 0),
            reviewer_expertise_tags=review_data.get("expertise_tags", [])
        )

        # Ajouter review au listing
        listing = self._find_listing_by_strategy_id(strategy_id)
        if listing:
            listing.reviews.append(review)
            listing.average_rating = self._calculate_average_rating(listing.reviews)

            # Récompenser le reviewer
            await self._reward_reviewer(reviewer_id, review)

        logger.info(f"Review submitted: {review_id}")
        return review

    async def purchase_strategy(
        self,
        buyer_id: str,
        listing_id: str,
        payment_method: str = "credit_card"
    ) -> MarketplaceTransaction:
        """Achat d'une stratégie"""

        if listing_id not in self.listings:
            raise ValueError(f"Listing not found: {listing_id}")

        listing = self.listings[listing_id]

        if listing.status != ListingStatus.ACTIVE:
            raise ValueError("Strategy not available for purchase")

        # Créer transaction
        transaction_id = str(uuid.uuid4())
        platform_fee = listing.price * self.platform_fee_rate
        seller_revenue = listing.price - platform_fee

        transaction = MarketplaceTransaction(
            transaction_id=transaction_id,
            strategy_id=listing.strategy_id,
            buyer_id=buyer_id,
            seller_id=listing.owner_id,
            transaction_type=TransactionType.PURCHASE,
            amount=listing.price,
            platform_fee=platform_fee,
            seller_revenue=seller_revenue,
            payment_method=payment_method
        )

        # Simuler traitement paiement
        payment_success = await self._process_payment(transaction)

        if payment_success:
            transaction.status = TransactionStatus.COMPLETED
            transaction.completed_at = datetime.utcnow()

            # Mettre à jour listing
            listing.downloads += 1
            listing.total_revenue += seller_revenue

            # Donner accès à l'acheteur
            await self._grant_strategy_access(buyer_id, listing)

        else:
            transaction.status = TransactionStatus.FAILED

        # Stocker transaction
        self.transactions[transaction_id] = transaction
        self.marketplace_metrics["total_transactions"] += 1
        self.marketplace_metrics["total_volume"] += listing.price

        logger.info(f"Purchase transaction: {transaction_id} - {transaction.status}")
        return transaction

    async def search_strategies(
        self,
        query: Optional[str] = None,
        category: Optional[StrategyCategory] = None,
        tags: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        price_range: Optional[Tuple[Decimal, Decimal]] = None,
        sort_by: str = "popularity"
    ) -> List[StrategyListing]:
        """Recherche de stratégies dans le marketplace"""

        results = []

        for listing in self.listings.values():
            if listing.status != ListingStatus.ACTIVE:
                continue

            # Filtres
            if query and query.lower() not in listing.name.lower() and query.lower() not in listing.description.lower():
                continue

            if category and listing.category != category:
                continue

            if tags and not any(tag in listing.tags for tag in tags):
                continue

            if min_rating and listing.average_rating < min_rating:
                continue

            if price_range and not (price_range[0] <= listing.price <= price_range[1]):
                continue

            results.append(listing)

        # Tri
        if sort_by == "popularity":
            results.sort(key=lambda x: x.downloads + x.views, reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda x: x.average_rating, reverse=True)
        elif sort_by == "newest":
            results.sort(key=lambda x: x.published_at or x.created_at, reverse=True)
        elif sort_by == "price_low":
            results.sort(key=lambda x: x.price)
        elif sort_by == "price_high":
            results.sort(key=lambda x: x.price, reverse=True)

        return results

    async def get_strategy_analytics(self, listing_id: str) -> Dict[str, Any]:
        """Analytics détaillées pour une stratégie"""

        if listing_id not in self.listings:
            raise ValueError(f"Listing not found: {listing_id}")

        listing = self.listings[listing_id]

        # Calculer métriques
        reviews = listing.reviews
        transactions = [t for t in self.transactions.values() if t.strategy_id == listing.strategy_id]

        analytics = {
            "performance": {
                "views": listing.views,
                "downloads": listing.downloads,
                "favorites": listing.favorites,
                "conversion_rate": listing.downloads / listing.views if listing.views > 0 else 0
            },
            "ratings": {
                "average_rating": listing.average_rating,
                "total_reviews": len(reviews),
                "rating_distribution": self._calculate_rating_distribution(reviews)
            },
            "financial": {
                "total_revenue": float(listing.total_revenue),
                "average_transaction": float(listing.total_revenue / len(transactions)) if transactions else 0,
                "monthly_recurring_revenue": float(listing.total_revenue * Decimal("0.1"))  # Estimation
            },
            "engagement": {
                "review_engagement": len(reviews) / listing.downloads if listing.downloads > 0 else 0,
                "community_score": self._calculate_community_score(listing)
            }
        }

        return analytics

    async def get_marketplace_leaderboard(self, metric: str = "revenue") -> List[Dict[str, Any]]:
        """Leaderboard du marketplace"""

        leaderboard = []

        for listing in self.listings.values():
            if listing.status != ListingStatus.ACTIVE:
                continue

            entry = {
                "listing_id": listing.listing_id,
                "name": listing.name,
                "owner_id": listing.owner_id,
                "category": listing.category.value,
                "rating": listing.average_rating,
                "downloads": listing.downloads,
                "revenue": float(listing.total_revenue)
            }

            if listing.performance_metrics:
                entry.update({
                    "sharpe_ratio": listing.performance_metrics.sharpe_ratio,
                    "max_drawdown": listing.performance_metrics.max_drawdown,
                    "total_return": listing.performance_metrics.total_return
                })

            leaderboard.append(entry)

        # Tri par métrique
        if metric == "revenue":
            leaderboard.sort(key=lambda x: x["revenue"], reverse=True)
        elif metric == "downloads":
            leaderboard.sort(key=lambda x: x["downloads"], reverse=True)
        elif metric == "rating":
            leaderboard.sort(key=lambda x: x["rating"], reverse=True)
        elif metric == "sharpe":
            leaderboard.sort(key=lambda x: x.get("sharpe_ratio", 0), reverse=True)

        return leaderboard[:50]  # Top 50

    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Statistiques globales du marketplace"""

        active_listings = [l for l in self.listings.values() if l.status == ListingStatus.ACTIVE]
        completed_transactions = [t for t in self.transactions.values() if t.status == TransactionStatus.COMPLETED]

        stats = {
            "overview": {
                "total_strategies": len(self.listings),
                "active_strategies": len(active_listings),
                "total_transactions": len(completed_transactions),
                "total_volume": float(sum(t.amount for t in completed_transactions)),
                "active_users": len(set(l.owner_id for l in active_listings))
            },
            "categories": self._get_category_stats(active_listings),
            "pricing": self._get_pricing_stats(active_listings),
            "quality": {
                "average_rating": np.mean([l.average_rating for l in active_listings if l.average_rating > 0]),
                "reviewed_percentage": len([l for l in active_listings if l.reviews]) / len(active_listings) * 100 if active_listings else 0
            },
            "growth": self._calculate_growth_metrics()
        }

        return stats

    # Méthodes utilitaires privées

    def _generate_strategy_id(self, strategy: Strategy, owner_id: str) -> str:
        """Génère un ID unique pour une stratégie"""
        strategy_signature = f"{strategy.__class__.__name__}_{owner_id}_{datetime.utcnow().isoformat()}"
        return hashlib.sha256(strategy_signature.encode()).hexdigest()[:16]

    async def _extract_strategy_code(self, strategy: Strategy) -> str:
        """Extrait le code source d'une stratégie"""
        import inspect
        return inspect.getsource(strategy.__class__)

    async def _validate_strategy_listing(
        self, listing: StrategyListing, strategy: Strategy
    ) -> Dict[str, Any]:
        """Valide un listing de stratégie"""

        validation = {
            "status": "pending",
            "reports": [],
            "compliance": {}
        }

        # Validation du code
        try:
            # Vérifier syntaxe
            compile(listing.strategy_code, '<string>', 'exec')
            validation["compliance"]["syntax_valid"] = True
        except SyntaxError:
            validation["compliance"]["syntax_valid"] = False
            validation["reports"].append("Syntax error in strategy code")

        # Validation de sécurité
        security_check = self._security_scan(listing.strategy_code)
        validation["compliance"]["security_safe"] = security_check["safe"]
        if not security_check["safe"]:
            validation["reports"].extend(security_check["issues"])

        # Validation des métriques
        if listing.performance_metrics:
            validation["compliance"]["has_performance_metrics"] = True
        else:
            validation["compliance"]["has_performance_metrics"] = False

        # Déterminer statut global
        if all(validation["compliance"].values()):
            validation["status"] = "approved"
        else:
            validation["status"] = "needs_review"

        return validation

    def _security_scan(self, code: str) -> Dict[str, Any]:
        """Scan de sécurité du code"""

        dangerous_patterns = [
            "import os", "import subprocess", "exec(", "eval(",
            "__import__", "open(", "file(", "input(", "raw_input("
        ]

        issues = []
        for pattern in dangerous_patterns:
            if pattern in code:
                issues.append(f"Potentially dangerous pattern found: {pattern}")

        return {
            "safe": len(issues) == 0,
            "issues": issues
        }

    async def _generate_performance_metrics(
        self, strategy: Strategy, backtest_data: pd.DataFrame
    ) -> StrategyMetrics:
        """Génère les métriques de performance"""

        # Simuler backtest complet
        try:
            results = await self.backtest_engine.run_backtest(strategy, backtest_data)

            return StrategyMetrics(
                total_return=float(results.performance_metrics.total_return),
                annualized_return=float(results.performance_metrics.annualized_return),
                sharpe_ratio=float(results.performance_metrics.sharpe_ratio),
                max_drawdown=float(results.performance_metrics.max_drawdown),
                calmar_ratio=float(results.performance_metrics.calmar_ratio),
                sortino_ratio=0.0,  # À calculer
                volatility=float(results.performance_metrics.volatility),
                var_95=0.0,  # À calculer
                cvar_95=0.0,  # À calculer
                beta=0.0,  # À calculer
                alpha=0.0,  # À calculer
                total_trades=results.performance_metrics.total_trades,
                win_rate=float(results.performance_metrics.win_rate),
                profit_factor=float(results.performance_metrics.profit_factor),
                avg_trade_return=float(results.performance_metrics.avg_trade_return),
                max_consecutive_losses=0,  # À calculer
                backtest_start=backtest_data.index[0],
                backtest_end=backtest_data.index[-1],
                market_regimes_tested=["normal", "volatile"]
            )

        except Exception as e:
            logger.error(f"Failed to generate performance metrics: {e}")
            # Retourner métriques par défaut
            return StrategyMetrics(
                total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, calmar_ratio=0.0, sortino_ratio=0.0,
                volatility=0.0, var_95=0.0, cvar_95=0.0, beta=0.0, alpha=0.0,
                total_trades=0, win_rate=0.0, profit_factor=0.0,
                avg_trade_return=0.0, max_consecutive_losses=0,
                backtest_start=datetime.utcnow(), backtest_end=datetime.utcnow()
            )

    def _find_listing_by_strategy_id(self, strategy_id: str) -> Optional[StrategyListing]:
        """Trouve un listing par strategy_id"""
        for listing in self.listings.values():
            if listing.strategy_id == strategy_id:
                return listing
        return None

    def _calculate_average_rating(self, reviews: List[StrategyReview]) -> float:
        """Calcule la note moyenne pondérée par réputation"""
        if not reviews:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for review in reviews:
            weight = 1.0 + review.reviewer_reputation  # Base 1.0 + reputation bonus
            weighted_sum += review.overall_rating * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _get_reviewer_reputation(self, reviewer_id: str) -> float:
        """Calcule la réputation d'un reviewer"""
        # Simulation - en production, basé sur historique
        return np.random.uniform(0.5, 2.0)

    async def _notify_potential_reviewers(self, listing: StrategyListing):
        """Notifie les reviewers potentiels"""
        # Simulation - en production, système de notifications
        logger.info(f"Notifying reviewers for strategy: {listing.name}")

    async def _reward_reviewer(self, reviewer_id: str, review: StrategyReview):
        """Récompense un reviewer"""
        reward = Decimal("10") * Decimal(str(review.time_spent_hours))
        logger.info(f"Rewarding reviewer {reviewer_id}: ${reward}")

    async def _process_payment(self, transaction: MarketplaceTransaction) -> bool:
        """Traite un paiement"""
        # Simulation - en production, intégration gateway paiement
        await asyncio.sleep(0.1)  # Simuler délai réseau
        return np.random.random() > 0.05  # 95% de succès

    async def _grant_strategy_access(self, buyer_id: str, listing: StrategyListing):
        """Donne accès à une stratégie à un acheteur"""
        # En production, système de permissions
        logger.info(f"Granting access to {buyer_id} for strategy {listing.name}")

    def _calculate_rating_distribution(self, reviews: List[StrategyReview]) -> Dict[int, int]:
        """Distribution des notes"""
        distribution = {i: 0 for i in range(1, 11)}
        for review in reviews:
            rating = int(round(review.overall_rating))
            distribution[rating] += 1
        return distribution

    def _calculate_community_score(self, listing: StrategyListing) -> float:
        """Score de communauté"""
        # Formule composite basée sur engagement
        if listing.downloads == 0:
            return 0.0

        review_factor = len(listing.reviews) / listing.downloads
        rating_factor = listing.average_rating / 10
        engagement_factor = listing.favorites / listing.downloads

        return (review_factor + rating_factor + engagement_factor) / 3

    def _get_category_stats(self, listings: List[StrategyListing]) -> Dict[str, int]:
        """Statistiques par catégorie"""
        stats = {}
        for listing in listings:
            category = listing.category.value
            stats[category] = stats.get(category, 0) + 1
        return stats

    def _get_pricing_stats(self, listings: List[StrategyListing]) -> Dict[str, Any]:
        """Statistiques de pricing"""
        prices = [float(l.price) for l in listings if l.price > 0]

        if not prices:
            return {"average": 0, "median": 0, "min": 0, "max": 0}

        return {
            "average": np.mean(prices),
            "median": np.median(prices),
            "min": np.min(prices),
            "max": np.max(prices)
        }

    def _calculate_growth_metrics(self) -> Dict[str, float]:
        """Métriques de croissance"""
        # Simulation - en production, basé sur données historiques
        return {
            "monthly_new_strategies": 15,
            "monthly_growth_rate": 8.5,
            "user_retention_rate": 85.2,
            "revenue_growth_rate": 12.3
        }