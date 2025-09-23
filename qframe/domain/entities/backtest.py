"""
Domain Layer: Backtest Entities
==============================

Entités du domaine pour le backtesting.
Contient les configurations, résultats et métriques de backtests.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd


class BacktestStatus(str, Enum):
    """Statuts d'un backtest"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BacktestType(str, Enum):
    """Types de backtest"""
    SINGLE_PERIOD = "single_period"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    ROLLING_WINDOW = "rolling_window"
    MULTI_STRATEGY = "multi_strategy"


class RebalanceFrequency(str, Enum):
    """Fréquences de rebalancement"""
    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class BacktestConfiguration:
    """Configuration d'un backtest"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Période de test
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime = field(default_factory=datetime.now)

    # Configuration financière
    initial_capital: Decimal = Decimal("100000")
    benchmark_symbol: Optional[str] = None

    # Stratégies à tester
    strategy_ids: List[str] = field(default_factory=list)
    strategy_allocations: Dict[str, Decimal] = field(default_factory=dict)  # allocation par stratégie

    # Configuration de trading
    transaction_cost: Decimal = Decimal("0.001")  # 0.1%
    slippage: Decimal = Decimal("0.0005")  # 0.05%
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY

    # Configuration de risk
    max_position_size: Decimal = Decimal("0.1")  # 10% max par position
    max_leverage: Decimal = Decimal("1.0")  # Pas de leverage par défaut
    stop_loss_percentage: Optional[Decimal] = None
    take_profit_percentage: Optional[Decimal] = None

    # Configuration spécifique au type de backtest
    backtest_type: BacktestType = BacktestType.SINGLE_PERIOD
    walk_forward_config: Optional['WalkForwardConfig'] = None
    monte_carlo_config: Optional['MonteCarloConfig'] = None

    # Métadonnées
    tags: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    def validate(self) -> List[str]:
        """Valide la configuration et retourne les erreurs"""
        errors = []

        if self.end_date <= self.start_date:
            errors.append("End date must be after start date")

        if self.initial_capital <= 0:
            errors.append("Initial capital must be positive")

        if not self.strategy_ids:
            errors.append("At least one strategy must be specified")

        if self.transaction_cost < 0 or self.transaction_cost > Decimal("0.1"):
            errors.append("Transaction cost must be between 0 and 10%")

        if self.max_position_size <= 0 or self.max_position_size > 1:
            errors.append("Max position size must be between 0 and 100%")

        # Validation des allocations
        if self.strategy_allocations:
            total_allocation = sum(self.strategy_allocations.values())
            if abs(total_allocation - Decimal("1.0")) > Decimal("0.01"):
                errors.append("Strategy allocations must sum to 100%")

        return errors


@dataclass
class WalkForwardConfig:
    """Configuration pour walk-forward analysis"""
    training_period_months: int = 12
    testing_period_months: int = 3
    step_months: int = 1
    min_training_observations: int = 252  # 1 année minimum
    reoptimize_parameters: bool = True


@dataclass
class MonteCarloConfig:
    """Configuration pour simulation Monte Carlo"""
    num_simulations: int = 1000
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.75, 0.95])
    bootstrap_method: str = "stationary"  # "stationary", "block", "circular"
    block_size: Optional[int] = None  # Pour block bootstrap


@dataclass
class BacktestMetrics:
    """Métriques de performance d'un backtest"""

    # Métriques de base
    total_return: Decimal = Decimal("0")
    annualized_return: Decimal = Decimal("0")
    volatility: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")

    # Métriques de risque
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_duration: int = 0  # jours
    value_at_risk_95: Decimal = Decimal("0")
    expected_shortfall_95: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")

    # Métriques de trading
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    average_trade_return: Decimal = Decimal("0")
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")

    # Métriques par rapport au benchmark
    alpha: Optional[Decimal] = None
    beta: Optional[Decimal] = None
    information_ratio: Optional[Decimal] = None
    tracking_error: Optional[Decimal] = None

    # Métriques avancées
    tail_ratio: Decimal = Decimal("0")
    skewness: Decimal = Decimal("0")
    kurtosis: Decimal = Decimal("0")
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    # Métriques de stabilité
    return_consistency: Decimal = Decimal("0")  # % de périodes positives
    rolling_sharpe_std: Decimal = Decimal("0")  # stabilité du Sharpe

    def get_summary_metrics(self) -> Dict[str, Decimal]:
        """Retourne les métriques principales pour un aperçu rapide"""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor
        }

    def assess_performance(self) -> Dict[str, str]:
        """Évalue la performance et retourne une assessment qualitative"""
        assessment = {}

        # Sharpe Ratio
        if self.sharpe_ratio >= Decimal("2.0"):
            assessment["sharpe"] = "EXCELLENT"
        elif self.sharpe_ratio >= Decimal("1.5"):
            assessment["sharpe"] = "VERY_GOOD"
        elif self.sharpe_ratio >= Decimal("1.0"):
            assessment["sharpe"] = "GOOD"
        elif self.sharpe_ratio >= Decimal("0.5"):
            assessment["sharpe"] = "FAIR"
        else:
            assessment["sharpe"] = "POOR"

        # Max Drawdown
        if abs(self.max_drawdown) <= Decimal("0.05"):
            assessment["drawdown"] = "EXCELLENT"
        elif abs(self.max_drawdown) <= Decimal("0.10"):
            assessment["drawdown"] = "GOOD"
        elif abs(self.max_drawdown) <= Decimal("0.20"):
            assessment["drawdown"] = "ACCEPTABLE"
        else:
            assessment["drawdown"] = "HIGH_RISK"

        # Win Rate
        if self.win_rate >= Decimal("0.60"):
            assessment["win_rate"] = "EXCELLENT"
        elif self.win_rate >= Decimal("0.55"):
            assessment["win_rate"] = "GOOD"
        elif self.win_rate >= Decimal("0.50"):
            assessment["win_rate"] = "FAIR"
        else:
            assessment["win_rate"] = "POOR"

        return assessment


@dataclass
class TradeExecution:
    """Représente une exécution de trade dans le backtest"""
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    price: Decimal
    value: Decimal
    commission: Decimal
    slippage: Decimal
    strategy_id: str
    signal_strength: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Résultat complet d'un backtest"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    configuration_id: str = ""
    name: str = ""

    # État du backtest
    status: BacktestStatus = BacktestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

    # Résultats financiers
    metrics: Optional[BacktestMetrics] = None
    initial_capital: Decimal = Decimal("0")
    final_capital: Decimal = Decimal("0")

    # Données temporelles
    portfolio_values: Optional[pd.Series] = None
    returns: Optional[pd.Series] = None
    positions: Optional[pd.DataFrame] = None
    drawdown_series: Optional[pd.Series] = None

    # Trades exécutés
    trades: List[TradeExecution] = field(default_factory=list)

    # Benchmark comparison
    benchmark_returns: Optional[pd.Series] = None
    benchmark_metrics: Optional[BacktestMetrics] = None

    # Pour walk-forward et Monte Carlo
    sub_results: List['BacktestResult'] = field(default_factory=list)
    confidence_intervals: Dict[str, Dict[str, Decimal]] = field(default_factory=dict)

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[timedelta]:
        """Durée d'exécution du backtest"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def total_trades(self) -> int:
        """Nombre total de trades"""
        return len(self.trades)

    @property
    def trading_period_days(self) -> int:
        """Nombre de jours de la période de trading"""
        if self.portfolio_values is not None and len(self.portfolio_values) > 1:
            return (self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days
        return 0

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques détaillées sur les trades"""
        if not self.trades:
            return {}

        trade_returns = []
        trade_values = []

        for trade in self.trades:
            trade_returns.append(float(trade.value))
            trade_values.append(float(trade.value))

        return {
            "total_trades": len(self.trades),
            "average_trade_value": sum(trade_values) / len(trade_values),
            "largest_trade": max(trade_values),
            "smallest_trade": min(trade_values),
            "trades_per_day": len(self.trades) / max(self.trading_period_days, 1)
        }

    def compare_to_benchmark(self) -> Optional[Dict[str, Decimal]]:
        """Compare les résultats au benchmark"""
        if not self.benchmark_metrics or not self.metrics:
            return None

        return {
            "excess_return": self.metrics.total_return - self.benchmark_metrics.total_return,
            "sharpe_difference": self.metrics.sharpe_ratio - self.benchmark_metrics.sharpe_ratio,
            "volatility_ratio": self.metrics.volatility / self.benchmark_metrics.volatility if self.benchmark_metrics.volatility > 0 else Decimal("0"),
            "max_drawdown_difference": self.metrics.max_drawdown - self.benchmark_metrics.max_drawdown
        }