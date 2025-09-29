"""
Domain Layer: Backtesting Service
================================

Service domain pour orchestrer l'exécution des backtests.
Coordonne les stratégies, données de marché et calculs de performance.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..entities.backtest import (
    BacktestConfiguration, BacktestResult, BacktestMetrics, BacktestStatus,
    BacktestType, TradeExecution, WalkForwardConfig, MonteCarloConfig
)
from ..entities.order import Order, OrderStatus, OrderSide
from ..entities.portfolio import Portfolio, Position
from ..repositories.backtest_repository import BacktestRepository
from ..repositories.strategy_repository import StrategyRepository
from ..repositories.portfolio_repository import PortfolioRepository


@dataclass
class MarketDataPoint:
    """Point de données de marché pour le backtesting"""
    timestamp: datetime
    symbol: str
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    metadata: Dict[str, Any] = None


class BacktestingService:
    """
    Service domain pour l'exécution de backtests.
    Orchestre les stratégies, données et calculs de performance.
    """

    def __init__(
        self,
        backtest_repository: BacktestRepository,
        strategy_repository: StrategyRepository,
        portfolio_repository: PortfolioRepository
    ):
        self.backtest_repository = backtest_repository
        self.strategy_repository = strategy_repository
        self.portfolio_repository = portfolio_repository

    async def run_backtest(self, config: BacktestConfiguration) -> BacktestResult:
        """
        Lance l'exécution d'un backtest selon la configuration donnée.

        Args:
            config: Configuration du backtest

        Returns:
            Résultat du backtest
        """
        # Valider la configuration
        validation_errors = config.validate()
        if validation_errors:
            result = BacktestResult(
                configuration_id=config.id,
                name=config.name,
                status=BacktestStatus.FAILED,
                error_message=f"Configuration invalid: {', '.join(validation_errors)}"
            )
            await self.backtest_repository.save_result(result)
            return result

        # Créer le résultat initial
        result = BacktestResult(
            configuration_id=config.id,
            name=config.name,
            status=BacktestStatus.RUNNING,
            start_time=datetime.utcnow(),
            initial_capital=config.initial_capital
        )

        try:
            # Sauvegarder le statut initial
            await self.backtest_repository.save_result(result)

            # Router vers le type de backtest approprié
            if config.backtest_type == BacktestType.SINGLE_PERIOD:
                await self._run_single_period_backtest(config, result)
            elif config.backtest_type == BacktestType.WALK_FORWARD:
                await self._run_walk_forward_backtest(config, result)
            elif config.backtest_type == BacktestType.MONTE_CARLO:
                await self._run_monte_carlo_backtest(config, result)
            else:
                raise ValueError(f"Unsupported backtest type: {config.backtest_type}")

            # Marquer comme terminé
            result.status = BacktestStatus.COMPLETED
            result.end_time = datetime.utcnow()

        except Exception as e:
            result.status = BacktestStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.utcnow()

        # Sauvegarder le résultat final
        await self.backtest_repository.save_result(result)
        return result

    async def _run_single_period_backtest(
        self,
        config: BacktestConfiguration,
        result: BacktestResult
    ) -> None:
        """Exécute un backtest sur une seule période"""

        # Récupérer les stratégies
        strategies = []
        for strategy_id in config.strategy_ids:
            strategy = await self.strategy_repository.find_by_id(strategy_id)
            if strategy:
                strategies.append(strategy)

        if not strategies:
            raise ValueError("No valid strategies found for backtest")

        # Simuler les données de marché (en production, récupérées d'un data provider)
        market_data = await self._generate_market_data(config.start_date, config.end_date)

        # Créer le portfolio initial
        portfolio = Portfolio(
            name=f"Backtest_{result.id}",
            initial_capital=config.initial_capital
        )

        # Exécuter la simulation
        portfolio_history, trades = await self._simulate_trading(
            config, strategies, market_data, portfolio
        )

        # Calculer les métriques
        returns = self._calculate_returns(portfolio_history)
        result.metrics = await self._calculate_metrics(returns, trades, config)

        # Sauvegarder les données temporelles
        result.portfolio_values = portfolio_history
        result.returns = returns
        result.trades = trades
        result.drawdown_series = self._calculate_drawdown_series(portfolio_history)
        result.final_capital = portfolio_history.iloc[-1] if len(portfolio_history) > 0 else config.initial_capital

    async def _run_walk_forward_backtest(
        self,
        config: BacktestConfiguration,
        result: BacktestResult
    ) -> None:
        """Exécute un backtest walk-forward"""

        if not config.walk_forward_config:
            raise ValueError("Walk-forward configuration is required")

        wf_config = config.walk_forward_config
        current_date = config.start_date
        sub_results = []

        while current_date < config.end_date:
            # Définir les périodes d'entraînement et de test
            train_end = current_date + timedelta(days=wf_config.training_period_months * 30)
            test_start = train_end
            test_end = train_end + timedelta(days=wf_config.testing_period_months * 30)

            if test_end > config.end_date:
                break

            # Créer une configuration pour cette période
            period_config = BacktestConfiguration(
                name=f"{config.name}_WF_{current_date.strftime('%Y%m%d')}",
                start_date=test_start,
                end_date=test_end,
                initial_capital=config.initial_capital,
                strategy_ids=config.strategy_ids,
                transaction_cost=config.transaction_cost,
                slippage=config.slippage,
                backtest_type=BacktestType.SINGLE_PERIOD
            )

            # Exécuter le backtest pour cette période
            period_result = await self.run_backtest(period_config)
            sub_results.append(period_result)

            # Avancer à la période suivante
            current_date += timedelta(days=wf_config.step_months * 30)

        # Agréger les résultats
        result.sub_results = sub_results
        result.metrics = await self._aggregate_walk_forward_metrics(sub_results)

    async def _run_monte_carlo_backtest(
        self,
        config: BacktestConfiguration,
        result: BacktestResult
    ) -> None:
        """Exécute une simulation Monte Carlo"""

        if not config.monte_carlo_config:
            raise ValueError("Monte Carlo configuration is required")

        mc_config = config.monte_carlo_config

        # Exécuter le backtest de base
        base_config = BacktestConfiguration(
            name=f"{config.name}_base",
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            strategy_ids=config.strategy_ids,
            transaction_cost=config.transaction_cost,
            slippage=config.slippage,
            backtest_type=BacktestType.SINGLE_PERIOD
        )

        base_result = await self.run_backtest(base_config)

        # Exécuter les simulations Monte Carlo
        simulation_results = []
        market_data = await self._generate_market_data(config.start_date, config.end_date)

        for i in range(mc_config.num_simulations):
            # Bootstrap des données
            bootstrapped_data = self._bootstrap_market_data(market_data, mc_config.bootstrap_method)

            # Simulation avec données bootstrappées
            sim_config = BacktestConfiguration(
                name=f"{config.name}_sim_{i}",
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=config.initial_capital,
                strategy_ids=config.strategy_ids,
                transaction_cost=config.transaction_cost,
                slippage=config.slippage,
                backtest_type=BacktestType.SINGLE_PERIOD
            )

            # Simuler rapidement sans sauvegarder
            sim_result = await self._quick_simulation(sim_config, bootstrapped_data)
            simulation_results.append(sim_result)

        # Calculer les intervalles de confiance
        result.confidence_intervals = self._calculate_confidence_intervals(
            simulation_results, mc_config.confidence_levels
        )
        result.sub_results = [base_result] + simulation_results[:10]  # Garder seulement quelques exemples
        result.metrics = base_result.metrics

    async def _simulate_trading(
        self,
        config: BacktestConfiguration,
        strategies: List[Any],
        market_data: pd.DataFrame,
        initial_portfolio: Portfolio
    ) -> Tuple[pd.Series, List[TradeExecution]]:
        """
        Simule le trading avec les stratégies sur les données de marché.

        Returns:
            Tuple (portfolio_values, trades)
        """
        portfolio = initial_portfolio
        portfolio_values = []
        trades = []

        # Simuler jour par jour
        for i in range(len(market_data)):
            current_data = market_data.iloc[:i+1]

            if len(current_data) < 50:  # Besoin de données minimum
                portfolio_values.append(float(portfolio.total_value))
                continue

            # Mettre à jour les prix des positions existantes
            await self._update_portfolio_values(portfolio, current_data.iloc[-1])

            # Générer des signaux avec toutes les stratégies
            all_signals = []
            for strategy in strategies:
                try:
                    signals = await self._generate_strategy_signals(strategy, current_data)
                    all_signals.extend(signals)
                except Exception as e:
                    continue  # Ignorer les erreurs de stratégie

            # Exécuter les trades
            if all_signals:
                executed_trades = await self._execute_signals(
                    portfolio, all_signals, current_data.iloc[-1], config
                )
                trades.extend(executed_trades)

            # Rebalancer si nécessaire
            if self._should_rebalance(current_data.index[-1], config.rebalance_frequency):
                rebalance_trades = await self._rebalance_portfolio(
                    portfolio, config, current_data.iloc[-1]
                )
                trades.extend(rebalance_trades)

            portfolio_values.append(float(portfolio.total_value))

        return pd.Series(portfolio_values, index=market_data.index), trades

    async def _generate_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Génère des données de marché synthétiques pour le backtesting"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)  # Pour la reproductibilité

        # Simuler des prix avec tendance et volatilité
        n_points = len(dates)
        returns = np.random.normal(0.0005, 0.02, n_points)  # Drift positif avec volatilité
        prices = 100 * np.exp(np.cumsum(returns))

        # Ajouter des régimes de marché
        volatility_regime = np.random.choice([0.5, 1.0, 2.0], n_points, p=[0.3, 0.5, 0.2])
        trend_regime = np.random.choice([-0.001, 0, 0.001], n_points, p=[0.2, 0.6, 0.2])

        for i in range(1, n_points):
            regime_return = returns[i] * volatility_regime[i] + trend_regime[i]
            prices[i] = prices[i-1] * (1 + regime_return)

        # Créer OHLCV
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
        volumes = np.random.lognormal(15, 0.5, n_points)

        return pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=dates)

    def _calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calcule les returns du portfolio"""
        return portfolio_values.pct_change().dropna()

    async def _calculate_metrics(
        self,
        returns: pd.Series,
        trades: List[TradeExecution],
        config: BacktestConfiguration
    ) -> BacktestMetrics:
        """Calcule les métriques de performance complètes"""

        if len(returns) == 0:
            return BacktestMetrics()

        # Métriques de base
        total_return = (returns + 1).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Métriques de risque
        drawdown_series = self._calculate_drawdown_series(returns.cumsum())
        max_drawdown = drawdown_series.min()
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown_series)

        var_95 = returns.quantile(0.05)
        expected_shortfall_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0

        # Métriques de trading
        winning_trades = len([t for t in trades if t.value > 0])
        losing_trades = len([t for t in trades if t.value < 0])
        win_rate = winning_trades / len(trades) if trades else 0

        profits = sum(t.value for t in trades if t.value > 0)
        losses = abs(sum(t.value for t in trades if t.value < 0))
        profit_factor = profits / losses if losses > 0 else float('inf')

        # Métriques avancées
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return BacktestMetrics(
            total_return=Decimal(str(total_return)),
            annualized_return=Decimal(str(annualized_return)),
            volatility=Decimal(str(volatility)),
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown=Decimal(str(max_drawdown)),
            max_drawdown_duration=max_drawdown_duration,
            value_at_risk_95=Decimal(str(var_95)),
            expected_shortfall_95=Decimal(str(expected_shortfall_95)),
            sortino_ratio=Decimal(str(sortino_ratio)),
            calmar_ratio=Decimal(str(calmar_ratio)),
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=Decimal(str(win_rate)),
            profit_factor=Decimal(str(profit_factor)),
            average_trade_return=Decimal(str(returns.mean())),
            tail_ratio=Decimal(str(abs(returns.quantile(0.95)) / abs(returns.quantile(0.05)))) if returns.quantile(0.05) != 0 else Decimal("0"),
            skewness=Decimal(str(returns.skew())),
            kurtosis=Decimal(str(returns.kurtosis()))
        )

    def _calculate_drawdown_series(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calcule la série de drawdown"""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown

    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calcule la durée maximale de drawdown en jours"""
        underwater = drawdown_series < 0
        periods = []
        current_period = 0

        for is_underwater in underwater:
            if is_underwater:
                current_period += 1
            else:
                if current_period > 0:
                    periods.append(current_period)
                current_period = 0

        return max(periods) if periods else 0

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calcule le ratio de Sortino"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = downside_returns.std()
        return returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else float('inf')

    def _bootstrap_market_data(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Bootstrap des données de marché pour Monte Carlo"""
        if method == "stationary":
            n = len(data)
            indices = np.random.choice(n, n, replace=True)
            return data.iloc[indices].reset_index(drop=True)
        # Autres méthodes de bootstrap peuvent être ajoutées
        return data

    def _calculate_confidence_intervals(
        self,
        simulation_results: List[BacktestResult],
        confidence_levels: List[float]
    ) -> Dict[str, Dict[str, Decimal]]:
        """Calcule les intervalles de confiance des simulations Monte Carlo"""
        intervals = {}

        metrics_data = {}
        for result in simulation_results:
            if result.metrics:
                for attr in ['sharpe_ratio', 'max_drawdown', 'total_return']:
                    if not hasattr(metrics_data, attr):
                        metrics_data[attr] = []
                    metrics_data[attr].append(float(getattr(result.metrics, attr)))

        for metric, values in metrics_data.items():
            if values:
                intervals[metric] = {}
                for cl in confidence_levels:
                    intervals[metric][f'p{int(cl*100)}'] = Decimal(str(np.percentile(values, cl*100)))

        return intervals

    async def _update_portfolio_values(self, portfolio: Portfolio, market_data: pd.Series) -> None:
        """Met à jour les valeurs du portfolio avec les nouveaux prix"""
        # Mise à jour simplifiée - en réalité, il faudrait mapper les symboles
        for position in portfolio.positions:
            if hasattr(market_data, 'close'):
                position.current_price = Decimal(str(market_data['close']))

    async def _generate_strategy_signals(self, strategy: Any, data: pd.DataFrame) -> List[Any]:
        """Génère des signaux pour une stratégie donnée"""
        # Placeholder - en réalité, on appellerait la méthode de génération de signaux de la stratégie
        return []

    async def _execute_signals(
        self,
        portfolio: Portfolio,
        signals: List[Any],
        market_data: pd.Series,
        config: BacktestConfiguration
    ) -> List[TradeExecution]:
        """Exécute les signaux de trading"""
        trades = []
        # Placeholder pour l'exécution des signaux
        return trades

    def _should_rebalance(self, current_date: datetime, frequency: str) -> bool:
        """Détermine si un rebalancement est nécessaire"""
        # Logique de rebalancement selon la fréquence
        return False

    async def _rebalance_portfolio(
        self,
        portfolio: Portfolio,
        config: BacktestConfiguration,
        market_data: pd.Series
    ) -> List[TradeExecution]:
        """Rebalance le portfolio"""
        trades = []
        # Placeholder pour le rebalancement
        return trades

    async def _quick_simulation(
        self,
        config: BacktestConfiguration,
        market_data: pd.DataFrame
    ) -> BacktestResult:
        """Simulation rapide pour Monte Carlo sans sauvegarde"""
        # Version simplifiée pour les simulations Monte Carlo
        result = BacktestResult(
            configuration_id=config.id,
            name=config.name,
            status=BacktestStatus.COMPLETED
        )
        # Simulation simplifiée
        return result

    async def _aggregate_walk_forward_metrics(
        self,
        sub_results: List[BacktestResult]
    ) -> BacktestMetrics:
        """Agrège les métriques de plusieurs périodes walk-forward"""
        valid_results = [r for r in sub_results if r.metrics]

        if not valid_results:
            return BacktestMetrics()

        # Moyennes pondérées des métriques
        total_return = sum(float(r.metrics.total_return) for r in valid_results) / len(valid_results)
        sharpe_ratio = sum(float(r.metrics.sharpe_ratio) for r in valid_results) / len(valid_results)
        max_drawdown = min(float(r.metrics.max_drawdown) for r in valid_results)
        win_rate = sum(float(r.metrics.win_rate) for r in valid_results) / len(valid_results)

        return BacktestMetrics(
            total_return=Decimal(str(total_return)),
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown=Decimal(str(max_drawdown)),
            win_rate=Decimal(str(win_rate)),
            total_trades=sum(r.metrics.total_trades for r in valid_results)
        )