"""
🏗️ QFrame Strategy Builder - Interactive Strategy Creation

High-level interface for building custom trading strategies using QFrame components.
"""

from typing import Any, Dict, List, Optional, Callable, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from qframe.core.interfaces import Strategy, Signal
from qframe.domain.entities.order import OrderSide, OrderType


@dataclass
class StrategyComponent:
    """Base class for strategy components"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalRule(StrategyComponent):
    """Signal generation rule"""
    condition: Callable[[pd.DataFrame], pd.Series] = None
    side: OrderSide = None
    strength: float = 1.0


@dataclass
class RiskRule(StrategyComponent):
    """Risk management rule"""
    check: Callable[[pd.DataFrame, List[Signal]], bool] = None
    max_positions: Optional[int] = None
    max_drawdown: Optional[float] = None


class StrategyBuilder:
    """
    🏗️ Interactive Strategy Builder

    Enables rapid prototyping of trading strategies with:
    - Signal rule composition
    - Risk management integration
    - Feature engineering pipeline
    - Backtesting integration
    """

    def __init__(self, name: str, qframe_api):
        """
        Initialize Strategy Builder

        Args:
            name: Strategy name
            qframe_api: QFrameResearch instance for integration
        """
        self.name = name
        self.qframe_api = qframe_api

        # Strategy components
        self.signal_rules: List[SignalRule] = []
        self.risk_rules: List[RiskRule] = []
        self.feature_pipeline: List[str] = []

        # Configuration
        self.config = {
            "initial_capital": 100000.0,
            "position_size": 0.1,
            "max_positions": 5,
            "stop_loss": None,
            "take_profit": None
        }

        # Generated strategy
        self._strategy_class = None

    def add_signal(
        self,
        name: str,
        condition: Union[str, Callable],
        side: OrderSide,
        strength: float = 1.0,
        description: str = ""
    ) -> "StrategyBuilder":
        """
        Add a signal generation rule

        Args:
            name: Rule name
            condition: Condition function or string expression
            side: Order side (BUY/SELL)
            strength: Signal strength (0-1)
            description: Rule description

        Returns:
            Self for chaining
        """
        if isinstance(condition, str):
            condition = self._compile_expression(condition)

        rule = SignalRule(
            name=name,
            description=description or f"{side.value} signal: {name}",
            condition=condition,
            side=side,
            strength=strength
        )

        self.signal_rules.append(rule)
        return self

    def add_buy_signal(self, name: str, condition: Union[str, Callable], **kwargs) -> "StrategyBuilder":
        """Add a buy signal rule"""
        return self.add_signal(name, condition, OrderSide.BUY, **kwargs)

    def add_sell_signal(self, name: str, condition: Union[str, Callable], **kwargs) -> "StrategyBuilder":
        """Add a sell signal rule"""
        return self.add_signal(name, condition, OrderSide.SELL, **kwargs)

    def add_risk_rule(
        self,
        name: str,
        check: Union[str, Callable],
        description: str = "",
        **parameters
    ) -> "StrategyBuilder":
        """
        Add a risk management rule

        Args:
            name: Rule name
            check: Risk check function or expression
            description: Rule description
            **parameters: Risk parameters

        Returns:
            Self for chaining
        """
        if isinstance(check, str):
            check = self._compile_risk_check(check)

        rule = RiskRule(
            name=name,
            description=description or f"Risk check: {name}",
            check=check,
            parameters=parameters
        )

        self.risk_rules.append(rule)
        return self

    def add_features(self, features: List[str]) -> "StrategyBuilder":
        """
        Add feature engineering steps

        Args:
            features: List of feature names to compute

        Returns:
            Self for chaining
        """
        self.feature_pipeline.extend(features)
        return self

    def set_config(self, **config) -> "StrategyBuilder":
        """
        Set strategy configuration

        Args:
            **config: Configuration parameters

        Returns:
            Self for chaining
        """
        self.config.update(config)
        return self

    def _compile_expression(self, expression: str) -> Callable:
        """Compile string expression to function"""
        def compiled_condition(data: pd.DataFrame) -> pd.Series:
            # Create local namespace with common functions
            namespace = {
                'data': data,
                'pd': pd,
                'np': np,
                'sma': lambda x, n: x.rolling(n).mean(),
                'ema': lambda x, n: x.ewm(span=n).mean(),
                'rsi': self._calculate_rsi,
                'bb_upper': lambda x, n: x.rolling(n).mean() + 2 * x.rolling(n).std(),
                'bb_lower': lambda x, n: x.rolling(n).mean() - 2 * x.rolling(n).std(),
                'close': data.get('close', data.get('Close', pd.Series())),
                'volume': data.get('volume', data.get('Volume', pd.Series())),
                'high': data.get('high', data.get('High', pd.Series())),
                'low': data.get('low', data.get('Low', pd.Series())),
            }

            try:
                result = eval(expression, {"__builtins__": {}}, namespace)
                if isinstance(result, (bool, np.bool_)):
                    return pd.Series([result] * len(data), index=data.index)
                return result.astype(bool) if hasattr(result, 'astype') else pd.Series(result, dtype=bool)
            except Exception as e:
                print(f"⚠️ Error in expression '{expression}': {e}")
                return pd.Series([False] * len(data), index=data.index)

        return compiled_condition

    def _compile_risk_check(self, expression: str) -> Callable:
        """Compile risk check expression"""
        def compiled_check(data: pd.DataFrame, signals: List[Signal]) -> bool:
            try:
                namespace = {
                    'data': data,
                    'signals': signals,
                    'len': len,
                    'any': any,
                    'all': all,
                    'max': max,
                    'min': min,
                }
                return bool(eval(expression, {"__builtins__": {}}, namespace))
            except Exception as e:
                print(f"⚠️ Error in risk check '{expression}': {e}")
                return False

        return compiled_check

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def build(self) -> Strategy:
        """
        Build the strategy from components

        Returns:
            Compiled Strategy instance
        """
        class BuiltStrategy(Strategy):
            def __init__(self, builder: StrategyBuilder):
                self.builder = builder
                self.name = builder.name
                self.signal_rules = builder.signal_rules
                self.risk_rules = builder.risk_rules
                self.config = builder.config

            def generate_signals(self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> List[Signal]:
                """Generate trading signals using built rules"""
                signals = []

                # Use features if available, otherwise use raw data
                working_data = features if features is not None else data

                # Generate signals from each rule
                for rule in self.signal_rules:
                    try:
                        condition_result = rule.condition(working_data)

                        # Create signals where condition is True
                        for timestamp, should_signal in condition_result.items():
                            if should_signal:
                                signals.append(Signal(
                                    timestamp=timestamp,
                                    side=rule.side,
                                    strength=rule.strength,
                                    source=rule.name,
                                    metadata={"rule": rule.name, "description": rule.description}
                                ))
                    except Exception as e:
                        print(f"⚠️ Error in signal rule '{rule.name}': {e}")

                # Apply risk filters
                filtered_signals = self._apply_risk_filters(working_data, signals)

                return filtered_signals

            def _apply_risk_filters(self, data: pd.DataFrame, signals: List[Signal]) -> List[Signal]:
                """Apply risk management filters to signals"""
                if not self.risk_rules:
                    return signals

                filtered_signals = []

                for signal in signals:
                    should_include = True

                    for risk_rule in self.risk_rules:
                        try:
                            if not risk_rule.check(data, [signal]):
                                should_include = False
                                break
                        except Exception as e:
                            print(f"⚠️ Error in risk rule '{risk_rule.name}': {e}")
                            should_include = False
                            break

                    if should_include:
                        filtered_signals.append(signal)

                return filtered_signals

            def get_strategy_info(self) -> Dict[str, Any]:
                """Get strategy information"""
                return {
                    "name": self.name,
                    "type": "BuiltStrategy",
                    "signal_rules": len(self.signal_rules),
                    "risk_rules": len(self.risk_rules),
                    "config": self.config,
                    "rule_descriptions": [rule.description for rule in self.signal_rules]
                }

        self._strategy_class = BuiltStrategy(self)
        return self._strategy_class

    async def backtest(
        self,
        data: Optional[pd.DataFrame] = None,
        symbol: str = "BTC/USDT",
        days: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Backtest the built strategy

        Args:
            data: Market data (optional, will fetch if not provided)
            symbol: Trading pair for data fetching
            days: Days of data to fetch
            **kwargs: Additional backtest parameters

        Returns:
            Backtest results
        """
        if not self._strategy_class:
            self.build()

        # Get data if not provided
        if data is None:
            print(f"📊 Fetching {days} days of {symbol} data...")
            data = await self.qframe_api.get_data(symbol, days=days)

        # Add features if specified
        if self.feature_pipeline:
            print(f"🔧 Computing {len(self.feature_pipeline)} features...")
            data = await self.qframe_api.compute_features(data)

        # Register strategy temporarily
        strategy_name = f"built_{self.name}"
        # TODO: Register with container for backtesting

        print(f"🚀 Running backtest for {self.name}...")

        # Run backtest using QFrame API
        return await self.qframe_api.backtest(
            strategy=strategy_name,
            data=data,
            initial_capital=self.config.get("initial_capital", 100000),
            **kwargs
        )

    def summary(self) -> str:
        """Get strategy summary"""
        summary = f"""
🏗️ Strategy Builder Summary: {self.name}
{'=' * 50}

📊 Signal Rules ({len(self.signal_rules)}):
"""
        for i, rule in enumerate(self.signal_rules, 1):
            summary += f"  {i}. {rule.name} ({rule.side.value}) - {rule.description}\n"

        summary += f"\n🛡️ Risk Rules ({len(self.risk_rules)}):\n"
        for i, rule in enumerate(self.risk_rules, 1):
            summary += f"  {i}. {rule.name} - {rule.description}\n"

        summary += f"\n🔧 Features ({len(self.feature_pipeline)}):\n"
        for i, feature in enumerate(self.feature_pipeline, 1):
            summary += f"  {i}. {feature}\n"

        summary += f"\n⚙️ Configuration:\n"
        for key, value in self.config.items():
            summary += f"  • {key}: {value}\n"

        return summary

    def __str__(self):
        return self.summary()


# Convenience functions for common strategies

def mean_reversion_strategy(
    name: str = "mean_reversion",
    lookback: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5
) -> StrategyBuilder:
    """Create a mean reversion strategy template"""
    from qframe.research import QFrameResearch

    qf = QFrameResearch()
    builder = StrategyBuilder(name, qf)

    # Add mean reversion signals
    builder.add_buy_signal(
        "mean_reversion_buy",
        f"(close - sma(close, {lookback})) / data.close.rolling({lookback}).std() < -{entry_threshold}",
        description=f"Buy when price is {entry_threshold} std below {lookback}-period mean"
    )

    builder.add_sell_signal(
        "mean_reversion_sell",
        f"(close - sma(close, {lookback})) / data.close.rolling({lookback}).std() > {exit_threshold}",
        description=f"Sell when price returns to {exit_threshold} std from mean"
    )

    # Add risk management
    builder.add_risk_rule(
        "max_positions",
        "len(signals) < 3",
        description="Limit to 3 concurrent positions"
    )

    builder.set_config(
        initial_capital=100000,
        position_size=0.2,
        max_positions=3
    )

    return builder


def momentum_strategy(
    name: str = "momentum",
    short_period: int = 10,
    long_period: int = 30,
    momentum_threshold: float = 0.02
) -> StrategyBuilder:
    """Create a momentum strategy template"""
    from qframe.research import QFrameResearch

    qf = QFrameResearch()
    builder = StrategyBuilder(name, qf)

    # Add momentum signals
    builder.add_buy_signal(
        "momentum_buy",
        f"(sma(close, {short_period}) / sma(close, {long_period}) - 1) > {momentum_threshold}",
        description=f"Buy when short MA > long MA by {momentum_threshold*100}%"
    )

    builder.add_sell_signal(
        "momentum_sell",
        f"(sma(close, {short_period}) / sma(close, {long_period}) - 1) < -{momentum_threshold}",
        description=f"Sell when short MA < long MA by {momentum_threshold*100}%"
    )

    # Add risk management
    builder.add_risk_rule(
        "trend_confirmation",
        f"(data.close.iloc[-1] / data.close.iloc[-{long_period}] - 1) > 0",
        description="Only trade in uptrend"
    )

    builder.set_config(
        initial_capital=100000,
        position_size=0.15,
        max_positions=5
    )

    return builder