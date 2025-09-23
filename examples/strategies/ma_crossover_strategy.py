"""
Moving Average Crossover Strategy Example
==========================================

Classic trend-following strategy using fast and slow moving average crossover.

BUY signal: When fast MA crosses above slow MA (Golden Cross)
SELL signal: When fast MA crosses below slow MA (Death Cross)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal

from qframe.domain.entities.strategy import Strategy
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence
from qframe.domain.services.signal_service import SignalService


class MovingAverageCrossoverStrategy:
    """
    Moving Average Crossover Strategy
    
    Parameters:
    -----------
    fast_period : int
        Period for fast moving average (e.g., 10)
    slow_period : int
        Period for slow moving average (e.g., 20)
    symbol : str
        Trading pair symbol (e.g., 'BTC/USDT')
    position_size : float
        Position size as percentage of portfolio (e.g., 0.5 for 50%)
    """
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        symbol: str = "BTC/USDT",
        position_size: float = 0.5
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.symbol = symbol
        self.position_size = position_size
        
        self.fast_ma: Optional[pd.Series] = None
        self.slow_ma: Optional[pd.Series] = None
        self.signals: List[TradingSignal] = []
        
    def calculate_indicators(self, price_data: pd.DataFrame) -> None:
        """
        Calculate moving averages from price data
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with 'close' column containing closing prices
        """
        self.fast_ma = price_data['close'].rolling(window=self.fast_period).mean()
        self.slow_ma = price_data['close'].rolling(window=self.slow_period).mean()
        
    def generate_signals(self, price_data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on MA crossover
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        List[Signal]
            List of generated trading signals
        """
        self.calculate_indicators(price_data)
        
        signals = []
        
        # Start from slow_period to ensure we have valid MA values
        for i in range(self.slow_period, len(price_data)):
            current_fast = self.fast_ma.iloc[i]
            current_slow = self.slow_ma.iloc[i]
            previous_fast = self.fast_ma.iloc[i-1]
            previous_slow = self.slow_ma.iloc[i-1]
            
            # Golden Cross: Fast MA crosses above Slow MA
            if previous_fast <= previous_slow and current_fast > current_slow:
                strength = self._calculate_signal_strength(current_fast, current_slow)
                signal = Signal(
                    symbol=self.symbol,
                    action=SignalAction.BUY,
                    timestamp=price_data.index[i] if hasattr(price_data.index[i], 'to_pydatetime') else datetime.now(),
                    strength=Decimal(str(strength)),
                    confidence=SignalConfidence.HIGH,
                    price=Decimal(str(price_data['close'].iloc[i])),
                    metadata={
                        'fast_ma': float(current_fast),
                        'slow_ma': float(current_slow),
                        'crossover_type': 'golden_cross'
                    }
                )
                signals.append(signal)

            # Death Cross: Fast MA crosses below Slow MA
            elif previous_fast >= previous_slow and current_fast < current_slow:
                strength = self._calculate_signal_strength(current_fast, current_slow)
                signal = Signal(
                    symbol=self.symbol,
                    action=SignalAction.SELL,
                    timestamp=price_data.index[i] if hasattr(price_data.index[i], 'to_pydatetime') else datetime.now(),
                    strength=Decimal(str(strength)),
                    confidence=SignalConfidence.HIGH,
                    price=Decimal(str(price_data['close'].iloc[i])),
                    metadata={
                        'fast_ma': float(current_fast),
                        'slow_ma': float(current_slow),
                        'crossover_type': 'death_cross'
                    }
                )
                signals.append(signal)
        
        self.signals = signals
        return signals
    
    def _calculate_signal_strength(self, fast_ma: float, slow_ma: float) -> float:
        """
        Calculate signal strength based on MA divergence
        
        Strength is based on percentage difference between MAs
        """
        divergence = abs(fast_ma - slow_ma) / slow_ma
        # Normalize to 0-1 range, cap at 1.0
        strength = min(divergence * 10, 1.0)
        return strength
    
    def get_position_size(self, current_price: Decimal, portfolio_value: Decimal) -> Decimal:
        """
        Calculate position size based on strategy parameters
        
        Parameters:
        -----------
        current_price : Decimal
            Current price of the asset
        portfolio_value : Decimal
            Total portfolio value
            
        Returns:
        --------
        Decimal
            Quantity to trade
        """
        allocation = portfolio_value * Decimal(str(self.position_size))
        quantity = allocation / current_price
        return quantity
    
    def get_strategy_state(self) -> Dict:
        """Get current strategy state for monitoring"""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'symbol': self.symbol,
            'position_size': self.position_size,
            'total_signals': len(self.signals),
            'last_fast_ma': float(self.fast_ma.iloc[-1]) if self.fast_ma is not None else None,
            'last_slow_ma': float(self.slow_ma.iloc[-1]) if self.slow_ma is not None else None
        }
    
    def backtest_summary(self, price_data: pd.DataFrame) -> Dict:
        """
        Generate a summary of backtest results
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Historical price data
            
        Returns:
        --------
        Dict
            Summary statistics
        """
        signals = self.generate_signals(price_data)
        
        # Calculate returns from signals
        trades = []
        position = None
        
        for signal in signals:
            if signal.action == SignalAction.BUY and position is None:
                position = {'entry_price': signal.price, 'entry_time': signal.timestamp}
            elif signal.action == SignalAction.SELL and position is not None:
                exit_price = signal.price
                pnl_pct = float((exit_price - position['entry_price']) / position['entry_price'])
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'holding_period': (signal.timestamp - position['entry_time']).total_seconds() / 86400  # days
                })
                position = None
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0
            }
        
        # Calculate statistics
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        total_return = sum(t['pnl_pct'] for t in trades)
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(trades) - len(winning_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'avg_return': total_return / len(trades) if trades else 0,
            'total_return': total_return,
            'avg_holding_period_days': np.mean([t['holding_period'] for t in trades]) if trades else 0,
            'best_trade': max(trades, key=lambda t: t['pnl_pct'])['pnl_pct'] if trades else 0,
            'worst_trade': min(trades, key=lambda t: t['pnl_pct'])['pnl_pct'] if trades else 0,
            'trades': trades
        }


def example_usage():
    """Example usage of the MA Crossover Strategy"""
    
    # Create sample price data (in real use, fetch from exchange or database)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate synthetic BTC price data for demonstration
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * (1 + returns).cumprod()
    
    price_data = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * (1 + abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - abs(np.random.normal(0, 0.01, len(dates)))),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Initialize strategy
    strategy = MovingAverageCrossoverStrategy(
        fast_period=10,
        slow_period=20,
        symbol='BTC/USDT',
        position_size=0.5
    )
    
    # Run backtest
    summary = strategy.backtest_summary(price_data)
    
    print("=" * 60)
    print("Moving Average Crossover Strategy - Backtest Results")
    print("=" * 60)
    print(f"\nStrategy Parameters:")
    print(f"  Fast MA Period: {strategy.fast_period}")
    print(f"  Slow MA Period: {strategy.slow_period}")
    print(f"  Symbol: {strategy.symbol}")
    print(f"  Position Size: {strategy.position_size * 100}%")
    
    print(f"\nBacktest Results:")
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Winning Trades: {summary['winning_trades']}")
    print(f"  Losing Trades: {summary['losing_trades']}")
    print(f"  Win Rate: {summary['win_rate']:.2%}")
    print(f"  Average Return per Trade: {summary['avg_return']:.2%}")
    print(f"  Total Return: {summary['total_return']:.2%}")
    print(f"  Average Holding Period: {summary['avg_holding_period_days']:.1f} days")
    print(f"  Best Trade: {summary['best_trade']:.2%}")
    print(f"  Worst Trade: {summary['worst_trade']:.2%}")
    
    # Show some example signals
    signals = strategy.signals[:5]  # First 5 signals
    if signals:
        print(f"\nExample Signals (first 5):")
        for i, signal in enumerate(signals, 1):
            print(f"  {i}. {signal.action.value} @ ${signal.price:,.2f} on {signal.timestamp.date()} "
                  f"(confidence: {signal.confidence.value}, strength: {float(signal.strength):.2f})")


if __name__ == "__main__":
    example_usage()
