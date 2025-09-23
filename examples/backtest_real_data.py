"""
Backtest with Real Historical Data
===================================

Complete example of backtesting MA Crossover strategy
using real historical data from Binance.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
import ccxt

from examples.strategies.ma_crossover_strategy import MovingAverageCrossoverStrategy


async def fetch_historical_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1d',
    days_back: int = 365
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance
    
    Parameters:
    -----------
    symbol : str
        Trading pair symbol
    timeframe : str
        Candlestick timeframe ('1m', '5m', '1h', '1d', etc.)
    days_back : int
        Number of days of historical data to fetch
        
    Returns:
    --------
    pd.DataFrame
        Historical price data
    """
    exchange = ccxt.binance()
    
    # Calculate start time
    since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    
    # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df


def calculate_portfolio_performance(
    trades: list,
    initial_capital: float = 10000.0
) -> dict:
    """
    Calculate portfolio performance metrics from trades
    
    Parameters:
    -----------
    trades : list
        List of trade dictionaries
    initial_capital : float
        Initial portfolio capital
        
    Returns:
    --------
    dict
        Performance metrics
    """
    if not trades:
        return {
            'final_value': initial_capital,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0
        }
    
    # Calculate portfolio value over time
    portfolio_values = [initial_capital]
    
    for trade in trades:
        current_value = portfolio_values[-1]
        trade_return = trade['pnl_pct']
        new_value = current_value * (1 + trade_return)
        portfolio_values.append(new_value)
    
    # Convert to pandas series for easier calculation
    values = pd.Series(portfolio_values)
    returns = values.pct_change().dropna()
    
    # Calculate metrics
    final_value = values.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # Sharpe Ratio (annualized, assuming 252 trading days)
    sharpe_ratio = 0
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5)
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar_ratio = 0
    if max_drawdown < 0:
        calmar_ratio = total_return / abs(max_drawdown)
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'portfolio_values': portfolio_values
    }


async def run_backtest():
    """Run complete backtest with real data"""
    
    print("=" * 70)
    print("Moving Average Crossover Strategy - Real Data Backtest")
    print("=" * 70)
    
    # Fetch historical data
    print("\n📊 Fetching historical data from Binance...")
    symbol = 'BTC/USDT'
    timeframe = '1d'
    days_back = 730  # 2 years
    
    price_data = await fetch_historical_data(symbol, timeframe, days_back)
    print(f"✅ Fetched {len(price_data)} days of data")
    print(f"   Period: {price_data.index[0].date()} to {price_data.index[-1].date()}")
    
    # Initialize strategy
    print("\n⚙️  Initializing strategy...")
    strategy = MovingAverageCrossoverStrategy(
        fast_period=10,
        slow_period=20,
        symbol=symbol,
        position_size=0.95  # Use 95% of portfolio per trade
    )
    
    # Run backtest
    print("🔄 Running backtest...")
    summary = strategy.backtest_summary(price_data)
    
    # Calculate portfolio performance
    initial_capital = 10000.0
    performance = calculate_portfolio_performance(
        summary['trades'],
        initial_capital
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    print("\n📈 Strategy Parameters:")
    print(f"   Fast MA Period: {strategy.fast_period} days")
    print(f"   Slow MA Period: {strategy.slow_period} days")
    print(f"   Symbol: {symbol}")
    print(f"   Position Size: {strategy.position_size * 100}%")
    
    print("\n💰 Portfolio Performance:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Value: ${performance['final_value']:,.2f}")
    print(f"   Total Return: {performance['total_return']:.2%}")
    print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {performance['max_drawdown']:.2%}")
    print(f"   Calmar Ratio: {performance['calmar_ratio']:.2f}")
    
    print("\n📊 Trading Statistics:")
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Winning Trades: {summary['winning_trades']}")
    print(f"   Losing Trades: {summary['losing_trades']}")
    print(f"   Win Rate: {summary['win_rate']:.2%}")
    print(f"   Average Return per Trade: {summary['avg_return']:.2%}")
    print(f"   Average Holding Period: {summary['avg_holding_period_days']:.1f} days")
    print(f"   Best Trade: {summary['best_trade']:.2%}")
    print(f"   Worst Trade: {summary['worst_trade']:.2%}")
    
    # Show recent trades
    if summary['trades']:
        print("\n📋 Recent Trades (last 5):")
        for i, trade in enumerate(summary['trades'][-5:], 1):
            entry = trade['entry_price']
            exit_price = trade['exit_price']
            pnl = trade['pnl_pct']
            sign = "📈" if pnl > 0 else "📉"
            print(f"   {sign} Trade {len(summary['trades'])-5+i}: "
                  f"Entry ${entry:,.2f} → Exit ${exit_price:,.2f} "
                  f"({pnl:+.2%})")
    
    # Buy & Hold comparison
    buy_hold_return = (price_data['close'].iloc[-1] - price_data['close'].iloc[0]) / price_data['close'].iloc[0]
    print(f"\n🔍 Comparison:")
    print(f"   Strategy Return: {performance['total_return']:.2%}")
    print(f"   Buy & Hold Return: {buy_hold_return:.2%}")
    
    if performance['total_return'] > buy_hold_return:
        outperformance = performance['total_return'] - buy_hold_return
        print(f"   ✅ Strategy outperformed by {outperformance:.2%}")
    else:
        underperformance = buy_hold_return - performance['total_return']
        print(f"   ❌ Strategy underperformed by {underperformance:.2%}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(run_backtest())
