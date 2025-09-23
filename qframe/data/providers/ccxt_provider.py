"""
CCXT Data Provider
==================

Data provider implementation using CCXT for cryptocurrency market data.
Integrated with QFrame's dependency injection system.
"""

from typing import Dict, Optional, List, Any
import pandas as pd
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
import asyncio
from loguru import logger

from qframe.core.interfaces import DataProvider
from qframe.core.container import injectable
from qframe.core.models import TimeFrame


@injectable
class CCXTProvider(DataProvider):
    """CCXT-based data provider for cryptocurrency markets."""
    
    def __init__(self, exchange_id: str = 'binance', config: Optional[Dict[str, Any]] = None):
        """Initialize CCXT provider.
        
        Args:
            exchange_id: Exchange identifier (binance, kraken, coinbase, etc.)
            config: Optional exchange configuration
        """
        self.exchange_id = exchange_id
        self.config = config or {'enableRateLimit': True, 'timeout': 10000}
        self.exchange: Optional[ccxt.Exchange] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize exchange connection."""
        if self._initialized:
            return
            
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class(self.config)
            await self.exchange.load_markets()
            self._initialized = True
            logger.info(f"Initialized CCXT provider for {self.exchange_id}")
        except Exception as e:
            logger.error(f"Failed to initialize CCXT provider: {e}")
            raise
            
    async def close(self) -> None:
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self._initialized = False
            
    def _timeframe_to_ccxt(self, timeframe: TimeFrame) -> str:
        """Convert TimeFrame enum to CCXT timeframe string."""
        mapping = {
            TimeFrame.M1: '1m',
            TimeFrame.M5: '5m',
            TimeFrame.M15: '15m',
            TimeFrame.M30: '30m',
            TimeFrame.H1: '1h',
            TimeFrame.H4: '4h',
            TimeFrame.D1: '1d',
            TimeFrame.W1: '1w',
        }
        return mapping.get(timeframe, '1h')
        
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: TimeFrame enum
            start_date: Start date for data
            end_date: End date for data  
            limit: Maximum number of candles
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self._initialized:
            await self.initialize()
            
        # Convert timeframe
        ccxt_timeframe = self._timeframe_to_ccxt(timeframe)
        
        # Default to last 100 candles if no dates specified
        if start_date is None and limit is None:
            limit = 100
            
        # Convert dates to milliseconds
        since_ms = None
        if start_date:
            since_ms = int(start_date.timestamp() * 1000)
            
        # Fetch data
        try:
            all_candles = []
            
            while True:
                candles = await self.exchange.fetch_ohlcv(
                    symbol,
                    ccxt_timeframe,
                    since=since_ms,
                    limit=min(500, limit) if limit else 500
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                
                # Check if we have enough data
                if limit and len(all_candles) >= limit:
                    all_candles = all_candles[:limit]
                    break
                    
                # Check if we've reached end_date
                if end_date:
                    last_timestamp = datetime.fromtimestamp(candles[-1][0] / 1000)
                    if last_timestamp >= end_date:
                        # Filter out candles after end_date
                        all_candles = [
                            c for c in all_candles
                            if datetime.fromtimestamp(c[0] / 1000) <= end_date
                        ]
                        break
                        
                # Update since for next iteration
                since_ms = candles[-1][0] + 1
                
                # If we got less than limit, we've reached the end
                if len(candles) < 500:
                    break
                    
                # Rate limiting
                await asyncio.sleep(self.exchange.rateLimit / 1000)
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
            
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
        return df
        
    async def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        """Fetch current ticker data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with ticker data (bid, ask, last, etc.)
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'last': ticker.get('last'),
                'volume': ticker.get('baseVolume'),
                'timestamp': ticker.get('timestamp')
            }
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            raise
            
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch order book data.
        
        Args:
            symbol: Trading symbol
            limit: Depth of order book
            
        Returns:
            Dictionary with bids and asks
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            order_book = await self.exchange.fetch_order_book(symbol, limit)
            return {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': order_book.get('timestamp')
            }
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            raise
            
    async def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols.
        
        Returns:
            List of symbol strings
        """
        if not self._initialized:
            await self.initialize()
            
        return list(self.exchange.markets.keys())