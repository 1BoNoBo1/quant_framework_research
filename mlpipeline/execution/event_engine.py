#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-Driven Execution Engine - Moteur événementiel unifié
UN SEUL chemin d'exécution pour backtest ET walk-forward
Boucle: Événements → Ordres → Fills → PnL
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled" 
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Ordre de trading"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_quantity: float = 0.0
    commission: float = 0.0

@dataclass
class Fill:
    """Exécution d'ordre"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    
@dataclass
class Position:
    """Position actuelle"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-8

@dataclass
class Portfolio:
    """État du portefeuille"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_commission: float = 0.0
    
    @property
    def total_value(self) -> float:
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        return sum(pos.realized_pnl + pos.unrealized_pnl for pos in self.positions.values())

class EventEngine:
    """
    MOTEUR ÉVÉNEMENTIEL UNIFIÉ
    
    Utilisé par TOUS les backtests (simple ET walk-forward)
    Garantit la cohérence : même boucle événements → ordres → fills → PnL
    """
    
    def __init__(self, 
                 initial_cash: float = 100000,
                 commission_rate: float = 0.002,  # 0.2%
                 slippage: float = 0.0001):       # 0.01%
        
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # État du moteur
        self.portfolio = Portfolio(cash=initial_cash)
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_counter = 0
        self.fill_counter = 0
        
        # Historique pour analyse
        self.equity_curve = []
        self.trade_history = []
        self.current_timestamp: Optional[datetime] = None
        self.current_prices: Dict[str, float] = {}
        
        logger.info(f"✅ EventEngine initialisé: ${initial_cash:,.0f} capital")
    
    def reset(self):
        """Reset complet du moteur pour nouveau backtest"""
        self.portfolio = Portfolio(cash=self.initial_cash)
        self.orders.clear()
        self.fills.clear()
        self.order_counter = 0
        self.fill_counter = 0
        self.equity_curve.clear()
        self.trade_history.clear()
        self.current_timestamp = None
        self.current_prices.clear()
        
        logger.debug("🔄 EventEngine reset")
    
    def on_market_data(self, timestamp: datetime, market_data: Dict[str, float]):
        """
        ÉVÉNEMENT MARCHÉ - Point d'entrée principal
        
        Args:
            timestamp: Timestamp du tick
            market_data: {"BTCUSDT": 45000.0, "ETHUSDT": 3000.0}
        """
        self.current_timestamp = timestamp
        self.current_prices.update(market_data)
        
        # 1. Mise à jour positions (unrealized PnL)
        self._update_positions()
        
        # 2. Traitement ordres pending
        self._process_pending_orders()
        
        # 3. Enregistrement equity
        self._record_equity_point()
    
    def submit_order(self, 
                    symbol: str,
                    side: OrderSide, 
                    quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None) -> str:
        """
        Soumission d'ordre - Interface unifiée
        
        Returns:
            order_id: ID de l'ordre créé
        """
        order_id = f"order_{self.order_counter}"
        self.order_counter += 1
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=self.current_timestamp
        )
        
        self.orders[order_id] = order
        
        logger.debug(f"📝 Ordre soumis: {order_id} {side.value} {quantity} {symbol}")
        return order_id
    
    def _process_pending_orders(self):
        """Traitement des ordres en attente"""
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                self._try_fill_order(order)
    
    def _try_fill_order(self, order: Order) -> bool:
        """
        Tentative d'exécution d'un ordre
        
        Returns:
            bool: True si ordre exécuté
        """
        if order.symbol not in self.current_prices:
            logger.warning(f"⚠️ Prix manquant pour {order.symbol}")
            return False
        
        current_price = self.current_prices[order.symbol]
        fill_price = self._calculate_fill_price(order, current_price)
        
        # Vérification fonds disponibles
        if not self._check_funds_available(order, fill_price):
            order.status = OrderStatus.REJECTED
            logger.warning(f"❌ Ordre rejeté (fonds insuffisants): {order.order_id}")
            return False
        
        # Exécution
        commission = abs(order.quantity * fill_price * self.commission_rate)
        
        fill = Fill(
            fill_id=f"fill_{self.fill_counter}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=self.current_timestamp,
            commission=commission
        )
        
        self.fill_counter += 1
        self.fills.append(fill)
        
        # Mise à jour ordre
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_quantity = order.quantity
        order.commission = commission
        
        # Mise à jour portfolio
        self._update_portfolio_from_fill(fill)
        
        logger.debug(f"✅ Ordre exécuté: {order.order_id} @ {fill_price:.2f}")
        return True
    
    def _calculate_fill_price(self, order: Order, current_price: float) -> float:
        """Calcul prix d'exécution avec slippage"""
        if order.order_type == OrderType.MARKET:
            # Market order: prix actuel + slippage
            slippage_adjustment = (1 + self.slippage if order.side == OrderSide.BUY 
                                 else 1 - self.slippage)
            return current_price * slippage_adjustment
        
        elif order.order_type == OrderType.LIMIT:
            # Limit order: prix limite si possible
            if order.side == OrderSide.BUY and current_price <= order.price:
                return order.price
            elif order.side == OrderSide.SELL and current_price >= order.price:
                return order.price
            else:
                # Limite non atteinte
                return None
        
        return current_price
    
    def _check_funds_available(self, order: Order, fill_price: float) -> bool:
        """Vérification fonds disponibles"""
        if order.side == OrderSide.BUY:
            required_cash = order.quantity * fill_price * (1 + self.commission_rate)
            return self.portfolio.cash >= required_cash
        else:  # SELL
            if order.symbol in self.portfolio.positions:
                available_qty = self.portfolio.positions[order.symbol].quantity
                return available_qty >= order.quantity
            return False
    
    def _update_portfolio_from_fill(self, fill: Fill):
        """Mise à jour portfolio après exécution"""
        symbol = fill.symbol
        
        # Création position si nécessaire
        if symbol not in self.portfolio.positions:
            self.portfolio.positions[symbol] = Position(symbol=symbol)
        
        position = self.portfolio.positions[symbol]
        
        # Calcul nouvelle position
        if fill.side == OrderSide.BUY:
            # Achat
            new_quantity = position.quantity + fill.quantity
            if new_quantity != 0:
                # Nouveau prix moyen pondéré
                total_cost = (position.quantity * position.avg_price + 
                            fill.quantity * fill.price)
                position.avg_price = total_cost / new_quantity
            
            position.quantity = new_quantity
            self.portfolio.cash -= fill.quantity * fill.price
        
        else:  # SELL
            # Vente - Calcul PnL réalisé
            if position.quantity > 0:  # Position longue
                realized_pnl = fill.quantity * (fill.price - position.avg_price)
                position.realized_pnl += realized_pnl
            
            position.quantity -= fill.quantity
            self.portfolio.cash += fill.quantity * fill.price
        
        # Commission
        self.portfolio.cash -= fill.commission
        self.portfolio.total_commission += fill.commission
        
        # Trade history
        self.trade_history.append({
            'timestamp': fill.timestamp,
            'symbol': symbol,
            'side': fill.side.value,
            'quantity': fill.quantity,
            'price': fill.price,
            'commission': fill.commission,
            'pnl': getattr(fill, 'realized_pnl', 0.0)
        })
    
    def _update_positions(self):
        """Mise à jour unrealized PnL"""
        for symbol, position in self.portfolio.positions.items():
            if not position.is_flat and symbol in self.current_prices:
                current_price = self.current_prices[symbol]
                position.unrealized_pnl = position.quantity * (current_price - position.avg_price)
    
    def _record_equity_point(self):
        """Enregistrement point d'équité"""
        equity_point = {
            'timestamp': self.current_timestamp,
            'cash': self.portfolio.cash,
            'total_value': self.portfolio.total_value,
            'total_pnl': self.portfolio.total_pnl,
            'total_commission': self.portfolio.total_commission
        }
        self.equity_curve.append(equity_point)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        MÉTRIQUES UNIFIÉES - Utilisées par walk-forward ET backtest simple
        """
        if not self.equity_curve:
            return {"error": "no_data"}
        
        # Conversion en DataFrame pour calculs
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Returns
        returns = equity_df['total_value'].pct_change().fillna(0)
        
        # Métriques principales
        total_return = (equity_df['total_value'].iloc[-1] / self.initial_cash) - 1
        sharpe_ratio = self._calculate_sharpe(returns)
        max_drawdown = self._calculate_max_drawdown(equity_df['total_value'])
        
        # Statistiques trades
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': int(total_trades),
            'win_rate': float(win_rate),
            'total_commission': float(self.portfolio.total_commission),
            'final_value': float(equity_df['total_value'].iloc[-1]),
            'returns_std': float(returns.std()),
            'returns_mean': float(returns.mean())
        }
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calcul Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calcul drawdown maximum"""
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min())
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """Export equity curve pour analyse"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Export trades pour analyse"""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)


class StrategyRunner:
    """
    Runner de stratégie utilisant le moteur événementiel
    Interface standardisée pour toutes les stratégies
    """
    
    def __init__(self, event_engine: EventEngine):
        self.engine = event_engine
        self.strategy_params = {}
        self.current_signals = {}
    
    def run_strategy(self, 
                    market_data_df: pd.DataFrame,
                    strategy_func: callable,
                    strategy_params: Dict[str, Any]) -> Dict[str, float]:
        """
        INTERFACE UNIFIÉE - Utilisée par walk-forward ET backtest
        
        Args:
            market_data_df: DataFrame avec colonnes [timestamp, symbol, price, ...]  
            strategy_func: Fonction génératrice de signaux
            strategy_params: Paramètres de la stratégie (frozen pour walk-forward)
        
        Returns:
            Dict: Métriques de performance standardisées
        """
        logger.info(f"🚀 Démarrage stratégie avec EventEngine")
        
        # Reset moteur pour nouveau run
        self.engine.reset()
        self.strategy_params = strategy_params
        
        # Génération signaux avec paramètres frozen
        signals_df = strategy_func(market_data_df, **strategy_params)
        
        # Simulation événementielle
        for idx, row in market_data_df.iterrows():
            timestamp = row['timestamp'] if 'timestamp' in row else idx
            
            # Prix du marché
            market_prices = {}
            if 'symbol' in row and 'close' in row:
                market_prices[row['symbol']] = row['close']
            else:
                # Format multi-colonnes (BTCUSDT_close, ETHUSDT_close, etc.)
                for col in market_data_df.columns:
                    if col.endswith('_close'):
                        symbol = col.replace('_close', '')
                        market_prices[symbol] = row[col]
            
            # Événement marché
            self.engine.on_market_data(timestamp, market_prices)
            
            # Signal de stratégie à ce timestamp
            if timestamp in signals_df.index:
                signal_row = signals_df.loc[timestamp]
                self._process_strategy_signal(signal_row, market_prices)
        
        # Retour métriques unifiées
        return self.engine.get_performance_metrics()
    
    def _process_strategy_signal(self, signal_row, market_prices: Dict[str, float]):
        """Traitement signal de stratégie"""
        for symbol in market_prices.keys():
            signal_col = f'{symbol}_signal'
            position_col = f'{symbol}_position_size'
            
            # Signal brut
            if signal_col in signal_row:
                signal = signal_row[signal_col]
            elif 'signal' in signal_row:
                signal = signal_row['signal']  
            else:
                continue
            
            # Taille de position
            if position_col in signal_row:
                position_size = signal_row[position_col]
            elif 'position_size' in signal_row:
                position_size = signal_row['position_size']
            else:
                position_size = 1000 if signal != 0 else 0  # Default size
            
            # Génération ordre si signal non nul
            if abs(signal) > 0:
                side = OrderSide.BUY if signal > 0 else OrderSide.SELL
                quantity = abs(position_size)
                
                if quantity > 0:
                    self.engine.submit_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        order_type=OrderType.MARKET
                    )


# Factory function pour création moteur standardisé
def create_event_engine(config: Dict[str, Any]) -> EventEngine:
    """
    Factory pour créer moteur événementiel avec config standard
    """
    return EventEngine(
        initial_cash=config.get('initial_capital', 100000),
        commission_rate=config.get('commission_rate', 0.002),
        slippage=config.get('slippage', 0.0001)
    )