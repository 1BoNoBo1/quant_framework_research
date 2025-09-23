"""
Unit tests for Portfolio domain entity
"""

import pytest
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from qframe.domain.entities.portfolio import Portfolio, Position, PortfolioStatus
from qframe.domain.entities.order import Order, OrderSide, OrderType, OrderStatus


class TestPortfolio:
    """Test suite for Portfolio entity"""
    
    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio"""
        return Portfolio(
            id=uuid4(),
            name="Test Portfolio",
            base_currency="USDT",
            initial_capital=Decimal("10000")
        )
    
    def test_create_portfolio(self):
        """Test creating a new portfolio"""
        portfolio = Portfolio(
            id=uuid4(),
            name="My Portfolio",
            base_currency="USDT",
            initial_capital=Decimal("10000")
        )
        
        assert portfolio.name == "My Portfolio"
        assert portfolio.base_currency == "USDT"
        assert portfolio.initial_capital == Decimal("10000")
        assert portfolio.cash_balance == Decimal("10000")
        assert len(portfolio.positions) == 0
        
    def test_add_position(self, portfolio):
        """Test adding a position to portfolio"""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            average_price=Decimal("50000")
        )
        
        portfolio.add_position(position)
        
        assert len(portfolio.positions) == 1
        assert "BTC/USDT" in portfolio.positions
        assert portfolio.positions["BTC/USDT"].quantity == Decimal("0.5")
        
    def test_update_position_on_buy(self, portfolio):
        """Test updating position on buy order"""
        # Initial position
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            average_price=Decimal("50000")
        )
        portfolio.add_position(position)
        portfolio.cash_balance = Decimal("5000")
        
        # Buy more
        buy_order = Order(
            id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            price=Decimal("60000")
        )
        buy_order.status = OrderStatus.FILLED
        buy_order.filled_quantity = Decimal("0.5")
        buy_order.average_fill_price = Decimal("60000")
        
        portfolio.update_position(buy_order)
        
        # Check updated position
        position = portfolio.positions["BTC/USDT"]
        assert position.quantity == Decimal("1.5")
        # Average price should be weighted: (1*50000 + 0.5*60000) / 1.5 = 53333.33
        assert position.average_price == pytest.approx(Decimal("53333.33"), rel=0.01)
        
    def test_update_position_on_sell(self, portfolio):
        """Test updating position on sell order"""
        # Initial position
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("2"),
            average_price=Decimal("50000")
        )
        portfolio.add_position(position)
        
        # Sell some
        sell_order = Order(
            id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            price=Decimal("60000")
        )
        sell_order.status = OrderStatus.FILLED
        sell_order.filled_quantity = Decimal("0.5")
        sell_order.average_fill_price = Decimal("60000")
        
        portfolio.update_position(sell_order)
        
        # Check updated position
        position = portfolio.positions["BTC/USDT"]
        assert position.quantity == Decimal("1.5")
        assert position.average_price == Decimal("50000")  # Unchanged on sell
        # Realized PnL: 0.5 * (60000 - 50000) = 5000
        assert position.realized_pnl == Decimal("5000")
        
    def test_calculate_total_value(self, portfolio):
        """Test calculating portfolio total value"""
        # Add positions
        btc_position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            average_price=Decimal("50000")
        )
        btc_position.current_price = Decimal("60000")
        
        eth_position = Position(
            symbol="ETH/USDT", 
            quantity=Decimal("10"),
            average_price=Decimal("3000")
        )
        eth_position.current_price = Decimal("3500")
        
        portfolio.add_position(btc_position)
        portfolio.add_position(eth_position)
        portfolio.cash_balance = Decimal("5000")
        
        total_value = portfolio.calculate_total_value()
        
        # 0.5 * 60000 + 10 * 3500 + 5000 = 30000 + 35000 + 5000 = 70000
        assert total_value == Decimal("70000")
        
    def test_calculate_unrealized_pnl(self, portfolio):
        """Test calculating unrealized PnL"""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            average_price=Decimal("50000")
        )
        position.current_price = Decimal("60000")
        
        portfolio.add_position(position)
        
        unrealized_pnl = portfolio.calculate_unrealized_pnl()
        
        # 1 * (60000 - 50000) = 10000
        assert unrealized_pnl == Decimal("10000")
        
    def test_get_allocation_percentages(self, portfolio):
        """Test getting allocation percentages"""
        # Add positions
        btc_position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            average_price=Decimal("50000")
        )
        btc_position.current_price = Decimal("50000")
        
        portfolio.add_position(btc_position)
        portfolio.cash_balance = Decimal("25000")
        
        allocations = portfolio.get_allocation_percentages()
        
        # BTC: 25000, Cash: 25000, Total: 50000
        assert allocations["BTC/USDT"] == pytest.approx(0.5, rel=0.01)
        assert allocations["CASH"] == pytest.approx(0.5, rel=0.01)
        
    def test_is_position_within_risk_limits(self, portfolio):
        """Test checking if position is within risk limits"""
        portfolio.risk_limits = {
            "max_position_size": 0.3,  # 30% max per position
            "max_leverage": 1.0
        }
        
        # Add a position that's 25% of portfolio
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            average_price=Decimal("50000")
        )
        position.current_price = Decimal("50000")
        
        portfolio.add_position(position)
        portfolio.cash_balance = Decimal("75000")
        
        # Should be within limits (25% < 30%)
        assert portfolio.is_position_within_risk_limits("BTC/USDT", Decimal("0.1"))
        
        # Additional 0.5 BTC would make it 50% - should exceed limits
        assert not portfolio.is_position_within_risk_limits("BTC/USDT", Decimal("0.5"))
