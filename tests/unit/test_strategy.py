"""
Unit tests for Strategy domain entity
"""

import pytest
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from qframe.domain.entities.strategy import Strategy, StrategyStatus
from qframe.domain.value_objects.signal import Signal, SignalAction


class TestStrategy:
    """Test suite for Strategy entity"""
    
    def test_create_strategy(self):
        """Test creating a new strategy"""
        strategy = Strategy(
            id=uuid4(),
            name="MA Crossover",
            description="Moving average crossover strategy",
            parameters={"fast_period": 10, "slow_period": 20}
        )
        
        assert strategy.name == "MA Crossover"
        assert strategy.status == StrategyStatus.INACTIVE
        assert strategy.parameters["fast_period"] == 10
        
    def test_activate_strategy(self):
        """Test activating a strategy"""
        strategy = Strategy(
            id=uuid4(),
            name="Test Strategy",
            description="Test",
            parameters={}
        )
        
        strategy.activate()
        assert strategy.status == StrategyStatus.ACTIVE
        assert strategy.is_active()
        
    def test_deactivate_strategy(self):
        """Test deactivating a strategy"""
        strategy = Strategy(
            id=uuid4(),
            name="Test Strategy",
            description="Test",
            parameters={}
        )
        
        strategy.activate()
        strategy.deactivate()
        
        assert strategy.status == StrategyStatus.INACTIVE
        assert not strategy.is_active()
        
    def test_update_parameters(self):
        """Test updating strategy parameters"""
        strategy = Strategy(
            id=uuid4(),
            name="Test Strategy",
            description="Test",
            parameters={"param1": 100}
        )
        
        strategy.update_parameters({"param1": 200, "param2": 50})
        
        assert strategy.parameters["param1"] == 200
        assert strategy.parameters["param2"] == 50
        assert strategy.version == 2  # Version should increment
        
    def test_record_performance(self):
        """Test recording strategy performance"""
        strategy = Strategy(
            id=uuid4(),
            name="Test Strategy",
            description="Test",
            parameters={}
        )
        
        strategy.record_performance(0.15, 0.05, 0.75, 100)
        
        assert strategy.performance_metrics["total_return"] == 0.15
        assert strategy.performance_metrics["max_drawdown"] == 0.05
        assert strategy.performance_metrics["win_rate"] == 0.75
        assert strategy.performance_metrics["total_trades"] == 100
        
    def test_strategy_equality(self):
        """Test strategy equality based on ID"""
        id1 = uuid4()
        strategy1 = Strategy(id=id1, name="S1", description="", parameters={})
        strategy2 = Strategy(id=id1, name="S2", description="", parameters={})
        strategy3 = Strategy(id=uuid4(), name="S1", description="", parameters={})
        
        assert strategy1 == strategy2  # Same ID
        assert strategy1 != strategy3  # Different ID
        
    @pytest.mark.parametrize("status,expected", [
        (StrategyStatus.ACTIVE, True),
        (StrategyStatus.INACTIVE, False),
        (StrategyStatus.PAUSED, False),
        (StrategyStatus.ERROR, False),
    ])
    def test_is_active_for_different_statuses(self, status, expected):
        """Test is_active method for different statuses"""
        strategy = Strategy(
            id=uuid4(),
            name="Test",
            description="",
            parameters={}
        )
        strategy.status = status
        assert strategy.is_active() == expected
