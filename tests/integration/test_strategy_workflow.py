"""
Integration tests for complete strategy workflow
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from qframe.application.commands.strategy_commands import (
    CreateStrategyCommand,
    UpdateStrategyCommand,
    ActivateStrategyCommand
)
from qframe.domain.entities.strategy import StrategyType
from qframe.application.handlers.strategy_command_handler import StrategyCommandHandler
from qframe.application.queries.strategy_queries import GetStrategyByIdQuery
from qframe.application.handlers.strategy_query_handler import StrategyQueryHandler

from qframe.infrastructure.persistence.memory_strategy_repository import MemoryStrategyRepository
from qframe.infrastructure.config.service_configuration import ServiceConfiguration
from qframe.core.config import FrameworkConfig, Environment


@pytest.mark.asyncio
class TestStrategyWorkflow:
    """Test complete strategy workflow"""
    
    @pytest.fixture
    async def setup(self):
        """Setup test environment"""
        from qframe.core.container import DIContainer

        config = FrameworkConfig(environment=Environment.TESTING)
        container = DIContainer()
        service_config = ServiceConfiguration(container, config)
        service_config.configure_all_services()
        
        # Get handlers
        command_handler = container.resolve(StrategyCommandHandler)
        query_handler = container.resolve(StrategyQueryHandler)
        
        return command_handler, query_handler
    
    async def test_create_and_activate_strategy(self, setup):
        """Test creating and activating a strategy"""
        command_handler, query_handler = setup
        
        # Create strategy
        create_command = CreateStrategyCommand(
            name="Test MA Strategy",
            description="Moving average test strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            universe=["BTC/USDT"],
            max_position_size=Decimal("0.1"),
            max_positions=5,
            risk_per_trade=Decimal("0.02")
        )
        
        strategy_id = await command_handler.handle_create_strategy(create_command)
        assert strategy_id is not None

        # Query the created strategy
        query = GetStrategyByIdQuery(strategy_id=strategy_id)
        retrieved = await query_handler.handle_get_by_id(query)
        assert retrieved.id == strategy_id
        assert retrieved.name == "Test MA Strategy"
        assert retrieved.status.value == "inactive"

        # Activate the strategy
        activate_command = ActivateStrategyCommand(strategy_id=strategy_id)
        await command_handler.handle_activate_strategy(activate_command)

        # Verify it's active
        query = GetStrategyByIdQuery(strategy_id=strategy_id)
        retrieved = await query_handler.handle_get_by_id(query)
        assert retrieved.status.value == "active"
    
    async def test_update_strategy_parameters(self, setup):
        """Test updating strategy parameters"""
        command_handler, query_handler = setup
        
        # Create strategy
        create_command = CreateStrategyCommand(
            name="Update Test Strategy",
            description="Strategy for update testing",
            strategy_type=StrategyType.MEAN_REVERSION,
            universe=["BTC/USDT"],
            max_position_size=Decimal("0.1"),
            max_positions=3,
            risk_per_trade=Decimal("0.02")
        )
        
        strategy_id = await command_handler.handle_create_strategy(create_command)

        # Get original strategy to check version
        query = GetStrategyByIdQuery(strategy_id=strategy_id)
        original_strategy = await query_handler.handle_get_by_id(query)
        original_version = original_strategy.version

        # Update parameters
        update_command = UpdateStrategyCommand(
            strategy_id=strategy_id,
            max_position_size=Decimal("0.15"),
            max_positions=5,
            risk_per_trade=Decimal("0.03")
        )

        await command_handler.handle_update_strategy(update_command)

        # Query updated strategy
        query = GetStrategyByIdQuery(strategy_id=strategy_id)
        updated = await query_handler.handle_get_by_id(query)
        assert updated.max_position_size == Decimal("0.15")
        assert updated.max_positions == 5
        assert updated.risk_per_trade == Decimal("0.03")
        assert updated.version > original_version
    
    async def test_concurrent_strategy_operations(self, setup):
        """Test concurrent operations on multiple strategies"""
        command_handler, query_handler = setup
        
        # Create multiple strategies concurrently
        create_tasks = []
        for i in range(5):
            command = CreateStrategyCommand(
                name=f"Concurrent Strategy {i}",
                description=f"Test strategy {i}",
                strategy_type=StrategyType.MOMENTUM,
                universe=[f"ETH/USDT"],
                max_position_size=Decimal("0.05"),
                max_positions=2,
                risk_per_trade=Decimal("0.01")
            )
            create_tasks.append(command_handler.handle_create_strategy(command))

        strategy_ids = await asyncio.gather(*create_tasks)
        assert len(strategy_ids) == 5

        # Activate all strategies concurrently
        activate_tasks = []
        for strategy_id in strategy_ids:
            command = ActivateStrategyCommand(strategy_id=strategy_id)
            activate_tasks.append(command_handler.handle_activate_strategy(command))

        await asyncio.gather(*activate_tasks)

        # Verify all are active
        for strategy_id in strategy_ids:
            query = GetStrategyByIdQuery(strategy_id=strategy_id)
            strategy = await query_handler.handle_get_by_id(query)
            assert strategy.status.value == "active"
    
    async def test_strategy_error_handling(self, setup):
        """Test error handling in strategy operations"""
        command_handler, query_handler = setup
        
        # Try to activate non-existent strategy
        fake_id = str(uuid4())
        activate_command = ActivateStrategyCommand(strategy_id=fake_id)

        with pytest.raises(Exception) as exc_info:
            await command_handler.handle_activate_strategy(activate_command)
        assert "not found" in str(exc_info.value).lower()

        # Try to create strategy with duplicate name
        create_command = CreateStrategyCommand(
            name="Duplicate Strategy",
            description="First strategy",
            strategy_type=StrategyType.ARBITRAGE,
            universe=["ADA/USDT", "DOT/USDT"],  # Arbitrage requires at least 2 symbols
            max_position_size=Decimal("0.08"),
            max_positions=4,
            risk_per_trade=Decimal("0.015")
        )

        await command_handler.handle_create_strategy(create_command)

        # Try to create another with same name
        with pytest.raises(Exception) as exc_info:
            await command_handler.handle_create_strategy(create_command)
        assert "already exists" in str(exc_info.value).lower()
