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
from qframe.application.handlers.strategy_command_handler import StrategyCommandHandler
from qframe.application.queries.strategy_queries import GetStrategyByIdQuery
from qframe.application.handlers.strategy_query_handler import StrategyQueryHandler

from qframe.infrastructure.persistence.memory_strategy_repository import MemoryStrategyRepository
from qframe.infrastructure.config.service_configuration import ServiceConfiguration
from qframe.infrastructure.config.environment_config import ApplicationConfig, Environment


@pytest.mark.asyncio
class TestStrategyWorkflow:
    """Test complete strategy workflow"""
    
    @pytest.fixture
    async def setup(self):
        """Setup test environment"""
        config = ApplicationConfig(environment=Environment.TESTING)
        service_config = ServiceConfiguration(config)
        service_config.configure()
        
        container = service_config.container
        
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
            parameters={
                "fast_period": 10,
                "slow_period": 20,
                "symbol": "BTC/USDT"
            }
        )
        
        strategy = await command_handler.handle(create_command)
        assert strategy.name == "Test MA Strategy"
        assert strategy.status.value == "inactive"
        
        # Query the created strategy
        query = GetStrategyByIdQuery(strategy_id=strategy.id)
        retrieved = await query_handler.handle(query)
        assert retrieved.id == strategy.id
        
        # Activate the strategy
        activate_command = ActivateStrategyCommand(strategy_id=strategy.id)
        activated = await command_handler.handle(activate_command)
        assert activated.status.value == "active"
        
        # Verify it's active
        query = GetStrategyByIdQuery(strategy_id=strategy.id)
        retrieved = await query_handler.handle(query)
        assert retrieved.status.value == "active"
    
    async def test_update_strategy_parameters(self, setup):
        """Test updating strategy parameters"""
        command_handler, query_handler = setup
        
        # Create strategy
        create_command = CreateStrategyCommand(
            name="Update Test Strategy",
            description="Strategy for update testing",
            parameters={"param1": 100}
        )
        
        strategy = await command_handler.handle(create_command)
        original_version = strategy.version
        
        # Update parameters
        update_command = UpdateStrategyCommand(
            strategy_id=strategy.id,
            parameters={"param1": 200, "param2": 50}
        )
        
        updated = await command_handler.handle(update_command)
        assert updated.parameters["param1"] == 200
        assert updated.parameters["param2"] == 50
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
                parameters={"index": i}
            )
            create_tasks.append(command_handler.handle(command))
        
        strategies = await asyncio.gather(*create_tasks)
        assert len(strategies) == 5
        
        # Activate all strategies concurrently
        activate_tasks = []
        for strategy in strategies:
            command = ActivateStrategyCommand(strategy_id=strategy.id)
            activate_tasks.append(command_handler.handle(command))
        
        activated = await asyncio.gather(*activate_tasks)
        assert all(s.status.value == "active" for s in activated)
    
    async def test_strategy_error_handling(self, setup):
        """Test error handling in strategy operations"""
        command_handler, query_handler = setup
        
        # Try to activate non-existent strategy
        fake_id = uuid4()
        activate_command = ActivateStrategyCommand(strategy_id=fake_id)
        
        with pytest.raises(Exception) as exc_info:
            await command_handler.handle(activate_command)
        assert "not found" in str(exc_info.value).lower()
        
        # Try to create strategy with duplicate name
        create_command = CreateStrategyCommand(
            name="Duplicate Strategy",
            description="First strategy",
            parameters={}
        )
        
        await command_handler.handle(create_command)
        
        # Try to create another with same name
        with pytest.raises(Exception) as exc_info:
            await command_handler.handle(create_command)
        assert "already exists" in str(exc_info.value).lower()
