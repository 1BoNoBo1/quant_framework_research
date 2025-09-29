"""
Base Command and Handler Classes
=================================

Base classes for CQRS command pattern implementation.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Type variables for generic command/response patterns
TCommand = TypeVar('TCommand')
TResult = TypeVar('TResult')


class CommandStatus(str, Enum):
    """Status of a command execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CommandResult:
    """Result of a command execution"""

    success: bool
    result_data: Optional[dict] = None
    message: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    @classmethod
    def success_result(cls, result_data: dict = None, message: str = None, execution_time_ms: float = None) -> 'CommandResult':
        """Create a successful command result"""
        return cls(
            success=True,
            result_data=result_data,
            message=message,
            execution_time_ms=execution_time_ms
        )

    @classmethod
    def failure_result(cls, error_message: str, execution_time_ms: float = None) -> 'CommandResult':
        """Create a failed command result"""
        return cls(
            success=False,
            error_message=error_message,
            execution_time_ms=execution_time_ms
        )


class Command:
    """
    Base class for all commands in the system.

    Commands represent actions to be performed.
    They are immutable and contain all necessary data for execution.
    """

    def __init__(self):
        self.command_id: str = str(uuid.uuid4())
        self.timestamp: datetime = datetime.utcnow()
        self.correlation_id: Optional[str] = None
        self.metadata: dict = {}

    @property
    def name(self) -> str:
        """Get the command name"""
        return self.__class__.__name__


class CommandHandler(ABC, Generic[TCommand]):
    """
    Base class for command handlers.

    Each command handler is responsible for executing a specific type of command.
    """

    @abstractmethod
    async def handle(self, command: TCommand) -> CommandResult:
        """
        Handle the given command.

        Args:
            command: The command to handle

        Returns:
            CommandResult: Result of the command execution
        """
        pass

    def get_command_type(self) -> type:
        """Get the type of command this handler processes"""
        # This would typically be determined by generic type parameters
        # For now, return None and let concrete handlers override
        return None


class BaseCommandHandler(CommandHandler):
    """
    Base implementation of command handler with common functionality.
    """

    def __init__(self):
        self.handler_name = self.__class__.__name__

    async def handle(self, command: TCommand) -> CommandResult:
        """
        Handle command with error handling and timing.
        """
        start_time = datetime.utcnow()

        try:
            # Validate command
            await self._validate(command)

            # Execute command
            result = await self._execute(command)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return CommandResult.success_result(
                result_data=result if isinstance(result, dict) else {"result": result},
                message="Command executed successfully",
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return CommandResult.failure_result(
                error_message=str(e),
                execution_time_ms=execution_time
            )

    async def _validate(self, command: TCommand) -> None:
        """
        Validate the command before execution.

        Args:
            command: The command to validate

        Raises:
            ValueError: If the command is invalid
        """
        # Default implementation - do nothing
        # Concrete handlers can override for validation
        pass

    @abstractmethod
    async def _execute(self, command: TCommand) -> TResult:
        """
        Execute the command.

        Args:
            command: The command to execute

        Returns:
            The result of command execution
        """
        pass