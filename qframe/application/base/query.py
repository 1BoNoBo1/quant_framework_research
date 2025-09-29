"""
Base Query and Handler Classes
===============================

Base classes for CQRS query pattern implementation.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Type variables for generic query/response patterns
TQuery = TypeVar('TQuery')
TResult = TypeVar('TResult')


class QueryStatus(str, Enum):
    """Status of a query execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueryResult:
    """Result of a query execution"""

    query_id: str
    status: QueryStatus
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    @classmethod
    def success(cls, query_id: str, result: Any = None, execution_time_ms: float = None) -> 'QueryResult':
        """Create a successful query result"""
        return cls(
            query_id=query_id,
            status=QueryStatus.COMPLETED,
            result=result,
            execution_time_ms=execution_time_ms
        )

    @classmethod
    def failure(cls, query_id: str, error: str, execution_time_ms: float = None) -> 'QueryResult':
        """Create a failed query result"""
        return cls(
            query_id=query_id,
            status=QueryStatus.FAILED,
            error=error,
            execution_time_ms=execution_time_ms
        )

    def is_success(self) -> bool:
        """Check if the query was successful"""
        return self.status == QueryStatus.COMPLETED and self.error is None

    def is_failure(self) -> bool:
        """Check if the query failed"""
        return self.status == QueryStatus.FAILED or self.error is not None


@dataclass(kw_only=True)
class Query:
    """Base query class with common fields"""

    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = field(default=None)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def get_query_name(self) -> str:
        """Get the query name"""
        return self.__class__.__name__


class QueryHandler(ABC, Generic[TQuery]):
    """
    Base class for query handlers.

    Each query handler is responsible for executing a specific type of query.
    """

    @abstractmethod
    async def handle(self, query: TQuery) -> QueryResult:
        """
        Handle the given query.

        Args:
            query: The query to handle

        Returns:
            QueryResult: Result of the query execution
        """
        pass

    def get_query_type(self) -> type:
        """Get the type of query this handler processes"""
        # This would typically be determined by generic type parameters
        # For now, return None and let concrete handlers override
        return None


class BaseQueryHandler(QueryHandler):
    """
    Base implementation of query handler with common functionality.
    """

    def __init__(self):
        self.handler_name = self.__class__.__name__

    async def handle(self, query: TQuery) -> QueryResult:
        """
        Handle query with error handling and timing.
        """
        start_time = datetime.utcnow()

        try:
            # Validate query
            await self._validate(query)

            # Execute query
            result = await self._execute(query)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return QueryResult.success(
                query_id=query.query_id,
                result=result,
                execution_time_ms=execution_time
            )

        except Exception as e:
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return QueryResult.failure(
                query_id=query.query_id,
                error=str(e),
                execution_time_ms=execution_time
            )

    async def _validate(self, query: TQuery) -> None:
        """
        Validate the query before execution.

        Args:
            query: The query to validate

        Raises:
            ValueError: If the query is invalid
        """
        if not query:
            raise ValueError("Query cannot be None")

        if not hasattr(query, 'query_id') or not query.query_id:
            raise ValueError("Query must have a valid query_id")

    @abstractmethod
    async def _execute(self, query: TQuery) -> Any:
        """
        Execute the query logic.

        Args:
            query: The query to execute

        Returns:
            The result of the query execution
        """
        pass


# Export main classes
__all__ = [
    'Query',
    'QueryResult',
    'QueryStatus',
    'QueryHandler',
    'BaseQueryHandler',
    'TQuery',
    'TResult'
]