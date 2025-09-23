"""
Infrastructure Layer: Event-Driven Architecture
===============================================

Architecture événementielle pour découpler les composants et gérer
les workflows complexes via des événements.
"""

from .core import (
    Event,
    EventMetadata,
    DomainEvent,
    SystemEvent,
    EventHandler,
    EventHandlerRegistry,
    EventBus
)

from .event_store import (
    EventStore,
    EventStream,
    EventSnapshot,
    InMemoryEventStore,
    EventStoreStatistics
)

from .saga import (
    SagaStep,
    SagaStepResult,
    SagaState,
    Saga,
    SagaManager,
    CompensationAction
)

from .projections import (
    Projection,
    ProjectionManager,
    ProjectionState,
    ReadModelUpdater
)

__all__ = [
    # Core event system
    'Event',
    'EventMetadata',
    'DomainEvent',
    'SystemEvent',
    'EventHandler',
    'EventHandlerRegistry',
    'EventBus',

    # Event store
    'EventStore',
    'EventStream',
    'EventSnapshot',
    'InMemoryEventStore',
    'EventStoreStatistics',

    # Saga system
    'SagaStep',
    'SagaStepResult',
    'SagaState',
    'Saga',
    'SagaManager',
    'CompensationAction',

    # Projections
    'Projection',
    'ProjectionManager',
    'ProjectionState',
    'ReadModelUpdater'
]