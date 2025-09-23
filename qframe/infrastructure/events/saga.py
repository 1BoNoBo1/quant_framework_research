"""
Infrastructure Layer: Saga Pattern
==================================

Implémentation du pattern Saga pour orchestrer des transactions distribuées
avec gestion des compensations et des échecs.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Type, Union
import traceback

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace
from .core import Event, DomainEvent, EventType, get_event_bus


class SagaStepResult(str, Enum):
    """Résultat d'une étape de saga"""
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    COMPENSATE = "compensate"


class SagaState(str, Enum):
    """État d'une saga"""
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    ABORTED = "aborted"


@dataclass
class CompensationAction:
    """Action de compensation pour une étape"""
    action_name: str
    action_data: Dict[str, Any]
    timeout_seconds: int = 30
    retry_count: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompensationAction':
        return cls(**data)


@dataclass
class SagaStep:
    """Étape d'une saga"""
    step_id: str
    step_name: str
    step_data: Dict[str, Any]
    compensation_action: Optional[CompensationAction] = None
    timeout_seconds: int = 60
    retry_count: int = 3

    # État d'exécution
    status: SagaStepResult = SagaStepResult.SUCCESS
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.executed_at:
            data["executed_at"] = self.executed_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SagaStep':
        if "executed_at" in data and data["executed_at"]:
            data["executed_at"] = datetime.fromisoformat(data["executed_at"])
        if "completed_at" in data and data["completed_at"]:
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        if "compensation_action" in data and data["compensation_action"]:
            data["compensation_action"] = CompensationAction.from_dict(data["compensation_action"])
        return cls(**data)


class SagaStepHandler(ABC):
    """Interface pour les handlers d'étapes de saga"""

    @abstractmethod
    async def execute(self, step: SagaStep, saga_context: Dict[str, Any]) -> SagaStepResult:
        """Exécuter l'étape"""
        pass

    @abstractmethod
    async def compensate(self, step: SagaStep, saga_context: Dict[str, Any]) -> bool:
        """Compenser l'étape en cas d'échec"""
        pass

    @property
    @abstractmethod
    def step_name(self) -> str:
        """Nom de l'étape"""
        pass


@dataclass
class SagaDefinition:
    """Définition d'une saga"""
    saga_type: str
    steps: List[SagaStep]
    timeout_seconds: int = 300
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    compensation_strategy: str = "reverse_order"  # "reverse_order" ou "parallel"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "saga_type": self.saga_type,
            "steps": [step.to_dict() for step in self.steps],
            "timeout_seconds": self.timeout_seconds,
            "retry_policy": self.retry_policy,
            "compensation_strategy": self.compensation_strategy
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SagaDefinition':
        steps = [SagaStep.from_dict(step_data) for step_data in data["steps"]]
        return cls(
            saga_type=data["saga_type"],
            steps=steps,
            timeout_seconds=data.get("timeout_seconds", 300),
            retry_policy=data.get("retry_policy", {}),
            compensation_strategy=data.get("compensation_strategy", "reverse_order")
        )


@dataclass
class SagaInstance:
    """Instance d'exécution d'une saga"""
    saga_id: str
    saga_type: str
    definition: SagaDefinition
    state: SagaState = SagaState.STARTED
    current_step_index: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    # Métadonnées d'exécution
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Corrélation
    correlation_id: Optional[str] = None
    triggered_by: Optional[str] = None

    @property
    def current_step(self) -> Optional[SagaStep]:
        """Obtenir l'étape courante"""
        if 0 <= self.current_step_index < len(self.definition.steps):
            return self.definition.steps[self.current_step_index]
        return None

    @property
    def completed_steps(self) -> List[SagaStep]:
        """Obtenir les étapes complétées"""
        return [
            step for step in self.definition.steps
            if step.status == SagaStepResult.SUCCESS
        ]

    @property
    def failed_steps(self) -> List[SagaStep]:
        """Obtenir les étapes échouées"""
        return [
            step for step in self.definition.steps
            if step.status == SagaStepResult.FAILURE
        ]

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "saga_id": self.saga_id,
            "saga_type": self.saga_type,
            "definition": self.definition.to_dict(),
            "state": self.state.value,
            "current_step_index": self.current_step_index,
            "context": self.context,
            "started_at": self.started_at.isoformat(),
            "correlation_id": self.correlation_id,
            "triggered_by": self.triggered_by,
            "error_message": self.error_message
        }

        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SagaInstance':
        definition = SagaDefinition.from_dict(data["definition"])

        instance = cls(
            saga_id=data["saga_id"],
            saga_type=data["saga_type"],
            definition=definition,
            state=SagaState(data["state"]),
            current_step_index=data["current_step_index"],
            context=data["context"],
            started_at=datetime.fromisoformat(data["started_at"]),
            correlation_id=data.get("correlation_id"),
            triggered_by=data.get("triggered_by"),
            error_message=data.get("error_message")
        )

        if "completed_at" in data and data["completed_at"]:
            instance.completed_at = datetime.fromisoformat(data["completed_at"])

        return instance


class Saga:
    """
    Orchestrateur de saga pour gérer l'exécution des étapes
    et les compensations en cas d'échec.
    """

    def __init__(self, instance: SagaInstance):
        self.instance = instance
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.tracer = get_tracer()
        self.event_bus = get_event_bus()

        # Handlers des étapes
        self._step_handlers: Dict[str, SagaStepHandler] = {}

    def register_step_handler(self, step_name: str, handler: SagaStepHandler):
        """Enregistrer un handler pour une étape"""
        self._step_handlers[step_name] = handler
        self.logger.debug(f"Registered handler for step '{step_name}'")

    @trace("saga.execute")
    async def execute(self) -> bool:
        """Exécuter la saga complète"""
        try:
            self.instance.state = SagaState.RUNNING

            # Publier événement de démarrage
            await self._publish_saga_event("saga.started")

            # Exécuter chaque étape
            for i, step in enumerate(self.instance.definition.steps):
                self.instance.current_step_index = i

                success = await self._execute_step(step)
                if not success:
                    # Échec d'une étape, démarrer la compensation
                    await self._compensate()
                    return False

            # Toutes les étapes ont réussi
            self.instance.state = SagaState.COMPLETED
            self.instance.completed_at = datetime.utcnow()

            # Publier événement de succès
            await self._publish_saga_event("saga.completed")

            self.logger.info(
                f"Saga {self.instance.saga_id} completed successfully",
                saga_id=self.instance.saga_id,
                saga_type=self.instance.saga_type,
                duration_ms=(self.instance.completed_at - self.instance.started_at).total_seconds() * 1000
            )

            # Métriques
            self.metrics.collector.increment_counter(
                "saga.completed",
                labels={"saga_type": self.instance.saga_type}
            )

            return True

        except Exception as e:
            self.instance.state = SagaState.FAILED
            self.instance.error_message = str(e)

            self.logger.error(
                f"Saga {self.instance.saga_id} failed with error: {e}",
                saga_id=self.instance.saga_id,
                error=e
            )

            # Publier événement d'échec
            await self._publish_saga_event("saga.failed", {"error": str(e)})

            # Métriques
            self.metrics.collector.increment_counter(
                "saga.failed",
                labels={"saga_type": self.instance.saga_type, "error_type": type(e).__name__}
            )

            # Tenter la compensation
            await self._compensate()
            return False

    async def _execute_step(self, step: SagaStep) -> bool:
        """Exécuter une étape individuelle"""
        handler = self._step_handlers.get(step.step_name)
        if not handler:
            raise ValueError(f"No handler registered for step '{step.step_name}'")

        step.executed_at = datetime.utcnow()
        step.attempts += 1

        self.logger.info(
            f"Executing saga step '{step.step_name}'",
            saga_id=self.instance.saga_id,
            step_id=step.step_id,
            step_name=step.step_name,
            attempt=step.attempts
        )

        # Publier événement de début d'étape
        await self._publish_saga_event("saga.step.started", {
            "step_id": step.step_id,
            "step_name": step.step_name
        })

        try:
            # Exécuter avec timeout
            result = await asyncio.wait_for(
                handler.execute(step, self.instance.context),
                timeout=step.timeout_seconds
            )

            if result == SagaStepResult.SUCCESS:
                step.status = SagaStepResult.SUCCESS
                step.completed_at = datetime.utcnow()

                # Publier événement de succès d'étape
                await self._publish_saga_event("saga.step.completed", {
                    "step_id": step.step_id,
                    "step_name": step.step_name
                })

                self.logger.info(
                    f"Saga step '{step.step_name}' completed successfully",
                    saga_id=self.instance.saga_id,
                    step_id=step.step_id
                )

                return True

            elif result == SagaStepResult.RETRY:
                if step.attempts < step.retry_count:
                    # Retry avec backoff
                    await asyncio.sleep(min(2 ** step.attempts, 60))
                    return await self._execute_step(step)
                else:
                    step.status = SagaStepResult.FAILURE
                    step.error_message = f"Max retries ({step.retry_count}) exceeded"
                    return False

            else:  # FAILURE ou COMPENSATE
                step.status = SagaStepResult.FAILURE
                return False

        except asyncio.TimeoutError:
            step.status = SagaStepResult.FAILURE
            step.error_message = f"Step timed out after {step.timeout_seconds} seconds"

            self.logger.error(
                f"Saga step '{step.step_name}' timed out",
                saga_id=self.instance.saga_id,
                step_id=step.step_id,
                timeout_seconds=step.timeout_seconds
            )

            return False

        except Exception as e:
            step.status = SagaStepResult.FAILURE
            step.error_message = str(e)

            self.logger.error(
                f"Saga step '{step.step_name}' failed with error: {e}",
                saga_id=self.instance.saga_id,
                step_id=step.step_id,
                error=e
            )

            # Publier événement d'échec d'étape
            await self._publish_saga_event("saga.step.failed", {
                "step_id": step.step_id,
                "step_name": step.step_name,
                "error": str(e)
            })

            return False

    async def _compensate(self):
        """Exécuter les compensations pour les étapes complétées"""
        self.instance.state = SagaState.COMPENSATING

        # Publier événement de début de compensation
        await self._publish_saga_event("saga.compensation.started")

        completed_steps = self.instance.completed_steps

        if self.instance.definition.compensation_strategy == "reverse_order":
            # Compenser dans l'ordre inverse
            for step in reversed(completed_steps):
                await self._compensate_step(step)
        else:
            # Compensation parallèle
            compensation_tasks = [
                self._compensate_step(step) for step in completed_steps
            ]
            await asyncio.gather(*compensation_tasks, return_exceptions=True)

        self.instance.state = SagaState.COMPENSATED
        self.instance.completed_at = datetime.utcnow()

        # Publier événement de fin de compensation
        await self._publish_saga_event("saga.compensation.completed")

        self.logger.info(
            f"Saga {self.instance.saga_id} compensation completed",
            saga_id=self.instance.saga_id,
            compensated_steps=len(completed_steps)
        )

    async def _compensate_step(self, step: SagaStep):
        """Compenser une étape individuelle"""
        handler = self._step_handlers.get(step.step_name)
        if not handler:
            self.logger.warning(f"No handler for compensation of step '{step.step_name}'")
            return

        self.logger.info(
            f"Compensating saga step '{step.step_name}'",
            saga_id=self.instance.saga_id,
            step_id=step.step_id
        )

        try:
            success = await handler.compensate(step, self.instance.context)
            if success:
                self.logger.info(
                    f"Successfully compensated step '{step.step_name}'",
                    saga_id=self.instance.saga_id,
                    step_id=step.step_id
                )
            else:
                self.logger.error(
                    f"Failed to compensate step '{step.step_name}'",
                    saga_id=self.instance.saga_id,
                    step_id=step.step_id
                )

        except Exception as e:
            self.logger.error(
                f"Error compensating step '{step.step_name}': {e}",
                saga_id=self.instance.saga_id,
                step_id=step.step_id,
                error=e
            )

    async def _publish_saga_event(self, event_type: str, additional_data: Dict[str, Any] = None):
        """Publier un événement de saga"""
        event_data = {
            "saga_id": self.instance.saga_id,
            "saga_type": self.instance.saga_type,
            "state": self.instance.state.value,
            "current_step_index": self.instance.current_step_index
        }

        if additional_data:
            event_data.update(additional_data)

        event = DomainEvent(
            event_type=EventType.STRATEGY_STARTED if "started" in event_type else EventType.STRATEGY_STOPPED,
            aggregate_id=self.instance.saga_id,
            data=event_data,
            correlation_id=self.instance.correlation_id,
            source=f"saga_{self.instance.saga_type}"
        )

        await self.event_bus.publish(event)


class SagaManager:
    """
    Gestionnaire des sagas pour orchestrer leur exécution
    et maintenir leur état.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

        # Instances de sagas actives
        self._active_sagas: Dict[str, Saga] = {}
        self._saga_definitions: Dict[str, SagaDefinition] = {}

        # Handlers globaux
        self._step_handlers: Dict[str, SagaStepHandler] = {}

        # Statistiques
        self._stats = {
            "sagas_started": 0,
            "sagas_completed": 0,
            "sagas_failed": 0,
            "sagas_compensated": 0
        }

    def register_saga_definition(self, definition: SagaDefinition):
        """Enregistrer une définition de saga"""
        self._saga_definitions[definition.saga_type] = definition
        self.logger.info(f"Registered saga definition: {definition.saga_type}")

    def register_step_handler(self, step_name: str, handler: SagaStepHandler):
        """Enregistrer un handler d'étape global"""
        self._step_handlers[step_name] = handler
        self.logger.info(f"Registered global step handler: {step_name}")

    async def start_saga(
        self,
        saga_type: str,
        context: Dict[str, Any],
        correlation_id: Optional[str] = None,
        triggered_by: Optional[str] = None
    ) -> str:
        """Démarrer une nouvelle saga"""
        if saga_type not in self._saga_definitions:
            raise ValueError(f"Unknown saga type: {saga_type}")

        definition = self._saga_definitions[saga_type]
        saga_id = str(uuid.uuid4())

        instance = SagaInstance(
            saga_id=saga_id,
            saga_type=saga_type,
            definition=definition,
            context=context,
            correlation_id=correlation_id,
            triggered_by=triggered_by
        )

        saga = Saga(instance)

        # Enregistrer les handlers d'étapes
        for step_name, handler in self._step_handlers.items():
            saga.register_step_handler(step_name, handler)

        self._active_sagas[saga_id] = saga
        self._stats["sagas_started"] += 1

        self.logger.info(
            f"Starting saga {saga_id} of type {saga_type}",
            saga_id=saga_id,
            saga_type=saga_type,
            correlation_id=correlation_id
        )

        # Démarrer l'exécution en arrière-plan
        asyncio.create_task(self._execute_saga(saga))

        return saga_id

    async def _execute_saga(self, saga: Saga):
        """Exécuter une saga et nettoyer après"""
        try:
            success = await saga.execute()
            if success:
                self._stats["sagas_completed"] += 1
            else:
                if saga.instance.state == SagaState.COMPENSATED:
                    self._stats["sagas_compensated"] += 1
                else:
                    self._stats["sagas_failed"] += 1

        except Exception as e:
            self.logger.error(f"Saga execution error: {e}", error=e)
            self._stats["sagas_failed"] += 1

        finally:
            # Nettoyer la saga active
            if saga.instance.saga_id in self._active_sagas:
                del self._active_sagas[saga.instance.saga_id]

    def get_saga(self, saga_id: str) -> Optional[Saga]:
        """Obtenir une saga active"""
        return self._active_sagas.get(saga_id)

    def list_active_sagas(self) -> List[str]:
        """Lister les IDs des sagas actives"""
        return list(self._active_sagas.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques du gestionnaire"""
        return {
            **self._stats,
            "active_sagas": len(self._active_sagas),
            "registered_definitions": len(self._saga_definitions),
            "registered_handlers": len(self._step_handlers)
        }


# Instance globale
_global_saga_manager = SagaManager()


def get_saga_manager() -> SagaManager:
    """Obtenir l'instance globale du gestionnaire de sagas"""
    return _global_saga_manager