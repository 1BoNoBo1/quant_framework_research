"""
Infrastructure Layer: Distributed Tracing System
===============================================

Système de tracing distribué pour suivre les flux à travers les services.
Compatible avec OpenTelemetry, Jaeger, Zipkin, etc.
"""

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from threading import Lock
import json
import random
import traceback


# Context variables pour le tracing
current_trace: ContextVar[Optional['Trace']] = ContextVar('current_trace', default=None)
current_span: ContextVar[Optional['Span']] = ContextVar('current_span', default=None)


class SpanKind(str, Enum):
    """Type de span"""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Statut d'un span"""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Contexte de propagation pour le tracing distribué"""
    trace_id: str
    span_id: str
    trace_flags: int = 0
    trace_state: Dict[str, str] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)

    def to_headers(self) -> Dict[str, str]:
        """Convertir en headers HTTP pour propagation"""
        headers = {
            "X-Trace-Id": self.trace_id,
            "X-Span-Id": self.span_id,
            "X-Trace-Flags": str(self.trace_flags)
        }

        if self.trace_state:
            headers["X-Trace-State"] = json.dumps(self.trace_state)

        if self.baggage:
            headers["X-Baggage"] = json.dumps(self.baggage)

        return headers

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['SpanContext']:
        """Créer un contexte depuis des headers HTTP"""
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")

        if not trace_id or not span_id:
            return None

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=int(headers.get("X-Trace-Flags", "0")),
            trace_state=json.loads(headers.get("X-Trace-State", "{}")),
            baggage=json.loads(headers.get("X-Baggage", "{}"))
        )


@dataclass
class Span:
    """Représentation d'un span dans une trace"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List['SpanEvent'] = field(default_factory=list)
    links: List['SpanLink'] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        """Durée en millisecondes"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def set_attribute(self, key: str, value: Any):
        """Définir un attribut"""
        self.attributes[key] = self._serialize_value(value)

    def set_attributes(self, attributes: Dict[str, Any]):
        """Définir plusieurs attributs"""
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Ajouter un événement au span"""
        event = SpanEvent(
            name=name,
            timestamp=datetime.utcnow(),
            attributes=attributes or {}
        )
        self.events.append(event)

    def set_status(self, status: SpanStatus, message: Optional[str] = None):
        """Définir le statut du span"""
        self.status = status
        self.status_message = message

    def record_exception(self, exception: Exception):
        """Enregistrer une exception"""
        self.set_status(SpanStatus.ERROR, str(exception))
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.stacktrace": traceback.format_exc()
            }
        )

    def end(self):
        """Terminer le span"""
        if not self.end_time:
            self.end_time = datetime.utcnow()

    def _serialize_value(self, value: Any) -> Any:
        """Sérialiser une valeur pour stockage"""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return str(value)


@dataclass
class SpanEvent:
    """Événement dans un span"""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """Lien vers un autre span"""
    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Représentation d'une trace complète"""
    trace_id: str
    root_span: Optional[Span] = None
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_span(self, span: Span):
        """Ajouter un span à la trace"""
        self.spans.append(span)
        if not self.root_span or not span.parent_span_id:
            self.root_span = span

    def get_span(self, span_id: str) -> Optional[Span]:
        """Récupérer un span par ID"""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_duration_ms(self) -> Optional[float]:
        """Durée totale de la trace"""
        if not self.spans:
            return None

        start_times = [s.start_time for s in self.spans]
        end_times = [s.end_time for s in self.spans if s.end_time]

        if not end_times:
            return None

        duration = (max(end_times) - min(start_times)).total_seconds() * 1000
        return duration

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour sérialisation"""
        return {
            "trace_id": self.trace_id,
            "duration_ms": self.get_duration_ms(),
            "span_count": len(self.spans),
            "metadata": self.metadata,
            "spans": [
                {
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "operation_name": span.operation_name,
                    "start_time": span.start_time.isoformat(),
                    "end_time": span.end_time.isoformat() if span.end_time else None,
                    "duration_ms": span.duration_ms,
                    "kind": span.kind.value,
                    "status": span.status.value,
                    "status_message": span.status_message,
                    "attributes": span.attributes,
                    "events": [
                        {
                            "name": e.name,
                            "timestamp": e.timestamp.isoformat(),
                            "attributes": e.attributes
                        }
                        for e in span.events
                    ]
                }
                for span in self.spans
            ]
        }


class Tracer:
    """
    Tracer principal pour créer et gérer les spans.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._traces: Dict[str, Trace] = {}
        self._lock = Lock()
        self._samplers: List[Callable[[str], bool]] = []

        # Sampler par défaut (100% des traces)
        self.add_sampler(lambda op: True)

    def add_sampler(self, sampler: Callable[[str], bool]):
        """Ajouter un sampler pour décider quelles traces capturer"""
        self._samplers.append(sampler)

    def should_sample(self, operation_name: str) -> bool:
        """Décider si une opération doit être tracée"""
        return any(sampler(operation_name) for sampler in self._samplers)

    def start_span(
        self,
        operation_name: str,
        parent: Optional[Span] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Démarrer un nouveau span"""

        # Vérifier le sampling
        if not self.should_sample(operation_name):
            # Retourner un span no-op
            return NoOpSpan()

        # Obtenir ou créer une trace
        if parent:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id
        else:
            # Vérifier s'il y a un span actuel dans le contexte
            parent = current_span.get()
            if parent:
                trace_id = parent.trace_id
                parent_span_id = parent.span_id
            else:
                # Nouvelle trace
                trace_id = self._generate_trace_id()
                parent_span_id = None

        # Créer le nouveau span
        span = Span(
            span_id=self._generate_span_id(),
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            kind=kind,
            attributes=attributes or {}
        )

        # Ajouter des attributs par défaut
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("span.kind", kind.value)

        # Ajouter à la trace
        with self._lock:
            if trace_id not in self._traces:
                self._traces[trace_id] = Trace(trace_id=trace_id)
            self._traces[trace_id].add_span(span)

        # Définir comme span actuel
        current_span.set(span)

        return span

    def start_trace(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Trace:
        """Démarrer une nouvelle trace"""
        span = self.start_span(operation_name, parent=None, attributes=attributes)
        trace = self._traces.get(span.trace_id)

        if trace:
            current_trace.set(trace)

        return trace

    def get_current_span(self) -> Optional[Span]:
        """Obtenir le span actuel"""
        return current_span.get()

    def get_current_trace(self) -> Optional[Trace]:
        """Obtenir la trace actuelle"""
        return current_trace.get()

    def inject_context(self, carrier: Dict[str, str]) -> Dict[str, str]:
        """Injecter le contexte de tracing dans un carrier (ex: headers HTTP)"""
        span = self.get_current_span()
        if not span:
            return carrier

        context = SpanContext(
            trace_id=span.trace_id,
            span_id=span.span_id
        )

        carrier.update(context.to_headers())
        return carrier

    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extraire le contexte de tracing d'un carrier"""
        return SpanContext.from_headers(carrier)

    def span(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Context manager pour créer un span"""
        return SpanContext(self, operation_name, kind, attributes)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Récupérer une trace par ID"""
        with self._lock:
            return self._traces.get(trace_id)

    def export_traces(self) -> List[Dict[str, Any]]:
        """Exporter toutes les traces pour analyse"""
        with self._lock:
            return [trace.to_dict() for trace in self._traces.values()]

    def clear_traces(self):
        """Effacer toutes les traces (pour les tests)"""
        with self._lock:
            self._traces.clear()

    def _generate_trace_id(self) -> str:
        """Générer un ID de trace unique"""
        return uuid.uuid4().hex

    def _generate_span_id(self) -> str:
        """Générer un ID de span unique"""
        return uuid.uuid4().hex[:16]

    def trace_operation(self, operation_name: str, **attributes):
        """Décorateur pour tracer une fonction"""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                with self.span(operation_name, attributes=attributes) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(SpanStatus.OK)
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator


class SpanContext:
    """Context manager pour les spans"""

    def __init__(
        self,
        tracer: Tracer,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.tracer = tracer
        self.operation_name = operation_name
        self.kind = kind
        self.attributes = attributes
        self.span = None

    def __enter__(self) -> Span:
        self.span = self.tracer.start_span(
            self.operation_name,
            kind=self.kind,
            attributes=self.attributes
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(SpanStatus.OK)
            self.span.end()


class NoOpSpan(Span):
    """Span no-op pour quand le sampling est désactivé"""

    def __init__(self):
        super().__init__(
            span_id="noop",
            trace_id="noop",
            parent_span_id=None,
            operation_name="noop",
            start_time=datetime.utcnow()
        )

    def set_attribute(self, key: str, value: Any):
        pass

    def set_attributes(self, attributes: Dict[str, Any]):
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        pass

    def set_status(self, status: SpanStatus, message: Optional[str] = None):
        pass

    def record_exception(self, exception: Exception):
        pass

    def end(self):
        pass


class TradingTracer(Tracer):
    """
    Tracer spécialisé pour les opérations de trading.
    """

    def trace_order_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal
    ) -> Span:
        """Tracer l'exécution d'un ordre"""
        span = self.start_span(
            "order.execute",
            kind=SpanKind.INTERNAL,
            attributes={
                "order.id": order_id,
                "order.symbol": symbol,
                "order.side": side,
                "order.quantity": float(quantity),
                "order.price": float(price),
                "order.value": float(quantity * price)
            }
        )
        return span

    def trace_strategy_signal(
        self,
        strategy_id: str,
        signal_type: str,
        confidence: float
    ) -> Span:
        """Tracer la génération d'un signal de stratégie"""
        span = self.start_span(
            "strategy.signal",
            kind=SpanKind.INTERNAL,
            attributes={
                "strategy.id": strategy_id,
                "signal.type": signal_type,
                "signal.confidence": confidence
            }
        )
        return span

    def trace_risk_check(
        self,
        check_type: str,
        portfolio_id: str,
        result: str
    ) -> Span:
        """Tracer une vérification de risque"""
        span = self.start_span(
            "risk.check",
            kind=SpanKind.INTERNAL,
            attributes={
                "risk.check_type": check_type,
                "portfolio.id": portfolio_id,
                "risk.result": result
            }
        )
        return span

    def trace_market_data_fetch(
        self,
        provider: str,
        symbol: str,
        data_type: str
    ) -> Span:
        """Tracer la récupération de données de marché"""
        span = self.start_span(
            "market_data.fetch",
            kind=SpanKind.CLIENT,
            attributes={
                "provider": provider,
                "symbol": symbol,
                "data.type": data_type
            }
        )
        return span


# Instance globale
_global_tracer = TradingTracer("qframe")


def get_tracer() -> TradingTracer:
    """Obtenir l'instance globale du tracer"""
    return _global_tracer


def trace(operation_name: str, **attributes):
    """Décorateur simple pour tracer une fonction"""
    return _global_tracer.trace_operation(operation_name, **attributes)