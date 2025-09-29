"""
Structured Logging Framework for QFrame Enterprise
================================================

Système de logging structuré avec métriques, tracing distribué,
et intégration observabilité moderne.
"""

import json
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import logging
import logging.handlers
from dataclasses import dataclass, field, asdict
from functools import wraps
import traceback
import uuid

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

try:
    import opentelemetry
    from opentelemetry import trace
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


class LogLevel(str, Enum):
    """Niveaux de log standardisés."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Formats de sortie supportés."""
    JSON = "json"
    CONSOLE = "console"
    STRUCTURED = "structured"


@dataclass
class LogContext:
    """Contexte de logging enrichi."""

    # Identification
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Exécution
    service_name: str = "qframe"
    service_version: str = "1.0.0"
    environment: str = "development"

    # Métadonnées métier
    strategy_name: Optional[str] = None
    portfolio_id: Optional[str] = None
    symbol: Optional[str] = None

    # Technique
    thread_id: str = field(default_factory=lambda: str(threading.get_ident()))
    process_id: int = field(default_factory=lambda: os.getpid() if 'os' in globals() else 0)

    # Tracing distribué
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Performance
    start_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PerformanceMetrics:
    """Métriques de performance pour logging."""

    operation: str
    duration_ms: float
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None

    # Métriques spécifiques trading
    orders_processed: Optional[int] = None
    signals_generated: Optional[int] = None
    features_calculated: Optional[int] = None

    # Métadonnées
    success: bool = True
    error_type: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Export pour logging."""
        return asdict(self)


class EnhancedLogRecord:
    """Enregistrement de log enrichi."""

    def __init__(
        self,
        level: LogLevel,
        message: str,
        context: LogContext,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        metrics: Optional[PerformanceMetrics] = None
    ):
        self.level = level
        self.message = message
        self.context = context
        self.extra = extra or {}
        self.exception = exception
        self.metrics = metrics
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation complète."""
        record = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'context': self.context.to_dict(),
            **self.extra
        }

        if self.exception:
            record['exception'] = {
                'type': type(self.exception).__name__,
                'message': str(self.exception),
                'traceback': traceback.format_exception(
                    type(self.exception),
                    self.exception,
                    self.exception.__traceback__
                )
            }

        if self.metrics:
            record['metrics'] = self.metrics.to_dict()

        return record


class LogHandler:
    """Handler de logging personnalisé."""

    def __init__(self, format_type: LogFormat = LogFormat.JSON):
        self.format_type = format_type

    def emit(self, record: EnhancedLogRecord) -> None:
        """Émet un log record."""
        if self.format_type == LogFormat.JSON:
            output = json.dumps(record.to_dict(), ensure_ascii=False, indent=None)
        elif self.format_type == LogFormat.CONSOLE:
            output = self._format_console(record)
        else:
            output = self._format_structured(record)

        print(output, file=sys.stdout if record.level.value not in ['ERROR', 'CRITICAL'] else sys.stderr)

    def _format_console(self, record: EnhancedLogRecord) -> str:
        """Format console lisible."""
        timestamp = record.timestamp.strftime('%H:%M:%S.%f')[:-3]
        level = record.level.value.ljust(8)

        # Contexte essentiel
        context_parts = []
        if record.context.request_id:
            context_parts.append(f"req={record.context.request_id[:8]}")
        if record.context.strategy_name:
            context_parts.append(f"strategy={record.context.strategy_name}")
        if record.context.symbol:
            context_parts.append(f"symbol={record.context.symbol}")

        context_str = f"[{','.join(context_parts)}]" if context_parts else ""

        base = f"{timestamp} {level} {context_str} {record.message}"

        if record.metrics:
            base += f" (took {record.metrics.duration_ms:.1f}ms)"

        if record.exception:
            base += f"\n  Exception: {record.exception}"

        return base

    def _format_structured(self, record: EnhancedLogRecord) -> str:
        """Format structuré mais lisible."""
        return f"{record.timestamp.isoformat()} [{record.level.value}] {record.message} | {json.dumps(record.context.to_dict(), ensure_ascii=False)}"


class StructuredLogger:
    """Logger structuré principal."""

    def __init__(
        self,
        name: str,
        context: Optional[LogContext] = None,
        handlers: Optional[List[LogHandler]] = None,
        min_level: LogLevel = LogLevel.INFO
    ):
        self.name = name
        self.context = context or LogContext(service_name=name)
        self.handlers = handlers or [LogHandler(LogFormat.CONSOLE)]
        self.min_level = min_level

        # Thread local storage pour contexte
        self._local = threading.local()

    def _should_log(self, level: LogLevel) -> bool:
        """Vérifie si le niveau doit être loggé."""
        levels = [LogLevel.TRACE, LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        return levels.index(level) >= levels.index(self.min_level)

    def _get_current_context(self) -> LogContext:
        """Récupère le contexte actuel (thread-safe)."""
        if hasattr(self._local, 'context'):
            return self._local.context
        return self.context

    def with_context(self, **kwargs) -> 'StructuredLogger':
        """Crée un logger avec contexte enrichi."""
        new_context = LogContext(**{**self.context.to_dict(), **kwargs})
        return StructuredLogger(
            name=self.name,
            context=new_context,
            handlers=self.handlers,
            min_level=self.min_level
        )

    @contextmanager
    def context_manager(self, **kwargs):
        """Context manager temporaire."""
        old_context = getattr(self._local, 'context', None)
        new_context = LogContext(**{**self._get_current_context().to_dict(), **kwargs})
        self._local.context = new_context

        try:
            yield self
        finally:
            if old_context:
                self._local.context = old_context
            else:
                delattr(self._local, 'context')

    def _log(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        metrics: Optional[PerformanceMetrics] = None
    ):
        """Méthode de logging interne."""
        if not self._should_log(level):
            return

        record = EnhancedLogRecord(
            level=level,
            message=message,
            context=self._get_current_context(),
            extra=extra,
            exception=exception,
            metrics=metrics
        )

        for handler in self.handlers:
            try:
                handler.emit(record)
            except Exception as e:
                # Fallback logging en cas d'erreur handler
                print(f"Logging handler error: {e}", file=sys.stderr)

    def trace(self, message: str, **kwargs):
        """Log niveau TRACE."""
        self._log(LogLevel.TRACE, message, kwargs)

    def debug(self, message: str, **kwargs):
        """Log niveau DEBUG."""
        self._log(LogLevel.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs):
        """Log niveau INFO."""
        self._log(LogLevel.INFO, message, kwargs)

    def warning(self, message: str, **kwargs):
        """Log niveau WARNING."""
        self._log(LogLevel.WARNING, message, kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log niveau ERROR."""
        self._log(LogLevel.ERROR, message, kwargs, exception=exception)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log niveau CRITICAL."""
        self._log(LogLevel.CRITICAL, message, kwargs, exception=exception)


class PerformanceLogger:
    """Logger spécialisé pour métriques de performance."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    @contextmanager
    def measure_operation(self, operation: str, **context):
        """Mesure automatique d'une opération."""
        start_time = time.time()
        start_context = {**context, 'operation': operation, 'start_time': start_time}

        with self.logger.context_manager(**start_context):
            self.logger.debug(f"Starting {operation}")

            try:
                yield
                duration_ms = (time.time() - start_time) * 1000

                metrics = PerformanceMetrics(
                    operation=operation,
                    duration_ms=duration_ms,
                    success=True
                )

                self.logger.info(
                    f"Completed {operation}",
                    metrics=metrics
                )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                metrics = PerformanceMetrics(
                    operation=operation,
                    duration_ms=duration_ms,
                    success=False,
                    error_type=type(e).__name__
                )

                self.logger.error(
                    f"Failed {operation}",
                    exception=e,
                    metrics=metrics
                )
                raise

    def log_metrics(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **metrics_data
    ):
        """Log explicite de métriques."""
        metrics = PerformanceMetrics(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **metrics_data
        )

        level = LogLevel.INFO if success else LogLevel.ERROR
        message = f"Metrics for {operation}"

        self.logger._log(level, message, metrics=metrics)


def performance_logged(operation: Optional[str] = None):
    """Décorateur pour logging automatique de performance."""

    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Essayer de récupérer le logger depuis les arguments
            logger = None
            for arg in args:
                if hasattr(arg, 'logger') and isinstance(arg.logger, StructuredLogger):
                    logger = PerformanceLogger(arg.logger)
                    break

            if not logger:
                # Logger par défaut
                logger = PerformanceLogger(get_default_logger())

            with logger.measure_operation(op_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


class LoggerFactory:
    """Factory pour créer des loggers configurés."""

    _loggers: Dict[str, StructuredLogger] = {}
    _default_config = {
        'min_level': LogLevel.INFO,
        'format': LogFormat.CONSOLE,
        'context': LogContext()
    }

    @classmethod
    def configure_defaults(
        self,
        min_level: LogLevel = LogLevel.INFO,
        format_type: LogFormat = LogFormat.CONSOLE,
        service_name: str = "qframe",
        environment: str = "development"
    ):
        """Configure les paramètres par défaut."""
        self._default_config = {
            'min_level': min_level,
            'format': format_type,
            'context': LogContext(
                service_name=service_name,
                environment=environment
            )
        }

    @classmethod
    def get_logger(
        cls,
        name: str,
        context: Optional[LogContext] = None,
        min_level: Optional[LogLevel] = None
    ) -> StructuredLogger:
        """Récupère ou crée un logger."""

        if name in cls._loggers:
            logger = cls._loggers[name]
            if context:
                logger = logger.with_context(**context.to_dict())
            return logger

        # Créer nouveau logger
        logger_context = context or cls._default_config['context']
        logger_min_level = min_level or cls._default_config['min_level']

        handlers = [LogHandler(cls._default_config['format'])]

        logger = StructuredLogger(
            name=name,
            context=logger_context,
            handlers=handlers,
            min_level=logger_min_level
        )

        cls._loggers[name] = logger
        return logger

    @classmethod
    def get_trading_logger(
        cls,
        strategy_name: str,
        portfolio_id: Optional[str] = None
    ) -> StructuredLogger:
        """Logger spécialisé pour trading."""
        context = LogContext(
            service_name="qframe-trading",
            strategy_name=strategy_name,
            portfolio_id=portfolio_id
        )

        return cls.get_logger(f"trading.{strategy_name}", context)

    @classmethod
    def get_data_logger(cls, symbol: str) -> StructuredLogger:
        """Logger spécialisé pour données."""
        context = LogContext(
            service_name="qframe-data",
            symbol=symbol
        )

        return cls.get_logger(f"data.{symbol}", context)


# Logger par défaut global
_default_logger: Optional[StructuredLogger] = None

def get_default_logger() -> StructuredLogger:
    """Récupère le logger par défaut."""
    global _default_logger
    if _default_logger is None:
        _default_logger = LoggerFactory.get_logger("qframe")
    return _default_logger


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.CONSOLE,
    service_name: str = "qframe",
    environment: str = "development"
):
    """Configuration globale du logging."""
    LoggerFactory.configure_defaults(
        min_level=level,
        format_type=format_type,
        service_name=service_name,
        environment=environment
    )

    global _default_logger
    _default_logger = None  # Force recreation


# Convenience imports
__all__ = [
    'LogLevel', 'LogFormat', 'LogContext', 'PerformanceMetrics',
    'StructuredLogger', 'PerformanceLogger', 'LoggerFactory',
    'performance_logged', 'get_default_logger', 'configure_logging'
]


if __name__ == "__main__":
    # Démonstration du système
    import os
    os.getpid = lambda: 12345  # Mock pour la démo

    # Configuration
    configure_logging(
        level=LogLevel.DEBUG,
        format_type=LogFormat.CONSOLE,
        service_name="qframe-demo",
        environment="development"
    )

    # Logger simple
    logger = get_default_logger()
    logger.info("🚀 Système de logging structuré initialisé")

    # Logger avec contexte
    trading_logger = LoggerFactory.get_trading_logger(
        strategy_name="DMN_LSTM",
        portfolio_id="portfolio_123"
    )

    with trading_logger.context_manager(symbol="BTC/USD", order_id="order_456"):
        trading_logger.info("📊 Génération signal de trading")

        # Performance logging
        perf_logger = PerformanceLogger(trading_logger)

        with perf_logger.measure_operation("feature_calculation", features_count=15):
            time.sleep(0.1)  # Simulation

        # Logger d'erreur
        try:
            raise ValueError("Erreur de démonstration")
        except Exception as e:
            trading_logger.error("❌ Erreur dans le calcul", exception=e)

    print("\n✅ Démonstration terminée")