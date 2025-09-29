"""
Infrastructure Layer: Structured Logging System
==============================================

Système de logging structuré avec correlation IDs, contexte métier,
et formatage JSON pour analyse dans ELK Stack ou DataDog.
"""

import json
import logging
import sys
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
import inspect


# Context variables pour correlation tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class LogLevel(str, Enum):
    """Niveaux de log étendus pour trading"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRADE = "TRADE"  # Logs spécifiques aux trades
    RISK = "RISK"    # Logs de risk management
    AUDIT = "AUDIT"  # Logs d'audit pour compliance


@dataclass
class LogContext:
    """Contexte enrichi pour les logs"""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    environment: str = "development"
    service_name: str = "qframe"
    service_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour JSON"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class StructuredLogger:
    """
    Logger structuré pour production avec support de:
    - Correlation IDs pour traçabilité
    - Contexte métier riche
    - Format JSON pour analyse
    - Performance metrics
    - Error tracking détaillé
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        output_format: str = "json",  # "json" ou "console"
        context: Optional[LogContext] = None
    ):
        self.name = name
        self.output_format = output_format
        self.context = context or LogContext()

        # Configurer le logger Python standard
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))

        # Créer le handler approprié
        handler = logging.StreamHandler(sys.stdout)

        if output_format == "json":
            handler.setFormatter(JsonFormatter(self.context))
        else:
            handler.setFormatter(ConsoleFormatter())

        self.logger.handlers.clear()
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def with_context(self, **kwargs) -> 'StructuredLogger':
        """Créer un nouveau logger avec contexte enrichi"""
        new_context = LogContext(**{**self.context.__dict__, **kwargs})
        return StructuredLogger(self.name, output_format=self.output_format, context=new_context)

    def debug(self, message: str, **kwargs):
        """Log niveau DEBUG"""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log niveau INFO"""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log niveau WARNING"""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log niveau ERROR avec exception tracking"""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
            kwargs['error_traceback'] = traceback.format_exc()
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log niveau CRITICAL"""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
            kwargs['error_traceback'] = traceback.format_exc()
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def trade(self, message: str, **kwargs):
        """Log spécifique aux trades"""
        self._log(LogLevel.TRADE, message, **kwargs)

    def risk(self, message: str, **kwargs):
        """Log spécifique au risk management"""
        self._log(LogLevel.RISK, message, **kwargs)

    def audit(self, message: str, **kwargs):
        """Log d'audit pour compliance"""
        self._log(LogLevel.AUDIT, message, **kwargs)

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Méthode interne de logging"""
        # Enrichir avec le contexte (sans 'message' pour éviter conflit LogRecord)
        extra_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level.value,
            'logger': self.name,
            **self.context.to_dict(),
            **self._get_correlation_context(),
            **self._sanitize_kwargs(kwargs)
        }

        # Ajouter les méta-données de code
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_frame = frame.f_back.f_back
            extra_data['source'] = {
                'file': caller_frame.f_code.co_filename.split('/')[-1],
                'function': caller_frame.f_code.co_name,
                'line': caller_frame.f_lineno
            }

        # Logger selon le niveau
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.TRADE: logging.INFO,
            LogLevel.RISK: logging.WARNING,
            LogLevel.AUDIT: logging.INFO
        }

        # Pour JSON, inclure le message dans la structure
        if self.output_format == "json":
            log_data = {'message': message, **extra_data}
            self.logger.log(level_map[level], json.dumps(log_data), extra=extra_data)
        else:
            self.logger.log(level_map[level], message, extra=extra_data)

    def _get_correlation_context(self) -> Dict[str, Any]:
        """Récupérer le contexte de correlation depuis ContextVars"""
        context = {}

        if correlation_id.get():
            context['correlation_id'] = correlation_id.get()
        if user_id.get():
            context['user_id'] = user_id.get()
        if session_id.get():
            context['session_id'] = session_id.get()
        if request_id.get():
            context['request_id'] = request_id.get()

        return context

    def _sanitize_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoyer et formater les kwargs pour JSON"""
        sanitized = {}

        for key, value in kwargs.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, Decimal):
                sanitized[key] = float(value)
            elif isinstance(value, datetime):
                sanitized[key] = value.isoformat()
            elif isinstance(value, Enum):
                sanitized[key] = value.value
            elif hasattr(value, '__dict__'):
                sanitized[key] = str(value)
            else:
                sanitized[key] = str(value)

        return sanitized

    def measure_performance(self, operation: str):
        """Context manager pour mesurer la performance"""
        return PerformanceTimer(self, operation)

    def log_trade_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        venue: str,
        **kwargs
    ):
        """Logger spécialisé pour l'exécution de trades"""
        self.trade(
            f"Trade executed: {side} {quantity} {symbol} @ {price} on {venue}",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            price=float(price),
            venue=venue,
            trade_value=float(quantity * price),
            **kwargs
        )

    def log_risk_breach(
        self,
        metric: str,
        current_value: Decimal,
        limit: Decimal,
        severity: str,
        **kwargs
    ):
        """Logger spécialisé pour les breaches de risk"""
        self.risk(
            f"Risk limit breach: {metric} = {current_value} (limit: {limit})",
            metric=metric,
            current_value=float(current_value),
            limit=float(limit),
            breach_percentage=float((current_value - limit) / limit * 100),
            severity=severity,
            **kwargs
        )

    def log_portfolio_update(
        self,
        portfolio_id: str,
        total_value: Decimal,
        pnl: Decimal,
        positions: int,
        **kwargs
    ):
        """Logger spécialisé pour les mises à jour de portfolio"""
        self.info(
            f"Portfolio updated: {portfolio_id} - Value: {total_value}, P&L: {pnl}",
            portfolio_id=portfolio_id,
            total_value=float(total_value),
            pnl=float(pnl),
            pnl_percentage=float(pnl / total_value * 100) if total_value > 0 else 0,
            positions_count=positions,
            **kwargs
        )


class JsonFormatter(logging.Formatter):
    """Formatter pour sortie JSON structurée"""

    def __init__(self, context: LogContext):
        super().__init__()
        self.context = context

    def format(self, record: logging.LogRecord) -> str:
        """Formater le record en JSON"""
        # Les données sont déjà dans extra
        if hasattr(record, 'extra'):
            return json.dumps(record.extra)

        # Fallback si pas de données extra
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            **self.context.to_dict()
        }

        if record.exc_info:
            log_data['error_traceback'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Formatter pour sortie console lisible"""

    def format(self, record: logging.LogRecord) -> str:
        """Formater le record pour la console"""
        # Couleurs ANSI pour la console
        colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'
        }

        color = colors.get(record.levelname, colors['RESET'])

        # Format de base
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        base_format = f"{color}[{timestamp}] [{record.levelname:8}] {record.name}: {record.getMessage()}{colors['RESET']}"

        # Ajouter le contexte s'il existe
        if hasattr(record, 'correlation_id'):
            base_format += f" (correlation_id={record.correlation_id})"

        if record.exc_info:
            base_format += f"\n{self.formatException(record.exc_info)}"

        return base_format


class PerformanceTimer:
    """Context manager pour mesurer la performance des opérations"""

    def __init__(self, logger: StructuredLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds() * 1000  # En millisecondes

        if exc_type:
            self.logger.error(
                f"Operation failed: {self.operation}",
                error=exc_val,
                duration_ms=duration
            )
        else:
            log_method = self.logger.debug if duration < 100 else self.logger.warning if duration < 1000 else self.logger.error
            log_method(
                f"Operation completed: {self.operation}",
                duration_ms=duration,
                performance_status="fast" if duration < 100 else "slow" if duration < 1000 else "critical"
            )


class LoggerFactory:
    """Factory pour créer des loggers configurés"""

    _loggers: Dict[str, StructuredLogger] = {}
    _default_context = LogContext()
    _default_level = "INFO"
    _default_format = "json"

    @classmethod
    def configure_defaults(
        cls,
        level: str = "INFO",
        format: str = "json",
        context: Optional[LogContext] = None
    ):
        """Configurer les paramètres par défaut"""
        cls._default_level = level
        cls._default_format = format
        if context:
            cls._default_context = context

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: Optional[str] = None,
        format: Optional[str] = None,
        context: Optional[LogContext] = None
    ) -> StructuredLogger:
        """Obtenir ou créer un logger"""
        if name not in cls._loggers:
            cls._loggers[name] = StructuredLogger(
                name=name,
                level=level or cls._default_level,
                output_format=format or cls._default_format,
                context=context or cls._default_context
            )
        return cls._loggers[name]

    @classmethod
    def set_correlation_id(cls, correlation_id_value: str):
        """Définir un correlation ID global"""
        correlation_id.set(correlation_id_value)

    @classmethod
    def set_user_context(cls, user_id_value: str, session_id_value: Optional[str] = None):
        """Définir le contexte utilisateur"""
        user_id.set(user_id_value)
        if session_id_value:
            session_id.set(session_id_value)

    @classmethod
    def set_request_context(cls, request_id_value: str):
        """Définir le contexte de requête"""
        request_id.set(request_id_value)

    @classmethod
    def create_correlation_id(cls) -> str:
        """Créer et définir un nouveau correlation ID"""
        new_id = str(uuid.uuid4())
        cls.set_correlation_id(new_id)
        return new_id


# Logger par défaut pour l'import direct
logger = LoggerFactory.get_logger(__name__)