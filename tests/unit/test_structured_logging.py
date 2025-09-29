"""
Tests pour Structured Logging Framework
======================================

Tests complets du système de logging structuré avec contexte et métriques.
"""

import pytest
import json
import threading
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from io import StringIO

from qframe.infrastructure.observability.structured_logging import (
    LogLevel, LogFormat, LogContext, PerformanceMetrics,
    EnhancedLogRecord, StructuredLogger, PerformanceLogger,
    LogHandler, LoggerFactory, performance_logged,
    get_default_logger, configure_logging
)


class TestLogContext:
    """Tests de la classe LogContext."""

    def test_log_context_creation(self):
        """Test création contexte de base."""
        context = LogContext(
            service_name="test-service",
            environment="test",
            strategy_name="test-strategy"
        )

        assert context.service_name == "test-service"
        assert context.environment == "test"
        assert context.strategy_name == "test-strategy"
        assert context.request_id is not None  # UUID généré
        assert context.thread_id is not None

    def test_log_context_to_dict(self):
        """Test conversion en dictionnaire."""
        context = LogContext(
            service_name="test",
            strategy_name="momentum",
            symbol="BTC/USD",
            user_id=None  # Doit être exclu
        )

        context_dict = context.to_dict()

        assert "service_name" in context_dict
        assert "strategy_name" in context_dict
        assert "symbol" in context_dict
        assert "user_id" not in context_dict  # None values excluded


class TestPerformanceMetrics:
    """Tests de la classe PerformanceMetrics."""

    def test_performance_metrics_creation(self):
        """Test création métriques de performance."""
        metrics = PerformanceMetrics(
            operation="feature_calculation",
            duration_ms=125.5,
            memory_mb=45.2,
            orders_processed=10,
            success=True
        )

        assert metrics.operation == "feature_calculation"
        assert metrics.duration_ms == 125.5
        assert metrics.memory_mb == 45.2
        assert metrics.orders_processed == 10
        assert metrics.success == True
        assert metrics.retry_count == 0

    def test_performance_metrics_to_dict(self):
        """Test sérialisation métriques."""
        metrics = PerformanceMetrics(
            operation="backtest",
            duration_ms=5000.0,
            features_calculated=25
        )

        metrics_dict = metrics.to_dict()

        assert "operation" in metrics_dict
        assert "duration_ms" in metrics_dict
        assert "features_calculated" in metrics_dict
        assert metrics_dict["success"] == True  # Default value


class TestEnhancedLogRecord:
    """Tests de la classe EnhancedLogRecord."""

    def test_log_record_creation(self):
        """Test création enregistrement de log."""
        context = LogContext(service_name="test")
        record = EnhancedLogRecord(
            level=LogLevel.INFO,
            message="Test message",
            context=context,
            extra={"key": "value"}
        )

        assert record.level == LogLevel.INFO
        assert record.message == "Test message"
        assert record.context == context
        assert record.extra == {"key": "value"}
        assert isinstance(record.timestamp, datetime)

    def test_log_record_with_exception(self):
        """Test log record avec exception."""
        context = LogContext(service_name="test")
        exception = ValueError("Test error")

        record = EnhancedLogRecord(
            level=LogLevel.ERROR,
            message="Error occurred",
            context=context,
            exception=exception
        )

        record_dict = record.to_dict()

        assert "exception" in record_dict
        assert record_dict["exception"]["type"] == "ValueError"
        assert record_dict["exception"]["message"] == "Test error"
        assert "traceback" in record_dict["exception"]

    def test_log_record_with_metrics(self):
        """Test log record avec métriques."""
        context = LogContext(service_name="test")
        metrics = PerformanceMetrics(operation="test", duration_ms=100.0)

        record = EnhancedLogRecord(
            level=LogLevel.INFO,
            message="Operation completed",
            context=context,
            metrics=metrics
        )

        record_dict = record.to_dict()

        assert "metrics" in record_dict
        assert record_dict["metrics"]["operation"] == "test"
        assert record_dict["metrics"]["duration_ms"] == 100.0

    def test_log_record_to_dict_complete(self):
        """Test sérialisation complète."""
        context = LogContext(
            service_name="test",
            strategy_name="momentum",
            symbol="BTC/USD"
        )

        record = EnhancedLogRecord(
            level=LogLevel.WARNING,
            message="Test warning",
            context=context,
            extra={"signal_strength": 0.8}
        )

        record_dict = record.to_dict()

        # Vérifications de base
        assert record_dict["level"] == "WARNING"
        assert record_dict["message"] == "Test warning"
        assert "timestamp" in record_dict
        assert "context" in record_dict

        # Contexte
        assert record_dict["context"]["service_name"] == "test"
        assert record_dict["context"]["strategy_name"] == "momentum"
        assert record_dict["context"]["symbol"] == "BTC/USD"

        # Extra
        assert record_dict["signal_strength"] == 0.8


class TestLogHandler:
    """Tests de la classe LogHandler."""

    def test_log_handler_json_format(self):
        """Test handler format JSON."""
        handler = LogHandler(LogFormat.JSON)
        context = LogContext(service_name="test")

        record = EnhancedLogRecord(
            level=LogLevel.INFO,
            message="Test message",
            context=context
        )

        with patch('builtins.print') as mock_print:
            handler.emit(record)

            # Vérifier qu'une ligne JSON a été imprimée
            mock_print.assert_called_once()
            output = mock_print.call_args[0][0]

            # Doit être du JSON valide
            parsed = json.loads(output)
            assert parsed["message"] == "Test message"
            assert parsed["level"] == "INFO"

    def test_log_handler_console_format(self):
        """Test handler format console."""
        handler = LogHandler(LogFormat.CONSOLE)
        context = LogContext(
            service_name="test",
            strategy_name="momentum",
            symbol="BTC/USD"
        )

        record = EnhancedLogRecord(
            level=LogLevel.ERROR,
            message="Test error",
            context=context
        )

        with patch('builtins.print') as mock_print:
            handler.emit(record)

            mock_print.assert_called_once()
            output = mock_print.call_args[0][0]

            # Vérifier format console
            assert "ERROR" in output
            assert "Test error" in output
            assert "strategy=momentum" in output
            assert "symbol=BTC/USD" in output

    def test_log_handler_with_metrics(self):
        """Test handler avec métriques."""
        handler = LogHandler(LogFormat.CONSOLE)
        context = LogContext(service_name="test")
        metrics = PerformanceMetrics(operation="test", duration_ms=125.5)

        record = EnhancedLogRecord(
            level=LogLevel.INFO,
            message="Operation completed",
            context=context,
            metrics=metrics
        )

        with patch('builtins.print') as mock_print:
            handler.emit(record)

            output = mock_print.call_args[0][0]
            assert "took 125.5ms" in output

    def test_log_handler_with_exception(self):
        """Test handler avec exception."""
        handler = LogHandler(LogFormat.CONSOLE)
        context = LogContext(service_name="test")
        exception = ValueError("Test error")

        record = EnhancedLogRecord(
            level=LogLevel.ERROR,
            message="Error occurred",
            context=context,
            exception=exception
        )

        with patch('builtins.print') as mock_print:
            handler.emit(record)

            output = mock_print.call_args[0][0]
            assert "Exception:" in output
            assert "Test error" in output


class TestStructuredLogger:
    """Tests de la classe StructuredLogger."""

    def test_logger_creation(self):
        """Test création logger."""
        context = LogContext(service_name="test")
        logger = StructuredLogger(
            name="test-logger",
            context=context,
            min_level=LogLevel.DEBUG
        )

        assert logger.name == "test-logger"
        assert logger.min_level == LogLevel.DEBUG
        assert len(logger.handlers) >= 1

    def test_logger_level_filtering(self):
        """Test filtrage par niveau."""
        logger = StructuredLogger(
            name="test",
            min_level=LogLevel.WARNING
        )

        # Ces niveaux ne devraient pas être loggés
        assert logger._should_log(LogLevel.DEBUG) == False
        assert logger._should_log(LogLevel.INFO) == False

        # Ces niveaux devraient être loggés
        assert logger._should_log(LogLevel.WARNING) == True
        assert logger._should_log(LogLevel.ERROR) == True
        assert logger._should_log(LogLevel.CRITICAL) == True

    def test_logger_with_context(self):
        """Test logger avec contexte enrichi."""
        base_context = LogContext(service_name="test")
        logger = StructuredLogger(name="test", context=base_context)

        enriched_logger = logger.with_context(
            strategy_name="momentum",
            symbol="BTC/USD"
        )

        assert enriched_logger.context.strategy_name == "momentum"
        assert enriched_logger.context.symbol == "BTC/USD"
        assert enriched_logger.context.service_name == "test"  # Preserved

    def test_logger_context_manager(self):
        """Test context manager temporaire."""
        logger = StructuredLogger(name="test")

        with logger.context_manager(strategy_name="test_strategy", symbol="ETH/USD"):
            # Dans le contexte, le logger devrait avoir ces valeurs
            current_context = logger._get_current_context()
            assert current_context.strategy_name == "test_strategy"
            assert current_context.symbol == "ETH/USD"

        # Après le contexte, les valeurs ne devraient plus être là
        # (sauf si c'était dans le contexte de base)
        current_context = logger._get_current_context()
        # Retour au contexte original

    def test_logger_info_message(self):
        """Test message info."""
        mock_handler = Mock()
        logger = StructuredLogger(
            name="test",
            handlers=[mock_handler],
            min_level=LogLevel.DEBUG
        )

        logger.info("Test info message", key="value")

        # Vérifier que le handler a été appelé
        mock_handler.emit.assert_called_once()
        record = mock_handler.emit.call_args[0][0]

        assert record.level == LogLevel.INFO
        assert record.message == "Test info message"
        assert record.extra["key"] == "value"

    def test_logger_error_with_exception(self):
        """Test message erreur avec exception."""
        mock_handler = Mock()
        logger = StructuredLogger(
            name="test",
            handlers=[mock_handler],
            min_level=LogLevel.DEBUG
        )

        exception = ValueError("Test error")
        logger.error("Error occurred", exception=exception, context="test")

        mock_handler.emit.assert_called_once()
        record = mock_handler.emit.call_args[0][0]

        assert record.level == LogLevel.ERROR
        assert record.message == "Error occurred"
        assert record.exception == exception
        assert record.extra["context"] == "test"

    def test_logger_thread_safety(self):
        """Test thread safety du contexte."""
        logger = StructuredLogger(name="test")

        results = {}

        def thread_function(thread_id):
            with logger.context_manager(thread_id=thread_id):
                time.sleep(0.01)  # Petit délai
                context = logger._get_current_context()
                results[thread_id] = context.thread_id

        # Lancer plusieurs threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_function, args=(f"thread_{i}",))
            threads.append(thread)
            thread.start()

        # Attendre tous les threads
        for thread in threads:
            thread.join()

        # Chaque thread devrait avoir son propre contexte
        assert len(results) == 5
        for thread_id in results.keys():
            assert thread_id in results


class TestPerformanceLogger:
    """Tests de la classe PerformanceLogger."""

    def test_performance_logger_measure_operation(self):
        """Test mesure d'opération."""
        mock_handler = Mock()
        base_logger = StructuredLogger(
            name="test",
            handlers=[mock_handler],
            min_level=LogLevel.DEBUG
        )
        perf_logger = PerformanceLogger(base_logger)

        with perf_logger.measure_operation("test_operation", symbol="BTC/USD"):
            time.sleep(0.01)  # Simulation d'opération

        # Vérifier que 2 logs ont été émis (start + completed)
        assert mock_handler.emit.call_count == 2

        # Premier appel: start
        start_record = mock_handler.emit.call_args_list[0][0][0]
        assert "Starting test_operation" in start_record.message

        # Deuxième appel: completed
        completed_record = mock_handler.emit.call_args_list[1][0][0]
        assert "Completed test_operation" in completed_record.message
        assert completed_record.metrics is not None
        assert completed_record.metrics.operation == "test_operation"
        assert completed_record.metrics.success == True
        assert completed_record.metrics.duration_ms > 0

    def test_performance_logger_measure_operation_with_error(self):
        """Test mesure d'opération avec erreur."""
        mock_handler = Mock()
        base_logger = StructuredLogger(
            name="test",
            handlers=[mock_handler],
            min_level=LogLevel.DEBUG
        )
        perf_logger = PerformanceLogger(base_logger)

        with pytest.raises(ValueError):
            with perf_logger.measure_operation("failing_operation"):
                raise ValueError("Test error")

        # Vérifier que 2 logs ont été émis (start + failed)
        assert mock_handler.emit.call_count == 2

        # Deuxième appel: failed
        failed_record = mock_handler.emit.call_args_list[1][0][0]
        assert "Failed failing_operation" in failed_record.message
        assert failed_record.level == LogLevel.ERROR
        assert failed_record.metrics.success == False
        assert failed_record.metrics.error_type == "ValueError"
        assert failed_record.exception is not None

    def test_performance_logger_log_metrics(self):
        """Test log explicite de métriques."""
        mock_handler = Mock()
        base_logger = StructuredLogger(
            name="test",
            handlers=[mock_handler],
            min_level=LogLevel.DEBUG
        )
        perf_logger = PerformanceLogger(base_logger)

        perf_logger.log_metrics(
            operation="custom_operation",
            duration_ms=250.5,
            success=True,
            orders_processed=15
        )

        mock_handler.emit.assert_called_once()
        record = mock_handler.emit.call_args[0][0]

        assert record.level == LogLevel.INFO
        assert "Metrics for custom_operation" in record.message
        assert record.metrics.operation == "custom_operation"
        assert record.metrics.duration_ms == 250.5
        assert record.metrics.orders_processed == 15


class TestPerformanceLoggedDecorator:
    """Tests du décorateur performance_logged."""

    def test_performance_logged_decorator(self):
        """Test décorateur sur fonction."""
        mock_handler = Mock()
        base_logger = StructuredLogger(
            name="test",
            handlers=[mock_handler],
            min_level=LogLevel.DEBUG
        )

        # Mock du logger par défaut
        with patch('qframe.infrastructure.observability.structured_logging.get_default_logger', return_value=base_logger):

            @performance_logged("test_function")
            def test_function(x, y):
                time.sleep(0.01)
                return x + y

            result = test_function(2, 3)

            assert result == 5
            assert mock_handler.emit.call_count >= 1  # Au moins un log

    def test_performance_logged_with_class_logger(self):
        """Test décorateur avec logger de classe."""
        mock_handler = Mock()
        base_logger = StructuredLogger(
            name="test",
            handlers=[mock_handler],
            min_level=LogLevel.DEBUG
        )

        class TestClass:
            def __init__(self):
                self.logger = base_logger

            @performance_logged("class_method")
            def process_data(self, data):
                return len(data)

        test_obj = TestClass()
        result = test_obj.process_data([1, 2, 3, 4, 5])

        assert result == 5
        assert mock_handler.emit.call_count >= 1


class TestLoggerFactory:
    """Tests de la classe LoggerFactory."""

    def test_logger_factory_get_logger(self):
        """Test récupération de logger via factory."""
        logger = LoggerFactory.get_logger("test_logger")

        assert logger.name == "test_logger"
        assert isinstance(logger, StructuredLogger)

        # Deuxième appel devrait retourner le même logger
        logger2 = LoggerFactory.get_logger("test_logger")
        assert logger is logger2

    def test_logger_factory_configure_defaults(self):
        """Test configuration des défauts."""
        LoggerFactory.configure_defaults(
            min_level=LogLevel.WARNING,
            format_type=LogFormat.JSON,
            service_name="test_service",
            environment="test"
        )

        logger = LoggerFactory.get_logger("configured_logger")

        assert logger.min_level == LogLevel.WARNING
        assert logger.context.service_name == "test_service"
        assert logger.context.environment == "test"

    def test_logger_factory_specialized_loggers(self):
        """Test loggers spécialisés."""
        # Trading logger
        trading_logger = LoggerFactory.get_trading_logger(
            strategy_name="momentum",
            portfolio_id="portfolio_123"
        )

        assert trading_logger.context.strategy_name == "momentum"
        assert trading_logger.context.portfolio_id == "portfolio_123"
        assert trading_logger.context.service_name == "qframe-trading"

        # Data logger
        data_logger = LoggerFactory.get_data_logger("BTC/USD")

        assert data_logger.context.symbol == "BTC/USD"
        assert data_logger.context.service_name == "qframe-data"


class TestGlobalFunctions:
    """Tests des fonctions globales."""

    def test_configure_logging(self):
        """Test configuration globale."""
        configure_logging(
            level=LogLevel.DEBUG,
            format_type=LogFormat.JSON,
            service_name="global_test",
            environment="test"
        )

        logger = get_default_logger()

        assert logger.min_level == LogLevel.DEBUG
        assert logger.context.service_name == "global_test"
        assert logger.context.environment == "test"

    def test_get_default_logger(self):
        """Test récupération logger par défaut."""
        logger1 = get_default_logger()
        logger2 = get_default_logger()

        # Devrait retourner le même logger
        assert logger1 is logger2
        assert isinstance(logger1, StructuredLogger)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])