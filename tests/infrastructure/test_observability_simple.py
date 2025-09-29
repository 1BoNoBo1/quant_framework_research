"""
Tests for Observability Infrastructure (Simplified)
==================================================

Suite de tests simplifiée pour les composants d'observabilité critiques.
Teste uniquement les métriques qui sont essentielles.
"""

import pytest
import json
import logging
from unittest.mock import Mock, patch
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any

from qframe.infrastructure.observability.metrics import MetricsCollector


@pytest.fixture
def metrics_collector():
    """Collecteur de métriques pour les tests."""
    return MetricsCollector()


class TestMetricsCollector:
    """Tests pour MetricsCollector."""

    def test_initialization(self, metrics_collector):
        """Test d'initialisation du collecteur."""
        assert metrics_collector.metrics == {}
        assert metrics_collector.counters == {}
        assert metrics_collector.histograms == {}

    def test_increment_counter(self, metrics_collector):
        """Test d'incrémentation de compteur."""
        # Act
        metrics_collector.increment("orders.created")
        metrics_collector.increment("orders.created")
        metrics_collector.increment("orders.created", value=3)

        # Assert
        assert metrics_collector.counters["orders.created"] == 5

    def test_increment_with_tags(self, metrics_collector):
        """Test d'incrémentation avec tags."""
        # Act
        metrics_collector.increment("orders.created", tags={"symbol": "BTC/USD"})
        metrics_collector.increment("orders.created", tags={"symbol": "ETH/USD"})

        # Assert
        assert "orders.created" in metrics_collector.counters

    def test_gauge_metric(self, metrics_collector):
        """Test de métrique gauge."""
        # Act
        metrics_collector.gauge("portfolio.value", 125000.50)
        metrics_collector.gauge("portfolio.value", 126000.75)

        # Assert
        assert metrics_collector.metrics["portfolio.value"] == 126000.75

    def test_histogram_metric(self, metrics_collector):
        """Test de métrique histogram."""
        # Act
        metrics_collector.histogram("request.duration", 150.5)
        metrics_collector.histogram("request.duration", 200.3)
        metrics_collector.histogram("request.duration", 95.7)

        # Assert
        assert "request.duration" in metrics_collector.histograms
        assert len(metrics_collector.histograms["request.duration"]) == 3

    def test_timing_context_manager(self, metrics_collector):
        """Test du context manager de timing."""
        # Act
        with metrics_collector.timer("database.query"):
            # Simuler une opération
            import time
            time.sleep(0.01)

        # Assert
        assert "database.query" in metrics_collector.histograms
        assert len(metrics_collector.histograms["database.query"]) == 1
        assert metrics_collector.histograms["database.query"][0] >= 10  # Au moins 10ms

    def test_custom_metric(self, metrics_collector):
        """Test de métrique personnalisée."""
        # Act
        metrics_collector.custom_metric(
            "sharpe_ratio",
            1.5,
            metric_type="gauge",
            tags={"strategy": "mean_reversion"}
        )

        # Assert
        assert metrics_collector.metrics["sharpe_ratio"] == 1.5

    def test_get_metrics_summary(self, metrics_collector):
        """Test de récupération du résumé des métriques."""
        # Arrange
        metrics_collector.increment("orders.total", value=10)
        metrics_collector.gauge("portfolio.value", 100000.0)
        metrics_collector.histogram("latency", 50.0)

        # Act
        summary = metrics_collector.get_metrics_summary()

        # Assert
        assert "counters" in summary
        assert "gauges" in summary
        assert "histograms" in summary
        assert summary["counters"]["orders.total"] == 10
        assert summary["gauges"]["portfolio.value"] == 100000.0

    def test_reset_metrics(self, metrics_collector):
        """Test de reset des métriques."""
        # Arrange
        metrics_collector.increment("test.counter")
        metrics_collector.gauge("test.gauge", 123.45)

        # Act
        metrics_collector.reset()

        # Assert
        assert metrics_collector.counters == {}
        assert metrics_collector.metrics == {}
        assert metrics_collector.histograms == {}


class TestMetricsIntegration:
    """Tests d'intégration des métriques."""

    def test_trading_metrics_workflow(self, metrics_collector):
        """Test du workflow de métriques de trading."""
        # Act - Simuler un workflow de trading
        metrics_collector.increment("orders.received")

        with metrics_collector.timer("order.validation"):
            import time
            time.sleep(0.005)  # Simuler validation

        metrics_collector.increment("orders.validated")

        with metrics_collector.timer("order.execution"):
            import time
            time.sleep(0.01)  # Simuler exécution

        metrics_collector.increment("orders.executed")
        metrics_collector.gauge("portfolio.total_value", 125000.0)
        metrics_collector.gauge("portfolio.unrealized_pnl", 2500.0)

        # Assert
        summary = metrics_collector.get_metrics_summary()
        assert summary["counters"]["orders.received"] == 1
        assert summary["counters"]["orders.validated"] == 1
        assert summary["counters"]["orders.executed"] == 1
        assert summary["gauges"]["portfolio.total_value"] == 125000.0
        assert "order.validation" in summary["histograms"]
        assert "order.execution" in summary["histograms"]

    def test_risk_monitoring_metrics(self, metrics_collector):
        """Test des métriques de monitoring de risque."""
        # Act - Simuler le monitoring de risque
        metrics_collector.gauge("risk.var_95", 5000.0)
        metrics_collector.gauge("risk.cvar_95", 7500.0)
        metrics_collector.gauge("risk.max_drawdown", 0.15)
        metrics_collector.gauge("risk.portfolio_leverage", 1.25)

        # Simuler des alertes de risque
        metrics_collector.increment("risk.alerts.high_var")
        metrics_collector.increment("risk.alerts.position_limit")

        # Assert
        summary = metrics_collector.get_metrics_summary()
        assert summary["gauges"]["risk.var_95"] == 5000.0
        assert summary["gauges"]["risk.max_drawdown"] == 0.15
        assert summary["counters"]["risk.alerts.high_var"] == 1
        assert summary["counters"]["risk.alerts.position_limit"] == 1

    def test_performance_monitoring_metrics(self, metrics_collector):
        """Test des métriques de monitoring de performance."""
        # Act - Simuler des métriques de performance
        strategy_metrics = [
            ("strategy.mean_reversion.sharpe_ratio", 1.8),
            ("strategy.mean_reversion.total_return", 0.15),
            ("strategy.mean_reversion.max_drawdown", -0.08),
            ("strategy.dmn_lstm.sharpe_ratio", 2.1),
            ("strategy.dmn_lstm.total_return", 0.22),
            ("strategy.dmn_lstm.max_drawdown", -0.12)
        ]

        for metric_name, value in strategy_metrics:
            metrics_collector.gauge(metric_name, value)

        # Métriques de trades
        metrics_collector.increment("trades.total", value=150)
        metrics_collector.increment("trades.winning", value=92)
        metrics_collector.increment("trades.losing", value=58)

        # Assert
        summary = metrics_collector.get_metrics_summary()
        assert summary["gauges"]["strategy.mean_reversion.sharpe_ratio"] == 1.8
        assert summary["gauges"]["strategy.dmn_lstm.sharpe_ratio"] == 2.1
        assert summary["counters"]["trades.total"] == 150
        assert summary["counters"]["trades.winning"] == 92

    def test_system_health_metrics(self, metrics_collector):
        """Test des métriques de santé système."""
        # Act - Simuler des métriques système
        metrics_collector.gauge("system.cpu_usage", 75.5)
        metrics_collector.gauge("system.memory_usage", 68.2)
        metrics_collector.gauge("system.disk_usage", 82.1)

        # Métriques de connectivité
        metrics_collector.increment("connections.database.success")
        metrics_collector.increment("connections.broker.success")
        metrics_collector.increment("connections.data_provider.error")

        # Latences
        metrics_collector.histogram("latency.database_query", 25.5)
        metrics_collector.histogram("latency.api_request", 150.2)
        metrics_collector.histogram("latency.market_data", 5.8)

        # Assert
        summary = metrics_collector.get_metrics_summary()
        assert summary["gauges"]["system.cpu_usage"] == 75.5
        assert summary["counters"]["connections.database.success"] == 1
        assert summary["counters"]["connections.data_provider.error"] == 1
        assert len(summary["histograms"]["latency.database_query"]) == 1

    def test_concurrent_metrics_collection(self, metrics_collector):
        """Test de collecte de métriques concurrente."""
        import threading
        import time

        def worker_thread(thread_id):
            for i in range(10):
                metrics_collector.increment(f"thread.{thread_id}.operations")
                metrics_collector.gauge(f"thread.{thread_id}.last_value", i)
                time.sleep(0.001)

        # Act - Lancer plusieurs threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Assert
        summary = metrics_collector.get_metrics_summary()
        for i in range(5):
            assert summary["counters"][f"thread.{i}.operations"] == 10
            assert summary["gauges"][f"thread.{i}.last_value"] == 9

    def test_metrics_with_business_context(self, metrics_collector):
        """Test de métriques avec contexte business."""
        # Act - Simuler des métriques avec contexte métier
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD"]

        for symbol in symbols:
            # Volume de trading par symbole
            volume = 1000 if symbol == "BTC/USD" else 500
            metrics_collector.gauge(f"trading.volume.{symbol.replace('/', '_')}", volume)

            # Spread par symbole
            spread = 0.01 if symbol == "BTC/USD" else 0.02
            metrics_collector.gauge(f"market.spread.{symbol.replace('/', '_')}", spread)

            # Nombre de trades par symbole
            trades_count = 25 if symbol == "BTC/USD" else 15
            metrics_collector.increment(f"trades.count.{symbol.replace('/', '_')}", value=trades_count)

        # Métriques globales
        metrics_collector.gauge("portfolio.total_exposure", 0.85)
        metrics_collector.gauge("portfolio.cash_ratio", 0.15)

        # Assert
        summary = metrics_collector.get_metrics_summary()
        assert summary["gauges"]["trading.volume.BTC_USD"] == 1000
        assert summary["gauges"]["market.spread.ETH_USD"] == 0.02
        assert summary["counters"]["trades.count.ADA_USD"] == 15
        assert summary["gauges"]["portfolio.total_exposure"] == 0.85


class TestMetricsPerformance:
    """Tests de performance des métriques."""

    def test_high_volume_metrics(self, metrics_collector):
        """Test de gestion de gros volumes de métriques."""
        import time

        # Act - Générer beaucoup de métriques
        start_time = time.time()

        for i in range(1000):
            metrics_collector.increment("high_volume.counter")
            metrics_collector.gauge(f"high_volume.gauge_{i % 10}", i * 1.5)
            metrics_collector.histogram("high_volume.histogram", i % 100)

        processing_time = time.time() - start_time

        # Assert
        assert metrics_collector.counters["high_volume.counter"] == 1000
        assert len(metrics_collector.metrics) >= 10  # Au moins 10 gauges différentes
        assert len(metrics_collector.histograms["high_volume.histogram"]) == 1000
        assert processing_time < 1.0  # Doit traiter en moins d'1 seconde

    def test_memory_efficiency(self, metrics_collector):
        """Test d'efficacité mémoire."""
        import sys

        # Mesurer la mémoire avant
        initial_size = sys.getsizeof(metrics_collector.__dict__)

        # Act - Ajouter de nombreuses métriques
        for i in range(100):
            metrics_collector.increment(f"memory_test.counter_{i}")
            metrics_collector.gauge(f"memory_test.gauge_{i}", i * 2.5)

        # Mesurer la mémoire après
        final_size = sys.getsizeof(metrics_collector.__dict__)

        # Assert - L'augmentation de mémoire doit être raisonnable
        memory_increase = final_size - initial_size
        assert memory_increase < 50000  # Moins de 50KB d'augmentation