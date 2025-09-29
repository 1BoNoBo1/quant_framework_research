"""
Tests for Domain Value Objects
==============================

Tests rapides pour les value objects du domain.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence
from qframe.domain.value_objects.performance_metrics import PerformanceMetrics


class TestSignal:
    """Tests pour Signal value object."""

    def test_signal_creation(self):
        """Test création d'un signal."""
        signal = Signal(
            symbol="BTC/USD",
            action=SignalAction.BUY,
            timestamp=datetime.now(),
            strength=Decimal("0.8"),
            confidence=SignalConfidence.HIGH,
            price=Decimal("45000.00"),
            strategy_id="test_strategy"
        )

        assert signal.symbol == "BTC/USD"
        assert signal.action == SignalAction.BUY
        assert signal.strength == Decimal("0.8")
        assert signal.confidence == SignalConfidence.HIGH

    def test_signal_actions(self):
        """Test des actions de signal."""
        assert SignalAction.BUY
        assert SignalAction.SELL
        assert SignalAction.HOLD

    def test_signal_confidence_levels(self):
        """Test des niveaux de confiance."""
        assert SignalConfidence.LOW
        assert SignalConfidence.MEDIUM
        assert SignalConfidence.HIGH

    def test_signal_validation(self):
        """Test validation des signaux."""
        # Signal valide
        signal = Signal(
            symbol="BTC/USD",
            action=SignalAction.BUY,
            timestamp=datetime.now(),
            strength=Decimal("0.8")
        )
        assert signal.is_valid()

        # Signal avec strength invalide
        invalid_signal = Signal(
            symbol="BTC/USD",
            action=SignalAction.BUY,
            timestamp=datetime.now(),
            strength=Decimal("1.5")  # > 1.0
        )
        assert not invalid_signal.is_valid()


class TestPerformanceMetrics:
    """Tests pour PerformanceMetrics value object."""

    def test_metrics_creation(self):
        """Test création de métriques."""
        metrics = PerformanceMetrics(
            total_return=Decimal("0.15"),
            sharpe_ratio=Decimal("1.8"),
            max_drawdown=Decimal("-0.08"),
            volatility=Decimal("0.20"),
            win_rate=Decimal("0.65")
        )

        assert metrics.total_return == Decimal("0.15")
        assert metrics.sharpe_ratio == Decimal("1.8")
        assert metrics.max_drawdown == Decimal("-0.08")

    def test_metrics_comparison(self):
        """Test comparaison de métriques."""
        metrics1 = PerformanceMetrics(
            total_return=Decimal("0.15"),
            sharpe_ratio=Decimal("1.8")
        )

        metrics2 = PerformanceMetrics(
            total_return=Decimal("0.12"),
            sharpe_ratio=Decimal("1.5")
        )

        assert metrics1.is_better_than(metrics2)
        assert not metrics2.is_better_than(metrics1)

    def test_metrics_summary(self):
        """Test résumé des métriques."""
        metrics = PerformanceMetrics(
            total_return=Decimal("0.15"),
            sharpe_ratio=Decimal("1.8"),
            max_drawdown=Decimal("-0.08")
        )

        summary = metrics.get_summary()

        assert isinstance(summary, dict)
        assert 'total_return' in summary
        assert 'sharpe_ratio' in summary
        assert 'max_drawdown' in summary