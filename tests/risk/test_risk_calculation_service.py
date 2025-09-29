"""
Tests for Risk Calculation Service
==================================

Tests ciblés pour le service de calcul de risque.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

from qframe.domain.services.risk_calculation_service import RiskCalculationService
from qframe.domain.entities.portfolio import Portfolio, Position
from qframe.domain.entities.risk_assessment import RiskAssessment, RiskLevel


@pytest.fixture
def risk_service():
    return RiskCalculationService()


@pytest.fixture
def sample_portfolio():
    portfolio = Portfolio(
        id="portfolio-001",
        name="Test Portfolio",
        initial_capital=Decimal("100000.00")
    )

    # Ajouter des positions
    positions = [
        Position(
            symbol="BTC/USD",
            quantity=Decimal("2.0"),
            average_price=Decimal("45000.00"),
            current_price=Decimal("46000.00")
        ),
        Position(
            symbol="ETH/USD",
            quantity=Decimal("10.0"),
            average_price=Decimal("3000.00"),
            current_price=Decimal("3100.00")
        )
    ]
    portfolio.positions = positions
    return portfolio


@pytest.fixture
def sample_returns():
    """Returns historiques pour tests VaR."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = pd.Series(
        np.random.normal(0.001, 0.02, 252),  # Moyenne 0.1%, volatilité 2%
        index=dates
    )
    return returns


class TestRiskCalculationService:
    """Tests pour RiskCalculationService."""

    def test_calculate_var_95(self, risk_service, sample_returns):
        """Test calcul VaR 95%."""
        var_95 = risk_service.calculate_var(sample_returns, confidence_level=0.95)

        assert isinstance(var_95, Decimal)
        assert var_95 < 0  # VaR devrait être négatif

    def test_calculate_var_99(self, risk_service, sample_returns):
        """Test calcul VaR 99%."""
        var_99 = risk_service.calculate_var(sample_returns, confidence_level=0.99)
        var_95 = risk_service.calculate_var(sample_returns, confidence_level=0.95)

        assert var_99 < var_95  # VaR 99% plus extrême que VaR 95%

    def test_calculate_cvar(self, risk_service, sample_returns):
        """Test calcul CVaR (Expected Shortfall)."""
        cvar_95 = risk_service.calculate_cvar(sample_returns, confidence_level=0.95)
        var_95 = risk_service.calculate_var(sample_returns, confidence_level=0.95)

        assert isinstance(cvar_95, Decimal)
        assert cvar_95 < var_95  # CVaR plus extrême que VaR

    def test_calculate_portfolio_var(self, risk_service, sample_portfolio):
        """Test VaR au niveau portfolio."""
        # Mock des returns pour simplifier
        with pytest.raises(NotImplementedError):
            risk_service.calculate_portfolio_var(sample_portfolio)

    def test_calculate_maximum_drawdown(self, risk_service):
        """Test calcul drawdown maximum."""
        # Série de valeurs de portfolio
        portfolio_values = pd.Series([100000, 105000, 98000, 102000, 95000, 110000])

        max_dd = risk_service.calculate_maximum_drawdown(portfolio_values)

        assert isinstance(max_dd, Decimal)
        assert max_dd <= 0  # Drawdown négatif ou zéro

    def test_assess_portfolio_risk(self, risk_service, sample_portfolio):
        """Test évaluation globale du risque."""
        assessment = risk_service.assess_portfolio_risk(sample_portfolio)

        assert isinstance(assessment, RiskAssessment)
        assert assessment.portfolio_id == sample_portfolio.id
        assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_calculate_position_concentration(self, risk_service, sample_portfolio):
        """Test calcul concentration des positions."""
        concentration = risk_service.calculate_position_concentration(sample_portfolio)

        assert isinstance(concentration, dict)
        assert len(concentration) == len(sample_portfolio.positions)
        assert all(0 <= weight <= 1 for weight in concentration.values())
        assert abs(sum(concentration.values()) - 1.0) < 0.01  # Somme ~= 1

    def test_stress_test(self, risk_service, sample_portfolio):
        """Test stress testing."""
        # Scenarios de stress (baisse de -10%, -20%, -30%)
        stress_scenarios = [-0.1, -0.2, -0.3]

        stress_results = risk_service.stress_test(sample_portfolio, stress_scenarios)

        assert isinstance(stress_results, dict)
        assert len(stress_results) == len(stress_scenarios)
        assert all(isinstance(result, Decimal) for result in stress_results.values())

    def test_risk_limit_validation(self, risk_service):
        """Test validation des limites de risque."""
        # Test avec différents niveaux de risque
        assert risk_service.is_within_risk_limits(var_95=Decimal("-0.02"), limit=Decimal("0.05"))
        assert not risk_service.is_within_risk_limits(var_95=Decimal("-0.08"), limit=Decimal("0.05"))

    def test_correlation_matrix(self, risk_service):
        """Test calcul matrice de corrélation."""
        # Mock data pour 3 actifs
        returns_data = pd.DataFrame({
            'BTC': np.random.normal(0.001, 0.03, 100),
            'ETH': np.random.normal(0.002, 0.04, 100),
            'ADA': np.random.normal(0.001, 0.05, 100)
        })

        corr_matrix = risk_service.calculate_correlation_matrix(returns_data)

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonale = 1

    def test_beta_calculation(self, risk_service):
        """Test calcul du beta."""
        # Returns actif et benchmark
        asset_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 100))

        beta = risk_service.calculate_beta(asset_returns, benchmark_returns)

        assert isinstance(beta, Decimal)
        assert beta > 0  # Beta positif pour corrélation positive

    def test_risk_metrics_summary(self, risk_service, sample_portfolio):
        """Test résumé des métriques de risque."""
        summary = risk_service.get_risk_summary(sample_portfolio)

        assert isinstance(summary, dict)
        assert 'total_exposure' in summary
        assert 'concentration' in summary
        assert 'estimated_var' in summary