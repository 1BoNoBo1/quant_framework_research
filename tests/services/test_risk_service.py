"""
Tests for Risk Service
=====================

Suite de tests complète pour le service de gestion des risques.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

from qframe.api.services.risk_service import RiskService
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.entities.risk_assessment import RiskAssessment
from qframe.domain.repositories.portfolio_repository import PortfolioRepository
from qframe.domain.repositories.order_repository import OrderRepository
from qframe.domain.repositories.risk_assessment_repository import RiskAssessmentRepository
from qframe.domain.services.risk_calculation_service import RiskCalculationService
from qframe.core.interfaces import MetricsCollector


@pytest.fixture
def mock_portfolio_repository():
    """Repository de portfolios mocké."""
    return Mock(spec=PortfolioRepository)


@pytest.fixture
def mock_order_repository():
    """Repository d'ordres mocké."""
    return Mock(spec=OrderRepository)


@pytest.fixture
def mock_risk_assessment_repository():
    """Repository d'évaluations de risque mocké."""
    return Mock(spec=RiskAssessmentRepository)


@pytest.fixture
def mock_risk_calculation_service():
    """Service de calcul de risque mocké."""
    return Mock(spec=RiskCalculationService)


@pytest.fixture
def mock_metrics_collector():
    """Collecteur de métriques mocké."""
    return Mock(spec=MetricsCollector)


@pytest.fixture
def risk_service(mock_portfolio_repository, mock_order_repository, mock_risk_assessment_repository,
                mock_risk_calculation_service, mock_metrics_collector):
    """Service de risque pour les tests."""
    return RiskService(
        portfolio_repository=mock_portfolio_repository,
        order_repository=mock_order_repository,
        risk_assessment_repository=mock_risk_assessment_repository,
        risk_calculation_service=mock_risk_calculation_service,
        metrics_collector=mock_metrics_collector
    )


@pytest.fixture
def sample_portfolio():
    """Portfolio de test."""
    portfolio = Portfolio(
        id="portfolio-001",
        name="Test Portfolio",
        initial_capital=Decimal("100000.00"),
        base_currency="USD"
    )
    portfolio.current_value = Decimal("95000.00")
    portfolio.cash_balance = Decimal("20000.00")
    portfolio.positions = {
        "BTC/USD": Decimal("1.5"),
        "ETH/USD": Decimal("10.0"),
        "ADA/USD": Decimal("1000.0")
    }
    return portfolio


@pytest.fixture
def sample_price_data():
    """Données de prix historiques."""
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
    np.random.seed(42)  # Pour la reproductibilité

    # Simulation de prix avec volatilité réaliste
    returns_btc = np.random.normal(0.001, 0.04, 252)  # 4% volatilité journalière
    returns_eth = np.random.normal(0.0005, 0.05, 252)  # 5% volatilité
    returns_ada = np.random.normal(0.0, 0.06, 252)     # 6% volatilité

    prices_btc = 45000 * np.exp(np.cumsum(returns_btc))
    prices_eth = 3000 * np.exp(np.cumsum(returns_eth))
    prices_ada = 1.5 * np.exp(np.cumsum(returns_ada))

    return pd.DataFrame({
        "date": dates,
        "BTC/USD": prices_btc,
        "ETH/USD": prices_eth,
        "ADA/USD": prices_ada
    })


@pytest.fixture
def sample_risk_assessment():
    """Évaluation de risque d'exemple."""
    return RiskAssessment(
        id="risk-001",
        portfolio_id="portfolio-001",
        assessment_date=datetime.now(),
        var_95=Decimal("5000.00"),
        cvar_95=Decimal("7500.00"),
        max_drawdown=Decimal("0.15"),
        volatility=Decimal("0.25"),
        beta=Decimal("1.2"),
        sharpe_ratio=Decimal("1.5")
    )


class TestRiskServiceBasic:
    """Tests de base pour RiskService."""

    async def test_calculate_portfolio_var(self, risk_service, mock_risk_calculation_service, sample_portfolio, sample_price_data):
        """Test de calcul du VaR du portfolio."""
        # Arrange
        expected_var = Decimal("5000.00")
        mock_risk_calculation_service.calculate_var.return_value = expected_var

        # Act
        result = await risk_service.calculate_portfolio_var(
            portfolio_id="portfolio-001",
            confidence_level=0.95,
            time_horizon_days=1
        )

        # Assert
        assert result == expected_var
        mock_risk_calculation_service.calculate_var.assert_called_once()

    async def test_calculate_portfolio_cvar(self, risk_service, mock_risk_calculation_service, sample_portfolio):
        """Test de calcul du CVaR du portfolio."""
        # Arrange
        expected_cvar = Decimal("7500.00")
        mock_risk_calculation_service.calculate_cvar.return_value = expected_cvar

        # Act
        result = await risk_service.calculate_portfolio_cvar(
            portfolio_id="portfolio-001",
            confidence_level=0.95,
            time_horizon_days=1
        )

        # Assert
        assert result == expected_cvar
        mock_risk_calculation_service.calculate_cvar.assert_called_once()

    async def test_get_portfolio_risk_metrics(self, risk_service, mock_portfolio_repository,
                                            mock_risk_calculation_service, sample_portfolio):
        """Test de récupération des métriques de risque du portfolio."""
        # Arrange
        mock_portfolio_repository.get_by_id.return_value = sample_portfolio

        risk_metrics = {
            "var_95": Decimal("5000.00"),
            "cvar_95": Decimal("7500.00"),
            "volatility": Decimal("0.25"),
            "beta": Decimal("1.2"),
            "sharpe_ratio": Decimal("1.5"),
            "max_drawdown": Decimal("0.15"),
            "sortino_ratio": Decimal("1.8")
        }
        mock_risk_calculation_service.calculate_comprehensive_risk.return_value = risk_metrics

        # Act
        result = await risk_service.get_portfolio_risk_metrics("portfolio-001")

        # Assert
        assert result["var_95"] == Decimal("5000.00")
        assert result["cvar_95"] == Decimal("7500.00")
        assert result["sharpe_ratio"] == Decimal("1.5")
        assert "volatility" in result
        assert "beta" in result

    async def test_assess_order_risk(self, risk_service, mock_portfolio_repository,
                                   mock_risk_calculation_service, sample_portfolio):
        """Test d'évaluation du risque d'un ordre."""
        # Arrange
        mock_portfolio_repository.get_by_id.return_value = sample_portfolio

        order_data = {
            "symbol": "BTC/USD",
            "side": OrderSide.BUY,
            "quantity": Decimal("0.5"),
            "price": Decimal("45000.00")
        }

        risk_impact = {
            "position_risk_increase": Decimal("0.05"),  # 5% d'augmentation
            "var_impact": Decimal("500.00"),
            "concentration_risk": Decimal("0.15"),
            "margin_requirement": Decimal("22500.00"),
            "risk_score": 7.2
        }
        mock_risk_calculation_service.assess_order_risk.return_value = risk_impact

        # Act
        result = await risk_service.assess_order_risk("portfolio-001", **order_data)

        # Assert
        assert result["risk_score"] == 7.2
        assert result["var_impact"] == Decimal("500.00")
        assert result["concentration_risk"] == Decimal("0.15")


class TestRiskServiceLimits:
    """Tests de gestion des limites de risque."""

    async def test_set_risk_limits(self, risk_service, mock_portfolio_repository, sample_portfolio):
        """Test de configuration des limites de risque."""
        # Arrange
        mock_portfolio_repository.get_by_id.return_value = sample_portfolio
        mock_portfolio_repository.save.return_value = sample_portfolio

        limits = {
            "max_var_percentage": 0.05,        # 5% du capital
            "max_position_size": 0.20,         # 20% par position
            "max_sector_exposure": 0.30,       # 30% par secteur
            "max_daily_loss": 2000.00,         # $2000 par jour
            "min_cash_percentage": 0.10        # 10% en cash
        }

        # Act
        result = await risk_service.set_risk_limits("portfolio-001", limits)

        # Assert
        assert result["max_var_percentage"] == 0.05
        assert result["max_position_size"] == 0.20
        assert result["status"] == "ACTIVE"
        mock_portfolio_repository.save.assert_called_once()

    async def test_check_risk_limits_compliance(self, risk_service, mock_portfolio_repository,
                                              mock_risk_calculation_service, sample_portfolio):
        """Test de vérification de conformité aux limites."""
        # Arrange
        mock_portfolio_repository.get_by_id.return_value = sample_portfolio

        # Portfolio avec limites configurées
        sample_portfolio.risk_limits = {
            "max_var_percentage": 0.05,
            "max_position_size": 0.20,
            "max_daily_loss": 2000.00
        }

        current_metrics = {
            "var_95": Decimal("6000.00"),  # Dépasse 5% de 100k = 5000
            "max_position_percentage": Decimal("0.15"),  # OK
            "daily_pnl": Decimal("-2500.00")  # Dépasse la limite de perte
        }
        mock_risk_calculation_service.calculate_current_metrics.return_value = current_metrics

        # Act
        compliance = await risk_service.check_risk_limits_compliance("portfolio-001")

        # Assert
        assert compliance["var_limit_breached"] is True
        assert compliance["position_limit_breached"] is False
        assert compliance["daily_loss_limit_breached"] is True
        assert compliance["overall_compliance"] is False

    async def test_get_risk_alerts(self, risk_service, mock_portfolio_repository,
                                 mock_risk_calculation_service, sample_portfolio):
        """Test de génération d'alertes de risque."""
        # Arrange
        mock_portfolio_repository.get_by_id.return_value = sample_portfolio

        alerts_data = [
            {
                "type": "VAR_BREACH",
                "severity": "HIGH",
                "message": "VaR exceeds 5% limit",
                "current_value": Decimal("6000.00"),
                "limit_value": Decimal("5000.00"),
                "timestamp": datetime.now()
            },
            {
                "type": "CONCENTRATION_RISK",
                "severity": "MEDIUM",
                "message": "BTC position exceeds 15% of portfolio",
                "current_value": Decimal("0.18"),
                "limit_value": Decimal("0.15"),
                "timestamp": datetime.now()
            }
        ]
        mock_risk_calculation_service.generate_alerts.return_value = alerts_data

        # Act
        alerts = await risk_service.get_risk_alerts("portfolio-001")

        # Assert
        assert len(alerts) == 2
        assert alerts[0]["type"] == "VAR_BREACH"
        assert alerts[0]["severity"] == "HIGH"
        assert alerts[1]["type"] == "CONCENTRATION_RISK"


class TestRiskServiceMonitoring:
    """Tests de monitoring des risques."""

    async def test_real_time_risk_monitoring(self, risk_service, mock_portfolio_repository,
                                           mock_risk_calculation_service):
        """Test de monitoring en temps réel."""
        # Arrange
        portfolio_ids = ["portfolio-001", "portfolio-002"]

        risk_updates = {
            "portfolio-001": {
                "var_95": Decimal("5200.00"),
                "volatility": Decimal("0.26"),
                "risk_score": 7.8
            },
            "portfolio-002": {
                "var_95": Decimal("3100.00"),
                "volatility": Decimal("0.18"),
                "risk_score": 5.2
            }
        }
        mock_risk_calculation_service.calculate_real_time_risk.return_value = risk_updates

        # Act
        monitoring_data = await risk_service.start_real_time_monitoring(portfolio_ids)

        # Assert
        assert "portfolio-001" in monitoring_data
        assert "portfolio-002" in monitoring_data
        assert monitoring_data["portfolio-001"]["risk_score"] == 7.8

    async def test_risk_scenario_analysis(self, risk_service, mock_risk_calculation_service, sample_portfolio):
        """Test d'analyse de scénarios de risque."""
        # Arrange
        scenarios = [
            {"name": "Market Crash", "btc_change": -0.30, "eth_change": -0.35, "ada_change": -0.40},
            {"name": "Bull Market", "btc_change": 0.50, "eth_change": 0.60, "ada_change": 0.70},
            {"name": "Volatility Spike", "btc_vol_multiplier": 2.0, "correlation_increase": 0.3}
        ]

        scenario_results = {
            "Market Crash": {
                "portfolio_value_change": Decimal("-32500.00"),
                "var_change": Decimal("8500.00"),
                "worst_asset": "ADA/USD"
            },
            "Bull Market": {
                "portfolio_value_change": Decimal("52000.00"),
                "var_change": Decimal("-1200.00"),
                "best_asset": "ADA/USD"
            },
            "Volatility Spike": {
                "portfolio_value_change": Decimal("-2100.00"),
                "var_change": Decimal("3800.00"),
                "correlation_impact": Decimal("0.15")
            }
        }
        mock_risk_calculation_service.run_scenario_analysis.return_value = scenario_results

        # Act
        results = await risk_service.run_scenario_analysis("portfolio-001", scenarios)

        # Assert
        assert len(results) == 3
        assert results["Market Crash"]["portfolio_value_change"] == Decimal("-32500.00")
        assert results["Bull Market"]["portfolio_value_change"] == Decimal("52000.00")

    async def test_stress_testing(self, risk_service, mock_risk_calculation_service):
        """Test de stress testing."""
        # Arrange
        stress_parameters = {
            "market_shock_percentage": 0.20,  # Choc de marché de 20%
            "correlation_shock": 0.80,        # Corrélations montent à 80%
            "volatility_multiplier": 2.5,     # Volatilité x2.5
            "liquidity_shock": 0.30          # Réduction de liquidité de 30%
        }

        stress_results = {
            "worst_case_loss": Decimal("25000.00"),
            "time_to_liquidation_days": 3.2,
            "recovery_time_estimate_days": 45,
            "critical_assets": ["BTC/USD", "ETH/USD"],
            "stress_score": 8.7
        }
        mock_risk_calculation_service.run_stress_test.return_value = stress_results

        # Act
        results = await risk_service.run_stress_test("portfolio-001", stress_parameters)

        # Assert
        assert results["worst_case_loss"] == Decimal("25000.00")
        assert results["stress_score"] == 8.7
        assert "BTC/USD" in results["critical_assets"]


class TestRiskServiceReporting:
    """Tests de rapports de risque."""

    async def test_generate_risk_report(self, risk_service, mock_portfolio_repository,
                                      mock_risk_assessment_repository, sample_portfolio, sample_risk_assessment):
        """Test de génération de rapport de risque."""
        # Arrange
        mock_portfolio_repository.get_by_id.return_value = sample_portfolio
        mock_risk_assessment_repository.get_latest_by_portfolio.return_value = sample_risk_assessment

        # Act
        report = await risk_service.generate_risk_report("portfolio-001")

        # Assert
        assert report["portfolio_id"] == "portfolio-001"
        assert report["assessment_date"] is not None
        assert "risk_metrics" in report
        assert "position_analysis" in report
        assert "recommendations" in report

    async def test_get_risk_trend_analysis(self, risk_service, mock_risk_assessment_repository):
        """Test d'analyse des tendances de risque."""
        # Arrange
        historical_assessments = [
            RiskAssessment(
                id=f"risk-{i:03d}",
                portfolio_id="portfolio-001",
                assessment_date=datetime.now() - timedelta(days=i),
                var_95=Decimal(f"{5000 + i*100}.00"),
                volatility=Decimal(f"{0.25 + i*0.01}"),
                sharpe_ratio=Decimal(f"{1.5 - i*0.05}")
            )
            for i in range(30)  # 30 jours d'historique
        ]
        mock_risk_assessment_repository.get_by_portfolio_date_range.return_value = historical_assessments

        # Act
        trend_analysis = await risk_service.get_risk_trend_analysis(
            "portfolio-001",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )

        # Assert
        assert "var_trend" in trend_analysis
        assert "volatility_trend" in trend_analysis
        assert "sharpe_trend" in trend_analysis
        assert trend_analysis["period_days"] == 30

    async def test_benchmark_risk_comparison(self, risk_service, mock_risk_calculation_service):
        """Test de comparaison avec des benchmarks."""
        # Arrange
        portfolio_metrics = {
            "var_95": Decimal("5000.00"),
            "volatility": Decimal("0.25"),
            "sharpe_ratio": Decimal("1.5"),
            "beta": Decimal("1.2")
        }

        benchmark_metrics = {
            "SPY": {"volatility": Decimal("0.18"), "sharpe_ratio": Decimal("1.1")},
            "QQQ": {"volatility": Decimal("0.22"), "sharpe_ratio": Decimal("1.3")},
            "BTC": {"volatility": Decimal("0.45"), "sharpe_ratio": Decimal("0.8")}
        }

        comparison_result = {
            "relative_volatility": {
                "vs_SPY": Decimal("1.39"),  # 39% plus volatile
                "vs_QQQ": Decimal("1.14"),
                "vs_BTC": Decimal("0.56")
            },
            "relative_sharpe": {
                "vs_SPY": Decimal("1.36"),  # 36% meilleur Sharpe
                "vs_QQQ": Decimal("1.15"),
                "vs_BTC": Decimal("1.88")
            },
            "risk_adjusted_performance": "OUTPERFORMING"
        }

        mock_risk_calculation_service.compare_with_benchmarks.return_value = comparison_result

        # Act
        comparison = await risk_service.benchmark_risk_comparison("portfolio-001", ["SPY", "QQQ", "BTC"])

        # Assert
        assert comparison["relative_volatility"]["vs_SPY"] == Decimal("1.39")
        assert comparison["risk_adjusted_performance"] == "OUTPERFORMING"


class TestRiskServiceConfiguration:
    """Tests de configuration du service de risque."""

    async def test_configure_risk_models(self, risk_service):
        """Test de configuration des modèles de risque."""
        # Arrange
        model_config = {
            "var_model": "HISTORICAL_SIMULATION",
            "confidence_levels": [0.95, 0.99],
            "time_horizons": [1, 5, 10],  # jours
            "correlation_model": "EXPONENTIAL_WEIGHTED",
            "volatility_model": "GARCH",
            "lookback_period_days": 252
        }

        # Act
        result = await risk_service.configure_risk_models(model_config)

        # Assert
        assert result["status"] == "CONFIGURED"
        assert result["var_model"] == "HISTORICAL_SIMULATION"
        assert 0.95 in result["confidence_levels"]

    async def test_calibrate_risk_models(self, risk_service, mock_risk_calculation_service, sample_price_data):
        """Test de calibration des modèles de risque."""
        # Arrange
        calibration_data = sample_price_data

        calibration_results = {
            "model_accuracy": 0.94,
            "backtesting_score": 0.89,
            "coverage_ratio": 0.96,
            "calibration_date": datetime.now(),
            "next_calibration_date": datetime.now() + timedelta(days=30)
        }
        mock_risk_calculation_service.calibrate_models.return_value = calibration_results

        # Act
        results = await risk_service.calibrate_risk_models(calibration_data)

        # Assert
        assert results["model_accuracy"] == 0.94
        assert results["coverage_ratio"] == 0.96
        assert "next_calibration_date" in results


class TestRiskServicePerformance:
    """Tests de performance du service de risque."""

    async def test_bulk_risk_calculation(self, risk_service, mock_risk_calculation_service):
        """Test de calcul de risque en masse."""
        # Arrange
        portfolio_ids = [f"portfolio-{i:03d}" for i in range(100)]

        bulk_results = {
            portfolio_id: {
                "var_95": Decimal(f"{5000 + i*100}.00"),
                "volatility": Decimal(f"{0.20 + i*0.001}"),
                "risk_score": 5.0 + i*0.1
            }
            for i, portfolio_id in enumerate(portfolio_ids)
        }
        mock_risk_calculation_service.calculate_bulk_risk.return_value = bulk_results

        # Act
        start_time = datetime.now()
        results = await risk_service.calculate_bulk_portfolio_risk(portfolio_ids)
        processing_time = (datetime.now() - start_time).total_seconds()

        # Assert
        assert len(results) == 100
        assert processing_time < 5.0  # Doit traiter en moins de 5 secondes
        assert "portfolio-000" in results
        assert "portfolio-099" in results

    async def test_concurrent_risk_monitoring(self, risk_service, mock_risk_calculation_service):
        """Test de monitoring concurrent."""
        import asyncio

        # Arrange
        portfolio_ids = ["portfolio-001", "portfolio-002", "portfolio-003"]

        def mock_calculate(portfolio_id):
            return {
                "var_95": Decimal("5000.00"),
                "risk_score": 7.5,
                "timestamp": datetime.now()
            }

        mock_risk_calculation_service.calculate_real_time_risk.side_effect = mock_calculate

        # Act
        tasks = [
            risk_service.calculate_portfolio_risk_metrics(portfolio_id)
            for portfolio_id in portfolio_ids
        ]
        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 3
        assert all("var_95" in result for result in results)


class TestRiskServiceIntegration:
    """Tests d'intégration."""

    @pytest.mark.integration
    async def test_end_to_end_risk_assessment(self, risk_service, mock_portfolio_repository,
                                            mock_risk_calculation_service, sample_portfolio):
        """Test de processus complet d'évaluation de risque."""
        # Arrange
        mock_portfolio_repository.get_by_id.return_value = sample_portfolio

        # Métriques calculées
        comprehensive_risk = {
            "var_95": Decimal("5000.00"),
            "cvar_95": Decimal("7500.00"),
            "volatility": Decimal("0.25"),
            "beta": Decimal("1.2"),
            "sharpe_ratio": Decimal("1.5"),
            "max_drawdown": Decimal("0.15")
        }
        mock_risk_calculation_service.calculate_comprehensive_risk.return_value = comprehensive_risk

        # Conformité aux limites
        compliance_check = {
            "overall_compliance": True,
            "var_limit_breached": False,
            "position_limit_breached": False
        }
        mock_risk_calculation_service.check_compliance.return_value = compliance_check

        # Act
        # 1. Calculer les métriques
        metrics = await risk_service.get_portfolio_risk_metrics("portfolio-001")

        # 2. Vérifier la conformité
        compliance = await risk_service.check_risk_limits_compliance("portfolio-001")

        # 3. Générer le rapport
        report = await risk_service.generate_risk_report("portfolio-001")

        # Assert
        assert metrics["var_95"] == Decimal("5000.00")
        assert compliance["overall_compliance"] is True
        assert report["portfolio_id"] == "portfolio-001"
        assert "risk_metrics" in report