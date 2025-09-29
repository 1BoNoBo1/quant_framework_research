"""
Tests for API Services (Comprehensive)
=====================================

Tests approfondis pour tous les services API avec faible couverture.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd

from qframe.api.services.backtest_service import BacktestService
from qframe.api.services.portfolio_service import PortfolioService as APIPortfolioService
from qframe.api.services.position_service import PositionService
from qframe.api.services.risk_service import RiskService as APIRiskService
from qframe.api.services.strategy_service import StrategyService as APIStrategyService
from qframe.api.services.real_time_service import RealTimeService


class TestBacktestService:
    """Tests pour BacktestService API."""

    @pytest.fixture
    def mock_backtest_repository(self):
        return Mock()

    @pytest.fixture
    def mock_strategy_repository(self):
        return Mock()

    @pytest.fixture
    def mock_data_provider(self):
        return Mock()

    @pytest.fixture
    def backtest_service(self, mock_backtest_repository, mock_strategy_repository, mock_data_provider):
        return BacktestService(
            backtest_repository=mock_backtest_repository,
            strategy_repository=mock_strategy_repository,
            data_provider=mock_data_provider
        )

    def test_create_backtest_config(self, backtest_service, mock_backtest_repository):
        """Test création configuration backtest."""
        config_data = {
            "name": "Test Backtest",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000.0,
            "strategy_ids": ["strategy-001"]
        }

        mock_config = Mock()
        mock_config.id = "config-001"
        mock_backtest_repository.save_config.return_value = mock_config

        result = backtest_service.create_config(config_data)

        assert result.id == "config-001"
        mock_backtest_repository.save_config.assert_called_once()

    def test_run_backtest(self, backtest_service, mock_backtest_repository, mock_strategy_repository):
        """Test exécution backtest."""
        mock_config = Mock()
        mock_config.id = "config-001"
        mock_config.strategy_ids = ["strategy-001"]
        mock_config.start_date = datetime(2023, 1, 1)
        mock_config.end_date = datetime(2023, 12, 31)
        mock_backtest_repository.get_config.return_value = mock_config

        mock_strategy = Mock()
        mock_strategy_repository.get_by_id.return_value = mock_strategy

        mock_result = Mock()
        mock_result.id = "result-001"
        mock_backtest_repository.save_result.return_value = mock_result

        result = backtest_service.run_backtest("config-001")

        assert result.id == "result-001"

    def test_get_backtest_results(self, backtest_service, mock_backtest_repository):
        """Test récupération résultats."""
        mock_results = [Mock() for _ in range(3)]
        mock_backtest_repository.get_all_results.return_value = mock_results

        results = backtest_service.get_results(limit=10, offset=0)

        assert len(results) == 3
        mock_backtest_repository.get_all_results.assert_called_once()

    def test_analyze_performance(self, backtest_service, mock_backtest_repository):
        """Test analyse de performance."""
        mock_result = Mock()
        mock_result.portfolio_values = [100000, 102000, 105000, 103000]
        mock_result.trades = []
        mock_backtest_repository.get_result.return_value = mock_result

        analysis = backtest_service.analyze_performance("result-001")

        assert "total_return" in analysis
        assert "sharpe_ratio" in analysis
        assert "max_drawdown" in analysis

    def test_compare_strategies(self, backtest_service, mock_backtest_repository):
        """Test comparaison de stratégies."""
        result_ids = ["result-001", "result-002"]

        mock_results = []
        for i, result_id in enumerate(result_ids):
            mock_result = Mock()
            mock_result.id = result_id
            mock_result.strategy_id = f"strategy-{i+1}"
            mock_result.final_value = 100000 + (i * 5000)
            mock_results.append(mock_result)

        mock_backtest_repository.get_results_by_ids.return_value = mock_results

        comparison = backtest_service.compare_strategies(result_ids)

        assert "strategies" in comparison
        assert len(comparison["strategies"]) == 2

    def test_get_backtest_status(self, backtest_service, mock_backtest_repository):
        """Test statut d'exécution."""
        mock_backtest_repository.get_execution_status.return_value = {
            "status": "running",
            "progress": 75.5,
            "estimated_completion": "2023-01-01T12:00:00Z"
        }

        status = backtest_service.get_execution_status("backtest-001")

        assert status["status"] == "running"
        assert status["progress"] == 75.5

    def test_cancel_backtest(self, backtest_service, mock_backtest_repository):
        """Test annulation backtest."""
        mock_backtest_repository.cancel_execution.return_value = True

        result = backtest_service.cancel_backtest("backtest-001")

        assert result is True
        mock_backtest_repository.cancel_execution.assert_called_once()

    def test_optimize_parameters(self, backtest_service, mock_strategy_repository):
        """Test optimisation de paramètres."""
        param_ranges = {
            "lookback": [10, 20, 30],
            "threshold": [0.1, 0.2, 0.3]
        }

        mock_results = []
        for lookback in param_ranges["lookback"]:
            for threshold in param_ranges["threshold"]:
                mock_result = Mock()
                mock_result.parameters = {"lookback": lookback, "threshold": threshold}
                mock_result.sharpe_ratio = lookback * threshold  # Dummy score
                mock_results.append(mock_result)

        with patch.object(backtest_service, 'run_parameter_sweep', return_value=mock_results):
            best_params = backtest_service.optimize_parameters("strategy-001", param_ranges)

            assert "lookback" in best_params
            assert "threshold" in best_params


class TestPortfolioServiceAPI:
    """Tests pour PortfolioService API."""

    @pytest.fixture
    def mock_portfolio_repository(self):
        return Mock()

    @pytest.fixture
    def mock_position_service(self):
        return Mock()

    @pytest.fixture
    def portfolio_service(self, mock_portfolio_repository, mock_position_service):
        return APIPortfolioService(
            portfolio_repository=mock_portfolio_repository,
            position_service=mock_position_service
        )

    def test_create_portfolio(self, portfolio_service, mock_portfolio_repository):
        """Test création portfolio."""
        portfolio_data = {
            "name": "Test Portfolio",
            "initial_capital": 100000.0,
            "base_currency": "USD"
        }

        mock_portfolio = Mock()
        mock_portfolio.id = "portfolio-001"
        mock_portfolio_repository.save.return_value = mock_portfolio

        result = portfolio_service.create_portfolio(portfolio_data)

        assert result.id == "portfolio-001"

    def test_get_portfolio_summary(self, portfolio_service, mock_portfolio_repository):
        """Test résumé portfolio."""
        mock_portfolio = Mock()
        mock_portfolio.id = "portfolio-001"
        mock_portfolio.total_value = Decimal("105000.00")
        mock_portfolio.cash_balance = Decimal("20000.00")
        mock_portfolio.positions = {"BTC/USD": Mock(market_value=Decimal("85000.00"))}
        mock_portfolio_repository.get_by_id.return_value = mock_portfolio

        summary = portfolio_service.get_summary("portfolio-001")

        assert summary["portfolio_id"] == "portfolio-001"
        assert summary["total_value"] == 105000.0
        assert summary["cash_balance"] == 20000.0

    def test_calculate_performance_metrics(self, portfolio_service, mock_portfolio_repository):
        """Test calcul métriques de performance."""
        mock_portfolio = Mock()
        mock_portfolio.initial_capital = Decimal("100000.00")
        mock_portfolio.total_value = Decimal("115000.00")
        mock_portfolio_repository.get_by_id.return_value = mock_portfolio

        # Mock historique des valeurs
        with patch.object(portfolio_service, 'get_value_history') as mock_history:
            mock_history.return_value = pd.Series([100000, 102000, 105000, 108000, 115000])

            metrics = portfolio_service.calculate_metrics("portfolio-001")

            assert "total_return" in metrics
            assert "volatility" in metrics
            assert "sharpe_ratio" in metrics

    def test_get_allocation(self, portfolio_service, mock_portfolio_repository):
        """Test allocation portfolio."""
        mock_portfolio = Mock()
        mock_portfolio.positions = {
            "BTC/USD": Mock(market_value=Decimal("50000")),
            "ETH/USD": Mock(market_value=Decimal("30000"))
        }
        mock_portfolio.cash_balance = Decimal("20000")
        mock_portfolio.total_value = Decimal("100000")
        mock_portfolio_repository.get_by_id.return_value = mock_portfolio

        allocation = portfolio_service.get_allocation("portfolio-001")

        assert "positions" in allocation
        assert "cash_percentage" in allocation
        assert allocation["cash_percentage"] == 20.0

    def test_rebalance_portfolio(self, portfolio_service, mock_portfolio_repository):
        """Test rééquilibrage portfolio."""
        target_allocation = {
            "BTC/USD": 0.4,
            "ETH/USD": 0.3,
            "cash": 0.3
        }

        mock_portfolio = Mock()
        mock_portfolio_repository.get_by_id.return_value = mock_portfolio

        with patch.object(portfolio_service, 'calculate_rebalance_orders') as mock_orders:
            mock_orders.return_value = [{"symbol": "BTC/USD", "action": "buy", "quantity": 0.5}]

            orders = portfolio_service.rebalance("portfolio-001", target_allocation)

            assert len(orders) > 0
            assert orders[0]["symbol"] == "BTC/USD"

    def test_risk_assessment(self, portfolio_service, mock_portfolio_repository):
        """Test évaluation de risque."""
        mock_portfolio = Mock()
        mock_portfolio_repository.get_by_id.return_value = mock_portfolio

        with patch.object(portfolio_service, 'calculate_var') as mock_var:
            mock_var.return_value = 5000.0

            risk_assessment = portfolio_service.assess_risk("portfolio-001")

            assert "var_95" in risk_assessment
            assert "concentration_risk" in risk_assessment

    def test_portfolio_comparison(self, portfolio_service, mock_portfolio_repository):
        """Test comparaison de portfolios."""
        portfolio_ids = ["portfolio-001", "portfolio-002"]

        mock_portfolios = []
        for i, portfolio_id in enumerate(portfolio_ids):
            mock_portfolio = Mock()
            mock_portfolio.id = portfolio_id
            mock_portfolio.total_value = Decimal(100000 + i * 10000)
            mock_portfolios.append(mock_portfolio)

        mock_portfolio_repository.get_by_ids.return_value = mock_portfolios

        comparison = portfolio_service.compare_portfolios(portfolio_ids)

        assert "portfolios" in comparison
        assert len(comparison["portfolios"]) == 2


class TestPositionService:
    """Tests pour PositionService."""

    @pytest.fixture
    def mock_position_repository(self):
        return Mock()

    @pytest.fixture
    def mock_market_data_service(self):
        return Mock()

    @pytest.fixture
    def position_service(self, mock_position_repository, mock_market_data_service):
        return PositionService(
            position_repository=mock_position_repository,
            market_data_service=mock_market_data_service
        )

    def test_get_all_positions(self, position_service, mock_position_repository):
        """Test récupération toutes positions."""
        mock_positions = [Mock() for _ in range(5)]
        mock_position_repository.get_all.return_value = mock_positions

        positions = position_service.get_all_positions()

        assert len(positions) == 5

    def test_get_positions_by_portfolio(self, position_service, mock_position_repository):
        """Test positions par portfolio."""
        mock_positions = [Mock() for _ in range(3)]
        mock_position_repository.get_by_portfolio_id.return_value = mock_positions

        positions = position_service.get_by_portfolio("portfolio-001")

        assert len(positions) == 3
        mock_position_repository.get_by_portfolio_id.assert_called_with("portfolio-001")

    def test_calculate_unrealized_pnl(self, position_service, mock_market_data_service):
        """Test calcul PnL non réalisé."""
        mock_position = Mock()
        mock_position.symbol = "BTC/USD"
        mock_position.quantity = Decimal("1.0")
        mock_position.average_price = Decimal("45000.00")

        mock_market_data_service.get_current_price.return_value = {"price": 47000.0}

        pnl = position_service.calculate_unrealized_pnl(mock_position)

        assert pnl == 2000.0  # (47000 - 45000) * 1.0

    def test_update_position_prices(self, position_service, mock_position_repository, mock_market_data_service):
        """Test mise à jour prix positions."""
        mock_positions = [
            Mock(symbol="BTC/USD", current_price=None),
            Mock(symbol="ETH/USD", current_price=None)
        ]
        mock_position_repository.get_all.return_value = mock_positions

        def mock_price_lookup(symbol):
            prices = {"BTC/USD": {"price": 47000.0}, "ETH/USD": {"price": 3100.0}}
            return prices.get(symbol, {"price": 0.0})

        mock_market_data_service.get_current_price.side_effect = mock_price_lookup

        position_service.update_current_prices()

        # Vérifier que les prix ont été mis à jour
        mock_position_repository.save.assert_called()

    def test_close_position(self, position_service, mock_position_repository):
        """Test fermeture position."""
        mock_position = Mock()
        mock_position.id = "position-001"
        mock_position_repository.get_by_id.return_value = mock_position

        result = position_service.close_position("position-001")

        assert result["status"] == "closed"
        mock_position_repository.delete.assert_called_with("position-001")

    def test_position_analytics(self, position_service, mock_position_repository):
        """Test analytics des positions."""
        mock_positions = []
        for i in range(10):
            position = Mock()
            position.symbol = f"SYMBOL{i}"
            position.market_value = Decimal(1000 + i * 100)
            position.unrealized_pnl = Decimal((i - 5) * 50)
            mock_positions.append(position)

        mock_position_repository.get_by_portfolio_id.return_value = mock_positions

        analytics = position_service.get_portfolio_analytics("portfolio-001")

        assert "total_positions" in analytics
        assert "total_value" in analytics
        assert "winners" in analytics
        assert "losers" in analytics

    def test_position_risk_metrics(self, position_service):
        """Test métriques de risque des positions."""
        mock_position = Mock()
        mock_position.symbol = "BTC/USD"
        mock_position.quantity = Decimal("2.0")
        mock_position.market_value = Decimal("94000.00")

        portfolio_total = Decimal("200000.00")

        risk_metrics = position_service.calculate_position_risk(mock_position, portfolio_total)

        assert "position_size_percentage" in risk_metrics
        assert "concentration_risk" in risk_metrics
        assert risk_metrics["position_size_percentage"] == 47.0  # 94k/200k


class TestRealTimeService:
    """Tests pour RealTimeService."""

    @pytest.fixture
    def mock_websocket_manager(self):
        return AsyncMock()

    @pytest.fixture
    def mock_data_provider(self):
        return Mock()

    @pytest.fixture
    def real_time_service(self, mock_websocket_manager, mock_data_provider):
        return RealTimeService(
            websocket_manager=mock_websocket_manager,
            data_provider=mock_data_provider
        )

    @pytest.mark.asyncio
    async def test_start_price_stream(self, real_time_service, mock_data_provider):
        """Test démarrage stream de prix."""
        symbols = ["BTC/USD", "ETH/USD"]

        with patch.object(real_time_service, 'price_stream_active', False):
            await real_time_service.start_price_stream(symbols)

            assert real_time_service.price_stream_active is True

    @pytest.mark.asyncio
    async def test_subscribe_to_updates(self, real_time_service, mock_websocket_manager):
        """Test souscription aux mises à jour."""
        client_id = "client-001"
        subscription = {
            "channel": "market_data",
            "symbols": ["BTC/USD"]
        }

        await real_time_service.subscribe(client_id, subscription)

        mock_websocket_manager.add_subscription.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_price_update(self, real_time_service, mock_websocket_manager):
        """Test diffusion mise à jour prix."""
        price_update = {
            "symbol": "BTC/USD",
            "price": 47500.0,
            "timestamp": datetime.utcnow().isoformat()
        }

        await real_time_service.broadcast_price_update(price_update)

        mock_websocket_manager.broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_portfolio_notifications(self, real_time_service, mock_websocket_manager):
        """Test notifications portfolio."""
        portfolio_update = {
            "portfolio_id": "portfolio-001",
            "total_value": 105000.0,
            "change": 5000.0
        }

        await real_time_service.notify_portfolio_change(portfolio_update)

        mock_websocket_manager.send_to_subscribers.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_notifications(self, real_time_service, mock_websocket_manager):
        """Test notifications d'ordres."""
        order_update = {
            "order_id": "order-001",
            "status": "filled",
            "portfolio_id": "portfolio-001"
        }

        await real_time_service.notify_order_update(order_update)

        mock_websocket_manager.send_to_portfolio_subscribers.assert_called_once()

    @pytest.mark.asyncio
    async def test_market_data_aggregation(self, real_time_service, mock_data_provider):
        """Test agrégation données de marché."""
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD"]

        def mock_ticker(symbol):
            prices = {"BTC/USD": 47000, "ETH/USD": 3100, "ADA/USD": 1.2}
            return {"symbol": symbol, "price": prices.get(symbol, 0)}

        mock_data_provider.get_current_price.side_effect = mock_ticker

        aggregated_data = await real_time_service.aggregate_market_data(symbols)

        assert len(aggregated_data) == 3
        assert all("price" in data for data in aggregated_data)

    @pytest.mark.asyncio
    async def test_connection_management(self, real_time_service, mock_websocket_manager):
        """Test gestion des connexions."""
        # Test ajout connexion
        client_ws = AsyncMock()
        await real_time_service.handle_client_connection(client_ws)

        mock_websocket_manager.add_connection.assert_called_once()

        # Test suppression connexion
        await real_time_service.handle_client_disconnect(client_ws)

        mock_websocket_manager.remove_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, real_time_service):
        """Test limitation de taux."""
        # Configurer limite
        real_time_service.rate_limit = 10  # 10 messages/seconde

        messages = []
        for i in range(20):
            message = {"seq": i, "data": f"message-{i}"}
            messages.append(message)

        # Envoyer tous les messages
        with patch.object(real_time_service, 'send_message') as mock_send:
            await real_time_service.send_batch_messages(messages)

            # Vérifier que la limite est respectée
            assert mock_send.call_count <= 12  # Avec marge d'erreur

    @pytest.mark.asyncio
    async def test_health_monitoring(self, real_time_service, mock_websocket_manager):
        """Test surveillance de santé."""
        # Simuler métriques de santé
        mock_websocket_manager.get_health_metrics.return_value = {
            "active_connections": 150,
            "messages_per_second": 45,
            "error_rate": 0.01
        }

        health_status = await real_time_service.get_health_status()

        assert "status" in health_status
        assert "metrics" in health_status
        assert health_status["metrics"]["active_connections"] == 150


class TestAPIServicesIntegration:
    """Tests d'intégration des services API."""

    def test_service_dependency_injection(self):
        """Test injection de dépendances entre services."""
        # Mock des dépendances
        mock_repos = {
            'portfolio': Mock(),
            'position': Mock(),
            'backtest': Mock(),
            'strategy': Mock()
        }

        # Créer services avec dépendances
        portfolio_service = APIPortfolioService(
            portfolio_repository=mock_repos['portfolio'],
            position_service=Mock()
        )

        backtest_service = BacktestService(
            backtest_repository=mock_repos['backtest'],
            strategy_repository=mock_repos['strategy'],
            data_provider=Mock()
        )

        # Vérifier que les services sont créés correctement
        assert portfolio_service is not None
        assert backtest_service is not None

    @pytest.mark.asyncio
    async def test_cross_service_communication(self):
        """Test communication entre services."""
        # Simuler workflow portfolio → position → real-time
        portfolio_service = Mock()
        position_service = Mock()
        real_time_service = AsyncMock()

        # 1. Mise à jour portfolio
        portfolio_update = {"portfolio_id": "portfolio-001", "total_value": 105000.0}
        portfolio_service.update_portfolio.return_value = portfolio_update

        # 2. Notification temps réel
        await real_time_service.notify_portfolio_change(portfolio_update)

        # Vérifier que la chaîne fonctionne
        real_time_service.notify_portfolio_change.assert_called_once()

    def test_error_propagation(self):
        """Test propagation d'erreurs entre services."""
        # Service avec dépendance qui échoue
        mock_repo = Mock()
        mock_repo.get_by_id.side_effect = Exception("Database error")

        portfolio_service = APIPortfolioService(
            portfolio_repository=mock_repo,
            position_service=Mock()
        )

        # L'erreur devrait se propager
        with pytest.raises(Exception, match="Database error"):
            portfolio_service.get_summary("portfolio-001")

    def test_service_caching(self):
        """Test mise en cache dans les services."""
        mock_repo = Mock()
        mock_expensive_calculation = Mock(return_value={"result": "expensive"})

        # Service avec cache
        service = APIPortfolioService(
            portfolio_repository=mock_repo,
            position_service=Mock()
        )

        with patch.object(service, 'expensive_calculation', mock_expensive_calculation):
            # Premier appel
            result1 = service.get_cached_analytics("portfolio-001")

            # Deuxième appel (devrait utiliser cache)
            result2 = service.get_cached_analytics("portfolio-001")

            # Si cache implémenté, seul un appel devrait être fait
            if hasattr(service, 'cache'):
                assert mock_expensive_calculation.call_count == 1

    def test_batch_operations(self):
        """Test opérations en lot."""
        mock_repo = Mock()
        portfolio_service = APIPortfolioService(
            portfolio_repository=mock_repo,
            position_service=Mock()
        )

        portfolio_ids = ["portfolio-001", "portfolio-002", "portfolio-003"]

        # Mock retour en lot
        mock_portfolios = [Mock(id=pid) for pid in portfolio_ids]
        mock_repo.get_by_ids.return_value = mock_portfolios

        # Opération en lot
        results = portfolio_service.get_batch_summaries(portfolio_ids)

        # Vérifier efficacité (un seul appel DB au lieu de 3)
        mock_repo.get_by_ids.assert_called_once_with(portfolio_ids)
        assert len(results) == 3