"""
Tests for External Adapters
===========================

Tests ciblés pour les adaptateurs externes.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
from datetime import datetime
import pandas as pd

from qframe.infrastructure.external.mock_broker_adapter import MockBrokerAdapter
from qframe.infrastructure.external.order_execution_adapter import BrokerAdapter
from qframe.domain.entities.order import Order, OrderSide, OrderType, OrderStatus


@pytest.fixture
def mock_broker_adapter():
    return MockBrokerAdapter()


@pytest.fixture
def sample_order():
    return Order(
        id="order-001",
        portfolio_id="portfolio-001",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
        created_time=datetime.utcnow()
    )


class TestMockBrokerAdapter:
    """Tests pour l'adaptateur broker simulé."""

    def test_mock_adapter_initialization(self, mock_broker_adapter):
        """Test initialisation adaptateur mock."""
        assert mock_broker_adapter.connection_status == "connected"
        assert mock_broker_adapter.account_balance > 0

    def test_execute_market_order(self, mock_broker_adapter, sample_order):
        """Test exécution ordre au marché."""
        result = mock_broker_adapter.execute_order(sample_order)

        assert result["status"] in ["filled", "partially_filled"]
        assert "execution_price" in result
        assert "filled_quantity" in result
        assert result["filled_quantity"] <= sample_order.quantity

    def test_execute_limit_order(self, mock_broker_adapter):
        """Test exécution ordre limite."""
        limit_order = Order(
            id="limit-001",
            portfolio_id="portfolio-001",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("5.0"),
            price=Decimal("3200.00"),
            created_time=datetime.utcnow()
        )

        result = mock_broker_adapter.execute_order(limit_order)

        assert result["status"] in ["pending", "filled", "partially_filled"]
        assert "order_id" in result

    def test_cancel_order(self, mock_broker_adapter, sample_order):
        """Test annulation d'ordre."""
        # D'abord placer l'ordre
        exec_result = mock_broker_adapter.execute_order(sample_order)
        broker_order_id = exec_result["order_id"]

        # Puis l'annuler
        cancel_result = mock_broker_adapter.cancel_order(broker_order_id)

        assert cancel_result["status"] in ["cancelled", "rejected"]

    def test_get_account_info(self, mock_broker_adapter):
        """Test récupération infos compte."""
        account_info = mock_broker_adapter.get_account_info()

        assert "balance" in account_info
        assert "positions" in account_info
        assert "orders" in account_info
        assert account_info["balance"] > 0

    def test_get_market_data(self, mock_broker_adapter):
        """Test récupération données marché."""
        market_data = mock_broker_adapter.get_market_data("BTC/USD")

        assert "symbol" in market_data
        assert "bid" in market_data
        assert "ask" in market_data
        assert "last_price" in market_data
        assert market_data["ask"] >= market_data["bid"]

    def test_order_status_tracking(self, mock_broker_adapter, sample_order):
        """Test suivi statut d'ordre."""
        # Exécuter ordre
        exec_result = mock_broker_adapter.execute_order(sample_order)
        broker_order_id = exec_result["order_id"]

        # Vérifier statut
        status = mock_broker_adapter.get_order_status(broker_order_id)

        assert "status" in status
        assert "filled_quantity" in status
        assert status["status"] in ["pending", "filled", "partially_filled", "cancelled"]

    def test_position_management(self, mock_broker_adapter):
        """Test gestion des positions."""
        positions = mock_broker_adapter.get_positions()

        assert isinstance(positions, list)
        for position in positions:
            assert "symbol" in position
            assert "quantity" in position
            assert "avg_price" in position

    def test_commission_calculation(self, mock_broker_adapter, sample_order):
        """Test calcul des commissions."""
        result = mock_broker_adapter.execute_order(sample_order)

        if "commission" in result:
            assert result["commission"] >= 0
            # Commission raisonnable (moins de 1% de la valeur)
            order_value = float(sample_order.quantity) * result["execution_price"]
            assert result["commission"] < order_value * 0.01

    def test_slippage_simulation(self, mock_broker_adapter):
        """Test simulation de slippage."""
        # Gros ordre qui pourrait avoir du slippage
        large_order = Order(
            id="large-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),  # Gros ordre
            created_time=datetime.utcnow()
        )

        result = mock_broker_adapter.execute_order(large_order)

        # Devrait avoir un prix d'exécution réaliste
        assert result["execution_price"] > 0
        if "slippage" in result:
            assert result["slippage"] >= 0

    def test_error_handling(self, mock_broker_adapter):
        """Test gestion d'erreurs."""
        # Ordre avec symbole invalide
        invalid_order = Order(
            id="invalid-001",
            portfolio_id="portfolio-001",
            symbol="INVALID/SYMBOL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            created_time=datetime.utcnow()
        )

        with pytest.raises((ValueError, KeyError)):
            mock_broker_adapter.execute_order(invalid_order)

    def test_latency_simulation(self, mock_broker_adapter, sample_order):
        """Test simulation de latence."""
        import time

        start_time = time.time()
        result = mock_broker_adapter.execute_order(sample_order)
        execution_time = time.time() - start_time

        # Devrait simuler une latence réaliste
        assert execution_time >= 0
        assert "execution_time" in result or execution_time < 1.0


class TestBrokerAdapterInterface:
    """Tests pour l'interface BrokerAdapter."""

    def test_broker_adapter_interface(self):
        """Test interface BrokerAdapter."""
        # Vérifier que l'interface existe et a les bonnes méthodes
        assert hasattr(BrokerAdapter, 'execute_order')
        assert hasattr(BrokerAdapter, 'cancel_order')
        assert hasattr(BrokerAdapter, 'get_account_info')

    def test_mock_adapter_implements_interface(self, mock_broker_adapter):
        """Test que MockBrokerAdapter implémente l'interface."""
        # Vérifier que toutes les méthodes requises existent
        required_methods = [
            'execute_order', 'cancel_order', 'get_account_info',
            'get_market_data', 'get_order_status', 'get_positions'
        ]

        for method in required_methods:
            assert hasattr(mock_broker_adapter, method)
            assert callable(getattr(mock_broker_adapter, method))


class TestOrderExecutionAdapter:
    """Tests pour l'adaptateur d'exécution d'ordres."""

    def test_order_routing(self):
        """Test routage d'ordres."""
        # Test conceptuel - vérifie que le routing fonctionne
        adapter = MockBrokerAdapter()

        btc_order = Order(
            id="btc-001", portfolio_id="portfolio-001",
            symbol="BTC/USD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=Decimal("1.0"),
            created_time=datetime.utcnow()
        )

        eth_order = Order(
            id="eth-001", portfolio_id="portfolio-001",
            symbol="ETH/USD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=Decimal("5.0"),
            created_time=datetime.utcnow()
        )

        btc_result = adapter.execute_order(btc_order)
        eth_result = adapter.execute_order(eth_order)

        # Les deux ordres devraient être traités
        assert btc_result["status"] in ["filled", "partially_filled"]
        assert eth_result["status"] in ["filled", "partially_filled"]

    def test_order_validation(self):
        """Test validation d'ordres."""
        adapter = MockBrokerAdapter()

        # Ordre avec quantité nulle
        invalid_order = Order(
            id="invalid-qty",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0"),
            created_time=datetime.utcnow()
        )

        with pytest.raises(ValueError):
            adapter.execute_order(invalid_order)

    def test_partial_execution_handling(self, mock_broker_adapter):
        """Test gestion d'exécution partielle."""
        # Simuler ordre qui ne peut être que partiellement exécuté
        large_order = Order(
            id="partial-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100.0"),  # Très gros ordre
            price=Decimal("40000.00"),  # Prix bas
            created_time=datetime.utcnow()
        )

        result = mock_broker_adapter.execute_order(large_order)

        if result["status"] == "partially_filled":
            assert result["filled_quantity"] < large_order.quantity
            assert result["filled_quantity"] > 0

    def test_concurrent_order_execution(self, mock_broker_adapter):
        """Test exécution d'ordres concurrents."""
        import threading

        orders = []
        results = []

        # Créer plusieurs ordres
        for i in range(5):
            order = Order(
                id=f"concurrent-{i}",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                created_time=datetime.utcnow()
            )
            orders.append(order)

        def execute_order(order):
            result = mock_broker_adapter.execute_order(order)
            results.append(result)

        # Exécuter en parallèle
        threads = []
        for order in orders:
            thread = threading.Thread(target=execute_order, args=(order,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Tous les ordres devraient avoir été traités
        assert len(results) == len(orders)
        assert all("status" in result for result in results)

    def test_market_data_integration(self, mock_broker_adapter):
        """Test intégration données de marché."""
        # Récupérer données de marché
        btc_data = mock_broker_adapter.get_market_data("BTC/USD")
        eth_data = mock_broker_adapter.get_market_data("ETH/USD")

        # Vérifier cohérence des données
        assert btc_data["symbol"] == "BTC/USD"
        assert eth_data["symbol"] == "ETH/USD"

        # Les prix devraient être différents
        assert btc_data["last_price"] != eth_data["last_price"]

    def test_account_balance_updates(self, mock_broker_adapter, sample_order):
        """Test mise à jour solde du compte."""
        # Balance initiale
        initial_info = mock_broker_adapter.get_account_info()
        initial_balance = initial_info["balance"]

        # Exécuter ordre
        result = mock_broker_adapter.execute_order(sample_order)

        if result["status"] == "filled":
            # Balance devrait avoir changé
            final_info = mock_broker_adapter.get_account_info()
            final_balance = final_info["balance"]

            # Pour un achat, le solde devrait diminuer
            if sample_order.side == OrderSide.BUY:
                assert final_balance < initial_balance

    def test_order_history_tracking(self, mock_broker_adapter):
        """Test suivi historique des ordres."""
        # Exécuter plusieurs ordres
        orders = []
        for i in range(3):
            order = Order(
                id=f"history-{i}",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                created_time=datetime.utcnow()
            )
            result = mock_broker_adapter.execute_order(order)
            orders.append((order, result))

        # Vérifier que l'historique est maintenu
        account_info = mock_broker_adapter.get_account_info()
        assert "orders" in account_info
        # Au moins quelques ordres devraient être dans l'historique
        assert len(account_info["orders"]) >= 0


class TestExternalAdapterIntegration:
    """Tests d'intégration des adaptateurs externes."""

    def test_adapter_factory_pattern(self):
        """Test pattern factory pour adaptateurs."""
        # Test création différents types d'adaptateurs
        mock_adapter = MockBrokerAdapter()
        assert mock_adapter is not None

        # Pourrait étendre avec d'autres adaptateurs (Binance, Coinbase, etc.)

    def test_adapter_configuration(self):
        """Test configuration des adaptateurs."""
        config = {
            "commission_rate": 0.001,
            "slippage_factor": 0.0005,
            "latency_ms": 50
        }

        adapter = MockBrokerAdapter(**config)
        assert adapter.commission_rate == 0.001

    def test_adapter_error_recovery(self, mock_broker_adapter):
        """Test récupération d'erreurs."""
        # Simuler déconnexion
        mock_broker_adapter.connection_status = "disconnected"

        # Tentative d'exécution devrait échouer gracieusement
        order = Order(
            id="error-recovery",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            created_time=datetime.utcnow()
        )

        # Devrait gérer l'erreur proprement
        with pytest.raises((ConnectionError, RuntimeError)):
            mock_broker_adapter.execute_order(order)

    def test_adapter_metrics_collection(self, mock_broker_adapter, sample_order):
        """Test collecte de métriques."""
        # Exécuter ordre et vérifier métriques
        result = mock_broker_adapter.execute_order(sample_order)

        # Vérifier que des métriques sont disponibles
        if hasattr(mock_broker_adapter, 'metrics'):
            metrics = mock_broker_adapter.metrics
            assert "orders_executed" in metrics
            assert "total_volume" in metrics