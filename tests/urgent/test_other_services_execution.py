"""
Tests d'Exécution Réelle - Autres Services Domain
=================================================

Tests qui EXÉCUTENT vraiment le code pour backtesting_service, risk_calculation_service, signal_service
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock

# Backtesting Service
from qframe.domain.services.backtesting_service import BacktestingService, MarketDataPoint
from qframe.domain.entities.backtest import (
    BacktestConfiguration, BacktestResult, BacktestMetrics, BacktestStatus,
    BacktestType, TradeExecution
)

# Risk Calculation Service
from qframe.domain.services.risk_calculation_service import (
    RiskCalculationService, RiskCalculationParams, MarketData
)

# Signal Service
from qframe.domain.services.signal_service import SignalService
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence

# Entities
from qframe.domain.entities.portfolio import Portfolio, Position, PortfolioSnapshot
from qframe.domain.entities.order import Order, OrderSide, OrderType


class TestMarketDataPointExecution:
    """Tests d'exécution réelle pour MarketDataPoint."""

    def test_market_data_point_creation_execution(self):
        """Test création point de données de marché."""
        # Exécuter création avec données OHLCV réelles
        data_point = MarketDataPoint(
            timestamp=datetime.utcnow(),
            symbol="BTC/USD",
            open_price=Decimal("49800"),
            high_price=Decimal("50200"),
            low_price=Decimal("49600"),
            close_price=Decimal("50000"),
            volume=Decimal("1250.5"),
            metadata={"source": "binance", "quality": "good"}
        )

        # Vérifier tous les champs
        assert data_point.symbol == "BTC/USD"
        assert data_point.open_price == Decimal("49800")
        assert data_point.high_price == Decimal("50200")
        assert data_point.low_price == Decimal("49600")
        assert data_point.close_price == Decimal("50000")
        assert data_point.volume == Decimal("1250.5")
        assert data_point.metadata["source"] == "binance"

        # Vérifier logique métier (OHLC cohérent)
        assert data_point.low_price <= data_point.open_price <= data_point.high_price
        assert data_point.low_price <= data_point.close_price <= data_point.high_price


class TestBacktestingServiceExecution:
    """Tests d'exécution réelle pour BacktestingService."""

    @pytest.fixture
    def mock_repositories(self):
        """Repositories mockés."""
        return {
            "backtest_repo": AsyncMock(),
            "strategy_repo": AsyncMock(),
            "portfolio_repo": AsyncMock()
        }

    @pytest.fixture
    def backtesting_service(self, mock_repositories):
        """Service de backtesting avec repos mockés."""
        return BacktestingService(
            backtest_repository=mock_repositories["backtest_repo"],
            strategy_repository=mock_repositories["strategy_repo"],
            portfolio_repository=mock_repositories["portfolio_repo"]
        )

    @pytest.fixture
    def sample_backtest_config(self):
        """Configuration de backtest réaliste."""
        return BacktestConfiguration(
            name="Test Backtest BTC Strategy",
            strategy_ids=["strategy-001", "strategy-002"],
            symbols=["BTC/USD", "ETH/USD"],
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow() - timedelta(days=1),
            initial_capital=Decimal("100000"),
            backtest_type=BacktestType.SINGLE_PERIOD,
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005")
        )

    @pytest.mark.asyncio
    async def test_backtesting_service_initialization_execution(self, backtesting_service, mock_repositories):
        """Test initialisation du service."""
        # Vérifier injection des dépendances
        assert backtesting_service.backtest_repository == mock_repositories["backtest_repo"]
        assert backtesting_service.strategy_repository == mock_repositories["strategy_repo"]
        assert backtesting_service.portfolio_repository == mock_repositories["portfolio_repo"]

    @pytest.mark.asyncio
    async def test_run_backtest_invalid_config_execution(self, backtesting_service):
        """Test backtest avec configuration invalide."""
        # Configuration invalide (dates inversées)
        invalid_config = BacktestConfiguration(
            name="Invalid Backtest",
            strategy_ids=[],  # Pas de stratégies
            symbols=["BTC/USD"],
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() - timedelta(days=30),  # Date fin avant début
            initial_capital=Decimal("0"),  # Capital nul
            backtest_type=BacktestType.SINGLE_PERIOD
        )

        # Exécuter backtest
        result = await backtesting_service.run_backtest(invalid_config)

        # Vérifier échec avec erreurs de validation
        assert isinstance(result, BacktestResult)
        assert result.status == BacktestStatus.FAILED
        assert "Configuration invalid" in result.error_message
        assert result.configuration_id == invalid_config.id

    @pytest.mark.asyncio
    async def test_run_backtest_no_strategies_execution(self, backtesting_service, sample_backtest_config, mock_repositories):
        """Test backtest sans stratégies valides."""
        # Mock repository pour ne retourner aucune stratégie
        mock_repositories["strategy_repo"].find_by_id.return_value = None

        # Exécuter backtest
        result = await backtesting_service.run_backtest(sample_backtest_config)

        # Vérifier échec
        assert result.status == BacktestStatus.FAILED
        assert "No valid strategies found" in result.error_message

    @pytest.mark.asyncio
    async def test_generate_market_data_execution(self, backtesting_service):
        """Test génération de données de marché."""
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow() - timedelta(days=1)

        # Exécuter génération (méthode privée testée directement)
        market_data = await backtesting_service._generate_market_data(start_date, end_date)

        # Vérifier données générées
        assert isinstance(market_data, list)
        assert len(market_data) > 0

        # Vérifier structure des données
        for data_point in market_data:
            assert isinstance(data_point, MarketDataPoint)
            assert data_point.timestamp >= start_date
            assert data_point.timestamp <= end_date
            assert data_point.open_price > 0
            assert data_point.volume > 0


class TestRiskCalculationServiceExecution:
    """Tests d'exécution réelle pour RiskCalculationService."""

    @pytest.fixture
    def risk_service(self):
        """Service de calcul de risque."""
        params = RiskCalculationParams(confidence_level=0.95)
        return RiskCalculationService(params)

    @pytest.fixture
    def sample_market_data(self):
        """Données de marché de test."""
        return MarketData(
            symbol="BTC/USD",
            prices=[
                Decimal("48000"), Decimal("49000"), Decimal("47000"),
                Decimal("50000"), Decimal("51000"), Decimal("49500"),
                Decimal("48500"), Decimal("50500"), Decimal("49800")
            ],
            timestamps=[
                datetime.utcnow() - timedelta(days=i) for i in range(9, 0, -1)
            ]
        )

    def test_risk_service_initialization_execution(self, risk_service):
        """Test initialisation service de risque."""
        assert risk_service.params.confidence_level == 0.95
        assert hasattr(risk_service, 'calculate_var')
        assert hasattr(risk_service, 'calculate_volatility')

    def test_calculate_returns_execution(self, risk_service, sample_market_data):
        """Test calcul des rendements."""
        # Exécuter calcul des rendements
        returns = risk_service.calculate_returns(sample_market_data.prices)

        # Vérifier résultat
        assert isinstance(returns, list)
        assert len(returns) == len(sample_market_data.prices) - 1  # n-1 returns pour n prix

        # Vérifier calculs
        for i, ret in enumerate(returns):
            expected_return = (sample_market_data.prices[i+1] - sample_market_data.prices[i]) / sample_market_data.prices[i]
            assert abs(ret - expected_return) < Decimal("0.0001")

    def test_calculate_volatility_execution(self, risk_service):
        """Test calcul de volatilité."""
        # Returns de test
        returns = [
            Decimal("0.02"), Decimal("-0.01"), Decimal("0.03"),
            Decimal("-0.02"), Decimal("0.01"), Decimal("-0.015"),
            Decimal("0.025"), Decimal("-0.005")
        ]

        # Exécuter calcul volatilité
        volatility = risk_service.calculate_volatility(returns)

        # Vérifier résultat
        assert isinstance(volatility, Decimal)
        assert volatility > 0

        # Vérifier que c'est cohérent avec std dev
        import statistics
        expected_vol = Decimal(str(statistics.stdev([float(r) for r in returns])))
        assert abs(volatility - expected_vol) < Decimal("0.001")

    def test_calculate_var_execution(self, risk_service):
        """Test calcul Value at Risk."""
        # Returns de test avec distribution connue
        returns = [
            Decimal("0.05"), Decimal("0.02"), Decimal("-0.01"), Decimal("0.03"),
            Decimal("-0.02"), Decimal("-0.04"), Decimal("0.01"), Decimal("-0.06"),
            Decimal("0.04"), Decimal("-0.03")
        ]

        # Exécuter calcul VaR
        var_result = risk_service.calculate_var(returns)

        # Vérifier résultat
        assert isinstance(var_result, Decimal)
        assert var_result >= Decimal("0")  # VaR toujours positive

        # Pour 95% confidence sur 10 returns, on prend le 5% pire
        sorted_returns = sorted([float(r) for r in returns])
        expected_var = abs(Decimal(str(sorted_returns[0])))  # Le pire return

        # VaR devrait être proche du pire return (6%)
        assert var_result >= Decimal("0.04")

    def test_market_data_processing_execution(self, risk_service, sample_market_data):
        """Test traitement des données de marché."""
        # Calculer returns à partir des prix
        sample_market_data.returns = risk_service.calculate_returns(sample_market_data.prices)

        # Calculer volatilité
        sample_market_data.volatility = risk_service.calculate_volatility(sample_market_data.returns)

        # Vérifier traitement complet
        assert sample_market_data.returns is not None
        assert len(sample_market_data.returns) > 0
        assert sample_market_data.volatility > Decimal("0")

        # Vérifier cohérence
        assert len(sample_market_data.returns) == len(sample_market_data.prices) - 1


class TestSignalServiceExecution:
    """Tests d'exécution réelle pour SignalService."""

    @pytest.fixture
    def signal_service(self):
        """Service de signaux."""
        return SignalService()

    @pytest.fixture
    def sample_signals(self):
        """Signaux de test."""
        return [
            Signal(
                symbol="BTC/USD",
                action=SignalAction.BUY,
                confidence=Decimal("0.85"),
                timestamp=datetime.utcnow(),
                signal_type=SignalType.ENTRY,
                metadata={"strategy": "mean_reversion", "indicator": "oversold"}
            ),
            Signal(
                symbol="ETH/USD",
                action=SignalAction.SELL,
                confidence=Decimal("0.72"),
                timestamp=datetime.utcnow() - timedelta(minutes=5),
                signal_type=SignalType.EXIT,
                metadata={"strategy": "momentum", "indicator": "overbought"}
            ),
            Signal(
                symbol="BTC/USD",
                action=SignalAction.HOLD,
                confidence=Decimal("0.45"),
                timestamp=datetime.utcnow() - timedelta(minutes=10),
                signal_type=SignalType.REBALANCE,
                metadata={"strategy": "portfolio_optimization"}
            )
        ]

    def test_signal_service_initialization_execution(self, signal_service):
        """Test initialisation service de signaux."""
        assert hasattr(signal_service, 'filter_signals')
        assert hasattr(signal_service, 'aggregate_signals')
        assert hasattr(signal_service, 'validate_signal')

    def test_filter_signals_by_confidence_execution(self, signal_service, sample_signals):
        """Test filtrage signaux par confiance."""
        min_confidence = Decimal("0.7")

        # Exécuter filtrage
        filtered_signals = signal_service.filter_signals_by_confidence(
            signals=sample_signals,
            min_confidence=min_confidence
        )

        # Vérifier filtrage
        assert isinstance(filtered_signals, list)
        assert len(filtered_signals) == 2  # BTC 0.85 et ETH 0.72

        for signal in filtered_signals:
            assert signal.confidence >= min_confidence

    def test_filter_signals_by_time_execution(self, signal_service, sample_signals):
        """Test filtrage signaux par fenêtre temporelle."""
        time_window = timedelta(minutes=8)

        # Exécuter filtrage
        recent_signals = signal_service.filter_signals_by_time(
            signals=sample_signals,
            time_window=time_window
        )

        # Vérifier filtrage temporel
        assert isinstance(recent_signals, list)
        assert len(recent_signals) == 2  # Signaux des 8 dernières minutes

        cutoff_time = datetime.utcnow() - time_window
        for signal in recent_signals:
            assert signal.timestamp >= cutoff_time

    def test_aggregate_signals_by_symbol_execution(self, signal_service, sample_signals):
        """Test agrégation signaux par symbole."""
        # Exécuter agrégation
        aggregated = signal_service.aggregate_signals_by_symbol(sample_signals)

        # Vérifier agrégation
        assert isinstance(aggregated, dict)
        assert "BTC/USD" in aggregated
        assert "ETH/USD" in aggregated

        # BTC a 2 signaux (BUY et HOLD)
        btc_signals = aggregated["BTC/USD"]
        assert len(btc_signals) == 2

        # ETH a 1 signal (SELL)
        eth_signals = aggregated["ETH/USD"]
        assert len(eth_signals) == 1

    def test_calculate_consensus_signal_execution(self, signal_service):
        """Test calcul signal de consensus."""
        # Signaux conflictuels pour BTC
        btc_signals = [
            Signal("BTC/USD", SignalAction.BUY, Decimal("0.8"), datetime.utcnow()),
            Signal("BTC/USD", SignalAction.BUY, Decimal("0.7"), datetime.utcnow()),
            Signal("BTC/USD", SignalAction.SELL, Decimal("0.6"), datetime.utcnow()),
            Signal("BTC/USD", SignalAction.HOLD, Decimal("0.5"), datetime.utcnow())
        ]

        # Exécuter calcul consensus
        consensus = signal_service.calculate_consensus_signal(btc_signals)

        # Vérifier consensus
        assert isinstance(consensus, Signal)
        assert consensus.symbol == "BTC/USD"

        # Avec 2 BUY (0.8, 0.7) vs 1 SELL (0.6) vs 1 HOLD (0.5)
        # Le consensus devrait pencher vers BUY
        assert consensus.action == SignalAction.BUY
        assert consensus.confidence > Decimal("0.5")

    def test_validate_signal_execution(self, signal_service):
        """Test validation de signal."""
        # Signal valide
        valid_signal = Signal(
            symbol="BTC/USD",
            action=SignalAction.BUY,
            confidence=Decimal("0.75"),
            timestamp=datetime.utcnow()
        )

        # Exécuter validation
        is_valid, errors = signal_service.validate_signal(valid_signal)

        # Vérifier validation
        assert is_valid is True
        assert len(errors) == 0

        # Signal invalide
        invalid_signal = Signal(
            symbol="",  # Symbole vide
            action=SignalAction.BUY,
            confidence=Decimal("1.5"),  # Confiance > 1
            timestamp=datetime.utcnow() + timedelta(hours=1)  # Futur
        )

        is_valid, errors = signal_service.validate_signal(invalid_signal)

        # Vérifier échec validation
        assert is_valid is False
        assert len(errors) > 0
        assert any("symbol" in error.lower() for error in errors)
        assert any("confidence" in error.lower() for error in errors)

    def test_prioritize_signals_execution(self, signal_service, sample_signals):
        """Test priorisation des signaux."""
        # Exécuter priorisation
        prioritized = signal_service.prioritize_signals(
            signals=sample_signals,
            max_signals=2
        )

        # Vérifier priorisation
        assert isinstance(prioritized, list)
        assert len(prioritized) <= 2

        # Les signaux doivent être triés par confiance décroissante
        if len(prioritized) == 2:
            assert prioritized[0].confidence >= prioritized[1].confidence

        # Le premier signal devrait être le BTC BUY (confidence 0.85)
        assert prioritized[0].symbol == "BTC/USD"
        assert prioritized[0].action == SignalAction.BUY

    def test_signal_service_integration_execution(self, signal_service, sample_signals):
        """Test d'intégration complète du service de signaux."""
        # Workflow complet: Filtrage → Agrégation → Consensus → Priorisation

        # 1. Filtrer par confiance minimale
        min_confidence = Decimal("0.6")
        confident_signals = signal_service.filter_signals_by_confidence(
            sample_signals, min_confidence
        )

        # 2. Filtrer par fenêtre temporelle
        time_window = timedelta(minutes=15)
        recent_signals = signal_service.filter_signals_by_time(
            confident_signals, time_window
        )

        # 3. Agréger par symbole
        aggregated = signal_service.aggregate_signals_by_symbol(recent_signals)

        # 4. Calculer consensus pour chaque symbole
        consensus_signals = []
        for symbol, signals in aggregated.items():
            if len(signals) > 1:
                consensus = signal_service.calculate_consensus_signal(signals)
                consensus_signals.append(consensus)
            else:
                consensus_signals.extend(signals)

        # 5. Prioriser les signaux finaux
        final_signals = signal_service.prioritize_signals(consensus_signals, max_signals=3)

        # 6. Vérifier résultat final
        assert isinstance(final_signals, list)
        assert len(final_signals) <= 3

        for signal in final_signals:
            assert isinstance(signal, Signal)
            assert signal.confidence >= min_confidence

        # 7. Valider tous les signaux finaux
        for signal in final_signals:
            is_valid, errors = signal_service.validate_signal(signal)
            assert is_valid is True
            assert len(errors) == 0