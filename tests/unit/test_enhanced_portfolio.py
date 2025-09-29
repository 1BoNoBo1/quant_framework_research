"""
Tests pour Enhanced Portfolio avec Pydantic
==========================================

Tests complets de validation, calculs, et métriques du portfolio entreprise.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any

from qframe.domain.entities.enhanced_portfolio import (
    EnhancedPortfolio, Position, RiskMetrics, PerformanceMetrics,
    PortfolioStatus, RiskLevel, CurrencyCode,
    create_portfolio, create_demo_portfolio
)


class TestPosition:
    """Tests de la classe Position."""

    def test_position_creation(self):
        """Test création position valide."""
        position = Position(
            symbol="BTC/USD",
            size=Decimal("1.5"),
            entry_price=45000.0,
            current_price=47000.0,
            strategy_name="Momentum"
        )

        assert position.symbol == "BTC/USD"
        assert position.size == Decimal("1.5")
        assert position.is_long == True
        assert position.market_value == Decimal("70500.0")  # 1.5 * 47000

    def test_position_validation_symbol(self):
        """Test validation du symbole."""
        # Symbole valide
        pos = Position(
            symbol="eth-usd",
            size=Decimal("1.0"),
            entry_price=3000.0,
            current_price=3100.0
        )
        assert pos.symbol == "ETH-USD"  # Uppercase automatique

        # Symbole invalide
        with pytest.raises(ValueError, match="Symbole doit contenir uniquement"):
            Position(
                symbol="BTC@USD",
                size=Decimal("1.0"),
                entry_price=45000.0,
                current_price=47000.0
            )

    def test_position_validation_size(self):
        """Test validation de la taille."""
        with pytest.raises(ValueError, match="Taille de position ne peut pas être zéro"):
            Position(
                symbol="BTC/USD",
                size=Decimal("0"),
                entry_price=45000.0,
                current_price=47000.0
            )

    def test_position_short_position(self):
        """Test position courte."""
        position = Position(
            symbol="BTC/USD",
            size=Decimal("-1.0"),
            entry_price=47000.0,
            current_price=45000.0
        )

        assert position.is_long == False
        assert position.market_value == Decimal("45000.0")  # abs(-1.0) * 45000

    def test_position_update_price(self):
        """Test mise à jour du prix."""
        position = Position(
            symbol="ETH/USD",
            size=Decimal("2.0"),
            entry_price=3000.0,
            current_price=3000.0
        )

        # Prix monte
        updated = position.update_current_price(3200.0)
        assert updated.current_price == 3200.0
        assert updated.unrealized_pnl == Decimal("400.0")  # 2.0 * (3200 - 3000)
        # PnL percent = pnl / market_value * 100 = 400 / 6000 * 100 = 6.67
        assert updated.unrealized_pnl_percent == pytest.approx(6.67, abs=0.1)

    def test_position_update_price_short(self):
        """Test mise à jour prix position courte."""
        position = Position(
            symbol="BTC/USD",
            size=Decimal("-0.5"),
            entry_price=47000.0,
            current_price=47000.0
        )

        # Prix baisse (profitable pour short)
        updated = position.update_current_price(45000.0)
        assert updated.unrealized_pnl == Decimal("1000.0")  # 0.5 * (47000 - 45000)

    def test_position_computed_fields(self):
        """Test des computed fields."""
        position = Position(
            symbol="BTC/USD",
            size=Decimal("1.0"),
            entry_price=45000.0,
            current_price=47000.0
        )

        assert position.market_value == Decimal("47000.0")
        assert position.is_long == True
        assert position.pnl_percentage == 0.0  # unrealized_pnl is None initially


class TestRiskMetrics:
    """Tests de la classe RiskMetrics."""

    def test_risk_metrics_creation(self):
        """Test création métriques de risque."""
        metrics = RiskMetrics(
            total_exposure=100000.0,
            leverage=2.5,
            max_position_weight=0.3
        )

        assert metrics.total_exposure == 100000.0
        assert metrics.leverage == 2.5
        assert metrics.risk_level == RiskLevel.MODERATE

    def test_risk_metrics_validation_leverage(self):
        """Test validation du levier."""
        with pytest.raises(ValueError, match="Levier ne peut pas dépasser 10x"):
            RiskMetrics(leverage=15.0)

    def test_risk_metrics_computed_score(self):
        """Test calcul du score de risque."""
        metrics = RiskMetrics(
            total_exposure=500000.0,  # Exposition importante
            leverage=3.0,            # Levier modéré
            max_position_weight=0.4, # Concentration élevée
            max_drawdown=0.15        # Drawdown significant
        )

        score = metrics.risk_score
        assert 0 <= score <= 100
        # Score doit être élevé avec ces paramètres
        assert score > 50


class TestPerformanceMetrics:
    """Tests de la classe PerformanceMetrics."""

    def test_performance_metrics_creation(self):
        """Test création métriques de performance."""
        start_time = datetime.now(timezone.utc)
        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.8,
            win_rate=0.65,
            period_start=start_time
        )

        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 1.8
        assert metrics.win_rate == 0.65

    def test_performance_metrics_validation_win_rate(self):
        """Test validation win rate."""
        start_time = datetime.now(timezone.utc)

        # Win rate invalide
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                win_rate=1.5,
                period_start=start_time
            )

    def test_performance_metrics_computed_fields(self):
        """Test computed fields."""
        start_time = datetime.now(timezone.utc)
        metrics = PerformanceMetrics(
            annualized_return=0.20,
            sharpe_ratio=2.0,
            period_start=start_time
        )

        # Risk adjusted return
        risk_adj = metrics.risk_adjusted_return
        assert risk_adj == 0.20 / 2.0  # annualized_return / sharpe_ratio


class TestEnhancedPortfolio:
    """Tests de la classe EnhancedPortfolio."""

    def test_portfolio_creation(self):
        """Test création portfolio basique."""
        portfolio = EnhancedPortfolio(
            name="Test Portfolio",
            owner_id="user_123",
            initial_capital=Decimal("100000"),
            current_balance=Decimal("95000"),
            base_currency=CurrencyCode.USD
        )

        assert portfolio.name == "Test Portfolio"
        assert portfolio.owner_id == "user_123"
        assert portfolio.total_equity == Decimal("95000")  # Pas de positions
        assert portfolio.position_count == 0
        assert portfolio.is_healthy == True

    def test_portfolio_validation_name(self):
        """Test validation du nom."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EnhancedPortfolio(
                name="   ",  # Nom vide après strip
                owner_id="user_123",
                initial_capital=Decimal("100000"),
                current_balance=Decimal("95000")
            )

    def test_portfolio_validation_balances(self):
        """Test validation des balances."""
        with pytest.raises(ValueError, match="Balance actuelle ne peut pas dépasser 10x"):
            EnhancedPortfolio(
                name="Test Portfolio",
                owner_id="user_123",
                initial_capital=Decimal("10000"),
                current_balance=Decimal("150000")  # 15x le capital initial
            )

    def test_portfolio_add_position(self):
        """Test ajout de position."""
        portfolio = create_portfolio(
            name="Test Portfolio",
            owner_id="user_123",
            initial_capital=50000
        )

        position = Position(
            symbol="BTC/USD",
            size=Decimal("1.0"),
            entry_price=45000.0,
            current_price=47000.0
        )

        updated_portfolio = portfolio.add_position(position)

        assert updated_portfolio.position_count == 1
        assert "BTC/USD" in updated_portfolio.position_symbols
        assert updated_portfolio.total_equity == Decimal("97000")  # 50000 + 47000
        assert updated_portfolio.version == 2  # Version incrémentée

    def test_portfolio_add_duplicate_position(self):
        """Test ajout position dupliquée."""
        portfolio = create_portfolio("Test", "user", 50000)

        position1 = Position(symbol="BTC/USD", size=Decimal("1.0"), entry_price=45000.0, current_price=47000.0)
        position2 = Position(symbol="BTC/USD", size=Decimal("0.5"), entry_price=46000.0, current_price=47000.0)

        portfolio = portfolio.add_position(position1)

        with pytest.raises(ValueError, match="Position existe déjà pour BTC/USD"):
            portfolio.add_position(position2)

    def test_portfolio_remove_position(self):
        """Test suppression de position."""
        portfolio = create_demo_portfolio()  # Portfolio avec positions
        initial_count = portfolio.position_count

        updated_portfolio = portfolio.remove_position("BTC/USD")

        assert updated_portfolio.position_count == initial_count - 1
        assert "BTC/USD" not in updated_portfolio.position_symbols
        assert updated_portfolio.version > portfolio.version

    def test_portfolio_remove_nonexistent_position(self):
        """Test suppression position inexistante."""
        portfolio = create_portfolio("Test", "user", 50000)

        with pytest.raises(ValueError, match="Position NONEXISTENT non trouvée"):
            portfolio.remove_position("NONEXISTENT")

    def test_portfolio_update_position_price(self):
        """Test mise à jour prix position."""
        portfolio = create_demo_portfolio()

        updated_portfolio = portfolio.update_position_price("BTC/USD", 50000.0)

        # Vérifier que la position BTC a été mise à jour
        btc_position = next(pos for pos in updated_portfolio.positions if pos.symbol == "BTC/USD")
        assert btc_position.current_price == 50000.0
        assert btc_position.unrealized_pnl is not None

    def test_portfolio_update_nonexistent_position_price(self):
        """Test mise à jour prix position inexistante."""
        portfolio = create_portfolio("Test", "user", 50000)

        with pytest.raises(ValueError, match="Position NONEXISTENT non trouvée"):
            portfolio.update_position_price("NONEXISTENT", 50000.0)

    def test_portfolio_calculate_risk_metrics(self):
        """Test calcul des métriques de risque."""
        portfolio = create_demo_portfolio()

        updated_portfolio = portfolio.calculate_risk_metrics()

        assert updated_portfolio.risk_metrics.total_exposure > 0
        assert updated_portfolio.risk_metrics.leverage > 0
        assert updated_portfolio.risk_metrics.max_position_weight > 0
        assert updated_portfolio.risk_metrics.concentration_index > 0

    def test_portfolio_computed_fields(self):
        """Test des computed fields du portfolio."""
        portfolio = create_demo_portfolio()

        # Total equity
        expected_equity = portfolio.current_balance + sum(pos.market_value for pos in portfolio.positions)
        assert portfolio.total_equity == expected_equity

        # Unrealized PnL
        expected_pnl = sum(pos.unrealized_pnl or Decimal('0') for pos in portfolio.positions)
        assert portfolio.unrealized_pnl == expected_pnl

        # Return percentage
        return_pct = portfolio.total_return_percent
        expected_return = float(((portfolio.total_equity - portfolio.initial_capital) / portfolio.initial_capital) * 100)
        assert abs(return_pct - expected_return) < 0.01

        # Position symbols
        symbols = portfolio.position_symbols
        assert all(isinstance(symbol, str) for symbol in symbols)
        assert len(symbols) == portfolio.position_count

    def test_portfolio_serialization(self):
        """Test sérialisation du portfolio."""
        portfolio = create_demo_portfolio()

        # Summary dict
        summary = portfolio.to_summary_dict()
        assert 'id' in summary
        assert 'total_equity' in summary
        assert 'risk_score' in summary
        assert isinstance(summary['total_equity'], float)

        # Detailed dict
        detailed = portfolio.to_detailed_dict()
        assert 'positions' in detailed
        assert 'risk_metrics' in detailed
        assert isinstance(detailed['positions'], list)


class TestFactoryFunctions:
    """Tests des fonctions factory."""

    def test_create_portfolio(self):
        """Test factory de création portfolio."""
        portfolio = create_portfolio(
            name="Trading Portfolio",
            owner_id="trader_123",
            initial_capital=100000.0,
            base_currency=CurrencyCode.USD
        )

        assert portfolio.name == "Trading Portfolio"
        assert portfolio.owner_id == "trader_123"
        assert portfolio.initial_capital == Decimal("100000.0")
        assert portfolio.current_balance == Decimal("100000.0")
        assert portfolio.base_currency == CurrencyCode.USD
        assert portfolio.performance_metrics is not None

    def test_create_demo_portfolio(self):
        """Test création portfolio de démonstration."""
        portfolio = create_demo_portfolio()

        assert portfolio.name == "Demo Trading Portfolio"
        assert portfolio.owner_id == "demo_user"
        assert portfolio.position_count > 0
        assert len(portfolio.positions) > 0

        # Vérifier positions de démo
        symbols = portfolio.position_symbols
        assert "BTC/USD" in symbols
        assert "ETH/USD" in symbols

        # Vérifier métriques calculées
        assert portfolio.risk_metrics.total_exposure > 0


class TestEdgeCases:
    """Tests des cas limites."""

    def test_portfolio_empty_positions(self):
        """Test portfolio sans positions."""
        portfolio = create_portfolio("Empty", "user", 10000)

        # Métriques avec portfolio vide
        updated = portfolio.calculate_risk_metrics()
        assert updated.risk_metrics.total_exposure == 0.0
        assert updated.risk_metrics.leverage >= 0.0

        # Computed fields
        assert portfolio.total_equity == portfolio.current_balance
        assert portfolio.unrealized_pnl == Decimal('0')
        assert len(portfolio.position_symbols) == 0

    def test_portfolio_max_position_limit(self):
        """Test limite de taille de position."""
        portfolio = EnhancedPortfolio(
            name="Limited Portfolio",
            owner_id="user_123",
            initial_capital=Decimal("100000"),
            current_balance=Decimal("100000"),
            max_position_size=Decimal("10000")  # Limite à 10k
        )

        # Position dans la limite
        small_position = Position(
            symbol="ETH/USD",
            size=Decimal("3.0"),
            entry_price=3000.0,
            current_price=3000.0  # 9k market value
        )
        portfolio = portfolio.add_position(small_position)

        # Position dépassant la limite
        large_position = Position(
            symbol="BTC/USD",
            size=Decimal("1.0"),
            entry_price=50000.0,
            current_price=50000.0  # 50k market value
        )

        with pytest.raises(ValueError, match="Position dépasse la limite"):
            portfolio.add_position(large_position)

    def test_portfolio_zero_division_protection(self):
        """Test protection contre division par zéro."""
        portfolio = create_portfolio("Test", "user", 0.01)  # Capital minimal

        # Return percentage avec capital très faible
        return_pct = portfolio.total_return_percent
        assert isinstance(return_pct, float)
        assert not (return_pct != return_pct)  # Pas NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])