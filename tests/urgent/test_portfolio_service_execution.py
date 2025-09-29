"""
Tests d'Exécution Réelle - Portfolio Service
===========================================

Tests qui EXÉCUTENT vraiment le code qframe.domain.services.portfolio_service
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List
from unittest.mock import Mock

from qframe.domain.services.portfolio_service import (
    PortfolioService, RebalancingPlan, AllocationOptimization,
    PortfolioPerformanceAnalysis
)
from qframe.domain.entities.portfolio import Portfolio, PortfolioSnapshot, PortfolioConstraints, RebalancingFrequency
from qframe.domain.entities.position import Position


class TestRebalancingPlanExecution:
    """Tests d'exécution réelle pour RebalancingPlan."""

    def test_rebalancing_plan_creation_execution(self):
        """Test création plan avec TOUS les calculs."""
        # Données réelles pour le plan
        target_allocations = {
            "BTC/USD": Decimal("0.50"),
            "ETH/USD": Decimal("0.30"),
            "CASH": Decimal("0.20")
        }

        current_allocations = {
            "BTC/USD": Decimal("0.60"),
            "ETH/USD": Decimal("0.25"),
            "CASH": Decimal("0.15")
        }

        trades_required = {
            "BTC/USD": Decimal("-1000"),  # Vendre $1000 de BTC
            "ETH/USD": Decimal("500"),    # Acheter $500 d'ETH
            "CASH": Decimal("500")        # Augmenter cash de $500
        }

        # Exécuter création complète
        plan = RebalancingPlan(
            portfolio_id="portfolio-001",
            timestamp=datetime.utcnow(),
            target_allocations=target_allocations,
            current_allocations=current_allocations,
            trades_required=trades_required,
            estimated_cost=Decimal("15.00"),
            reason="Threshold-based rebalancing"
        )

        # Exécuter tous les calculs métier
        assert plan.get_trade_value() == Decimal("2000")  # |1000| + |500| + |500|

        symbols_to_buy = plan.get_symbols_to_buy()
        assert symbols_to_buy == ["ETH/USD", "CASH"]

        symbols_to_sell = plan.get_symbols_to_sell()
        assert symbols_to_sell == ["BTC/USD"]


class TestAllocationOptimizationExecution:
    """Tests d'exécution réelle pour AllocationOptimization."""

    def test_allocation_optimization_calculations_execution(self):
        """Test calculs d'optimisation d'allocation."""
        original = {
            "BTC/USD": Decimal("0.70"),
            "ETH/USD": Decimal("0.30"),
            "CASH": Decimal("0.00")
        }

        optimized = {
            "BTC/USD": Decimal("0.50"),
            "ETH/USD": Decimal("0.30"),
            "CASH": Decimal("0.20")
        }

        # Exécuter création avec calculs métier
        optimization = AllocationOptimization(
            original_allocations=original,
            optimized_allocations=optimized,
            expected_return=Decimal("0.12"),
            expected_risk=Decimal("0.25"),
            sharpe_ratio=Decimal("0.40"),
            optimization_method="risk_adjusted",
            constraints_applied=["max_allocation_constraint"]
        )

        # Exécuter calculs de changements
        changes = optimization.get_allocation_changes()

        assert changes["BTC/USD"] == Decimal("-0.20")  # Réduction de 20%
        assert changes["ETH/USD"] == Decimal("0.00")   # Pas de changement
        assert changes["CASH"] == Decimal("0.20")      # Augmentation de 20%


class TestPortfolioServiceExecution:
    """Tests d'exécution réelle pour PortfolioService."""

    @pytest.fixture
    def portfolio_service(self):
        """Service avec taux sans risque configuré."""
        return PortfolioService(risk_free_rate=Decimal("0.03"))

    @pytest.fixture
    def sample_portfolio(self):
        """Portfolio de test avec positions réelles."""
        # Positions réelles
        positions = {
            "BTC/USD": Position(
                symbol="BTC/USD",
                quantity=Decimal("0.5"),
                average_price=Decimal("45000"),
                current_price=Decimal("50000"),
                market_value=Decimal("25000")  # 0.5 * 50000
            ),
            "ETH/USD": Position(
                symbol="ETH/USD",
                quantity=Decimal("10"),
                average_price=Decimal("2800"),
                current_price=Decimal("3000"),
                market_value=Decimal("30000")  # 10 * 3000
            ),
            "CASH": Position(
                symbol="CASH",
                quantity=Decimal("15000"),
                average_price=Decimal("1"),
                current_price=Decimal("1"),
                market_value=Decimal("15000")
            )
        }

        # Contraintes réelles
        constraints = PortfolioConstraints(
            max_position_size=Decimal("0.4"),
            max_leverage=Decimal("1.0"),
            allowed_symbols=["BTC/USD", "ETH/USD", "CASH"],
            rebalancing_frequency=RebalancingFrequency.WEEKLY
        )

        # Allocations cibles
        target_allocations = {
            "BTC/USD": Decimal("0.35"),  # 35%
            "ETH/USD": Decimal("0.40"),  # 40%
            "CASH": Decimal("0.25")      # 25%
        }

        portfolio = Portfolio(
            id="test-portfolio-001",
            name="Test Portfolio",
            initial_capital=Decimal("60000"),
            base_currency="USD",
            positions=positions,
            target_allocations=target_allocations,
            constraints=constraints
        )

        # Calculer la valeur totale
        portfolio.total_value = sum(pos.market_value for pos in positions.values())

        return portfolio

    @pytest.fixture
    def portfolio_with_history(self, sample_portfolio):
        """Portfolio avec historique de snapshots."""
        portfolio = sample_portfolio

        # Créer historique de 30 jours avec évolution réaliste
        base_date = datetime.utcnow() - timedelta(days=30)
        snapshots = []

        base_value = Decimal("65000")  # Valeur de départ
        for i in range(30):
            # Évolution avec volatilité réaliste
            variation = Decimal(str((i % 7 - 3) * 0.02))  # ±6% variation cyclique
            daily_value = base_value * (Decimal("1") + variation)

            snapshot = PortfolioSnapshot(
                timestamp=base_date + timedelta(days=i),
                total_value=daily_value,
                positions_snapshot=portfolio.positions.copy(),
                cash_balance=Decimal("15000") + variation * Decimal("1000")
            )
            snapshots.append(snapshot)
            base_value = daily_value

        portfolio.snapshots = snapshots
        return portfolio

    def test_create_rebalancing_plan_execution(self, portfolio_service, sample_portfolio):
        """Test création plan de rééquilibrage avec calculs réels."""
        # Calculer allocations actuelles réelles
        # BTC: 25000/70000 = 35.7%, ETH: 30000/70000 = 42.9%, CASH: 15000/70000 = 21.4%

        # Exécuter création du plan
        plan = portfolio_service.create_rebalancing_plan(
            portfolio=sample_portfolio,
            rebalancing_threshold=Decimal("0.05"),
            transaction_cost_rate=Decimal("0.002")
        )

        # Vérifier que le plan a été créé
        assert plan is not None
        assert isinstance(plan, RebalancingPlan)
        assert plan.portfolio_id == sample_portfolio.id

        # Vérifier calculs des allocations actuelles
        current_allocs = plan.current_allocations
        assert "BTC/USD" in current_allocs
        assert "ETH/USD" in current_allocs
        assert "CASH" in current_allocs

        # Vérifier que les trades ont été calculés
        assert len(plan.trades_required) > 0

        # Vérifier calcul du coût
        assert plan.estimated_cost > 0

        # Vérifier valeur totale des trades
        trade_value = plan.get_trade_value()
        assert trade_value > 0

    def test_rebalancing_frequency_check_execution(self, portfolio_service, sample_portfolio):
        """Test vérification fréquence de rééquilibrage."""
        # Test avec dernière date de rééquilibrage récente
        sample_portfolio.last_rebalanced_at = datetime.utcnow() - timedelta(days=3)

        # Exécuter vérification WEEKLY (devrait être False - trop récent)
        should_rebalance = portfolio_service.should_rebalance_by_frequency(
            sample_portfolio, RebalancingFrequency.WEEKLY
        )
        assert should_rebalance is False

        # Test avec dernière date ancienne
        sample_portfolio.last_rebalanced_at = datetime.utcnow() - timedelta(days=10)

        # Exécuter vérification WEEKLY (devrait être True - plus d'une semaine)
        should_rebalance = portfolio_service.should_rebalance_by_frequency(
            sample_portfolio, RebalancingFrequency.WEEKLY
        )
        assert should_rebalance is True

        # Test DAILY
        should_rebalance_daily = portfolio_service.should_rebalance_by_frequency(
            sample_portfolio, RebalancingFrequency.DAILY
        )
        assert should_rebalance_daily is True

        # Test MANUAL (toujours False)
        should_rebalance_manual = portfolio_service.should_rebalance_by_frequency(
            sample_portfolio, RebalancingFrequency.MANUAL
        )
        assert should_rebalance_manual is False

    def test_execute_rebalancing_plan_execution(self, portfolio_service, sample_portfolio):
        """Test exécution réelle d'un plan de rééquilibrage."""
        # Créer un plan réel
        plan = portfolio_service.create_rebalancing_plan(sample_portfolio)

        if plan:  # Seulement si un plan est nécessaire
            # Exécuter le plan
            execution_log = portfolio_service.execute_rebalancing_plan(sample_portfolio, plan)

            # Vérifier l'exécution
            assert isinstance(execution_log, list)
            assert len(execution_log) > 0

            # Vérifier que les messages contiennent des actions réelles
            log_text = " ".join(execution_log)
            assert any(action in log_text for action in ["BUY", "SELL"])
            assert "completed" in log_text.lower()

            # Vérifier mise à jour des dates
            assert sample_portfolio.last_rebalanced_at is not None
            assert sample_portfolio.updated_at is not None

    def test_equal_weight_optimization_execution(self, portfolio_service):
        """Test optimisation allocation équi-pondérée."""
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD"]

        # Exécuter optimisation
        optimization = portfolio_service.optimize_allocation_equal_weight(symbols)

        # Vérifier résultat
        assert isinstance(optimization, AllocationOptimization)
        assert optimization.optimization_method == "equal_weight"

        # Vérifier allocation équitable
        expected_weight = Decimal("1") / len(symbols)  # 25% chacun
        for symbol in symbols:
            assert optimization.optimized_allocations[symbol] == expected_weight

        # Vérifier somme = 1
        total_weight = sum(optimization.optimized_allocations.values())
        assert abs(total_weight - Decimal("1")) < Decimal("0.0001")

    def test_equal_weight_with_constraints_execution(self, portfolio_service):
        """Test optimisation équi-pondérée avec contraintes."""
        symbols = ["BTC/USD", "ETH/USD", "CASH"]

        # Contraintes réelles
        constraints = {
            "BTC/USD": (Decimal("0.20"), Decimal("0.40")),  # Min 20%, Max 40%
            "CASH": (Decimal("0.15"), Decimal("0.30"))       # Min 15%, Max 30%
        }

        # Exécuter optimisation avec contraintes
        optimization = portfolio_service.optimize_allocation_equal_weight(
            symbols, constraints
        )

        # Vérifier contraintes appliquées
        assert len(optimization.constraints_applied) > 0

        # Vérifier respect des contraintes
        for symbol, (min_w, max_w) in constraints.items():
            actual_weight = optimization.optimized_allocations[symbol]
            assert actual_weight >= min_w
            assert actual_weight <= max_w

        # Vérifier renormalisation
        total_weight = sum(optimization.optimized_allocations.values())
        assert abs(total_weight - Decimal("1")) < Decimal("0.01")

    def test_risk_parity_optimization_execution(self, portfolio_service):
        """Test optimisation par parité de risque."""
        symbols = ["BTC/USD", "ETH/USD", "CASH"]

        # Estimations de risque réalistes
        risk_estimates = {
            "BTC/USD": Decimal("0.40"),  # Très volatile
            "ETH/USD": Decimal("0.35"),  # Volatile
            "CASH": Decimal("0.01")      # Très peu risqué
        }

        # Exécuter optimisation
        optimization = portfolio_service.optimize_allocation_risk_parity(
            symbols, risk_estimates
        )

        # Vérifier résultat
        assert optimization.optimization_method == "risk_parity"

        # Vérifier logique de parité de risque
        # CASH devrait avoir le plus gros poids (risque le plus faible)
        cash_weight = optimization.optimized_allocations["CASH"]
        btc_weight = optimization.optimized_allocations["BTC/USD"]

        assert cash_weight > btc_weight  # Cash moins risqué → plus de poids

        # Calculer contribution au risque pour validation
        risk_contributions = {}
        for symbol in symbols:
            weight = optimization.optimized_allocations[symbol]
            risk = risk_estimates[symbol]
            risk_contributions[symbol] = weight * risk

        # En risk parity, les contributions devraient être plus équilibrées
        contrib_values = list(risk_contributions.values())
        contrib_range = max(contrib_values) - min(contrib_values)
        assert contrib_range < Decimal("0.3")  # Contributions relativement équilibrées

    def test_momentum_optimization_execution(self, portfolio_service):
        """Test optimisation basée sur le momentum."""
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD"]

        # Scores de momentum réalistes
        momentum_scores = {
            "BTC/USD": Decimal("0.15"),   # Momentum positif
            "ETH/USD": Decimal("0.08"),   # Momentum modéré
            "ADA/USD": Decimal("-0.05")   # Momentum négatif
        }

        # Exécuter optimisation
        optimization = portfolio_service.optimize_allocation_momentum(
            symbols, momentum_scores, lookback_period=12
        )

        # Vérifier résultat
        assert optimization.optimization_method == "momentum"

        # Vérifier logique momentum
        btc_weight = optimization.optimized_allocations["BTC/USD"]
        eth_weight = optimization.optimized_allocations["ETH/USD"]
        ada_weight = optimization.optimized_allocations["ADA/USD"]

        # BTC (momentum le plus élevé) devrait avoir le plus gros poids
        assert btc_weight > eth_weight
        assert eth_weight > ada_weight  # ETH > ADA (momentum positif vs négatif)

        # Vérifier que les poids sont positifs même pour momentum négatif
        assert ada_weight >= 0

    def test_momentum_optimization_no_positive_momentum_execution(self, portfolio_service):
        """Test optimisation momentum avec que des scores négatifs."""
        symbols = ["BTC/USD", "ETH/USD"]

        # Tous momentum négatifs
        momentum_scores = {
            "BTC/USD": Decimal("-0.10"),
            "ETH/USD": Decimal("-0.15")
        }

        # Exécuter optimisation
        optimization = portfolio_service.optimize_allocation_momentum(
            symbols, momentum_scores
        )

        # Devrait fallback vers allocation équi-pondérée
        expected_weight = Decimal("0.5")
        assert optimization.optimized_allocations["BTC/USD"] == expected_weight
        assert optimization.optimized_allocations["ETH/USD"] == expected_weight

    def test_portfolio_performance_analysis_execution(self, portfolio_service, portfolio_with_history):
        """Test analyse complète de performance."""
        # Exécuter analyse de performance
        analysis = portfolio_service.analyze_portfolio_performance(
            portfolio_with_history,
            analysis_period_days=30
        )

        # Vérifier création de l'analyse
        assert isinstance(analysis, PortfolioPerformanceAnalysis)
        assert analysis.portfolio_id == portfolio_with_history.id
        assert analysis.analysis_period_days == 30

        # Vérifier calculs de base
        assert analysis.total_return != Decimal("0")
        assert analysis.annualized_return != Decimal("0")
        assert analysis.volatility >= Decimal("0")
        assert analysis.max_drawdown >= Decimal("0")

        # Vérifier métriques optionnelles
        assert analysis.win_rate is not None
        assert analysis.best_day is not None
        assert analysis.worst_day is not None
        assert analysis.var_95 is not None

        # Vérifier sérialisation
        analysis_dict = analysis.to_dict()
        assert isinstance(analysis_dict, dict)
        assert "total_return" in analysis_dict
        assert "sharpe_ratio" in analysis_dict
        assert isinstance(analysis_dict["total_return"], float)

    def test_portfolio_performance_analysis_with_benchmark_execution(self, portfolio_service, portfolio_with_history):
        """Test analyse de performance avec benchmark."""
        # Créer benchmark returns réalistes
        benchmark_returns = [Decimal(str(0.01 * (i % 5 - 2))) for i in range(30)]  # ±2% cyclique

        # Exécuter analyse avec benchmark
        analysis = portfolio_service.analyze_portfolio_performance(
            portfolio_with_history,
            analysis_period_days=30,
            benchmark_returns=benchmark_returns
        )

        # Vérifier métriques relatives au benchmark
        assert analysis.benchmark_return is not None
        assert analysis.alpha is not None
        assert analysis.beta is not None
        assert analysis.information_ratio is not None

        # Vérifier que les calculs sont logiques
        assert isinstance(analysis.beta, Decimal)
        assert analysis.beta > 0  # Beta positif attendu pour un portfolio d'actifs

    def test_portfolio_comparison_execution(self, portfolio_service, sample_portfolio):
        """Test comparaison de portfolios."""
        # Créer plusieurs portfolios pour comparaison
        portfolio1 = sample_portfolio

        # Portfolio 2 avec allocations différentes
        portfolio2 = Portfolio(
            id="test-portfolio-002",
            name="Conservative Portfolio",
            initial_capital=Decimal("50000"),
            base_currency="USD"
        )

        # Ajouter historique simplifié aux portfolios
        for portfolio in [portfolio1, portfolio2]:
            snapshots = []
            base_value = portfolio.initial_capital
            for i in range(10):
                snapshot = PortfolioSnapshot(
                    timestamp=datetime.utcnow() - timedelta(days=10-i),
                    total_value=base_value + Decimal(str(i * 1000)),  # Croissance linéaire
                    positions_snapshot={},
                    cash_balance=Decimal("10000")
                )
                snapshots.append(snapshot)
            portfolio.snapshots = snapshots

        # Exécuter comparaison
        portfolios = [portfolio1, portfolio2]
        comparison = portfolio_service.compare_portfolios(portfolios, metric="sharpe_ratio")

        # Vérifier résultat
        assert isinstance(comparison, list)
        assert len(comparison) == 2

        # Vérifier structure du résultat
        for portfolio, analysis in comparison:
            assert isinstance(portfolio, Portfolio)
            assert isinstance(analysis, PortfolioPerformanceAnalysis)

        # Vérifier tri (devrait être par sharpe_ratio décroissant)
        first_sharpe = comparison[0][1].sharpe_ratio
        second_sharpe = comparison[1][1].sharpe_ratio
        assert first_sharpe >= second_sharpe

    def test_calculate_current_allocations_execution(self, portfolio_service, sample_portfolio):
        """Test calcul allocations actuelles."""
        # Exécuter calcul
        allocations = portfolio_service._calculate_current_allocations(sample_portfolio)

        # Vérifier résultat
        assert isinstance(allocations, dict)
        assert "BTC/USD" in allocations
        assert "ETH/USD" in allocations
        assert "CASH" in allocations

        # Vérifier calculs corrects
        total_value = sample_portfolio.total_value
        btc_allocation = allocations["BTC/USD"]
        eth_allocation = allocations["ETH/USD"]
        cash_allocation = allocations["CASH"]

        # Vérifier pourcentages
        assert btc_allocation == Decimal("25000") / total_value  # 25000/70000
        assert eth_allocation == Decimal("30000") / total_value  # 30000/70000
        assert cash_allocation == Decimal("15000") / total_value # 15000/70000

        # Vérifier somme = 1
        total_allocation = sum(allocations.values())
        assert abs(total_allocation - Decimal("1")) < Decimal("0.0001")

    def test_calculate_risk_metrics_execution(self, portfolio_service, portfolio_with_history):
        """Test calcul métriques de risque."""
        # Exécuter calcul des métriques
        risk_metrics = portfolio_service.calculate_risk_metrics(portfolio_with_history)

        # Vérifier résultat
        assert risk_metrics is not None
        assert isinstance(risk_metrics, dict)

        # Vérifier métriques requises
        required_metrics = [
            "volatility", "var_95", "max_drawdown", "sharpe_ratio",
            "avg_daily_return", "observation_count"
        ]

        for metric in required_metrics:
            assert metric in risk_metrics
            assert isinstance(risk_metrics[metric], (int, float))

        # Vérifier valeurs logiques
        assert risk_metrics["volatility"] >= 0
        assert risk_metrics["var_95"] >= 0
        assert risk_metrics["max_drawdown"] >= 0
        assert risk_metrics["observation_count"] > 0

    def test_calculate_concentration_risk_execution(self, portfolio_service, sample_portfolio):
        """Test calcul risque de concentration."""
        # Exécuter calcul
        concentration_risk = portfolio_service.calculate_concentration_risk(sample_portfolio)

        # Vérifier résultat
        assert isinstance(concentration_risk, Decimal)
        assert Decimal("0") <= concentration_risk <= Decimal("1")

        # Test avec portfolio très concentré (une seule position)
        concentrated_portfolio = Portfolio(
            id="concentrated-portfolio",
            name="Concentrated",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )

        concentrated_portfolio.positions = {
            "BTC/USD": Position(
                symbol="BTC/USD",
                quantity=Decimal("2"),
                average_price=Decimal("50000"),
                current_price=Decimal("50000"),
                market_value=Decimal("100000")
            )
        }
        concentrated_portfolio.total_value = Decimal("100000")

        concentrated_risk = portfolio_service.calculate_concentration_risk(concentrated_portfolio)

        # Portfolio concentré devrait avoir un risque plus élevé
        assert concentrated_risk > concentration_risk

    def test_estimate_correlation_risk_execution(self, portfolio_service, sample_portfolio):
        """Test estimation risque de corrélation."""
        # Exécuter estimation
        correlation_risk = portfolio_service.estimate_correlation_risk(sample_portfolio)

        # Vérifier résultat
        assert isinstance(correlation_risk, Decimal)
        assert Decimal("0") <= correlation_risk <= Decimal("1")

        # Test avec portfolio full crypto
        crypto_portfolio = Portfolio(
            id="crypto-portfolio",
            name="All Crypto",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )

        crypto_portfolio.positions = {
            "BTC/USD": Position(symbol="BTC/USD", quantity=Decimal("1"),
                              average_price=Decimal("50000"), current_price=Decimal("50000"),
                              market_value=Decimal("50000")),
            "ETH/USD": Position(symbol="ETH/USD", quantity=Decimal("10"),
                              average_price=Decimal("3000"), current_price=Decimal("3000"),
                              market_value=Decimal("30000")),
            "ADA/USD": Position(symbol="ADA/USD", quantity=Decimal("10000"),
                              average_price=Decimal("2"), current_price=Decimal("2"),
                              market_value=Decimal("20000"))
        }

        crypto_correlation_risk = portfolio_service.estimate_correlation_risk(crypto_portfolio)

        # Portfolio full crypto devrait avoir un risque de corrélation plus élevé
        assert crypto_correlation_risk >= Decimal("0.6")  # Haute corrélation attendue

    def test_daily_returns_calculation_execution(self, portfolio_service, portfolio_with_history):
        """Test calcul des rendements journaliers."""
        # Exécuter calcul des rendements
        returns = portfolio_service._calculate_daily_returns(portfolio_with_history.snapshots)

        # Vérifier résultat
        assert isinstance(returns, list)
        assert len(returns) == len(portfolio_with_history.snapshots) - 1  # n-1 rendements pour n snapshots

        # Vérifier types des rendements
        for ret in returns:
            assert isinstance(ret, Decimal)

        # Vérifier logique des calculs
        if len(returns) >= 2:
            # Premier rendement basé sur snapshots[0] et snapshots[1]
            snapshot1 = portfolio_with_history.snapshots[0]
            snapshot2 = portfolio_with_history.snapshots[1]

            expected_return = (snapshot2.total_value - snapshot1.total_value) / snapshot1.total_value
            assert abs(returns[0] - expected_return) < Decimal("0.0001")

    def test_volatility_calculation_execution(self, portfolio_service):
        """Test calcul de volatilité."""
        # Rendements de test
        test_returns = [
            Decimal("0.02"), Decimal("-0.01"), Decimal("0.03"),
            Decimal("-0.02"), Decimal("0.01"), Decimal("0.004"),
            Decimal("-0.015"), Decimal("0.025")
        ]

        # Exécuter calcul de volatilité
        volatility = portfolio_service._calculate_volatility(test_returns)

        # Vérifier résultat
        assert isinstance(volatility, Decimal)
        assert volatility > 0

        # Vérifier que c'est annualisé (devrait être plus grand que la volatilité journalière)
        import statistics
        daily_vol = Decimal(str(statistics.stdev([float(r) for r in test_returns])))
        assert volatility > daily_vol  # Volatilité annualisée > volatilité journalière

    def test_max_drawdown_calculation_execution(self, portfolio_service, portfolio_with_history):
        """Test calcul du drawdown maximum."""
        # Exécuter calcul
        max_drawdown = portfolio_service._calculate_max_drawdown(portfolio_with_history.snapshots)

        # Vérifier résultat
        assert isinstance(max_drawdown, Decimal)
        assert max_drawdown >= Decimal("0")

        # Test avec snapshots simulant un vrai drawdown
        declining_snapshots = []
        values = [Decimal("100000"), Decimal("95000"), Decimal("85000"),
                 Decimal("80000"), Decimal("90000"), Decimal("95000")]

        for i, value in enumerate(values):
            snapshot = PortfolioSnapshot(
                timestamp=datetime.utcnow() - timedelta(days=len(values)-i),
                total_value=value,
                positions_snapshot={},
                cash_balance=Decimal("10000")
            )
            declining_snapshots.append(snapshot)

        # Calculer drawdown sur données avec baisse
        dd = portfolio_service._calculate_max_drawdown(declining_snapshots)

        # Le drawdown devrait être de 20% (100k -> 80k)
        expected_dd = Decimal("0.20")
        assert abs(dd - expected_dd) < Decimal("0.01")

    def test_beta_calculation_execution(self, portfolio_service):
        """Test calcul du beta."""
        # Returns de test corrélés
        portfolio_returns = [Decimal("0.02"), Decimal("-0.01"), Decimal("0.03"), Decimal("-0.015")]
        benchmark_returns = [Decimal("0.015"), Decimal("-0.008"), Decimal("0.025"), Decimal("-0.012")]

        # Exécuter calcul beta
        beta = portfolio_service._calculate_beta(portfolio_returns, benchmark_returns)

        # Vérifier résultat
        assert isinstance(beta, Decimal)
        assert beta > 0  # Beta positif pour des returns corrélés positivement

        # Test avec returns parfaitement corrélés (beta devrait être proche de 1)
        perfect_correlation_bench = portfolio_returns.copy()
        beta_perfect = portfolio_service._calculate_beta(portfolio_returns, perfect_correlation_bench)
        assert abs(beta_perfect - Decimal("1")) < Decimal("0.1")

    def test_var_calculation_execution(self, portfolio_service):
        """Test calcul Value at Risk."""
        # Returns de test avec distribution réaliste
        test_returns = [
            Decimal("0.05"), Decimal("0.02"), Decimal("0.01"), Decimal("-0.01"),
            Decimal("-0.02"), Decimal("-0.03"), Decimal("-0.05"), Decimal("-0.08"),
            Decimal("0.03"), Decimal("0.01")
        ]

        # Exécuter calcul VaR 95%
        var_95 = portfolio_service._calculate_var(test_returns, Decimal("0.95"))

        # Vérifier résultat
        assert isinstance(var_95, Decimal)
        assert var_95 >= 0  # VaR est toujours positive (valeur absolue)

        # Pour 95% confidence sur 10 returns, on devrait avoir le 5% pire return
        # Le pire return est -8%, donc VaR devrait être proche de 8%
        assert var_95 >= Decimal("0.05")  # Au moins 5%

    def test_information_ratio_calculation_execution(self, portfolio_service):
        """Test calcul ratio d'information."""
        # Returns avec alpha positif
        portfolio_returns = [Decimal("0.03"), Decimal("0.01"), Decimal("0.02"), Decimal("-0.01")]
        benchmark_returns = [Decimal("0.02"), Decimal("0.005"), Decimal("0.015"), Decimal("-0.015")]

        # Exécuter calcul
        info_ratio = portfolio_service._calculate_information_ratio(portfolio_returns, benchmark_returns)

        # Vérifier résultat
        assert isinstance(info_ratio, Decimal)

        # Portfolio performe mieux que benchmark, info ratio devrait être positif
        assert info_ratio > 0

    def test_portfolio_service_integration_execution(self, portfolio_service, sample_portfolio):
        """Test d'intégration complète du service."""
        # Workflow complet: optimisation → plan → exécution → analyse

        # 1. Optimiser allocation
        symbols = list(sample_portfolio.positions.keys())
        optimization = portfolio_service.optimize_allocation_equal_weight(symbols)

        # 2. Créer plan avec allocation optimisée
        plan = portfolio_service.create_rebalancing_plan(
            sample_portfolio,
            target_allocations=optimization.optimized_allocations
        )

        # 3. Exécuter si nécessaire
        if plan:
            execution_log = portfolio_service.execute_rebalancing_plan(sample_portfolio, plan)
            assert len(execution_log) > 0

        # 4. Analyser performance (avec historique simulé)
        # Ajouter quelques snapshots pour l'analyse
        snapshots = []
        for i in range(5):
            snapshot = PortfolioSnapshot(
                timestamp=datetime.utcnow() - timedelta(days=5-i),
                total_value=sample_portfolio.total_value + Decimal(str(i * 1000)),
                positions_snapshot=sample_portfolio.positions.copy(),
                cash_balance=Decimal("15000")
            )
            snapshots.append(snapshot)
        sample_portfolio.snapshots = snapshots

        analysis = portfolio_service.analyze_portfolio_performance(sample_portfolio)

        # 5. Vérifier workflow complet
        assert isinstance(optimization, AllocationOptimization)
        assert isinstance(analysis, PortfolioPerformanceAnalysis)

        # 6. Calculer métriques de risque
        risk_metrics = portfolio_service.calculate_risk_metrics(sample_portfolio)
        if risk_metrics:
            assert "volatility" in risk_metrics
            assert "sharpe_ratio" in risk_metrics

        # 7. Calculer risques de concentration et corrélation
        concentration_risk = portfolio_service.calculate_concentration_risk(sample_portfolio)
        correlation_risk = portfolio_service.estimate_correlation_risk(sample_portfolio)

        assert isinstance(concentration_risk, Decimal)
        assert isinstance(correlation_risk, Decimal)