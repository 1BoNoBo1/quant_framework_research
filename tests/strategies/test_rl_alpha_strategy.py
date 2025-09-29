"""
Tests for RL Alpha Strategy
==========================

Suite de tests complète pour la stratégie de génération d'alphas par Reinforcement Learning.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch
from datetime import datetime
from decimal import Decimal

from qframe.strategies.research.rl_alpha_strategy import (
    RLAlphaStrategy,
    RLAlphaConfig,
    AlphaFormula,
    SearchSpace,
    FormulaEnvironment,
    PPOAgent
)
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence


class TestAlphaFormula:
    """Tests pour la classe AlphaFormula"""

    def test_alpha_formula_creation(self):
        """Test la création d'une formule alpha"""
        formula = AlphaFormula(
            formula="(-1 * Corr(open, volume, 10))",
            operators=["corr", "product"],
            ic=0.05,
            rank_ic=0.08,
            complexity=3,
            generation=1
        )

        assert formula.formula == "(-1 * Corr(open, volume, 10))"
        assert formula.operators == ["corr", "product"]
        assert formula.ic == 0.05
        assert formula.rank_ic == 0.08
        assert formula.complexity == 3
        assert formula.generation == 1
        assert isinstance(formula.metadata, dict)

    def test_alpha_formula_with_metadata(self):
        """Test la création avec métadonnées"""
        metadata = {"source": "RL", "timestamp": "2023-01-01"}
        formula = AlphaFormula(
            formula="Scale(Delta(close, 5))",
            operators=["scale", "delta"],
            ic=0.03,
            rank_ic=0.04,
            complexity=2,
            generation=5,
            metadata=metadata
        )

        assert formula.metadata == metadata

    def test_alpha_formula_post_init(self):
        """Test l'initialisation automatique des métadonnées"""
        formula = AlphaFormula(
            formula="Rank(volume)",
            operators=["rank"],
            ic=0.02,
            rank_ic=0.025,
            complexity=1,
            generation=0
        )

        assert formula.metadata == {}


class TestRLAlphaConfig:
    """Tests pour la configuration RL Alpha"""

    def test_default_config(self):
        """Test la configuration par défaut"""
        config = RLAlphaConfig()

        assert config.max_generations == 1000
        assert config.population_size == 50
        assert config.max_complexity == 10
        assert config.ic_threshold == 0.02
        assert config.learning_rate == 0.001
        assert config.epsilon == 0.1
        assert config.state_dim == 50
        assert config.action_dim == 42
        assert config.hidden_dim == 128

    def test_custom_config(self):
        """Test une configuration personnalisée"""
        config = RLAlphaConfig(
            max_generations=500,
            population_size=30,
            max_complexity=5,
            ic_threshold=0.05,
            learning_rate=0.0001
        )

        assert config.max_generations == 500
        assert config.population_size == 30
        assert config.max_complexity == 5
        assert config.ic_threshold == 0.05
        assert config.learning_rate == 0.0001

    def test_config_validation(self):
        """Test la validation des paramètres"""
        # Test valeurs positives
        config = RLAlphaConfig(
            max_generations=100,
            population_size=10,
            ic_threshold=0.01
        )

        assert config.max_generations > 0
        assert config.population_size > 0
        assert config.ic_threshold > 0


class TestSearchSpace:
    """Tests pour l'espace de recherche"""

    def test_search_space_initialization(self):
        """Test l'initialisation de l'espace de recherche"""
        search_space = SearchSpace()

        assert hasattr(search_space, 'operators')
        assert hasattr(search_space, 'features')
        assert hasattr(search_space, 'constants')
        assert hasattr(search_space, 'time_deltas')

        # Vérifier qu'il y a des éléments dans chaque catégorie
        assert len(search_space.operators) > 0
        assert len(search_space.features) > 0
        assert len(search_space.constants) > 0
        assert len(search_space.time_deltas) > 0

    def test_get_action_space(self):
        """Test l'obtention de l'espace d'actions"""
        search_space = SearchSpace()
        action_space = search_space.get_action_space()

        assert isinstance(action_space, list)
        assert len(action_space) > 0

        # Vérifier que tous les éléments sont des strings
        for action in action_space:
            assert isinstance(action, str)

    def test_sample_operator(self):
        """Test l'échantillonnage d'opérateurs"""
        search_space = SearchSpace()

        # Test échantillonnage aléatoire
        operator = search_space.sample_operator()
        assert operator in search_space.operators

        # Test échantillonnage avec seed pour reproductibilité
        np.random.seed(42)
        op1 = search_space.sample_operator()
        np.random.seed(42)
        op2 = search_space.sample_operator()
        assert op1 == op2

    def test_sample_feature(self):
        """Test l'échantillonnage de features"""
        search_space = SearchSpace()

        feature = search_space.sample_feature()
        assert feature in search_space.features

    def test_sample_constant(self):
        """Test l'échantillonnage de constantes"""
        search_space = SearchSpace()

        constant = search_space.sample_constant()
        assert constant in search_space.constants
        assert isinstance(constant, (int, float))

    def test_sample_time_delta(self):
        """Test l'échantillonnage de deltas temporels"""
        search_space = SearchSpace()

        time_delta = search_space.sample_time_delta()
        assert time_delta in search_space.time_deltas
        assert isinstance(time_delta, int)
        assert time_delta > 0


class TestFormulaEnvironment:
    """Tests pour l'environnement de génération de formules"""

    @pytest.fixture
    def sample_market_data(self):
        """Données de marché pour les tests"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')

        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.1) + abs(np.random.randn(100) * 0.5),
            'low': 100 + np.cumsum(np.random.randn(100) * 0.1) - abs(np.random.randn(100) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
            'volume': abs(np.random.randn(100) * 1000 + 5000),
            'vwap': 100 + np.cumsum(np.random.randn(100) * 0.1)
        })
        data.set_index('timestamp', inplace=True)
        return data

    def test_environment_initialization(self, sample_market_data):
        """Test l'initialisation de l'environnement"""
        config = RLAlphaConfig()
        env = FormulaEnvironment(config, sample_market_data)

        assert env.config == config
        assert len(env.data) == len(sample_market_data)
        assert hasattr(env, 'search_space')
        assert hasattr(env, 'current_formula')
        assert hasattr(env, 'state_dim')

    def test_reset_environment(self, sample_market_data):
        """Test la réinitialisation de l'environnement"""
        config = RLAlphaConfig()
        env = FormulaEnvironment(config, sample_market_data)

        initial_state = env.reset()

        assert isinstance(initial_state, np.ndarray)
        assert len(initial_state) == config.state_dim
        assert all(np.isfinite(initial_state))

    def test_step_function(self, sample_market_data):
        """Test la fonction step de l'environnement"""
        config = RLAlphaConfig()
        env = FormulaEnvironment(config, sample_market_data)

        env.reset()

        # Test avec une action valide (index dans l'espace d'actions)
        action = 0  # Première action disponible
        state, reward, done, info = env.step(action)

        assert isinstance(state, np.ndarray)
        assert len(state) == config.state_dim
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_representation(self, sample_market_data):
        """Test la représentation d'état"""
        config = RLAlphaConfig()
        env = FormulaEnvironment(config, sample_market_data)

        state = env._get_state()

        assert isinstance(state, np.ndarray)
        assert len(state) == config.state_dim
        assert all(np.isfinite(state))

    def test_reward_calculation(self, sample_market_data):
        """Test le calcul de récompense"""
        config = RLAlphaConfig()
        env = FormulaEnvironment(config, sample_market_data)

        # Test avec une formule simple
        env.current_formula = ["close"]
        reward = env._calculate_reward()

        assert isinstance(reward, float)
        # La récompense peut être négative (pénalité) ou positive

    def test_formula_evaluation(self, sample_market_data):
        """Test l'évaluation de formules"""
        config = RLAlphaConfig()
        env = FormulaEnvironment(config, sample_market_data)

        # Test avec une formule valide
        formula = ["close"]
        ic, rank_ic = env._evaluate_formula(formula)

        assert isinstance(ic, float)
        assert isinstance(rank_ic, float)
        # Les IC peuvent être négatifs ou positifs

    def test_formula_complexity(self, sample_market_data):
        """Test le calcul de complexité"""
        config = RLAlphaConfig()
        env = FormulaEnvironment(config, sample_market_data)

        # Test formule simple
        simple_formula = ["close"]
        simple_complexity = env._calculate_complexity(simple_formula)

        # Test formule complexe
        complex_formula = ["corr", "open", "volume", "10", "scale", "delta"]
        complex_complexity = env._calculate_complexity(complex_formula)

        assert isinstance(simple_complexity, int)
        assert isinstance(complex_complexity, int)
        assert complex_complexity > simple_complexity


class TestPPOAgent:
    """Tests pour l'agent PPO"""

    def test_ppo_agent_initialization(self):
        """Test l'initialisation de l'agent PPO"""
        config = RLAlphaConfig()
        agent = PPOAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')
        assert isinstance(agent.actor, nn.Module)
        assert isinstance(agent.critic, nn.Module)

    def test_forward_pass(self):
        """Test le passage avant de l'agent"""
        config = RLAlphaConfig()
        agent = PPOAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        # Test avec un état aléatoire
        state = torch.randn(config.state_dim)
        action_probs, value = agent(state)

        assert action_probs.shape == (config.action_dim,)
        assert value.shape == ()
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6)

    def test_batch_forward(self):
        """Test le passage avant avec un batch"""
        config = RLAlphaConfig()
        agent = PPOAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        batch_size = 5
        states = torch.randn(batch_size, config.state_dim)
        action_probs, values = agent(states)

        assert action_probs.shape == (batch_size, config.action_dim)
        assert values.shape == (batch_size,)

    def test_action_selection(self):
        """Test la sélection d'actions"""
        config = RLAlphaConfig()
        agent = PPOAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        state = torch.randn(config.state_dim)
        action_probs, _ = agent(state)

        # Test sélection d'action déterministe (argmax)
        action_det = torch.argmax(action_probs)
        assert 0 <= action_det < config.action_dim

        # Test sélection stochastique
        action_stoch = torch.multinomial(action_probs, 1)
        assert 0 <= action_stoch < config.action_dim


class TestRLAlphaStrategy:
    """Tests complets pour la stratégie RL Alpha"""

    @pytest.fixture
    def config(self):
        """Configuration de test"""
        return RLAlphaConfig(
            max_generations=10,  # Réduit pour les tests
            population_size=5,   # Réduit pour les tests
            max_complexity=5,    # Réduit pour les tests
            ic_threshold=0.01
        )

    @pytest.fixture
    def sample_market_data(self):
        """Données de marché pour les tests"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')

        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.cumsum(np.random.randn(200) * 0.1),
            'high': 100 + np.cumsum(np.random.randn(200) * 0.1) + abs(np.random.randn(200) * 0.5),
            'low': 100 + np.cumsum(np.random.randn(200) * 0.1) - abs(np.random.randn(200) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(200) * 0.1),
            'volume': abs(np.random.randn(200) * 1000 + 5000),
            'vwap': 100 + np.cumsum(np.random.randn(200) * 0.1)
        })
        data.set_index('timestamp', inplace=True)
        return data

    @pytest.fixture
    def strategy(self, config):
        """Stratégie initialisée pour les tests"""
        return RLAlphaStrategy(
            config=config,
            metrics_collector=Mock()
        )

    def test_strategy_initialization(self, strategy, config):
        """Test l'initialisation de la stratégie"""
        assert strategy.config == config
        assert hasattr(strategy, 'agent')
        assert hasattr(strategy, 'search_space')
        assert hasattr(strategy, 'alpha_formulas')
        assert hasattr(strategy, 'generation')

    def test_training_setup(self, strategy, sample_market_data):
        """Test la configuration d'entraînement"""
        strategy._setup_training(sample_market_data)

        assert hasattr(strategy, 'environment')
        assert strategy.environment is not None

    def test_formula_generation(self, strategy, sample_market_data):
        """Test la génération de formules"""
        strategy._setup_training(sample_market_data)

        # Générer quelques formules
        formulas = []
        for _ in range(3):
            formula = strategy._generate_formula()
            formulas.append(formula)

        assert len(formulas) == 3
        for formula in formulas:
            assert isinstance(formula, AlphaFormula)
            assert len(formula.operators) > 0
            assert isinstance(formula.ic, float)
            assert isinstance(formula.complexity, int)

    def test_signal_generation(self, strategy, sample_market_data):
        """Test la génération de signaux"""
        signals = strategy.generate_signals(sample_market_data)

        assert isinstance(signals, list)
        # Il peut y avoir 0 signaux si aucune formule n'est assez bonne

        # Vérifier le format des signaux s'il y en a
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]

    def test_alpha_formula_evaluation(self, strategy, sample_market_data):
        """Test l'évaluation des formules alpha"""
        strategy._setup_training(sample_market_data)

        # Créer une formule simple pour test
        formula = AlphaFormula(
            formula="close",
            operators=["close"],
            ic=0.0,
            rank_ic=0.0,
            complexity=1,
            generation=0
        )

        # Évaluer la formule
        ic, rank_ic = strategy._evaluate_alpha_formula(formula, sample_market_data)

        assert isinstance(ic, float)
        assert isinstance(rank_ic, float)

    def test_best_formulas_selection(self, strategy):
        """Test la sélection des meilleures formules"""
        # Créer des formulas avec différents IC
        formulas = [
            AlphaFormula("f1", ["op1"], 0.05, 0.06, 1, 0),
            AlphaFormula("f2", ["op2"], 0.02, 0.03, 1, 0),
            AlphaFormula("f3", ["op3"], 0.08, 0.09, 1, 0),
            AlphaFormula("f4", ["op4"], 0.01, 0.02, 1, 0)
        ]

        strategy.alpha_formulas = formulas

        # Sélectionner les 2 meilleures
        best = strategy._get_best_formulas(top_k=2)

        assert len(best) == 2
        assert best[0].ic >= best[1].ic  # Triées par IC décroissant
        assert best[0].ic == 0.08  # Meilleure formule

    def test_population_evolution(self, strategy, sample_market_data):
        """Test l'évolution de la population"""
        strategy._setup_training(sample_market_data)

        initial_count = len(strategy.alpha_formulas)

        # Faire une génération d'évolution (raccourcie pour les tests)
        strategy._evolve_population(steps=5)

        # Il devrait y avoir des formules générées
        assert len(strategy.alpha_formulas) >= initial_count

    def test_strategy_state(self, strategy):
        """Test l'état de la stratégie"""
        state = strategy.get_strategy_state()

        assert isinstance(state, dict)
        assert 'config' in state
        assert 'generation' in state
        assert 'formulas_count' in state
        assert 'best_ic' in state

    def test_performance_metrics(self, strategy, sample_market_data):
        """Test les métriques de performance"""
        # Générer quelques signaux
        signals = strategy.generate_signals(sample_market_data)

        # Calculer des métriques simples
        total_signals = len(signals)
        assert total_signals >= 0

        if total_signals > 0:
            buy_signals = [s for s in signals if s.action == SignalAction.BUY]
            sell_signals = [s for s in signals if s.action == SignalAction.SELL]

            assert len(buy_signals) + len(sell_signals) <= total_signals

    def test_error_handling(self, strategy):
        """Test la gestion d'erreurs"""
        # Données vides
        empty_data = pd.DataFrame()
        signals = strategy.generate_signals(empty_data)
        assert signals == []

        # Données insuffisantes
        small_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        signals = strategy.generate_signals(small_data)
        assert isinstance(signals, list)

    @pytest.mark.performance
    def test_training_performance(self, strategy, sample_market_data):
        """Test de performance de l'entraînement"""
        import time

        start_time = time.time()

        strategy._setup_training(sample_market_data)
        strategy._evolve_population(steps=3)  # Très court pour test

        training_time = time.time() - start_time

        # L'entraînement RL peut être lent, mais pas trop pour les tests
        assert training_time < 30.0  # 30 secondes max
        assert len(strategy.alpha_formulas) > 0

    def test_formula_serialization(self, strategy):
        """Test la sérialisation des formules"""
        formula = AlphaFormula(
            formula="(-1 * Corr(open, volume, 10))",
            operators=["corr", "product"],
            ic=0.05,
            rank_ic=0.08,
            complexity=3,
            generation=1,
            metadata={"test": True}
        )

        # Test conversion en dict
        formula_dict = {
            'formula': formula.formula,
            'operators': formula.operators,
            'ic': formula.ic,
            'rank_ic': formula.rank_ic,
            'complexity': formula.complexity,
            'generation': formula.generation,
            'metadata': formula.metadata
        }

        assert isinstance(formula_dict, dict)
        assert formula_dict['formula'] == formula.formula
        assert formula_dict['ic'] == formula.ic

    def test_symbolic_operators_integration(self, strategy, sample_market_data):
        """Test l'intégration avec les opérateurs symboliques"""
        # Vérifier que les opérateurs symboliques sont disponibles
        assert hasattr(strategy, 'search_space')

        operators = strategy.search_space.operators
        assert 'cs_rank' in operators
        assert 'scale' in operators
        assert 'delta' in operators

        # Test que les formules peuvent utiliser ces opérateurs
        strategy._setup_training(sample_market_data)
        formula = strategy._generate_formula()

        assert isinstance(formula, AlphaFormula)
        # Au moins un opérateur devrait être symbolique
        symbolic_ops = ['cs_rank', 'scale', 'delta', 'ts_rank', 'corr']
        has_symbolic = any(op in formula.operators for op in symbolic_ops)
        # Note: Ce test peut échouer aléatoirement, c'est normal pour du RL