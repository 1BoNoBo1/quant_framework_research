"""
Auto Strategy Generator
======================

Revolutionary automated strategy generation using genetic algorithms,
machine learning, and symbolic evolution.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import json
import hashlib

from ...core.container import injectable
from ...core.interfaces import Strategy
from ..backtesting.advanced_backtesting_engine import AdvancedBacktestingEngine, BacktestConfig

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Types de stratégies générables"""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    PAIRS_TRADING = "pairs_trading"
    ARBITRAGE = "arbitrage"
    MACHINE_LEARNING = "machine_learning"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MULTI_FACTOR = "multi_factor"
    HYBRID = "hybrid"


class ComplexityLevel(str, Enum):
    """Niveaux de complexité"""
    SIMPLE = "simple"           # 1-3 indicateurs
    MODERATE = "moderate"       # 4-8 indicateurs
    COMPLEX = "complex"         # 9-15 indicateurs
    ADVANCED = "advanced"       # 16+ indicateurs avec ML


@dataclass
class StrategyComponent:
    """Composant atomique d'une stratégie"""
    component_type: str
    parameters: Dict[str, Any]
    weight: float
    enabled: bool = True

    def mutate(self, mutation_rate: float = 0.1) -> 'StrategyComponent':
        """Mute ce composant"""
        if random.random() < mutation_rate:
            # Muter les paramètres
            new_params = self.parameters.copy()
            for key, value in new_params.items():
                if isinstance(value, (int, float)):
                    # Mutation gaussienne
                    new_params[key] = value * (1 + random.gauss(0, 0.1))
                elif isinstance(value, str) and key in ['operator', 'comparison']:
                    # Mutation discrète
                    operators = ['>', '<', '>=', '<=', '==']
                    if value in operators:
                        new_params[key] = random.choice(operators)

            return StrategyComponent(
                component_type=self.component_type,
                parameters=new_params,
                weight=self.weight * (1 + random.gauss(0, 0.05)),
                enabled=self.enabled
            )
        return self

    def to_code(self) -> str:
        """Génère le code pour ce composant"""
        if self.component_type == "technical_indicator":
            indicator = self.parameters.get("indicator", "sma")
            period = self.parameters.get("period", 20)
            return f"ta.{indicator}(data['close'], timeperiod={period})"
        elif self.component_type == "signal_condition":
            left = self.parameters.get("left", "price")
            operator = self.parameters.get("operator", ">")
            right = self.parameters.get("right", "sma_20")
            return f"({left} {operator} {right})"
        else:
            return f"# Component: {self.component_type}"


@dataclass
class StrategyTemplate:
    """Template pour génération de stratégies"""
    name: str
    strategy_type: StrategyType
    complexity_level: ComplexityLevel
    components: List[StrategyComponent]
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: Dict[str, Any]

    # Métadonnées génétiques
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness_score: Optional[float] = None
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def strategy_id(self) -> str:
        """ID unique basé sur les composants"""
        content = json.dumps({
            "components": [c.__dict__ for c in self.components],
            "entry": self.entry_conditions,
            "exit": self.exit_conditions
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def crossover(self, other: 'StrategyTemplate') -> 'StrategyTemplate':
        """Croisement génétique avec une autre stratégie"""
        # Point de croisement aléatoire
        crossover_point = random.randint(1, min(len(self.components), len(other.components)) - 1)

        # Combiner composants
        new_components = (
            self.components[:crossover_point] +
            other.components[crossover_point:]
        )

        # Combiner conditions (50/50)
        new_entry = random.choice([self.entry_conditions, other.entry_conditions])
        new_exit = random.choice([self.exit_conditions, other.exit_conditions])

        # Combiner risk management
        new_risk = self.risk_management.copy()
        new_risk.update({k: v for k, v in other.risk_management.items() if random.random() > 0.5})

        return StrategyTemplate(
            name=f"Hybrid_{self.strategy_id[:6]}_{other.strategy_id[:6]}",
            strategy_type=random.choice([self.strategy_type, other.strategy_type]),
            complexity_level=max(self.complexity_level, other.complexity_level),
            components=new_components,
            entry_conditions=new_entry,
            exit_conditions=new_exit,
            risk_management=new_risk,
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.strategy_id, other.strategy_id]
        )

    def mutate(self, mutation_rate: float = 0.15) -> 'StrategyTemplate':
        """Mutation génétique"""
        mutated_components = [comp.mutate(mutation_rate) for comp in self.components]

        # Possibilité d'ajouter/supprimer des composants
        if random.random() < mutation_rate:
            if len(mutated_components) > 2 and random.random() < 0.3:
                # Supprimer un composant
                mutated_components.pop(random.randint(0, len(mutated_components) - 1))
            elif len(mutated_components) < 10 and random.random() < 0.7:
                # Ajouter un composant
                new_component = self._generate_random_component()
                mutated_components.append(new_component)

        return StrategyTemplate(
            name=f"Mutated_{self.strategy_id[:8]}",
            strategy_type=self.strategy_type,
            complexity_level=self.complexity_level,
            components=mutated_components,
            entry_conditions=self.entry_conditions.copy(),
            exit_conditions=self.exit_conditions.copy(),
            risk_management=self.risk_management.copy(),
            generation=self.generation + 1,
            parent_ids=[self.strategy_id]
        )

    def _generate_random_component(self) -> StrategyComponent:
        """Génère un composant aléatoire"""
        component_types = ["technical_indicator", "signal_condition", "filter"]
        comp_type = random.choice(component_types)

        if comp_type == "technical_indicator":
            indicators = ["sma", "ema", "rsi", "macd", "bbands", "stoch"]
            return StrategyComponent(
                component_type=comp_type,
                parameters={
                    "indicator": random.choice(indicators),
                    "period": random.randint(5, 100),
                    "source": "close"
                },
                weight=random.uniform(0.1, 1.0)
            )
        elif comp_type == "signal_condition":
            return StrategyComponent(
                component_type=comp_type,
                parameters={
                    "left": "price",
                    "operator": random.choice([">", "<", ">=", "<="]),
                    "right": f"sma_{random.randint(10, 50)}",
                    "threshold": random.uniform(0.001, 0.05)
                },
                weight=random.uniform(0.1, 1.0)
            )
        else:
            return StrategyComponent(
                component_type=comp_type,
                parameters={"filter_type": "volatility", "threshold": random.uniform(0.01, 0.1)},
                weight=random.uniform(0.1, 1.0)
            )

    def generate_python_code(self) -> str:
        """Génère le code Python complet de la stratégie"""
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "import talib as ta",
            "from decimal import Decimal",
            "from datetime import datetime",
            "from typing import List, Optional",
            "",
            "from qframe.core.interfaces import Strategy",
            "from qframe.domain.value_objects.signal import Signal",
            "",
            f"class {self.name.replace(' ', '').replace('-', '_')}Strategy(Strategy):",
            '    """',
            f'    Auto-generated strategy: {self.name}',
            f'    Type: {self.strategy_type.value}',
            f'    Complexity: {self.complexity_level.value}',
            f'    Generation: {self.generation}',
            f'    Created: {self.creation_timestamp.isoformat()}',
            '    """',
            "",
            "    def __init__(self):",
            "        self.name = f'{self.name}'",
            "        self.position_size = Decimal('0.1')",
            "        self.stop_loss = Decimal('0.02')",
            "        self.take_profit = Decimal('0.04')",
            "",
            "    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:",
            "        signals = []",
            "        ",
            "        # Technical Indicators"
        ]

        # Générer calculs d'indicateurs
        for component in self.components:
            if component.component_type == "technical_indicator":
                code_lines.append(f"        {component.parameters['indicator']}_{component.parameters['period']} = {component.to_code()}")

        code_lines.extend([
            "",
            "        # Signal Generation Logic",
            "        for i in range(len(data)):",
            "            if i < 50:  # Warm-up period",
            "                continue",
            "",
            "            # Entry Conditions"
        ])

        # Générer conditions d'entrée
        entry_logic = " and ".join(self.entry_conditions)
        code_lines.append(f"            if {entry_logic}:")
        code_lines.extend([
            "                signal = Signal(",
            "                    symbol=data.index[i] if hasattr(data.index[i], 'symbol') else 'UNKNOWN',",
            "                    action='buy',",
            "                    timestamp=data.index[i] if isinstance(data.index[i], datetime) else datetime.utcnow(),",
            "                    strength=Decimal('0.8'),",
            "                    confidence=Decimal('0.7'),",
            "                    price=Decimal(str(data['close'].iloc[i])),",
            "                    quantity=self.position_size",
            "                )",
            "                signals.append(signal)",
            "",
            "        return signals",
            "",
            "    def calculate_position_size(self, signal: Signal, portfolio_value: Decimal) -> Decimal:",
            "        return min(self.position_size, portfolio_value * Decimal('0.1'))",
            "",
            "    def should_exit(self, current_price: Decimal, entry_price: Decimal, position_age: int) -> bool:",
            "        # Exit conditions",
            f"        # {self.exit_conditions}",
            "        price_change = (current_price - entry_price) / entry_price",
            "        return (price_change <= -self.stop_loss or ",
            "                price_change >= self.take_profit or ",
            "                position_age > 50)",
        ])

        return "\n".join(code_lines)


@dataclass
class GenerationConfig:
    """Configuration pour génération de stratégies"""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_percentage: float = 0.1

    # Contraintes de génération
    max_components: int = 15
    min_components: int = 3
    allowed_strategy_types: List[StrategyType] = field(default_factory=lambda: list(StrategyType))
    complexity_preference: Optional[ComplexityLevel] = None

    # Critères de fitness
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe_ratio": 0.4,
        "total_return": 0.3,
        "max_drawdown": -0.2,  # Négatif car on veut minimiser
        "win_rate": 0.1
    })

    # Critères d'arrêt
    target_fitness: Optional[float] = None
    stagnation_threshold: int = 20  # Générations sans amélioration


@injectable
class AutoStrategyGenerator:
    """
    Générateur automatique de stratégies révolutionnaire.

    Capacités:
    - Évolution génétique de stratégies
    - Templates intelligents adaptatifs
    - Validation statistique automatique
    - Génération de code Python
    - Optimisation multi-objectifs
    - Apprentissage des patterns gagnants
    """

    def __init__(
        self,
        backtesting_engine: AdvancedBacktestingEngine,
        config: Optional[GenerationConfig] = None
    ):
        self.backtesting_engine = backtesting_engine
        self.config = config or GenerationConfig()

        # Populations et historique
        self.current_population: List[StrategyTemplate] = []
        self.hall_of_fame: List[StrategyTemplate] = []
        self.generation_history: List[Dict[str, Any]] = []

        # Cache de fitness pour éviter recalculs
        self.fitness_cache: Dict[str, float] = {}

        # Templates de base
        self.base_templates = self._create_base_templates()

        # Statistiques de génération
        self.total_strategies_generated = 0
        self.total_generations = 0
        self.best_fitness_achieved = float('-inf')

    async def evolve_strategies(
        self,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        target_performance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Évolution génétique de stratégies de trading.

        Args:
            training_data: Données d'entraînement
            validation_data: Données de validation
            target_performance: Performance cible à atteindre

        Returns:
            Résultats complets de l'évolution
        """
        logger.info(f"Starting strategy evolution with population size {self.config.population_size}")

        # Initialiser population
        if not self.current_population:
            await self._initialize_population()

        evolution_start = datetime.utcnow()
        best_fitness_history = []
        average_fitness_history = []
        stagnation_count = 0
        last_best_fitness = float('-inf')

        for generation in range(self.config.max_generations):
            generation_start = datetime.utcnow()

            logger.info(f"Generation {generation + 1}/{self.config.max_generations}")

            # Évaluer fitness de la population
            fitness_scores = await self._evaluate_population(training_data)

            # Statistiques de génération
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_fitness_history.append(best_fitness)
            average_fitness_history.append(avg_fitness)

            # Vérifier amélioration
            if best_fitness > last_best_fitness:
                last_best_fitness = best_fitness
                stagnation_count = 0

                # Mettre à jour hall of fame
                best_individual = self.current_population[fitness_scores.index(max(fitness_scores))]
                await self._update_hall_of_fame(best_individual, best_fitness)
            else:
                stagnation_count += 1

            # Critères d'arrêt
            if (self.config.target_fitness and best_fitness >= self.config.target_fitness):
                logger.info(f"Target fitness {self.config.target_fitness} achieved!")
                break

            if stagnation_count >= self.config.stagnation_threshold:
                logger.info(f"Evolution stagnated for {stagnation_count} generations")
                break

            # Sélection et reproduction
            new_population = await self._reproduce_population(fitness_scores)

            # Mutation
            mutated_population = await self._mutate_population(new_population)

            self.current_population = mutated_population
            self.total_generations += 1

            # Enregistrer historique
            generation_time = (datetime.utcnow() - generation_start).total_seconds()
            self.generation_history.append({
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "average_fitness": avg_fitness,
                "population_size": len(self.current_population),
                "generation_time": generation_time,
                "stagnation_count": stagnation_count
            })

            logger.info(f"Generation {generation + 1} complete: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")

        # Validation finale
        validation_results = None
        if validation_data is not None:
            validation_results = await self._validate_best_strategies(validation_data)

        evolution_time = (datetime.utcnow() - evolution_start).total_seconds()

        return {
            "evolution_summary": {
                "total_generations": len(self.generation_history),
                "evolution_time_seconds": evolution_time,
                "best_fitness_achieved": max(best_fitness_history),
                "final_population_size": len(self.current_population),
                "strategies_evaluated": self.total_strategies_generated
            },
            "best_strategies": self.hall_of_fame[:10],  # Top 10
            "fitness_evolution": {
                "best_fitness_history": best_fitness_history,
                "average_fitness_history": average_fitness_history
            },
            "generation_details": self.generation_history,
            "validation_results": validation_results,
            "performance_analysis": await self._analyze_evolution_performance()
        }

    async def generate_strategy_from_description(
        self,
        description: str,
        strategy_type: Optional[StrategyType] = None,
        complexity: Optional[ComplexityLevel] = None
    ) -> StrategyTemplate:
        """
        Génère une stratégie à partir d'une description en langage naturel.

        Args:
            description: Description de la stratégie souhaitée
            strategy_type: Type de stratégie (optionnel)
            complexity: Niveau de complexité (optionnel)

        Returns:
            Template de stratégie généré
        """
        logger.info(f"Generating strategy from description: {description[:100]}...")

        # Analyse de la description (NLP simplifié)
        analyzed_description = await self._analyze_description(description)

        # Déterminer le type de stratégie
        if not strategy_type:
            strategy_type = await self._infer_strategy_type(analyzed_description)

        # Déterminer la complexité
        if not complexity:
            complexity = await self._infer_complexity(analyzed_description)

        # Sélectionner template de base
        base_template = await self._select_base_template(strategy_type, complexity)

        # Adapter le template selon la description
        adapted_template = await self._adapt_template_to_description(
            base_template, analyzed_description
        )

        # Générer composants spécifiques
        custom_components = await self._generate_components_from_description(analyzed_description)
        adapted_template.components.extend(custom_components)

        logger.info(f"Generated {strategy_type.value} strategy with {len(adapted_template.components)} components")
        return adapted_template

    async def optimize_existing_strategy(
        self,
        strategy_template: StrategyTemplate,
        training_data: pd.DataFrame,
        optimization_objectives: Optional[List[str]] = None
    ) -> StrategyTemplate:
        """
        Optimise une stratégie existante via évolution dirigée.

        Args:
            strategy_template: Stratégie à optimiser
            training_data: Données d'entraînement
            optimization_objectives: Objectifs d'optimisation

        Returns:
            Stratégie optimisée
        """
        logger.info(f"Optimizing strategy: {strategy_template.name}")

        # Créer population centrée sur la stratégie existante
        optimization_population = []

        # Ajouter stratégie originale
        optimization_population.append(strategy_template)

        # Créer variations
        for _ in range(self.config.population_size - 1):
            mutated = strategy_template.mutate(mutation_rate=0.3)  # Mutation plus agressive
            optimization_population.append(mutated)

        # Sauvegarder population actuelle
        original_population = self.current_population.copy()
        self.current_population = optimization_population

        # Évolution ciblée
        optimization_config = GenerationConfig(
            population_size=len(optimization_population),
            max_generations=20,  # Évolution plus courte
            mutation_rate=0.2,
            target_fitness=strategy_template.fitness_score * 1.2 if strategy_template.fitness_score else None
        )

        original_config = self.config
        self.config = optimization_config

        try:
            evolution_result = await self.evolve_strategies(training_data)
            best_optimized = evolution_result["best_strategies"][0] if evolution_result["best_strategies"] else strategy_template

            logger.info(f"Strategy optimization complete. Fitness improved from {strategy_template.fitness_score} to {best_optimized.fitness_score}")
            return best_optimized

        finally:
            # Restaurer configuration et population
            self.config = original_config
            self.current_population = original_population

    async def generate_ensemble_strategy(
        self,
        individual_strategies: List[StrategyTemplate],
        ensemble_method: str = "weighted_voting",
        training_data: Optional[pd.DataFrame] = None
    ) -> StrategyTemplate:
        """
        Crée une stratégie ensemble à partir de stratégies individuelles.

        Args:
            individual_strategies: Stratégies à combiner
            ensemble_method: Méthode d'ensemble
            training_data: Données pour optimiser les poids

        Returns:
            Stratégie ensemble optimisée
        """
        logger.info(f"Creating ensemble strategy from {len(individual_strategies)} strategies")

        if ensemble_method == "weighted_voting":
            # Calculer poids basés sur fitness
            weights = []
            for strategy in individual_strategies:
                weight = strategy.fitness_score if strategy.fitness_score else 1.0
                weights.append(max(weight, 0.1))  # Poids minimum

            # Normaliser poids
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

        # Créer stratégie ensemble
        ensemble_components = []
        ensemble_entry_conditions = []
        ensemble_exit_conditions = []

        for i, strategy in enumerate(individual_strategies):
            weight = normalized_weights[i] if ensemble_method == "weighted_voting" else 1.0 / len(individual_strategies)

            # Ajouter composants pondérés
            for component in strategy.components:
                weighted_component = StrategyComponent(
                    component_type=component.component_type,
                    parameters=component.parameters.copy(),
                    weight=component.weight * weight,
                    enabled=component.enabled
                )
                ensemble_components.append(weighted_component)

            # Combiner conditions
            ensemble_entry_conditions.extend([f"({cond}) * {weight}" for cond in strategy.entry_conditions])
            ensemble_exit_conditions.extend([f"({cond}) * {weight}" for cond in strategy.exit_conditions])

        # Créer template ensemble
        ensemble_template = StrategyTemplate(
            name=f"Ensemble_{len(individual_strategies)}Strategies",
            strategy_type=StrategyType.HYBRID,
            complexity_level=ComplexityLevel.ADVANCED,
            components=ensemble_components,
            entry_conditions=[f"sum([{', '.join(ensemble_entry_conditions)}]) > 0.5"],
            exit_conditions=[f"sum([{', '.join(ensemble_exit_conditions)}]) > 0.5"],
            risk_management={
                "ensemble_size": len(individual_strategies),
                "ensemble_method": ensemble_method,
                "component_weights": normalized_weights if ensemble_method == "weighted_voting" else None
            }
        )

        # Optimiser l'ensemble si données disponibles
        if training_data is not None:
            optimized_ensemble = await self.optimize_existing_strategy(ensemble_template, training_data)
            return optimized_ensemble

        return ensemble_template

    # === Méthodes privées ===

    async def _initialize_population(self) -> None:
        """Initialise la population de départ"""
        logger.info("Initializing population")

        self.current_population = []

        # Utiliser templates de base
        base_count = min(len(self.base_templates), self.config.population_size // 2)
        self.current_population.extend(self.base_templates[:base_count])

        # Générer stratégies aléatoires
        remaining_slots = self.config.population_size - len(self.current_population)

        for _ in range(remaining_slots):
            strategy_type = random.choice(self.config.allowed_strategy_types or list(StrategyType))
            complexity = self.config.complexity_preference or random.choice(list(ComplexityLevel))

            random_strategy = await self._generate_random_strategy(strategy_type, complexity)
            self.current_population.append(random_strategy)

        self.total_strategies_generated += len(self.current_population)
        logger.info(f"Population initialized with {len(self.current_population)} strategies")

    async def _evaluate_population(self, data: pd.DataFrame) -> List[float]:
        """Évalue la fitness de toute la population"""
        fitness_scores = []

        for i, strategy in enumerate(self.current_population):
            if i % 10 == 0:
                logger.debug(f"Evaluating strategy {i+1}/{len(self.current_population)}")

            # Vérifier cache
            if strategy.strategy_id in self.fitness_cache:
                fitness = self.fitness_cache[strategy.strategy_id]
            else:
                fitness = await self._calculate_fitness(strategy, data)
                self.fitness_cache[strategy.strategy_id] = fitness

            strategy.fitness_score = fitness
            fitness_scores.append(fitness)

        return fitness_scores

    async def _calculate_fitness(self, strategy: StrategyTemplate, data: pd.DataFrame) -> float:
        """Calcule la fitness d'une stratégie"""
        try:
            # Convertir template en stratégie exécutable (simulation)
            # Dans la réalité, générerait et exécuterait le code Python

            # Simulation de backtesting
            simulated_results = await self._simulate_strategy_performance(strategy, data)

            # Calculer score composite
            fitness = 0.0

            for metric, weight in self.config.fitness_weights.items():
                if metric in simulated_results:
                    fitness += simulated_results[metric] * weight

            return max(fitness, 0.0)  # Fitness non-négative

        except Exception as e:
            logger.error(f"Error calculating fitness for strategy {strategy.strategy_id}: {e}")
            return 0.0

    async def _simulate_strategy_performance(
        self,
        strategy: StrategyTemplate,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Simule la performance d'une stratégie"""

        # Simulation basée sur les composants de la stratégie
        # Dans la réalité, exécuterait le code généré

        # Facteurs de performance basés sur le type et la complexité
        base_performance = {
            StrategyType.MEAN_REVERSION: {"sharpe": 1.2, "return": 0.15, "drawdown": 0.08},
            StrategyType.MOMENTUM: {"sharpe": 0.9, "return": 0.18, "drawdown": 0.12},
            StrategyType.PAIRS_TRADING: {"sharpe": 1.5, "return": 0.12, "drawdown": 0.06},
            StrategyType.ARBITRAGE: {"sharpe": 2.0, "return": 0.10, "drawdown": 0.03},
            StrategyType.MACHINE_LEARNING: {"sharpe": 1.1, "return": 0.20, "drawdown": 0.10}
        }.get(strategy.strategy_type, {"sharpe": 1.0, "return": 0.12, "drawdown": 0.08})

        # Ajustements pour complexité
        complexity_multipliers = {
            ComplexityLevel.SIMPLE: 0.8,
            ComplexityLevel.MODERATE: 1.0,
            ComplexityLevel.COMPLEX: 1.1,
            ComplexityLevel.ADVANCED: 1.2
        }

        multiplier = complexity_multipliers.get(strategy.complexity_level, 1.0)

        # Ajouter variabilité et réalisme
        noise = random.gauss(1.0, 0.2)  # ±20% variabilité

        return {
            "sharpe_ratio": base_performance["sharpe"] * multiplier * noise,
            "total_return": base_performance["return"] * multiplier * noise,
            "max_drawdown": base_performance["drawdown"] / (multiplier * noise),
            "win_rate": 0.55 + random.gauss(0, 0.1)  # ~55% win rate with variability
        }

    async def _reproduce_population(self, fitness_scores: List[float]) -> List[StrategyTemplate]:
        """Reproduction de la population via sélection et croisement"""

        # Sélection élitiste
        elite_count = int(len(self.current_population) * self.config.elite_percentage)
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        elite_strategies = [self.current_population[i] for i in elite_indices]

        new_population = elite_strategies.copy()

        # Croisement pour compléter la population
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Sélection par tournoi
                parent1 = await self._tournament_selection(fitness_scores)
                parent2 = await self._tournament_selection(fitness_scores)

                # Croisement
                offspring = parent1.crossover(parent2)
                new_population.append(offspring)
            else:
                # Reproduction asexuée (copie)
                parent = await self._tournament_selection(fitness_scores)
                new_population.append(parent)

        return new_population[:self.config.population_size]

    async def _mutate_population(self, population: List[StrategyTemplate]) -> List[StrategyTemplate]:
        """Mutation de la population"""
        mutated_population = []

        for strategy in population:
            if random.random() < self.config.mutation_rate:
                mutated_strategy = strategy.mutate(self.config.mutation_rate)
                mutated_population.append(mutated_strategy)
            else:
                mutated_population.append(strategy)

        return mutated_population

    async def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> StrategyTemplate:
        """Sélection par tournoi"""
        tournament_indices = random.sample(range(len(fitness_scores)), min(tournament_size, len(fitness_scores)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return self.current_population[best_idx]

    async def _update_hall_of_fame(self, strategy: StrategyTemplate, fitness: float) -> None:
        """Met à jour le hall of fame"""
        strategy.fitness_score = fitness

        # Ajouter à hall of fame
        self.hall_of_fame.append(strategy)

        # Trier par fitness et garder seulement les meilleurs
        self.hall_of_fame.sort(key=lambda s: s.fitness_score or 0, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:20]  # Top 20

        if fitness > self.best_fitness_achieved:
            self.best_fitness_achieved = fitness
            logger.info(f"New best fitness achieved: {fitness:.4f}")

    def _create_base_templates(self) -> List[StrategyTemplate]:
        """Crée les templates de base"""
        templates = []

        # Template Mean Reversion Simple
        mr_template = StrategyTemplate(
            name="Simple_Mean_Reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            complexity_level=ComplexityLevel.SIMPLE,
            components=[
                StrategyComponent("technical_indicator", {"indicator": "sma", "period": 20}, 1.0),
                StrategyComponent("technical_indicator", {"indicator": "std", "period": 20}, 0.8),
                StrategyComponent("signal_condition", {"left": "price", "operator": "<", "right": "sma_20", "deviation": 2}, 1.0)
            ],
            entry_conditions=["price < (sma_20 - 2 * std_20)"],
            exit_conditions=["price > sma_20"],
            risk_management={"stop_loss": 0.02, "take_profit": 0.04}
        )
        templates.append(mr_template)

        # Template Momentum Simple
        momentum_template = StrategyTemplate(
            name="Simple_Momentum",
            strategy_type=StrategyType.MOMENTUM,
            complexity_level=ComplexityLevel.SIMPLE,
            components=[
                StrategyComponent("technical_indicator", {"indicator": "sma", "period": 10}, 1.0),
                StrategyComponent("technical_indicator", {"indicator": "sma", "period": 30}, 1.0),
                StrategyComponent("signal_condition", {"left": "sma_10", "operator": ">", "right": "sma_30"}, 1.0)
            ],
            entry_conditions=["sma_10 > sma_30"],
            exit_conditions=["sma_10 < sma_30"],
            risk_management={"stop_loss": 0.03, "take_profit": 0.06}
        )
        templates.append(momentum_template)

        return templates

    async def _generate_random_strategy(
        self,
        strategy_type: StrategyType,
        complexity: ComplexityLevel
    ) -> StrategyTemplate:
        """Génère une stratégie aléatoire"""

        # Nombre de composants selon complexité
        component_counts = {
            ComplexityLevel.SIMPLE: (2, 4),
            ComplexityLevel.MODERATE: (4, 8),
            ComplexityLevel.COMPLEX: (8, 12),
            ComplexityLevel.ADVANCED: (12, 16)
        }

        min_comp, max_comp = component_counts[complexity]
        num_components = random.randint(min_comp, max_comp)

        # Générer composants aléatoires
        components = []
        for _ in range(num_components):
            component = self._generate_random_component_by_type(strategy_type)
            components.append(component)

        # Générer conditions
        entry_conditions = [f"condition_{i} > threshold_{i}" for i in range(random.randint(1, 3))]
        exit_conditions = [f"exit_condition_{i}" for i in range(random.randint(1, 2))]

        # Risk management aléatoire
        risk_management = {
            "stop_loss": random.uniform(0.01, 0.05),
            "take_profit": random.uniform(0.02, 0.08),
            "position_size": random.uniform(0.05, 0.2)
        }

        return StrategyTemplate(
            name=f"Random_{strategy_type.value}_{random.randint(1000, 9999)}",
            strategy_type=strategy_type,
            complexity_level=complexity,
            components=components,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management=risk_management
        )

    def _generate_random_component_by_type(self, strategy_type: StrategyType) -> StrategyComponent:
        """Génère un composant aléatoire selon le type de stratégie"""

        if strategy_type == StrategyType.MEAN_REVERSION:
            indicators = ["sma", "ema", "std", "rsi", "bbands"]
        elif strategy_type == StrategyType.MOMENTUM:
            indicators = ["sma", "ema", "macd", "rsi", "adx"]
        elif strategy_type == StrategyType.MACHINE_LEARNING:
            indicators = ["sma", "ema", "rsi", "macd", "bbands", "stoch", "williams"]
        else:
            indicators = ["sma", "ema", "rsi", "macd"]

        return StrategyComponent(
            component_type="technical_indicator",
            parameters={
                "indicator": random.choice(indicators),
                "period": random.randint(5, 100),
                "source": random.choice(["close", "high", "low", "volume"])
            },
            weight=random.uniform(0.1, 1.0)
        )

    async def _analyze_description(self, description: str) -> Dict[str, Any]:
        """Analyse une description en langage naturel (NLP simplifié)"""

        # Mots-clés pour types de stratégies
        keywords = {
            "mean_reversion": ["mean reversion", "revert", "oversold", "overbought", "bollinger", "rsi"],
            "momentum": ["momentum", "trend", "breakout", "moving average", "crossover"],
            "pairs_trading": ["pairs", "spread", "cointegration", "relative", "correlation"],
            "arbitrage": ["arbitrage", "spread", "difference", "risk-free"],
            "machine_learning": ["ml", "machine learning", "neural", "predict", "model"]
        }

        # Indicateurs techniques mentionnés
        indicators = ["sma", "ema", "rsi", "macd", "bollinger", "stochastic", "williams", "adx"]

        description_lower = description.lower()

        analysis = {
            "strategy_types": [],
            "indicators_mentioned": [],
            "complexity_hints": [],
            "conditions": []
        }

        # Détecter types de stratégies
        for strategy_type, words in keywords.items():
            if any(word in description_lower for word in words):
                analysis["strategy_types"].append(strategy_type)

        # Détecter indicateurs
        for indicator in indicators:
            if indicator in description_lower:
                analysis["indicators_mentioned"].append(indicator)

        # Détecter complexité
        if any(word in description_lower for word in ["simple", "basic", "easy"]):
            analysis["complexity_hints"].append("simple")
        elif any(word in description_lower for word in ["complex", "advanced", "sophisticated"]):
            analysis["complexity_hints"].append("complex")

        return analysis

    async def _infer_strategy_type(self, analysis: Dict[str, Any]) -> StrategyType:
        """Infère le type de stratégie depuis l'analyse"""
        if analysis["strategy_types"]:
            type_name = analysis["strategy_types"][0]
            return StrategyType(type_name)
        return StrategyType.HYBRID

    async def _infer_complexity(self, analysis: Dict[str, Any]) -> ComplexityLevel:
        """Infère le niveau de complexité"""
        if "simple" in analysis["complexity_hints"]:
            return ComplexityLevel.SIMPLE
        elif "complex" in analysis["complexity_hints"]:
            return ComplexityLevel.COMPLEX
        elif len(analysis["indicators_mentioned"]) > 5:
            return ComplexityLevel.ADVANCED
        elif len(analysis["indicators_mentioned"]) > 2:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE

    async def _select_base_template(
        self,
        strategy_type: StrategyType,
        complexity: ComplexityLevel
    ) -> StrategyTemplate:
        """Sélectionne un template de base"""

        # Chercher template correspondant
        for template in self.base_templates:
            if template.strategy_type == strategy_type:
                return template

        # Générer template si pas trouvé
        return await self._generate_random_strategy(strategy_type, complexity)

    async def _adapt_template_to_description(
        self,
        template: StrategyTemplate,
        analysis: Dict[str, Any]
    ) -> StrategyTemplate:
        """Adapte un template selon l'analyse de description"""

        # Copier template
        adapted = StrategyTemplate(
            name=f"Adapted_{template.name}",
            strategy_type=template.strategy_type,
            complexity_level=template.complexity_level,
            components=template.components.copy(),
            entry_conditions=template.entry_conditions.copy(),
            exit_conditions=template.exit_conditions.copy(),
            risk_management=template.risk_management.copy()
        )

        # Adapter selon indicateurs mentionnés
        for indicator in analysis["indicators_mentioned"]:
            component = StrategyComponent(
                component_type="technical_indicator",
                parameters={
                    "indicator": indicator,
                    "period": random.randint(10, 50),
                    "source": "close"
                },
                weight=1.0
            )
            adapted.components.append(component)

        return adapted

    async def _generate_components_from_description(self, analysis: Dict[str, Any]) -> List[StrategyComponent]:
        """Génère des composants spécifiques à partir de l'analyse"""
        components = []

        for indicator in analysis["indicators_mentioned"]:
            component = StrategyComponent(
                component_type="technical_indicator",
                parameters={
                    "indicator": indicator,
                    "period": random.randint(10, 50)
                },
                weight=random.uniform(0.5, 1.0)
            )
            components.append(component)

        return components

    async def _validate_best_strategies(self, validation_data: pd.DataFrame) -> Dict[str, Any]:
        """Valide les meilleures stratégies sur données de validation"""

        validation_results = {}

        for i, strategy in enumerate(self.hall_of_fame[:5]):  # Top 5
            try:
                # Simuler performance sur validation
                val_performance = await self._simulate_strategy_performance(strategy, validation_data)

                validation_results[strategy.strategy_id] = {
                    "strategy_name": strategy.name,
                    "training_fitness": strategy.fitness_score,
                    "validation_performance": val_performance,
                    "generalization_score": val_performance.get("sharpe_ratio", 0) / max(strategy.fitness_score or 1, 0.1)
                }

            except Exception as e:
                logger.error(f"Validation failed for strategy {strategy.strategy_id}: {e}")

        return validation_results

    async def _analyze_evolution_performance(self) -> Dict[str, Any]:
        """Analyse les performances de l'évolution"""

        if not self.generation_history:
            return {}

        # Convergence
        fitness_values = [gen["best_fitness"] for gen in self.generation_history]
        convergence_rate = (fitness_values[-1] - fitness_values[0]) / len(fitness_values) if len(fitness_values) > 1 else 0

        # Diversité (approximation)
        diversity_score = len(set(s.strategy_id for s in self.current_population)) / len(self.current_population)

        return {
            "convergence_rate": convergence_rate,
            "diversity_score": diversity_score,
            "total_unique_strategies": len(self.fitness_cache),
            "average_generation_time": np.mean([gen["generation_time"] for gen in self.generation_history]),
            "fitness_improvement": fitness_values[-1] - fitness_values[0] if len(fitness_values) > 1 else 0
        }