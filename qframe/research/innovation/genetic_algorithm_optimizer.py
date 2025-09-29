"""
Genetic Algorithm Optimizer
===========================

Advanced genetic algorithm for strategy parameter optimization.
Multi-objective optimization with sophisticated selection and crossover.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
import asyncio
import json
import random
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from qframe.core.container import injectable
from qframe.core.interfaces import Strategy, DataProvider
from qframe.core.config import FrameworkConfig

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Méthodes de sélection génétique"""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    ELITIST = "elitist"
    NSGA2 = "nsga2"  # Non-dominated Sorting Genetic Algorithm II


class CrossoverMethod(Enum):
    """Méthodes de croisement"""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND_ALPHA = "blend_alpha"
    SIMULATED_BINARY = "simulated_binary"


class MutationMethod(Enum):
    """Méthodes de mutation"""
    RANDOM = "random"
    GAUSSIAN = "gaussian"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"
    CREEP = "creep"


@dataclass
class Parameter:
    """Définition d'un paramètre à optimiser"""
    name: str
    param_type: str  # "float", "int", "categorical"
    min_value: Optional[Union[float, int]] = None
    max_value: Optional[Union[float, int]] = None
    possible_values: Optional[List[Any]] = None
    step: Optional[Union[float, int]] = None
    importance: float = 1.0  # Poids pour l'optimisation


@dataclass
class ObjectiveFunction:
    """Fonction objectif pour l'optimisation"""
    name: str
    weight: float
    maximize: bool = True
    constraint: Optional[Callable] = None


@dataclass
class Individual:
    """Individu dans la population génétique"""
    genes: Dict[str, Any]  # Paramètres de l'individu
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0  # Fitness globale
    rank: int = 0
    crowding_distance: float = 0.0

    # Métadonnées
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    individual_id: str = field(default_factory=lambda: f"ind_{int(time.time() * 1000000) % 1000000}")

    # Résultats d'évaluation
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    backtest_results: Optional[Dict[str, float]] = None
    validation_score: float = 0.0


@dataclass
class Population:
    """Population d'individus"""
    individuals: List[Individual]
    generation: int = 0
    size: int = 0

    # Statistiques de la population
    best_fitness: float = float('-inf')
    average_fitness: float = 0.0
    worst_fitness: float = float('inf')
    diversity_score: float = 0.0

    # Historique
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    best_individual: Optional[Individual] = None

    def __post_init__(self):
        self.size = len(self.individuals)


@dataclass
class OptimizationConfig:
    """Configuration de l'optimisation génétique"""
    population_size: int = 100
    max_generations: int = 50
    elite_size: int = 10
    tournament_size: int = 5

    # Probabilités
    crossover_probability: float = 0.8
    mutation_probability: float = 0.1
    elite_probability: float = 0.1

    # Méthodes
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN

    # Paramètres spécifiques aux méthodes
    mutation_strength: float = 0.1
    crossover_alpha: float = 0.5
    adaptive_mutation: bool = True

    # Critères d'arrêt
    convergence_threshold: float = 1e-6
    stagnation_generations: int = 10
    target_fitness: Optional[float] = None

    # Multi-objectifs
    multi_objective: bool = False
    pareto_front_size: int = 50


@injectable
class GeneticAlgorithmOptimizer:
    """
    Optimiseur génétique avancé pour stratégies quantitatives.

    Fonctionnalités:
    - Optimisation multi-objectifs (NSGA-II)
    - Sélection, croisement et mutation sophistiqués
    - Adaptation dynamique des paramètres
    - Évaluation parallèle des fitness
    - Conservation de la diversité génétique
    - Contraintes et pénalités
    """

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.current_population: Optional[Population] = None
        self.optimization_history: List[Population] = []
        self.executor = ThreadPoolExecutor(max_workers=8)

        # Configuration par défaut
        self.default_config = OptimizationConfig()

        logger.info("Genetic Algorithm Optimizer initialized")

    async def optimize_strategy_parameters(
        self,
        strategy_class: type,
        parameters: List[Parameter],
        objective_functions: List[ObjectiveFunction],
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        optimization_config: Optional[OptimizationConfig] = None
    ) -> Dict[str, Any]:
        """
        Optimise les paramètres d'une stratégie avec algorithme génétique.

        Args:
            strategy_class: Classe de stratégie à optimiser
            parameters: Liste des paramètres à optimiser
            objective_functions: Fonctions objectifs à maximiser/minimiser
            training_data: Données d'entraînement
            validation_data: Données de validation optionnelles
            optimization_config: Configuration de l'optimisation

        Returns:
            Résultats d'optimisation avec meilleurs paramètres
        """
        config = optimization_config or self.default_config

        logger.info(f"Starting genetic optimization for {strategy_class.__name__}")
        logger.info(f"Parameters: {[p.name for p in parameters]}")
        logger.info(f"Objectives: {[obj.name for obj in objective_functions]}")

        start_time = time.time()

        # Initialiser la population
        initial_population = await self._create_initial_population(
            parameters, config.population_size
        )

        # Évaluer la population initiale
        await self._evaluate_population(
            initial_population, strategy_class, objective_functions,
            training_data, validation_data
        )

        self.current_population = initial_population
        self.optimization_history.append(copy.deepcopy(initial_population))

        best_fitness_history = []
        stagnation_counter = 0

        # Boucle évolutionnaire principale
        for generation in range(config.max_generations):
            logger.info(f"Generation {generation + 1}/{config.max_generations}")

            # Créer nouvelle génération
            new_population = await self._evolve_population(
                self.current_population, parameters, config
            )

            # Évaluer la nouvelle population
            await self._evaluate_population(
                new_population, strategy_class, objective_functions,
                training_data, validation_data
            )

            # Sélection de la population suivante
            if config.multi_objective:
                self.current_population = await self._nsga2_selection(
                    self.current_population, new_population, config
                )
            else:
                self.current_population = await self._elitist_selection(
                    self.current_population, new_population, config
                )

            # Mettre à jour les statistiques
            await self._update_population_stats(self.current_population)
            self.optimization_history.append(copy.deepcopy(self.current_population))

            # Vérifier la convergence
            best_fitness_history.append(self.current_population.best_fitness)

            if len(best_fitness_history) > config.stagnation_generations:
                recent_improvement = (
                    best_fitness_history[-1] - best_fitness_history[-config.stagnation_generations]
                )
                if abs(recent_improvement) < config.convergence_threshold:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

                if stagnation_counter >= config.stagnation_generations:
                    logger.info(f"Convergence atteinte à la génération {generation + 1}")
                    break

            # Adaptation dynamique des paramètres
            if config.adaptive_mutation:
                config.mutation_probability = self._adaptive_mutation_rate(
                    generation, config.max_generations, stagnation_counter
                )

            logger.info(f"Best fitness: {self.current_population.best_fitness:.6f}")
            logger.info(f"Average fitness: {self.current_population.average_fitness:.6f}")
            logger.info(f"Diversity: {self.current_population.diversity_score:.6f}")

        # Préparer les résultats
        optimization_time = time.time() - start_time
        results = await self._prepare_optimization_results(
            parameters, objective_functions, optimization_time, config
        )

        logger.info(f"Genetic optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best fitness achieved: {self.current_population.best_fitness:.6f}")

        return results

    async def multi_objective_optimization(
        self,
        strategy_class: type,
        parameters: List[Parameter],
        objective_functions: List[ObjectiveFunction],
        training_data: pd.DataFrame,
        pareto_front_size: int = 50
    ) -> Dict[str, Any]:
        """
        Optimisation multi-objectifs avec NSGA-II.

        Returns:
            Front de Pareto avec solutions non-dominées
        """
        config = OptimizationConfig(
            multi_objective=True,
            pareto_front_size=pareto_front_size,
            selection_method=SelectionMethod.NSGA2
        )

        return await self.optimize_strategy_parameters(
            strategy_class, parameters, objective_functions, training_data,
            optimization_config=config
        )

    async def adaptive_optimization(
        self,
        strategy_class: type,
        parameters: List[Parameter],
        objective_functions: List[ObjectiveFunction],
        data_windows: List[pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Optimisation adaptative sur plusieurs fenêtres temporelles.

        Args:
            data_windows: Liste de fenêtres de données temporelles

        Returns:
            Paramètres optimaux adaptatifs
        """
        logger.info("Starting adaptive optimization across time windows")

        adaptive_results = {}
        all_optimal_params = []

        for i, window_data in enumerate(data_windows):
            logger.info(f"Optimizing window {i + 1}/{len(data_windows)}")

            # Optimisation pour cette fenêtre
            window_results = await self.optimize_strategy_parameters(
                strategy_class, parameters, objective_functions, window_data
            )

            adaptive_results[f"window_{i}"] = window_results
            all_optimal_params.append(window_results["best_parameters"])

        # Analyser la stabilité des paramètres
        stability_analysis = await self._analyze_parameter_stability(
            parameters, all_optimal_params
        )

        # Calculer paramètres consensuels
        consensus_parameters = await self._calculate_consensus_parameters(
            parameters, all_optimal_params
        )

        adaptive_results.update({
            "stability_analysis": stability_analysis,
            "consensus_parameters": consensus_parameters,
            "parameter_evolution": all_optimal_params
        })

        return adaptive_results

    async def _create_initial_population(
        self, parameters: List[Parameter], population_size: int
    ) -> Population:
        """Crée la population initiale"""
        individuals = []

        for i in range(population_size):
            genes = {}
            for param in parameters:
                genes[param.name] = self._generate_random_gene(param)

            individual = Individual(genes=genes, generation=0)
            individuals.append(individual)

        return Population(individuals=individuals, generation=0)

    def _generate_random_gene(self, parameter: Parameter) -> Any:
        """Génère une valeur aléatoire pour un paramètre"""
        if parameter.param_type == "float":
            return random.uniform(parameter.min_value, parameter.max_value)
        elif parameter.param_type == "int":
            return random.randint(parameter.min_value, parameter.max_value)
        elif parameter.param_type == "categorical":
            return random.choice(parameter.possible_values)
        else:
            raise ValueError(f"Unknown parameter type: {parameter.param_type}")

    async def _evaluate_population(
        self,
        population: Population,
        strategy_class: type,
        objective_functions: List[ObjectiveFunction],
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ):
        """Évalue tous les individus d'une population en parallèle"""

        # Créer les tâches d'évaluation
        evaluation_tasks = []
        for individual in population.individuals:
            task = self._evaluate_individual(
                individual, strategy_class, objective_functions,
                training_data, validation_data
            )
            evaluation_tasks.append(task)

        # Exécuter en parallèle
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Traiter les résultats
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation failed for individual {i}: {result}")
                # Assigner fitness très faible
                population.individuals[i].fitness = float('-inf')
            else:
                population.individuals[i].fitness_scores = result["fitness_scores"]
                population.individuals[i].fitness = result["combined_fitness"]
                population.individuals[i].evaluation_results = result["details"]

    async def _evaluate_individual(
        self,
        individual: Individual,
        strategy_class: type,
        objective_functions: List[ObjectiveFunction],
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Évalue un individu spécifique"""
        try:
            # Créer instance de stratégie avec les paramètres
            strategy = strategy_class(**individual.genes)

            # Simuler backtesting (simplifié)
            backtest_results = await self._simulate_strategy_backtest(
                strategy, training_data, validation_data
            )

            # Calculer fitness pour chaque objectif
            fitness_scores = {}
            for objective in objective_functions:
                if objective.name in backtest_results:
                    score = backtest_results[objective.name]

                    # Appliquer contrainte si définie
                    if objective.constraint and not objective.constraint(score):
                        score = float('-inf') if objective.maximize else float('inf')

                    fitness_scores[objective.name] = score
                else:
                    fitness_scores[objective.name] = 0.0

            # Combiner les fitness (pondération)
            combined_fitness = sum(
                fitness_scores[obj.name] * obj.weight * (1 if obj.maximize else -1)
                for obj in objective_functions
            )

            return {
                "fitness_scores": fitness_scores,
                "combined_fitness": combined_fitness,
                "details": backtest_results
            }

        except Exception as e:
            logger.error(f"Individual evaluation failed: {e}")
            return {
                "fitness_scores": {obj.name: float('-inf') for obj in objective_functions},
                "combined_fitness": float('-inf'),
                "details": {"error": str(e)}
            }

    async def _simulate_strategy_backtest(
        self,
        strategy: Strategy,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Simule un backtest de stratégie (version simplifiée)"""

        # Simulation de métriques de performance
        # Dans un vrai système, ceci ferait un backtest complet

        returns = training_data["close"].pct_change().dropna()

        # Simuler signals et performance
        n_signals = len(returns) // 20  # Signal tous les 20 periods
        random_signals = np.random.choice([-1, 0, 1], size=n_signals, p=[0.3, 0.4, 0.3])

        # Calculer return simple basé sur les signaux
        signal_returns = []
        for signal in random_signals:
            if signal != 0:
                # Prendre 5 returns suivants
                period_returns = np.random.choice(returns.values, 5, replace=True)
                signal_return = np.sum(period_returns) * signal
                signal_returns.append(signal_return)

        if not signal_returns:
            signal_returns = [0.0]

        signal_returns = np.array(signal_returns)

        # Calculer métriques
        total_return = np.sum(signal_returns)
        volatility = np.std(signal_returns) if len(signal_returns) > 1 else 0.1
        sharpe_ratio = total_return / volatility if volatility > 0 else 0
        max_drawdown = np.min(np.cumsum(signal_returns)) if len(signal_returns) > 0 else 0

        # Ajouter un peu de bruit basé sur les paramètres
        param_quality = sum(
            abs(hash(str(v)) % 100) / 100 for v in strategy.__dict__.values()
            if isinstance(v, (int, float))
        ) / max(len(strategy.__dict__), 1)

        noise_factor = 1 + (param_quality - 0.5) * 0.2  # ±10% based on parameters

        return {
            "total_return": total_return * noise_factor,
            "sharpe_ratio": sharpe_ratio * noise_factor,
            "max_drawdown": abs(max_drawdown),
            "volatility": volatility,
            "win_rate": len([r for r in signal_returns if r > 0]) / len(signal_returns) if signal_returns.size > 0 else 0.5,
            "profit_factor": abs(sum([r for r in signal_returns if r > 0])) / abs(sum([r for r in signal_returns if r < 0])) if sum([r for r in signal_returns if r < 0]) != 0 else 1.0
        }

    async def _evolve_population(
        self,
        population: Population,
        parameters: List[Parameter],
        config: OptimizationConfig
    ) -> Population:
        """Crée une nouvelle génération via sélection, croisement et mutation"""

        new_individuals = []

        # Élitisme - garder les meilleurs
        elite_size = int(config.population_size * config.elite_probability)
        sorted_individuals = sorted(
            population.individuals, key=lambda x: x.fitness, reverse=True
        )
        elites = sorted_individuals[:elite_size]
        new_individuals.extend(copy.deepcopy(elites))

        # Générer le reste via croisement et mutation
        while len(new_individuals) < config.population_size:
            # Sélection des parents
            parent1 = await self._select_parent(population, config)
            parent2 = await self._select_parent(population, config)

            # Croisement
            if random.random() < config.crossover_probability:
                child1, child2 = await self._crossover(parent1, parent2, parameters, config)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # Mutation
            if random.random() < config.mutation_probability:
                await self._mutate(child1, parameters, config)
            if random.random() < config.mutation_probability:
                await self._mutate(child2, parameters, config)

            # Mettre à jour métadonnées
            child1.generation = population.generation + 1
            child2.generation = population.generation + 1
            child1.parent_ids = [parent1.individual_id, parent2.individual_id]
            child2.parent_ids = [parent1.individual_id, parent2.individual_id]

            new_individuals.extend([child1, child2])

        # Tronquer à la taille désirée
        new_individuals = new_individuals[:config.population_size]

        return Population(
            individuals=new_individuals,
            generation=population.generation + 1
        )

    async def _select_parent(
        self, population: Population, config: OptimizationConfig
    ) -> Individual:
        """Sélectionne un parent selon la méthode configurée"""

        if config.selection_method == SelectionMethod.TOURNAMENT:
            return await self._tournament_selection(population, config.tournament_size)
        elif config.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return await self._roulette_wheel_selection(population)
        elif config.selection_method == SelectionMethod.RANK_BASED:
            return await self._rank_based_selection(population)
        else:
            # Sélection aléatoire par défaut
            return random.choice(population.individuals)

    async def _tournament_selection(
        self, population: Population, tournament_size: int
    ) -> Individual:
        """Sélection par tournoi"""
        tournament = random.sample(population.individuals, min(tournament_size, len(population.individuals)))
        return max(tournament, key=lambda x: x.fitness)

    async def _roulette_wheel_selection(self, population: Population) -> Individual:
        """Sélection par roulette"""
        # Ajuster fitness pour qu'elles soient positives
        min_fitness = min(ind.fitness for ind in population.individuals)
        adjusted_fitness = [ind.fitness - min_fitness + 1 for ind in population.individuals]

        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            return random.choice(population.individuals)

        pick = random.uniform(0, total_fitness)
        current = 0
        for i, individual in enumerate(population.individuals):
            current += adjusted_fitness[i]
            if current >= pick:
                return individual

        return population.individuals[-1]

    async def _rank_based_selection(self, population: Population) -> Individual:
        """Sélection basée sur le rang"""
        sorted_individuals = sorted(population.individuals, key=lambda x: x.fitness)
        ranks = list(range(1, len(sorted_individuals) + 1))

        total_rank = sum(ranks)
        pick = random.uniform(0, total_rank)
        current = 0

        for i, individual in enumerate(sorted_individuals):
            current += ranks[i]
            if current >= pick:
                return individual

        return sorted_individuals[-1]

    async def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        parameters: List[Parameter],
        config: OptimizationConfig
    ) -> Tuple[Individual, Individual]:
        """Effectue le croisement entre deux parents"""

        child1_genes = copy.deepcopy(parent1.genes)
        child2_genes = copy.deepcopy(parent2.genes)

        if config.crossover_method == CrossoverMethod.UNIFORM:
            for param in parameters:
                if random.random() < 0.5:
                    child1_genes[param.name], child2_genes[param.name] = (
                        child2_genes[param.name], child1_genes[param.name]
                    )

        elif config.crossover_method == CrossoverMethod.ARITHMETIC:
            alpha = config.crossover_alpha
            for param in parameters:
                if param.param_type in ["float", "int"]:
                    val1, val2 = parent1.genes[param.name], parent2.genes[param.name]
                    child1_genes[param.name] = alpha * val1 + (1 - alpha) * val2
                    child2_genes[param.name] = (1 - alpha) * val1 + alpha * val2

                    # Arrondir pour les entiers
                    if param.param_type == "int":
                        child1_genes[param.name] = int(round(child1_genes[param.name]))
                        child2_genes[param.name] = int(round(child2_genes[param.name]))

        elif config.crossover_method == CrossoverMethod.SINGLE_POINT:
            param_names = [p.name for p in parameters]
            crossover_point = random.randint(1, len(param_names) - 1)

            for i, param_name in enumerate(param_names):
                if i >= crossover_point:
                    child1_genes[param_name], child2_genes[param_name] = (
                        child2_genes[param_name], child1_genes[param_name]
                    )

        child1 = Individual(genes=child1_genes)
        child2 = Individual(genes=child2_genes)

        return child1, child2

    async def _mutate(
        self, individual: Individual, parameters: List[Parameter], config: OptimizationConfig
    ):
        """Effectue la mutation d'un individu"""

        for param in parameters:
            if random.random() < config.mutation_probability:

                if config.mutation_method == MutationMethod.GAUSSIAN:
                    if param.param_type == "float":
                        current_val = individual.genes[param.name]
                        mutation_range = (param.max_value - param.min_value) * config.mutation_strength
                        mutation = np.random.normal(0, mutation_range)
                        new_val = current_val + mutation

                        # Contraindre dans les limites
                        new_val = max(param.min_value, min(param.max_value, new_val))
                        individual.genes[param.name] = new_val

                    elif param.param_type == "int":
                        current_val = individual.genes[param.name]
                        mutation_range = max(1, int((param.max_value - param.min_value) * config.mutation_strength))
                        mutation = int(np.random.normal(0, mutation_range))
                        new_val = current_val + mutation

                        new_val = max(param.min_value, min(param.max_value, new_val))
                        individual.genes[param.name] = new_val

                elif config.mutation_method == MutationMethod.RANDOM:
                    individual.genes[param.name] = self._generate_random_gene(param)

                elif param.param_type == "categorical":
                    individual.genes[param.name] = random.choice(param.possible_values)

    async def _nsga2_selection(
        self,
        current_population: Population,
        new_population: Population,
        config: OptimizationConfig
    ) -> Population:
        """Sélection NSGA-II pour optimisation multi-objectifs"""

        # Combiner les deux populations
        combined_individuals = current_population.individuals + new_population.individuals

        # Tri non-dominé
        fronts = await self._non_dominated_sort(combined_individuals)

        # Sélectionner individus pour la prochaine génération
        next_generation = []
        front_index = 0

        while len(next_generation) + len(fronts[front_index]) <= config.population_size:
            next_generation.extend(fronts[front_index])
            front_index += 1

            if front_index >= len(fronts):
                break

        # Compléter avec crowding distance si nécessaire
        if len(next_generation) < config.population_size and front_index < len(fronts):
            remaining_slots = config.population_size - len(next_generation)
            last_front = fronts[front_index]

            # Calculer crowding distance
            await self._calculate_crowding_distance(last_front)

            # Trier par crowding distance décroissante
            last_front.sort(key=lambda x: x.crowding_distance, reverse=True)
            next_generation.extend(last_front[:remaining_slots])

        return Population(
            individuals=next_generation,
            generation=current_population.generation + 1
        )

    async def _non_dominated_sort(self, individuals: List[Individual]) -> List[List[Individual]]:
        """Tri non-dominé pour NSGA-II"""
        fronts = []
        domination_counts = {}
        dominated_solutions = {}

        # Initialiser
        for individual in individuals:
            domination_counts[individual.individual_id] = 0
            dominated_solutions[individual.individual_id] = []

        # Calculer dominance
        for i, ind1 in enumerate(individuals):
            for j, ind2 in enumerate(individuals):
                if i != j:
                    if self._dominates(ind1, ind2):
                        dominated_solutions[ind1.individual_id].append(ind2.individual_id)
                    elif self._dominates(ind2, ind1):
                        domination_counts[ind1.individual_id] += 1

        # Premier front (non-dominés)
        current_front = []
        for individual in individuals:
            if domination_counts[individual.individual_id] == 0:
                individual.rank = 0
                current_front.append(individual)

        fronts.append(current_front)

        # Fronts suivants
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []

            for individual in fronts[front_index]:
                for dominated_id in dominated_solutions[individual.individual_id]:
                    dominated_individual = next(
                        ind for ind in individuals if ind.individual_id == dominated_id
                    )
                    domination_counts[dominated_id] -= 1

                    if domination_counts[dominated_id] == 0:
                        dominated_individual.rank = front_index + 1
                        next_front.append(dominated_individual)

            if next_front:
                fronts.append(next_front)
            else:
                break

            front_index += 1

        return fronts

    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Vérifie si ind1 domine ind2 (toutes objectives meilleures ou égales, au moins une strictement meilleure)"""
        better_in_any = False

        for objective_name, score1 in ind1.fitness_scores.items():
            score2 = ind2.fitness_scores.get(objective_name, 0)

            if score1 < score2:  # Assumant maximisation
                return False
            elif score1 > score2:
                better_in_any = True

        return better_in_any

    async def _calculate_crowding_distance(self, front: List[Individual]):
        """Calcule la crowding distance pour un front"""

        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return

        # Initialiser distances
        for individual in front:
            individual.crowding_distance = 0

        # Pour chaque objectif
        objective_names = list(front[0].fitness_scores.keys())

        for objective_name in objective_names:
            # Trier par cet objectif
            front.sort(key=lambda x: x.fitness_scores[objective_name])

            # Distance infinie pour les extrêmes
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Range de l'objectif
            max_obj = front[-1].fitness_scores[objective_name]
            min_obj = front[0].fitness_scores[objective_name]
            obj_range = max_obj - min_obj

            if obj_range == 0:
                continue

            # Calculer distances
            for i in range(1, len(front) - 1):
                if front[i].crowding_distance != float('inf'):
                    distance = (
                        front[i + 1].fitness_scores[objective_name] -
                        front[i - 1].fitness_scores[objective_name]
                    ) / obj_range
                    front[i].crowding_distance += distance

    async def _elitist_selection(
        self,
        current_population: Population,
        new_population: Population,
        config: OptimizationConfig
    ) -> Population:
        """Sélection élitiste simple"""

        # Combiner toutes les populations
        all_individuals = current_population.individuals + new_population.individuals

        # Trier par fitness
        all_individuals.sort(key=lambda x: x.fitness, reverse=True)

        # Prendre les meilleurs
        selected = all_individuals[:config.population_size]

        return Population(
            individuals=selected,
            generation=current_population.generation + 1
        )

    async def _update_population_stats(self, population: Population):
        """Met à jour les statistiques de la population"""
        if not population.individuals:
            return

        fitness_values = [ind.fitness for ind in population.individuals]

        population.best_fitness = max(fitness_values)
        population.average_fitness = np.mean(fitness_values)
        population.worst_fitness = min(fitness_values)

        # Calculer diversité (variance des fitness)
        population.diversity_score = np.std(fitness_values) if len(fitness_values) > 1 else 0

        # Mettre à jour historique
        population.fitness_history.append(population.best_fitness)
        population.diversity_history.append(population.diversity_score)

        # Meilleur individu
        population.best_individual = max(population.individuals, key=lambda x: x.fitness)

    def _adaptive_mutation_rate(
        self, current_generation: int, max_generations: int, stagnation_counter: int
    ) -> float:
        """Calcule un taux de mutation adaptatif"""

        # Taux de base décroissant avec les générations
        base_rate = 0.1 * (1 - current_generation / max_generations)

        # Augmenter si stagnation
        stagnation_boost = min(stagnation_counter * 0.02, 0.1)

        return min(base_rate + stagnation_boost, 0.3)

    async def _prepare_optimization_results(
        self,
        parameters: List[Parameter],
        objective_functions: List[ObjectiveFunction],
        optimization_time: float,
        config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Prépare les résultats d'optimisation"""

        if not self.current_population or not self.current_population.best_individual:
            return {"error": "No valid results"}

        best_individual = self.current_population.best_individual

        # Historique de convergence
        convergence_history = []
        for i, population in enumerate(self.optimization_history):
            convergence_history.append({
                "generation": i,
                "best_fitness": population.best_fitness,
                "average_fitness": population.average_fitness,
                "diversity": population.diversity_score
            })

        # Analyse de Pareto (si multi-objectifs)
        pareto_front = []
        if config.multi_objective and len(self.optimization_history) > 0:
            pareto_front = await self._extract_pareto_front(self.current_population)

        return {
            "best_parameters": best_individual.genes,
            "best_fitness": best_individual.fitness,
            "best_fitness_scores": best_individual.fitness_scores,
            "convergence_history": convergence_history,
            "pareto_front": pareto_front,
            "optimization_time": optimization_time,
            "total_generations": len(self.optimization_history),
            "final_population_size": len(self.current_population.individuals),
            "parameter_ranges": {p.name: (p.min_value, p.max_value) for p in parameters},
            "objectives": [{"name": obj.name, "weight": obj.weight, "maximize": obj.maximize} for obj in objective_functions],
            "algorithm_config": {
                "population_size": config.population_size,
                "selection_method": config.selection_method.value,
                "crossover_method": config.crossover_method.value,
                "mutation_method": config.mutation_method.value
            }
        }

    async def _extract_pareto_front(self, population: Population) -> List[Dict[str, Any]]:
        """Extrait le front de Pareto de la population"""
        # Tri non-dominé pour identifier le premier front
        fronts = await self._non_dominated_sort(population.individuals)

        if not fronts:
            return []

        pareto_front = []
        for individual in fronts[0]:  # Premier front = Pareto optimal
            pareto_front.append({
                "parameters": individual.genes,
                "fitness_scores": individual.fitness_scores,
                "combined_fitness": individual.fitness
            })

        return pareto_front

    async def _analyze_parameter_stability(
        self,
        parameters: List[Parameter],
        all_optimal_params: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyse la stabilité des paramètres optimaux"""

        stability_analysis = {}

        for param in parameters:
            param_values = [params[param.name] for params in all_optimal_params]

            if param.param_type in ["float", "int"]:
                stability_analysis[param.name] = {
                    "mean": np.mean(param_values),
                    "std": np.std(param_values),
                    "min": np.min(param_values),
                    "max": np.max(param_values),
                    "coefficient_of_variation": np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else float('inf')
                }
            else:
                # Pour les paramètres catégoriels
                value_counts = {}
                for value in param_values:
                    value_counts[value] = value_counts.get(value, 0) + 1

                stability_analysis[param.name] = {
                    "value_distribution": value_counts,
                    "most_frequent": max(value_counts, key=value_counts.get),
                    "frequency": max(value_counts.values()) / len(param_values)
                }

        return stability_analysis

    async def _calculate_consensus_parameters(
        self,
        parameters: List[Parameter],
        all_optimal_params: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calcule des paramètres de consensus"""

        consensus_params = {}

        for param in parameters:
            param_values = [params[param.name] for params in all_optimal_params]

            if param.param_type in ["float", "int"]:
                # Utiliser la médiane pour plus de robustesse
                consensus_value = np.median(param_values)
                if param.param_type == "int":
                    consensus_value = int(round(consensus_value))
                consensus_params[param.name] = consensus_value
            else:
                # Pour les catégoriels, prendre le plus fréquent
                value_counts = {}
                for value in param_values:
                    value_counts[value] = value_counts.get(value, 0) + 1
                consensus_params[param.name] = max(value_counts, key=value_counts.get)

        return consensus_params