"""
Innovation Engine
================

Automated strategy generation and research innovation platform.
"""

from .auto_strategy_generator import (
    AutoStrategyGenerator,
    StrategyTemplate,
    GenerationConfig,
    StrategyComponent,
    StrategyType
)
from .research_paper_integrator import (
    ResearchPaperIntegrator,
    PaperImplementation,
    ImplementationStatus,
    PaperMetadata,
    PaperType,
    MethodologyExtraction
)
from .ab_testing_framework import (
    ABTestingFramework,
    ABTest,
    TestResult,
    TestGroup,
    TestStatus,
    TestType,
    StatisticalTest
)
from .genetic_algorithm_optimizer import (
    GeneticAlgorithmOptimizer,
    Individual,
    Population,
    Parameter,
    ObjectiveFunction,
    OptimizationConfig,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod
)

__all__ = [
    # Auto Strategy Generator
    "AutoStrategyGenerator",
    "StrategyTemplate",
    "GenerationConfig",
    "StrategyComponent",
    "StrategyType",

    # Research Paper Integrator
    "ResearchPaperIntegrator",
    "PaperImplementation",
    "ImplementationStatus",
    "PaperMetadata",
    "PaperType",
    "MethodologyExtraction",

    # A/B Testing Framework
    "ABTestingFramework",
    "ABTest",
    "TestResult",
    "TestGroup",
    "TestStatus",
    "TestType",
    "StatisticalTest",

    # Genetic Algorithm Optimizer
    "GeneticAlgorithmOptimizer",
    "Individual",
    "Population",
    "Parameter",
    "ObjectiveFunction",
    "OptimizationConfig",
    "SelectionMethod",
    "CrossoverMethod",
    "MutationMethod"
]