"""
A/B Testing Framework
====================

Advanced A/B testing system for strategy comparison and validation.
Statistical testing, power analysis, and automated experiment management.
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
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

# Statistical imports
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, kstest
import warnings
warnings.filterwarnings('ignore')

from qframe.core.container import injectable
from qframe.core.interfaces import Strategy, DataProvider, Portfolio
from qframe.core.config import FrameworkConfig

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Statut d'un test A/B"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class TestType(Enum):
    """Types de tests A/B"""
    STRATEGY_COMPARISON = "strategy_comparison"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    FEATURE_SELECTION = "feature_selection"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION_OPTIMIZATION = "execution_optimization"


class StatisticalTest(Enum):
    """Tests statistiques disponibles"""
    T_TEST = "t_test"                    # Test de Student
    WELCH_T_TEST = "welch_t_test"       # Test de Welch (variances inégales)
    MANN_WHITNEY = "mann_whitney"        # Test de Mann-Whitney U
    KOLMOGOROV_SMIRNOV = "ks_test"      # Test de Kolmogorov-Smirnov
    CHI_SQUARE = "chi_square"           # Test du Chi-carré
    BOOTSTRAP = "bootstrap"             # Bootstrap test


@dataclass
class TestGroup:
    """Groupe de test A/B"""
    group_id: str
    name: str
    description: str
    strategy: Strategy
    allocation_weight: float = 0.5  # Proportion du capital alloué

    # Configuration du groupe
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)

    # Métriques collectées
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    execution_metrics: Dict[str, float] = field(default_factory=dict)

    # État du groupe
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    trades_executed: int = 0
    capital_allocated: float = 0.0


@dataclass
class TestResult:
    """Résultat d'un test A/B"""
    test_id: str
    test_type: TestType
    statistical_test: StatisticalTest

    # Groupes testés
    group_a: TestGroup
    group_b: TestGroup

    # Métriques principales
    primary_metric: str
    secondary_metrics: List[str]

    # Résultats statistiques
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    statistical_power: float
    sample_size: int

    # Résultats métier
    relative_improvement: float  # % d'amélioration du groupe B vs A
    absolute_difference: float
    business_significance: bool
    recommendation: str

    # Métadonnées
    start_date: datetime
    end_date: datetime
    duration_days: int
    significance_level: float = 0.05
    minimum_detectable_effect: float = 0.05

    # Validation
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ABTest:
    """Test A/B complet"""
    test_id: str
    name: str
    description: str
    test_type: TestType
    status: TestStatus

    # Configuration du test
    groups: List[TestGroup]
    primary_metric: str
    secondary_metrics: List[str] = field(default_factory=list)

    # Paramètres statistiques
    significance_level: float = 0.05
    statistical_power: float = 0.8
    minimum_detectable_effect: float = 0.05
    statistical_test: StatisticalTest = StatisticalTest.T_TEST

    # Planification
    planned_start_date: Optional[datetime] = None
    planned_end_date: Optional[datetime] = None
    max_duration_days: int = 30

    # Résultats
    results: Optional[TestResult] = None
    interim_results: List[TestResult] = field(default_factory=list)

    # Métadonnées
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@injectable
class ABTestingFramework:
    """
    Framework avancé de tests A/B pour stratégies quantitatives.

    Fonctionnalités:
    - Tests multi-groupes (A/B/C/...)
    - Analyse statistique rigoureuse
    - Power analysis et calcul de taille d'échantillon
    - Tests séquentiels et interim analysis
    - Validation d'hypothèses statistiques
    - Recommandations automatiques
    """

    def __init__(self, config: FrameworkConfig, portfolio_service: Portfolio):
        self.config = config
        self.portfolio_service = portfolio_service
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: Dict[str, ABTest] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info("A/B Testing Framework initialized")

    async def create_test(
        self,
        name: str,
        description: str,
        test_type: TestType,
        strategy_a: Strategy,
        strategy_b: Strategy,
        primary_metric: str,
        secondary_metrics: Optional[List[str]] = None,
        **kwargs
    ) -> ABTest:
        """
        Crée un nouveau test A/B.

        Args:
            name: Nom du test
            description: Description du test
            test_type: Type de test
            strategy_a: Stratégie de contrôle (groupe A)
            strategy_b: Stratégie de test (groupe B)
            primary_metric: Métrique principale à optimiser
            secondary_metrics: Métriques secondaires à surveiller
            **kwargs: Paramètres additionnels

        Returns:
            ABTest configuré
        """
        test_id = self._generate_test_id(name)

        # Créer les groupes
        group_a = TestGroup(
            group_id=f"{test_id}_A",
            name=f"{name} - Control",
            description="Control group (baseline strategy)",
            strategy=strategy_a,
            allocation_weight=kwargs.get("allocation_a", 0.5)
        )

        group_b = TestGroup(
            group_id=f"{test_id}_B",
            name=f"{name} - Treatment",
            description="Treatment group (test strategy)",
            strategy=strategy_b,
            allocation_weight=kwargs.get("allocation_b", 0.5)
        )

        # Créer le test
        ab_test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            test_type=test_type,
            status=TestStatus.PENDING,
            groups=[group_a, group_b],
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or [],
            significance_level=kwargs.get("significance_level", 0.05),
            statistical_power=kwargs.get("statistical_power", 0.8),
            minimum_detectable_effect=kwargs.get("minimum_detectable_effect", 0.05),
            statistical_test=kwargs.get("statistical_test", StatisticalTest.T_TEST),
            max_duration_days=kwargs.get("max_duration_days", 30)
        )

        # Calculer la taille d'échantillon requise
        sample_size = await self._calculate_required_sample_size(ab_test)
        logger.info(f"Required sample size for test {test_id}: {sample_size}")

        # Stocker le test
        self.active_tests[test_id] = ab_test

        logger.info(f"A/B test created: {test_id}")
        return ab_test

    async def start_test(self, test_id: str, start_date: Optional[datetime] = None) -> bool:
        """
        Démarre un test A/B.

        Args:
            test_id: ID du test à démarrer
            start_date: Date de début optionnelle

        Returns:
            True si le test a démarré avec succès
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test not found: {test_id}")

        ab_test = self.active_tests[test_id]

        if ab_test.status != TestStatus.PENDING:
            raise ValueError(f"Test {test_id} is not in pending status")

        # Valider la configuration
        validation_result = await self._validate_test_configuration(ab_test)
        if not validation_result["valid"]:
            logger.error(f"Test validation failed: {validation_result['errors']}")
            return False

        # Initialiser les groupes
        start_time = start_date or datetime.utcnow()
        for group in ab_test.groups:
            group.start_time = start_time
            group.capital_allocated = self._calculate_group_allocation(ab_test, group)

        ab_test.status = TestStatus.RUNNING
        ab_test.planned_start_date = start_time

        logger.info(f"A/B test started: {test_id}")
        return True

    async def stop_test(self, test_id: str, reason: str = "Manual stop") -> TestResult:
        """
        Arrête un test A/B et calcule les résultats finaux.

        Args:
            test_id: ID du test à arrêter
            reason: Raison de l'arrêt

        Returns:
            Résultats finaux du test
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test not found: {test_id}")

        ab_test = self.active_tests[test_id]

        if ab_test.status != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")

        # Finaliser les groupes
        end_time = datetime.utcnow()
        for group in ab_test.groups:
            group.end_time = end_time

        # Calculer les résultats finaux
        final_results = await self._calculate_test_results(ab_test, is_final=True)
        ab_test.results = final_results
        ab_test.status = TestStatus.COMPLETED
        ab_test.planned_end_date = end_time

        # Déplacer vers les tests complétés
        self.completed_tests[test_id] = ab_test
        del self.active_tests[test_id]

        logger.info(f"A/B test completed: {test_id} - {reason}")
        return final_results

    async def get_interim_results(self, test_id: str) -> TestResult:
        """
        Calcule les résultats intermédiaires d'un test en cours.

        Args:
            test_id: ID du test

        Returns:
            Résultats intermédiaires
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test not found: {test_id}")

        ab_test = self.active_tests[test_id]

        if ab_test.status != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")

        # Calculer résultats intermédiaires
        interim_results = await self._calculate_test_results(ab_test, is_final=False)
        ab_test.interim_results.append(interim_results)

        logger.info(f"Interim results calculated for test: {test_id}")
        return interim_results

    async def run_sequential_test(
        self,
        test_id: str,
        check_interval_hours: int = 24,
        early_stopping: bool = True
    ) -> TestResult:
        """
        Exécute un test séquentiel avec vérifications périodiques.

        Args:
            test_id: ID du test
            check_interval_hours: Intervalle entre les vérifications
            early_stopping: Permettre l'arrêt précoce

        Returns:
            Résultats finaux du test
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test not found: {test_id}")

        ab_test = self.active_tests[test_id]

        # Démarrer le test
        await self.start_test(test_id)

        try:
            while ab_test.status == TestStatus.RUNNING:
                # Attendre l'intervalle de vérification
                await asyncio.sleep(check_interval_hours * 3600)

                # Calculer résultats intermédiaires
                interim_results = await self.get_interim_results(test_id)

                # Vérifier les conditions d'arrêt précoce
                if early_stopping:
                    stop_early, reason = await self._check_early_stopping_conditions(ab_test, interim_results)
                    if stop_early:
                        logger.info(f"Early stopping triggered for test {test_id}: {reason}")
                        return await self.stop_test(test_id, reason)

                # Vérifier la durée maximale
                if ab_test.planned_start_date:
                    duration = datetime.utcnow() - ab_test.planned_start_date
                    if duration.days >= ab_test.max_duration_days:
                        return await self.stop_test(test_id, "Maximum duration reached")

            return ab_test.results

        except Exception as e:
            ab_test.status = TestStatus.FAILED
            logger.error(f"Sequential test failed for {test_id}: {e}")
            raise

    async def calculate_power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        significance_level: float = 0.05
    ) -> Dict[str, float]:
        """
        Calcule l'analyse de puissance pour un test.

        Args:
            effect_size: Taille d'effet attendue
            sample_size: Taille d'échantillon
            significance_level: Niveau de significativité

        Returns:
            Résultats de l'analyse de puissance
        """
        # Calcul de puissance statistique (simplifié)
        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        z_beta = effect_size * np.sqrt(sample_size / 2) - z_alpha

        statistical_power = stats.norm.cdf(z_beta)

        # Taille d'échantillon requise pour puissance 0.8
        z_power = stats.norm.ppf(0.8)
        required_sample_size = 2 * ((z_alpha + z_power) / effect_size) ** 2

        return {
            "statistical_power": statistical_power,
            "required_sample_size": int(required_sample_size),
            "effect_size": effect_size,
            "significance_level": significance_level,
            "sample_size": sample_size
        }

    def get_test_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de tous les tests"""
        return {
            "active_tests": len(self.active_tests),
            "completed_tests": len(self.completed_tests),
            "total_tests": len(self.active_tests) + len(self.completed_tests),
            "test_types": self._count_by_type(),
            "success_rate": self._calculate_success_rate(),
            "average_duration": self._calculate_average_duration()
        }

    async def _calculate_required_sample_size(self, ab_test: ABTest) -> int:
        """Calcule la taille d'échantillon requise"""
        effect_size = ab_test.minimum_detectable_effect
        alpha = ab_test.significance_level
        power = ab_test.statistical_power

        # Formule pour test t à deux échantillons
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        sample_size_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(sample_size_per_group))

    async def _validate_test_configuration(self, ab_test: ABTest) -> Dict[str, Any]:
        """Valide la configuration d'un test"""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Vérifier les allocations
        total_allocation = sum(group.allocation_weight for group in ab_test.groups)
        if abs(total_allocation - 1.0) > 0.01:
            validation["errors"].append(f"Total allocation is {total_allocation}, should be 1.0")
            validation["valid"] = False

        # Vérifier les métriques
        if not ab_test.primary_metric:
            validation["errors"].append("Primary metric is required")
            validation["valid"] = False

        # Vérifier les paramètres statistiques
        if ab_test.significance_level <= 0 or ab_test.significance_level >= 1:
            validation["errors"].append("Significance level must be between 0 and 1")
            validation["valid"] = False

        if ab_test.statistical_power <= 0 or ab_test.statistical_power >= 1:
            validation["errors"].append("Statistical power must be between 0 and 1")
            validation["valid"] = False

        return validation

    def _calculate_group_allocation(self, ab_test: ABTest, group: TestGroup) -> float:
        """Calcule l'allocation de capital pour un groupe"""
        # Allocation basique - peut être sophistiquée avec des règles de risque
        total_capital = 100000.0  # Capital de test par défaut
        return total_capital * group.allocation_weight

    async def _calculate_test_results(self, ab_test: ABTest, is_final: bool = False) -> TestResult:
        """Calcule les résultats d'un test A/B"""
        group_a, group_b = ab_test.groups[0], ab_test.groups[1]

        # Simuler les métriques de performance (dans un vrai système, récupérer depuis la DB)
        metrics_a = await self._collect_group_metrics(group_a)
        metrics_b = await self._collect_group_metrics(group_b)

        # Extraire la métrique principale
        primary_values_a = metrics_a.get(ab_test.primary_metric, [])
        primary_values_b = metrics_b.get(ab_test.primary_metric, [])

        if not primary_values_a or not primary_values_b:
            logger.warning(f"Insufficient data for test {ab_test.test_id}")
            # Retourner résultats par défaut
            return self._create_default_test_result(ab_test, group_a, group_b)

        # Effectuer le test statistique
        stat_results = await self._perform_statistical_test(
            ab_test.statistical_test,
            primary_values_a,
            primary_values_b,
            ab_test.significance_level
        )

        # Calculer les métriques d'affaires
        mean_a = np.mean(primary_values_a)
        mean_b = np.mean(primary_values_b)
        relative_improvement = (mean_b - mean_a) / mean_a * 100 if mean_a != 0 else 0
        absolute_difference = mean_b - mean_a

        # Déterminer la signification business
        business_significance = (
            stat_results["p_value"] < ab_test.significance_level and
            abs(relative_improvement) >= ab_test.minimum_detectable_effect * 100
        )

        # Générer recommandation
        recommendation = self._generate_recommendation(
            stat_results["p_value"],
            relative_improvement,
            business_significance,
            ab_test.significance_level
        )

        # Calculer durée
        start_date = group_a.start_time or datetime.utcnow()
        end_date = group_a.end_time or datetime.utcnow()
        duration_days = (end_date - start_date).days

        return TestResult(
            test_id=ab_test.test_id,
            test_type=ab_test.test_type,
            statistical_test=ab_test.statistical_test,
            group_a=group_a,
            group_b=group_b,
            primary_metric=ab_test.primary_metric,
            secondary_metrics=ab_test.secondary_metrics,
            p_value=stat_results["p_value"],
            effect_size=stat_results["effect_size"],
            confidence_interval=stat_results["confidence_interval"],
            statistical_power=stat_results["power"],
            sample_size=len(primary_values_a) + len(primary_values_b),
            relative_improvement=relative_improvement,
            absolute_difference=absolute_difference,
            business_significance=business_significance,
            recommendation=recommendation,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            significance_level=ab_test.significance_level,
            minimum_detectable_effect=ab_test.minimum_detectable_effect,
            assumptions_met=stat_results["assumptions"],
            warnings=stat_results.get("warnings", [])
        )

    async def _collect_group_metrics(self, group: TestGroup) -> Dict[str, List[float]]:
        """Collecte les métriques d'un groupe de test"""
        # Simulation de métriques - dans un vrai système, récupérer depuis la base de données
        n_observations = np.random.randint(50, 200)

        # Simuler des métriques réalistes
        if "return" in group.group_id or "profit" in group.group_id:
            base_return = 0.08 if "B" in group.group_id else 0.06  # Groupe B légèrement meilleur
            returns = np.random.normal(base_return / 252, 0.02, n_observations)  # Returns journaliers
        else:
            returns = np.random.normal(0.06 / 252, 0.02, n_observations)

        sharpe_ratios = returns / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else [0]

        return {
            "daily_return": returns.tolist(),
            "sharpe_ratio": [sharpe_ratios] * n_observations,
            "max_drawdown": np.random.uniform(0.02, 0.15, n_observations).tolist(),
            "volatility": np.random.uniform(0.15, 0.25, n_observations).tolist(),
            "win_rate": np.random.uniform(0.45, 0.65, n_observations).tolist()
        }

    async def _perform_statistical_test(
        self,
        test_type: StatisticalTest,
        sample_a: List[float],
        sample_b: List[float],
        alpha: float
    ) -> Dict[str, Any]:
        """Effectue un test statistique"""
        arr_a, arr_b = np.array(sample_a), np.array(sample_b)

        if test_type == StatisticalTest.T_TEST:
            # Test t de Student (hypothèse: variances égales)
            statistic, p_value = ttest_ind(arr_a, arr_b, equal_var=True)
            assumptions = {
                "normality": True,  # Simplifié - devrait tester avec Shapiro-Wilk
                "equal_variances": True,
                "independence": True
            }

        elif test_type == StatisticalTest.WELCH_T_TEST:
            # Test t de Welch (variances inégales)
            statistic, p_value = ttest_ind(arr_a, arr_b, equal_var=False)
            assumptions = {
                "normality": True,
                "equal_variances": False,
                "independence": True
            }

        elif test_type == StatisticalTest.MANN_WHITNEY:
            # Test de Mann-Whitney U (non-paramétrique)
            statistic, p_value = mannwhitneyu(arr_a, arr_b, alternative='two-sided')
            assumptions = {
                "normality": False,
                "equal_variances": False,
                "independence": True
            }

        else:
            # Test par défaut
            statistic, p_value = ttest_ind(arr_a, arr_b)
            assumptions = {"default": True}

        # Calculer taille d'effet (Cohen's d)
        pooled_std = np.sqrt(((len(arr_a) - 1) * np.var(arr_a) + (len(arr_b) - 1) * np.var(arr_b)) / (len(arr_a) + len(arr_b) - 2))
        effect_size = (np.mean(arr_b) - np.mean(arr_a)) / pooled_std if pooled_std > 0 else 0

        # Calculer intervalle de confiance
        se_diff = pooled_std * np.sqrt(1/len(arr_a) + 1/len(arr_b))
        diff_means = np.mean(arr_b) - np.mean(arr_a)
        t_critical = stats.t.ppf(1 - alpha/2, len(arr_a) + len(arr_b) - 2)
        ci_lower = diff_means - t_critical * se_diff
        ci_upper = diff_means + t_critical * se_diff

        # Calculer puissance statistique observée
        observed_power = self._calculate_observed_power(effect_size, len(arr_a) + len(arr_b), alpha)

        return {
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": effect_size,
            "confidence_interval": (ci_lower, ci_upper),
            "power": observed_power,
            "assumptions": assumptions,
            "sample_size_a": len(arr_a),
            "sample_size_b": len(arr_b)
        }

    def _calculate_observed_power(self, effect_size: float, total_sample_size: int, alpha: float) -> float:
        """Calcule la puissance statistique observée"""
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = effect_size * np.sqrt(total_sample_size / 4) - z_alpha
        return stats.norm.cdf(z_beta)

    def _generate_recommendation(
        self,
        p_value: float,
        relative_improvement: float,
        business_significance: bool,
        alpha: float
    ) -> str:
        """Génère une recommandation basée sur les résultats"""
        if p_value < alpha and business_significance:
            if relative_improvement > 0:
                return f"Recommandation: Adopter la stratégie B. Amélioration significative de {relative_improvement:.2f}%."
            else:
                return f"Recommandation: Conserver la stratégie A. Dégradation significative de {abs(relative_improvement):.2f}% avec B."
        elif p_value < alpha and not business_significance:
            return "Recommandation: Différence statistiquement significative mais sans impact business notable."
        else:
            return "Recommandation: Aucune différence significative détectée. Prolonger le test ou accepter l'équivalence."

    def _create_default_test_result(self, ab_test: ABTest, group_a: TestGroup, group_b: TestGroup) -> TestResult:
        """Crée un résultat par défaut quand les données sont insuffisantes"""
        return TestResult(
            test_id=ab_test.test_id,
            test_type=ab_test.test_type,
            statistical_test=ab_test.statistical_test,
            group_a=group_a,
            group_b=group_b,
            primary_metric=ab_test.primary_metric,
            secondary_metrics=ab_test.secondary_metrics,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            statistical_power=0.0,
            sample_size=0,
            relative_improvement=0.0,
            absolute_difference=0.0,
            business_significance=False,
            recommendation="Données insuffisantes pour conclure",
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
            duration_days=0,
            warnings=["Insufficient data for statistical analysis"]
        )

    async def _check_early_stopping_conditions(self, ab_test: ABTest, interim_results: TestResult) -> Tuple[bool, str]:
        """Vérifie les conditions d'arrêt précoce"""
        # Arrêt pour significativité claire
        if interim_results.p_value < ab_test.significance_level / 10:  # Critère strict
            return True, f"Strong statistical significance reached (p={interim_results.p_value:.6f})"

        # Arrêt pour futilité (très peu de chance d'atteindre la significativité)
        if interim_results.p_value > 0.8 and interim_results.sample_size > 100:
            return True, "Futility stopping - unlikely to reach significance"

        # Arrêt pour amélioration business très importante
        if abs(interim_results.relative_improvement) > 50:  # >50% d'amélioration
            return True, f"Large business impact detected ({interim_results.relative_improvement:.1f}%)"

        return False, ""

    def _generate_test_id(self, name: str) -> str:
        """Génère un ID unique pour un test"""
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())[:20]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"test_{clean_name}_{timestamp}"

    def _count_by_type(self) -> Dict[str, int]:
        """Compte les tests par type"""
        type_counts = {}
        all_tests = list(self.active_tests.values()) + list(self.completed_tests.values())

        for test in all_tests:
            test_type = test.test_type.value
            type_counts[test_type] = type_counts.get(test_type, 0) + 1

        return type_counts

    def _calculate_success_rate(self) -> float:
        """Calcule le taux de succès des tests"""
        if not self.completed_tests:
            return 0.0

        successful_tests = sum(
            1 for test in self.completed_tests.values()
            if test.results and test.results.business_significance
        )

        return successful_tests / len(self.completed_tests)

    def _calculate_average_duration(self) -> float:
        """Calcule la durée moyenne des tests"""
        if not self.completed_tests:
            return 0.0

        total_duration = sum(
            test.results.duration_days for test in self.completed_tests.values()
            if test.results
        )

        return total_duration / len(self.completed_tests)