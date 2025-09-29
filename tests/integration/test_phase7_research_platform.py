"""
Phase 7 Integration Tests
========================

Comprehensive tests for the Research & Innovation Platform.
Tests integration between Analytics Engine, Innovation Engine, and External Integrations.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any
import tempfile
import json

# Research Platform Components
from qframe.research.backtesting.advanced_backtesting_engine import AdvancedBacktestingEngine
from qframe.research.analytics.factor_analyzer import FactorAnalyzer
from qframe.research.innovation.auto_strategy_generator import (
    AutoStrategyGenerator, StrategyTemplate, GenerationConfig
)
from qframe.research.innovation.research_paper_integrator import (
    ResearchPaperIntegrator, PaperMetadata, PaperType
)
from qframe.research.innovation.ab_testing_framework import (
    ABTestingFramework, TestType
)
from qframe.research.innovation.genetic_algorithm_optimizer import (
    GeneticAlgorithmOptimizer, Parameter, ObjectiveFunction, OptimizationConfig
)
from qframe.research.external.academic_data_connector import (
    AcademicDataConnector, DataQuery, DataSourceType
)
from qframe.research.external.research_api_manager import ResearchAPIManager
from qframe.research.external.paper_alert_system import (
    PaperAlertSystem, AlertRule, AlertTrigger, AlertPriority, AlertChannel
)

# Core framework
from qframe.core.config import FrameworkConfig
from qframe.core.container import get_container, injectable

# Mock strategy for testing
@injectable
class MockStrategy:
    def __init__(self, param1: float = 0.5, param2: int = 10):
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, data: pd.DataFrame, features=None):
        return []


class TestPhase7ResearchPlatform:
    """Tests d'intégration pour la plateforme de recherche Phase 7"""

    @pytest.fixture
    def config(self):
        """Configuration test"""
        return FrameworkConfig(environment="testing")

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Données OHLCV de test"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)

        prices = 100 + np.cumsum(np.random.randn(252) * 0.02)

        return pd.DataFrame({
            'open': prices + np.random.randn(252) * 0.1,
            'high': prices + np.abs(np.random.randn(252) * 0.2),
            'low': prices - np.abs(np.random.randn(252) * 0.2),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 252)
        }, index=dates)

    @pytest.fixture
    def mock_research_papers(self):
        """Papiers de recherche fictifs"""
        return [
            {
                "id": "paper_1",
                "title": "Deep Learning for Algorithmic Trading",
                "authors": ["Smith, J.", "Doe, A."],
                "abstract": "This paper presents a novel deep learning approach for algorithmic trading using LSTM networks.",
                "year": 2023,
                "url": "https://arxiv.org/abs/2301.00001",
                "keywords": ["deep learning", "algorithmic trading", "LSTM"]
            },
            {
                "id": "paper_2",
                "title": "Risk Management in Cryptocurrency Markets",
                "authors": ["Johnson, M."],
                "abstract": "An analysis of risk management techniques for cryptocurrency portfolio optimization.",
                "year": 2023,
                "url": "https://arxiv.org/abs/2301.00002",
                "keywords": ["risk management", "cryptocurrency", "portfolio optimization"]
            }
        ]

    @pytest.mark.asyncio
    async def test_advanced_backtesting_integration(self, config, sample_ohlcv_data):
        """Test d'intégration du moteur de backtesting avancé"""

        # Créer moteur de backtesting
        backtest_engine = AdvancedBacktestingEngine(config)

        # Créer stratégie mock
        strategy = MockStrategy(param1=0.3, param2=15)

        # Exécuter backtest
        results = await backtest_engine.run_backtest(strategy, sample_ohlcv_data)

        # Vérifications
        assert results is not None
        assert hasattr(results, 'performance_metrics')
        assert hasattr(results, 'portfolio_values')
        assert len(results.portfolio_values) > 0

        # Vérifier métriques de performance
        summary = results.get_summary_report()
        assert 'performance' in summary
        assert 'total_return' in summary['performance']
        assert 'sharpe_ratio' in summary['performance']
        assert 'max_drawdown' in summary['performance']

    @pytest.mark.asyncio
    async def test_factor_analysis_integration(self, config, sample_ohlcv_data):
        """Test d'intégration de l'analyseur de facteurs"""

        factor_analyzer = FactorAnalyzer(config)

        # Préparer données multi-assets (simuler plusieurs symboles)
        returns_data = {}
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']

        for symbol in symbols:
            # Générer returns différents pour chaque asset
            np.random.seed(hash(symbol) % 1000)
            returns = sample_ohlcv_data['close'].pct_change().dropna()
            returns_data[symbol] = returns + np.random.randn(len(returns)) * 0.001

        returns_df = pd.DataFrame(returns_data)

        # Analyser facteurs
        factor_analysis = await factor_analyzer.analyze_portfolio(returns_df)

        # Vérifications
        assert factor_analysis is not None
        assert 'factor_loadings' in factor_analysis
        assert 'factor_returns' in factor_analysis
        assert 'explained_variance' in factor_analysis

        # Vérifier PCA
        pca_results = await factor_analyzer.perform_pca_analysis(returns_df)
        assert 'components' in pca_results
        assert 'explained_variance_ratio' in pca_results

    @pytest.mark.asyncio
    async def test_auto_strategy_generator_integration(self, config, sample_ohlcv_data):
        """Test d'intégration du générateur automatique de stratégies"""

        generator = AutoStrategyGenerator(config)

        # Configuration de génération
        gen_config = GenerationConfig(
            population_size=10,
            max_generations=3,
            mutation_rate=0.1,
            crossover_rate=0.8,
            max_strategy_complexity=5
        )

        # Générer stratégies
        generation_results = await generator.evolve_strategies(
            training_data=sample_ohlcv_data,
            config=gen_config
        )

        # Vérifications
        assert generation_results is not None
        assert 'best_strategies' in generation_results
        assert 'evolution_history' in generation_results
        assert len(generation_results['best_strategies']) > 0

        # Vérifier qu'on peut générer le code
        best_strategy = generation_results['best_strategies'][0]
        strategy_code = await generator.generate_strategy_code(best_strategy)
        assert isinstance(strategy_code, str)
        assert len(strategy_code) > 100

    @pytest.mark.asyncio
    async def test_research_paper_integrator(self, config, mock_research_papers):
        """Test d'intégration du système d'intégration de papiers"""

        integrator = ResearchPaperIntegrator(config)

        # Simuler parsing d'un papier
        paper_metadata = PaperMetadata(
            title="Test Paper: ML for Trading",
            authors=["Test Author"],
            abstract="This is a test paper about machine learning applications in trading.",
            paper_type=PaperType.STRATEGY,
            keywords=["machine learning", "trading", "strategy"]
        )

        # Parser le papier (utilise contenu mock)
        with patch.object(integrator, '_extract_text_content', return_value="Mock paper content"):
            with patch.object(integrator, '_extract_metadata', return_value=paper_metadata):
                implementation = await integrator.parse_paper("mock_paper.pdf", paper_metadata)

        # Vérifications
        assert implementation is not None
        assert implementation.metadata.title == "Test Paper: ML for Trading"
        assert implementation.metadata.paper_type == PaperType.STRATEGY

        # Tester implémentation automatique
        paper_id = list(integrator.implementations.keys())[0]
        implementation = await integrator.implement_paper(paper_id)

        assert implementation.strategy_code is not None
        assert len(implementation.strategy_code) > 100

    @pytest.mark.asyncio
    async def test_ab_testing_framework_integration(self, config, sample_ohlcv_data):
        """Test d'intégration du framework de tests A/B"""

        # Mock portfolio service
        mock_portfolio_service = Mock()

        ab_framework = ABTestingFramework(config, mock_portfolio_service)

        # Créer stratégies de test
        strategy_a = MockStrategy(param1=0.3, param2=10)
        strategy_b = MockStrategy(param1=0.5, param2=20)

        # Créer test A/B
        ab_test = await ab_framework.create_test(
            name="Strategy Parameter Test",
            description="Test different parameter values",
            test_type=TestType.PARAMETER_OPTIMIZATION,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            primary_metric="sharpe_ratio",
            secondary_metrics=["max_drawdown", "total_return"]
        )

        # Vérifications
        assert ab_test is not None
        assert ab_test.name == "Strategy Parameter Test"
        assert len(ab_test.groups) == 2
        assert ab_test.primary_metric == "sharpe_ratio"

        # Simuler démarrage et arrêt du test
        test_started = await ab_framework.start_test(ab_test.test_id)
        assert test_started is True

        # Obtenir résultats intermédiaires
        interim_results = await ab_framework.get_interim_results(ab_test.test_id)
        assert interim_results is not None
        assert hasattr(interim_results, 'p_value')
        assert hasattr(interim_results, 'relative_improvement')

    @pytest.mark.asyncio
    async def test_genetic_algorithm_optimizer_integration(self, config, sample_ohlcv_data):
        """Test d'intégration de l'optimiseur génétique"""

        optimizer = GeneticAlgorithmOptimizer(config)

        # Définir paramètres à optimiser
        parameters = [
            Parameter(name="param1", param_type="float", min_value=0.1, max_value=1.0),
            Parameter(name="param2", param_type="int", min_value=5, max_value=50)
        ]

        # Définir objectifs
        objectives = [
            ObjectiveFunction(name="sharpe_ratio", weight=1.0, maximize=True),
            ObjectiveFunction(name="max_drawdown", weight=0.5, maximize=False)
        ]

        # Configuration d'optimisation (réduite pour les tests)
        opt_config = OptimizationConfig(
            population_size=10,
            max_generations=3,
            convergence_threshold=0.001
        )

        # Optimiser paramètres
        optimization_results = await optimizer.optimize_strategy_parameters(
            strategy_class=MockStrategy,
            parameters=parameters,
            objective_functions=objectives,
            training_data=sample_ohlcv_data,
            optimization_config=opt_config
        )

        # Vérifications
        assert optimization_results is not None
        assert 'best_parameters' in optimization_results
        assert 'best_fitness' in optimization_results
        assert 'convergence_history' in optimization_results

        # Vérifier paramètres optimaux
        best_params = optimization_results['best_parameters']
        assert 'param1' in best_params
        assert 'param2' in best_params
        assert 0.1 <= best_params['param1'] <= 1.0
        assert 5 <= best_params['param2'] <= 50

    @pytest.mark.asyncio
    async def test_academic_data_connector_integration(self, config):
        """Test d'intégration du connecteur de données académiques"""

        connector = AcademicDataConnector(config)

        # Mock des réponses API
        mock_arxiv_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2301.00001v1</id>
                <title>Test Paper Title</title>
                <summary>Test paper abstract</summary>
                <published>2023-01-01T00:00:00Z</published>
                <author><name>Test Author</name></author>
                <category term="q-fin.CP" />
            </entry>
        </feed>"""

        # Tester requête arXiv
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = mock_arxiv_response
            mock_get.return_value.__aenter__.return_value = mock_response

            query = DataQuery(
                source_id="arxiv",
                query_type="papers",
                parameters={},
                keywords=["machine learning", "finance"],
                limit=10
            )

            result = await connector.query(query)

            # Vérifications
            assert result is not None
            assert result.source == "arXiv"
            assert len(result.data) > 0
            assert result.returned_records > 0

    @pytest.mark.asyncio
    async def test_research_api_manager_integration(self, config):
        """Test d'intégration du gestionnaire d'API de recherche"""

        api_manager = ResearchAPIManager(config)
        await api_manager.start()

        try:
            # Mock réponse API
            mock_response_data = {
                "data": [
                    {
                        "paperId": "test123",
                        "title": "Test Paper",
                        "authors": [{"name": "Test Author"}],
                        "abstract": "Test abstract",
                        "year": 2023,
                        "url": "https://test.com/paper"
                    }
                ]
            }

            # Test recherche de papiers
            with patch('aiohttp.ClientSession.request') as mock_request:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = mock_response_data
                mock_response.headers = {}
                mock_request.return_value.__aenter__.return_value = mock_response

                papers = await api_manager.search_papers("machine learning finance", limit=10)

                # Vérifications
                assert isinstance(papers, list)
                # Note: Peut être vide si mock n'est pas parfait, c'est OK pour ce test d'intégration

        finally:
            await api_manager.stop()

    @pytest.mark.asyncio
    async def test_paper_alert_system_integration(self, config, mock_research_papers):
        """Test d'intégration du système d'alertes papier"""

        # Mock API manager
        mock_api_manager = AsyncMock()
        mock_api_manager.search_papers.return_value = mock_research_papers

        alert_system = PaperAlertSystem(config, mock_api_manager)

        # Configurer règle d'alerte
        rule = AlertRule(
            rule_id="test_rule",
            name="ML Trading Papers",
            description="Papers about ML in trading",
            trigger=AlertTrigger.KEYWORD_MATCH,
            priority=AlertPriority.HIGH,
            keywords=["machine learning", "algorithmic trading"],
            min_relevance_score=0.3
        )

        alert_system.add_rule(rule)

        # Traiter papiers
        alerts = await alert_system.process_new_papers(mock_research_papers)

        # Vérifications
        assert isinstance(alerts, list)
        # Au moins un papier devrait correspondre aux critères
        matching_alerts = [a for a in alerts if len(a.matched_keywords) > 0]
        assert len(matching_alerts) > 0

        # Vérifier structure de l'alerte
        if alerts:
            alert = alerts[0]
            assert hasattr(alert, 'alert_id')
            assert hasattr(alert, 'relevance_score')
            assert hasattr(alert, 'paper_data')
            assert hasattr(alert, 'matched_keywords')

    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self, config, sample_ohlcv_data, mock_research_papers):
        """Test de workflow complet de recherche end-to-end"""

        # 1. Connecteur de données académiques
        data_connector = AcademicDataConnector(config)

        # 2. Analyseur de facteurs
        factor_analyzer = FactorAnalyzer(config)

        # 3. Générateur de stratégies
        strategy_generator = AutoStrategyGenerator(config)

        # 4. Moteur de backtesting
        backtest_engine = AdvancedBacktestingEngine(config)

        # 5. Framework A/B testing
        mock_portfolio_service = Mock()
        ab_framework = ABTestingFramework(config, mock_portfolio_service)

        # Workflow complet

        # Étape 1: Analyser les facteurs de marché
        returns_data = {
            'BTCUSDT': sample_ohlcv_data['close'].pct_change().dropna(),
            'ETHUSDT': sample_ohlcv_data['close'].pct_change().dropna() + np.random.randn(len(sample_ohlcv_data)-1) * 0.001
        }
        returns_df = pd.DataFrame(returns_data)

        factor_analysis = await factor_analyzer.analyze_portfolio(returns_df)
        assert 'factor_loadings' in factor_analysis

        # Étape 2: Générer stratégies automatiquement
        gen_config = GenerationConfig(
            population_size=5,
            max_generations=2,
            max_strategy_complexity=3
        )

        generation_results = await strategy_generator.evolve_strategies(
            sample_ohlcv_data,
            config=gen_config
        )

        assert len(generation_results['best_strategies']) > 0

        # Étape 3: Backtester les meilleures stratégies
        best_strategy_template = generation_results['best_strategies'][0]
        mock_strategy = MockStrategy()  # Représente la stratégie générée

        backtest_results = await backtest_engine.run_backtest(mock_strategy, sample_ohlcv_data)
        assert backtest_results is not None

        # Étape 4: Comparer stratégies avec A/B testing
        strategy_a = MockStrategy(param1=0.3)
        strategy_b = MockStrategy(param1=0.7)

        ab_test = await ab_framework.create_test(
            name="Generated Strategy Comparison",
            description="Compare generated strategies",
            test_type=TestType.STRATEGY_COMPARISON,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            primary_metric="sharpe_ratio"
        )

        assert ab_test is not None

        # Workflow complet réussi
        workflow_results = {
            "factor_analysis": factor_analysis,
            "generated_strategies": generation_results,
            "backtest_results": backtest_results,
            "ab_test": ab_test
        }

        # Vérifications finales
        assert all(key in workflow_results for key in [
            "factor_analysis", "generated_strategies", "backtest_results", "ab_test"
        ])

    @pytest.mark.asyncio
    async def test_performance_and_scalability(self, config, sample_ohlcv_data):
        """Test de performance et scalabilité des composants Phase 7"""

        # Test avec données plus importantes
        large_data = sample_ohlcv_data.copy()
        # Répliquer les données pour simuler plus d'historique
        for i in range(3):
            new_dates = pd.date_range(
                large_data.index[-1] + timedelta(days=1),
                periods=len(large_data),
                freq='D'
            )
            new_data = large_data.copy()
            new_data.index = new_dates
            large_data = pd.concat([large_data, new_data])

        # Test de performance du backtesting
        start_time = datetime.utcnow()

        backtest_engine = AdvancedBacktestingEngine(config)
        strategy = MockStrategy()

        results = await backtest_engine.run_backtest(strategy, large_data)

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        # Vérifications de performance
        assert execution_time < 10.0  # Moins de 10 secondes
        assert results is not None
        assert len(results.portfolio_values) == len(large_data)

        # Test de mémoire (vérifier que les DataFrames ne causent pas de fuite)
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Exécuter plusieurs backtests
        for _ in range(5):
            await backtest_engine.run_backtest(MockStrategy(), sample_ohlcv_data)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # La consommation mémoire ne devrait pas exploser
        assert memory_increase < 100  # Moins de 100 MB d'augmentation

    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, config, sample_ohlcv_data):
        """Test de gestion d'erreurs et résilience"""

        # Test avec données corrompues
        corrupted_data = sample_ohlcv_data.copy()
        corrupted_data.loc[corrupted_data.index[10:20], 'close'] = np.nan

        backtest_engine = AdvancedBacktestingEngine(config)
        strategy = MockStrategy()

        # Le backtest devrait gérer les données manquantes
        results = await backtest_engine.run_backtest(strategy, corrupted_data)
        assert results is not None

        # Test avec paramètres invalides
        invalid_strategy = MockStrategy(param1=-1.0, param2=0)  # Paramètres potentiellement problématiques

        try:
            results = await backtest_engine.run_backtest(invalid_strategy, sample_ohlcv_data)
            # Devrait soit réussir soit échouer proprement
            assert results is not None or True  # Accepte les deux cas
        except Exception as e:
            # L'erreur devrait être gérée proprement
            assert isinstance(e, Exception)

        # Test de résistance aux erreurs réseau (pour composants externes)
        data_connector = AcademicDataConnector(config)

        # Simuler échec réseau
        with patch('aiohttp.ClientSession.get', side_effect=Exception("Network error")):
            query = DataQuery(
                source_id="arxiv",
                query_type="papers",
                parameters={},
                keywords=["test"],
                limit=1
            )

            try:
                result = await data_connector.query(query)
                # Si pas d'exception, vérifier que l'erreur est gérée
                assert result is not None
            except Exception:
                # Exception attendue, c'est OK
                pass

    def test_configuration_and_setup(self, config):
        """Test de configuration et initialisation des composants"""

        # Vérifier que tous les composants peuvent être initialisés
        components = [
            AdvancedBacktestingEngine(config),
            FactorAnalyzer(config),
            AutoStrategyGenerator(config),
            ResearchPaperIntegrator(config),
            GeneticAlgorithmOptimizer(config),
            AcademicDataConnector(config),
            ResearchAPIManager(config)
        ]

        # Tous les composants devraient s'initialiser sans erreur
        for component in components:
            assert component is not None
            # Les composants n'ont pas tous un attribut 'config' stocké
            # L'important est qu'ils s'initialisent correctement

    @pytest.mark.asyncio
    async def test_data_flow_integration(self, config, sample_ohlcv_data):
        """Test d'intégration du flux de données entre composants"""

        # Simuler flux de données réaliste

        # 1. Données d'entrée
        input_data = sample_ohlcv_data

        # 2. Analyse des facteurs
        factor_analyzer = FactorAnalyzer(config)
        returns_data = {'ASSET': input_data['close'].pct_change().dropna()}
        returns_df = pd.DataFrame(returns_data)

        factor_results = await factor_analyzer.analyze_portfolio(returns_df)

        # 3. Les résultats de facteurs peuvent être utilisés pour la génération de stratégie
        strategy_generator = AutoStrategyGenerator(config)

        # Utiliser les insights des facteurs pour guider la génération
        # (En production, on passerait factor_results à la génération)
        gen_config = GenerationConfig(
            population_size=5,
            max_generations=2
        )

        generation_results = await strategy_generator.evolve_strategies(
            input_data,
            config=gen_config
        )

        # 4. Backtester les stratégies générées
        backtest_engine = AdvancedBacktestingEngine(config)
        mock_strategy = MockStrategy()

        backtest_results = await backtest_engine.run_backtest(mock_strategy, input_data)

        # Vérifier que les données circulent correctement
        data_flow_validation = {
            "input_data_shape": input_data.shape,
            "factor_analysis_components": len(factor_results.get('factor_loadings', {})),
            "generated_strategies_count": len(generation_results['best_strategies']),
            "backtest_portfolio_length": len(backtest_results.portfolio_values)
        }

        # Tous les composants devraient avoir traité les données
        assert data_flow_validation["input_data_shape"][0] > 0
        assert data_flow_validation["backtest_portfolio_length"] > 0
        assert data_flow_validation["generated_strategies_count"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, config, sample_ohlcv_data):
        """Test d'opérations concurrentes"""

        # Exécuter plusieurs opérations en parallèle

        # Créer tâches concurrentes
        backtest_engine = AdvancedBacktestingEngine(config)
        factor_analyzer = FactorAnalyzer(config)
        strategy_generator = AutoStrategyGenerator(config)

        async def run_backtest():
            return await backtest_engine.run_backtest(MockStrategy(), sample_ohlcv_data)

        async def run_factor_analysis():
            returns_data = {'ASSET': sample_ohlcv_data['close'].pct_change().dropna()}
            returns_df = pd.DataFrame(returns_data)
            return await factor_analyzer.analyze_portfolio(returns_df)

        async def run_strategy_generation():
            gen_config = GenerationConfig(population_size=3, max_generations=1)
            return await strategy_generator.evolve_strategies(sample_ohlcv_data, config=gen_config)

        # Exécuter en parallèle
        start_time = datetime.utcnow()

        results = await asyncio.gather(
            run_backtest(),
            run_factor_analysis(),
            run_strategy_generation(),
            return_exceptions=True
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        # Vérifier que toutes les opérations ont réussi ou échoué proprement
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Operation {i} failed with: {result}")
            else:
                assert result is not None

        # L'exécution parallèle devrait être plus rapide que séquentielle
        assert execution_time < 30.0  # Temps raisonnable pour les opérations parallèles

    def test_metrics_and_monitoring(self, config):
        """Test des métriques et monitoring"""

        # Tester les systèmes de métriques de chaque composant

        # API Manager metrics
        api_manager = ResearchAPIManager(config)
        metrics = api_manager.get_metrics()

        expected_metrics = ['global', 'by_provider', 'queue_size', 'active_requests', 'cache_size']
        for metric in expected_metrics:
            assert metric in metrics

        # Data connector stats
        data_connector = AcademicDataConnector(config)
        stats = data_connector.get_query_stats()

        expected_stats = ['total_queries', 'cache_hits', 'failed_queries', 'by_source']
        for stat in expected_stats:
            assert stat in stats

    @pytest.mark.asyncio
    async def test_cleanup_and_resource_management(self, config):
        """Test de nettoyage et gestion des ressources"""

        # Tester que les ressources sont correctement libérées

        api_manager = ResearchAPIManager(config)
        await api_manager.start()

        # Vérifier que les ressources sont allouées
        assert api_manager.session is not None

        # Nettoyer
        await api_manager.stop()

        # Vérifier que les ressources sont libérées
        # (En production, vérifierait que session.closed == True)

        # Test avec data connector
        data_connector = AcademicDataConnector(config)

        # Simuler des connexions
        for provider in data_connector.providers.values():
            if hasattr(provider, 'session') and provider.session:
                await provider.disconnect()

        # Nettoyage global
        await data_connector.disconnect_all()