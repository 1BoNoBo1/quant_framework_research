"""
Tests pour Scientific Report Generator
=====================================

Tests pour le générateur de rapports scientifiques QFrame.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from qframe.research.reports.scientific_report_generator import ScientificReportGenerator


class TestScientificReportGenerator:
    """Tests pour ScientificReportGenerator."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.generator = ScientificReportGenerator()

        # Données de test
        self.sample_backtest_results = {
            'total_return': 0.565,
            'sharpe_ratio': 2.254,
            'max_drawdown': 0.0497,
            'win_rate': 0.60,
            'total_trades': 544
        }

        # Données de marché simulées
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        self.sample_market_data = pd.DataFrame({
            'timestamp': dates,
            'close': 50000 + np.cumsum(np.random.normal(0, 100, 100)),
            'volume': np.random.lognormal(10, 0.5, 100)
        })

    def test_generator_initialization(self):
        """Test initialisation du générateur."""
        assert self.generator is not None
        assert hasattr(self.generator, 'generate_strategy_performance_report')

    def test_generate_basic_report(self):
        """Test génération d'un rapport basique."""
        report = self.generator.generate_strategy_performance_report(
            strategy_name="TestStrategy",
            backtest_results=self.sample_backtest_results,
            market_data=self.sample_market_data
        )

        assert report is not None
        assert hasattr(report, 'sections')
        assert len(report.sections) >= 5  # Au moins 5 sections

    def test_report_sections_content(self):
        """Test contenu des sections du rapport."""
        report = self.generator.generate_strategy_performance_report(
            strategy_name="TestStrategy",
            backtest_results=self.sample_backtest_results,
            market_data=self.sample_market_data
        )

        section_titles = [section.title for section in report.sections]

        # Vérifier que les sections essentielles sont présentes
        expected_sections = [
            "Executive Summary",
            "Performance Analysis",
            "Risk Analysis"
        ]

        for expected in expected_sections:
            assert any(expected in title for title in section_titles)

    def test_export_to_markdown(self):
        """Test export vers Markdown."""
        report = self.generator.generate_strategy_performance_report(
            strategy_name="TestStrategy",
            backtest_results=self.sample_backtest_results,
            market_data=self.sample_market_data
        )

        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp:
            self.generator.export_to_markdown(report, tmp.name)

            # Vérifier que le fichier existe et n'est pas vide
            assert Path(tmp.name).exists()
            assert Path(tmp.name).stat().st_size > 0

            # Nettoyer
            Path(tmp.name).unlink()

    def test_export_to_html(self):
        """Test export vers HTML."""
        report = self.generator.generate_strategy_performance_report(
            strategy_name="TestStrategy",
            backtest_results=self.sample_backtest_results,
            market_data=self.sample_market_data
        )

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            self.generator.export_to_html(report, tmp.name)

            # Vérifier que le fichier existe et n'est pas vide
            assert Path(tmp.name).exists()
            assert Path(tmp.name).stat().st_size > 0

            # Nettoyer
            Path(tmp.name).unlink()

    def test_performance_metrics_validation(self):
        """Test validation des métriques de performance."""
        # Test avec métriques valides
        valid_results = {
            'total_return': 0.15,
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.05,
            'win_rate': 0.65
        }

        report = self.generator.generate_strategy_performance_report(
            strategy_name="ValidStrategy",
            backtest_results=valid_results,
            market_data=self.sample_market_data
        )

        assert report is not None

    def test_market_data_validation(self):
        """Test validation des données de marché."""
        # Test avec données minimales
        minimal_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'close': [50000] * 10
        })

        report = self.generator.generate_strategy_performance_report(
            strategy_name="MinimalDataStrategy",
            backtest_results=self.sample_backtest_results,
            market_data=minimal_data
        )

        assert report is not None

    def test_error_handling_empty_data(self):
        """Test gestion d'erreurs avec données vides."""
        empty_data = pd.DataFrame()

        with pytest.raises(Exception):
            self.generator.generate_strategy_performance_report(
                strategy_name="EmptyDataStrategy",
                backtest_results=self.sample_backtest_results,
                market_data=empty_data
            )

    def test_error_handling_missing_metrics(self):
        """Test gestion d'erreurs avec métriques manquantes."""
        incomplete_results = {
            'total_return': 0.15
            # Métriques manquantes
        }

        # Devrait fonctionner avec des valeurs par défaut
        report = self.generator.generate_strategy_performance_report(
            strategy_name="IncompleteStrategy",
            backtest_results=incomplete_results,
            market_data=self.sample_market_data
        )

        assert report is not None

    def test_feature_analysis_integration(self):
        """Test intégration avec analyse des features."""
        feature_analysis = {
            'features_generated': 18,
            'feature_quality': 0.156,
            'alpha_signals': 245
        }

        report = self.generator.generate_strategy_performance_report(
            strategy_name="FeatureStrategy",
            backtest_results=self.sample_backtest_results,
            market_data=self.sample_market_data,
            feature_analysis=feature_analysis
        )

        assert report is not None
        # Vérifier qu'une section feature analysis est présente
        section_titles = [section.title for section in report.sections]
        assert any("Feature" in title for title in section_titles)

    def test_validation_results_integration(self):
        """Test intégration avec résultats de validation."""
        validation_results = {
            'overall_validation': 87.3,
            'data_quality_score': 100.0,
            'overfitting_checks': 87.5
        }

        report = self.generator.generate_strategy_performance_report(
            strategy_name="ValidatedStrategy",
            backtest_results=self.sample_backtest_results,
            market_data=self.sample_market_data,
            validation_results=validation_results
        )

        assert report is not None
        # Vérifier qu'une section validation est présente
        section_titles = [section.title for section in report.sections]
        assert any("Validation" in title for title in section_titles)