"""
QFrame Research Reports Module
==============================

Professional scientific reporting system for quantitative research results.
"""

from .scientific_report_generator import ScientificReportGenerator
from .performance_analyzer import PerformanceAnalyzer
from .risk_analyzer import RiskAnalyzer
from .statistical_validator import StatisticalValidator

__all__ = [
    'ScientificReportGenerator',
    'PerformanceAnalyzer',
    'RiskAnalyzer',
    'StatisticalValidator'
]