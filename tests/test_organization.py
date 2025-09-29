#!/usr/bin/env python
"""
📋 Test Organization and Management

Provides utilities for organizing, categorizing and managing
the QFrame test suite for better maintainability and CI/CD integration.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TestCategoryInfo:
    """Information about a test category"""
    name: str
    description: str
    marker: str
    estimated_duration: str
    criticality: str


class TestOrganizer:
    """Organizes and categorizes QFrame tests"""

    # Test categories with their characteristics
    CATEGORIES = {
        'unit': TestCategoryInfo(
            name='Unit Tests',
            description='Fast, isolated tests for individual components',
            marker='unit',
            estimated_duration='< 1s per test',
            criticality='critical'
        ),
        'integration': TestCategoryInfo(
            name='Integration Tests',
            description='Tests for component interactions and workflows',
            marker='integration',
            estimated_duration='1-10s per test',
            criticality='critical'
        ),
        'ui': TestCategoryInfo(
            name='UI Tests',
            description='Tests for web interface components and interactions',
            marker='ui',
            estimated_duration='2-15s per test',
            criticality='important'
        ),
        'strategies': TestCategoryInfo(
            name='Strategy Tests',
            description='Tests for trading strategies and algorithms',
            marker='strategies',
            estimated_duration='5-30s per test',
            criticality='critical'
        ),
        'backtesting': TestCategoryInfo(
            name='Backtesting Tests',
            description='Tests for historical backtesting engine',
            marker='backtesting',
            estimated_duration='10-60s per test',
            criticality='important'
        ),
        'data': TestCategoryInfo(
            name='Data Provider Tests',
            description='Tests for external data sources and providers',
            marker='data',
            estimated_duration='5-20s per test',
            criticality='important'
        ),
        'risk': TestCategoryInfo(
            name='Risk Management Tests',
            description='Tests for risk calculation and management',
            marker='risk',
            estimated_duration='2-10s per test',
            criticality='critical'
        ),
        'performance': TestCategoryInfo(
            name='Performance Tests',
            description='Tests for system performance and benchmarks',
            marker='performance',
            estimated_duration='30-300s per test',
            criticality='optional'
        ),
        'slow': TestCategoryInfo(
            name='Slow Tests',
            description='Long-running tests requiring patience',
            marker='slow',
            estimated_duration='> 30s per test',
            criticality='optional'
        )
    }

    # Test execution plans for different scenarios
    EXECUTION_PLANS = {
        'quick': {
            'description': 'Fast feedback for development',
            'includes': ['unit'],
            'excludes': ['slow', 'performance'],
            'parallel': True,
            'timeout': 120
        },
        'ci': {
            'description': 'Complete CI/CD pipeline tests',
            'includes': ['unit', 'integration', 'strategies', 'risk'],
            'excludes': ['slow', 'performance'],
            'parallel': True,
            'timeout': 600
        },
        'full': {
            'description': 'Complete test suite including slow tests',
            'includes': ['unit', 'integration', 'ui', 'strategies', 'backtesting', 'data', 'risk'],
            'excludes': [],
            'parallel': True,
            'timeout': 1800
        },
        'critical': {
            'description': 'Only critical tests for production validation',
            'includes': ['unit', 'integration', 'strategies', 'risk'],
            'excludes': ['slow', 'performance'],
            'filters': ['critical'],
            'parallel': True,
            'timeout': 300
        },
        'performance': {
            'description': 'Performance and benchmark tests only',
            'includes': ['performance'],
            'excludes': [],
            'parallel': False,
            'timeout': 3600
        }
    }

    @classmethod
    def get_pytest_command(cls, plan_name: str = 'quick') -> List[str]:
        """Generate pytest command for a specific execution plan"""
        plan = cls.EXECUTION_PLANS.get(plan_name)
        if not plan:
            raise ValueError(f"Unknown test plan: {plan_name}")

        cmd = ['pytest', 'tests/']

        # Add markers for inclusion
        if plan['includes']:
            markers = ' or '.join(plan['includes'])
            cmd.extend(['-m', f'({markers})'])

        # Add markers for exclusion
        if plan['excludes']:
            excludes = ' or '.join(plan['excludes'])
            if '-m' in cmd:
                # Combine with existing marker expression
                idx = cmd.index('-m') + 1
                cmd[idx] = f"{cmd[idx]} and not ({excludes})"
            else:
                cmd.extend(['-m', f'not ({excludes})'])

        # Add criticality filters
        if plan.get('filters'):
            filters = ' or '.join(plan['filters'])
            if '-m' in cmd:
                idx = cmd.index('-m') + 1
                cmd[idx] = f"{cmd[idx]} and ({filters})"
            else:
                cmd.extend(['-m', filters])

        # Add parallel execution
        if plan['parallel']:
            cmd.extend(['-n', 'auto'])

        # Add timeout
        if plan['timeout']:
            cmd.extend(['--timeout', str(plan['timeout'])])

        # Add common options
        cmd.extend([
            '--tb=short',
            '--durations=10',
            '-v'
        ])

        return cmd

    @classmethod
    def print_test_organization(cls):
        """Print test organization information"""
        print("📋 QFrame Test Organization")
        print("=" * 60)

        print("\n🏷️ Test Categories:")
        for category, info in cls.CATEGORIES.items():
            print(f"  {info.marker:<12} - {info.name}")
            print(f"  {'':>12}   {info.description}")
            print(f"  {'':>12}   Duration: {info.estimated_duration}")
            print(f"  {'':>12}   Criticality: {info.criticality}")
            print()

        print("\n⚡ Execution Plans:")
        for plan_name, plan in cls.EXECUTION_PLANS.items():
            print(f"  {plan_name:<12} - {plan['description']}")
            print(f"  {'':>12}   Includes: {', '.join(plan['includes'])}")
            if plan['excludes']:
                print(f"  {'':>12}   Excludes: {', '.join(plan['excludes'])}")
            print(f"  {'':>12}   Timeout: {plan['timeout']}s")
            print()

        print("\n🚀 Usage Examples:")
        print("  make test-quick      # Quick development tests")
        print("  make test-ci         # CI/CD pipeline tests")
        print("  make test-all        # Complete test suite")
        print("  make test-critical   # Production validation")
        print("  make test-performance # Performance benchmarks")

    @classmethod
    def validate_test_markers(cls) -> Dict[str, List[str]]:
        """Validate that all tests have appropriate markers"""
        import subprocess
        import json

        # Get all test items with their markers
        result = subprocess.run([
            'poetry', 'run', 'pytest', 'tests/',
            '--collect-only', '--quiet', '--json-report', '--json-report-file=/tmp/test_report.json'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error collecting tests: {result.stderr}")
            return {}

        # Analyze markers
        issues = defaultdict(list)
        try:
            with open('/tmp/test_report.json', 'r') as f:
                report = json.load(f)

            for test in report.get('tests', []):
                test_name = test.get('name', '')
                markers = [m.get('name', '') for m in test.get('markers', [])]

                # Check for required markers
                category_markers = [m for m in markers if m in cls.CATEGORIES]
                if not category_markers:
                    issues['missing_category'].append(test_name)

                # Check for criticality markers
                criticality_markers = [m for m in markers if m in ['critical', 'important', 'optional']]
                if not criticality_markers:
                    issues['missing_criticality'].append(test_name)

        except Exception as e:
            print(f"Error analyzing test markers: {e}")

        return dict(issues)


@pytest.mark.unit
@pytest.mark.critical
class TestTestOrganization:
    """Tests for the test organization system itself"""

    def test_execution_plan_generation(self):
        """Test that execution plans generate valid pytest commands"""
        for plan_name in TestOrganizer.EXECUTION_PLANS:
            cmd = TestOrganizer.get_pytest_command(plan_name)
            assert isinstance(cmd, list)
            assert 'pytest' in cmd[0]
            assert 'tests/' in cmd

    def test_all_categories_defined(self):
        """Test that all expected categories are defined"""
        expected_categories = {
            'unit', 'integration', 'ui', 'strategies',
            'backtesting', 'data', 'risk', 'performance', 'slow'
        }
        actual_categories = set(TestOrganizer.CATEGORIES.keys())
        assert expected_categories.issubset(actual_categories)

    def test_execution_plans_validity(self):
        """Test that execution plans reference valid categories"""
        all_categories = set(TestOrganizer.CATEGORIES.keys())

        for plan_name, plan in TestOrganizer.EXECUTION_PLANS.items():
            # Check includes reference valid categories
            for category in plan['includes']:
                assert category in all_categories, f"Invalid category '{category}' in plan '{plan_name}'"

            # Check excludes reference valid categories
            for category in plan['excludes']:
                assert category in all_categories, f"Invalid category '{category}' in plan '{plan_name}'"


def main():
    """Main entry point for test organization utilities"""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'info':
            TestOrganizer.print_test_organization()
        elif command == 'validate':
            issues = TestOrganizer.validate_test_markers()
            if issues:
                print("⚠️ Test marker validation issues found:")
                for issue_type, tests in issues.items():
                    print(f"\n{issue_type.replace('_', ' ').title()}:")
                    for test in tests[:10]:  # Show first 10
                        print(f"  - {test}")
                    if len(tests) > 10:
                        print(f"  ... and {len(tests) - 10} more")
            else:
                print("✅ All tests have appropriate markers")
        elif command in TestOrganizer.EXECUTION_PLANS:
            cmd = TestOrganizer.get_pytest_command(command)
            print("Generated pytest command:")
            print(" ".join(cmd))
        else:
            print(f"Unknown command: {command}")
            print("Available commands: info, validate, or execution plan names")
    else:
        TestOrganizer.print_test_organization()


if __name__ == "__main__":
    main()