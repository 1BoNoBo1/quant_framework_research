#!/usr/bin/env python3
"""
Script de test runner personnalisé pour QFrame.
Fournit une interface simple pour exécuter différents types de tests.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime


class QFrameTestRunner:
    """Runner de tests personnalisé pour QFrame."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        self.coverage_dir = self.project_root / "htmlcov"

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Exécute une commande et retourne True si succès."""
        print(f"\n🔄 {description}")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=True)
            print(f"✅ {description} - SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} - FAILED (exit code: {e.returncode})")
            return False

    def run_unit_tests(self, coverage: bool = True, verbose: bool = False) -> bool:
        """Exécute les tests unitaires."""
        cmd = ["poetry", "run", "pytest", "tests/unit/"]

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        if coverage:
            cmd.extend([
                "--cov=qframe",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ])

        cmd.extend(["-m", "not slow"])

        return self.run_command(cmd, "Running Unit Tests")

    def run_ui_tests(self, coverage: bool = True, verbose: bool = False) -> bool:
        """Exécute les tests d'interface utilisateur."""
        cmd = ["poetry", "run", "pytest", "tests/ui/"]

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        if coverage:
            cmd.extend([
                "--cov=qframe.ui",
                "--cov-append",
                "--cov-report=term-missing"
            ])

        cmd.extend(["-m", "ui and not slow"])

        return self.run_command(cmd, "Running UI Tests")

    def run_integration_tests(self, verbose: bool = False) -> bool:
        """Exécute les tests d'intégration."""
        cmd = ["poetry", "run", "pytest", "tests/integration/"]

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        cmd.extend([
            "--cov=qframe",
            "--cov-append",
            "--timeout=300"
        ])

        return self.run_command(cmd, "Running Integration Tests")

    def run_backtesting_tests(self, coverage: bool = True, verbose: bool = False) -> bool:
        """Exécute les tests spécifiques au backtesting."""
        cmd = ["poetry", "run", "pytest"]

        # Inclure tous les tests marqués backtesting
        cmd.extend(["-m", "backtesting"])

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        if coverage:
            cmd.extend([
                "--cov=qframe",
                "--cov-append",
                "--cov-report=term-missing"
            ])

        return self.run_command(cmd, "Running Backtesting Tests")

    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Exécute les tests de performance."""
        cmd = ["poetry", "run", "pytest"]
        cmd.extend(["-m", "performance or slow"])
        cmd.extend(["--timeout=600"])

        if verbose:
            cmd.append("-v")

        return self.run_command(cmd, "Running Performance Tests")

    def run_all_tests(self, coverage: bool = True, verbose: bool = False) -> bool:
        """Exécute tous les tests."""
        cmd = ["poetry", "run", "pytest", "tests/"]

        if verbose:
            cmd.append("-v")

        if coverage:
            cmd.extend([
                "--cov=qframe",
                "--cov-branch",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                "--cov-fail-under=75"
            ])

        return self.run_command(cmd, "Running All Tests")

    def run_quick_tests(self, verbose: bool = False) -> bool:
        """Exécute les tests rapides pour feedback immédiat."""
        cmd = ["poetry", "run", "pytest", "tests/unit/", "tests/ui/"]
        cmd.extend(["-m", "not slow and not performance"])

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        cmd.extend([
            "--cov=qframe",
            "--cov-report=term-missing",
            "--maxfail=5"  # S'arrêter après 5 échecs
        ])

        return self.run_command(cmd, "Running Quick Tests")

    def run_lint_checks(self) -> bool:
        """Exécute les vérifications de lint."""
        success = True

        # Black formatting check
        if not self.run_command(
            ["poetry", "run", "black", "--check", "qframe/"],
            "Checking code formatting with Black"
        ):
            success = False

        # Ruff linting
        if not self.run_command(
            ["poetry", "run", "ruff", "check", "qframe/"],
            "Linting code with Ruff"
        ):
            success = False

        # MyPy type checking
        if not self.run_command(
            ["poetry", "run", "mypy", "qframe/", "--ignore-missing-imports"],
            "Type checking with MyPy"
        ):
            success = False

        return success

    def fix_formatting(self) -> bool:
        """Corrige automatiquement le formatage."""
        success = True

        # Format with Black
        if not self.run_command(
            ["poetry", "run", "black", "qframe/"],
            "Formatting code with Black"
        ):
            success = False

        # Fix with Ruff
        if not self.run_command(
            ["poetry", "run", "ruff", "check", "--fix", "qframe/"],
            "Auto-fixing issues with Ruff"
        ):
            success = False

        return success

    def generate_test_report(self) -> bool:
        """Génère un rapport de test détaillé."""
        cmd = [
            "poetry", "run", "pytest", "tests/",
            "--cov=qframe",
            "--cov-branch",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--junit-xml=junit.xml",
            "--html=report.html",
            "--self-contained-html"
        ]

        success = self.run_command(cmd, "Generating comprehensive test report")

        if success:
            print(f"\n📊 Test reports generated:")
            print(f"  - HTML Coverage: {self.coverage_dir}/index.html")
            print(f"  - XML Coverage: {self.project_root}/coverage.xml")
            print(f"  - Test Report: {self.project_root}/report.html")

        return success

    def clean_test_artifacts(self) -> bool:
        """Nettoie les artefacts de test."""
        artifacts = [
            ".coverage",
            "coverage.xml",
            "coverage.json",
            "junit.xml",
            "report.html",
            "htmlcov",
            ".pytest_cache",
            "__pycache__"
        ]

        for artifact in artifacts:
            artifact_path = self.project_root / artifact
            if artifact_path.exists():
                if artifact_path.is_dir():
                    subprocess.run(["rm", "-rf", str(artifact_path)])
                else:
                    artifact_path.unlink()
                print(f"🗑️  Removed {artifact}")

        print("✅ Test artifacts cleaned")
        return True

    def show_coverage_summary(self) -> None:
        """Affiche un résumé de coverage."""
        coverage_file = self.project_root / "coverage.json"

        if not coverage_file.exists():
            print("❌ No coverage data found. Run tests with coverage first.")
            return

        try:
            with open(coverage_file) as f:
                coverage_data = json.load(f)

            total_coverage = coverage_data.get("totals", {})

            print("\n📊 Coverage Summary:")
            print(f"  Lines: {total_coverage.get('percent_covered', 0):.1f}%")
            print(f"  Branches: {total_coverage.get('percent_covered_display', 'N/A')}")
            print(f"  Missing: {total_coverage.get('missing_lines', 0)} lines")

            # Coverage par fichier
            files = coverage_data.get("files", {})
            print(f"\n📁 Coverage by module:")

            for file_path, file_data in sorted(files.items()):
                if "qframe/ui" in file_path:
                    module_name = "UI Components"
                elif "qframe/core" in file_path:
                    module_name = "Core Framework"
                elif "qframe/strategies" in file_path:
                    module_name = "Strategies"
                else:
                    module_name = "Other"

                coverage_pct = file_data.get("summary", {}).get("percent_covered", 0)
                if coverage_pct < 75:
                    status = "❌"
                elif coverage_pct < 90:
                    status = "⚠️"
                else:
                    status = "✅"

                print(f"  {status} {module_name}: {coverage_pct:.1f}%")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Error reading coverage data: {e}")

    def run_test_suite(self, suite_name: str, **kwargs) -> bool:
        """Exécute une suite de tests spécifique."""
        suites = {
            "unit": self.run_unit_tests,
            "ui": self.run_ui_tests,
            "integration": self.run_integration_tests,
            "backtesting": self.run_backtesting_tests,
            "performance": self.run_performance_tests,
            "all": self.run_all_tests,
            "quick": self.run_quick_tests
        }

        if suite_name not in suites:
            print(f"❌ Unknown test suite: {suite_name}")
            print(f"Available suites: {', '.join(suites.keys())}")
            return False

        return suites[suite_name](**kwargs)


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="QFrame Test Runner - Execute tests with ease",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s unit                    # Run unit tests
  %(prog)s ui --verbose           # Run UI tests with verbose output
  %(prog)s all --no-coverage      # Run all tests without coverage
  %(prog)s quick                  # Run quick tests for fast feedback
  %(prog)s lint                   # Run linting checks
  %(prog)s report                 # Generate comprehensive test report
  %(prog)s clean                  # Clean test artifacts
        """
    )

    parser.add_argument(
        "command",
        choices=[
            "unit", "ui", "integration", "backtesting", "performance",
            "all", "quick", "lint", "fix", "report", "clean", "coverage"
        ],
        help="Test command to execute"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage collection"
    )

    args = parser.parse_args()

    runner = QFrameTestRunner()

    print("🧪 QFrame Test Runner")
    print("=" * 50)

    success = True

    if args.command == "lint":
        success = runner.run_lint_checks()
    elif args.command == "fix":
        success = runner.fix_formatting()
    elif args.command == "report":
        success = runner.generate_test_report()
    elif args.command == "clean":
        success = runner.clean_test_artifacts()
    elif args.command == "coverage":
        runner.show_coverage_summary()
    else:
        # Test suites
        kwargs = {
            "verbose": args.verbose,
            "coverage": not args.no_coverage
        }
        success = runner.run_test_suite(args.command, **kwargs)

    if success:
        print("\n🎉 All operations completed successfully!")
        if args.command in ["unit", "ui", "all", "quick"] and not args.no_coverage:
            print(f"📊 Coverage report: {runner.coverage_dir}/index.html")
    else:
        print("\n❌ Some operations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()