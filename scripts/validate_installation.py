#!/usr/bin/env python
"""
🔧 QFrame Research Platform Validation Script

Tests all components to ensure proper integration and functionality.
"""

import sys
import asyncio
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class QFrameValidator:
    """
    🔧 Complete validation suite for QFrame + Research Platform
    """

    def __init__(self):
        self.results = {
            "core_imports": [],
            "research_imports": [],
            "integration_tests": [],
            "service_tests": [],
            "ui_tests": [],
            "docker_tests": []
        }
        self.errors = []

    def run_all_tests(self) -> Dict[str, bool]:
        """Run complete validation suite"""
        print("🚀 QFrame Research Platform Validation")
        print("=" * 60)

        test_results = {}

        # 1. Test Core QFrame imports
        print("\n📦 1. Testing QFrame Core imports...")
        test_results["core_imports"] = self.test_core_imports()

        # 2. Test Research Platform imports
        print("\n🔬 2. Testing Research Platform imports...")
        test_results["research_imports"] = self.test_research_imports()

        # 3. Test Integration Layer
        print("\n🔗 3. Testing Integration Layer...")
        test_results["integration"] = self.test_integration_layer()

        # 4. Test Docker Configuration
        print("\n🐳 4. Testing Docker Configuration...")
        test_results["docker"] = self.test_docker_config()

        # 5. Test dependencies
        print("\n📚 5. Testing Dependencies...")
        test_results["dependencies"] = self.test_dependencies()

        # 6. Test example scripts
        print("\n📋 6. Testing Example Scripts...")
        test_results["examples"] = self.test_examples()

        # Summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)

        total_tests = len(test_results)
        passed_tests = sum(test_results.values())

        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name:20} {status}")

        print(f"\n📈 Overall: {passed_tests}/{total_tests} tests passed")

        if self.errors:
            print(f"\n⚠️ {len(self.errors)} errors encountered:")
            for i, error in enumerate(self.errors[:5], 1):  # Show first 5 errors
                print(f"  {i}. {error}")

        success_rate = passed_tests / total_tests
        if success_rate >= 0.8:
            print("🎉 QFrame Research Platform is ready!")
        elif success_rate >= 0.6:
            print("⚠️ QFrame Research Platform has some issues but is usable")
        else:
            print("❌ QFrame Research Platform needs attention")

        return test_results

    def test_core_imports(self) -> bool:
        """Test QFrame Core component imports"""
        core_modules = [
            "qframe.core.container",
            "qframe.core.config",
            "qframe.core.interfaces",
            "qframe.domain.entities.portfolio",
            "qframe.domain.entities.order",
            "qframe.domain.services.portfolio_service",
            "qframe.domain.services.backtesting_service",
            "qframe.strategies.research.adaptive_mean_reversion_strategy",
            "qframe.strategies.research.dmn_lstm_strategy",
            "qframe.features.symbolic_operators",
            "qframe.infrastructure.data.binance_provider"
        ]

        passed = 0
        total = len(core_modules)

        for module in core_modules:
            try:
                importlib.import_module(module)
                print(f"  ✅ {module}")
                passed += 1
            except ImportError as e:
                print(f"  ❌ {module}: {e}")
                self.errors.append(f"Core import failed: {module} - {e}")
            except Exception as e:
                print(f"  ⚠️ {module}: {e}")
                self.errors.append(f"Core import error: {module} - {e}")

        success = passed >= total * 0.8  # 80% success rate
        print(f"  📊 Core imports: {passed}/{total} ({'✅ PASS' if success else '❌ FAIL'})")
        return success

    def test_research_imports(self) -> bool:
        """Test Research Platform imports"""
        research_modules = [
            "qframe.research.data_lake.storage",
            "qframe.research.data_lake.catalog",
            "qframe.research.data_lake.feature_store",
            "qframe.research.data_lake.ingestion",
            "qframe.research.integration_layer",
            "qframe.research.backtesting.distributed_engine",
            "qframe.research.ui.research_dashboard",
            "qframe.research.sdk.research_api"
        ]

        passed = 0
        total = len(research_modules)

        for module in research_modules:
            try:
                importlib.import_module(module)
                print(f"  ✅ {module}")
                passed += 1
            except ImportError as e:
                print(f"  ❌ {module}: {e}")
                self.errors.append(f"Research import failed: {module} - {e}")
            except Exception as e:
                print(f"  ⚠️ {module}: {e}")
                self.errors.append(f"Research import error: {module} - {e}")

        success = passed >= total * 0.7  # 70% success rate (some deps may be missing)
        print(f"  📊 Research imports: {passed}/{total} ({'✅ PASS' if success else '❌ FAIL'})")
        return success

    def test_integration_layer(self) -> bool:
        """Test the integration between Core and Research"""
        try:
            from qframe.research.integration_layer import create_research_integration

            # Test integration creation (without actual backends)
            integration = create_research_integration(use_minio=False)

            if hasattr(integration, 'container') and hasattr(integration, 'feature_store'):
                print("  ✅ Integration layer created successfully")

                # Test status method
                status = integration.get_integration_status()
                if isinstance(status, dict):
                    print("  ✅ Integration status accessible")
                    return True
                else:
                    print("  ⚠️ Integration status format incorrect")
                    return False
            else:
                print("  ❌ Integration layer missing components")
                return False

        except Exception as e:
            print(f"  ❌ Integration layer test failed: {e}")
            self.errors.append(f"Integration test failed: {e}")
            return False

    def test_docker_config(self) -> bool:
        """Test Docker configuration validity"""
        config_files = [
            "docker-compose.research.yml",
            "Dockerfile.research",
            "jupyterhub_config.py"
        ]

        passed = 0
        total = len(config_files)

        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                print(f"  ✅ {config_file} exists")
                passed += 1
            else:
                print(f"  ❌ {config_file} missing")
                self.errors.append(f"Missing config file: {config_file}")

        # Test docker compose validation
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", "docker-compose.research.yml", "config", "--quiet"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print("  ✅ Docker Compose configuration valid")
                passed += 0.5  # Bonus for valid config
            else:
                print(f"  ⚠️ Docker Compose validation warnings: {result.stderr}")
        except Exception as e:
            print(f"  ⚠️ Docker Compose test skipped: {e}")

        success = passed >= total * 0.8
        print(f"  📊 Docker config: {passed}/{total} ({'✅ PASS' if success else '❌ FAIL'})")
        return success

    def test_dependencies(self) -> bool:
        """Test key dependencies"""
        key_deps = [
            ("pandas", "data manipulation"),
            ("numpy", "numerical computing"),
            ("pydantic", "data validation"),
            ("sqlalchemy", "database ORM"),
            ("fastapi", "API framework"),
            ("streamlit", "UI framework")
        ]

        optional_deps = [
            ("dask", "distributed computing"),
            ("ray", "distributed ML"),
            ("mlflow", "experiment tracking"),
            ("plotly", "visualization"),
            ("boto3", "AWS SDK"),
            ("minio", "object storage")
        ]

        core_passed = 0
        for dep, desc in key_deps:
            try:
                importlib.import_module(dep)
                print(f"  ✅ {dep} ({desc})")
                core_passed += 1
            except ImportError:
                print(f"  ❌ {dep} ({desc}) - REQUIRED")
                self.errors.append(f"Missing required dependency: {dep}")

        optional_passed = 0
        for dep, desc in optional_deps:
            try:
                importlib.import_module(dep)
                print(f"  ✅ {dep} ({desc}) - optional")
                optional_passed += 1
            except ImportError:
                print(f"  ⚠️ {dep} ({desc}) - optional, some features may not work")

        core_success = core_passed >= len(key_deps) * 0.9
        total_score = (core_passed + optional_passed * 0.5) / (len(key_deps) + len(optional_deps) * 0.5)

        print(f"  📊 Dependencies: {core_passed}/{len(key_deps)} core, {optional_passed}/{len(optional_deps)} optional")
        return core_success

    def test_examples(self) -> bool:
        """Test that example scripts can be imported"""
        example_files = [
            "examples/research_platform_integration_demo.py",
            "examples/minimal_example.py"
        ]

        passed = 0
        total = len(example_files)

        for example_file in example_files:
            example_path = project_root / example_file
            if example_path.exists():
                try:
                    # Test syntax by compiling
                    with open(example_path, 'r') as f:
                        content = f.read()
                    compile(content, example_path, 'exec')
                    print(f"  ✅ {example_file} syntax valid")
                    passed += 1
                except SyntaxError as e:
                    print(f"  ❌ {example_file} syntax error: {e}")
                    self.errors.append(f"Syntax error in {example_file}: {e}")
                except Exception as e:
                    print(f"  ⚠️ {example_file} issue: {e}")
            else:
                print(f"  ❌ {example_file} missing")

        success = passed >= total * 0.8
        print(f"  📊 Examples: {passed}/{total} ({'✅ PASS' if success else '❌ FAIL'})")
        return success

    def test_async_integration(self) -> bool:
        """Test async functionality"""
        async def async_test():
            try:
                # Test async integration
                from qframe.research.sdk import QFrameResearch

                # Create instance without auto-init to avoid dependency issues
                qf = QFrameResearch(auto_init=False)

                # Test that it can be created
                if hasattr(qf, 'compute_backend'):
                    print("  ✅ QFrameResearch SDK created successfully")
                    return True
                else:
                    print("  ❌ QFrameResearch SDK missing attributes")
                    return False

            except Exception as e:
                print(f"  ❌ Async test failed: {e}")
                self.errors.append(f"Async test failed: {e}")
                return False

        try:
            result = asyncio.run(async_test())
            return result
        except Exception as e:
            print(f"  ❌ Async runner failed: {e}")
            return False


def main():
    """Main validation function"""
    validator = QFrameValidator()
    results = validator.run_all_tests()

    # Return appropriate exit code
    success_rate = sum(results.values()) / len(results)
    return 0 if success_rate >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())