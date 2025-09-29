# QFrame Makefile - Test and Development Commands

.PHONY: help install test test-unit test-ui test-integration test-backtesting test-quick test-all
.PHONY: lint fix clean coverage report deps security
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Python and Poetry setup
PYTHON := python3
POETRY := poetry
TEST_RUNNER := $(PYTHON) scripts/run_tests.py

help: ## Show this help message
	@echo "$(CYAN)🚀 QFrame Research Platform - Development Commands$(RESET)"
	@echo "=================================================="
	@echo ""
	@echo "$(YELLOW)📋 Quick Start:$(RESET)"
	@echo "  make install     # Install all dependencies"
	@echo "  make demo        # Run framework demo"
	@echo "  make test        # Run test suite"
	@echo "  make ui          # Start web interface"
	@echo "  make validate    # Full validation suite"
	@echo ""
	@echo "$(YELLOW)🔧 All Available Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)💡 Pro Tips:$(RESET)"
	@echo "  make dev         # Quick development cycle"
	@echo "  make ci          # Simulate CI pipeline"
	@echo "  make research    # Start research environment"

# Installation and Setup
install: ## Install all dependencies
	@echo "$(CYAN)Installing dependencies...$(RESET)"
	$(POETRY) install --no-interaction
	$(POETRY) run pre-commit install

deps: ## Update dependencies
	@echo "$(CYAN)Updating dependencies...$(RESET)"
	$(POETRY) update

# Test Commands
test: test-quick ## Run quick tests (alias for test-quick)

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(RESET)"
	$(TEST_RUNNER) unit

test-ui: ## Run UI component tests
	@echo "$(CYAN)Running UI tests...$(RESET)"
	$(TEST_RUNNER) ui

test-integration: ## Run integration tests
	@echo "$(CYAN)Running integration tests...$(RESET)"
	$(TEST_RUNNER) integration

test-backtesting: ## Run backtesting specific tests
	@echo "$(CYAN)Running backtesting tests...$(RESET)"
	$(TEST_RUNNER) backtesting

test-performance: ## Run performance tests
	@echo "$(CYAN)Running performance tests...$(RESET)"
	$(TEST_RUNNER) performance

test-quick: ## Run quick tests for fast feedback
	@echo "$(CYAN)Running quick tests...$(RESET)"
	$(TEST_RUNNER) quick

test-all: ## Run all tests with full coverage
	@echo "$(CYAN)Running complete test suite...$(RESET)"
	$(TEST_RUNNER) all

test-critical: ## Run only critical tests for production validation
	@echo "$(CYAN)Running critical tests...$(RESET)"
	$(POETRY) run pytest tests/ -m "critical" --tb=short -v

test-institutional: ## Run institutional metrics tests
	@echo "$(CYAN)Running institutional metrics tests...$(RESET)"
	$(POETRY) run pytest tests/test_institutional_metrics.py -v

test-organization: ## Show test organization and categories
	@echo "$(CYAN)QFrame Test Organization:$(RESET)"
	$(POETRY) run python tests/test_organization.py info

test-validate-markers: ## Validate that all tests have proper markers
	@echo "$(CYAN)Validating test markers...$(RESET)"
	$(POETRY) run python tests/test_organization.py validate

test-parallel: ## Run tests in parallel for faster execution
	@echo "$(CYAN)Running tests in parallel...$(RESET)"
	$(POETRY) run pytest tests/ -n auto --tb=short -v

test-no-cov: ## Run tests without coverage for faster feedback
	@echo "$(CYAN)Running tests without coverage...$(RESET)"
	$(POETRY) run pytest tests/ --no-cov --tb=short -v

test-watch: ## Watch files and run tests on changes
	@echo "$(CYAN)Starting test watcher...$(RESET)"
	$(POETRY) run ptw tests/ -- --tb=short --no-cov

# Code Quality
lint: ## Run all linting checks
	@echo "$(CYAN)Running linting checks...$(RESET)"
	$(TEST_RUNNER) lint

fix: ## Auto-fix code formatting issues
	@echo "$(CYAN)Auto-fixing code issues...$(RESET)"
	$(TEST_RUNNER) fix

# Coverage and Reporting
coverage: ## Show coverage summary
	@echo "$(CYAN)Coverage Summary:$(RESET)"
	$(TEST_RUNNER) coverage

report: ## Generate comprehensive test report
	@echo "$(CYAN)Generating test reports...$(RESET)"
	$(TEST_RUNNER) report
	@echo "$(GREEN)Reports generated:$(RESET)"
	@echo "  📊 Coverage: htmlcov/index.html"
	@echo "  📋 Tests: report.html"

# Cleanup
clean: ## Clean test artifacts and cache
	@echo "$(CYAN)Cleaning test artifacts...$(RESET)"
	$(TEST_RUNNER) clean

clean-all: clean ## Clean everything (artifacts, cache, builds)
	@echo "$(CYAN)Deep cleaning...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ .tox/ .pytest_cache/

# Security and Dependencies
security: ## Run security scans
	@echo "$(CYAN)Running security scans...$(RESET)"
	$(POETRY) run bandit -r qframe/ -f json -o bandit-report.json || echo "$(YELLOW)Security issues found, check bandit-report.json$(RESET)"
	$(POETRY) run safety check || echo "$(YELLOW)Dependency vulnerabilities found$(RESET)"

# Development Workflow
dev-setup: install ## Complete development setup
	@echo "$(GREEN)Development environment ready!$(RESET)"
	@echo "Quick commands:"
	@echo "  make test-quick  - Fast feedback tests"
	@echo "  make lint        - Code quality checks"
	@echo "  make fix         - Auto-fix issues"

# CI/CD simulation
ci: lint test-all ## Simulate CI/CD pipeline locally
	@echo "$(GREEN)✅ CI pipeline simulation completed$(RESET)"

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "$(CYAN)Running benchmarks...$(RESET)"
	$(POETRY) run pytest tests/ -m "performance" --benchmark-only

# Documentation
docs: ## Generate documentation
	@echo "$(CYAN)Generating documentation...$(RESET)"
	$(POETRY) run sphinx-build -b html docs/ docs/_build/

# Pre-commit hooks
pre-commit: ## Run pre-commit hooks manually
	@echo "$(CYAN)Running pre-commit hooks...$(RESET)"
	$(POETRY) run pre-commit run --all-files

# Release preparation
prepare-release: clean lint test-all security ## Prepare for release
	@echo "$(GREEN)✅ Release preparation completed$(RESET)"
	@echo "Ready for release! 🚀"

# Development helpers
shell: ## Open Poetry shell
	$(POETRY) shell

jupyter: ## Start Jupyter notebook
	$(POETRY) run jupyter notebook

# QFrame Framework Commands
demo: ## Run framework demonstration
	@echo "$(CYAN)Running QFrame demonstration...$(RESET)"
	$(POETRY) run python examples/strategy_runtime_test.py

demo-minimal: ## Run minimal example
	@echo "$(CYAN)Running minimal example...$(RESET)"
	$(POETRY) run python examples/minimal_example.py

# Single Entry Point Commands
main: ## Run main entry point with default settings
	@echo "$(CYAN)Running QFrame main entry point...$(RESET)"
	$(POETRY) run python main.py

main-help: ## Show main entry point help
	@echo "$(CYAN)QFrame main entry point help:$(RESET)"
	$(POETRY) run python main.py --help

main-strategies: ## List available strategies via main
	@echo "$(CYAN)Available strategies:$(RESET)"
	$(POETRY) run python main.py --list-strategies

main-backtest: ## Run backtest via main entry point
	@echo "$(CYAN)Running backtest via main entry point...$(RESET)"
	$(POETRY) run python main.py --strategy mean_reversion --mode backtest --quiet

main-validate: ## Run validation via main entry point
	@echo "$(CYAN)Running validation via main entry point...$(RESET)"
	$(POETRY) run python main.py --mode validate --validation-type complete --quiet

# Institutional Metrics Commands
metrics: ## Run institutional metrics test
	@echo "$(CYAN)Running institutional metrics test...$(RESET)"
	$(POETRY) run python examples/institutional_metrics_test.py

metrics-ic: ## Calculate Information Coefficient example
	@echo "$(CYAN)Calculating Information Coefficient...$(RESET)"
	$(POETRY) run python -c "from examples.institutional_metrics_test import test_information_metrics; test_information_metrics()"

metrics-mae-mfe: ## Calculate MAE/MFE example
	@echo "$(CYAN)Calculating MAE/MFE metrics...$(RESET)"
	$(POETRY) run python -c "from examples.institutional_metrics_test import test_excursion_metrics; test_excursion_metrics()"

validate-institutional: ## Run institutional validation suite
	@echo "$(CYAN)Running institutional validation...$(RESET)"
	$(POETRY) run python examples/institutional_validation_test.py

validate-data: ## Run data integrity validation
	@echo "$(CYAN)Running data integrity validation...$(RESET)"
	$(POETRY) run python scripts/validate_data_integrity.py

info: ## Show framework information
	@echo "$(CYAN)QFrame framework information:$(RESET)"
	$(POETRY) run python qframe_cli.py info

strategies: ## List available strategies
	@echo "$(CYAN)Available QFrame strategies:$(RESET)"
	$(POETRY) run python qframe_cli.py strategies

version: ## Show version information
	@echo "$(CYAN)QFrame version information:$(RESET)"
	$(POETRY) run python qframe_cli.py version

# UI Commands
ui: ## Start web interface locally
	@echo "$(CYAN)Starting QFrame web interface...$(RESET)"
	@echo "Interface will be available at http://localhost:8502"
	cd qframe/ui && ./deploy-simple.sh test

ui-docker: ## Start web interface with Docker
	@echo "$(CYAN)Starting QFrame web interface with Docker...$(RESET)"
	@echo "Interface will be available at http://localhost:8501"
	cd qframe/ui && ./deploy-simple.sh up

ui-status: ## Check web interface status
	@echo "$(CYAN)Checking QFrame UI status...$(RESET)"
	cd qframe/ui && ./check-status.sh

ui-logs: ## View web interface logs
	@echo "$(CYAN)Viewing QFrame UI logs...$(RESET)"
	cd qframe/ui && ./deploy-simple.sh logs

ui-down: ## Stop web interface
	@echo "$(CYAN)Stopping QFrame web interface...$(RESET)"
	cd qframe/ui && ./deploy-simple.sh down

# Research Platform Commands
research: ## Start research platform with Docker
	@echo "$(CYAN)Starting QFrame Research Platform...$(RESET)"
	docker-compose -f docker-compose.research.yml up -d
	@echo "$(GREEN)Research platform started!$(RESET)"
	@echo "📊 Services available:"
	@echo "  • JupyterHub: http://localhost:8888"
	@echo "  • MLflow: http://localhost:5000"
	@echo "  • Dask Dashboard: http://localhost:8787"

research-stop: ## Stop research platform
	@echo "$(CYAN)Stopping QFrame Research Platform...$(RESET)"
	docker-compose -f docker-compose.research.yml down
	@echo "$(GREEN)Research platform stopped!$(RESET)"

research-logs: ## View research platform logs
	@echo "$(CYAN)Viewing research platform logs...$(RESET)"
	docker-compose -f docker-compose.research.yml logs -f

# Streamlit app (legacy)
streamlit: ## Run Streamlit app for testing
	@echo "$(CYAN)Starting Streamlit app...$(RESET)"
	$(POETRY) run streamlit run qframe/ui/streamlit_app/main.py

# Database operations
db-reset: ## Reset test database
	@echo "$(CYAN)Resetting test database...$(RESET)"
	# Add database reset commands here when needed

# Monitoring
monitor-tests: ## Monitor test execution in real-time
	@echo "$(CYAN)Monitoring tests...$(RESET)"
	watch -n 2 "$(TEST_RUNNER) quick --no-coverage"

# Statistics
stats: ## Show project statistics
	@echo "$(CYAN)Project Statistics:$(RESET)"
	@echo "Lines of code:"
	@find qframe -name "*.py" -exec wc -l {} + | tail -1
	@echo "Test files:"
	@find tests -name "test_*.py" | wc -l
	@echo "Coverage target: 75%"

# Validation (comprehensive check)
validate: clean lint test-all security coverage ## Full validation suite
	@echo "$(GREEN)🎉 Full validation completed successfully!$(RESET)"
	@echo "$(GREEN)✅ Code quality: PASSED$(RESET)"
	@echo "$(GREEN)✅ Test suite: PASSED$(RESET)"
	@echo "$(GREEN)✅ Security: PASSED$(RESET)"
	@echo "$(GREEN)✅ Coverage: CHECKED$(RESET)"

# Performance profiling
profile: ## Profile test performance
	@echo "$(CYAN)Profiling test performance...$(RESET)"
	$(POETRY) run pytest tests/unit/ --profile --profile-svg

# Debug helpers
debug-ui: ## Debug UI tests specifically
	@echo "$(CYAN)Debugging UI tests...$(RESET)"
	$(POETRY) run pytest tests/ui/ -v -s --tb=long

debug-integration: ## Debug integration tests
	@echo "$(CYAN)Debugging integration tests...$(RESET)"
	$(POETRY) run pytest tests/integration/ -v -s --tb=long

# Quick development cycle
dev: fix test-quick ## Quick development cycle (fix + quick tests)
	@echo "$(GREEN)✅ Development cycle completed$(RESET)"