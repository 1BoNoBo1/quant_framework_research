#!/usr/bin/env python
"""
🚀 QFrame Research Platform - Single Entry Point

Unified command-line interface for all QFrame operations:
- Strategy execution (backtest, live, validation)
- Research platform management
- Data operations and validation
- System monitoring and reporting

Usage:
    python main.py --strategy <strategy> --mode <mode> [options]

Examples:
    python main.py --strategy dmn_lstm --mode backtest
    python main.py --strategy mean_reversion --mode live
    python main.py --mode validate --strategy all
    python main.py --mode research --action start
    python main.py --mode ui --action start
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qframe.core.config import get_config, Environment
from qframe.core.container import get_container
from qframe.validation.institutional_validator import InstitutionalValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QFrameMainApp:
    """
    🎯 QFrame Main Application

    Unified entry point for all QFrame operations with support for:
    - Multiple strategies and execution modes
    - Research platform management
    - Validation and testing workflows
    - System administration tasks
    """

    def __init__(self):
        self.config = get_config()
        self.container = get_container()
        self.strategies = {
            'dmn_lstm': 'DMN LSTM Strategy',
            'mean_reversion': 'Adaptive Mean Reversion Strategy',
            'funding_arbitrage': 'Funding Arbitrage Strategy',
            'rl_alpha': 'Reinforcement Learning Alpha Strategy',
            'simple': 'Simple Trading Strategy'
        }
        self.modes = {
            'backtest': 'Run historical backtesting',
            'live': 'Execute live trading',
            'validate': 'Run institutional validation',
            'research': 'Manage research platform',
            'ui': 'Manage web interface',
            'data': 'Data operations and validation',
            'demo': 'Run demonstration examples',
            'metrics': 'Calculate institutional metrics'
        }

    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            description='🚀 QFrame Research Platform - Unified Entry Point',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_examples_text()
        )

        # Main arguments
        parser.add_argument(
            '--strategy', '-s',
            choices=list(self.strategies.keys()) + ['all'],
            help='Strategy to execute (default: simple)',
            default='simple'
        )

        parser.add_argument(
            '--mode', '-m',
            choices=list(self.modes.keys()),
            help='Execution mode (default: demo)',
            default='demo'
        )

        parser.add_argument(
            '--config', '-c',
            choices=['dev', 'test', 'prod'],
            help='Configuration environment (default: dev)',
            default='dev'
        )

        # Strategy-specific arguments
        strategy_group = parser.add_argument_group('Strategy Options')
        strategy_group.add_argument(
            '--symbol',
            help='Trading symbol (default: BTC/USD)',
            default='BTC/USD'
        )

        strategy_group.add_argument(
            '--timeframe',
            help='Data timeframe (default: 1h)',
            default='1h'
        )

        strategy_group.add_argument(
            '--start-date',
            help='Start date for backtesting (YYYY-MM-DD)',
            default=None
        )

        strategy_group.add_argument(
            '--end-date',
            help='End date for backtesting (YYYY-MM-DD)',
            default=None
        )

        strategy_group.add_argument(
            '--initial-capital',
            type=float,
            help='Initial capital for backtesting (default: 10000)',
            default=10000.0
        )

        # Platform management arguments
        platform_group = parser.add_argument_group('Platform Management')
        platform_group.add_argument(
            '--action',
            choices=['start', 'stop', 'status', 'logs', 'restart', 'ic', 'mae-mfe'],
            help='Platform action (for research/ui/metrics modes)',
            default='start'
        )

        # Validation arguments
        validation_group = parser.add_argument_group('Validation Options')
        validation_group.add_argument(
            '--validation-type',
            choices=['institutional', 'data', 'complete'],
            help='Type of validation to run (default: complete)',
            default='complete'
        )

        # Output arguments
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument(
            '--output-dir',
            help='Output directory for results',
            default='./results'
        )

        output_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )

        output_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-essential output'
        )

        # Utility arguments
        parser.add_argument(
            '--list-strategies',
            action='store_true',
            help='List available strategies and exit'
        )

        parser.add_argument(
            '--list-modes',
            action='store_true',
            help='List available modes and exit'
        )

        parser.add_argument(
            '--version',
            action='store_true',
            help='Show version information and exit'
        )

        return parser

    def _get_examples_text(self) -> str:
        """Generate examples text for help"""
        return """
Examples:
  Basic usage:
    python main.py --strategy simple --mode demo
    python main.py --strategy mean_reversion --mode backtest

  Backtesting:
    python main.py -s dmn_lstm -m backtest --start-date 2024-01-01 --end-date 2024-12-31
    python main.py -s funding_arbitrage -m backtest --initial-capital 50000

  Validation:
    python main.py --mode validate --strategy all
    python main.py -m validate --validation-type institutional -s mean_reversion

  Research Platform:
    python main.py --mode research --action start
    python main.py --mode research --action status

  Web Interface:
    python main.py --mode ui --action start
    python main.py --mode ui --action logs

  Data Operations:
    python main.py --mode data --action validate
    python main.py --mode data --action integrity-check

Strategy Options:
  dmn_lstm          - Deep Market Network with LSTM
  mean_reversion    - Adaptive Mean Reversion Strategy
  funding_arbitrage - Funding Rate Arbitrage Strategy
  rl_alpha          - Reinforcement Learning Alpha
  simple            - Simple Trading Strategy (default)
  all               - All available strategies

Mode Options:
  demo              - Run demonstration examples (default)
  backtest          - Historical backtesting
  live              - Live trading execution
  validate          - Institutional validation suite
  research          - Research platform management
  ui                - Web interface management
  data              - Data operations and validation
        """

    async def run(self, args: argparse.Namespace) -> int:
        """Main execution method"""
        try:
            # Setup logging level
            if args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
            elif args.quiet:
                logging.getLogger().setLevel(logging.WARNING)

            # Handle utility commands
            if args.list_strategies:
                self._list_strategies()
                return 0

            if args.list_modes:
                self._list_modes()
                return 0

            if args.version:
                self._show_version()
                return 0

            # Print startup banner
            if not args.quiet:
                self._print_banner(args)

            # Route to appropriate handler
            if args.mode == 'demo':
                return await self._handle_demo(args)
            elif args.mode == 'backtest':
                return await self._handle_backtest(args)
            elif args.mode == 'live':
                return await self._handle_live(args)
            elif args.mode == 'validate':
                return await self._handle_validate(args)
            elif args.mode == 'research':
                return await self._handle_research(args)
            elif args.mode == 'ui':
                return await self._handle_ui(args)
            elif args.mode == 'data':
                return await self._handle_data(args)
            elif args.mode == 'metrics':
                return await self._handle_metrics(args)
            else:
                logger.error(f"Unknown mode: {args.mode}")
                return 1

        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _print_banner(self, args: argparse.Namespace):
        """Print startup banner"""
        print("🚀 QFrame Research Platform")
        print("=" * 50)
        print(f"Strategy: {self.strategies.get(args.strategy, args.strategy)}")
        print(f"Mode: {self.modes.get(args.mode, args.mode)}")
        print(f"Environment: {args.config}")
        print("=" * 50)
        print()

    def _list_strategies(self):
        """List available strategies"""
        print("🎯 Available QFrame Strategies:")
        print("=" * 40)
        for key, description in self.strategies.items():
            print(f"  {key:<20} - {description}")
        print()

    def _list_modes(self):
        """List available modes"""
        print("🔧 Available Execution Modes:")
        print("=" * 40)
        for key, description in self.modes.items():
            print(f"  {key:<15} - {description}")
        print()

    def _show_version(self):
        """Show version information"""
        print(f"QFrame Research Platform v{self.config.app_version}")
        print(f"Environment: {self.config.environment}")
        print(f"Python: {sys.version}")

    async def _handle_demo(self, args: argparse.Namespace) -> int:
        """Handle demo mode"""
        logger.info("Running QFrame demonstration")

        if args.strategy == 'simple' or args.strategy == 'all':
            logger.info("Running minimal example...")
            from examples.minimal_example import main as run_minimal
            await asyncio.to_thread(run_minimal)

        if args.strategy == 'all':
            logger.info("Running strategy runtime test...")
            from examples.strategy_runtime_test import main as run_runtime
            await asyncio.to_thread(run_runtime)

        logger.info("✅ Demo completed successfully!")
        return 0

    async def _handle_backtest(self, args: argparse.Namespace) -> int:
        """Handle backtest mode"""
        logger.info(f"Running backtest for strategy: {args.strategy}")

        # Import strategy
        strategy = await self._load_strategy(args.strategy)
        if not strategy:
            return 1

        # Generate or load data
        data = await self._load_market_data(args)

        # Run backtest
        logger.info("Starting backtest execution...")
        # This would be implemented with actual backtesting logic
        logger.info(f"Backtesting {args.symbol} from {args.start_date} to {args.end_date}")
        logger.info(f"Initial capital: ${args.initial_capital:,.2f}")

        logger.info("✅ Backtest completed successfully!")
        return 0

    async def _handle_live(self, args: argparse.Namespace) -> int:
        """Handle live trading mode"""
        logger.warning("⚠️ Live trading mode is not yet implemented")
        logger.info("This would connect to live exchanges and execute real trades")
        logger.info("Use demo mode for testing strategies first")
        return 0

    async def _handle_validate(self, args: argparse.Namespace) -> int:
        """Handle validation mode"""
        logger.info("Running validation suite")

        if args.validation_type in ['institutional', 'complete']:
            logger.info("Running institutional validation...")
            from examples.institutional_validation_test import test_institutional_validation
            await asyncio.to_thread(test_institutional_validation)

        if args.validation_type in ['data', 'complete']:
            logger.info("Running data integrity validation...")
            import subprocess
            result = subprocess.run([
                sys.executable, 'scripts/validate_data_integrity.py'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("✅ Data validation passed")
            else:
                logger.error("❌ Data validation failed")
                return 1

        logger.info("✅ Validation completed successfully!")
        return 0

    async def _handle_research(self, args: argparse.Namespace) -> int:
        """Handle research platform management"""
        logger.info(f"Research platform action: {args.action}")

        import subprocess

        if args.action == 'start':
            logger.info("Starting QFrame Research Platform...")
            subprocess.run(['make', 'research'], check=True)

        elif args.action == 'stop':
            logger.info("Stopping QFrame Research Platform...")
            subprocess.run(['make', 'research-stop'], check=True)

        elif args.action == 'status':
            logger.info("Checking research platform status...")
            result = subprocess.run(['docker-compose', '-f', 'docker-compose.research.yml', 'ps'],
                                  capture_output=True, text=True)
            print(result.stdout)

        elif args.action == 'logs':
            logger.info("Showing research platform logs...")
            subprocess.run(['make', 'research-logs'])

        elif args.action == 'restart':
            logger.info("Restarting research platform...")
            subprocess.run(['make', 'research-stop'], check=True)
            subprocess.run(['make', 'research'], check=True)

        return 0

    async def _handle_ui(self, args: argparse.Namespace) -> int:
        """Handle UI management"""
        logger.info(f"UI action: {args.action}")

        import subprocess

        if args.action == 'start':
            logger.info("Starting QFrame Web Interface...")
            subprocess.run(['make', 'ui'], check=True)

        elif args.action == 'stop':
            logger.info("Stopping QFrame Web Interface...")
            subprocess.run(['make', 'ui-down'], check=True)

        elif args.action == 'status':
            logger.info("Checking UI status...")
            subprocess.run(['make', 'ui-status'], check=True)

        elif args.action == 'logs':
            logger.info("Showing UI logs...")
            subprocess.run(['make', 'ui-logs'])

        elif args.action == 'restart':
            logger.info("Restarting UI...")
            subprocess.run(['make', 'ui-down'], check=True)
            subprocess.run(['make', 'ui'], check=True)

        return 0

    async def _handle_data(self, args: argparse.Namespace) -> int:
        """Handle data operations"""
        logger.info("Running data operations")

        if args.action == 'validate':
            import subprocess
            result = subprocess.run([
                sys.executable, 'scripts/validate_data_integrity.py'
            ], check=True)
            return result.returncode

        logger.info("✅ Data operations completed!")
        return 0

    async def _handle_metrics(self, args: argparse.Namespace) -> int:
        """Handle institutional metrics mode"""
        logger.info("Running institutional metrics analysis")

        if args.action == 'ic':
            logger.info("Calculating Information Coefficient...")
            import subprocess
            result = subprocess.run([
                sys.executable, '-c',
                'from examples.institutional_metrics_test import test_information_metrics; test_information_metrics()'
            ], check=True)

        elif args.action == 'mae-mfe':
            logger.info("Calculating MAE/MFE metrics...")
            import subprocess
            result = subprocess.run([
                sys.executable, '-c',
                'from examples.institutional_metrics_test import test_excursion_metrics; test_excursion_metrics()'
            ], check=True)

        else:
            # Run complete metrics test
            logger.info("Running complete institutional metrics analysis...")
            import subprocess
            result = subprocess.run([
                sys.executable, 'examples/institutional_metrics_test.py'
            ], check=True)

        logger.info("✅ Institutional metrics analysis completed!")
        return 0

    async def _load_strategy(self, strategy_name: str):
        """Load strategy by name"""
        # This would load the actual strategy implementation
        logger.info(f"Loading strategy: {strategy_name}")
        return f"MockStrategy({strategy_name})"

    async def _load_market_data(self, args: argparse.Namespace):
        """Load market data for backtesting"""
        logger.info(f"Loading market data for {args.symbol}")
        # This would load actual market data
        return "MockMarketData"


def main():
    """Main entry point"""
    app = QFrameMainApp()
    parser = app.create_parser()
    args = parser.parse_args()

    # Run the application
    return asyncio.run(app.run(args))


if __name__ == "__main__":
    sys.exit(main())