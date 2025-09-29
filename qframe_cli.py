#!/usr/bin/env python
"""
QFrame CLI - Alternative Entry Point
=====================================

Alternative CLI qui contourne le bug Typer pour permettre l'utilisation du framework.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from qframe import __version__
from qframe.core.config import FrameworkConfig


def cmd_version(args):
    """Show QFrame version"""
    print(f"QFrame version {__version__}")


def cmd_info(args):
    """Show framework information"""
    config = FrameworkConfig()
    print("QFrame Configuration")
    print("-" * 40)
    print(f"Version: {__version__}")
    print(f"Environment: {config.environment.value}")
    print(f"Project Root: {config.project_root}")
    print(f"Data Directory: {config.data_dir}")
    print(f"Logs Directory: {config.logs_dir}")
    print(f"Log Level: {config.log_level.value}")


def cmd_strategies(args):
    """List available strategies"""
    strategies = [
        ("DMN LSTM", "Deep Market Networks with LSTM", "Research"),
        ("Mean Reversion", "Statistical mean reversion with ML", "Research"),
        ("Funding Arbitrage", "Cross-exchange funding arbitrage", "Research"),
        ("RL Alpha", "Reinforcement Learning alpha generation", "Research"),
        ("Grid Trading", "Grid-based systematic trading", "Production"),
    ]

    print("Available Strategies")
    print("-" * 60)
    for name, desc, type_ in strategies:
        print(f"{name:20} | {desc:30} | {type_}")


def cmd_test(args):
    """Run framework tests"""
    cmd = ["poetry", "run", "pytest", "tests/"]
    if args.verbose:
        cmd.append("-v")
    if args.coverage:
        cmd.extend(["--cov=qframe", "--cov-report=term-missing"])

    print("Running tests...")
    result = subprocess.run(cmd)
    return result.returncode


def cmd_backtest(args):
    """Run strategy backtest"""
    print(f"Backtesting {args.strategy} strategy")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")

    # Import and run backtest here
    import asyncio
    from examples.backtest_example import run_backtest_example

    try:
        asyncio.run(run_backtest_example())
        print("✓ Backtest completed successfully")
    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        return 1

    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog='qframe',
        description='QFrame - Professional Quantitative Trading Framework'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Version command
    parser_version = subparsers.add_parser('version', help='Show QFrame version')
    parser_version.set_defaults(func=cmd_version)

    # Info command
    parser_info = subparsers.add_parser('info', help='Show framework information')
    parser_info.set_defaults(func=cmd_info)

    # Strategies command
    parser_strategies = subparsers.add_parser('strategies', help='List available strategies')
    parser_strategies.set_defaults(func=cmd_strategies)

    # Test command
    parser_test = subparsers.add_parser('test', help='Run framework tests')
    parser_test.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser_test.add_argument('-c', '--coverage', action='store_true', help='Show coverage report')
    parser_test.set_defaults(func=cmd_test)

    # Backtest command
    parser_backtest = subparsers.add_parser('backtest', help='Run strategy backtest')
    parser_backtest.add_argument('strategy', help='Strategy name to backtest')
    parser_backtest.add_argument('-s', '--symbol', default='BTC/USDT', help='Trading symbol')
    parser_backtest.add_argument('--start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser_backtest.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser_backtest.set_defaults(func=cmd_backtest)

    # Parse arguments
    args = parser.parse_args()

    # Run command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())