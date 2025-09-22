"""
QFrame CLI - Command Line Interface
====================================

Interface en ligne de commande pour le framework quantitatif.
"""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
from typing import Optional

from qframe import __version__
from qframe.core.config import FrameworkConfig, Environment

app = typer.Typer(
    name="qframe",
    help="QFrame - Professional Quantitative Trading Framework",
    add_completion=False
)
console = Console()


@app.command()
def version():
    """Show QFrame version"""
    console.print(f"[bold green]QFrame[/bold green] version {__version__}")


@app.command()
def info():
    """Show framework information and configuration"""
    config = FrameworkConfig()

    table = Table(title="QFrame Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Version", __version__)
    table.add_row("Environment", config.environment.value)
    table.add_row("Project Root", str(config.project_root))
    table.add_row("Data Directory", str(config.data_dir))
    table.add_row("Logs Directory", str(config.logs_dir))
    table.add_row("Log Level", config.log_level.value)

    console.print(table)


@app.command()
def strategies():
    """List available trading strategies"""
    strategies_list = [
        ("DMN LSTM", "Deep Market Networks with LSTM architecture", "Research"),
        ("Mean Reversion", "Statistical mean reversion with ML optimization", "Research"),
        ("Funding Arbitrage", "Cross-exchange funding rate arbitrage", "Research"),
        ("RL Alpha", "Reinforcement Learning alpha generation", "Research"),
        ("Grid Trading", "Grid-based systematic trading", "Production"),
    ]

    table = Table(title="Available Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Type", style="yellow")

    for name, desc, type_ in strategies_list:
        table.add_row(name, desc, type_)

    console.print(table)


@app.command()
def test(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    coverage: bool = typer.Option(False, "--coverage", "-c", help="Show coverage report")
):
    """Run framework tests"""
    import subprocess

    cmd = ["poetry", "run", "pytest", "tests/"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=qframe", "--cov-report=term-missing"])

    console.print("[bold cyan]Running tests...[/bold cyan]")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        console.print("[bold green]✓ All tests passed![/bold green]")
    else:
        console.print("[bold red]✗ Some tests failed[/bold red]")

    return result.returncode


@app.command()
def backtest(
    strategy: str = typer.Argument(..., help="Strategy name to backtest"),
    symbol: str = typer.Option("BTC/USDT", "--symbol", "-s", help="Trading symbol"),
    start: str = typer.Option("2024-01-01", "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2024-12-31", "--end", help="End date (YYYY-MM-DD)")
):
    """Run strategy backtest"""
    console.print(f"[bold cyan]Backtesting {strategy} strategy[/bold cyan]")
    console.print(f"Symbol: {symbol}")
    console.print(f"Period: {start} to {end}")

    # TODO: Implement actual backtesting logic
    console.print("[yellow]Backtesting engine not yet implemented[/yellow]")


@app.command()
def train(
    strategy: str = typer.Argument(..., help="Strategy name to train"),
    data_path: Optional[Path] = typer.Option(None, "--data", "-d", help="Path to training data")
):
    """Train a machine learning strategy"""
    console.print(f"[bold cyan]Training {strategy} strategy[/bold cyan]")

    if data_path:
        console.print(f"Using data: {data_path}")

    # TODO: Implement actual training logic
    console.print("[yellow]Training pipeline not yet implemented[/yellow]")


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()