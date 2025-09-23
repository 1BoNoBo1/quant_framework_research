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
import asyncio
import subprocess
import sys
import os

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
    cmd = ["poetry", "run", "pytest", "tests/"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=qframe", "--cov-report=term-missing"])

    console.print("[bold cyan]Running tests...[/bold cyan]")
    try:
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            console.print("[bold green]✓ All tests passed![/bold green]")
        else:
            console.print("[bold red]✗ Some tests failed[/bold red]")
        return result.returncode
    except Exception as e:
        console.print(f"[bold red]Error running tests: {e}[/bold red]")
        return 1


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


@app.command("download")
def download_data(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTC/USDT)"),
    exchange: str = typer.Option("binance", "--exchange", "-e", help="Exchange to use"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t",
                                 help="Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)"),
    days: int = typer.Option(30, "--days", "-d", help="Number of days to download"),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
                                        help="Output directory (default: data/)"),
    list_symbols: bool = typer.Option(False, "--list", help="List available symbols")
):
    """Download market data using the existing data download scripts"""

    # Set default output directory
    if output is None:
        config = FrameworkConfig()
        output = config.data_dir

    # Build command to run the download script
    script_path = Path(__file__).parent.parent.parent / "scripts" / "download_market_data.py"

    if not script_path.exists():
        console.print(f"[bold red]Error: Download script not found at {script_path}[/bold red]")
        return 1

    cmd = [sys.executable, str(script_path)]

    if list_symbols:
        cmd.extend(["--exchange", exchange, "--list-symbols"])
        console.print(f"[bold cyan]Listing available symbols on {exchange}...[/bold cyan]")
    else:
        cmd.extend([
            "--exchange", exchange,
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--days", str(days),
            "--output", str(output)
        ])
        console.print(f"[bold cyan]Downloading {symbol} data from {exchange}...[/bold cyan]")
        console.print(f"Exchange: {exchange}")
        console.print(f"Symbol: {symbol}")
        console.print(f"Timeframe: {timeframe}")
        console.print(f"Days: {days}")
        console.print(f"Output: {output}")

    try:
        # Run the download script
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            if not list_symbols:
                console.print(f"[bold green]✓ Successfully downloaded {symbol} data![/bold green]")
                console.print(f"Data saved to: {output}")
        else:
            console.print("[bold red]✗ Download failed[/bold red]")

        return result.returncode

    except Exception as e:
        console.print(f"[bold red]Error running download script: {e}[/bold red]")
        return 1


@app.command("batch-download")
def batch_download_data(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c",
                                             help="YAML configuration file"),
    mode: str = typer.Option("batch", "--mode", "-m",
                           help="Download mode: batch, symbol, all"),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s",
                                       help="Symbol for mode=symbol"),
    group: str = typer.Option("crypto_majors", "--group", "-g",
                            help="Symbol group for mode=batch"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    period: str = typer.Option("medium", "--period", "-p",
                             help="Period: short, medium, long")
):
    """Batch download multiple symbols using configuration"""

    # Build command to run the batch download script
    script_path = Path(__file__).parent.parent.parent / "scripts" / "batch_download.py"

    if not script_path.exists():
        console.print(f"[bold red]Error: Batch download script not found at {script_path}[/bold red]")
        return 1

    cmd = [sys.executable, str(script_path)]

    if config_file:
        cmd.extend(["--config", str(config_file)])

    cmd.extend([
        "--mode", mode,
        "--group", group,
        "--timeframe", timeframe,
        "--period", period
    ])

    if symbol and mode == "symbol":
        cmd.extend(["--symbol", symbol])

    console.print(f"[bold cyan]Running batch download...[/bold cyan]")
    console.print(f"Mode: {mode}")
    console.print(f"Group: {group}")
    console.print(f"Timeframe: {timeframe}")
    console.print(f"Period: {period}")

    if symbol:
        console.print(f"Symbol: {symbol}")

    try:
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            console.print("[bold green]✓ Batch download completed successfully![/bold green]")
        else:
            console.print("[bold red]✗ Batch download failed[/bold red]")

        return result.returncode

    except Exception as e:
        console.print(f"[bold red]Error running batch download script: {e}[/bold red]")
        return 1


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()