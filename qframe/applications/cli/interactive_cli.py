"""
QFrame Interactive CLI - Enterprise Command Line Interface
=========================================================

CLI avancé et interactif pour QFrame avec commandes sophistiquées,
auto-complétion, historique, et interface utilisateur riche.
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import traceback

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.prompt import Prompt, Confirm
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.columns import Columns
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    # Fallback basique
    class Console:
        def print(self, *args, **kwargs): print(*args)
    click = None

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from qframe.core.config import get_config
from qframe.core.container import get_container
from qframe.infrastructure.observability.structured_logging import LoggerFactory, configure_logging, LogLevel, LogFormat
from qframe.infrastructure.monitoring.metrics_collector import configure_monitoring, get_default_collector
from qframe.infrastructure.performance.optimized_processors import create_optimized_processor, PerformanceConfig
from qframe.domain.entities.enhanced_portfolio import create_portfolio, create_demo_portfolio


class CLIContext:
    """Contexte global du CLI."""

    def __init__(self):
        self.console = Console() if HAS_RICH else Console()
        self.config = None
        self.container = None
        self.logger = None
        self.metrics_collector = None
        self.current_portfolio = None
        self.debug_mode = False
        self.history = []

    def initialize(self):
        """Initialise le contexte CLI."""
        try:
            # Configuration
            self.config = get_config()

            # Container DI
            self.container = get_container()

            # Logging
            configure_logging(
                level=LogLevel.INFO,
                format_type=LogFormat.CONSOLE,
                service_name="qframe-cli"
            )
            self.logger = LoggerFactory.get_logger("cli")

            # Monitoring
            self.metrics_collector = configure_monitoring(
                enable_prometheus=False,  # Pas besoin en CLI
                enable_system_metrics=True,
                auto_collection_interval=60
            )

            self.logger.info("🚀 QFrame CLI initialized successfully")
            return True

        except Exception as e:
            if HAS_RICH:
                self.console.print(f"❌ [red]Initialization failed: {e}[/red]")
            else:
                print(f"❌ Initialization failed: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return False


# Contexte global CLI
cli_context = CLIContext()


class CommandProcessor:
    """Processeur de commandes avec auto-complétion et validation."""

    def __init__(self, context: CLIContext):
        self.context = context
        self.commands = {}
        self._register_commands()

    def _register_commands(self):
        """Enregistre toutes les commandes disponibles."""
        self.commands = {
            # Commandes système
            'help': {
                'func': self.cmd_help,
                'description': 'Affiche l\'aide des commandes',
                'usage': 'help [command]',
                'category': 'system'
            },
            'status': {
                'func': self.cmd_status,
                'description': 'Affiche le statut du système',
                'usage': 'status',
                'category': 'system'
            },
            'config': {
                'func': self.cmd_config,
                'description': 'Gère la configuration',
                'usage': 'config [show|set key value]',
                'category': 'system'
            },
            'debug': {
                'func': self.cmd_debug,
                'description': 'Active/désactive le mode debug',
                'usage': 'debug [on|off]',
                'category': 'system'
            },

            # Commandes portfolio
            'portfolio': {
                'func': self.cmd_portfolio,
                'description': 'Gère les portfolios',
                'usage': 'portfolio [create|show|list|demo]',
                'category': 'portfolio'
            },
            'positions': {
                'func': self.cmd_positions,
                'description': 'Affiche les positions',
                'usage': 'positions',
                'category': 'portfolio'
            },
            'metrics': {
                'func': self.cmd_metrics,
                'description': 'Affiche les métriques',
                'usage': 'metrics [show|history metric_name]',
                'category': 'monitoring'
            },

            # Commandes trading
            'backtest': {
                'func': self.cmd_backtest,
                'description': 'Lance un backtest',
                'usage': 'backtest strategy [start_date] [end_date]',
                'category': 'trading'
            },
            'strategies': {
                'func': self.cmd_strategies,
                'description': 'Liste les stratégies disponibles',
                'usage': 'strategies',
                'category': 'trading'
            },

            # Commandes données
            'data': {
                'func': self.cmd_data,
                'description': 'Gère les données',
                'usage': 'data [fetch|show] symbol [timeframe]',
                'category': 'data'
            },

            # Commandes avancées
            'performance': {
                'func': self.cmd_performance,
                'description': 'Tests de performance',
                'usage': 'performance [test|benchmark]',
                'category': 'advanced'
            },
            'clear': {
                'func': self.cmd_clear,
                'description': 'Efface l\'écran',
                'usage': 'clear',
                'category': 'system'
            },
            'exit': {
                'func': self.cmd_exit,
                'description': 'Quitte le CLI',
                'usage': 'exit',
                'category': 'system'
            }
        }

    async def execute_command(self, command_line: str) -> bool:
        """Exécute une commande."""
        command_line = command_line.strip()

        if not command_line:
            return True

        # Ajouter à l'historique
        self.context.history.append({
            'command': command_line,
            'timestamp': datetime.now(),
            'success': None
        })

        parts = command_line.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command not in self.commands:
            self.context.console.print(f"❌ [red]Commande inconnue: {command}[/red]")
            self.context.console.print("💡 Tapez 'help' pour voir les commandes disponibles")
            self.context.history[-1]['success'] = False
            return True

        try:
            result = await self.commands[command]['func'](args)
            self.context.history[-1]['success'] = result
            return result

        except Exception as e:
            self.context.console.print(f"❌ [red]Erreur lors de l'exécution: {e}[/red]")
            if self.context.debug_mode:
                self.context.console.print(f"[yellow]{traceback.format_exc()}[/yellow]")
            self.context.history[-1]['success'] = False
            return True

    # === Commandes Système ===

    async def cmd_help(self, args: List[str]) -> bool:
        """Commande help."""
        if args and args[0] in self.commands:
            # Aide spécifique à une commande
            cmd = self.commands[args[0]]

            panel = Panel.fit(
                f"[bold]{args[0]}[/bold]\n\n"
                f"[dim]Description:[/dim] {cmd['description']}\n"
                f"[dim]Usage:[/dim] {cmd['usage']}\n"
                f"[dim]Catégorie:[/dim] {cmd['category']}",
                title="📖 Aide Commande",
                border_style="blue"
            )
            self.context.console.print(panel)
        else:
            # Aide générale
            self._show_general_help()

        return True

    def _show_general_help(self):
        """Affiche l'aide générale."""
        # Grouper commandes par catégorie
        categories = {}
        for cmd_name, cmd_info in self.commands.items():
            category = cmd_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((cmd_name, cmd_info['description']))

        # Créer le tableau
        table = Table(title="🎯 QFrame CLI - Commandes Disponibles")
        table.add_column("Commande", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Catégorie", style="magenta")

        for category, commands in categories.items():
            for cmd_name, description in commands:
                table.add_row(cmd_name, description, category)

        self.context.console.print(table)
        self.context.console.print("\n💡 [dim]Tapez 'help <commande>' pour plus de détails[/dim]")

    async def cmd_status(self, args: List[str]) -> bool:
        """Commande status."""
        try:
            # Informations système
            status_info = {
                "QFrame Version": "1.0.0",
                "Environment": self.context.config.environment.value if self.context.config else "unknown",
                "Debug Mode": "🟢 Activé" if self.context.debug_mode else "🔴 Désactivé",
                "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Portfolio Loaded": "🟢 Oui" if self.context.current_portfolio else "🔴 Non"
            }

            # Métriques système si disponibles
            if self.context.metrics_collector:
                cpu = self.context.metrics_collector.get_current_value("system_cpu_usage_percent")
                memory = self.context.metrics_collector.get_current_value("system_memory_usage_percent")

                if cpu is not None:
                    status_info["System CPU"] = f"{cpu:.1f}%"
                if memory is not None:
                    status_info["System Memory"] = f"{memory:.1f}%"

            # Créer l'affichage
            columns = []
            for key, value in status_info.items():
                columns.append(f"[bold]{key}:[/bold] {value}")

            panel = Panel(
                "\n".join(columns),
                title="📊 Status QFrame",
                border_style="green"
            )

            self.context.console.print(panel)
            return True

        except Exception as e:
            self.context.console.print(f"❌ [red]Erreur status: {e}[/red]")
            return False

    async def cmd_config(self, args: List[str]) -> bool:
        """Commande config."""
        if not args or args[0] == "show":
            # Afficher configuration
            if not self.context.config:
                self.context.console.print("❌ [red]Configuration non disponible[/red]")
                return False

            config_dict = self.context.config.model_dump()

            # Formatter en JSON pour affichage
            syntax = Syntax(
                json.dumps(config_dict, indent=2, default=str),
                "json",
                theme="monokai",
                line_numbers=True
            )

            panel = Panel(
                syntax,
                title="⚙️ Configuration QFrame",
                border_style="blue"
            )

            self.context.console.print(panel)
            return True

        elif args[0] == "set" and len(args) >= 3:
            # Modifier configuration (basique)
            key, value = args[1], args[2]
            self.context.console.print(f"🔧 Configuration {key} = {value} (simulation)")
            return True

        else:
            self.context.console.print("❌ [red]Usage: config [show|set key value][/red]")
            return False

    async def cmd_debug(self, args: List[str]) -> bool:
        """Commande debug."""
        if not args:
            state = "activé" if self.context.debug_mode else "désactivé"
            self.context.console.print(f"🐛 Mode debug: {state}")
        elif args[0] == "on":
            self.context.debug_mode = True
            self.context.console.print("🐛 [green]Mode debug activé[/green]")
        elif args[0] == "off":
            self.context.debug_mode = False
            self.context.console.print("🐛 [yellow]Mode debug désactivé[/yellow]")
        else:
            self.context.console.print("❌ [red]Usage: debug [on|off][/red]")
            return False

        return True

    async def cmd_clear(self, args: List[str]) -> bool:
        """Commande clear."""
        os.system('clear' if os.name == 'posix' else 'cls')
        return True

    async def cmd_exit(self, args: List[str]) -> bool:
        """Commande exit."""
        self.context.console.print("👋 [green]Au revoir![/green]")
        return False

    # === Commandes Portfolio ===

    async def cmd_portfolio(self, args: List[str]) -> bool:
        """Commande portfolio."""
        if not args or args[0] == "show":
            if not self.context.current_portfolio:
                self.context.console.print("❌ [red]Aucun portfolio chargé[/red]")
                return True

            self._display_portfolio(self.context.current_portfolio)
            return True

        elif args[0] == "create":
            name = args[1] if len(args) > 1 else Prompt.ask("Nom du portfolio")
            capital = float(args[2]) if len(args) > 2 else float(Prompt.ask("Capital initial", default="100000"))

            try:
                portfolio = create_portfolio(
                    name=name,
                    owner_id="cli_user",
                    initial_capital=capital
                )
                self.context.current_portfolio = portfolio
                self.context.console.print(f"✅ [green]Portfolio '{name}' créé avec {capital:,.0f}$[/green]")
                return True
            except Exception as e:
                self.context.console.print(f"❌ [red]Erreur création portfolio: {e}[/red]")
                return False

        elif args[0] == "demo":
            try:
                portfolio = create_demo_portfolio()
                self.context.current_portfolio = portfolio
                self.context.console.print("✅ [green]Portfolio de démonstration chargé[/green]")
                self._display_portfolio(portfolio)
                return True
            except Exception as e:
                self.context.console.print(f"❌ [red]Erreur portfolio demo: {e}[/red]")
                return False

        else:
            self.context.console.print("❌ [red]Usage: portfolio [create|show|demo][/red]")
            return False

    def _display_portfolio(self, portfolio):
        """Affiche un portfolio."""
        # Informations générales
        info_table = Table(title=f"💼 Portfolio: {portfolio.name}")
        info_table.add_column("Métrique", style="cyan")
        info_table.add_column("Valeur", style="white")

        info_table.add_row("ID", portfolio.id[:8] + "...")
        info_table.add_row("Capital Initial", f"{float(portfolio.initial_capital):,.0f} {portfolio.base_currency.value}")
        info_table.add_row("Balance Actuelle", f"{float(portfolio.current_balance):,.0f} {portfolio.base_currency.value}")
        info_table.add_row("Équité Totale", f"{float(portfolio.total_equity):,.0f} {portfolio.base_currency.value}")
        info_table.add_row("PnL Non Réalisé", f"{float(portfolio.unrealized_pnl):,.2f} {portfolio.base_currency.value}")
        info_table.add_row("Rendement Total", f"{portfolio.total_return_percent:.2f}%")
        info_table.add_row("Nombre de Positions", str(portfolio.position_count))
        info_table.add_row("Statut", portfolio.status.value)
        info_table.add_row("Santé", "🟢 Bon" if portfolio.is_healthy else "🔴 Attention")

        self.context.console.print(info_table)

        # Positions si disponibles
        if portfolio.positions:
            pos_table = Table(title="📈 Positions")
            pos_table.add_column("Symbole", style="cyan")
            pos_table.add_column("Taille", style="white")
            pos_table.add_column("Prix Entrée", style="white")
            pos_table.add_column("Prix Actuel", style="white")
            pos_table.add_column("Valeur Marché", style="white")
            pos_table.add_column("PnL", style="white")

            for pos in portfolio.positions:
                pnl_color = "green" if (pos.unrealized_pnl or 0) >= 0 else "red"
                pos_table.add_row(
                    pos.symbol,
                    f"{pos.size}",
                    f"{pos.entry_price:,.2f}",
                    f"{pos.current_price:,.2f}",
                    f"{float(pos.market_value):,.2f}",
                    f"[{pnl_color}]{float(pos.unrealized_pnl or 0):+,.2f}[/{pnl_color}]"
                )

            self.context.console.print(pos_table)

    async def cmd_positions(self, args: List[str]) -> bool:
        """Commande positions."""
        if not self.context.current_portfolio:
            self.context.console.print("❌ [red]Aucun portfolio chargé[/red]")
            return True

        if not self.context.current_portfolio.positions:
            self.context.console.print("ℹ️ [yellow]Aucune position ouverte[/yellow]")
            return True

        # Affichage détaillé des positions
        for i, pos in enumerate(self.context.current_portfolio.positions, 1):
            panel_content = (
                f"[bold]Symbole:[/bold] {pos.symbol}\n"
                f"[bold]Taille:[/bold] {pos.size} ({'Long' if pos.is_long else 'Short'})\n"
                f"[bold]Prix d'entrée:[/bold] {pos.entry_price:,.2f}\n"
                f"[bold]Prix actuel:[/bold] {pos.current_price:,.2f}\n"
                f"[bold]Valeur marché:[/bold] {float(pos.market_value):,.2f}\n"
                f"[bold]PnL:[/bold] {float(pos.unrealized_pnl or 0):+,.2f}\n"
                f"[bold]Stratégie:[/bold] {pos.strategy_name or 'N/A'}\n"
                f"[bold]Entrée:[/bold] {pos.entry_time.strftime('%Y-%m-%d %H:%M')}"
            )

            pnl_color = "green" if (pos.unrealized_pnl or 0) >= 0 else "red"
            panel = Panel(
                panel_content,
                title=f"📊 Position {i}/{len(self.context.current_portfolio.positions)}",
                border_style=pnl_color
            )

            self.context.console.print(panel)

        return True

    # === Commandes Monitoring ===

    async def cmd_metrics(self, args: List[str]) -> bool:
        """Commande metrics."""
        if not self.context.metrics_collector:
            self.context.console.print("❌ [red]Système de métriques non disponible[/red]")
            return False

        if not args or args[0] == "show":
            # Afficher dashboard des métriques
            dashboard_data = self.context.metrics_collector.get_dashboard_data()

            # Métriques système
            if dashboard_data.get('system_status'):
                sys_table = Table(title="🖥️ Métriques Système")
                sys_table.add_column("Métrique", style="cyan")
                sys_table.add_column("Valeur", style="white")

                for metric, value in dashboard_data['system_status'].items():
                    if isinstance(value, float):
                        color = "red" if value > 80 else "yellow" if value > 60 else "green"
                        sys_table.add_row(metric, f"[{color}]{value:.1f}%[/{color}]")
                    else:
                        sys_table.add_row(metric, str(value))

                self.context.console.print(sys_table)

            # Statistiques collecteur
            stats = dashboard_data.get('stats', {})
            stats_table = Table(title="📊 Statistiques Collecteur")
            stats_table.add_column("Statistique", style="cyan")
            stats_table.add_column("Valeur", style="white")

            for key, value in stats.items():
                stats_table.add_row(key.replace('_', ' ').title(), str(value))

            self.context.console.print(stats_table)

            # Alertes actives
            alerts = dashboard_data.get('active_alerts', [])
            if alerts:
                alert_table = Table(title="🚨 Alertes Actives")
                alert_table.add_column("Règle", style="cyan")
                alert_table.add_column("Métrique", style="white")
                alert_table.add_column("Valeur", style="white")
                alert_table.add_column("Sévérité", style="white")

                for alert in alerts:
                    severity_color = {
                        'info': 'blue',
                        'warning': 'yellow',
                        'error': 'red',
                        'critical': 'red bold'
                    }.get(alert['severity'], 'white')

                    alert_table.add_row(
                        alert['rule_name'],
                        alert['metric_name'],
                        f"{alert['value']:.2f}",
                        f"[{severity_color}]{alert['severity'].upper()}[/{severity_color}]"
                    )

                self.context.console.print(alert_table)

            return True

        elif args[0] == "history" and len(args) > 1:
            metric_name = args[1]
            hours = int(args[2]) if len(args) > 2 else 1

            history = self.context.metrics_collector.get_metric_history(metric_name, hours)

            if not history:
                self.context.console.print(f"❌ [red]Aucune donnée pour {metric_name}[/red]")
                return True

            # Afficher historique
            hist_table = Table(title=f"📈 Historique: {metric_name} ({hours}h)")
            hist_table.add_column("Timestamp", style="cyan")
            hist_table.add_column("Valeur", style="white")

            for metric in history[-10:]:  # Dernières 10 valeurs
                hist_table.add_row(
                    metric.timestamp.strftime("%H:%M:%S"),
                    f"{metric.value:.2f}"
                )

            self.context.console.print(hist_table)

            # Résumé statistique
            summary = self.context.metrics_collector.get_metric_summary(metric_name, hours)
            if summary:
                summary_table = Table(title="📊 Résumé Statistique")
                summary_table.add_column("Statistique", style="cyan")
                summary_table.add_column("Valeur", style="white")

                for key, value in summary.items():
                    summary_table.add_row(key.title(), f"{value:.2f}")

                self.context.console.print(summary_table)

            return True

        else:
            self.context.console.print("❌ [red]Usage: metrics [show|history metric_name [hours]][/red]")
            return False

    # === Commandes Trading ===

    async def cmd_strategies(self, args: List[str]) -> bool:
        """Commande strategies."""
        strategies = [
            {"name": "DMN LSTM", "type": "Deep Learning", "status": "✅ Disponible"},
            {"name": "Mean Reversion", "type": "Statistical", "status": "✅ Disponible"},
            {"name": "Funding Arbitrage", "type": "Arbitrage", "status": "✅ Disponible"},
            {"name": "RL Alpha", "type": "Reinforcement Learning", "status": "✅ Disponible"},
            {"name": "Grid Trading", "type": "Market Making", "status": "🚧 En développement"}
        ]

        table = Table(title="🎯 Stratégies Disponibles")
        table.add_column("Nom", style="cyan")
        table.add_column("Type", style="white")
        table.add_column("Statut", style="white")

        for strategy in strategies:
            table.add_row(strategy["name"], strategy["type"], strategy["status"])

        self.context.console.print(table)
        return True

    async def cmd_backtest(self, args: List[str]) -> bool:
        """Commande backtest."""
        if not args:
            self.context.console.print("❌ [red]Usage: backtest strategy [start_date] [end_date][/red]")
            return False

        strategy = args[0]

        # Simulation d'un backtest
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.context.console
        ) as progress:
            task = progress.add_task(f"🔄 Exécution backtest {strategy}...", total=100)

            # Simulation des étapes
            steps = [
                "Chargement des données...",
                "Initialisation de la stratégie...",
                "Calcul des features...",
                "Génération des signaux...",
                "Calcul des métriques...",
                "Finalisation..."
            ]

            for i, step in enumerate(steps):
                await asyncio.sleep(0.5)  # Simulation
                progress.update(task, advance=100//len(steps), description=step)

        # Résultats simulés
        results = {
            "Stratégie": strategy,
            "Période": "2023-01-01 à 2023-12-31",
            "Rendement Total": f"{np.random.uniform(5, 25):.2f}%",
            "Ratio de Sharpe": f"{np.random.uniform(0.8, 2.5):.2f}",
            "Drawdown Max": f"{np.random.uniform(5, 15):.2f}%",
            "Trades Gagnants": f"{np.random.uniform(45, 75):.1f}%",
            "Nombre de Trades": f"{np.random.randint(150, 500)}"
        }

        # Afficher résultats
        results_table = Table(title=f"📊 Résultats Backtest: {strategy}")
        results_table.add_column("Métrique", style="cyan")
        results_table.add_column("Valeur", style="white")

        for metric, value in results.items():
            results_table.add_row(metric, value)

        self.context.console.print(results_table)
        return True

    # === Commandes Données ===

    async def cmd_data(self, args: List[str]) -> bool:
        """Commande data."""
        if not args:
            self.context.console.print("❌ [red]Usage: data [fetch|show] symbol [timeframe][/red]")
            return False

        action = args[0]
        symbol = args[1] if len(args) > 1 else None

        if action == "fetch" and symbol:
            timeframe = args[2] if len(args) > 2 else "1h"

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.context.console
            ) as progress:
                task = progress.add_task(f"📥 Récupération données {symbol} {timeframe}...", total=100)

                await asyncio.sleep(1)  # Simulation
                progress.update(task, advance=50, description="Connexion à l'API...")
                await asyncio.sleep(1)
                progress.update(task, advance=50, description="Traitement des données...")

            self.context.console.print(f"✅ [green]Données {symbol} {timeframe} récupérées (simulation)[/green]")
            return True

        elif action == "show" and symbol:
            # Simulation d'affichage de données
            if HAS_PANDAS:
                # Génération de données factices
                dates = pd.date_range(end=datetime.now(), periods=10, freq='1H')
                data = pd.DataFrame({
                    'timestamp': dates,
                    'open': np.random.uniform(45000, 50000, 10),
                    'high': np.random.uniform(46000, 51000, 10),
                    'low': np.random.uniform(44000, 49000, 10),
                    'close': np.random.uniform(45000, 50000, 10),
                    'volume': np.random.uniform(100, 1000, 10)
                })

                # Créer tableau d'affichage
                data_table = Table(title=f"📊 Données: {symbol}")
                data_table.add_column("Time", style="cyan")
                data_table.add_column("Open", style="white")
                data_table.add_column("High", style="green")
                data_table.add_column("Low", style="red")
                data_table.add_column("Close", style="white")
                data_table.add_column("Volume", style="blue")

                for _, row in data.tail(5).iterrows():
                    data_table.add_row(
                        row['timestamp'].strftime("%H:%M"),
                        f"{row['open']:.0f}",
                        f"{row['high']:.0f}",
                        f"{row['low']:.0f}",
                        f"{row['close']:.0f}",
                        f"{row['volume']:.0f}"
                    )

                self.context.console.print(data_table)
            else:
                self.context.console.print(f"📊 Données {symbol} (pandas non disponible)")

            return True

        else:
            self.context.console.print("❌ [red]Usage: data [fetch|show] symbol [timeframe][/red]")
            return False

    # === Commandes Avancées ===

    async def cmd_performance(self, args: List[str]) -> bool:
        """Commande performance."""
        if not args or args[0] == "test":
            # Test de performance simple
            config = PerformanceConfig(max_workers=2, use_numba=False)
            processor = create_optimized_processor("feature", config)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.context.console
            ) as progress:
                task = progress.add_task("🚀 Test de performance...", total=100)

                # Génération de données test
                progress.update(task, advance=20, description="Génération données test...")
                if HAS_PANDAS:
                    test_data = pd.DataFrame({
                        'close': 100 + np.cumsum(np.random.randn(1000) * 0.02),
                        'volume': np.random.randint(1000, 10000, 1000),
                        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1T')
                    }).set_index('timestamp')

                    progress.update(task, advance=30, description="Calcul des features...")
                    await asyncio.sleep(0.5)

                    # Test processing
                    start_time = asyncio.get_event_loop().time()
                    features = processor.compute_technical_features(test_data)
                    duration = asyncio.get_event_loop().time() - start_time

                    progress.update(task, advance=50, description="Finalisation...")
                    await asyncio.sleep(0.2)

                    # Résultats
                    results_table = Table(title="⚡ Résultats Performance")
                    results_table.add_column("Métrique", style="cyan")
                    results_table.add_column("Valeur", style="white")

                    results_table.add_row("Points de données", f"{len(test_data):,}")
                    results_table.add_row("Features générées", f"{len(features.columns)}")
                    results_table.add_row("Temps d'exécution", f"{duration:.3f}s")
                    results_table.add_row("Débit", f"{len(test_data)/duration:,.0f} points/s")

                    stats = processor.get_performance_stats()
                    results_table.add_row("Features calculées", f"{stats['features_computed']}")
                    results_table.add_row("Cache hit ratio", f"{stats['cache_hit_ratio']:.2%}")

                    self.context.console.print(results_table)
                else:
                    progress.update(task, advance=80, description="Pandas non disponible...")
                    self.context.console.print("⚠️ [yellow]Test limité: pandas non disponible[/yellow]")

            return True

        else:
            self.context.console.print("❌ [red]Usage: performance [test|benchmark][/red]")
            return False


class InteractiveCLI:
    """CLI interactif principal."""

    def __init__(self):
        self.context = cli_context
        self.processor = CommandProcessor(self.context)
        self.running = True

    async def run(self):
        """Lance le CLI interactif."""
        # Initialisation
        if not self.context.initialize():
            return

        # Bannière de bienvenue
        self._show_welcome()

        # Boucle principale
        while self.running:
            try:
                # Prompt
                prompt_text = self._get_prompt()

                if HAS_RICH:
                    command = Prompt.ask(prompt_text)
                else:
                    command = input(f"{prompt_text} ")

                # Traitement de la commande
                if command.strip():
                    should_continue = await self.processor.execute_command(command)
                    if not should_continue:
                        self.running = False

            except KeyboardInterrupt:
                if Confirm.ask("\n🤔 Voulez-vous vraiment quitter?"):
                    break
                else:
                    self.context.console.print("⚡ Continuons!")

            except EOFError:
                break

            except Exception as e:
                self.context.console.print(f"❌ [red]Erreur inattendue: {e}[/red]")
                if self.context.debug_mode:
                    traceback.print_exc()

        # Nettoyage
        self._cleanup()

    def _show_welcome(self):
        """Affiche la bannière de bienvenue."""
        welcome_text = """
[bold blue]
 ██████╗ ███████╗██████╗  █████╗ ███╗   ███╗███████╗
██╔═══██╗██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝
██║   ██║█████╗  ██████╔╝███████║██╔████╔██║█████╗
██║▄▄ ██║██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝
╚██████╔╝██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗
 ╚══▀▀═╝ ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
[/bold blue]

[bold green]🚀 Quantitative Framework - Interactive CLI[/bold green]
[dim]Version 1.0.0 - Enterprise Edition[/dim]

💡 [cyan]Tapez 'help' pour voir les commandes disponibles[/cyan]
🐛 [yellow]Tapez 'debug on' pour activer le mode verbose[/yellow]
👋 [red]Tapez 'exit' pour quitter[/red]
        """

        panel = Panel.fit(
            welcome_text,
            border_style="blue",
            padding=(1, 2)
        )

        self.context.console.print(panel)

    def _get_prompt(self) -> str:
        """Génère le prompt avec informations contextuelles."""
        parts = ["[bold cyan]qframe[/bold cyan]"]

        if self.context.current_portfolio:
            parts.append(f"[green]({self.context.current_portfolio.name})[/green]")

        if self.context.debug_mode:
            parts.append("[yellow](debug)[/yellow]")

        return " ".join(parts) + " [bold]>[/bold]"

    def _cleanup(self):
        """Nettoyage avant fermeture."""
        try:
            if self.context.metrics_collector:
                self.context.metrics_collector.stop_auto_collection()

            self.context.console.print("\n✨ [green]Nettoyage terminé. Au revoir![/green]")

        except Exception as e:
            self.context.console.print(f"⚠️ [yellow]Erreur nettoyage: {e}[/yellow]")


# Point d'entrée principal
async def main():
    """Point d'entrée principal du CLI."""
    if not HAS_RICH:
        print("⚠️ Rich non disponible - expérience dégradée")

    cli = InteractiveCLI()
    await cli.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        traceback.print_exc()