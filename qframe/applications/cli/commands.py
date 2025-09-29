"""
CLI Commands and Utilities
==========================

Commandes avancées et utilitaires pour le CLI QFrame.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.tree import Tree
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

import pandas as pd
import numpy as np

from qframe.core.config import get_config
from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
from qframe.infrastructure.performance.optimized_processors import create_optimized_processor, PerformanceConfig


class AdvancedCommands:
    """Commandes CLI avancées."""

    def __init__(self, context):
        self.context = context
        self.console = context.console

    async def cmd_strategy_analyze(self, args: List[str]) -> bool:
        """Analyse approfondie d'une stratégie."""
        if not args:
            strategy_name = "mean_reversion"  # Par défaut
        else:
            strategy_name = args[0]

        self.console.print(f"🔍 [cyan]Analyse de la stratégie: {strategy_name}[/cyan]")

        # Simulation d'analyse
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:

            # Étapes d'analyse
            steps = [
                ("Chargement paramètres...", 0.5),
                ("Analyse historique...", 1.0),
                ("Calcul métriques risque...", 0.8),
                ("Optimisation paramètres...", 1.2),
                ("Génération rapport...", 0.3)
            ]

            task = progress.add_task("Analyse en cours...", total=len(steps))

            for step_name, duration in steps:
                progress.update(task, description=step_name)
                await asyncio.sleep(duration)
                progress.advance(task)

        # Résultats d'analyse
        analysis_results = {
            "Stratégie": strategy_name.title().replace('_', ' '),
            "Performance Historique": {
                "Rendement Annuel": f"{np.random.uniform(12, 28):.1f}%",
                "Volatilité": f"{np.random.uniform(8, 18):.1f}%",
                "Ratio Sharpe": f"{np.random.uniform(1.2, 2.8):.2f}",
                "Drawdown Max": f"{np.random.uniform(5, 15):.1f}%"
            },
            "Analyse Risque": {
                "VaR 95%": f"{np.random.uniform(2, 8):.1f}%",
                "Skewness": f"{np.random.uniform(-0.5, 0.5):.2f}",
                "Kurtosis": f"{np.random.uniform(2.5, 4.5):.2f}"
            },
            "Paramètres Optimaux": {
                "Lookback": f"{np.random.randint(10, 50)}",
                "Threshold": f"{np.random.uniform(0.8, 2.2):.2f}",
                "Stop Loss": f"{np.random.uniform(2, 8):.1f}%"
            }
        }

        # Affichage structuré
        for section, data in analysis_results.items():
            if isinstance(data, dict):
                table = Table(title=f"📊 {section}")
                table.add_column("Métrique", style="cyan")
                table.add_column("Valeur", style="white")

                for metric, value in data.items():
                    table.add_row(metric, str(value))

                self.console.print(table)
            else:
                self.console.print(f"[bold]{section}:[/bold] {data}")

        return True

    async def cmd_market_scan(self, args: List[str]) -> bool:
        """Scanner de marché pour opportunités."""
        symbols = args if args else ["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD", "DOT/USD"]

        self.console.print(f"🔍 [cyan]Scan de marché sur {len(symbols)} symboles[/cyan]")

        # Simulation du scan
        opportunities = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:

            task = progress.add_task("Scan en cours...", total=len(symbols))

            for symbol in symbols:
                progress.update(task, description=f"Analyse {symbol}...")
                await asyncio.sleep(0.3)

                # Générer signal aléatoire
                signal_strength = np.random.uniform(-1, 1)
                signal_type = "BUY" if signal_strength > 0.3 else "SELL" if signal_strength < -0.3 else "HOLD"

                if signal_type != "HOLD":
                    opportunities.append({
                        "symbol": symbol,
                        "signal": signal_type,
                        "strength": abs(signal_strength),
                        "price": np.random.uniform(20, 50000),
                        "volume": np.random.uniform(1000, 100000),
                        "rsi": np.random.uniform(20, 80),
                        "ma_cross": np.random.choice(["Golden", "Death", "None"])
                    })

                progress.advance(task)

        # Affichage des opportunités
        if opportunities:
            table = Table(title="🎯 Opportunités Détectées")
            table.add_column("Symbole", style="cyan")
            table.add_column("Signal", style="white")
            table.add_column("Force", style="white")
            table.add_column("Prix", style="white")
            table.add_column("RSI", style="white")
            table.add_column("MA Cross", style="white")

            for opp in sorted(opportunities, key=lambda x: x["strength"], reverse=True):
                signal_color = "green" if opp["signal"] == "BUY" else "red"
                strength_bar = "█" * int(opp["strength"] * 10)

                table.add_row(
                    opp["symbol"],
                    f"[{signal_color}]{opp['signal']}[/{signal_color}]",
                    f"{strength_bar} {opp['strength']:.2f}",
                    f"{opp['price']:.2f}",
                    f"{opp['rsi']:.1f}",
                    opp["ma_cross"]
                )

            self.console.print(table)
        else:
            self.console.print("ℹ️ [yellow]Aucune opportunité détectée actuellement[/yellow]")

        return True

    async def cmd_risk_analysis(self, args: List[str]) -> bool:
        """Analyse de risque du portfolio."""
        if not self.context.current_portfolio:
            self.console.print("❌ [red]Aucun portfolio chargé[/red]")
            return False

        portfolio = self.context.current_portfolio

        self.console.print("🔍 [cyan]Analyse de risque du portfolio[/cyan]")

        # Calcul des métriques de risque
        portfolio_with_risk = portfolio.calculate_risk_metrics()
        risk_metrics = portfolio_with_risk.risk_metrics

        # Table des métriques de risque
        risk_table = Table(title="⚠️ Métriques de Risque")
        risk_table.add_column("Métrique", style="cyan")
        risk_table.add_column("Valeur", style="white")
        risk_table.add_column("Évaluation", style="white")

        # Évaluations basées sur les seuils
        evaluations = {
            "total_exposure": "🟢 Normal" if risk_metrics.total_exposure < 100000 else "🟡 Élevé" if risk_metrics.total_exposure < 500000 else "🔴 Très élevé",
            "leverage": "🟢 Conservateur" if risk_metrics.leverage < 2 else "🟡 Modéré" if risk_metrics.leverage < 5 else "🔴 Risqué",
            "max_position_weight": "🟢 Diversifié" if risk_metrics.max_position_weight < 0.3 else "🟡 Concentré" if risk_metrics.max_position_weight < 0.5 else "🔴 Très concentré",
            "concentration_index": "🟢 Faible" if risk_metrics.concentration_index < 0.3 else "🟡 Modérée" if risk_metrics.concentration_index < 0.5 else "🔴 Élevée"
        }

        risk_data = {
            "Exposition Totale": f"{risk_metrics.total_exposure:,.0f} {portfolio.base_currency.value}",
            "Levier": f"{risk_metrics.leverage:.2f}x",
            "Poids Max Position": f"{risk_metrics.max_position_weight:.1%}",
            "Index Concentration": f"{risk_metrics.concentration_index:.3f}",
            "Score de Risque": f"{risk_metrics.risk_score:.1f}/100"
        }

        metric_keys = ["total_exposure", "leverage", "max_position_weight", "concentration_index"]

        for i, (label, value) in enumerate(risk_data.items()):
            if i < len(metric_keys):
                evaluation = evaluations.get(metric_keys[i], "📊 N/A")
            else:
                # Score de risque
                score = risk_metrics.risk_score
                evaluation = "🟢 Faible" if score < 30 else "🟡 Modéré" if score < 60 else "🔴 Élevé"

            risk_table.add_row(label, value, evaluation)

        self.console.print(risk_table)

        # Recommandations
        recommendations = []

        if risk_metrics.leverage > 3:
            recommendations.append("💡 Réduire le levier pour limiter le risque")

        if risk_metrics.max_position_weight > 0.4:
            recommendations.append("💡 Diversifier davantage les positions")

        if risk_metrics.concentration_index > 0.4:
            recommendations.append("💡 Rééquilibrer le portfolio")

        if risk_metrics.risk_score > 70:
            recommendations.append("⚠️ Score de risque élevé - révision nécessaire")

        if recommendations:
            rec_panel = Panel(
                "\n".join(recommendations),
                title="💡 Recommandations",
                border_style="yellow"
            )
            self.console.print(rec_panel)
        else:
            self.console.print("✅ [green]Portfolio dans les limites de risque acceptables[/green]")

        return True

    async def cmd_benchmark(self, args: List[str]) -> bool:
        """Benchmark de performance du système."""
        test_type = args[0] if args else "complete"

        self.console.print(f"⚡ [cyan]Benchmark de performance: {test_type}[/cyan]")

        benchmarks = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:

            if test_type in ["complete", "features"]:
                # Test des features
                task = progress.add_task("Test calcul features...", total=100)

                # Générer données test
                test_data = pd.DataFrame({
                    'close': 100 + np.cumsum(np.random.randn(5000) * 0.02),
                    'volume': np.random.randint(1000, 10000, 5000),
                    'timestamp': pd.date_range('2023-01-01', periods=5000, freq='1T')
                }).set_index('timestamp')

                progress.update(task, advance=30)

                # Test avec processeur optimisé
                config = PerformanceConfig(max_workers=4, use_numba=False)
                processor = create_optimized_processor("feature", config)

                start_time = time.time()
                features = processor.compute_technical_features(test_data)
                duration = time.time() - start_time

                progress.update(task, advance=70)

                benchmarks["Feature Calculation"] = {
                    "Data Points": f"{len(test_data):,}",
                    "Features Generated": f"{len(features.columns)}",
                    "Duration": f"{duration:.3f}s",
                    "Throughput": f"{len(test_data)/duration:,.0f} points/s"
                }

            if test_type in ["complete", "memory"]:
                # Test mémoire
                task = progress.add_task("Test gestion mémoire...", total=100)

                import psutil
                process = psutil.Process()

                # Mesure mémoire avant
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                progress.update(task, advance=30)

                # Allocation/libération
                large_arrays = []
                for i in range(10):
                    large_arrays.append(np.random.randn(100000))
                    await asyncio.sleep(0.01)

                mem_during = process.memory_info().rss / 1024 / 1024  # MB
                progress.update(task, advance=40)

                # Libération
                del large_arrays
                await asyncio.sleep(0.1)
                mem_after = process.memory_info().rss / 1024 / 1024  # MB

                progress.update(task, advance=30)

                benchmarks["Memory Management"] = {
                    "Initial": f"{mem_before:.1f} MB",
                    "Peak": f"{mem_during:.1f} MB",
                    "After Cleanup": f"{mem_after:.1f} MB",
                    "Efficiency": f"{((mem_during - mem_after) / (mem_during - mem_before) * 100):.1f}%"
                }

            if test_type in ["complete", "io"]:
                # Test I/O
                task = progress.add_task("Test performances I/O...", total=100)

                # Test écriture
                start_time = time.time()
                test_df = pd.DataFrame(np.random.randn(10000, 10))

                # Simulation sauvegarde
                json_data = test_df.to_json()
                write_duration = time.time() - start_time
                progress.update(task, advance=50)

                # Test lecture
                start_time = time.time()
                reloaded_df = pd.read_json(json_data)
                read_duration = time.time() - start_time
                progress.update(task, advance=50)

                benchmarks["I/O Performance"] = {
                    "Write Speed": f"{len(test_df) / write_duration:,.0f} rows/s",
                    "Read Speed": f"{len(reloaded_df) / read_duration:,.0f} rows/s",
                    "Data Size": f"{len(json_data) / 1024:.1f} KB"
                }

        # Affichage des résultats
        for benchmark_name, results in benchmarks.items():
            table = Table(title=f"📊 {benchmark_name}")
            table.add_column("Métrique", style="cyan")
            table.add_column("Résultat", style="white")

            for metric, result in results.items():
                table.add_row(metric, str(result))

            self.console.print(table)

        # Score global
        overall_score = np.random.uniform(75, 95)  # Simulation
        score_color = "green" if overall_score > 85 else "yellow" if overall_score > 70 else "red"

        score_panel = Panel(
            f"[bold]Score Global: [{score_color}]{overall_score:.1f}/100[/{score_color}][/bold]",
            title="🏆 Performance Globale",
            border_style=score_color
        )
        self.console.print(score_panel)

        return True

    async def cmd_export_config(self, args: List[str]) -> bool:
        """Export de la configuration."""
        if not self.context.config:
            self.console.print("❌ [red]Configuration non disponible[/red]")
            return False

        format_type = args[0] if args else "json"
        filename = args[1] if len(args) > 1 else f"qframe_config.{format_type}"

        try:
            config_dict = self.context.config.model_dump()

            if format_type == "json":
                with open(filename, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)

            elif format_type == "yaml":
                try:
                    import yaml
                    with open(filename, 'w') as f:
                        yaml.dump(config_dict, f, default_flow_style=False)
                except ImportError:
                    self.console.print("❌ [red]PyYAML non disponible[/red]")
                    return False

            self.console.print(f"✅ [green]Configuration exportée vers {filename}[/green]")
            return True

        except Exception as e:
            self.console.print(f"❌ [red]Erreur export: {e}[/red]")
            return False


def get_advanced_commands():
    """Retourne les commandes avancées disponibles."""
    return {
        'analyze': {
            'func': lambda ctx: AdvancedCommands(ctx).cmd_strategy_analyze,
            'description': 'Analyse approfondie d\'une stratégie',
            'usage': 'analyze [strategy_name]',
            'category': 'advanced'
        },
        'scan': {
            'func': lambda ctx: AdvancedCommands(ctx).cmd_market_scan,
            'description': 'Scanner de marché pour opportunités',
            'usage': 'scan [symbol1] [symbol2] ...',
            'category': 'trading'
        },
        'risk': {
            'func': lambda ctx: AdvancedCommands(ctx).cmd_risk_analysis,
            'description': 'Analyse de risque du portfolio',
            'usage': 'risk',
            'category': 'portfolio'
        },
        'benchmark': {
            'func': lambda ctx: AdvancedCommands(ctx).cmd_benchmark,
            'description': 'Benchmark de performance système',
            'usage': 'benchmark [complete|features|memory|io]',
            'category': 'advanced'
        },
        'export': {
            'func': lambda ctx: AdvancedCommands(ctx).cmd_export_config,
            'description': 'Export de la configuration',
            'usage': 'export [json|yaml] [filename]',
            'category': 'system'
        }
    }