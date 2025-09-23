"""
Advanced Visualization Pipeline for QFrame
==========================================

Implémente des pipelines de visualisation sophistiquées selon les standards Claude Flow,
avec génération automatique de dashboards et graphiques financiers interactifs.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings

# Imports de visualisation
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Basic matplotlib will be used.")

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-darkgrid')
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration pour les visualisations."""
    theme: str = "plotly_dark"
    figure_size: Tuple[int, int] = (1200, 800)
    dpi: int = 300
    interactive: bool = True
    save_format: str = "html"
    color_scheme: str = "financial"

class AdvancedFinancialVisualizer:
    """
    Visualiseur avancé pour données financières.

    Génère automatiquement des graphiques sophistiqués pour
    analyse quantitative et reporting de performance.
    """

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.color_palette = self._get_financial_colors()

    def create_strategy_dashboard(
        self,
        strategy_data: Dict[str, Any],
        market_data: pd.DataFrame,
        signals: pd.DataFrame = None,
        output_path: str = None
    ) -> str:
        """
        Crée un dashboard complet pour une stratégie.

        Args:
            strategy_data: Données de performance de la stratégie
            market_data: Données OHLCV
            signals: Signaux de trading (optionnel)
            output_path: Chemin de sauvegarde

        Returns:
            HTML du dashboard
        """
        logger.info("Génération du dashboard de stratégie")

        if not PLOTLY_AVAILABLE:
            return self._create_basic_dashboard(strategy_data, market_data)

        # Créer la grille de subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Performance Cumulative', 'Drawdown',
                'Prix et Signaux', 'Distribution des Returns',
                'Rolling Sharpe Ratio', 'Volume Analysis',
                'Risk Metrics', 'Monthly Returns Heatmap'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"colspan": 2}, None],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08
        )

        # 1. Performance cumulative
        self._add_performance_chart(fig, strategy_data, row=1, col=1)

        # 2. Drawdown
        self._add_drawdown_chart(fig, strategy_data, row=1, col=2)

        # 3. Prix et signaux
        self._add_price_signals_chart(fig, market_data, signals, row=2, col=1)

        # 4. Distribution des returns
        self._add_returns_distribution(fig, strategy_data, row=3, col=1)

        # 5. Rolling Sharpe
        self._add_rolling_sharpe(fig, strategy_data, row=3, col=2)

        # 6. Volume analysis
        self._add_volume_analysis(fig, market_data, row=4, col=1)

        # 7. Risk metrics
        self._add_risk_metrics(fig, strategy_data, row=4, col=2)

        # Mise en forme
        fig.update_layout(
            height=1600,
            showlegend=True,
            title_text="Dashboard de Stratégie Quantitative",
            title_x=0.5,
            template=self.config.theme
        )

        # Sauvegarde
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Dashboard sauvegardé: {output_path}")

        return fig.to_html()

    def create_risk_analysis_report(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> str:
        """
        Génère un rapport d'analyse de risque complet.

        Args:
            returns: Série des returns de la stratégie
            benchmark_returns: Returns du benchmark (optionnel)
            confidence_levels: Niveaux de confiance pour VaR

        Returns:
            HTML du rapport de risque
        """
        logger.info("Génération du rapport d'analyse de risque")

        if not PLOTLY_AVAILABLE:
            return self._create_basic_risk_report(returns)

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Value at Risk', 'Rolling Volatility',
                'QQ Plot', 'Autocorrelation',
                'Risk Decomposition', 'Stress Testing'
            ]
        )

        # 1. VaR Analysis
        self._add_var_analysis(fig, returns, confidence_levels, row=1, col=1)

        # 2. Rolling Volatility
        self._add_rolling_volatility(fig, returns, row=1, col=2)

        # 3. QQ Plot
        self._add_qq_plot(fig, returns, row=2, col=1)

        # 4. Autocorrelation
        self._add_autocorrelation(fig, returns, row=2, col=2)

        # 5. Risk Decomposition
        self._add_risk_decomposition(fig, returns, row=3, col=1)

        # 6. Stress Testing
        self._add_stress_testing(fig, returns, row=3, col=2)

        fig.update_layout(
            height=1200,
            title_text="Analyse de Risque Quantitative",
            title_x=0.5,
            template=self.config.theme
        )

        return fig.to_html()

    def create_performance_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_names: List[str]
    ) -> str:
        """
        Crée une analyse d'attribution de performance.

        Args:
            strategy_returns: Returns de la stratégie
            factor_returns: Returns des facteurs
            factor_names: Noms des facteurs

        Returns:
            HTML de l'analyse d'attribution
        """
        logger.info("Génération de l'analyse d'attribution de performance")

        # Régression factorielle simple
        attribution_results = self._calculate_factor_attribution(
            strategy_returns, factor_returns, factor_names
        )

        if not PLOTLY_AVAILABLE:
            return self._create_basic_attribution_report(attribution_results)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Factor Exposures', 'Attribution Breakdown',
                'Rolling Betas', 'Active Return'
            ]
        )

        # 1. Factor Exposures
        self._add_factor_exposures(fig, attribution_results, row=1, col=1)

        # 2. Attribution Breakdown
        self._add_attribution_breakdown(fig, attribution_results, row=1, col=2)

        # 3. Rolling Betas
        self._add_rolling_betas(fig, strategy_returns, factor_returns, row=2, col=1)

        # 4. Active Return
        self._add_active_return(fig, attribution_results, row=2, col=2)

        fig.update_layout(
            height=800,
            title_text="Attribution de Performance",
            title_x=0.5,
            template=self.config.theme
        )

        return fig.to_html()

    def create_regime_analysis(
        self,
        data: pd.DataFrame,
        regime_labels: pd.Series,
        strategy_returns: pd.Series
    ) -> str:
        """
        Analyse de performance par régime de marché.

        Args:
            data: Données de marché
            regime_labels: Labels des régimes
            strategy_returns: Returns de la stratégie

        Returns:
            HTML de l'analyse de régimes
        """
        logger.info("Génération de l'analyse par régime")

        if not PLOTLY_AVAILABLE:
            return self._create_basic_regime_report(regime_labels, strategy_returns)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Régimes Temporels', 'Performance par Régime',
                'Volatilité par Régime', 'Sharpe par Régime'
            ]
        )

        # 1. Régimes temporels
        self._add_regime_timeline(fig, regime_labels, row=1, col=1)

        # 2. Performance par régime
        self._add_regime_performance(fig, regime_labels, strategy_returns, row=1, col=2)

        # 3. Volatilité par régime
        self._add_regime_volatility(fig, regime_labels, strategy_returns, row=2, col=1)

        # 4. Sharpe par régime
        self._add_regime_sharpe(fig, regime_labels, strategy_returns, row=2, col=2)

        fig.update_layout(
            height=800,
            title_text="Analyse par Régime de Marché",
            title_x=0.5,
            template=self.config.theme
        )

        return fig.to_html()

    # Méthodes pour ajouter des graphiques spécifiques
    def _add_performance_chart(self, fig, strategy_data: Dict, row: int, col: int):
        """Ajoute le graphique de performance cumulative."""
        if 'cumulative_returns' in strategy_data:
            returns = strategy_data['cumulative_returns']
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns.values,
                    mode='lines',
                    name='Performance Cumulative',
                    line=dict(color=self.color_palette['primary'])
                ),
                row=row, col=col
            )

    def _add_drawdown_chart(self, fig, strategy_data: Dict, row: int, col: int):
        """Ajoute le graphique de drawdown."""
        if 'drawdown' in strategy_data:
            drawdown = strategy_data['drawdown']
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    fill='tonexty',
                    name='Drawdown',
                    line=dict(color=self.color_palette['danger'])
                ),
                row=row, col=col
            )

    def _add_price_signals_chart(self, fig, market_data: pd.DataFrame, signals: pd.DataFrame, row: int, col: int):
        """Ajoute le graphique prix + signaux."""
        # Prix
        fig.add_trace(
            go.Scatter(
                x=market_data.index,
                y=market_data['close'],
                mode='lines',
                name='Prix',
                line=dict(color=self.color_palette['neutral'])
            ),
            row=row, col=col
        )

        # Signaux si disponibles
        if signals is not None and not signals.empty:
            buy_signals = signals[signals['action'] == 'buy']
            sell_signals = signals[signals['action'] == 'sell']

            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['price'],
                        mode='markers',
                        name='Signaux Achat',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color=self.color_palette['success']
                        )
                    ),
                    row=row, col=col
                )

            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['price'],
                        mode='markers',
                        name='Signaux Vente',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color=self.color_palette['danger']
                        )
                    ),
                    row=row, col=col
                )

    def _add_returns_distribution(self, fig, strategy_data: Dict, row: int, col: int):
        """Ajoute l'histogramme de distribution des returns."""
        if 'returns' in strategy_data:
            returns = strategy_data['returns'].dropna()
            fig.add_trace(
                go.Histogram(
                    x=returns.values,
                    nbinsx=50,
                    name='Distribution Returns',
                    marker=dict(color=self.color_palette['primary'])
                ),
                row=row, col=col
            )

    def _add_rolling_sharpe(self, fig, strategy_data: Dict, row: int, col: int):
        """Ajoute le graphique du Sharpe ratio mobile."""
        if 'rolling_sharpe' in strategy_data:
            sharpe = strategy_data['rolling_sharpe']
            fig.add_trace(
                go.Scatter(
                    x=sharpe.index,
                    y=sharpe.values,
                    mode='lines',
                    name='Sharpe Mobile',
                    line=dict(color=self.color_palette['info'])
                ),
                row=row, col=col
            )

            # Ligne de référence à 1.0
            fig.add_hline(
                y=1.0,
                line_dash="dash",
                line_color=self.color_palette['warning'],
                row=row, col=col
            )

    def _add_volume_analysis(self, fig, market_data: pd.DataFrame, row: int, col: int):
        """Ajoute l'analyse de volume."""
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            fig.add_trace(
                go.Bar(
                    x=volume.index,
                    y=volume.values,
                    name='Volume',
                    marker=dict(color=self.color_palette['secondary'])
                ),
                row=row, col=col
            )

    def _add_risk_metrics(self, fig, strategy_data: Dict, row: int, col: int):
        """Ajoute les métriques de risque."""
        risk_metrics = strategy_data.get('risk_metrics', {})

        if risk_metrics:
            metrics_names = list(risk_metrics.keys())
            metrics_values = list(risk_metrics.values())

            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=metrics_values,
                    name='Risk Metrics',
                    marker=dict(color=self.color_palette['warning'])
                ),
                row=row, col=col
            )

    def _add_var_analysis(self, fig, returns: pd.Series, confidence_levels: List[float], row: int, col: int):
        """Ajoute l'analyse VaR."""
        var_values = [returns.quantile(1 - cl) for cl in confidence_levels]
        cl_labels = [f"VaR {cl:.0%}" for cl in confidence_levels]

        fig.add_trace(
            go.Bar(
                x=cl_labels,
                y=var_values,
                name='Value at Risk',
                marker=dict(color=self.color_palette['danger'])
            ),
            row=row, col=col
        )

    def _add_rolling_volatility(self, fig, returns: pd.Series, row: int, col: int):
        """Ajoute la volatilité mobile."""
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)  # Annualisée

        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatilité Mobile (30j)',
                line=dict(color=self.color_palette['warning'])
            ),
            row=row, col=col
        )

    def _calculate_factor_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_names: List[str]
    ) -> Dict[str, Any]:
        """Calcule l'attribution factorielle."""
        from sklearn.linear_model import LinearRegression

        # Aligner les données
        aligned_data = pd.concat([strategy_returns, factor_returns], axis=1, join='inner')
        y = aligned_data.iloc[:, 0].values.reshape(-1, 1)
        X = aligned_data.iloc[:, 1:].values

        # Régression
        model = LinearRegression()
        model.fit(X, y.ravel())

        # Résultats
        betas = model.coef_
        alpha = model.intercept_
        r_squared = model.score(X, y.ravel())

        return {
            'betas': dict(zip(factor_names, betas)),
            'alpha': alpha,
            'r_squared': r_squared,
            'factor_names': factor_names
        }

    def _get_financial_colors(self) -> Dict[str, str]:
        """Palette de couleurs pour graphiques financiers."""
        return {
            'primary': '#1f77b4',      # Bleu
            'secondary': '#ff7f0e',    # Orange
            'success': '#2ca02c',      # Vert
            'danger': '#d62728',       # Rouge
            'warning': '#ff7f0e',      # Jaune/Orange
            'info': '#17a2b8',         # Cyan
            'neutral': '#6c757d'       # Gris
        }

    def _create_basic_dashboard(self, strategy_data: Dict, market_data: pd.DataFrame) -> str:
        """Fallback dashboard basique sans Plotly."""
        return "<html><body><h1>Dashboard basique</h1><p>Plotly non disponible</p></body></html>"

    def _create_basic_risk_report(self, returns: pd.Series) -> str:
        """Fallback rapport de risque basique."""
        var_95 = returns.quantile(0.05)
        return f"<html><body><h1>Rapport de Risque</h1><p>VaR 95%: {var_95:.2%}</p></body></html>"

    def _create_basic_attribution_report(self, attribution_results: Dict) -> str:
        """Fallback attribution basique."""
        return "<html><body><h1>Attribution basique</h1></body></html>"

    def _create_basic_regime_report(self, regime_labels: pd.Series, strategy_returns: pd.Series) -> str:
        """Fallback analyse de régimes basique."""
        return "<html><body><h1>Analyse de régimes basique</h1></body></html>"

    # Méthodes pour les autres graphiques (stubs pour l'exemple)
    def _add_qq_plot(self, fig, returns: pd.Series, row: int, col: int):
        """Ajoute un QQ plot."""
        pass

    def _add_autocorrelation(self, fig, returns: pd.Series, row: int, col: int):
        """Ajoute l'autocorrélation."""
        pass

    def _add_risk_decomposition(self, fig, returns: pd.Series, row: int, col: int):
        """Ajoute la décomposition de risque."""
        pass

    def _add_stress_testing(self, fig, returns: pd.Series, row: int, col: int):
        """Ajoute les résultats de stress testing."""
        pass

    def _add_factor_exposures(self, fig, attribution_results: Dict, row: int, col: int):
        """Ajoute les expositions factorielles."""
        pass

    def _add_attribution_breakdown(self, fig, attribution_results: Dict, row: int, col: int):
        """Ajoute la décomposition d'attribution."""
        pass

    def _add_rolling_betas(self, fig, strategy_returns: pd.Series, factor_returns: pd.DataFrame, row: int, col: int):
        """Ajoute les betas mobiles."""
        pass

    def _add_active_return(self, fig, attribution_results: Dict, row: int, col: int):
        """Ajoute l'active return."""
        pass

    def _add_regime_timeline(self, fig, regime_labels: pd.Series, row: int, col: int):
        """Ajoute la timeline des régimes."""
        pass

    def _add_regime_performance(self, fig, regime_labels: pd.Series, strategy_returns: pd.Series, row: int, col: int):
        """Ajoute la performance par régime."""
        pass

    def _add_regime_volatility(self, fig, regime_labels: pd.Series, strategy_returns: pd.Series, row: int, col: int):
        """Ajoute la volatilité par régime."""
        pass

    def _add_regime_sharpe(self, fig, regime_labels: pd.Series, strategy_returns: pd.Series, row: int, col: int):
        """Ajoute le Sharpe par régime."""
        pass

# Factory function
def create_visualizer(theme: str = "plotly_dark") -> AdvancedFinancialVisualizer:
    """Factory pour créer un visualiseur."""
    config = VisualizationConfig(theme=theme)
    return AdvancedFinancialVisualizer(config)

# Exemple d'usage
if __name__ == "__main__":
    # Données d'exemple
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    cumulative_returns = (1 + returns).cumprod()

    strategy_data = {
        'returns': returns,
        'cumulative_returns': cumulative_returns,
        'drawdown': (cumulative_returns / cumulative_returns.expanding().max() - 1),
        'rolling_sharpe': returns.rolling(30).mean() / returns.rolling(30).std() * np.sqrt(252),
        'risk_metrics': {
            'VaR_95': returns.quantile(0.05),
            'Max_DD': -0.12,
            'Volatility': returns.std() * np.sqrt(252)
        }
    }

    market_data = pd.DataFrame({
        'close': cumulative_returns * 100,
        'volume': np.random.randint(1000, 10000, 252)
    }, index=dates)

    # Créer le visualiseur
    visualizer = create_visualizer()

    # Générer le dashboard
    dashboard = visualizer.create_strategy_dashboard(strategy_data, market_data)
    print("Dashboard généré avec succès!")

    # Générer le rapport de risque
    risk_report = visualizer.create_risk_analysis_report(returns)
    print("Rapport de risque généré!")