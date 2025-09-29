"""
⚠️ Risk Monitor Component
Surveillance risque temps réel - VaR, limites, alertes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy import stats
import math


class RiskMonitor:
    """Moniteur de risque avancé avec calculs VaR et stress testing."""

    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        """Initialise les données de session pour le monitoring des risques."""
        if 'risk_config' not in st.session_state:
            st.session_state.risk_config = {
                'max_portfolio_var': 5000.0,  # VaR maximum du portefeuille
                'max_position_size': 50000.0,  # Taille max d'une position
                'max_leverage': 5.0,  # Leverage maximum
                'max_correlation': 0.8,  # Corrélation maximum entre positions
                'max_drawdown': 10.0,  # Drawdown maximum (%)
                'position_limit_pct': 20.0,  # % maximum du portefeuille par position
                'daily_loss_limit': 2000.0,  # Perte journalière maximum
                'margin_call_threshold': 80.0,  # Seuil de margin call (%)
                'emergency_stop_enabled': True
            }

        if 'risk_metrics' not in st.session_state:
            st.session_state.risk_metrics = {}
            self._calculate_risk_metrics()

        if 'risk_alerts' not in st.session_state:
            st.session_state.risk_alerts = []

        if 'var_history' not in st.session_state:
            st.session_state.var_history = []
            self._generate_var_history()

    def _generate_var_history(self):
        """Génère l'historique VaR pour les graphiques."""
        base_time = datetime.now() - timedelta(days=30)
        history = []

        for i in range(30):  # 30 jours d'historique
            timestamp = base_time + timedelta(days=i)

            # Simulation VaR évoluant
            base_var = 2000 + 500 * math.sin(i * 0.2) + np.random.normal(0, 200)
            portfolio_var_95 = max(1000, base_var)
            portfolio_var_99 = portfolio_var_95 * 1.5

            # Simulation autres métriques
            portfolio_value = 100000 + np.random.normal(0, 5000)
            max_leverage = np.random.uniform(1.0, 3.0)
            portfolio_beta = np.random.uniform(0.8, 1.4)

            history.append({
                'timestamp': timestamp,
                'portfolio_var_95': portfolio_var_95,
                'portfolio_var_99': portfolio_var_99,
                'portfolio_value': portfolio_value,
                'max_leverage': max_leverage,
                'portfolio_beta': portfolio_beta,
                'drawdown': np.random.uniform(0, 8)
            })

        st.session_state.var_history = history

    def _calculate_risk_metrics(self):
        """Calcule les métriques de risque en temps réel."""
        # Récupération des positions
        positions = st.session_state.get('positions', [])

        if not positions:
            st.session_state.risk_metrics = {
                'portfolio_var_95': 0.0,
                'portfolio_var_99': 0.0,
                'max_individual_loss': 0.0,
                'correlation_risk': 0.0,
                'leverage_ratio': 0.0,
                'concentration_risk': 0.0,
                'liquidity_risk': 'LOW',
                'market_exposure': 0.0
            }
            return

        # Calculs des métriques
        total_exposure = sum([pos['size'] * pos['current_price'] for pos in positions])
        total_margin = sum([pos['margin'] for pos in positions])
        account_balance = 100000.0  # Simulation

        # VaR Portfolio (simulation Monte Carlo simplifiée)
        var_95, var_99 = self._calculate_portfolio_var(positions)

        # Leverage ratio
        leverage_ratio = total_exposure / account_balance if account_balance > 0 else 0

        # Concentration risk (position la plus grande en % du portefeuille)
        position_sizes = [pos['margin'] for pos in positions]
        max_position_pct = max(position_sizes) / sum(position_sizes) * 100 if position_sizes else 0

        # Corrélation risk (simulation)
        correlation_risk = self._calculate_correlation_risk(positions)

        # Max individual loss
        max_individual_loss = max([abs(pos['unrealized_pnl']) for pos in positions] + [0])

        # Market exposure
        long_exposure = sum([pos['size'] * pos['current_price'] for pos in positions if pos['side'] == 'Long'])
        short_exposure = sum([pos['size'] * pos['current_price'] for pos in positions if pos['side'] == 'Short'])
        net_exposure = abs(long_exposure - short_exposure)

        st.session_state.risk_metrics = {
            'portfolio_var_95': var_95,
            'portfolio_var_99': var_99,
            'max_individual_loss': max_individual_loss,
            'correlation_risk': correlation_risk,
            'leverage_ratio': leverage_ratio,
            'concentration_risk': max_position_pct,
            'liquidity_risk': self._assess_liquidity_risk(positions),
            'market_exposure': net_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure
        }

    def _calculate_portfolio_var(self, positions: List[Dict]) -> Tuple[float, float]:
        """Calcule la VaR du portefeuille (95% et 99%)."""
        if not positions:
            return 0.0, 0.0

        # Simulation Monte Carlo simplifiée
        n_simulations = 1000
        portfolio_returns = []

        for _ in range(n_simulations):
            total_change = 0

            for pos in positions:
                # Simulation d'un changement de prix (volatilité quotidienne ~2%)
                price_change = np.random.normal(0, 0.02)
                position_change = pos['size'] * pos['current_price'] * price_change

                # Application du leverage
                leveraged_change = position_change * pos.get('leverage', 1.0)

                if pos['side'] == 'Long':
                    total_change += leveraged_change
                else:
                    total_change -= leveraged_change

            portfolio_returns.append(total_change)

        # Calcul VaR
        var_95 = abs(np.percentile(portfolio_returns, 5))  # 95% confiance
        var_99 = abs(np.percentile(portfolio_returns, 1))  # 99% confiance

        return var_95, var_99

    def _calculate_correlation_risk(self, positions: List[Dict]) -> float:
        """Évalue le risque de corrélation entre les positions."""
        if len(positions) < 2:
            return 0.0

        # Simulation de corrélations entre les assets
        symbols = [pos['symbol'] for pos in positions]
        n = len(symbols)

        # Matrice de corrélation simulée (simplifiée)
        correlations = []
        for i in range(n):
            for j in range(i+1, n):
                # Simulation corrélation basée sur les types d'assets
                if 'BTC' in symbols[i] and 'ETH' in symbols[j]:
                    corr = 0.7  # Crypto corrélées
                elif 'USDT' in symbols[i] and 'USDT' in symbols[j]:
                    corr = 0.9  # Mêmes paires de base
                else:
                    corr = np.random.uniform(0.3, 0.8)

                correlations.append(corr)

        return max(correlations) if correlations else 0.0

    def _assess_liquidity_risk(self, positions: List[Dict]) -> str:
        """Évalue le risque de liquidité."""
        if not positions:
            return 'LOW'

        # Simulation basée sur les symbols
        high_liquidity = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        total_exposure = sum([pos['size'] * pos['current_price'] for pos in positions])

        high_liq_exposure = sum([
            pos['size'] * pos['current_price']
            for pos in positions
            if pos['symbol'] in high_liquidity
        ])

        liquidity_ratio = high_liq_exposure / total_exposure if total_exposure > 0 else 1.0

        if liquidity_ratio > 0.8:
            return 'LOW'
        elif liquidity_ratio > 0.5:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def render(self):
        """Rendu principal du moniteur de risque."""
        st.header("⚠️ Risk Monitor")

        # Mise à jour automatique
        if st.button("🔄 Recalculer Risques", key="risk_refresh"):
            self._calculate_risk_metrics()
            st.success("✅ Métriques de risque recalculées")

        # Métriques principales
        self._render_risk_dashboard()

        # Onglets de monitoring
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 VaR & Métriques",
            "⚖️ Limites & Seuils",
            "🔥 Stress Testing",
            "⚠️ Alertes",
            "⚙️ Configuration"
        ])

        with tab1:
            self._render_var_metrics()

        with tab2:
            self._render_limits_monitoring()

        with tab3:
            self._render_stress_testing()

        with tab4:
            self._render_risk_alerts()

        with tab5:
            self._render_risk_configuration()

    def _render_risk_dashboard(self):
        """Dashboard principal des risques."""
        metrics = st.session_state.risk_metrics
        config = st.session_state.risk_config

        # Indicateurs principaux
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            var_95 = metrics.get('portfolio_var_95', 0)
            max_var = config['max_portfolio_var']
            var_status = '🔴' if var_95 > max_var else '🟡' if var_95 > max_var * 0.8 else '🟢'

            st.metric(
                f"{var_status} VaR 95%",
                f"${var_95:,.0f}",
                f"{var_95/max_var*100:.1f}% de la limite"
            )

        with col2:
            leverage = metrics.get('leverage_ratio', 0)
            max_leverage = config['max_leverage']
            lev_status = '🔴' if leverage > max_leverage else '🟡' if leverage > max_leverage * 0.8 else '🟢'

            st.metric(
                f"{lev_status} Leverage",
                f"{leverage:.2f}x",
                f"Max: {max_leverage}x"
            )

        with col3:
            concentration = metrics.get('concentration_risk', 0)
            max_concentration = config['position_limit_pct']
            conc_status = '🔴' if concentration > max_concentration else '🟡' if concentration > max_concentration * 0.8 else '🟢'

            st.metric(
                f"{conc_status} Concentration",
                f"{concentration:.1f}%",
                f"Max: {max_concentration}%"
            )

        with col4:
            correlation = metrics.get('correlation_risk', 0)
            max_corr = config['max_correlation']
            corr_status = '🔴' if correlation > max_corr else '🟡' if correlation > max_corr * 0.9 else '🟢'

            st.metric(
                f"{corr_status} Corrélation Max",
                f"{correlation:.2f}",
                f"Seuil: {max_corr:.2f}"
            )

        with col5:
            liquidity = metrics.get('liquidity_risk', 'LOW')
            liq_colors = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}

            st.metric(
                f"{liq_colors.get(liquidity, '⚪')} Liquidité",
                liquidity,
                "Risk Level"
            )

    def _render_var_metrics(self):
        """Métriques VaR détaillées."""
        st.subheader("📊 Value at Risk & Métriques")

        metrics = st.session_state.risk_metrics

        # Graphique VaR historique
        col1, col2 = st.columns(2)

        with col1:
            self._render_var_history_chart()

        with col2:
            self._render_exposure_breakdown()

        # Métriques détaillées
        st.subheader("📈 Métriques Détaillées")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📉 VaR 99%", f"${metrics.get('portfolio_var_99', 0):,.0f}")
            st.metric("💥 Perte Max Individuelle", f"${metrics.get('max_individual_loss', 0):,.0f}")

        with col2:
            st.metric("📊 Exposition Nette", f"${metrics.get('market_exposure', 0):,.0f}")
            st.metric("🔗 Risque Corrélation", f"{metrics.get('correlation_risk', 0):.2f}")

        with col3:
            long_exp = metrics.get('long_exposure', 0)
            st.metric("📈 Exposition Long", f"${long_exp:,.0f}")
            short_exp = metrics.get('short_exposure', 0)
            st.metric("📉 Exposition Short", f"${short_exp:,.0f}")

        with col4:
            net_exp = long_exp - short_exp
            direction = "🟢 Net Long" if net_exp > 0 else "🔴 Net Short"
            st.metric(direction, f"${abs(net_exp):,.0f}")

            hedge_ratio = min(long_exp, short_exp) / max(long_exp, short_exp) if max(long_exp, short_exp) > 0 else 0
            st.metric("⚖️ Hedge Ratio", f"{hedge_ratio:.2f}")

    def _render_var_history_chart(self):
        """Graphique historique VaR."""
        st.subheader("📊 Historique VaR")

        history = st.session_state.var_history

        if not history:
            st.info("Aucune donnée historique")
            return

        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = go.Figure()

        # VaR 95%
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_var_95'],
            mode='lines',
            name='VaR 95%',
            line=dict(color='#FF9800', width=2)
        ))

        # VaR 99%
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_var_99'],
            mode='lines',
            name='VaR 99%',
            line=dict(color='#F44336', width=2)
        ))

        # Limite VaR
        max_var = st.session_state.risk_config['max_portfolio_var']
        fig.add_hline(y=max_var, line_dash="dash", line_color="red",
                     annotation_text=f"Limite: ${max_var:,.0f}")

        fig.update_layout(
            height=300,
            xaxis_title="Date",
            yaxis_title="VaR ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_exposure_breakdown(self):
        """Breakdown de l'exposition par asset."""
        st.subheader("🥧 Exposition par Asset")

        positions = st.session_state.get('positions', [])

        if not positions:
            st.info("Aucune position")
            return

        # Calcul expositions
        exposures = []
        for pos in positions:
            exposure = pos['size'] * pos['current_price']
            exposures.append({
                'Symbol': pos['symbol'],
                'Exposition': exposure,
                'Side': pos['side'],
                'Leverage': pos.get('leverage', 1.0)
            })

        df = pd.DataFrame(exposures)

        # Graphique en camembert
        fig = px.pie(
            df,
            values='Exposition',
            names='Symbol',
            title="Répartition Exposition"
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    def _render_limits_monitoring(self):
        """Surveillance des limites et seuils."""
        st.subheader("⚖️ Surveillance des Limites")

        config = st.session_state.risk_config
        metrics = st.session_state.risk_metrics
        positions = st.session_state.get('positions', [])

        # Statut des limites
        limits_status = []

        # VaR Limit
        var_95 = metrics.get('portfolio_var_95', 0)
        var_usage = var_95 / config['max_portfolio_var'] * 100
        limits_status.append({
            'Limite': 'VaR 95%',
            'Actuel': f"${var_95:,.0f}",
            'Maximum': f"${config['max_portfolio_var']:,.0f}",
            'Utilisation (%)': var_usage,
            'Statut': '🔴' if var_usage > 100 else '🟡' if var_usage > 80 else '🟢'
        })

        # Leverage Limit
        leverage = metrics.get('leverage_ratio', 0)
        lev_usage = leverage / config['max_leverage'] * 100
        limits_status.append({
            'Limite': 'Leverage',
            'Actuel': f"{leverage:.2f}x",
            'Maximum': f"{config['max_leverage']:.1f}x",
            'Utilisation (%)': lev_usage,
            'Statut': '🔴' if lev_usage > 100 else '🟡' if lev_usage > 80 else '🟢'
        })

        # Position Size Limits
        if positions:
            max_position_value = max([pos['size'] * pos['current_price'] for pos in positions])
            pos_usage = max_position_value / config['max_position_size'] * 100
            limits_status.append({
                'Limite': 'Taille Position Max',
                'Actuel': f"${max_position_value:,.0f}",
                'Maximum': f"${config['max_position_size']:,.0f}",
                'Utilisation (%)': pos_usage,
                'Statut': '🔴' if pos_usage > 100 else '🟡' if pos_usage > 80 else '🟢'
            })

        # Concentration Limit
        concentration = metrics.get('concentration_risk', 0)
        conc_usage = concentration / config['position_limit_pct'] * 100
        limits_status.append({
            'Limite': 'Concentration',
            'Actuel': f"{concentration:.1f}%",
            'Maximum': f"{config['position_limit_pct']:.1f}%",
            'Utilisation (%)': conc_usage,
            'Statut': '🔴' if conc_usage > 100 else '🟡' if conc_usage > 80 else '🟢'
        })

        # Tableau des limites
        df_limits = pd.DataFrame(limits_status)
        st.dataframe(df_limits, use_container_width=True, hide_index=True)

        # Graphique utilisation des limites
        fig = go.Figure()

        colors = ['red' if status['Utilisation (%)'] > 100 else
                 'orange' if status['Utilisation (%)'] > 80 else 'green'
                 for status in limits_status]

        fig.add_trace(go.Bar(
            x=[status['Limite'] for status in limits_status],
            y=[status['Utilisation (%)'] for status in limits_status],
            marker_color=colors,
            text=[f"{status['Utilisation (%)']:.1f}%" for status in limits_status],
            textposition='auto'
        ))

        fig.add_hline(y=100, line_dash="dash", line_color="red",
                     annotation_text="Limite 100%")
        fig.add_hline(y=80, line_dash="dash", line_color="orange",
                     annotation_text="Seuil 80%")

        fig.update_layout(
            height=400,
            title="Utilisation des Limites de Risque",
            xaxis_title="Type de Limite",
            yaxis_title="Utilisation (%)",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_stress_testing(self):
        """Stress testing et scenarios."""
        st.subheader("🔥 Stress Testing")

        positions = st.session_state.get('positions', [])

        if not positions:
            st.info("Aucune position pour le stress testing")
            return

        # Scenarios de stress
        scenarios = [
            {'name': 'Crash -20%', 'market_move': -0.20, 'volatility_mult': 2.0},
            {'name': 'Rally +15%', 'market_move': 0.15, 'volatility_mult': 1.5},
            {'name': 'Flash Crash -10%', 'market_move': -0.10, 'volatility_mult': 5.0},
            {'name': 'Volatilité Extrême', 'market_move': 0.0, 'volatility_mult': 10.0},
            {'name': 'Corrélation 1.0', 'market_move': -0.05, 'correlation': 1.0}
        ]

        # Calcul impact scenarios
        scenario_results = []

        for scenario in scenarios:
            total_impact = 0

            for pos in positions:
                # Impact prix
                price_impact = pos['current_price'] * scenario['market_move']
                position_impact = pos['size'] * price_impact

                # Application direction
                if pos['side'] == 'Long':
                    total_impact += position_impact
                else:
                    total_impact -= position_impact

                # Application leverage
                total_impact *= pos.get('leverage', 1.0)

            scenario_results.append({
                'Scenario': scenario['name'],
                'Impact PnL': total_impact,
                'Impact %': total_impact / 100000 * 100,  # vs capital initial
                'Severity': 'HIGH' if abs(total_impact) > 10000 else 'MEDIUM' if abs(total_impact) > 5000 else 'LOW'
            })

        # Tableau résultats
        df_stress = pd.DataFrame(scenario_results)

        # Coloration
        def color_impact(val):
            if val > 0:
                return 'color: #00D4AA'
            elif val < -5000:
                return 'color: #FF4336; font-weight: bold'
            elif val < -2000:
                return 'color: #FF9800'
            else:
                return 'color: #FF4336'

        styled_df = df_stress.style.applymap(color_impact, subset=['Impact PnL'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Graphique stress scenarios
        fig = go.Figure()

        colors = ['red' if result['Impact PnL'] < -5000 else
                 'orange' if result['Impact PnL'] < 0 else 'green'
                 for result in scenario_results]

        fig.add_trace(go.Bar(
            x=[result['Scenario'] for result in scenario_results],
            y=[result['Impact PnL'] for result in scenario_results],
            marker_color=colors,
            text=[f"${result['Impact PnL']:,.0f}" for result in scenario_results],
            textposition='auto'
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            height=400,
            title="Impact des Scenarios de Stress",
            xaxis_title="Scenario",
            yaxis_title="Impact PnL ($)",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Recommandations
        worst_scenario = min(scenario_results, key=lambda x: x['Impact PnL'])
        if worst_scenario['Impact PnL'] < -5000:
            st.error(f"⚠️ Risque Élevé: Le scenario '{worst_scenario['Scenario']}' "
                    f"pourrait causer une perte de ${abs(worst_scenario['Impact PnL']):,.0f}")

    def _render_risk_alerts(self):
        """Alertes de risque actives."""
        st.subheader("⚠️ Alertes de Risque")

        # Génération alertes automatiques
        alerts = self._generate_risk_alerts()

        if alerts:
            for alert in alerts:
                severity_colors = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
                severity_color = severity_colors.get(alert['severity'], '⚪')

                with st.container():
                    col1, col2, col3 = st.columns([1, 5, 1])

                    with col1:
                        st.write(severity_color)

                    with col2:
                        st.write(f"**{alert['title']}**")
                        st.caption(alert['description'])
                        if alert.get('recommendation'):
                            st.info(f"💡 {alert['recommendation']}")

                    with col3:
                        if st.button("✅", key=f"resolve_alert_{alert['id']}"):
                            self._resolve_alert(alert['id'])

                    st.markdown("---")
        else:
            st.success("✅ Aucune alerte de risque active")

        # Bouton d'urgence
        if st.session_state.risk_config.get('emergency_stop_enabled', True):
            st.markdown("### 🚨 Contrôles d'Urgence")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🛑 ARRÊT D'URGENCE", type="primary", use_container_width=True):
                    self._emergency_stop()

            with col2:
                if st.button("📉 FERMER TOUTES POSITIONS", use_container_width=True):
                    self._close_all_positions()

    def _render_risk_configuration(self):
        """Configuration des paramètres de risque."""
        st.subheader("⚙️ Configuration Risque")

        config = st.session_state.risk_config

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💰 Limites Financières")

            config['max_portfolio_var'] = st.number_input(
                "VaR Maximum Portfolio ($)",
                min_value=1000.0,
                max_value=50000.0,
                value=config['max_portfolio_var'],
                step=500.0
            )

            config['max_position_size'] = st.number_input(
                "Taille Position Maximum ($)",
                min_value=1000.0,
                max_value=100000.0,
                value=config['max_position_size'],
                step=1000.0
            )

            config['daily_loss_limit'] = st.number_input(
                "Limite Perte Journalière ($)",
                min_value=500.0,
                max_value=10000.0,
                value=config['daily_loss_limit'],
                step=100.0
            )

        with col2:
            st.markdown("#### ⚖️ Limites Techniques")

            config['max_leverage'] = st.slider(
                "Leverage Maximum",
                min_value=1.0,
                max_value=10.0,
                value=config['max_leverage'],
                step=0.1
            )

            config['max_correlation'] = st.slider(
                "Corrélation Maximum",
                min_value=0.5,
                max_value=1.0,
                value=config['max_correlation'],
                step=0.05
            )

            config['position_limit_pct'] = st.slider(
                "Position Maximum (% portfolio)",
                min_value=5.0,
                max_value=50.0,
                value=config['position_limit_pct'],
                step=1.0
            )

        st.markdown("#### 🚨 Contrôles d'Urgence")

        col3, col4 = st.columns(2)
        with col3:
            config['emergency_stop_enabled'] = st.checkbox(
                "Arrêt d'urgence activé",
                value=config['emergency_stop_enabled']
            )

        with col4:
            config['margin_call_threshold'] = st.slider(
                "Seuil Margin Call (%)",
                min_value=50.0,
                max_value=95.0,
                value=config['margin_call_threshold'],
                step=1.0
            )

        # Sauvegarde configuration
        if st.button("💾 Sauvegarder Configuration", type="primary"):
            st.session_state.risk_config = config
            st.success("✅ Configuration sauvegardée")

    def _generate_risk_alerts(self) -> List[Dict]:
        """Génère les alertes de risque automatiques."""
        alerts = []
        metrics = st.session_state.risk_metrics
        config = st.session_state.risk_config

        # Alerte VaR
        var_95 = metrics.get('portfolio_var_95', 0)
        if var_95 > config['max_portfolio_var']:
            alerts.append({
                'id': 'var_exceeded',
                'severity': 'HIGH',
                'title': 'VaR Limite Dépassée',
                'description': f"VaR à ${var_95:,.0f} dépasse la limite de ${config['max_portfolio_var']:,.0f}",
                'recommendation': 'Réduire la taille des positions ou ajouter du hedging'
            })

        # Alerte Leverage
        leverage = metrics.get('leverage_ratio', 0)
        if leverage > config['max_leverage']:
            alerts.append({
                'id': 'leverage_exceeded',
                'severity': 'HIGH',
                'title': 'Leverage Excessif',
                'description': f"Leverage à {leverage:.2f}x dépasse la limite de {config['max_leverage']:.1f}x",
                'recommendation': 'Réduire les positions avec leverage ou augmenter le capital'
            })

        # Alerte Concentration
        concentration = metrics.get('concentration_risk', 0)
        if concentration > config['position_limit_pct']:
            alerts.append({
                'id': 'concentration_risk',
                'severity': 'MEDIUM',
                'title': 'Risque de Concentration',
                'description': f"Position représente {concentration:.1f}% du portfolio (limite: {config['position_limit_pct']:.1f}%)",
                'recommendation': 'Diversifier le portefeuille ou réduire la position dominante'
            })

        # Alerte Corrélation
        correlation = metrics.get('correlation_risk', 0)
        if correlation > config['max_correlation']:
            alerts.append({
                'id': 'high_correlation',
                'severity': 'MEDIUM',
                'title': 'Corrélation Élevée',
                'description': f"Corrélation maximum à {correlation:.2f} (limite: {config['max_correlation']:.2f})",
                'recommendation': 'Ajouter des positions décorrélées ou réduire les positions similaires'
            })

        return alerts

    def _emergency_stop(self):
        """Arrêt d'urgence du trading."""
        st.session_state.emergency_stop_active = True
        st.session_state.trading_active = False
        st.error("🚨 ARRÊT D'URGENCE ACTIVÉ - Trading suspendu")

    def _close_all_positions(self):
        """Ferme toutes les positions."""
        st.session_state.positions = []
        st.warning("⚠️ Toutes les positions ont été fermées")

    def _resolve_alert(self, alert_id: str):
        """Marque une alerte comme résolue."""
        # Dans une vraie application, on ajouterait à un log des alertes résolues
        st.success(f"✅ Alerte {alert_id} marquée comme résolue")


# Instance globale pour utilisation dans les pages
risk_monitor = RiskMonitor()