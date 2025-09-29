"""
📊 Position Monitor Component
Surveillance temps réel des positions - PnL, exposition, métriques
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math


class PositionMonitor:
    """Moniteur de positions avancé avec analytics temps réel."""

    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        """Initialise les données de session pour le monitoring."""
        if 'positions' not in st.session_state:
            st.session_state.positions = []
            self._create_sample_positions()

        if 'position_history' not in st.session_state:
            st.session_state.position_history = []
            self._generate_position_history()

        if 'position_alerts' not in st.session_state:
            st.session_state.position_alerts = []

    def _create_sample_positions(self):
        """Crée des positions d'exemple pour la démonstration."""
        base_time = datetime.now()

        positions = [
            {
                'id': 'pos_001',
                'symbol': 'BTC/USDT',
                'side': 'Long',
                'size': 0.75,
                'entry_price': 42800.0,
                'current_price': 43250.0,
                'mark_price': 43240.0,
                'leverage': 1.0,
                'margin': 32100.0,
                'unrealized_pnl': 337.5,
                'realized_pnl': 0.0,
                'entry_time': base_time - timedelta(hours=4),
                'last_update': base_time,
                'stop_loss': 40500.0,
                'take_profit': 45000.0,
                'funding_fee': -2.5,
                'commission': 21.4
            },
            {
                'id': 'pos_002',
                'symbol': 'ETH/USDT',
                'side': 'Short',
                'size': 8.0,
                'entry_price': 2680.0,
                'current_price': 2650.0,
                'mark_price': 2649.5,
                'leverage': 2.0,
                'margin': 10720.0,
                'unrealized_pnl': 244.0,
                'realized_pnl': 150.0,
                'entry_time': base_time - timedelta(hours=2),
                'last_update': base_time,
                'stop_loss': 2750.0,
                'take_profit': 2500.0,
                'funding_fee': -1.8,
                'commission': 10.7
            },
            {
                'id': 'pos_003',
                'symbol': 'BNB/USDT',
                'side': 'Long',
                'size': 50.0,
                'entry_price': 234.5,
                'current_price': 235.2,
                'mark_price': 235.1,
                'leverage': 1.5,
                'margin': 7816.7,
                'unrealized_pnl': 35.0,
                'realized_pnl': -45.0,
                'entry_time': base_time - timedelta(minutes=30),
                'last_update': base_time,
                'stop_loss': 220.0,
                'take_profit': 250.0,
                'funding_fee': -0.5,
                'commission': 5.8
            }
        ]
        st.session_state.positions = positions

    def _generate_position_history(self):
        """Génère l'historique des PnL pour les graphiques."""
        base_time = datetime.now() - timedelta(hours=24)
        history = []

        for i in range(288):  # 24h en intervalles de 5 minutes
            timestamp = base_time + timedelta(minutes=5*i)

            # Simulation PnL positions
            btc_pnl = 300 + np.random.normal(0, 50) + 30 * math.sin(i * 0.02)
            eth_pnl = 150 + np.random.normal(0, 30) + 20 * math.cos(i * 0.015)
            bnb_pnl = 20 + np.random.normal(0, 15)

            total_pnl = btc_pnl + eth_pnl + bnb_pnl

            history.append({
                'timestamp': timestamp,
                'btc_pnl': btc_pnl,
                'eth_pnl': eth_pnl,
                'bnb_pnl': bnb_pnl,
                'total_pnl': total_pnl,
                'cumulative_pnl': sum([h.get('total_pnl', 0) for h in history]) + total_pnl
            })

        st.session_state.position_history = history

    def render(self):
        """Rendu principal du moniteur de positions."""
        st.header("📊 Position Monitor")

        # Auto-refresh
        if st.button("🔄 Actualiser Positions", key="position_refresh"):
            self._update_positions()

        # Métriques globales
        self._render_portfolio_metrics()

        # Onglets de monitoring
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Positions Actives",
            "📊 Analytics",
            "⚠️ Alertes & Risques",
            "📋 Historique PnL"
        ])

        with tab1:
            self._render_active_positions()

        with tab2:
            self._render_position_analytics()

        with tab3:
            self._render_risk_alerts()

        with tab4:
            self._render_pnl_history()

    def _render_portfolio_metrics(self):
        """Métriques globales du portefeuille."""
        positions = st.session_state.positions

        if not positions:
            st.info("Aucune position ouverte")
            return

        # Calculs agrégés
        total_margin = sum([pos['margin'] for pos in positions])
        total_unrealized_pnl = sum([pos['unrealized_pnl'] for pos in positions])
        total_realized_pnl = sum([pos['realized_pnl'] for pos in positions])
        total_fees = sum([pos['funding_fee'] + pos['commission'] for pos in positions])

        net_pnl = total_unrealized_pnl + total_realized_pnl - total_fees
        account_balance = 100000.0  # Simulation
        total_exposure = sum([pos['size'] * pos['current_price'] for pos in positions])

        # Affichage métriques
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "💰 Margin Total",
                f"${total_margin:,.2f}",
                f"{total_margin/account_balance*100:.1f}% du capital"
            )

        with col2:
            pnl_color = "normal" if total_unrealized_pnl >= 0 else "inverse"
            st.metric(
                "📈 PnL Non Réalisé",
                f"${total_unrealized_pnl:,.2f}",
                f"{total_unrealized_pnl/total_margin*100:+.2f}%",
                delta_color=pnl_color
            )

        with col3:
            st.metric(
                "💸 PnL Réalisé",
                f"${total_realized_pnl:,.2f}",
                f"{total_realized_pnl/account_balance*100:+.2f}%"
            )

        with col4:
            st.metric(
                "🌐 Exposition Totale",
                f"${total_exposure:,.0f}",
                f"{total_exposure/account_balance:.1f}x capital"
            )

        with col5:
            net_color = "normal" if net_pnl >= 0 else "inverse"
            st.metric(
                "🎯 PnL Net",
                f"${net_pnl:,.2f}",
                f"{net_pnl/account_balance*100:+.2f}%",
                delta_color=net_color
            )

    def _render_active_positions(self):
        """Tableau détaillé des positions actives."""
        st.subheader("📈 Positions Actives")

        positions = st.session_state.positions

        if not positions:
            st.info("Aucune position ouverte")
            return

        # Filtrages
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_symbol = st.selectbox(
                "Filtrer par Symbol",
                options=['Tous'] + list(set([pos['symbol'] for pos in positions]))
            )
        with col2:
            filter_side = st.selectbox(
                "Filtrer par Direction",
                options=['Tous', 'Long', 'Short']
            )
        with col3:
            sort_by = st.selectbox(
                "Trier par",
                options=['unrealized_pnl', 'size', 'entry_time', 'symbol']
            )

        # Application filtres
        filtered_positions = positions
        if filter_symbol != 'Tous':
            filtered_positions = [p for p in filtered_positions if p['symbol'] == filter_symbol]
        if filter_side != 'Tous':
            filtered_positions = [p for p in filtered_positions if p['side'] == filter_side]

        # Tri
        filtered_positions = sorted(filtered_positions, key=lambda x: x[sort_by], reverse=True)

        # Tableau positions
        for pos in filtered_positions:
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 2])

                with col1:
                    direction_emoji = '🟢' if pos['side'] == 'Long' else '🔴'
                    st.write(f"**{direction_emoji} {pos['symbol']}**")
                    st.caption(f"ID: {pos['id']}")

                with col2:
                    st.write(f"**{pos['size']}**")
                    st.caption(f"{pos['leverage']}x leverage")

                with col3:
                    st.write(f"${pos['entry_price']:,.2f}")
                    st.caption(f"Mark: ${pos['mark_price']:,.2f}")

                with col4:
                    pnl_pct = pos['unrealized_pnl'] / pos['margin'] * 100
                    pnl_color = '🟢' if pos['unrealized_pnl'] >= 0 else '🔴'
                    st.write(f"{pnl_color} ${pos['unrealized_pnl']:,.2f}")
                    st.caption(f"{pnl_pct:+.2f}%")

                with col5:
                    # Distances stop/take profit
                    current_price = pos['current_price']
                    if pos['side'] == 'Long':
                        sl_distance = (current_price - pos['stop_loss']) / current_price * 100
                        tp_distance = (pos['take_profit'] - current_price) / current_price * 100
                    else:
                        sl_distance = (pos['stop_loss'] - current_price) / current_price * 100
                        tp_distance = (current_price - pos['take_profit']) / current_price * 100

                    st.write(f"SL: {sl_distance:.1f}%")
                    st.caption(f"TP: {tp_distance:.1f}%")

                with col6:
                    col6a, col6b, col6c = st.columns(3)
                    with col6a:
                        if st.button("✏️", key=f"edit_pos_{pos['id']}", help="Modifier"):
                            self._edit_position(pos)
                    with col6b:
                        if st.button("❌", key=f"close_pos_{pos['id']}", help="Fermer"):
                            self._close_position(pos['id'])
                    with col6c:
                        if st.button("📊", key=f"details_pos_{pos['id']}", help="Détails"):
                            self._show_position_details(pos)

                # Barre de progression pour les stop/take profit
                progress_container = st.container()
                with progress_container:
                    if pos['side'] == 'Long':
                        min_price = min(pos['stop_loss'], pos['entry_price'], pos['current_price'])
                        max_price = max(pos['take_profit'], pos['entry_price'], pos['current_price'])
                    else:
                        min_price = min(pos['take_profit'], pos['entry_price'], pos['current_price'])
                        max_price = max(pos['stop_loss'], pos['entry_price'], pos['current_price'])

                    if max_price > min_price:
                        progress = (pos['current_price'] - min_price) / (max_price - min_price)
                        progress = max(0, min(1, progress))

                        st.progress(progress, text=f"Position: ${pos['current_price']:,.2f}")

                st.markdown("---")

    def _render_position_analytics(self):
        """Analytics avancées des positions."""
        st.subheader("📊 Analytics Positions")

        positions = st.session_state.positions

        if not positions:
            st.info("Aucune position pour l'analyse")
            return

        # Graphiques analytics
        col1, col2 = st.columns(2)

        with col1:
            self._render_pnl_breakdown()

        with col2:
            self._render_exposure_allocation()

        # Métriques détaillées
        self._render_performance_metrics()

        # Analyse des corrélations
        self._render_correlation_matrix()

    def _render_pnl_breakdown(self):
        """Breakdown du PnL par position."""
        st.subheader("💰 Breakdown PnL")

        positions = st.session_state.positions

        symbols = [pos['symbol'] for pos in positions]
        pnls = [pos['unrealized_pnl'] for pos in positions]
        colors = ['#00D4AA' if pnl >= 0 else '#FF4336' for pnl in pnls]

        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=pnls,
                marker_color=colors,
                text=[f"${pnl:+.2f}" for pnl in pnls],
                textposition='auto'
            )
        ])

        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Symbol",
            yaxis_title="PnL ($)",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig, use_container_width=True)

    def _render_exposure_allocation(self):
        """Allocation de l'exposition par asset."""
        st.subheader("🥧 Allocation Exposition")

        positions = st.session_state.positions

        symbols = [pos['symbol'].split('/')[0] for pos in positions]  # Récupère le symbole de base
        exposures = [pos['size'] * pos['current_price'] for pos in positions]

        fig = go.Figure(data=[
            go.Pie(
                labels=symbols,
                values=exposures,
                hole=0.4,
                textinfo='label+percent',
                textposition='outside'
            )
        ])

        fig.update_layout(
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_performance_metrics(self):
        """Métriques de performance détaillées."""
        st.subheader("📈 Métriques Performance")

        positions = st.session_state.positions

        # Calculs métriques
        pnls = [pos['unrealized_pnl'] for pos in positions]
        margins = [pos['margin'] for pos in positions]

        win_rate = len([pnl for pnl in pnls if pnl > 0]) / len(pnls) * 100 if pnls else 0
        avg_win = np.mean([pnl for pnl in pnls if pnl > 0]) if any(pnl > 0 for pnl in pnls) else 0
        avg_loss = np.mean([pnl for pnl in pnls if pnl < 0]) if any(pnl < 0 for pnl in pnls) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        total_pnl = sum(pnls)
        total_margin = sum(margins)
        roi = total_pnl / total_margin * 100 if total_margin > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        with col2:
            st.metric("📈 Gain Moyen", f"${avg_win:.2f}")
        with col3:
            st.metric("📉 Perte Moyenne", f"${avg_loss:.2f}")
        with col4:
            st.metric("⚖️ Profit Factor", f"{profit_factor:.2f}")

        # ROI par position
        st.subheader("💹 ROI par Position")

        roi_data = []
        for pos in positions:
            position_roi = pos['unrealized_pnl'] / pos['margin'] * 100
            roi_data.append({
                'Symbol': pos['symbol'],
                'ROI (%)': position_roi,
                'Durée': (datetime.now() - pos['entry_time']).total_seconds() / 3600  # heures
            })

        roi_df = pd.DataFrame(roi_data)
        if not roi_df.empty:
            st.dataframe(roi_df, use_container_width=True)

    def _render_correlation_matrix(self):
        """Matrice de corrélation des positions."""
        st.subheader("🔗 Corrélations")

        # Simulation données de corrélation
        symbols = [pos['symbol'] for pos in st.session_state.positions]

        if len(symbols) < 2:
            st.info("Minimum 2 positions requises pour la corrélation")
            return

        # Génération matrice corrélation simulée
        n = len(symbols)
        correlation_matrix = np.random.rand(n, n)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Symétrique
        np.fill_diagonal(correlation_matrix, 1)  # Diagonale = 1

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix,
            texttemplate="%{text:.2f}",
            textfont={"size": 10}
        ))

        fig.update_layout(
            height=300,
            title="Corrélation entre Positions"
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_risk_alerts(self):
        """Alertes et surveillance des risques."""
        st.subheader("⚠️ Alertes & Surveillance")

        # Génération alertes automatiques
        alerts = self._generate_risk_alerts()

        if alerts:
            for alert in alerts:
                alert_color = {
                    'HIGH': '🔴',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(alert['severity'], '⚪')

                with st.container():
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col1:
                        st.write(alert_color)
                    with col2:
                        st.write(f"**{alert['title']}**")
                        st.caption(alert['description'])
                    with col3:
                        if st.button("❌", key=f"dismiss_{alert['id']}"):
                            self._dismiss_alert(alert['id'])
                    st.markdown("---")
        else:
            st.success("✅ Aucune alerte active")

        # Configuration des seuils d'alerte
        st.subheader("⚙️ Configuration Alertes")

        col1, col2 = st.columns(2)
        with col1:
            pnl_alert_threshold = st.slider("Seuil Alerte PnL (%)", -20, 20, -10)
            margin_alert_threshold = st.slider("Seuil Utilisation Margin (%)", 50, 100, 80)

        with col2:
            correlation_alert = st.checkbox("Alertes Corrélation Élevée")
            funding_alert = st.checkbox("Alertes Frais de Financement")

    def _render_pnl_history(self):
        """Historique graphique du PnL."""
        st.subheader("📋 Historique PnL")

        history = st.session_state.position_history

        if not history:
            st.info("Aucun historique disponible")
            return

        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Graphique PnL par asset
        fig = go.Figure()

        assets = ['btc_pnl', 'eth_pnl', 'bnb_pnl']
        colors = ['#F7931A', '#627EEA', '#F3BA2F']
        names = ['BTC', 'ETH', 'BNB']

        for asset, color, name in zip(assets, colors, names):
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[asset],
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))

        # PnL cumulé
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['total_pnl'],
            mode='lines',
            name='Total',
            line=dict(color='white', width=3, dash='dash')
        ))

        fig.update_layout(
            height=400,
            xaxis_title="Temps",
            yaxis_title="PnL ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Métriques historique
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_pnl = df['total_pnl'].max()
            st.metric("📈 PnL Maximum", f"${max_pnl:.2f}")
        with col2:
            min_pnl = df['total_pnl'].min()
            st.metric("📉 PnL Minimum", f"${min_pnl:.2f}")
        with col3:
            pnl_volatility = df['total_pnl'].std()
            st.metric("📊 Volatilité", f"${pnl_volatility:.2f}")
        with col4:
            sharpe_approx = df['total_pnl'].mean() / pnl_volatility if pnl_volatility > 0 else 0
            st.metric("📐 Sharpe (approx)", f"{sharpe_approx:.2f}")

    def _generate_risk_alerts(self) -> List[Dict]:
        """Génère les alertes de risque automatiques."""
        alerts = []
        positions = st.session_state.positions

        for pos in positions:
            # Alerte PnL négatif important
            pnl_pct = pos['unrealized_pnl'] / pos['margin'] * 100
            if pnl_pct < -10:
                alerts.append({
                    'id': f"pnl_alert_{pos['id']}",
                    'severity': 'HIGH' if pnl_pct < -15 else 'MEDIUM',
                    'title': f"PnL Négatif {pos['symbol']}",
                    'description': f"Position en perte de {pnl_pct:.1f}% ({pos['unrealized_pnl']:+.2f}$)"
                })

            # Alerte stop loss proche
            current_price = pos['current_price']
            if pos['side'] == 'Long':
                distance_to_sl = (current_price - pos['stop_loss']) / current_price * 100
            else:
                distance_to_sl = (pos['stop_loss'] - current_price) / current_price * 100

            if distance_to_sl < 5:
                alerts.append({
                    'id': f"sl_alert_{pos['id']}",
                    'severity': 'HIGH' if distance_to_sl < 2 else 'MEDIUM',
                    'title': f"Stop Loss Proche {pos['symbol']}",
                    'description': f"Stop loss à {distance_to_sl:.1f}% du prix actuel"
                })

        return alerts

    def _update_positions(self):
        """Met à jour les positions avec de nouvelles données."""
        for pos in st.session_state.positions:
            # Simulation mise à jour prix
            volatility = 0.002  # 0.2% volatilité
            price_change = np.random.normal(0, volatility)
            pos['current_price'] *= (1 + price_change)
            pos['mark_price'] = pos['current_price'] * (1 + np.random.normal(0, 0.0001))

            # Recalcul PnL
            if pos['side'] == 'Long':
                pos['unrealized_pnl'] = pos['size'] * (pos['current_price'] - pos['entry_price'])
            else:
                pos['unrealized_pnl'] = pos['size'] * (pos['entry_price'] - pos['current_price'])

            pos['last_update'] = datetime.now()

        st.success("✅ Positions mises à jour")

    def _edit_position(self, position: Dict):
        """Interface d'édition de position."""
        st.session_state.editing_position = position
        st.info(f"✏️ Édition position {position['symbol']} - {position['id']}")

    def _close_position(self, position_id: str):
        """Ferme une position."""
        st.session_state.positions = [
            pos for pos in st.session_state.positions
            if pos['id'] != position_id
        ]
        st.success(f"❌ Position {position_id} fermée")
        st.rerun()

    def _show_position_details(self, position: Dict):
        """Affiche les détails d'une position."""
        with st.expander(f"📊 Détails Position {position['symbol']}", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Informations de base**")
                st.write(f"ID: {position['id']}")
                st.write(f"Symbol: {position['symbol']}")
                st.write(f"Direction: {position['side']}")
                st.write(f"Taille: {position['size']}")
                st.write(f"Leverage: {position['leverage']}x")

            with col2:
                st.write("**Prix et PnL**")
                st.write(f"Prix d'entrée: ${position['entry_price']:,.2f}")
                st.write(f"Prix actuel: ${position['current_price']:,.2f}")
                st.write(f"Prix mark: ${position['mark_price']:,.2f}")
                st.write(f"PnL non réalisé: ${position['unrealized_pnl']:+.2f}")
                st.write(f"PnL réalisé: ${position['realized_pnl']:+.2f}")

            with col3:
                st.write("**Risk Management**")
                st.write(f"Stop Loss: ${position['stop_loss']:,.2f}")
                st.write(f"Take Profit: ${position['take_profit']:,.2f}")
                st.write(f"Margin: ${position['margin']:,.2f}")
                st.write(f"Frais funding: ${position['funding_fee']:+.2f}")
                st.write(f"Commission: ${position['commission']:+.2f}")

    def _dismiss_alert(self, alert_id: str):
        """Supprime une alerte."""
        st.session_state.position_alerts = [
            alert for alert in st.session_state.position_alerts
            if alert['id'] != alert_id
        ]


# Instance globale pour utilisation dans les pages
position_monitor = PositionMonitor()