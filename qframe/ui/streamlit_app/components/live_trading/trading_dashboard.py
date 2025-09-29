"""
🎯 Trading Dashboard Component
Core du Live Trading - Vue d'ensemble temps réel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time


class TradingDashboard:
    """Composant principal du tableau de bord trading live."""

    def __init__(self):
        self.refresh_interval = 3  # seconds
        self.init_session_state()

    def init_session_state(self):
        """Initialise les données de session pour le dashboard."""
        if 'dashboard_data' not in st.session_state:
            st.session_state.dashboard_data = {
                'account_balance': 100000.0,
                'total_pnl': 0.0,
                'daily_pnl': 0.0,
                'open_positions': 0,
                'active_orders': 0,
                'last_update': datetime.now()
            }

        if 'price_data' not in st.session_state:
            st.session_state.price_data = self._generate_initial_price_data()

        if 'pnl_history' not in st.session_state:
            st.session_state.pnl_history = []
            # Génération historique initiale
            base_time = datetime.now() - timedelta(hours=24)
            for i in range(288):  # 24h * 12 (5min intervals)
                timestamp = base_time + timedelta(minutes=5*i)
                pnl = np.random.normal(0, 50) * i / 100  # Drift progressif
                st.session_state.pnl_history.append({
                    'timestamp': timestamp,
                    'pnl': pnl,
                    'cumulative_pnl': sum([p['pnl'] for p in st.session_state.pnl_history]) + pnl
                })

    def _generate_initial_price_data(self) -> Dict[str, Dict]:
        """Génère les données de prix initiales pour les cryptos principales."""
        base_prices = {
            'BTC/USDT': 43250.0,
            'ETH/USDT': 2650.0,
            'BNB/USDT': 235.0,
            'XRP/USDT': 0.52,
            'ADA/USDT': 0.38,
            'SOL/USDT': 95.0,
            'DOT/USDT': 5.2,
            'MATIC/USDT': 0.72
        }

        price_data = {}
        base_time = datetime.now() - timedelta(hours=1)

        for symbol, base_price in base_prices.items():
            history = []
            current_price = base_price

            for i in range(60):  # 1h de données minute par minute
                timestamp = base_time + timedelta(minutes=i)
                # Random walk avec volatilité réaliste
                change_pct = np.random.normal(0, 0.002)  # 0.2% volatilité moyenne
                current_price *= (1 + change_pct)

                history.append({
                    'timestamp': timestamp,
                    'price': current_price,
                    'volume': np.random.uniform(1000000, 10000000),
                    'change_pct': change_pct * 100
                })

            price_data[symbol] = {
                'current_price': current_price,
                'history': history,
                'change_24h': np.random.uniform(-5, 5),
                'volume_24h': np.random.uniform(100000000, 1000000000)
            }

        return price_data

    def render(self):
        """Rendu principal du dashboard trading."""
        st.header("🎯 Trading Dashboard")

        # Auto-refresh
        if st.button("🔄 Actualiser", key="dashboard_refresh"):
            self._update_real_time_data()

        # Métriques principales
        self._render_key_metrics()

        # Graphiques principaux
        col1, col2 = st.columns(2)

        with col1:
            self._render_pnl_chart()

        with col2:
            self._render_portfolio_composition()

        # Tableaux de données
        self._render_market_overview()
        self._render_active_positions()

        # Auto-refresh toutes les 3 secondes
        time.sleep(self.refresh_interval)
        st.rerun()

    def _render_key_metrics(self):
        """Affiche les métriques clés en temps réel."""
        data = st.session_state.dashboard_data

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="💰 Balance Totale",
                value=f"${data['account_balance']:,.2f}",
                delta=f"{data['daily_pnl']:+.2f}"
            )

        with col2:
            st.metric(
                label="📈 PnL Total",
                value=f"${data['total_pnl']:,.2f}",
                delta=f"{data['total_pnl']/data['account_balance']*100:+.2f}%"
            )

        with col3:
            st.metric(
                label="📊 PnL Journalier",
                value=f"${data['daily_pnl']:,.2f}",
                delta=f"{data['daily_pnl']/data['account_balance']*100:+.2f}%"
            )

        with col4:
            st.metric(
                label="🎯 Positions Ouvertes",
                value=str(data['open_positions']),
                delta=f"+{np.random.randint(0, 3)}" if np.random.random() > 0.5 else "0"
            )

        with col5:
            st.metric(
                label="📋 Ordres Actifs",
                value=str(data['active_orders']),
                delta=f"+{np.random.randint(0, 2)}" if np.random.random() > 0.7 else "0"
            )

        # Dernière mise à jour
        st.caption(f"🕐 Dernière mise à jour: {data['last_update'].strftime('%H:%M:%S')}")

    def _render_pnl_chart(self):
        """Graphique PnL en temps réel."""
        st.subheader("📈 PnL en Temps Réel")

        if not st.session_state.pnl_history:
            st.info("Aucune donnée PnL disponible")
            return

        df = pd.DataFrame(st.session_state.pnl_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = go.Figure()

        # Ligne PnL cumulatif
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cumulative_pnl'],
            mode='lines',
            name='PnL Cumulé',
            line=dict(color='#00D4AA', width=2),
            fill='tonexty' if df['cumulative_pnl'].iloc[-1] > 0 else 'tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)' if df['cumulative_pnl'].iloc[-1] > 0 else 'rgba(255, 67, 54, 0.1)'
        ))

        # Ligne zéro
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Temps",
            yaxis_title="PnL ($)",
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_portfolio_composition(self):
        """Graphique composition du portefeuille."""
        st.subheader("🥧 Composition Portefeuille")

        # Simulation composition portefeuille
        assets = ['BTC', 'ETH', 'BNB', 'Cash', 'Others']
        values = [45000, 25000, 12000, 15000, 3000]

        fig = go.Figure(data=[go.Pie(
            labels=assets,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(
                colors=['#F7931A', '#627EEA', '#F3BA2F', '#4CAF50', '#FF9800'],
                line=dict(color='#FFFFFF', width=2)
            )
        )])

        fig.update_layout(
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_market_overview(self):
        """Aperçu du marché en temps réel."""
        st.subheader("🌍 Aperçu Marché")

        price_data = st.session_state.price_data

        market_df = pd.DataFrame([
            {
                'Symbol': symbol,
                'Prix': f"${data['current_price']:,.2f}" if data['current_price'] > 1 else f"${data['current_price']:.4f}",
                'Change 24h': f"{data['change_24h']:+.2f}%",
                'Volume 24h': f"${data['volume_24h']/1000000:.1f}M",
                'Trend': '🟢' if data['change_24h'] > 0 else '🔴'
            }
            for symbol, data in price_data.items()
        ])

        # Configuration colonnes avec couleurs
        def color_negative_red(val):
            if '+' in str(val):
                return 'color: #00D4AA'
            elif '-' in str(val):
                return 'color: #FF4336'
            return ''

        styled_df = market_df.style.applymap(color_negative_red, subset=['Change 24h'])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

    def _render_active_positions(self):
        """Affichage des positions actives."""
        st.subheader("📊 Positions Actives")

        # Simulation positions actives
        if 'active_positions' not in st.session_state:
            st.session_state.active_positions = self._generate_sample_positions()

        if not st.session_state.active_positions:
            st.info("Aucune position ouverte")
            return

        positions_df = pd.DataFrame(st.session_state.active_positions)

        # Configuration du dataframe avec couleurs
        def highlight_pnl(val):
            if val > 0:
                return 'background-color: rgba(0, 212, 170, 0.1); color: #00D4AA'
            elif val < 0:
                return 'background-color: rgba(255, 67, 54, 0.1); color: #FF4336'
            return ''

        styled_positions = positions_df.style.applymap(highlight_pnl, subset=['PnL ($)', 'PnL (%)'])

        st.dataframe(
            styled_positions,
            use_container_width=True,
            hide_index=True
        )

        # Actions rapides
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Nouvelle Position", key="new_position"):
                st.info("Redirection vers Order Manager...")

        with col2:
            if st.button("⚠️ Fermer Toutes", key="close_all"):
                st.warning("Confirmation requise pour fermer toutes les positions")

        with col3:
            if st.button("🔒 Stop Loss Global", key="global_stop"):
                st.error("Stop Loss Global activé!")

    def _generate_sample_positions(self) -> List[Dict]:
        """Génère des positions d'exemple."""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        positions = []

        for i, symbol in enumerate(symbols):
            if np.random.random() > 0.3:  # 70% chance d'avoir une position
                size = np.random.uniform(0.1, 2.0)
                entry_price = st.session_state.price_data[symbol]['current_price'] * np.random.uniform(0.95, 1.05)
                current_price = st.session_state.price_data[symbol]['current_price']

                side = np.random.choice(['Long', 'Short'])
                if side == 'Long':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                pnl_usd = size * entry_price * (pnl_pct / 100)

                positions.append({
                    'Symbol': symbol,
                    'Side': side,
                    'Size': f"{size:.3f}",
                    'Entry Price': f"${entry_price:,.2f}",
                    'Current Price': f"${current_price:,.2f}",
                    'PnL (%)': f"{pnl_pct:+.2f}%",
                    'PnL ($)': f"{pnl_usd:+.2f}"
                })

        return positions

    def _update_real_time_data(self):
        """Met à jour les données en temps réel."""
        current_time = datetime.now()

        # Mise à jour prix
        for symbol in st.session_state.price_data:
            old_price = st.session_state.price_data[symbol]['current_price']
            change = np.random.normal(0, 0.001)  # 0.1% volatilité
            new_price = old_price * (1 + change)

            st.session_state.price_data[symbol]['current_price'] = new_price
            st.session_state.price_data[symbol]['change_24h'] += change * 100

            # Ajouter à l'historique
            st.session_state.price_data[symbol]['history'].append({
                'timestamp': current_time,
                'price': new_price,
                'volume': np.random.uniform(1000000, 10000000),
                'change_pct': change * 100
            })

            # Garder seulement les 60 dernières minutes
            if len(st.session_state.price_data[symbol]['history']) > 60:
                st.session_state.price_data[symbol]['history'].pop(0)

        # Mise à jour PnL
        if st.session_state.active_positions:
            total_pnl_change = np.random.normal(0, 25)
            st.session_state.pnl_history.append({
                'timestamp': current_time,
                'pnl': total_pnl_change,
                'cumulative_pnl': st.session_state.pnl_history[-1]['cumulative_pnl'] + total_pnl_change
            })

            # Garder seulement les 288 dernières entrées (24h)
            if len(st.session_state.pnl_history) > 288:
                st.session_state.pnl_history.pop(0)

        # Mise à jour métriques dashboard
        st.session_state.dashboard_data['last_update'] = current_time
        st.session_state.dashboard_data['total_pnl'] = st.session_state.pnl_history[-1]['cumulative_pnl'] if st.session_state.pnl_history else 0
        st.session_state.dashboard_data['daily_pnl'] = sum([p['pnl'] for p in st.session_state.pnl_history[-288:]])  # Dernières 24h
        st.session_state.dashboard_data['open_positions'] = len(st.session_state.active_positions)

    def get_dashboard_summary(self) -> Dict:
        """Retourne un résumé des données du dashboard."""
        return {
            'total_balance': st.session_state.dashboard_data['account_balance'],
            'total_pnl': st.session_state.dashboard_data['total_pnl'],
            'daily_pnl': st.session_state.dashboard_data['daily_pnl'],
            'open_positions': st.session_state.dashboard_data['open_positions'],
            'active_orders': st.session_state.dashboard_data['active_orders'],
            'last_update': st.session_state.dashboard_data['last_update']
        }


# Instance global pour utilisation dans les pages
trading_dashboard = TradingDashboard()