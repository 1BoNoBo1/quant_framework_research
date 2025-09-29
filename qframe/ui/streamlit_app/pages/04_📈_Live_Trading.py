"""
Live Trading Interface - Interface complète pour trading en temps réel
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import json

# Import des composants
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_client import APIClient
from components.utils import init_session_state, get_cached_data
from components.charts import create_line_chart, create_candlestick_chart
from components.tables import create_data_table

# Configuration de la page
st.set_page_config(
    page_title="QFrame - Live Trading",
    page_icon="📈",
    layout="wide"
)

# Initialisation
api_client = APIClient()
init_session_state()

# Styles CSS personnalisés
st.markdown("""
<style>
    .live-trading-header {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .trading-card {
        background: #1a1a2e;
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 255, 136, 0.1);
    }
    .position-card {
        background: #0f0f23;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff88;
    }
    .order-card {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #6b88ff;
    }
    .profit-positive {
        color: #00ff88;
        font-weight: bold;
    }
    .profit-negative {
        color: #ff6b6b;
        font-weight: bold;
    }
    .status-live {
        background-color: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        display: inline-block;
        margin: 0.25rem;
    }
    .status-pending {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        display: inline-block;
        margin: 0.25rem;
    }
    .status-error {
        background-color: rgba(255, 107, 107, 0.1);
        border: 1px solid #ff6b6b;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        display: inline-block;
        margin: 0.25rem;
    }
    .market-price {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .price-up { color: #00ff88; }
    .price-down { color: #ff6b6b; }
    .price-neutral { color: #6b88ff; }

    .risk-gauge {
        text-align: center;
        padding: 1rem;
    }
    .risk-low { color: #00ff88; }
    .risk-medium { color: #ffc107; }
    .risk-high { color: #ff6b6b; }

    .quick-action-btn {
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
        font-weight: bold;
        cursor: pointer;
    }
    .btn-buy {
        background-color: #00ff88;
        color: black;
    }
    .btn-sell {
        background-color: #ff6b6b;
        color: white;
    }
    .btn-close {
        background-color: #ffc107;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="live-trading-header">
    <h1>📈 Live Trading Interface</h1>
    <p>Interface complète pour trading en temps réel avec gestion des risques intégrée</p>
</div>
""", unsafe_allow_html=True)

# Initialisation session state pour live trading
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'active_orders' not in st.session_state:
    st.session_state.active_orders = {}
if 'trading_balance' not in st.session_state:
    st.session_state.trading_balance = 100000.0  # Capital initial
if 'pnl_history' not in st.session_state:
    st.session_state.pnl_history = []

# Navigation principale
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "💹 Order Management",
    "📈 Positions",
    "⚠️ Risk Monitor",
    "📱 Alerts & News",
    "⚙️ Settings"
])

# ================== TAB 1: Dashboard ==================
with tab1:
    st.header("📊 Trading Dashboard")

    # Status du trading
    col_status1, col_status2, col_status3 = st.columns(3)

    with col_status1:
        if st.button("🔴 Start Trading" if not st.session_state.trading_active else "🟢 Stop Trading"):
            st.session_state.trading_active = not st.session_state.trading_active
            if st.session_state.trading_active:
                st.success("✅ Trading activé!")
            else:
                st.warning("⏸️ Trading désactivé!")
            st.rerun()

    with col_status2:
        trading_status = "🟢 LIVE" if st.session_state.trading_active else "🔴 STOPPED"
        st.markdown(f"<div class='status-live'><strong>Status: {trading_status}</strong></div>",
                   unsafe_allow_html=True)

    with col_status3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"<div class='status-live'><strong>Time: {current_time}</strong></div>",
                   unsafe_allow_html=True)

    # Métriques principales du portfolio
    st.subheader("💰 Portfolio Overview")

    col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)

    # Calcul des métriques (simulées)
    total_balance = st.session_state.trading_balance
    positions_value = sum([pos.get('market_value', 0) for pos in st.session_state.positions.values()])
    available_cash = total_balance - positions_value
    daily_pnl = np.random.uniform(-2000, 3000)  # Simulé
    total_pnl = sum([pos.get('unrealized_pnl', 0) for pos in st.session_state.positions.values()])

    with col_met1:
        st.metric("Total Balance", f"${total_balance:,.2f}", f"{daily_pnl:+.2f}")

    with col_met2:
        st.metric("Available Cash", f"${available_cash:,.2f}")

    with col_met3:
        st.metric("Positions Value", f"${positions_value:,.2f}")

    with col_met4:
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("Unrealized PnL", f"${total_pnl:,.2f}", delta_color=pnl_color)

    with col_met5:
        num_positions = len(st.session_state.positions)
        st.metric("Open Positions", num_positions)

    # Graphiques de performance
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Graphique PnL en temps réel
        st.subheader("📈 Real-Time PnL")

        # Générer données PnL simulées
        if len(st.session_state.pnl_history) < 100:
            # Initialiser avec des données
            times = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
            base_pnl = 0
            pnl_data = []

            for time_point in times:
                base_pnl += np.random.normal(0, 50)
                pnl_data.append({
                    'time': time_point,
                    'pnl': base_pnl,
                    'balance': total_balance + base_pnl
                })

            st.session_state.pnl_history = pnl_data

        # Mise à jour en temps réel
        if st.session_state.trading_active:
            current_pnl = st.session_state.pnl_history[-1]['pnl'] + np.random.normal(0, 25)
            st.session_state.pnl_history.append({
                'time': datetime.now(),
                'pnl': current_pnl,
                'balance': total_balance + current_pnl
            })

            # Garder seulement les 100 derniers points
            if len(st.session_state.pnl_history) > 100:
                st.session_state.pnl_history = st.session_state.pnl_history[-100:]

        # Créer le graphique
        df_pnl = pd.DataFrame(st.session_state.pnl_history)

        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=df_pnl['time'],
            y=df_pnl['pnl'],
            mode='lines',
            name='PnL',
            line=dict(color='#00ff88' if df_pnl['pnl'].iloc[-1] >= 0 else '#ff6b6b', width=3)
        ))

        fig_pnl.update_layout(
            title="PnL Evolution (Real-Time)",
            xaxis_title="Time",
            yaxis_title="PnL ($)",
            template='plotly_dark',
            height=350
        )

        st.plotly_chart(fig_pnl, use_container_width=True)

    with col_chart2:
        # Market prices en temps réel
        st.subheader("📊 Market Prices")

        # Prix simulés pour instruments populaires
        instruments = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AAPL', 'GOOGL']
        prices = {
            'BTC/USDT': 67500 + np.random.uniform(-500, 500),
            'ETH/USDT': 2650 + np.random.uniform(-50, 50),
            'SOL/USDT': 145 + np.random.uniform(-5, 5),
            'AAPL': 175.50 + np.random.uniform(-2, 2),
            'GOOGL': 142.30 + np.random.uniform(-1.5, 1.5)
        }

        # Affichage des prix
        for instrument, price in prices.items():
            change_pct = np.random.uniform(-2, 2)
            change_color = "profit-positive" if change_pct >= 0 else "profit-negative"

            st.markdown(f"""
            <div class="position-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>{instrument}</strong>
                    <div>
                        <span class="market-price">${price:,.2f}</span>
                        <span class="{change_color}">({change_pct:+.2f}%)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Quick Actions
    st.subheader("⚡ Quick Actions")

    col_action1, col_action2, col_action3, col_action4 = st.columns(4)

    with col_action1:
        if st.button("🛒 Quick Buy BTC", use_container_width=True):
            st.success("🛒 Buy order placed for BTC!")

    with col_action2:
        if st.button("📉 Quick Sell ETH", use_container_width=True):
            st.success("📉 Sell order placed for ETH!")

    with col_action3:
        if st.button("🔄 Close All Positions", use_container_width=True):
            st.warning("🔄 All positions closed!")
            st.session_state.positions = {}

    with col_action4:
        if st.button("⏸️ Emergency Stop", use_container_width=True):
            st.error("🛑 Emergency stop activated!")
            st.session_state.trading_active = False

# ================== TAB 2: Order Management ==================
with tab2:
    st.header("💹 Order Management")

    col_order_left, col_order_right = st.columns([2, 1])

    with col_order_left:
        # Formulaire de placement d'ordre
        st.subheader("📝 Place New Order")

        with st.expander("🛒 Buy Order", expanded=True):
            col_buy1, col_buy2 = st.columns(2)

            with col_buy1:
                buy_symbol = st.selectbox(
                    "Symbol",
                    ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AAPL', 'GOOGL', 'MSFT'],
                    key="buy_symbol"
                )

                buy_order_type = st.selectbox(
                    "Order Type",
                    ['Market', 'Limit', 'Stop', 'Stop-Limit'],
                    key="buy_order_type"
                )

                buy_quantity = st.number_input(
                    "Quantity",
                    min_value=0.001,
                    value=0.1,
                    step=0.001,
                    format="%.3f",
                    key="buy_quantity"
                )

            with col_buy2:
                if buy_order_type in ['Limit', 'Stop-Limit']:
                    buy_price = st.number_input(
                        "Price",
                        min_value=0.01,
                        value=67500.0,
                        step=0.01,
                        key="buy_price"
                    )

                if buy_order_type in ['Stop', 'Stop-Limit']:
                    buy_stop_price = st.number_input(
                        "Stop Price",
                        min_value=0.01,
                        value=67000.0,
                        step=0.01,
                        key="buy_stop_price"
                    )

                # Calcul valeur estimée
                estimated_price = buy_price if buy_order_type == 'Limit' else prices.get(buy_symbol, 0)
                estimated_value = buy_quantity * estimated_price
                st.metric("Estimated Value", f"${estimated_value:,.2f}")

            if st.button("🛒 Place Buy Order", type="primary", use_container_width=True):
                order_id = f"BUY_{len(st.session_state.active_orders)+1:03d}"
                st.session_state.active_orders[order_id] = {
                    'id': order_id,
                    'symbol': buy_symbol,
                    'side': 'BUY',
                    'type': buy_order_type,
                    'quantity': buy_quantity,
                    'price': buy_price if buy_order_type in ['Limit', 'Stop-Limit'] else None,
                    'status': 'PENDING',
                    'timestamp': datetime.now()
                }
                st.success(f"✅ Buy order {order_id} placed successfully!")
                st.rerun()

        with st.expander("📉 Sell Order"):
            col_sell1, col_sell2 = st.columns(2)

            with col_sell1:
                sell_symbol = st.selectbox(
                    "Symbol",
                    ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AAPL', 'GOOGL', 'MSFT'],
                    key="sell_symbol"
                )

                sell_order_type = st.selectbox(
                    "Order Type",
                    ['Market', 'Limit', 'Stop', 'Stop-Limit'],
                    key="sell_order_type"
                )

                sell_quantity = st.number_input(
                    "Quantity",
                    min_value=0.001,
                    value=0.1,
                    step=0.001,
                    format="%.3f",
                    key="sell_quantity"
                )

            with col_sell2:
                if sell_order_type in ['Limit', 'Stop-Limit']:
                    sell_price = st.number_input(
                        "Price",
                        min_value=0.01,
                        value=67500.0,
                        step=0.01,
                        key="sell_price"
                    )

                if sell_order_type in ['Stop', 'Stop-Limit']:
                    sell_stop_price = st.number_input(
                        "Stop Price",
                        min_value=0.01,
                        value=68000.0,
                        step=0.01,
                        key="sell_stop_price"
                    )

                # Calcul valeur estimée
                estimated_price = sell_price if sell_order_type == 'Limit' else prices.get(sell_symbol, 0)
                estimated_value = sell_quantity * estimated_price
                st.metric("Estimated Value", f"${estimated_value:,.2f}")

            if st.button("📉 Place Sell Order", type="primary", use_container_width=True):
                order_id = f"SELL_{len(st.session_state.active_orders)+1:03d}"
                st.session_state.active_orders[order_id] = {
                    'id': order_id,
                    'symbol': sell_symbol,
                    'side': 'SELL',
                    'type': sell_order_type,
                    'quantity': sell_quantity,
                    'price': sell_price if sell_order_type in ['Limit', 'Stop-Limit'] else None,
                    'status': 'PENDING',
                    'timestamp': datetime.now()
                }
                st.success(f"✅ Sell order {order_id} placed successfully!")
                st.rerun()

    with col_order_right:
        # Ordres actifs
        st.subheader("📋 Active Orders")

        if not st.session_state.active_orders:
            st.info("No active orders")
        else:
            for order_id, order in st.session_state.active_orders.items():
                status_class = "status-pending" if order['status'] == 'PENDING' else "status-live"

                st.markdown(f"""
                <div class="order-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{order['symbol']}</strong><br>
                            <small>{order['side']} • {order['type']}</small>
                        </div>
                        <div style="text-align: right;">
                            <div>{order['quantity']}</div>
                            <div class="{status_class}">{order['status']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_cancel, col_modify = st.columns(2)
                with col_cancel:
                    if st.button(f"❌ Cancel", key=f"cancel_{order_id}"):
                        del st.session_state.active_orders[order_id]
                        st.success(f"Order {order_id} cancelled")
                        st.rerun()

                with col_modify:
                    if st.button(f"✏️ Modify", key=f"modify_{order_id}"):
                        st.info(f"Modify order {order_id}")

    # Historique des ordres
    st.subheader("📚 Order History")

    # Générer historique simulé
    order_history = []
    for i in range(10):
        order_history.append({
            'Order ID': f"ORD_{i+1:03d}",
            'Symbol': np.random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT']),
            'Side': np.random.choice(['BUY', 'SELL']),
            'Quantity': f"{np.random.uniform(0.01, 1.0):.3f}",
            'Price': f"${np.random.uniform(1000, 70000):.2f}",
            'Status': np.random.choice(['FILLED', 'CANCELLED', 'PARTIALLY_FILLED']),
            'Time': (datetime.now() - timedelta(hours=np.random.randint(1, 24))).strftime('%H:%M:%S')
        })

    df_history = pd.DataFrame(order_history)
    st.dataframe(df_history, use_container_width=True)

# ================== TAB 3: Positions ==================
with tab3:
    st.header("📈 Position Monitor")

    # Résumé des positions
    col_pos1, col_pos2, col_pos3, col_pos4 = st.columns(4)

    # Générer positions simulées si vides
    if not st.session_state.positions:
        st.session_state.positions = {
            'BTC/USDT': {
                'symbol': 'BTC/USDT',
                'quantity': 0.15,
                'avg_price': 66800.0,
                'current_price': 67500.0,
                'market_value': 0.15 * 67500.0,
                'unrealized_pnl': 0.15 * (67500.0 - 66800.0),
                'side': 'LONG'
            },
            'ETH/USDT': {
                'symbol': 'ETH/USDT',
                'quantity': 2.5,
                'avg_price': 2680.0,
                'current_price': 2650.0,
                'market_value': 2.5 * 2650.0,
                'unrealized_pnl': 2.5 * (2650.0 - 2680.0),
                'side': 'LONG'
            }
        }

    # Calculs positions
    total_positions = len(st.session_state.positions)
    total_market_value = sum([pos['market_value'] for pos in st.session_state.positions.values()])
    total_unrealized_pnl = sum([pos['unrealized_pnl'] for pos in st.session_state.positions.values()])
    long_positions = len([pos for pos in st.session_state.positions.values() if pos['side'] == 'LONG'])

    with col_pos1:
        st.metric("Total Positions", total_positions)

    with col_pos2:
        st.metric("Market Value", f"${total_market_value:,.2f}")

    with col_pos3:
        pnl_color = "normal" if total_unrealized_pnl >= 0 else "inverse"
        st.metric("Unrealized PnL", f"${total_unrealized_pnl:,.2f}", delta_color=pnl_color)

    with col_pos4:
        st.metric("Long Positions", f"{long_positions}/{total_positions}")

    # Détail des positions
    st.subheader("🎯 Position Details")

    if not st.session_state.positions:
        st.info("No open positions")
    else:
        for symbol, position in st.session_state.positions.items():
            pnl_class = "profit-positive" if position['unrealized_pnl'] >= 0 else "profit-negative"
            pnl_pct = (position['unrealized_pnl'] / (position['quantity'] * position['avg_price'])) * 100

            st.markdown(f"""
            <div class="position-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4>{position['symbol']} ({position['side']})</h4>
                        <div>Quantity: {position['quantity']}</div>
                        <div>Avg Price: ${position['avg_price']:,.2f}</div>
                    </div>
                    <div style="text-align: right;">
                        <div><strong>Current: ${position['current_price']:,.2f}</strong></div>
                        <div>Value: ${position['market_value']:,.2f}</div>
                        <div class="{pnl_class}">PnL: ${position['unrealized_pnl']:,.2f} ({pnl_pct:+.2f}%)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_close, col_reduce, col_hedge = st.columns(3)

            with col_close:
                if st.button(f"🔒 Close {symbol}", key=f"close_{symbol}"):
                    del st.session_state.positions[symbol]
                    st.success(f"Position {symbol} closed")
                    st.rerun()

            with col_reduce:
                if st.button(f"📉 Reduce 50%", key=f"reduce_{symbol}"):
                    st.session_state.positions[symbol]['quantity'] *= 0.5
                    st.session_state.positions[symbol]['market_value'] *= 0.5
                    st.session_state.positions[symbol]['unrealized_pnl'] *= 0.5
                    st.success(f"Position {symbol} reduced by 50%")
                    st.rerun()

            with col_hedge:
                if st.button(f"🛡️ Add Hedge", key=f"hedge_{symbol}"):
                    st.info(f"Hedge order placed for {symbol}")

    # Graphique allocation
    if st.session_state.positions:
        st.subheader("📊 Position Allocation")

        symbols = list(st.session_state.positions.keys())
        values = [pos['market_value'] for pos in st.session_state.positions.values()]

        fig_allocation = go.Figure(data=[go.Pie(
            labels=symbols,
            values=values,
            hole=0.4
        )])

        fig_allocation.update_layout(
            title="Portfolio Allocation by Market Value",
            template='plotly_dark',
            height=400
        )

        st.plotly_chart(fig_allocation, use_container_width=True)

# ================== TAB 4: Risk Monitor ==================
with tab4:
    st.header("⚠️ Risk Management Monitor")

    # Métriques de risque globales
    st.subheader("🎯 Global Risk Metrics")

    col_risk1, col_risk2, col_risk3, col_risk4 = st.columns(4)

    # Calculs de risque simulés
    portfolio_var = np.random.uniform(2000, 8000)  # VaR
    max_drawdown = np.random.uniform(5, 15)  # Max DD %
    leverage_ratio = total_market_value / st.session_state.trading_balance
    risk_score = min(100, max(0, 50 + np.random.uniform(-30, 30)))

    with col_risk1:
        var_color = "🟢" if portfolio_var < 5000 else "🟡" if portfolio_var < 7000 else "🔴"
        st.metric("Portfolio VaR (95%)", f"{var_color} ${portfolio_var:,.0f}")

    with col_risk2:
        dd_color = "🟢" if max_drawdown < 10 else "🟡" if max_drawdown < 15 else "🔴"
        st.metric("Max Drawdown", f"{dd_color} {max_drawdown:.1f}%")

    with col_risk3:
        leverage_color = "🟢" if leverage_ratio < 1.5 else "🟡" if leverage_ratio < 2.0 else "🔴"
        st.metric("Leverage Ratio", f"{leverage_color} {leverage_ratio:.2f}x")

    with col_risk4:
        if risk_score < 30:
            risk_color = "🟢"
            risk_level = "LOW"
        elif risk_score < 70:
            risk_color = "🟡"
            risk_level = "MEDIUM"
        else:
            risk_color = "🔴"
            risk_level = "HIGH"

        st.metric("Risk Score", f"{risk_color} {risk_score:.0f}/100")
        st.markdown(f"<div class='risk-gauge risk-{risk_level.lower()}'><strong>{risk_level} RISK</strong></div>",
                   unsafe_allow_html=True)

    # Limites de risque
    st.subheader("🚨 Risk Limits")

    col_limit1, col_limit2 = st.columns(2)

    with col_limit1:
        st.markdown("### Position Limits")

        # Limite par position
        max_position_size = st.slider("Max Position Size (%)", 0, 50, 25)
        max_daily_loss = st.slider("Max Daily Loss ($)", 1000, 20000, 5000)
        max_portfolio_var = st.slider("Max Portfolio VaR ($)", 3000, 15000, 8000)

        # Vérification des limites
        position_violations = []
        for symbol, pos in st.session_state.positions.items():
            position_pct = (pos['market_value'] / st.session_state.trading_balance) * 100
            if position_pct > max_position_size:
                position_violations.append(f"{symbol}: {position_pct:.1f}% > {max_position_size}%")

        if position_violations:
            st.error("🚨 Position Limit Violations:")
            for violation in position_violations:
                st.error(f"• {violation}")
        else:
            st.success("✅ All position limits respected")

    with col_limit2:
        st.markdown("### Risk Controls")

        # Contrôles automatiques
        auto_stop_loss = st.checkbox("Auto Stop-Loss", value=True)
        emergency_liquidation = st.checkbox("Emergency Liquidation", value=True)
        correlation_limits = st.checkbox("Correlation Limits", value=False)

        # Status des contrôles
        if auto_stop_loss:
            st.success("✅ Stop-loss automatique activé")

        if emergency_liquidation:
            st.success("✅ Liquidation d'urgence activée")

        if correlation_limits:
            st.info("📊 Limites de corrélation activées")

        # Actions d'urgence
        st.markdown("### 🚨 Emergency Actions")

        if st.button("🛑 Emergency Stop All", type="primary"):
            st.session_state.trading_active = False
            st.error("🛑 EMERGENCY STOP ACTIVATED!")

        if st.button("💰 Reduce All Positions 50%"):
            for symbol in st.session_state.positions:
                st.session_state.positions[symbol]['quantity'] *= 0.5
                st.session_state.positions[symbol]['market_value'] *= 0.5
            st.warning("📉 All positions reduced by 50%")
            st.rerun()

    # Graphique de risque temporel
    st.subheader("📈 Risk Evolution")

    # Générer données de risque temporelles
    risk_times = [datetime.now() - timedelta(minutes=i) for i in range(60, 0, -1)]
    risk_values = [50 + 20 * np.sin(i/10) + np.random.normal(0, 5) for i in range(60)]

    df_risk = pd.DataFrame({
        'time': risk_times,
        'risk_score': risk_values
    })

    fig_risk = go.Figure()

    # Zone de risque
    fig_risk.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Low Risk")
    fig_risk.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk")

    # Ligne de risque
    colors = ['green' if r < 30 else 'orange' if r < 70 else 'red' for r in risk_values]

    fig_risk.add_trace(go.Scatter(
        x=df_risk['time'],
        y=df_risk['risk_score'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='orange', width=3),
        marker=dict(size=6, color=colors)
    ))

    fig_risk.update_layout(
        title="Risk Score Evolution (Last Hour)",
        xaxis_title="Time",
        yaxis_title="Risk Score",
        template='plotly_dark',
        height=400,
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(fig_risk, use_container_width=True)

# ================== TAB 5: Alerts & News ==================
with tab5:
    st.header("📱 Alerts & Market News")

    col_alert1, col_alert2 = st.columns(2)

    with col_alert1:
        st.subheader("🚨 Active Alerts")

        # Alertes système simulées
        alerts = [
            {
                'type': 'WARNING',
                'message': 'BTC price approaching resistance at $68,000',
                'time': '14:23:15',
                'priority': 'MEDIUM'
            },
            {
                'type': 'INFO',
                'message': 'ETH volume spike detected (+45%)',
                'time': '14:20:32',
                'priority': 'LOW'
            },
            {
                'type': 'CRITICAL',
                'message': 'Portfolio VaR exceeded daily limit',
                'time': '14:15:08',
                'priority': 'HIGH'
            }
        ]

        for alert in alerts:
            alert_class = "status-error" if alert['priority'] == 'HIGH' else "status-pending" if alert['priority'] == 'MEDIUM' else "status-live"
            icon = "🚨" if alert['priority'] == 'HIGH' else "⚠️" if alert['priority'] == 'MEDIUM' else "ℹ️"

            st.markdown(f"""
            <div class="{alert_class}">
                <div style="display: flex; justify-content: between; align-items: center;">
                    <div>{icon} <strong>{alert['type']}</strong></div>
                    <div style="font-size: 0.8em;">{alert['time']}</div>
                </div>
                <div style="margin-top: 0.5rem;">{alert['message']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Configuration des alertes
        st.subheader("⚙️ Alert Settings")

        price_alerts = st.checkbox("Price Movement Alerts", value=True)
        volume_alerts = st.checkbox("Volume Spike Alerts", value=True)
        risk_alerts = st.checkbox("Risk Limit Alerts", value=True)
        news_alerts = st.checkbox("Market News Alerts", value=False)

        alert_threshold = st.slider("Price Alert Threshold (%)", 1, 10, 3)

    with col_alert2:
        st.subheader("📰 Market News Feed")

        # News simulées
        news_items = [
            {
                'title': 'Bitcoin ETF sees $2.1B inflow this week',
                'source': 'CoinDesk',
                'time': '5 min ago',
                'impact': 'BULLISH'
            },
            {
                'title': 'Federal Reserve hints at rate cut in December',
                'source': 'Reuters',
                'time': '12 min ago',
                'impact': 'BULLISH'
            },
            {
                'title': 'Ethereum network congestion causes gas spike',
                'source': 'The Block',
                'time': '18 min ago',
                'impact': 'BEARISH'
            },
            {
                'title': 'Major DeFi protocol announces v3 upgrade',
                'source': 'DeFi Pulse',
                'time': '25 min ago',
                'impact': 'NEUTRAL'
            }
        ]

        for news in news_items:
            impact_color = "profit-positive" if news['impact'] == 'BULLISH' else "profit-negative" if news['impact'] == 'BEARISH' else "profit-neutral"
            impact_icon = "📈" if news['impact'] == 'BULLISH' else "📉" if news['impact'] == 'BEARISH' else "➡️"

            st.markdown(f"""
            <div class="position-card">
                <div style="display: flex; justify-content: between; align-items: flex-start;">
                    <div style="flex: 1;">
                        <strong>{news['title']}</strong><br>
                        <small>{news['source']} • {news['time']}</small>
                    </div>
                    <div class="{impact_color}" style="margin-left: 1rem;">
                        {impact_icon} {news['impact']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Sentiment de marché
        st.subheader("😊 Market Sentiment")

        sentiment_score = 72  # Sur 100
        sentiment_color = "profit-positive" if sentiment_score > 60 else "profit-negative" if sentiment_score < 40 else "profit-neutral"

        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div class="market-price {sentiment_color}">{sentiment_score}/100</div>
            <div><strong>OPTIMISTIC</strong></div>
            <div style="margin-top: 1rem;">
                <span style="color: #00ff88;">Fear: 28</span> •
                <span style="color: #ffc107;">Greed: 72</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================== TAB 6: Settings ==================
with tab6:
    st.header("⚙️ Trading Settings")

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        st.subheader("🔧 Trading Configuration")

        # Paramètres de trading
        default_order_size = st.number_input("Default Order Size", min_value=0.001, value=0.1, step=0.001)
        default_leverage = st.slider("Default Leverage", 1.0, 5.0, 1.0, 0.1)
        auto_confirm_orders = st.checkbox("Auto-confirm orders", value=False)
        enable_stop_loss = st.checkbox("Enable Stop-Loss", value=True)

        if enable_stop_loss:
            default_stop_loss = st.slider("Default Stop-Loss (%)", 1, 20, 5)

        enable_take_profit = st.checkbox("Enable Take-Profit", value=True)

        if enable_take_profit:
            default_take_profit = st.slider("Default Take-Profit (%)", 5, 50, 15)

        st.subheader("📊 Display Settings")

        refresh_rate = st.selectbox("Refresh Rate", [1, 2, 5, 10, 30], index=2)
        show_advanced_charts = st.checkbox("Show Advanced Charts", value=True)
        dark_mode = st.checkbox("Dark Mode", value=True)

    with col_set2:
        st.subheader("🔐 Security Settings")

        two_factor_auth = st.checkbox("Two-Factor Authentication", value=True)
        require_confirmation = st.checkbox("Require Order Confirmation", value=True)
        ip_whitelist = st.checkbox("IP Whitelist", value=False)

        st.subheader("💾 Data & Backup")

        auto_backup = st.checkbox("Auto-backup trades", value=True)
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])

        if st.button("📥 Export Trading Data"):
            st.success("Trading data exported successfully!")

        if st.button("🔄 Backup Settings"):
            st.success("Settings backed up successfully!")

        st.subheader("🚨 Risk Limits")

        max_daily_trades = st.number_input("Max Daily Trades", min_value=1, value=50)
        max_position_count = st.number_input("Max Open Positions", min_value=1, value=10)
        max_exposure_per_symbol = st.slider("Max Exposure per Symbol (%)", 5, 50, 25)

        # Sauvegarde des paramètres
        if st.button("💾 Save Settings", type="primary"):
            st.success("✅ Settings saved successfully!")

# Auto-refresh pour les données en temps réel
if st.session_state.trading_active:
    time.sleep(1)
    st.rerun()

# Sidebar avec informations
with st.sidebar:
    st.markdown("### 📈 Live Trading")

    # Status en temps réel
    status_color = "🟢" if st.session_state.trading_active else "🔴"
    st.markdown(f"**Status**: {status_color} {'LIVE' if st.session_state.trading_active else 'STOPPED'}")

    st.metric("Balance", f"${st.session_state.trading_balance:,.2f}")
    st.metric("Open Positions", len(st.session_state.positions))
    st.metric("Active Orders", len(st.session_state.active_orders))

    st.markdown("### ⚡ Quick Stats")
    daily_volume = np.random.uniform(50000, 200000)
    st.metric("Daily Volume", f"${daily_volume:,.0f}")

    trades_today = np.random.randint(15, 45)
    st.metric("Trades Today", trades_today)

    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [📊 Portfolio Analysis](#)
    - [📈 Advanced Charts](#)
    - [⚠️ Risk Reports](#)
    - [💹 Trading History](#)
    """)

    # Actions rapides
    st.markdown("### ⚡ Emergency")
    if st.button("🛑 STOP ALL", use_container_width=True):
        st.session_state.trading_active = False
        st.error("🛑 ALL TRADING STOPPED!")
        st.rerun()