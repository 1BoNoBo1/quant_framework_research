"""
📋 Order Manager Component
Gestion complète des ordres trading - Placement, suivi, modification
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal
from enum import Enum
import uuid


class OrderType(Enum):
    """Types d'ordres supportés."""
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "Stop Limit"
    TRAILING_STOP = "Trailing Stop"


class OrderSide(Enum):
    """Côtés des ordres."""
    BUY = "Buy"
    SELL = "Sell"


class OrderStatus(Enum):
    """Statuts des ordres."""
    PENDING = "Pending"
    PARTIAL = "Partial"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


class OrderManager:
    """Gestionnaire d'ordres trading complet."""

    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        """Initialise les données de session pour la gestion d'ordres."""
        if 'orders' not in st.session_state:
            st.session_state.orders = []
            self._create_sample_orders()

        if 'order_form' not in st.session_state:
            st.session_state.order_form = {
                'symbol': 'BTC/USDT',
                'side': OrderSide.BUY.value,
                'type': OrderType.MARKET.value,
                'quantity': 0.01,
                'price': None,
                'stop_price': None,
                'take_profit': None,
                'stop_loss': None
            }

        if 'quick_trade_amounts' not in st.session_state:
            st.session_state.quick_trade_amounts = [10, 25, 50, 100, 250, 500, 1000]

    def _create_sample_orders(self):
        """Crée des ordres d'exemple pour la démonstration."""
        sample_orders = [
            {
                'id': str(uuid.uuid4())[:8],
                'symbol': 'BTC/USDT',
                'side': OrderSide.BUY.value,
                'type': OrderType.LIMIT.value,
                'quantity': 0.5,
                'price': 42800.0,
                'filled_quantity': 0.0,
                'status': OrderStatus.PENDING.value,
                'created_at': datetime.now() - timedelta(minutes=15),
                'updated_at': datetime.now() - timedelta(minutes=15)
            },
            {
                'id': str(uuid.uuid4())[:8],
                'symbol': 'ETH/USDT',
                'side': OrderSide.SELL.value,
                'type': OrderType.STOP.value,
                'quantity': 2.0,
                'price': 2600.0,
                'stop_price': 2610.0,
                'filled_quantity': 1.2,
                'status': OrderStatus.PARTIAL.value,
                'created_at': datetime.now() - timedelta(hours=2),
                'updated_at': datetime.now() - timedelta(minutes=5)
            },
            {
                'id': str(uuid.uuid4())[:8],
                'symbol': 'BNB/USDT',
                'side': OrderSide.BUY.value,
                'type': OrderType.MARKET.value,
                'quantity': 10.0,
                'price': 234.5,
                'filled_quantity': 10.0,
                'status': OrderStatus.FILLED.value,
                'created_at': datetime.now() - timedelta(hours=1),
                'updated_at': datetime.now() - timedelta(hours=1)
            }
        ]
        st.session_state.orders.extend(sample_orders)

    def render(self):
        """Rendu principal du gestionnaire d'ordres."""
        st.header("📋 Order Manager")

        # Onglets principaux
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚀 Nouveau Ordre",
            "📊 Ordres Actifs",
            "📈 Quick Trading",
            "📋 Historique"
        ])

        with tab1:
            self._render_order_form()

        with tab2:
            self._render_active_orders()

        with tab3:
            self._render_quick_trading()

        with tab4:
            self._render_order_history()

    def _render_order_form(self):
        """Formulaire de création d'ordre avancé."""
        st.subheader("🚀 Créer Nouvel Ordre")

        # Récupération des prix en temps réel depuis le dashboard
        if 'price_data' in st.session_state:
            current_prices = {
                symbol: data['current_price']
                for symbol, data in st.session_state.price_data.items()
            }
        else:
            current_prices = {'BTC/USDT': 43250.0, 'ETH/USDT': 2650.0}

        # Layout en colonnes
        col1, col2 = st.columns(2)

        with col1:
            # Sélection symbol
            symbol = st.selectbox(
                "💱 Symbol",
                options=list(current_prices.keys()),
                index=list(current_prices.keys()).index(st.session_state.order_form['symbol'])
            )
            st.session_state.order_form['symbol'] = symbol

            # Prix actuel
            current_price = current_prices.get(symbol, 0)
            st.info(f"💰 Prix actuel: ${current_price:,.2f}")

            # Side et Type
            side = st.radio(
                "📈📉 Côté",
                options=[OrderSide.BUY.value, OrderSide.SELL.value],
                horizontal=True,
                index=0 if st.session_state.order_form['side'] == OrderSide.BUY.value else 1
            )
            st.session_state.order_form['side'] = side

            order_type = st.selectbox(
                "🎯 Type d'Ordre",
                options=[ot.value for ot in OrderType],
                index=[ot.value for ot in OrderType].index(st.session_state.order_form['type'])
            )
            st.session_state.order_form['type'] = order_type

        with col2:
            # Quantité
            quantity = st.number_input(
                "📊 Quantité",
                min_value=0.001,
                max_value=1000.0,
                value=st.session_state.order_form['quantity'],
                step=0.001,
                format="%.6f"
            )
            st.session_state.order_form['quantity'] = quantity

            # Valeur notionnelle
            notional_value = quantity * current_price
            st.caption(f"💵 Valeur: ${notional_value:,.2f}")

            # Prix et stops selon le type d'ordre
            if order_type in [OrderType.LIMIT.value, OrderType.STOP_LIMIT.value]:
                price = st.number_input(
                    "💰 Prix Limite",
                    min_value=0.01,
                    value=current_price,
                    step=0.01,
                    format="%.2f"
                )
                st.session_state.order_form['price'] = price

            if order_type in [OrderType.STOP.value, OrderType.STOP_LIMIT.value]:
                stop_price = st.number_input(
                    "🛑 Prix Stop",
                    min_value=0.01,
                    value=current_price * 0.95 if side == OrderSide.BUY.value else current_price * 1.05,
                    step=0.01,
                    format="%.2f"
                )
                st.session_state.order_form['stop_price'] = stop_price

        # Section Risk Management
        st.markdown("---")
        st.subheader("⚖️ Risk Management")

        col3, col4 = st.columns(2)
        with col3:
            use_take_profit = st.checkbox("✅ Take Profit")
            if use_take_profit:
                take_profit = st.number_input(
                    "🎯 Take Profit Price",
                    min_value=0.01,
                    value=current_price * 1.05 if side == OrderSide.BUY.value else current_price * 0.95,
                    step=0.01,
                    format="%.2f"
                )
                st.session_state.order_form['take_profit'] = take_profit

        with col4:
            use_stop_loss = st.checkbox("🛑 Stop Loss")
            if use_stop_loss:
                stop_loss = st.number_input(
                    "⛔ Stop Loss Price",
                    min_value=0.01,
                    value=current_price * 0.95 if side == OrderSide.BUY.value else current_price * 1.05,
                    step=0.01,
                    format="%.2f"
                )
                st.session_state.order_form['stop_loss'] = stop_loss

        # Calcul des métriques de risque
        if 'take_profit' in st.session_state.order_form and 'stop_loss' in st.session_state.order_form:
            tp_price = st.session_state.order_form.get('take_profit', current_price)
            sl_price = st.session_state.order_form.get('stop_loss', current_price)

            if side == OrderSide.BUY.value:
                potential_profit = (tp_price - current_price) / current_price * 100
                potential_loss = (current_price - sl_price) / current_price * 100
            else:
                potential_profit = (current_price - tp_price) / current_price * 100
                potential_loss = (sl_price - current_price) / current_price * 100

            risk_reward = potential_profit / potential_loss if potential_loss > 0 else 0

            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric("📈 Profit Potentiel", f"{potential_profit:.2f}%")
            with col6:
                st.metric("📉 Perte Potentielle", f"{potential_loss:.2f}%")
            with col7:
                st.metric("⚖️ Risk/Reward", f"{risk_reward:.2f}")

        # Boutons d'action
        st.markdown("---")
        col8, col9, col10 = st.columns(3)

        with col8:
            if st.button("🚀 Placer Ordre", type="primary", use_container_width=True):
                self._place_order()

        with col9:
            if st.button("🔄 Reset Form", use_container_width=True):
                st.session_state.order_form = {
                    'symbol': 'BTC/USDT',
                    'side': OrderSide.BUY.value,
                    'type': OrderType.MARKET.value,
                    'quantity': 0.01,
                    'price': None,
                    'stop_price': None,
                    'take_profit': None,
                    'stop_loss': None
                }
                st.rerun()

        with col10:
            if st.button("📊 Calculateur Position", use_container_width=True):
                self._show_position_calculator()

    def _render_active_orders(self):
        """Affichage et gestion des ordres actifs."""
        st.subheader("📊 Ordres Actifs")

        active_orders = [
            order for order in st.session_state.orders
            if order['status'] in [OrderStatus.PENDING.value, OrderStatus.PARTIAL.value]
        ]

        if not active_orders:
            st.info("Aucun ordre actif")
            return

        # Filtrages et tri
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_symbol = st.selectbox(
                "Filtrer par Symbol",
                options=['Tous'] + list(set([order['symbol'] for order in active_orders]))
            )
        with col2:
            filter_side = st.selectbox(
                "Filtrer par Side",
                options=['Tous', OrderSide.BUY.value, OrderSide.SELL.value]
            )
        with col3:
            sort_by = st.selectbox(
                "Trier par",
                options=['created_at', 'symbol', 'quantity', 'status']
            )

        # Application des filtres
        filtered_orders = active_orders
        if filter_symbol != 'Tous':
            filtered_orders = [o for o in filtered_orders if o['symbol'] == filter_symbol]
        if filter_side != 'Tous':
            filtered_orders = [o for o in filtered_orders if o['side'] == filter_side]

        # Tri
        filtered_orders = sorted(filtered_orders, key=lambda x: x[sort_by], reverse=True)

        # Tableau des ordres
        for i, order in enumerate(filtered_orders):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])

                with col1:
                    st.write(f"**{order['symbol']}**")
                    st.caption(f"ID: {order['id']}")

                with col2:
                    color = '🟢' if order['side'] == OrderSide.BUY.value else '🔴'
                    st.write(f"{color} {order['side']}")
                    st.caption(f"{order['type']}")

                with col3:
                    filled_pct = (order['filled_quantity'] / order['quantity']) * 100
                    st.write(f"{order['quantity']:.3f}")
                    st.caption(f"Exécuté: {filled_pct:.1f}%")

                with col4:
                    if order.get('price'):
                        st.write(f"${order['price']:,.2f}")
                    else:
                        st.write("Market")

                    status_color = {
                        OrderStatus.PENDING.value: '🟡',
                        OrderStatus.PARTIAL.value: '🟠',
                        OrderStatus.FILLED.value: '🟢',
                        OrderStatus.CANCELLED.value: '⚫',
                        OrderStatus.REJECTED.value: '🔴'
                    }
                    st.caption(f"{status_color.get(order['status'], '❓')} {order['status']}")

                with col5:
                    col5a, col5b, col5c = st.columns(3)
                    with col5a:
                        if st.button("✏️", key=f"edit_{order['id']}", help="Modifier"):
                            self._edit_order(order)
                    with col5b:
                        if st.button("❌", key=f"cancel_{order['id']}", help="Annuler"):
                            self._cancel_order(order['id'])
                    with col5c:
                        if st.button("📊", key=f"details_{order['id']}", help="Détails"):
                            self._show_order_details(order)

                st.markdown("---")

    def _render_quick_trading(self):
        """Interface de trading rapide."""
        st.subheader("📈 Quick Trading")

        # Sélection symbol pour quick trade
        if 'price_data' in st.session_state:
            symbols = list(st.session_state.price_data.keys())
            current_prices = st.session_state.price_data
        else:
            symbols = ['BTC/USDT', 'ETH/USDT']
            current_prices = {'BTC/USDT': {'current_price': 43250}, 'ETH/USDT': {'current_price': 2650}}

        selected_symbol = st.selectbox("💱 Symbol Quick Trade", symbols)
        current_price = current_prices[selected_symbol]['current_price']

        st.info(f"💰 {selected_symbol}: ${current_price:,.2f}")

        # Montants prédéfinis
        st.subheader("💵 Montants Prédéfinis")

        amounts = st.session_state.quick_trade_amounts
        cols = st.columns(len(amounts))

        for i, amount in enumerate(amounts):
            with cols[i]:
                quantity = amount / current_price
                if st.button(f"${amount}", key=f"quick_buy_{amount}", use_container_width=True):
                    self._quick_trade(selected_symbol, OrderSide.BUY, quantity, amount)
                st.caption(f"{quantity:.6f}")

        st.markdown("---")

        # Quick Sell (positions existantes)
        st.subheader("💰 Quick Sell")

        # Simulation positions pour quick sell
        positions = self._get_user_positions(selected_symbol)

        if positions:
            for position in positions:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"Position: {position['quantity']:.6f} {selected_symbol}")
                with col2:
                    st.write(f"PnL: {position['pnl']:+.2f}%")
                with col3:
                    if st.button("🔴 Vendre", key=f"quick_sell_{position['id']}"):
                        self._quick_trade(selected_symbol, OrderSide.SELL, position['quantity'], position['value'])
        else:
            st.info(f"Aucune position en {selected_symbol}")

        # Configuration quick trade
        st.markdown("---")
        st.subheader("⚙️ Configuration Quick Trade")

        col1, col2 = st.columns(2)
        with col1:
            auto_stop_loss = st.checkbox("🛑 Stop Loss Auto (-5%)")
            auto_take_profit = st.checkbox("✅ Take Profit Auto (+10%)")

        with col2:
            if auto_stop_loss:
                sl_pct = st.slider("Stop Loss %", 1, 20, 5)
            if auto_take_profit:
                tp_pct = st.slider("Take Profit %", 5, 50, 10)

    def _render_order_history(self):
        """Historique complet des ordres."""
        st.subheader("📋 Historique des Ordres")

        # Filtres temporels
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Date début", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("Date fin", datetime.now())
        with col3:
            status_filter = st.multiselect(
                "Statuts",
                options=[status.value for status in OrderStatus],
                default=[OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]
            )

        # Filtrage des ordres
        filtered_orders = [
            order for order in st.session_state.orders
            if order['status'] in status_filter
            and start_date <= order['created_at'].date() <= end_date
        ]

        if not filtered_orders:
            st.info("Aucun ordre dans les critères sélectionnés")
            return

        # DataFrame pour l'historique
        df = pd.DataFrame(filtered_orders)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['execution_rate'] = (df['filled_quantity'] / df['quantity'] * 100).round(2)

        # Métriques de l'historique
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Ordres", len(filtered_orders))
        with col2:
            filled_orders = len([o for o in filtered_orders if o['status'] == OrderStatus.FILLED.value])
            st.metric("Ordres Exécutés", filled_orders)
        with col3:
            avg_execution = df['execution_rate'].mean()
            st.metric("Taux Exécution Moyen", f"{avg_execution:.1f}%")
        with col4:
            total_volume = sum([o['quantity'] * o.get('price', 0) for o in filtered_orders])
            st.metric("Volume Total", f"${total_volume:,.0f}")

        # Tableau historique
        display_df = df[[
            'created_at', 'symbol', 'side', 'type',
            'quantity', 'price', 'filled_quantity', 'status'
        ]].copy()

        display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = [
            'Date', 'Symbol', 'Side', 'Type',
            'Quantité', 'Prix', 'Exécuté', 'Statut'
        ]

        st.dataframe(display_df, use_container_width=True)

        # Export des données
        if st.button("📥 Exporter CSV"):
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Télécharger",
                data=csv,
                file_name=f"order_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    def _place_order(self):
        """Place un nouvel ordre."""
        form_data = st.session_state.order_form

        new_order = {
            'id': str(uuid.uuid4())[:8],
            'symbol': form_data['symbol'],
            'side': form_data['side'],
            'type': form_data['type'],
            'quantity': form_data['quantity'],
            'price': form_data.get('price'),
            'stop_price': form_data.get('stop_price'),
            'filled_quantity': 0.0,
            'status': OrderStatus.PENDING.value,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }

        # Simulation d'exécution pour ordres market
        if form_data['type'] == OrderType.MARKET.value:
            new_order['status'] = OrderStatus.FILLED.value
            new_order['filled_quantity'] = form_data['quantity']
            if 'price_data' in st.session_state:
                new_order['price'] = st.session_state.price_data[form_data['symbol']]['current_price']

        st.session_state.orders.append(new_order)

        st.success(f"✅ Ordre {new_order['id']} créé avec succès!")
        st.balloons()

    def _quick_trade(self, symbol: str, side: OrderSide, quantity: float, value: float):
        """Exécute un trade rapide."""
        quick_order = {
            'id': str(uuid.uuid4())[:8],
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.MARKET.value,
            'quantity': quantity,
            'price': value / quantity,
            'filled_quantity': quantity,
            'status': OrderStatus.FILLED.value,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }

        st.session_state.orders.append(quick_order)

        action = "Achat" if side == OrderSide.BUY else "Vente"
        st.success(f"✅ {action} rapide exécuté: {quantity:.6f} {symbol} pour ${value:.2f}")

    def _cancel_order(self, order_id: str):
        """Annule un ordre."""
        for order in st.session_state.orders:
            if order['id'] == order_id:
                order['status'] = OrderStatus.CANCELLED.value
                order['updated_at'] = datetime.now()
                st.success(f"❌ Ordre {order_id} annulé")
                st.rerun()
                break

    def _edit_order(self, order: Dict):
        """Ouvre l'interface d'édition d'ordre."""
        st.session_state.editing_order = order
        st.info(f"✏️ Édition de l'ordre {order['id']} - Utilisez le formulaire ci-dessus")

    def _show_order_details(self, order: Dict):
        """Affiche les détails d'un ordre."""
        with st.expander(f"📊 Détails Ordre {order['id']}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Symbol:** {order['symbol']}")
                st.write(f"**Side:** {order['side']}")
                st.write(f"**Type:** {order['type']}")
                st.write(f"**Statut:** {order['status']}")
            with col2:
                st.write(f"**Quantité:** {order['quantity']}")
                st.write(f"**Prix:** ${order.get('price', 'N/A')}")
                st.write(f"**Exécuté:** {order['filled_quantity']}")
                st.write(f"**Créé:** {order['created_at'].strftime('%Y-%m-%d %H:%M')}")

    def _show_position_calculator(self):
        """Affiche le calculateur de position."""
        with st.expander("📊 Calculateur de Position", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                account_balance = st.number_input("💰 Balance Compte", value=100000.0)
                risk_percentage = st.slider("⚖️ Risque par Trade (%)", 0.5, 10.0, 2.0, 0.1)
                entry_price = st.number_input("📈 Prix d'Entrée", value=43250.0)

            with col2:
                stop_loss_price = st.number_input("🛑 Prix Stop Loss", value=41000.0)
                risk_per_unit = abs(entry_price - stop_loss_price)
                max_risk_amount = account_balance * (risk_percentage / 100)
                position_size = max_risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                st.metric("💵 Risque Maximum", f"${max_risk_amount:.2f}")
                st.metric("📊 Taille Position", f"{position_size:.6f}")
                st.metric("💰 Valeur Position", f"${position_size * entry_price:.2f}")

    def _get_user_positions(self, symbol: str) -> List[Dict]:
        """Retourne les positions simulées de l'utilisateur."""
        # Simulation de positions basée sur les ordres remplis
        filled_orders = [
            order for order in st.session_state.orders
            if order['status'] == OrderStatus.FILLED.value and order['symbol'] == symbol
        ]

        if not filled_orders:
            return []

        # Calcul position nette
        total_bought = sum([o['filled_quantity'] for o in filled_orders if o['side'] == OrderSide.BUY.value])
        total_sold = sum([o['filled_quantity'] for o in filled_orders if o['side'] == OrderSide.SELL.value])
        net_position = total_bought - total_sold

        if net_position <= 0:
            return []

        # Prix moyen d'achat
        total_cost = sum([
            o['filled_quantity'] * o['price']
            for o in filled_orders
            if o['side'] == OrderSide.BUY.value and o.get('price')
        ])
        avg_price = total_cost / total_bought if total_bought > 0 else 0

        # PnL
        current_price = st.session_state.price_data.get(symbol, {}).get('current_price', avg_price)
        pnl_pct = (current_price - avg_price) / avg_price * 100 if avg_price > 0 else 0

        return [{
            'id': 'position_1',
            'quantity': net_position,
            'avg_price': avg_price,
            'current_price': current_price,
            'value': net_position * current_price,
            'pnl': pnl_pct
        }]


# Instance globale pour utilisation dans les pages
order_manager = OrderManager()