"""
Table Components for QFrame Streamlit App
=========================================

Composants de tableaux réutilisables pour l'interface Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime


def display_portfolio_table(portfolios: List[Dict]) -> None:
    """
    Affiche un tableau des portfolios
    """
    if not portfolios:
        st.info("No portfolios found")
        return

    df = pd.DataFrame(portfolios)

    # Formatage des colonnes
    if 'total_value' in df.columns:
        df['total_value'] = df['total_value'].apply(lambda x: f"${x:,.2f}")

    if 'initial_capital' in df.columns:
        df['initial_capital'] = df['initial_capital'].apply(lambda x: f"${x:,.2f}")

    if 'pnl' in df.columns:
        df['pnl'] = df['pnl'].apply(lambda x: f"${x:,.2f}")

    if 'pnl_percentage' in df.columns:
        df['pnl_percentage'] = df['pnl_percentage'].apply(lambda x: f"{x:+.2%}")

    # Renommage des colonnes
    column_mapping = {
        'id': 'ID',
        'name': 'Name',
        'total_value': 'Total Value',
        'initial_capital': 'Initial Capital',
        'pnl': 'P&L',
        'pnl_percentage': 'P&L %',
        'status': 'Status',
        'created_at': 'Created'
    }

    df = df.rename(columns=column_mapping)

    # Sélection des colonnes à afficher
    display_columns = [col for col in column_mapping.values() if col in df.columns]
    df_display = df[display_columns]

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )


def display_positions_table(positions: List[Dict]) -> None:
    """
    Affiche un tableau des positions
    """
    if not positions:
        st.info("No positions found")
        return

    df = pd.DataFrame(positions)

    # Formatage des colonnes
    if 'quantity' in df.columns:
        df['quantity'] = df['quantity'].apply(lambda x: f"{x:.6f}")

    if 'average_price' in df.columns:
        df['average_price'] = df['average_price'].apply(lambda x: f"${x:,.2f}")

    if 'current_price' in df.columns:
        df['current_price'] = df['current_price'].apply(lambda x: f"${x:,.2f}")

    if 'market_value' in df.columns:
        df['market_value'] = df['market_value'].apply(lambda x: f"${x:,.2f}")

    if 'unrealized_pnl' in df.columns:
        df['unrealized_pnl'] = df['unrealized_pnl'].apply(lambda x: f"${x:+,.2f}")

    if 'unrealized_pnl_percentage' in df.columns:
        df['unrealized_pnl_percentage'] = df['unrealized_pnl_percentage'].apply(lambda x: f"{x:+.2%}")

    # Renommage des colonnes
    column_mapping = {
        'symbol': 'Symbol',
        'quantity': 'Quantity',
        'average_price': 'Avg Price',
        'current_price': 'Current Price',
        'market_value': 'Market Value',
        'unrealized_pnl': 'Unrealized P&L',
        'unrealized_pnl_percentage': 'Unrealized P&L %'
    }

    df = df.rename(columns=column_mapping)

    # Sélection des colonnes à afficher
    display_columns = [col for col in column_mapping.values() if col in df.columns]
    df_display = df[display_columns]

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )


def display_orders_table(orders: List[Dict], show_actions: bool = False) -> None:
    """
    Affiche un tableau des ordres
    """
    if not orders:
        st.info("No orders found")
        return

    df = pd.DataFrame(orders)

    # Formatage des colonnes
    if 'quantity' in df.columns:
        df['quantity'] = df['quantity'].apply(lambda x: f"{x:.6f}")

    if 'price' in df.columns:
        df['price'] = df['price'].apply(lambda x: f"${x:,.2f}" if x else "Market")

    if 'filled_quantity' in df.columns:
        df['filled_quantity'] = df['filled_quantity'].apply(lambda x: f"{x:.6f}")

    if 'average_fill_price' in df.columns:
        df['average_fill_price'] = df['average_fill_price'].apply(lambda x: f"${x:,.2f}" if x else "-")

    # Formatage des timestamps
    timestamp_columns = ['created_time', 'updated_time']
    for col in timestamp_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Renommage des colonnes
    column_mapping = {
        'id': 'ID',
        'symbol': 'Symbol',
        'side': 'Side',
        'order_type': 'Type',
        'quantity': 'Quantity',
        'price': 'Price',
        'filled_quantity': 'Filled',
        'average_fill_price': 'Avg Fill Price',
        'status': 'Status',
        'created_time': 'Created',
        'updated_time': 'Updated'
    }

    df = df.rename(columns=column_mapping)

    # Sélection des colonnes à afficher
    display_columns = [col for col in column_mapping.values() if col in df.columns]
    df_display = df[display_columns]

    # Couleurs pour les différents statuts
    def color_status(val):
        if val == 'filled':
            return 'color: green'
        elif val == 'cancelled':
            return 'color: red'
        elif val == 'pending':
            return 'color: orange'
        return ''

    if 'Status' in df_display.columns:
        df_styled = df_display.style.applymap(color_status, subset=['Status'])
        st.dataframe(df_styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Actions sur les ordres
    if show_actions and not df.empty:
        st.subheader("Order Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            order_ids = df['ID'].tolist() if 'ID' in df.columns else []
            selected_order = st.selectbox("Select Order", order_ids)

        with col2:
            if st.button("Cancel Order", type="secondary"):
                if selected_order:
                    st.info(f"Cancelling order {selected_order}...")
                    # TODO: Implémenter l'annulation d'ordre via API

        with col3:
            if st.button("View Details", type="secondary"):
                if selected_order:
                    st.info(f"Viewing details for order {selected_order}...")
                    # TODO: Afficher les détails de l'ordre


def display_strategies_table(strategies: List[Dict], show_controls: bool = False) -> None:
    """
    Affiche un tableau des stratégies
    """
    if not strategies:
        st.info("No strategies found")
        return

    df = pd.DataFrame(strategies)

    # Formatage des colonnes
    if 'total_return' in df.columns:
        df['total_return'] = df['total_return'].apply(lambda x: f"{x:+.2%}")

    if 'sharpe_ratio' in df.columns:
        df['sharpe_ratio'] = df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")

    if 'max_drawdown' in df.columns:
        df['max_drawdown'] = df['max_drawdown'].apply(lambda x: f"{x:.2%}")

    # Formatage des timestamps
    timestamp_columns = ['created_at', 'updated_at', 'last_signal_time']
    for col in timestamp_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Renommage des colonnes
    column_mapping = {
        'id': 'ID',
        'name': 'Name',
        'status': 'Status',
        'total_return': 'Total Return',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'total_trades': 'Total Trades',
        'win_rate': 'Win Rate',
        'last_signal_time': 'Last Signal'
    }

    df = df.rename(columns=column_mapping)

    # Sélection des colonnes à afficher
    display_columns = [col for col in column_mapping.values() if col in df.columns]
    df_display = df[display_columns]

    # Couleurs pour les statuts
    def color_status(val):
        if val == 'active':
            return 'color: green'
        elif val == 'stopped':
            return 'color: red'
        elif val == 'paused':
            return 'color: orange'
        return ''

    if 'Status' in df_display.columns:
        df_styled = df_display.style.applymap(color_status, subset=['Status'])
        st.dataframe(df_styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Contrôles des stratégies
    if show_controls and not df.empty:
        st.subheader("Strategy Controls")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            strategy_ids = df['ID'].tolist() if 'ID' in df.columns else []
            selected_strategy = st.selectbox("Select Strategy", strategy_ids)

        with col2:
            if st.button("Start", type="primary"):
                if selected_strategy:
                    st.success(f"Starting strategy {selected_strategy}...")
                    # TODO: Implémenter le démarrage via API

        with col3:
            if st.button("Stop", type="secondary"):
                if selected_strategy:
                    st.warning(f"Stopping strategy {selected_strategy}...")
                    # TODO: Implémenter l'arrêt via API

        with col4:
            if st.button("Configure", type="secondary"):
                if selected_strategy:
                    st.info(f"Configuring strategy {selected_strategy}...")
                    # TODO: Afficher l'interface de configuration


def display_alerts_table(alerts: List[Dict]) -> None:
    """
    Affiche un tableau des alertes
    """
    if not alerts:
        st.info("No alerts found")
        return

    df = pd.DataFrame(alerts)

    # Formatage des timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Renommage des colonnes
    column_mapping = {
        'id': 'ID',
        'severity': 'Severity',
        'title': 'Title',
        'message': 'Message',
        'timestamp': 'Time',
        'acknowledged': 'Ack',
        'source': 'Source'
    }

    df = df.rename(columns=column_mapping)

    # Sélection des colonnes à afficher
    display_columns = [col for col in column_mapping.values() if col in df.columns]
    df_display = df[display_columns]

    # Couleurs pour les niveaux de sévérité
    def color_severity(val):
        if val == 'critical':
            return 'color: red; font-weight: bold'
        elif val == 'warning':
            return 'color: orange; font-weight: bold'
        elif val == 'info':
            return 'color: blue'
        return ''

    if 'Severity' in df_display.columns:
        df_styled = df_display.style.applymap(color_severity, subset=['Severity'])
        st.dataframe(df_styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df_display, use_container_width=True, hide_index=True)


def display_risk_metrics_table(risk_data: Dict) -> None:
    """
    Affiche un tableau des métriques de risque
    """
    if not risk_data:
        st.info("No risk data available")
        return

    # Créer un DataFrame à partir des métriques de risque
    metrics = {
        'Metric': ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'Max Drawdown', 'Volatility', 'Sharpe Ratio', 'Beta'],
        'Value': [
            f"{risk_data.get('var_95', 0):.2%}",
            f"{risk_data.get('var_99', 0):.2%}",
            f"{risk_data.get('cvar_95', 0):.2%}",
            f"{risk_data.get('max_drawdown', 0):.2%}",
            f"{risk_data.get('volatility', 0):.2%}",
            f"{risk_data.get('sharpe_ratio', 0):.2f}",
            f"{risk_data.get('beta', 0):.2f}"
        ],
        'Status': [
            get_risk_status(risk_data.get('var_95', 0), 0.05, 0.1),
            get_risk_status(risk_data.get('var_99', 0), 0.08, 0.15),
            get_risk_status(risk_data.get('cvar_95', 0), 0.06, 0.12),
            get_risk_status(risk_data.get('max_drawdown', 0), 0.03, 0.07),
            get_risk_status(risk_data.get('volatility', 0), 0.15, 0.25),
            'Good' if risk_data.get('sharpe_ratio', 0) > 1 else 'Poor',
            'Neutral'
        ]
    }

    df = pd.DataFrame(metrics)

    # Couleurs pour les statuts
    def color_status(val):
        if val == 'Good':
            return 'color: green'
        elif val == 'Warning':
            return 'color: orange'
        elif val == 'Critical':
            return 'color: red'
        return ''

    df_styled = df.style.applymap(color_status, subset=['Status'])
    st.dataframe(df_styled, use_container_width=True, hide_index=True)


def get_risk_status(value: float, warning_threshold: float, critical_threshold: float) -> str:
    """
    Détermine le statut de risque basé sur les seuils
    """
    if value < warning_threshold:
        return 'Good'
    elif value < critical_threshold:
        return 'Warning'
    else:
        return 'Critical'


def display_performance_summary_table(performance_data: Dict) -> None:
    """
    Affiche un tableau de résumé des performances
    """
    if not performance_data:
        st.info("No performance data available")
        return

    summary = {
        'Metric': [
            'Total Return',
            'Annualized Return',
            'Total Trades',
            'Win Rate',
            'Average Win',
            'Average Loss',
            'Profit Factor',
            'Maximum Drawdown',
            'Recovery Factor'
        ],
        'Value': [
            f"{performance_data.get('total_return', 0):.2%}",
            f"{performance_data.get('annualized_return', 0):.2%}",
            f"{performance_data.get('total_trades', 0):,}",
            f"{performance_data.get('win_rate', 0):.1%}",
            f"${performance_data.get('average_win', 0):,.2f}",
            f"${performance_data.get('average_loss', 0):,.2f}",
            f"{performance_data.get('profit_factor', 0):.2f}",
            f"{performance_data.get('max_drawdown', 0):.2%}",
            f"{performance_data.get('recovery_factor', 0):.2f}"
        ]
    }

    df = pd.DataFrame(summary)
    st.dataframe(df, use_container_width=True, hide_index=True)


def create_interactive_table(
    data: List[Dict],
    columns: List[str] = None,
    formatters: Dict[str, Callable] = None,
    filterable: bool = True,
    sortable: bool = True
) -> None:
    """
    Crée un tableau interactif avec filtres et tri
    """
    if not data:
        st.info("No data available")
        return

    df = pd.DataFrame(data)

    if columns:
        df = df[columns]

    if formatters:
        for col, formatter in formatters.items():
            if col in df.columns:
                df[col] = df[col].apply(formatter)

    if filterable:
        # Ajouter des filtres
        st.subheader("Filters")
        filter_cols = st.columns(min(len(df.columns), 4))

        filters = {}
        for i, col in enumerate(df.columns[:4]):  # Limiter à 4 filtres
            with filter_cols[i]:
                if df[col].dtype == 'object':
                    unique_values = df[col].unique()
                    filters[col] = st.multiselect(f"Filter {col}", unique_values, default=unique_values)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    filters[col] = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))

        # Appliquer les filtres
        for col, filter_val in filters.items():
            if isinstance(filter_val, list):
                df = df[df[col].isin(filter_val)]
            elif isinstance(filter_val, tuple):
                df = df[(df[col] >= filter_val[0]) & (df[col] <= filter_val[1])]

    st.dataframe(df, use_container_width=True, hide_index=True)