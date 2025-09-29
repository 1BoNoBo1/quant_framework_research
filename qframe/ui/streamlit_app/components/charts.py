"""
Chart Components for QFrame Streamlit App
=========================================

Composants de graphiques réutilisables pour l'interface Streamlit.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


def create_portfolio_value_chart(data: List[Dict], title: str = "Portfolio Value Over Time") -> go.Figure:
    """
    Crée un graphique de valeur de portfolio dans le temps
    """
    if not data:
        return create_empty_chart("No portfolio data available")

    df = pd.DataFrame(data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['total_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00ff88', width=2),
        hovertemplate='<b>%{y:$,.2f}</b><br>%{x}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value (USD)",
        template="plotly_dark",
        height=400,
        showlegend=False
    )

    return fig


def create_portfolio_allocation_chart(positions: List[Dict], title: str = "Portfolio Allocation") -> go.Figure:
    """
    Crée un graphique en secteurs de l'allocation du portfolio
    """
    if not positions:
        return create_empty_chart("No position data available")

    df = pd.DataFrame(positions)

    fig = go.Figure(data=[go.Pie(
        labels=df['symbol'],
        values=df['market_value'],
        hole=0.4,
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400
    )

    return fig


def create_strategy_performance_chart(strategies: List[Dict], title: str = "Strategy Performance") -> go.Figure:
    """
    Crée un graphique de performance des stratégies
    """
    if not strategies:
        return create_empty_chart("No strategy data available")

    df = pd.DataFrame(strategies)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['name'],
        y=df['total_return'],
        marker_color=df['total_return'].apply(lambda x: '#00ff88' if x >= 0 else '#ff4444'),
        hovertemplate='<b>%{x}</b><br>Return: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Strategy",
        yaxis_title="Total Return (%)",
        template="plotly_dark",
        height=400,
        showlegend=False
    )

    return fig


def create_risk_metrics_chart(risk_data: Dict, title: str = "Risk Metrics") -> go.Figure:
    """
    Crée un graphique de métriques de risque
    """
    if not risk_data:
        return create_empty_chart("No risk data available")

    # Créer des jauges pour les métriques de risque
    metrics = ['var_95', 'max_drawdown', 'volatility']
    values = [risk_data.get(metric, 0) for metric in metrics]
    labels = ['VaR 95%', 'Max Drawdown', 'Volatility']

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=labels
    )

    for i, (value, label) in enumerate(zip(values, labels)):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value * 100,  # Convertir en pourcentage
                title={'text': label},
                gauge={
                    'axis': {'range': [None, 20]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 10], 'color': "yellow"},
                        {'range': [10, 20], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 15
                    }
                }
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=300
    )

    return fig


def create_orders_timeline_chart(orders: List[Dict], title: str = "Orders Timeline") -> go.Figure:
    """
    Crée un timeline des ordres
    """
    if not orders:
        return create_empty_chart("No orders data available")

    df = pd.DataFrame(orders)

    # Créer couleurs selon le type d'ordre
    colors = {
        'BUY': '#00ff88',
        'SELL': '#ff4444'
    }

    fig = go.Figure()

    for order_type in df['side'].unique():
        mask = df['side'] == order_type
        subset = df[mask]

        fig.add_trace(go.Scatter(
            x=subset['created_time'],
            y=subset['price'],
            mode='markers',
            name=order_type,
            marker=dict(
                color=colors.get(order_type, '#888888'),
                size=subset['quantity'] * 10,  # Taille basée sur la quantité
                opacity=0.7
            ),
            hovertemplate=f'<b>{order_type}</b><br>' +
                         'Price: $%{y:,.2f}<br>' +
                         'Quantity: %{text}<br>' +
                         'Time: %{x}<extra></extra>',
            text=subset['quantity']
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=400
    )

    return fig


def create_pnl_chart(pnl_data: List[Dict], title: str = "P&L Over Time") -> go.Figure:
    """
    Crée un graphique de P&L dans le temps
    """
    if not pnl_data:
        return create_empty_chart("No P&L data available")

    df = pd.DataFrame(pnl_data)

    fig = go.Figure()

    # P&L réalisé
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['realized_pnl'],
        mode='lines',
        name='Realized P&L',
        line=dict(color='#00ff88', width=2),
        fill='tonexty' if 'unrealized_pnl' in df.columns else 'tozeroy'
    ))

    # P&L non réalisé si disponible
    if 'unrealized_pnl' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['unrealized_pnl'],
            mode='lines',
            name='Unrealized P&L',
            line=dict(color='#ffaa00', width=2),
            fill='tozeroy'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="P&L (USD)",
        template="plotly_dark",
        height=400
    )

    # Ligne de référence à zéro
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig


def create_market_data_chart(market_data: List[Dict], title: str = "Price Chart") -> go.Figure:
    """
    Crée un graphique de données de marché (OHLC)
    """
    if not market_data:
        return create_empty_chart("No market data available")

    df = pd.DataFrame(market_data)

    fig = go.Figure(data=go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False
    )

    return fig


def create_volume_chart(market_data: List[Dict], title: str = "Volume") -> go.Figure:
    """
    Crée un graphique de volume
    """
    if not market_data:
        return create_empty_chart("No volume data available")

    df = pd.DataFrame(market_data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name='Volume',
        marker_color='rgba(158,202,225,0.8)',
        hovertemplate='Volume: %{y:,.0f}<br>Time: %{x}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Volume",
        template="plotly_dark",
        height=200,
        showlegend=False
    )

    return fig


def create_combined_price_volume_chart(market_data: List[Dict], title: str = "Price & Volume") -> go.Figure:
    """
    Crée un graphique combiné prix et volume
    """
    if not market_data:
        return create_empty_chart("No market data available")

    df = pd.DataFrame(market_data)

    # Créer subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=['Price', 'Volume']
    )

    # Graphique de prix
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ),
        row=1, col=1
    )

    # Graphique de volume
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.8)'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
    )

    return fig


def create_empty_chart(message: str) -> go.Figure:
    """
    Crée un graphique vide avec un message
    """
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=16, color="gray")
    )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return fig


def display_metric_card(title: str, value: Any, delta: Optional[float] = None, format_func: callable = None):
    """
    Affiche une carte de métrique avec Streamlit
    """
    if format_func:
        formatted_value = format_func(value)
    else:
        formatted_value = str(value)

    if delta is not None:
        delta_color = "normal" if delta >= 0 else "inverse"
        st.metric(
            label=title,
            value=formatted_value,
            delta=f"{delta:+.2%}" if isinstance(delta, float) else str(delta),
            delta_color=delta_color
        )
    else:
        st.metric(label=title, value=formatted_value)


def display_chart_with_controls(fig: go.Figure, key: str = None):
    """
    Affiche un graphique avec des contrôles
    """
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("📊 Export", key=f"export_{key}"):
            # TODO: Implémenter l'export
            st.info("Export functionality coming soon")

    with col2:
        if st.button("🔄 Refresh", key=f"refresh_{key}"):
            st.rerun()

    with col3:
        if st.button("⚙️ Settings", key=f"settings_{key}"):
            # TODO: Implémenter les paramètres de graphique
            st.info("Chart settings coming soon")

    st.plotly_chart(fig, use_container_width=True)