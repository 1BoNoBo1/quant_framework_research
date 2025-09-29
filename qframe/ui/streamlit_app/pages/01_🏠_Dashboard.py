"""
Page Dashboard - QFrame Streamlit App
=====================================

Tableau de bord principal avec métriques de performance, graphiques et données en temps réel.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from qframe.ui.streamlit_app.utils.session_state import (
    init_session, SessionStateManager, require_api_connection
)
from qframe.ui.streamlit_app.utils.api_client import get_api_client
from qframe.ui.streamlit_app.components.charts import (
    create_portfolio_value_chart,
    create_portfolio_allocation_chart,
    create_strategy_performance_chart,
    create_pnl_chart,
    create_orders_timeline_chart,
    create_risk_metrics_chart,
    display_metric_card,
    display_chart_with_controls
)
from qframe.ui.streamlit_app.components.tables import (
    display_portfolio_table,
    display_orders_table,
    display_alerts_table
)

# Configuration de la page
st.set_page_config(
    page_title="Dashboard - QFrame",
    page_icon="🏠",
    layout="wide"
)

# Initialiser la session
init_session()

@require_api_connection
def main():
    """Page Dashboard principale"""

    st.title("🏠 Dashboard - Vue d'ensemble")

    client = get_api_client()

    # Métriques principales
    render_key_metrics(client)

    # Graphiques principaux
    render_main_charts(client)

    # Tableaux de données
    render_data_tables(client)

    # Sidebar avec contrôles
    render_sidebar_controls()

def render_key_metrics(client):
    """Affiche les métriques clés"""

    st.subheader("📊 Métriques clés")

    # Récupérer les données du dashboard
    dashboard_data = SessionStateManager.get_cached_data('dashboard_data')
    if not dashboard_data:
        dashboard_data = client.get_dashboard_data()
        if dashboard_data:
            SessionStateManager.cache_data('dashboard_data', dashboard_data, ttl_seconds=60)

    if not dashboard_data:
        st.warning("Impossible de récupérer les données du dashboard")
        return

    # Affichage des métriques en colonnes
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_value = dashboard_data.get('total_portfolio_value', 0)
        daily_change = dashboard_data.get('daily_pnl_pct', 0)
        display_metric_card(
            "Valeur totale",
            total_value,
            delta=daily_change,
            format_func=lambda x: f"${x:,.2f}"
        )

    with col2:
        daily_pnl = dashboard_data.get('daily_pnl', 0)
        display_metric_card(
            "P&L quotidien",
            daily_pnl,
            format_func=lambda x: f"${x:,.2f}"
        )

    with col3:
        active_strategies = dashboard_data.get('active_strategies', 0)
        total_strategies = dashboard_data.get('total_strategies', 0)
        display_metric_card(
            "Stratégies actives",
            f"{active_strategies}/{total_strategies}"
        )

    with col4:
        daily_trades = dashboard_data.get('daily_trades', 0)
        avg_trades = dashboard_data.get('avg_daily_trades', 0)
        delta = ((daily_trades - avg_trades) / max(avg_trades, 1)) if avg_trades > 0 else 0
        display_metric_card(
            "Trades (24h)",
            daily_trades,
            delta=delta
        )

    with col5:
        win_rate = dashboard_data.get('win_rate', 0)
        display_metric_card(
            "Taux de réussite",
            win_rate,
            format_func=lambda x: f"{x:.1%}"
        )

    # Métriques de risque secondaires
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        max_drawdown = dashboard_data.get('max_drawdown', 0)
        display_metric_card(
            "Max Drawdown",
            max_drawdown,
            format_func=lambda x: f"{x:.2%}"
        )

    with col2:
        sharpe_ratio = dashboard_data.get('sharpe_ratio', 0)
        display_metric_card(
            "Sharpe Ratio",
            f"{sharpe_ratio:.2f}"
        )

    with col3:
        var_95 = dashboard_data.get('var_95', 0)
        display_metric_card(
            "VaR 95%",
            var_95,
            format_func=lambda x: f"${x:,.0f}"
        )

    with col4:
        open_orders = dashboard_data.get('open_orders', 0)
        display_metric_card(
            "Ordres ouverts",
            open_orders
        )

def render_main_charts(client):
    """Affiche les graphiques principaux"""

    st.subheader("📈 Graphiques de performance")

    # Récupérer les données de performance
    portfolio_data = SessionStateManager.get_cached_data('portfolio_performance')
    if not portfolio_data:
        portfolios = client.get_portfolios()
        if portfolios and len(portfolios) > 0:
            # Prendre le premier portfolio ou celui sélectionné
            selected_id = SessionStateManager.get_selected_portfolio()
            if not selected_id and portfolios:
                selected_id = portfolios[0]['id']
                SessionStateManager.set_portfolio(selected_id)

            if selected_id:
                portfolio_data = client.get_portfolio_performance(selected_id)
                if portfolio_data:
                    SessionStateManager.cache_data('portfolio_performance', portfolio_data, ttl_seconds=300)

    # Layout en onglets pour organiser les graphiques
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "🎯 Stratégies", "⚠️ Risque", "📋 Ordres"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Graphique de valeur du portfolio
            if portfolio_data and 'value_history' in portfolio_data:
                fig_value = create_portfolio_value_chart(
                    portfolio_data['value_history'],
                    "Évolution de la valeur du portfolio"
                )
                display_chart_with_controls(fig_value, "portfolio_value")
            else:
                st.info("Données de performance du portfolio non disponibles")

        with col2:
            # Graphique d'allocation
            portfolios = client.get_portfolios()
            if portfolios:
                selected_id = SessionStateManager.get_selected_portfolio()
                if selected_id:
                    positions = client.get_portfolio_positions(selected_id)
                    if positions:
                        fig_allocation = create_portfolio_allocation_chart(
                            positions,
                            "Allocation actuelle du portfolio"
                        )
                        display_chart_with_controls(fig_allocation, "portfolio_allocation")
                    else:
                        st.info("Aucune position dans le portfolio")

        # P&L Chart sur toute la largeur
        if portfolio_data and 'pnl_history' in portfolio_data:
            fig_pnl = create_pnl_chart(
                portfolio_data['pnl_history'],
                "Historique P&L"
            )
            display_chart_with_controls(fig_pnl, "pnl_history")

    with tab2:
        # Performance des stratégies
        strategies = client.get_strategies()
        if strategies:
            strategy_performance = []
            for strategy in strategies:
                perf = client.get_strategy_performance(strategy['id'])
                if perf:
                    strategy_performance.append({
                        'name': strategy['name'],
                        'total_return': perf.get('total_return', 0),
                        'sharpe_ratio': perf.get('sharpe_ratio', 0),
                        'max_drawdown': perf.get('max_drawdown', 0)
                    })

            if strategy_performance:
                fig_strategies = create_strategy_performance_chart(
                    strategy_performance,
                    "Performance des stratégies"
                )
                display_chart_with_controls(fig_strategies, "strategy_performance")
            else:
                st.info("Données de performance des stratégies non disponibles")
        else:
            st.info("Aucune stratégie configurée")

    with tab3:
        # Métriques de risque
        selected_portfolio = SessionStateManager.get_selected_portfolio()
        if selected_portfolio:
            risk_assessments = client.get_risk_assessments(selected_portfolio)
            if risk_assessments and len(risk_assessments) > 0:
                latest_risk = risk_assessments[0]  # Le plus récent
                fig_risk = create_risk_metrics_chart(
                    latest_risk,
                    "Métriques de risque actuelles"
                )
                display_chart_with_controls(fig_risk, "risk_metrics")
            else:
                st.info("Évaluation de risque non disponible")

                # Bouton pour calculer le risque
                if st.button("🔍 Calculer le risque"):
                    with st.spinner("Calcul en cours..."):
                        risk_result = client.calculate_portfolio_risk(selected_portfolio)
                        if risk_result:
                            st.success("Risque calculé avec succès")
                            st.rerun()
                        else:
                            st.error("Erreur lors du calcul du risque")

    with tab4:
        # Timeline des ordres
        orders = client.get_orders(limit=50)
        if orders:
            fig_orders = create_orders_timeline_chart(
                orders,
                "Timeline des ordres récents"
            )
            display_chart_with_controls(fig_orders, "orders_timeline")
        else:
            st.info("Aucun ordre récent")

def render_data_tables(client):
    """Affiche les tableaux de données"""

    st.subheader("📋 Données détaillées")

    # Layout en colonnes pour les tableaux
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏦 Portfolios**")
        portfolios = client.get_portfolios()
        if portfolios:
            display_portfolio_table(portfolios)
        else:
            st.info("Aucun portfolio trouvé")

        # Ordres récents
        st.markdown("**📋 Ordres récents**")
        orders = client.get_orders(limit=10)
        if orders:
            display_orders_table(orders)
        else:
            st.info("Aucun ordre récent")

    with col2:
        # Alertes récentes
        st.markdown("**🚨 Alertes récentes**")
        alerts = client.get_alerts(limit=10)
        if alerts:
            display_alerts_table(alerts)
        else:
            st.info("Aucune alerte récente")

        # Stratégies
        st.markdown("**🎯 Stratégies**")
        strategies = client.get_strategies()
        if strategies:
            for strategy in strategies:
                status_icon = "🟢" if strategy.get('status') == 'active' else "🔴"
                st.write(f"{status_icon} **{strategy.get('name', 'N/A')}**")
                st.caption(f"Status: {strategy.get('status', 'unknown')}")
        else:
            st.info("Aucune stratégie configurée")

def render_sidebar_controls():
    """Affiche les contrôles dans la sidebar"""

    with st.sidebar:
        st.markdown("---")
        st.subheader("🎛️ Contrôles Dashboard")

        # Sélection du portfolio
        portfolios = get_api_client().get_portfolios()
        if portfolios:
            portfolio_options = {p['id']: p['name'] for p in portfolios}
            current_portfolio = SessionStateManager.get_selected_portfolio()

            selected = st.selectbox(
                "Portfolio actif",
                options=list(portfolio_options.keys()),
                format_func=lambda x: portfolio_options[x],
                index=list(portfolio_options.keys()).index(current_portfolio) if current_portfolio in portfolio_options else 0
            )

            if selected != current_portfolio:
                SessionStateManager.set_portfolio(selected)
                st.rerun()

        # Contrôles de rafraîchissement
        st.markdown("**🔄 Rafraîchissement**")

        if st.button("Actualiser données", use_container_width=True):
            SessionStateManager.clear_cache()
            st.success("Données actualisées")
            st.rerun()

        if st.button("Vider tout le cache", use_container_width=True):
            SessionStateManager.clear_cache()
            st.success("Cache vidé")

        # Affichage de l'état du cache
        st.markdown("**💾 État du cache**")
        cached_keys = list(SessionStateManager.get('cached_data', {}).keys())
        if cached_keys:
            for key in cached_keys:
                st.caption(f"✅ {key}")
        else:
            st.caption("Cache vide")

        # Auto-refresh
        auto_refresh = SessionStateManager.get('auto_refresh', True)
        if auto_refresh:
            interval = SessionStateManager.get('refresh_interval', 30)
            st.rerun(delay=interval)

if __name__ == "__main__":
    main()