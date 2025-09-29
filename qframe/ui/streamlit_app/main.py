"""
QFrame Streamlit Application
============================

Application principale pour l'interface web QFrame utilisant Streamlit.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Ajouter le chemin du projet pour les imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from qframe.ui.streamlit_app.utils.session_state import init_session, SessionStateManager
from qframe.ui.streamlit_app.utils.api_client import check_api_connection, get_api_client
from qframe.ui.streamlit_app.components.charts import display_metric_card

# Configuration de la page
st.set_page_config(
    page_title="QFrame - Quantitative Trading Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/qframe',
        'Report a bug': "https://github.com/your-repo/qframe/issues",
        'About': "QFrame - Framework de trading quantitatif professionnel"
    }
)

# CSS personnalisé
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }

    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    .sidebar .sidebar-content {
        background-color: #262730;
    }

    .css-1d391kg {
        padding-top: 1rem;
    }

    .status-connected {
        color: #00ff88;
        font-weight: bold;
    }

    .status-disconnected {
        color: #ff4444;
        font-weight: bold;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Point d'entrée principal de l'application"""

    # Initialiser la session
    init_session()

    # Sidebar pour navigation et configuration
    with st.sidebar:
        st.title("🎯 QFrame")
        st.markdown("---")

        # Vérification de la connexion API
        st.subheader("🔌 Connexion API")
        api_connected = check_api_connection()
        SessionStateManager.set('api_connected', api_connected)

        if api_connected:
            st.markdown('<span class="status-connected">✅ Connecté</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-disconnected">❌ Déconnecté</span>', unsafe_allow_html=True)
            st.info("Démarrez le serveur QFrame sur http://localhost:8000")

        st.markdown("---")

        # Configuration d'affichage
        st.subheader("⚙️ Configuration")

        # Auto-refresh
        auto_refresh = st.checkbox(
            "Rafraîchissement automatique",
            value=SessionStateManager.get('auto_refresh', True)
        )
        SessionStateManager.set('auto_refresh', auto_refresh)

        if auto_refresh:
            refresh_interval = st.slider(
                "Intervalle (secondes)",
                5, 300,
                SessionStateManager.get('refresh_interval', 30)
            )
            SessionStateManager.set('refresh_interval', refresh_interval)

        # Filtres de temps
        date_range = st.selectbox(
            "Période d'affichage",
            ["1h", "6h", "24h", "7d", "30d"],
            index=2,  # 24h par défaut
            key="date_range_select"
        )
        SessionStateManager.set('date_range', date_range)

        # Filtre symboles
        symbol_filter = st.selectbox(
            "Filtre symboles",
            ["All", "BTC", "ETH", "Majors", "Altcoins"],
            key="symbol_filter_select"
        )
        SessionStateManager.set('symbol_filter', symbol_filter)

        st.markdown("---")

        # Actions rapides
        st.subheader("🚀 Actions")

        if st.button("🔄 Actualiser", use_container_width=True):
            SessionStateManager.clear_cache()
            st.rerun()

        if st.button("🧹 Vider cache", use_container_width=True):
            SessionStateManager.clear_cache()
            st.success("Cache vidé")

        # Informations système
        st.markdown("---")
        st.subheader("📊 Système")

        if api_connected:
            client = get_api_client()
            system_info = client.get_system_info()

            if system_info:
                st.metric("Uptime", system_info.get('uptime', 'N/A'))
                st.metric("Version", system_info.get('version', 'N/A'))

                if 'memory_usage' in system_info:
                    memory_pct = system_info['memory_usage'].get('percent', 0)
                    st.metric("Mémoire", f"{memory_pct:.1f}%")

        # Debug session state
        if st.checkbox("Mode debug"):
            debug_info = SessionStateManager.debug_session_state()
            st.json(debug_info)

    # Contenu principal
    st.title("📊 QFrame - Dashboard Principal")

    if not api_connected:
        st.error("❌ Connexion API requise pour afficher les données")
        st.info("Veuillez vérifier que le serveur QFrame est démarré.")
        return

    # Dashboard principal
    render_main_dashboard()

def render_main_dashboard():
    """Rendu du dashboard principal"""

    client = get_api_client()

    # Métriques principales
    st.subheader("📈 Vue d'ensemble")

    # Récupérer les données du dashboard
    dashboard_data = client.get_dashboard_data()

    if dashboard_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_value = dashboard_data.get('total_portfolio_value', 0)
            display_metric_card(
                "Valeur totale",
                f"${total_value:,.2f}",
                delta=dashboard_data.get('daily_pnl_pct', 0)
            )

        with col2:
            active_strategies = dashboard_data.get('active_strategies', 0)
            display_metric_card("Stratégies actives", active_strategies)

        with col3:
            daily_trades = dashboard_data.get('daily_trades', 0)
            display_metric_card("Trades (24h)", daily_trades)

        with col4:
            open_orders = dashboard_data.get('open_orders', 0)
            display_metric_card("Ordres ouverts", open_orders)

    # Alertes système
    alerts = client.get_alerts(limit=5)
    if alerts:
        st.subheader("🚨 Alertes récentes")

        for alert in alerts:
            severity = alert.get('severity', 'info')
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "🔵")

            with st.expander(f"{icon} {alert.get('title', 'Alert')}"):
                st.write(alert.get('message', ''))
                st.caption(f"Créé le: {alert.get('created_at', 'N/A')}")

                if st.button(f"Marquer comme lu", key=f"ack_{alert.get('id')}"):
                    client.acknowledge_alert(alert.get('id'))
                    st.success("Alerte marquée comme lue")
                    st.rerun()

    # Performance récente
    st.subheader("📊 Performance récente")

    # Navigation vers les pages spécialisées
    st.subheader("🧭 Navigation rapide")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📁 Portfolios", use_container_width=True):
            st.switch_page("pages/02_📁_Portfolios.py")

    with col2:
        if st.button("🎯 Stratégies", use_container_width=True):
            st.switch_page("pages/03_🎯_Strategies.py")

    with col3:
        if st.button("⚠️ Gestion risques", use_container_width=True):
            st.switch_page("pages/05_⚠️_Risk_Management.py")

    # Auto-refresh
    if SessionStateManager.get('auto_refresh', True):
        interval = SessionStateManager.get('refresh_interval', 30)
        st.rerun(delay=interval)

if __name__ == "__main__":
    main()