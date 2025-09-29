"""
Page Strategies - QFrame Streamlit App
======================================

Gestion complète des stratégies de trading : création, configuration, monitoring et performance.
"""

import streamlit as st
import sys
from pathlib import Path
import json

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from qframe.ui.streamlit_app.utils.session_state import (
    init_session, SessionStateManager, require_api_connection
)
from qframe.ui.streamlit_app.utils.api_client import get_api_client, format_currency, format_percentage
from qframe.ui.streamlit_app.components.charts import (
    create_strategy_performance_chart,
    create_pnl_chart,
    display_chart_with_controls
)
from qframe.ui.streamlit_app.components.tables import (
    display_strategies_table
)

# Configuration de la page
st.set_page_config(
    page_title="Strategies - QFrame",
    page_icon="🎯",
    layout="wide"
)

# Initialiser la session
init_session()

@require_api_connection
def main():
    """Page principale des stratégies"""

    st.title("🎯 Gestion des Stratégies")

    # Onglets pour organiser les fonctionnalités
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "➕ Créer", "⚙️ Configuration", "📈 Performance"])

    with tab1:
        render_strategies_overview()

    with tab2:
        render_create_strategy()

    with tab3:
        render_strategy_configuration()

    with tab4:
        render_strategy_performance()

def render_strategies_overview():
    """Vue d'ensemble des stratégies"""

    client = get_api_client()

    # Récupérer les stratégies
    strategies = client.get_strategies()

    if not strategies:
        st.info("Aucune stratégie trouvée. Créez votre première stratégie dans l'onglet 'Créer'.")
        return

    st.subheader("📊 Toutes les stratégies")

    # Tableau des stratégies
    display_strategies_table(strategies)

    # Métriques globales
    st.subheader("📈 Métriques globales")

    active_strategies = sum(1 for s in strategies if s.get('status') == 'active')
    total_strategies = len(strategies)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Stratégies actives", f"{active_strategies}/{total_strategies}")

    with col2:
        avg_return = sum(s.get('total_return', 0) for s in strategies) / len(strategies) if strategies else 0
        st.metric("Retour moyen", format_percentage(avg_return))

    with col3:
        profitable_strategies = sum(1 for s in strategies if s.get('total_return', 0) > 0)
        st.metric("Stratégies profitables", f"{profitable_strategies}/{total_strategies}")

    with col4:
        total_trades = sum(s.get('total_trades', 0) for s in strategies)
        st.metric("Total trades", total_trades)

    # Actions rapides
    st.subheader("🚀 Actions rapides")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("▶️ Démarrer toutes", use_container_width=True):
            start_all_strategies(client, strategies)

    with col2:
        if st.button("⏹️ Arrêter toutes", use_container_width=True):
            stop_all_strategies(client, strategies)

    with col3:
        if st.button("🔄 Actualiser", use_container_width=True):
            SessionStateManager.clear_cache('strategies')
            st.rerun()

    with col4:
        if st.button("📊 Recalculer perf", use_container_width=True):
            recalculate_all_performance(client, strategies)

    # Détails d'une stratégie sélectionnée
    st.subheader("🔍 Détails de la stratégie")

    strategy_options = {s['id']: f"{s['name']} ({s.get('status', 'unknown')})" for s in strategies}

    selected_id = st.selectbox(
        "Sélectionnez une stratégie",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        key="strategy_overview_select"
    )

    if selected_id:
        render_strategy_details(client, selected_id)

def render_strategy_details(client, strategy_id):
    """Affiche les détails d'une stratégie spécifique"""

    # Récupérer les détails de la stratégie
    strategy = client.get_strategy(strategy_id)
    if not strategy:
        st.error("Impossible de récupérer les détails de la stratégie")
        return

    # Informations de base
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Informations**")
        st.write(f"**Nom:** {strategy.get('name', 'N/A')}")
        st.write(f"**Type:** {strategy.get('type', 'N/A')}")
        st.write(f"**Statut:** {strategy.get('status', 'N/A')}")
        st.write(f"**Portfolio:** {strategy.get('portfolio_id', 'N/A')}")
        st.write(f"**Créé le:** {strategy.get('created_at', 'N/A')}")

    with col2:
        st.markdown("**📊 Configuration**")
        config = strategy.get('config', {})
        if config:
            for key, value in config.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write("Aucune configuration spécifique")

    # Performance
    performance = client.get_strategy_performance(strategy_id)
    if performance:
        st.markdown("**💰 Performance**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Retour total", format_percentage(performance.get('total_return', 0)))

        with col2:
            st.metric("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")

        with col3:
            st.metric("Max Drawdown", format_percentage(performance.get('max_drawdown', 0)))

        with col4:
            st.metric("Total trades", performance.get('total_trades', 0))

        # Graphiques de performance
        if 'pnl_history' in performance:
            fig_pnl = create_pnl_chart(
                performance['pnl_history'],
                f"P&L de {strategy.get('name', 'Strategy')}"
            )
            display_chart_with_controls(fig_pnl, f"strategy_pnl_{strategy_id}")

    # Actions sur la stratégie
    st.markdown("**🎛️ Actions**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if strategy.get('status') != 'active':
            if st.button("▶️ Démarrer", key=f"start_{strategy_id}"):
                result = client.start_strategy(strategy_id)
                if result:
                    st.success("Stratégie démarrée")
                    st.rerun()
                else:
                    st.error("Erreur lors du démarrage")

    with col2:
        if strategy.get('status') == 'active':
            if st.button("⏹️ Arrêter", key=f"stop_{strategy_id}"):
                result = client.stop_strategy(strategy_id)
                if result:
                    st.success("Stratégie arrêtée")
                    st.rerun()
                else:
                    st.error("Erreur lors de l'arrêt")

    with col3:
        if st.button("⚙️ Configurer", key=f"config_{strategy_id}"):
            st.session_state.configuring_strategy = strategy_id

    with col4:
        if st.button("📈 Analyser", key=f"analyze_{strategy_id}"):
            SessionStateManager.set_strategy(strategy_id)
            st.switch_page("pages/03_🎯_Strategies.py")

def render_create_strategy():
    """Interface de création de stratégie"""

    st.subheader("➕ Créer une nouvelle stratégie")

    # Types de stratégies disponibles
    strategy_types = {
        "mean_reversion": "Mean Reversion",
        "momentum": "Momentum",
        "arbitrage": "Arbitrage",
        "grid_trading": "Grid Trading",
        "dmn_lstm": "DMN LSTM",
        "rl_alpha": "RL Alpha Generation"
    }

    with st.form("create_strategy"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Nom de la stratégie", placeholder="Ex: Mean Reversion BTC")
            strategy_type = st.selectbox("Type de stratégie", options=list(strategy_types.keys()),
                                       format_func=lambda x: strategy_types[x])

        with col2:
            # Sélection du portfolio
            client = get_api_client()
            portfolios = client.get_portfolios()
            if portfolios:
                portfolio_options = {p['id']: p['name'] for p in portfolios}
                portfolio_id = st.selectbox(
                    "Portfolio cible",
                    options=list(portfolio_options.keys()),
                    format_func=lambda x: portfolio_options[x]
                )
            else:
                st.error("Aucun portfolio disponible. Créez d'abord un portfolio.")
                portfolio_id = None

            description = st.text_area("Description", placeholder="Description de la stratégie...")

        # Configuration spécifique au type
        st.markdown("**⚙️ Configuration de la stratégie**")

        config = {}

        if strategy_type == "mean_reversion":
            col1, col2 = st.columns(2)
            with col1:
                config['lookback_short'] = st.number_input("Période courte", value=10, min_value=1)
                config['lookback_long'] = st.number_input("Période longue", value=50, min_value=1)
            with col2:
                config['z_entry'] = st.number_input("Seuil d'entrée (Z-score)", value=2.0, step=0.1)
                config['z_exit'] = st.number_input("Seuil de sortie (Z-score)", value=0.5, step=0.1)

        elif strategy_type == "momentum":
            col1, col2 = st.columns(2)
            with col1:
                config['momentum_period'] = st.number_input("Période momentum", value=20, min_value=1)
                config['min_momentum'] = st.number_input("Momentum minimum", value=0.02, step=0.01)
            with col2:
                config['lookback_period'] = st.number_input("Période de référence", value=252, min_value=1)

        elif strategy_type == "grid_trading":
            col1, col2 = st.columns(2)
            with col1:
                config['grid_size'] = st.number_input("Taille de grille (%)", value=2.0, step=0.1)
                config['num_levels'] = st.number_input("Nombre de niveaux", value=10, min_value=1)
            with col2:
                config['base_amount'] = st.number_input("Montant de base", value=100.0, step=10.0)

        # Configuration générale
        with st.expander("⚙️ Configuration avancée"):
            config['position_size'] = st.slider("Taille de position (%)", 1, 100, 10)
            config['max_positions'] = st.number_input("Positions max simultanées", value=5, min_value=1)
            config['stop_loss'] = st.number_input("Stop Loss (%)", value=5.0, step=0.1, min_value=0.0)
            config['take_profit'] = st.number_input("Take Profit (%)", value=10.0, step=0.1, min_value=0.0)

        submitted = st.form_submit_button("🚀 Créer la stratégie", type="primary")

        if submitted:
            if not name:
                st.error("Le nom de la stratégie est requis")
                return

            if not portfolio_id:
                st.error("Un portfolio est requis")
                return

            # Préparer les données
            strategy_data = {
                "name": name,
                "type": strategy_type,
                "portfolio_id": portfolio_id,
                "description": description,
                "config": config
            }

            # Créer la stratégie
            result = client.create_strategy(strategy_data)

            if result:
                st.success(f"✅ Stratégie '{name}' créée avec succès !")
                st.json(result)

                # Définir comme stratégie sélectionnée
                SessionStateManager.set_strategy(result.get('id'))

                # Vider le cache pour rafraîchir la liste
                SessionStateManager.clear_cache('strategies')

                st.balloons()
            else:
                st.error("❌ Erreur lors de la création de la stratégie")

def render_strategy_configuration():
    """Interface de configuration des stratégies"""

    st.subheader("⚙️ Configuration des stratégies")

    client = get_api_client()
    strategies = client.get_strategies()

    if not strategies:
        st.info("Aucune stratégie à configurer")
        return

    # Sélection de la stratégie à configurer
    strategy_options = {s['id']: s['name'] for s in strategies}
    selected_id = st.selectbox(
        "Stratégie à configurer",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        key="config_strategy_select"
    )

    if not selected_id:
        return

    strategy = client.get_strategy(selected_id)
    if not strategy:
        st.error("Impossible de récupérer la stratégie")
        return

    st.subheader(f"Configuration de: {strategy.get('name')}")

    # Configuration actuelle
    current_config = strategy.get('config', {})

    with st.form("strategy_config"):
        st.markdown("**🎛️ Paramètres de trading**")

        col1, col2 = st.columns(2)

        with col1:
            position_size = st.slider(
                "Taille de position (%)",
                1, 100,
                int(current_config.get('position_size', 10))
            )

            max_positions = st.number_input(
                "Positions max simultanées",
                value=current_config.get('max_positions', 5),
                min_value=1
            )

        with col2:
            stop_loss = st.number_input(
                "Stop Loss (%)",
                value=current_config.get('stop_loss', 5.0),
                step=0.1,
                min_value=0.0
            )

            take_profit = st.number_input(
                "Take Profit (%)",
                value=current_config.get('take_profit', 10.0),
                step=0.1,
                min_value=0.0
            )

        # Configuration spécifique au type de stratégie
        strategy_type = strategy.get('type')

        if strategy_type == "mean_reversion":
            st.markdown("**📊 Paramètres Mean Reversion**")
            col1, col2 = st.columns(2)

            with col1:
                lookback_short = st.number_input(
                    "Période courte",
                    value=current_config.get('lookback_short', 10),
                    min_value=1
                )
                z_entry = st.number_input(
                    "Seuil d'entrée (Z-score)",
                    value=current_config.get('z_entry', 2.0),
                    step=0.1
                )

            with col2:
                lookback_long = st.number_input(
                    "Période longue",
                    value=current_config.get('lookback_long', 50),
                    min_value=1
                )
                z_exit = st.number_input(
                    "Seuil de sortie (Z-score)",
                    value=current_config.get('z_exit', 0.5),
                    step=0.1
                )

        # Configuration de risque
        st.markdown("**⚠️ Gestion du risque**")

        col1, col2 = st.columns(2)

        with col1:
            max_daily_loss = st.number_input(
                "Perte quotidienne max (%)",
                value=current_config.get('max_daily_loss', 5.0),
                step=0.1,
                min_value=0.0
            )

        with col2:
            max_drawdown = st.number_input(
                "Drawdown max (%)",
                value=current_config.get('max_drawdown', 10.0),
                step=0.1,
                min_value=0.0
            )

        # Horaires de trading
        st.markdown("**🕐 Horaires de trading**")

        col1, col2 = st.columns(2)

        with col1:
            start_hour = st.time_input(
                "Heure de début",
                value=None
            )

        with col2:
            end_hour = st.time_input(
                "Heure de fin",
                value=None
            )

        # Boutons d'action
        col1, col2, col3 = st.columns(3)

        with col1:
            save_config = st.form_submit_button("💾 Sauvegarder", type="primary")

        with col2:
            test_config = st.form_submit_button("🧪 Tester config")

        with col3:
            reset_config = st.form_submit_button("🔄 Reset défaut")

        if save_config:
            # Construire la nouvelle configuration
            new_config = {
                'position_size': position_size,
                'max_positions': max_positions,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'max_daily_loss': max_daily_loss,
                'max_drawdown': max_drawdown
            }

            # Ajouter configuration spécifique au type
            if strategy_type == "mean_reversion":
                new_config.update({
                    'lookback_short': lookback_short,
                    'lookback_long': lookback_long,
                    'z_entry': z_entry,
                    'z_exit': z_exit
                })

            # Mettre à jour la stratégie
            update_data = {"config": new_config}
            result = client.update_strategy(selected_id, update_data)

            if result:
                st.success("✅ Configuration sauvegardée")
                SessionStateManager.clear_cache('strategies')
                st.rerun()
            else:
                st.error("❌ Erreur lors de la sauvegarde")

        if test_config:
            st.info("🧪 Test de configuration - Fonctionnalité à implémenter")

        if reset_config:
            st.warning("🔄 Reset à la configuration par défaut - Fonctionnalité à implémenter")

def render_strategy_performance():
    """Analyse de performance des stratégies"""

    st.subheader("📈 Performance des stratégies")

    client = get_api_client()
    strategies = client.get_strategies()

    if not strategies:
        st.info("Aucune stratégie à analyser")
        return

    # Sélection multiple pour comparaison
    strategy_options = {s['id']: s['name'] for s in strategies}
    selected_ids = st.multiselect(
        "Stratégies à analyser",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        default=list(strategy_options.keys())[:3]  # Sélectionner les 3 premières
    )

    if not selected_ids:
        st.warning("Sélectionnez au moins une stratégie")
        return

    # Collecte des données de performance
    performance_data = []
    for strategy_id in selected_ids:
        strategy = next(s for s in strategies if s['id'] == strategy_id)
        perf = client.get_strategy_performance(strategy_id)

        if perf:
            performance_data.append({
                'name': strategy['name'],
                'total_return': perf.get('total_return', 0),
                'sharpe_ratio': perf.get('sharpe_ratio', 0),
                'max_drawdown': perf.get('max_drawdown', 0),
                'total_trades': perf.get('total_trades', 0),
                'win_rate': perf.get('win_rate', 0),
                'avg_trade_return': perf.get('avg_trade_return', 0)
            })

    if not performance_data:
        st.warning("Aucune donnée de performance disponible")
        return

    # Graphique de comparaison
    if len(performance_data) > 1:
        fig_comparison = create_strategy_performance_chart(
            performance_data,
            "Comparaison des performances"
        )
        display_chart_with_controls(fig_comparison, "strategy_comparison")

    # Tableau de performance détaillé
    st.subheader("📊 Tableau de performance")

    import pandas as pd
    df = pd.DataFrame(performance_data)

    # Formater les colonnes
    df['Retour (%)'] = df['total_return'].apply(lambda x: f"{x:.2%}")
    df['Sharpe'] = df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    df['Max DD (%)'] = df['max_drawdown'].apply(lambda x: f"{x:.2%}")
    df['Taux réussite (%)'] = df['win_rate'].apply(lambda x: f"{x:.2%}")

    # Afficher le tableau
    display_df = df[['name', 'Retour (%)', 'Sharpe', 'Max DD (%)', 'total_trades', 'Taux réussite (%)']].copy()
    display_df.columns = ['Stratégie', 'Retour', 'Sharpe', 'Max DD', 'Trades', 'Taux réussite']

    st.dataframe(display_df, use_container_width=True)

    # Métriques de risque détaillées
    st.subheader("⚠️ Analyse de risque")

    for strategy_id in selected_ids:
        strategy_name = strategy_options[strategy_id]

        with st.expander(f"Risque détaillé - {strategy_name}"):
            perf = client.get_strategy_performance(strategy_id)

            if perf:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Volatilité", f"{perf.get('volatility', 0):.2%}")

                with col2:
                    st.metric("VaR 95%", f"${perf.get('var_95', 0):,.0f}")

                with col3:
                    st.metric("Calmar Ratio", f"{perf.get('calmar_ratio', 0):.2f}")

                with col4:
                    st.metric("Sortino Ratio", f"{perf.get('sortino_ratio', 0):.2f}")

                # Graphique P&L historique
                if 'pnl_history' in perf:
                    fig_pnl = create_pnl_chart(
                        perf['pnl_history'],
                        f"P&L historique - {strategy_name}"
                    )
                    display_chart_with_controls(fig_pnl, f"detailed_pnl_{strategy_id}")
            else:
                st.info("Données de performance non disponibles")

# Fonctions utilitaires
def start_all_strategies(client, strategies):
    """Démarre toutes les stratégies inactives"""
    started = 0
    for strategy in strategies:
        if strategy.get('status') != 'active':
            result = client.start_strategy(strategy['id'])
            if result:
                started += 1

    if started > 0:
        st.success(f"✅ {started} stratégies démarrées")
        SessionStateManager.clear_cache('strategies')
        st.rerun()
    else:
        st.info("Aucune stratégie à démarrer")

def stop_all_strategies(client, strategies):
    """Arrête toutes les stratégies actives"""
    stopped = 0
    for strategy in strategies:
        if strategy.get('status') == 'active':
            result = client.stop_strategy(strategy['id'])
            if result:
                stopped += 1

    if stopped > 0:
        st.success(f"✅ {stopped} stratégies arrêtées")
        SessionStateManager.clear_cache('strategies')
        st.rerun()
    else:
        st.info("Aucune stratégie à arrêter")

def recalculate_all_performance(client, strategies):
    """Recalcule la performance de toutes les stratégies"""
    with st.spinner("Recalcul des performances..."):
        # Ici on pourrait appeler un endpoint de recalcul global
        # Pour l'instant on vide juste le cache
        SessionStateManager.clear_cache()
        st.success("Performance recalculée")

if __name__ == "__main__":
    main()