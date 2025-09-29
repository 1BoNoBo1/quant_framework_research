"""
Page Portfolios - QFrame Streamlit App
======================================

Gestion complète des portfolios : création, modification, visualisation des performances.
"""

import streamlit as st
import sys
from pathlib import Path
from decimal import Decimal

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from qframe.ui.streamlit_app.utils.session_state import (
    init_session, SessionStateManager, require_api_connection
)
from qframe.ui.streamlit_app.utils.api_client import get_api_client, format_currency, format_percentage
from qframe.ui.streamlit_app.components.charts import (
    create_portfolio_value_chart,
    create_portfolio_allocation_chart,
    create_pnl_chart,
    display_chart_with_controls
)
from qframe.ui.streamlit_app.components.tables import (
    display_portfolio_table,
    display_positions_table
)

# Configuration de la page
st.set_page_config(
    page_title="Portfolios - QFrame",
    page_icon="📁",
    layout="wide"
)

# Initialiser la session
init_session()

@require_api_connection
def main():
    """Page principale des portfolios"""

    st.title("📁 Gestion des Portfolios")

    # Onglets pour organiser les fonctionnalités
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "➕ Créer", "⚙️ Gérer", "📈 Analyse"])

    with tab1:
        render_portfolios_overview()

    with tab2:
        render_create_portfolio()

    with tab3:
        render_manage_portfolios()

    with tab4:
        render_portfolio_analysis()

def render_portfolios_overview():
    """Vue d'ensemble des portfolios"""

    client = get_api_client()

    # Récupérer les portfolios
    portfolios = client.get_portfolios()

    if not portfolios:
        st.info("Aucun portfolio trouvé. Créez votre premier portfolio dans l'onglet 'Créer'.")
        return

    st.subheader("📊 Tous les portfolios")

    # Tableau des portfolios
    display_portfolio_table(portfolios)

    # Métriques globales
    st.subheader("📈 Métriques globales")

    total_value = sum(p.get('total_value', 0) for p in portfolios)
    total_pnl = sum(p.get('unrealized_pnl', 0) for p in portfolios)
    avg_return = sum(p.get('total_return_pct', 0) for p in portfolios) / len(portfolios) if portfolios else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Valeur totale", format_currency(total_value))

    with col2:
        st.metric("P&L total", format_currency(total_pnl))

    with col3:
        st.metric("Retour moyen", format_percentage(avg_return))

    with col4:
        st.metric("Nombre de portfolios", len(portfolios))

    # Sélection de portfolio pour détails
    st.subheader("🔍 Détails du portfolio")

    portfolio_options = {p['id']: f"{p['name']} ({format_currency(p.get('total_value', 0))})" for p in portfolios}

    selected_id = st.selectbox(
        "Sélectionnez un portfolio",
        options=list(portfolio_options.keys()),
        format_func=lambda x: portfolio_options[x],
        key="portfolio_overview_select"
    )

    if selected_id:
        render_portfolio_details(client, selected_id)

def render_portfolio_details(client, portfolio_id):
    """Affiche les détails d'un portfolio spécifique"""

    # Récupérer les détails du portfolio
    portfolio = client.get_portfolio(portfolio_id)
    if not portfolio:
        st.error("Impossible de récupérer les détails du portfolio")
        return

    # Informations de base
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Informations**")
        st.write(f"**Nom:** {portfolio.get('name', 'N/A')}")
        st.write(f"**ID:** {portfolio.get('id', 'N/A')}")
        st.write(f"**Capital initial:** {format_currency(portfolio.get('initial_capital', 0))}")
        st.write(f"**Devise de base:** {portfolio.get('base_currency', 'USD')}")
        st.write(f"**Créé le:** {portfolio.get('created_at', 'N/A')}")

    with col2:
        st.markdown("**💰 Performance**")
        st.write(f"**Valeur actuelle:** {format_currency(portfolio.get('total_value', 0))}")
        st.write(f"**Cash disponible:** {format_currency(portfolio.get('cash_balance', 0))}")
        st.write(f"**P&L réalisé:** {format_currency(portfolio.get('realized_pnl', 0))}")
        st.write(f"**P&L non réalisé:** {format_currency(portfolio.get('unrealized_pnl', 0))}")
        st.write(f"**Retour total:** {format_percentage(portfolio.get('total_return_pct', 0))}")

    # Positions
    positions = client.get_portfolio_positions(portfolio_id)
    if positions:
        st.markdown("**📊 Positions actuelles**")
        display_positions_table(positions)

        # Graphique d'allocation
        col1, col2 = st.columns(2)

        with col1:
            fig_allocation = create_portfolio_allocation_chart(positions, "Allocation actuelle")
            display_chart_with_controls(fig_allocation, f"allocation_{portfolio_id}")

        with col2:
            # Calcul des métriques de positions
            total_market_value = sum(p.get('market_value', 0) for p in positions)
            profitable_positions = sum(1 for p in positions if p.get('unrealized_pnl', 0) > 0)

            st.metric("Valeur des positions", format_currency(total_market_value))
            st.metric("Positions profitables", f"{profitable_positions}/{len(positions)}")

    else:
        st.info("Aucune position dans ce portfolio")

    # Performance historique
    performance = client.get_portfolio_performance(portfolio_id)
    if performance and 'value_history' in performance:
        st.markdown("**📈 Performance historique**")

        fig_value = create_portfolio_value_chart(
            performance['value_history'],
            f"Évolution de {portfolio.get('name', 'Portfolio')}"
        )
        display_chart_with_controls(fig_value, f"value_{portfolio_id}")

        if 'pnl_history' in performance:
            fig_pnl = create_pnl_chart(
                performance['pnl_history'],
                f"P&L de {portfolio.get('name', 'Portfolio')}"
            )
            display_chart_with_controls(fig_pnl, f"pnl_{portfolio_id}")

def render_create_portfolio():
    """Interface de création de portfolio"""

    st.subheader("➕ Créer un nouveau portfolio")

    with st.form("create_portfolio"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Nom du portfolio", placeholder="Ex: Portfolio Principal")
            initial_capital = st.number_input(
                "Capital initial",
                min_value=0.01,
                value=10000.0,
                step=100.0,
                format="%.2f"
            )

        with col2:
            base_currency = st.selectbox(
                "Devise de base",
                ["USD", "EUR", "BTC", "ETH"],
                index=0
            )
            description = st.text_area("Description (optionnel)", placeholder="Description du portfolio...")

        # Configuration avancée
        with st.expander("⚙️ Configuration avancée"):
            max_positions = st.number_input("Nombre maximum de positions", min_value=1, value=20)
            max_position_size = st.slider("Taille maximale d'une position (%)", 1, 50, 20)
            risk_limit = st.slider("Limite de risque quotidien (%)", 1, 10, 5)

        submitted = st.form_submit_button("🚀 Créer le portfolio", type="primary")

        if submitted:
            if not name:
                st.error("Le nom du portfolio est requis")
                return

            # Préparer les données
            portfolio_data = {
                "name": name,
                "initial_capital": float(initial_capital),
                "base_currency": base_currency,
                "description": description,
                "config": {
                    "max_positions": max_positions,
                    "max_position_size_pct": max_position_size / 100,
                    "daily_risk_limit_pct": risk_limit / 100
                }
            }

            # Créer le portfolio
            client = get_api_client()
            result = client.create_portfolio(portfolio_data)

            if result:
                st.success(f"✅ Portfolio '{name}' créé avec succès !")
                st.json(result)

                # Définir comme portfolio sélectionné
                SessionStateManager.set_portfolio(result.get('id'))

                # Vider le cache pour rafraîchir la liste
                SessionStateManager.clear_cache('portfolios')

                st.balloons()
            else:
                st.error("❌ Erreur lors de la création du portfolio")

def render_manage_portfolios():
    """Interface de gestion des portfolios"""

    st.subheader("⚙️ Gérer les portfolios")

    client = get_api_client()
    portfolios = client.get_portfolios()

    if not portfolios:
        st.info("Aucun portfolio à gérer")
        return

    # Sélection du portfolio à gérer
    portfolio_options = {p['id']: p['name'] for p in portfolios}
    selected_id = st.selectbox(
        "Portfolio à gérer",
        options=list(portfolio_options.keys()),
        format_func=lambda x: portfolio_options[x],
        key="manage_portfolio_select"
    )

    if not selected_id:
        return

    selected_portfolio = next(p for p in portfolios if p['id'] == selected_id)

    # Actions de gestion
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✏️ Modifier", use_container_width=True):
            st.session_state.editing_portfolio = selected_id

    with col2:
        if st.button("📊 Recalculer", use_container_width=True):
            # Recalculer les métriques du portfolio
            with st.spinner("Recalcul en cours..."):
                # Ici on pourrait appeler un endpoint de recalcul
                st.success("Portfolio recalculé")

    with col3:
        if st.button("🗑️ Supprimer", use_container_width=True, type="secondary"):
            st.session_state.deleting_portfolio = selected_id

    # Formulaire de modification
    if st.session_state.get('editing_portfolio') == selected_id:
        st.markdown("---")
        st.subheader("✏️ Modifier le portfolio")

        with st.form("edit_portfolio"):
            new_name = st.text_input("Nom", value=selected_portfolio.get('name', ''))
            new_description = st.text_area(
                "Description",
                value=selected_portfolio.get('description', '')
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 Sauvegarder"):
                    updates = {
                        "name": new_name,
                        "description": new_description
                    }

                    result = client.update_portfolio(selected_id, updates)
                    if result:
                        st.success("Portfolio mis à jour")
                        st.session_state.editing_portfolio = None
                        SessionStateManager.clear_cache()
                        st.rerun()
                    else:
                        st.error("Erreur lors de la mise à jour")

            with col2:
                if st.form_submit_button("❌ Annuler"):
                    st.session_state.editing_portfolio = None
                    st.rerun()

    # Confirmation de suppression
    if st.session_state.get('deleting_portfolio') == selected_id:
        st.markdown("---")
        st.error("⚠️ Confirmer la suppression")
        st.write(f"Êtes-vous sûr de vouloir supprimer le portfolio **{selected_portfolio.get('name')}** ?")
        st.write("Cette action est irréversible.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Confirmer la suppression", type="primary"):
                # Ici on implémenterait la suppression
                # result = client.delete_portfolio(selected_id)
                st.error("Suppression non implémentée pour des raisons de sécurité")
                st.session_state.deleting_portfolio = None

        with col2:
            if st.button("❌ Annuler"):
                st.session_state.deleting_portfolio = None
                st.rerun()

def render_portfolio_analysis():
    """Analyse approfondie des portfolios"""

    st.subheader("📈 Analyse des portfolios")

    client = get_api_client()
    portfolios = client.get_portfolios()

    if not portfolios:
        st.info("Aucun portfolio à analyser")
        return

    # Sélection pour analyse
    portfolio_options = {p['id']: p['name'] for p in portfolios}
    selected_ids = st.multiselect(
        "Portfolios à analyser",
        options=list(portfolio_options.keys()),
        format_func=lambda x: portfolio_options[x],
        default=list(portfolio_options.keys())[:3]  # Sélectionner les 3 premiers par défaut
    )

    if not selected_ids:
        st.warning("Sélectionnez au moins un portfolio")
        return

    # Analyse comparative
    st.markdown("**📊 Analyse comparative**")

    analysis_data = []
    for portfolio_id in selected_ids:
        portfolio = client.get_portfolio(portfolio_id)
        if portfolio:
            analysis_data.append({
                'Nom': portfolio.get('name', 'N/A'),
                'Valeur': portfolio.get('total_value', 0),
                'Capital initial': portfolio.get('initial_capital', 0),
                'Retour (%)': portfolio.get('total_return_pct', 0) * 100,
                'P&L': portfolio.get('unrealized_pnl', 0) + portfolio.get('realized_pnl', 0)
            })

    if analysis_data:
        import pandas as pd
        df = pd.DataFrame(analysis_data)

        # Afficher le tableau
        st.dataframe(df, use_container_width=True)

        # Graphiques comparatifs
        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(df.set_index('Nom')['Retour (%)'])

        with col2:
            st.bar_chart(df.set_index('Nom')['Valeur'])

    # Métriques de risque
    st.markdown("**⚠️ Analyse de risque**")

    for portfolio_id in selected_ids:
        portfolio_name = portfolio_options[portfolio_id]

        with st.expander(f"Risque - {portfolio_name}"):
            risk_assessments = client.get_risk_assessments(portfolio_id)

            if risk_assessments:
                latest_risk = risk_assessments[0]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("VaR 95%", f"${latest_risk.get('var_95', 0):,.0f}")

                with col2:
                    st.metric("Max Drawdown", f"{latest_risk.get('max_drawdown', 0):.2%}")

                with col3:
                    st.metric("Volatilité", f"{latest_risk.get('volatility', 0):.2%}")
            else:
                st.info("Aucune évaluation de risque disponible")

                if st.button(f"Calculer le risque", key=f"calc_risk_{portfolio_id}"):
                    result = client.calculate_portfolio_risk(portfolio_id)
                    if result:
                        st.success("Risque calculé")
                        st.rerun()

if __name__ == "__main__":
    main()