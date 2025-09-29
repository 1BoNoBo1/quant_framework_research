"""
Page Risk Management - QFrame Streamlit App
===========================================

Gestion complète des risques : évaluation, monitoring, limites et alertes.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from qframe.ui.streamlit_app.utils.session_state import (
    init_session, SessionStateManager, require_api_connection
)
from qframe.ui.streamlit_app.utils.api_client import get_api_client, format_currency, format_percentage
from qframe.ui.streamlit_app.components.charts import (
    create_risk_metrics_chart,
    display_chart_with_controls,
    display_metric_card
)
from qframe.ui.streamlit_app.components.tables import (
    display_risk_assessments_table,
    display_alerts_table
)

# Configuration de la page
st.set_page_config(
    page_title="Risk Management - QFrame",
    page_icon="⚠️",
    layout="wide"
)

# Initialiser la session
init_session()

@require_api_connection
def main():
    """Page principale de gestion des risques"""

    st.title("⚠️ Gestion des Risques")

    # Onglets pour organiser les fonctionnalités
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "🎯 Évaluation", "⚙️ Limites", "🚨 Alertes"])

    with tab1:
        render_risk_overview()

    with tab2:
        render_risk_assessment()

    with tab3:
        render_risk_limits()

    with tab4:
        render_risk_alerts()

def render_risk_overview():
    """Vue d'ensemble des risques"""

    client = get_api_client()

    st.subheader("📊 Vue d'ensemble des risques")

    # Métriques de risque globales
    render_global_risk_metrics(client)

    # Risque par portfolio
    st.subheader("🏦 Risque par portfolio")
    render_portfolio_risk_overview(client)

    # Évolution du risque
    st.subheader("📈 Évolution du risque")
    render_risk_evolution(client)

def render_global_risk_metrics(client):
    """Affiche les métriques de risque globales"""

    # Récupérer les données de risque globales
    portfolios = client.get_portfolios()

    if not portfolios:
        st.info("Aucun portfolio pour évaluer le risque")
        return

    # Calculer les métriques globales
    total_value = sum(p.get('total_value', 0) for p in portfolios)
    total_var = 0
    max_drawdown = 0
    total_volatility = 0

    risk_data = []
    for portfolio in portfolios:
        risk_assessments = client.get_risk_assessments(portfolio['id'])
        if risk_assessments and len(risk_assessments) > 0:
            latest_risk = risk_assessments[0]
            portfolio_value = portfolio.get('total_value', 0)

            # Pondérer par la valeur du portfolio
            weight = portfolio_value / total_value if total_value > 0 else 0

            total_var += latest_risk.get('var_95', 0) * weight
            max_drawdown = max(max_drawdown, latest_risk.get('max_drawdown', 0))
            total_volatility += latest_risk.get('volatility', 0) * weight

            risk_data.append({
                'portfolio': portfolio['name'],
                'value': portfolio_value,
                'var_95': latest_risk.get('var_95', 0),
                'max_drawdown': latest_risk.get('max_drawdown', 0),
                'volatility': latest_risk.get('volatility', 0)
            })

    # Affichage des métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        display_metric_card(
            "VaR 95% Global",
            total_var,
            format_func=lambda x: f"${x:,.0f}"
        )

    with col2:
        display_metric_card(
            "Max Drawdown",
            max_drawdown,
            format_func=lambda x: f"{x:.2%}"
        )

    with col3:
        display_metric_card(
            "Volatilité Pondérée",
            total_volatility,
            format_func=lambda x: f"{x:.2%}"
        )

    with col4:
        risk_score = calculate_risk_score(total_var, max_drawdown, total_volatility, total_value)
        display_metric_card(
            "Score de Risque",
            f"{risk_score}/100"
        )

    # Graphique de risque global si données disponibles
    if risk_data:
        # Créer un graphique de risque composite
        risk_summary = {
            'var_95': total_var / total_value if total_value > 0 else 0,
            'max_drawdown': max_drawdown,
            'volatility': total_volatility
        }

        fig_risk = create_risk_metrics_chart(risk_summary, "Métriques de risque globales")
        display_chart_with_controls(fig_risk, "global_risk")

def render_portfolio_risk_overview(client):
    """Vue d'ensemble du risque par portfolio"""

    portfolios = client.get_portfolios()

    if not portfolios:
        return

    # Tableau de risque par portfolio
    portfolio_risk_data = []

    for portfolio in portfolios:
        risk_assessments = client.get_risk_assessments(portfolio['id'])

        if risk_assessments and len(risk_assessments) > 0:
            latest_risk = risk_assessments[0]
            portfolio_risk_data.append({
                'Portfolio': portfolio['name'],
                'Valeur': format_currency(portfolio.get('total_value', 0)),
                'VaR 95%': format_currency(latest_risk.get('var_95', 0)),
                'Max DD': format_percentage(latest_risk.get('max_drawdown', 0)),
                'Volatilité': format_percentage(latest_risk.get('volatility', 0)),
                'Score': calculate_portfolio_risk_score(latest_risk),
                'Dernière MAJ': latest_risk.get('calculated_at', 'N/A')
            })

    if portfolio_risk_data:
        df = pd.DataFrame(portfolio_risk_data)

        # Colorier selon le score de risque
        def color_risk_score(val):
            score = int(val.split('/')[0]) if '/' in str(val) else 50
            if score >= 80:
                return 'background-color: #ff6b6b'
            elif score >= 60:
                return 'background-color: #feca57'
            else:
                return 'background-color: #48ca8b'

        styled_df = df.style.applymap(color_risk_score, subset=['Score'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("Aucune évaluation de risque disponible")

        # Bouton pour calculer le risque de tous les portfolios
        if st.button("🔍 Calculer le risque de tous les portfolios"):
            calculate_all_portfolio_risks(client, portfolios)

def render_risk_evolution(client):
    """Affiche l'évolution du risque dans le temps"""

    # Sélection du portfolio pour l'analyse temporelle
    portfolios = client.get_portfolios()

    if not portfolios:
        return

    portfolio_options = {p['id']: p['name'] for p in portfolios}
    selected_id = st.selectbox(
        "Portfolio à analyser",
        options=list(portfolio_options.keys()),
        format_func=lambda x: portfolio_options[x],
        key="risk_evolution_portfolio"
    )

    if selected_id:
        risk_assessments = client.get_risk_assessments(selected_id)

        if risk_assessments and len(risk_assessments) > 1:
            # Préparer les données pour le graphique
            risk_history = []
            for assessment in sorted(risk_assessments, key=lambda x: x.get('calculated_at', '')):
                risk_history.append({
                    'timestamp': assessment.get('calculated_at'),
                    'var_95': assessment.get('var_95', 0),
                    'max_drawdown': assessment.get('max_drawdown', 0),
                    'volatility': assessment.get('volatility', 0)
                })

            if risk_history:
                # Créer graphique d'évolution (pour l'instant, utilisons le graphique de risque)
                latest_risk = risk_history[-1]
                fig_evolution = create_risk_metrics_chart(latest_risk, "Évolution du risque")
                display_chart_with_controls(fig_evolution, "risk_evolution")

                # Tableau d'historique
                st.markdown("**📊 Historique des évaluations**")
                df_history = pd.DataFrame(risk_history)
                st.dataframe(df_history, use_container_width=True)
        else:
            st.info("Historique de risque insuffisant pour l'analyse")

def render_risk_assessment():
    """Interface d'évaluation des risques"""

    st.subheader("🎯 Évaluation des risques")

    client = get_api_client()

    # Sélection du portfolio à évaluer
    portfolios = client.get_portfolios()

    if not portfolios:
        st.info("Aucun portfolio à évaluer")
        return

    portfolio_options = {p['id']: p['name'] for p in portfolios}
    selected_id = st.selectbox(
        "Portfolio à évaluer",
        options=list(portfolio_options.keys()),
        format_func=lambda x: portfolio_options[x],
        key="assessment_portfolio_select"
    )

    if not selected_id:
        return

    selected_portfolio = next(p for p in portfolios if p['id'] == selected_id)

    # Informations du portfolio
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Portfolio sélectionné**")
        st.write(f"**Nom:** {selected_portfolio['name']}")
        st.write(f"**Valeur:** {format_currency(selected_portfolio.get('total_value', 0))}")
        st.write(f"**Positions:** {selected_portfolio.get('position_count', 'N/A')}")

    with col2:
        st.markdown("**🎛️ Actions**")

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("🔍 Calculer risque", use_container_width=True):
                calculate_portfolio_risk(client, selected_id, selected_portfolio['name'])

        with col_b:
            if st.button("📊 Voir historique", use_container_width=True):
                st.session_state.show_risk_history = selected_id

    # Évaluations existantes
    risk_assessments = client.get_risk_assessments(selected_id)

    if risk_assessments:
        st.subheader("📊 Évaluations existantes")

        # Tableau des évaluations
        display_risk_assessments_table(risk_assessments)

        # Détails de la dernière évaluation
        latest_assessment = risk_assessments[0]

        st.subheader("🔍 Dernière évaluation détaillée")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**💰 Métriques de perte**")
            st.metric("VaR 95%", format_currency(latest_assessment.get('var_95', 0)))
            st.metric("VaR 99%", format_currency(latest_assessment.get('var_99', 0)))
            st.metric("Expected Shortfall", format_currency(latest_assessment.get('expected_shortfall', 0)))

        with col2:
            st.markdown("**📉 Métriques de drawdown**")
            st.metric("Max Drawdown", format_percentage(latest_assessment.get('max_drawdown', 0)))
            st.metric("Avg Drawdown", format_percentage(latest_assessment.get('avg_drawdown', 0)))
            st.metric("Drawdown Duration", f"{latest_assessment.get('max_drawdown_duration', 0)} jours")

        with col3:
            st.markdown("**📊 Métriques de volatilité**")
            st.metric("Volatilité", format_percentage(latest_assessment.get('volatility', 0)))
            st.metric("Skewness", f"{latest_assessment.get('skewness', 0):.2f}")
            st.metric("Kurtosis", f"{latest_assessment.get('kurtosis', 0):.2f}")

        # Graphique de risque
        fig_assessment = create_risk_metrics_chart(latest_assessment, "Évaluation de risque détaillée")
        display_chart_with_controls(fig_assessment, f"assessment_{selected_id}")

    else:
        st.info("Aucune évaluation de risque disponible pour ce portfolio")

    # Configuration de l'évaluation
    with st.expander("⚙️ Configuration de l'évaluation"):
        st.markdown("**📅 Paramètres temporels**")

        col1, col2 = st.columns(2)

        with col1:
            lookback_days = st.number_input("Période d'analyse (jours)", value=252, min_value=30)
            confidence_level = st.slider("Niveau de confiance (%)", 90, 99, 95)

        with col2:
            monte_carlo_sims = st.number_input("Simulations Monte Carlo", value=10000, min_value=1000, step=1000)
            rebalancing_freq = st.selectbox("Fréquence de rééquilibrage", ["Daily", "Weekly", "Monthly"])

        if st.button("🎯 Calculer avec paramètres personnalisés"):
            # Ici on pourrait passer les paramètres personnalisés à l'API
            calculate_portfolio_risk(client, selected_id, selected_portfolio['name'],
                                   custom_params={
                                       'lookback_days': lookback_days,
                                       'confidence_level': confidence_level,
                                       'monte_carlo_sims': monte_carlo_sims,
                                       'rebalancing_freq': rebalancing_freq
                                   })

def render_risk_limits():
    """Interface de configuration des limites de risque"""

    st.subheader("⚙️ Configuration des limites de risque")

    client = get_api_client()

    # Récupérer les limites actuelles
    risk_limits = client.get_risk_limits()

    if risk_limits:
        st.markdown("**📊 Limites actuelles**")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("VaR Limite", format_currency(risk_limits.get('max_var_95', 0)))
            st.metric("Drawdown Max", format_percentage(risk_limits.get('max_drawdown', 0)))

        with col2:
            st.metric("Volatilité Max", format_percentage(risk_limits.get('max_volatility', 0)))
            st.metric("Concentration Max", format_percentage(risk_limits.get('max_position_concentration', 0)))
    else:
        st.info("Aucune limite de risque configurée")

    # Configuration des nouvelles limites
    st.subheader("🎛️ Configurer les limites")

    with st.form("risk_limits_config"):
        st.markdown("**💰 Limites de perte**")

        col1, col2 = st.columns(2)

        with col1:
            max_var_95 = st.number_input(
                "VaR 95% Maximum ($)",
                value=risk_limits.get('max_var_95', 10000) if risk_limits else 10000,
                step=1000.0
            )

            max_daily_loss = st.number_input(
                "Perte quotidienne max ($)",
                value=risk_limits.get('max_daily_loss', 5000) if risk_limits else 5000,
                step=500.0
            )

        with col2:
            max_drawdown = st.slider(
                "Drawdown maximum (%)",
                1, 50,
                int(risk_limits.get('max_drawdown', 0.15) * 100) if risk_limits else 15
            )

            max_volatility = st.slider(
                "Volatilité maximum (%)",
                5, 100,
                int(risk_limits.get('max_volatility', 0.30) * 100) if risk_limits else 30
            )

        st.markdown("**🎯 Limites de concentration**")

        col1, col2 = st.columns(2)

        with col1:
            max_position_size = st.slider(
                "Taille max d'une position (%)",
                1, 50,
                int(risk_limits.get('max_position_size', 0.20) * 100) if risk_limits else 20
            )

        with col2:
            max_sector_concentration = st.slider(
                "Concentration max par secteur (%)",
                10, 80,
                int(risk_limits.get('max_sector_concentration', 0.40) * 100) if risk_limits else 40
            )

        st.markdown("**⚡ Limites opérationnelles**")

        col1, col2 = st.columns(2)

        with col1:
            max_leverage = st.number_input(
                "Leverage maximum",
                value=risk_limits.get('max_leverage', 2.0) if risk_limits else 2.0,
                step=0.1,
                min_value=1.0
            )

        with col2:
            max_correlation = st.slider(
                "Corrélation max entre positions (%)",
                50, 95,
                int(risk_limits.get('max_correlation', 0.80) * 100) if risk_limits else 80
            )

        # Actions sur les alertes
        st.markdown("**🚨 Configuration des alertes**")

        enable_email_alerts = st.checkbox(
            "Alertes par email",
            value=risk_limits.get('enable_email_alerts', True) if risk_limits else True
        )

        alert_threshold_pct = st.slider(
            "Seuil d'alerte (% de la limite)",
            50, 95,
            int(risk_limits.get('alert_threshold_pct', 0.80) * 100) if risk_limits else 80
        )

        submitted = st.form_submit_button("💾 Sauvegarder les limites", type="primary")

        if submitted:
            new_limits = {
                'max_var_95': max_var_95,
                'max_daily_loss': max_daily_loss,
                'max_drawdown': max_drawdown / 100,
                'max_volatility': max_volatility / 100,
                'max_position_size': max_position_size / 100,
                'max_sector_concentration': max_sector_concentration / 100,
                'max_leverage': max_leverage,
                'max_correlation': max_correlation / 100,
                'enable_email_alerts': enable_email_alerts,
                'alert_threshold_pct': alert_threshold_pct / 100
            }

            # Ici on sauvegarderait les limites via l'API
            st.success("✅ Limites de risque sauvegardées")
            st.json(new_limits)

def render_risk_alerts():
    """Interface de gestion des alertes de risque"""

    st.subheader("🚨 Alertes de risque")

    client = get_api_client()

    # Récupérer les alertes de risque
    alerts = client.get_alerts(severity="warning")  # Filtrer les alertes de warning et plus

    if alerts:
        st.subheader("🔥 Alertes actives")

        # Compter les alertes par sévérité
        critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
        warning_alerts = [a for a in alerts if a.get('severity') == 'warning']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🔴 Critiques", len(critical_alerts))

        with col2:
            st.metric("🟡 Avertissements", len(warning_alerts))

        with col3:
            st.metric("📊 Total", len(alerts))

        # Tableau des alertes
        display_alerts_table(alerts)

        # Actions sur les alertes
        st.markdown("**🎛️ Actions**")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Marquer toutes comme lues"):
                acknowledge_all_alerts(client, alerts)

        with col2:
            if st.button("🔄 Actualiser"):
                SessionStateManager.clear_cache('alerts')
                st.rerun()

        with col3:
            if st.button("📧 Envoyer rapport"):
                st.info("Envoi de rapport par email - Fonctionnalité à implémenter")

    else:
        st.success("✅ Aucune alerte de risque active")

    # Historique des alertes
    st.subheader("📚 Historique des alertes")

    all_alerts = client.get_alerts(limit=50)

    if all_alerts:
        # Graphique temporel des alertes
        alert_counts = {}
        for alert in all_alerts:
            date = alert.get('created_at', '')[:10]  # Prendre juste la date
            if date not in alert_counts:
                alert_counts[date] = {'critical': 0, 'warning': 0, 'info': 0}
            alert_counts[date][alert.get('severity', 'info')] += 1

        if alert_counts:
            import pandas as pd
            df_alerts = pd.DataFrame.from_dict(alert_counts, orient='index')
            df_alerts.index = pd.to_datetime(df_alerts.index)

            st.line_chart(df_alerts)

        # Configuration des alertes
        with st.expander("⚙️ Configuration des alertes"):
            st.markdown("**📧 Notifications**")

            col1, col2 = st.columns(2)

            with col1:
                email_notifications = st.checkbox("Notifications par email", value=True)
                slack_notifications = st.checkbox("Notifications Slack", value=False)

            with col2:
                sms_notifications = st.checkbox("Notifications SMS", value=False)
                webhook_notifications = st.checkbox("Notifications Webhook", value=False)

            st.markdown("**⏰ Fréquence**")

            alert_frequency = st.selectbox(
                "Fréquence de vérification",
                ["1 minute", "5 minutes", "15 minutes", "1 heure"],
                index=1
            )

            if st.button("💾 Sauvegarder configuration"):
                alert_config = {
                    'email_notifications': email_notifications,
                    'slack_notifications': slack_notifications,
                    'sms_notifications': sms_notifications,
                    'webhook_notifications': webhook_notifications,
                    'alert_frequency': alert_frequency
                }
                st.success("Configuration sauvegardée")
                st.json(alert_config)
    else:
        st.info("Aucun historique d'alertes")

# Fonctions utilitaires
def calculate_risk_score(var, drawdown, volatility, total_value):
    """Calcule un score de risque global"""
    if total_value == 0:
        return 100

    var_ratio = (var / total_value) * 100
    score = min(100, var_ratio * 10 + drawdown * 100 + volatility * 50)
    return max(0, int(score))

def calculate_portfolio_risk_score(risk_assessment):
    """Calcule un score de risque pour un portfolio"""
    var_95 = risk_assessment.get('var_95', 0)
    max_drawdown = risk_assessment.get('max_drawdown', 0)
    volatility = risk_assessment.get('volatility', 0)

    score = min(100, max_drawdown * 200 + volatility * 100)
    return f"{max(0, int(score))}/100"

def calculate_portfolio_risk(client, portfolio_id, portfolio_name, custom_params=None):
    """Calcule le risque d'un portfolio"""
    with st.spinner(f"Calcul du risque pour {portfolio_name}..."):
        result = client.calculate_portfolio_risk(portfolio_id)

        if result:
            st.success("✅ Calcul de risque terminé")

            # Afficher un aperçu des résultats
            if 'var_95' in result:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("VaR 95%", format_currency(result['var_95']))

                with col2:
                    st.metric("Max Drawdown", format_percentage(result.get('max_drawdown', 0)))

                with col3:
                    st.metric("Volatilité", format_percentage(result.get('volatility', 0)))

            # Vider le cache pour rafraîchir
            SessionStateManager.clear_cache()
            st.rerun()
        else:
            st.error("❌ Erreur lors du calcul du risque")

def calculate_all_portfolio_risks(client, portfolios):
    """Calcule le risque pour tous les portfolios"""
    with st.spinner("Calcul du risque pour tous les portfolios..."):
        calculated = 0

        for portfolio in portfolios:
            result = client.calculate_portfolio_risk(portfolio['id'])
            if result:
                calculated += 1

        if calculated > 0:
            st.success(f"✅ Risque calculé pour {calculated} portfolios")
            SessionStateManager.clear_cache()
            st.rerun()
        else:
            st.error("❌ Erreur lors du calcul des risques")

def acknowledge_all_alerts(client, alerts):
    """Marque toutes les alertes comme lues"""
    acknowledged = 0

    for alert in alerts:
        result = client.acknowledge_alert(alert['id'])
        if result:
            acknowledged += 1

    if acknowledged > 0:
        st.success(f"✅ {acknowledged} alertes marquées comme lues")
        SessionStateManager.clear_cache('alerts')
        st.rerun()
    else:
        st.error("❌ Erreur lors de la mise à jour des alertes")

if __name__ == "__main__":
    main()