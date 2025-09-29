"""
Session State Management for QFrame Streamlit App
=================================================

Gestion centralisée de l'état de session pour l'application Streamlit.
"""

import streamlit as st
from typing import Any, Dict, Optional
from datetime import datetime


class SessionStateManager:
    """Gestionnaire d'état de session pour Streamlit"""

    @staticmethod
    def init_session_state():
        """Initialise l'état de session par défaut"""

        # Configuration de l'API
        if 'api_connected' not in st.session_state:
            st.session_state.api_connected = False

        # Données mises en cache
        if 'cached_data' not in st.session_state:
            st.session_state.cached_data = {}

        if 'last_update' not in st.session_state:
            st.session_state.last_update = {}

        # Portfolio sélectionné
        if 'selected_portfolio_id' not in st.session_state:
            st.session_state.selected_portfolio_id = None

        # Stratégie sélectionnée
        if 'selected_strategy_id' not in st.session_state:
            st.session_state.selected_strategy_id = None

        # Paramètres d'affichage
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True

        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 30  # secondes

        if 'theme' not in st.session_state:
            st.session_state.theme = "dark"

        # Filtres
        if 'date_range' not in st.session_state:
            st.session_state.date_range = "24h"

        if 'symbol_filter' not in st.session_state:
            st.session_state.symbol_filter = "All"

        # Alertes
        if 'unread_alerts' not in st.session_state:
            st.session_state.unread_alerts = 0

        # Navigation
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Dashboard"

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Récupère une valeur de l'état de session"""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Définit une valeur dans l'état de session"""
        st.session_state[key] = value

    @staticmethod
    def cache_data(key: str, data: Any, ttl_seconds: int = 300) -> None:
        """Met en cache des données avec TTL"""
        st.session_state.cached_data[key] = data
        st.session_state.last_update[key] = datetime.now()

    @staticmethod
    def get_cached_data(key: str, ttl_seconds: int = 300) -> Optional[Any]:
        """Récupère des données mises en cache si elles ne sont pas expirées"""
        if key not in st.session_state.cached_data:
            return None

        if key not in st.session_state.last_update:
            return None

        # Vérifier l'expiration
        last_update = st.session_state.last_update[key]
        if (datetime.now() - last_update).total_seconds() > ttl_seconds:
            # Données expirées
            del st.session_state.cached_data[key]
            del st.session_state.last_update[key]
            return None

        return st.session_state.cached_data[key]

    @staticmethod
    def clear_cache(key: str = None) -> None:
        """Vide le cache (tout ou une clé spécifique)"""
        if key:
            st.session_state.cached_data.pop(key, None)
            st.session_state.last_update.pop(key, None)
        else:
            st.session_state.cached_data.clear()
            st.session_state.last_update.clear()

    @staticmethod
    def set_portfolio(portfolio_id: str) -> None:
        """Définit le portfolio sélectionné"""
        SessionStateManager.set('selected_portfolio_id', portfolio_id)
        # Vider le cache des données liées au portfolio
        SessionStateManager.clear_cache('portfolio_positions')
        SessionStateManager.clear_cache('portfolio_performance')

    @staticmethod
    def get_selected_portfolio() -> Optional[str]:
        """Récupère l'ID du portfolio sélectionné"""
        return SessionStateManager.get('selected_portfolio_id')

    @staticmethod
    def set_strategy(strategy_id: str) -> None:
        """Définit la stratégie sélectionnée"""
        SessionStateManager.set('selected_strategy_id', strategy_id)
        # Vider le cache des données liées à la stratégie
        SessionStateManager.clear_cache('strategy_performance')

    @staticmethod
    def get_selected_strategy() -> Optional[str]:
        """Récupère l'ID de la stratégie sélectionnée"""
        return SessionStateManager.get('selected_strategy_id')

    @staticmethod
    def toggle_auto_refresh() -> None:
        """Bascule le rafraîchissement automatique"""
        current = SessionStateManager.get('auto_refresh', True)
        SessionStateManager.set('auto_refresh', not current)

    @staticmethod
    def set_refresh_interval(interval: int) -> None:
        """Définit l'intervalle de rafraîchissement"""
        SessionStateManager.set('refresh_interval', interval)

    @staticmethod
    def increment_alerts() -> None:
        """Incrémente le compteur d'alertes non lues"""
        current = SessionStateManager.get('unread_alerts', 0)
        SessionStateManager.set('unread_alerts', current + 1)

    @staticmethod
    def clear_alerts() -> None:
        """Remet à zéro le compteur d'alertes"""
        SessionStateManager.set('unread_alerts', 0)

    @staticmethod
    def get_display_config() -> Dict[str, Any]:
        """Récupère la configuration d'affichage"""
        return {
            'auto_refresh': SessionStateManager.get('auto_refresh', True),
            'refresh_interval': SessionStateManager.get('refresh_interval', 30),
            'theme': SessionStateManager.get('theme', 'dark'),
            'date_range': SessionStateManager.get('date_range', '24h'),
            'symbol_filter': SessionStateManager.get('symbol_filter', 'All')
        }

    @staticmethod
    def update_display_config(config: Dict[str, Any]) -> None:
        """Met à jour la configuration d'affichage"""
        for key, value in config.items():
            SessionStateManager.set(key, value)

    @staticmethod
    def get_navigation_state() -> Dict[str, Any]:
        """Récupère l'état de navigation"""
        return {
            'current_page': SessionStateManager.get('current_page', 'Dashboard'),
            'selected_portfolio': SessionStateManager.get_selected_portfolio(),
            'selected_strategy': SessionStateManager.get_selected_strategy()
        }

    @staticmethod
    def set_current_page(page: str) -> None:
        """Définit la page actuelle"""
        SessionStateManager.set('current_page', page)

    @staticmethod
    def debug_session_state() -> Dict[str, Any]:
        """Retourne l'état de session pour débogage"""
        return {
            'session_keys': list(st.session_state.keys()),
            'cached_data_keys': list(st.session_state.get('cached_data', {}).keys()),
            'selected_portfolio': SessionStateManager.get_selected_portfolio(),
            'selected_strategy': SessionStateManager.get_selected_strategy(),
            'api_connected': SessionStateManager.get('api_connected', False),
            'current_page': SessionStateManager.get('current_page', 'Dashboard')
        }


# Fonction d'initialisation automatique
def init_session():
    """Fonction d'initialisation à appeler au début de chaque page"""
    SessionStateManager.init_session_state()


# Décorateur pour les fonctions qui nécessitent une connexion API
def require_api_connection(func):
    """Décorateur qui vérifie la connexion API avant d'exécuter une fonction"""
    def wrapper(*args, **kwargs):
        if not SessionStateManager.get('api_connected', False):
            st.error("❌ Connexion API requise")
            st.stop()
        return func(*args, **kwargs)
    return wrapper


# Décorateur pour les fonctions qui nécessitent un portfolio sélectionné
def require_portfolio_selection(func):
    """Décorateur qui vérifie qu'un portfolio est sélectionné"""
    def wrapper(*args, **kwargs):
        if not SessionStateManager.get_selected_portfolio():
            st.warning("⚠️ Veuillez sélectionner un portfolio")
            st.stop()
        return func(*args, **kwargs)
    return wrapper