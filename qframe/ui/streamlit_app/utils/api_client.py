"""
QFrame API Client for Streamlit
===============================

Client API pour communiquer avec le backend QFrame depuis l'interface Streamlit.
"""

import os
import requests
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class QFrameAPIClient:
    """Client API pour QFrame Backend"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("QFRAME_API_URL", "http://localhost:8000")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Effectue une requête HTTP avec gestion d'erreur"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            st.error(f"Erreur API: {e}")
            return None

    # ===============================
    # Health & System
    # ===============================

    def get_health(self) -> Optional[Dict]:
        """Vérifie l'état de santé du système"""
        return self._request("GET", "/health")

    def get_system_info(self) -> Optional[Dict]:
        """Récupère les informations système"""
        return self._request("GET", "/system/info")

    # ===============================
    # Dashboard & Metrics
    # ===============================

    def get_dashboard_data(self) -> Optional[Dict]:
        """Récupère les données du dashboard"""
        return self._request("GET", "/dashboard")

    def get_metrics(self) -> Optional[Dict]:
        """Récupère les métriques système"""
        return self._request("GET", "/metrics")

    def get_performance_metrics(self) -> Optional[Dict]:
        """Récupère les métriques de performance"""
        return self._request("GET", "/metrics/performance")

    # ===============================
    # Portfolio Management
    # ===============================

    def get_portfolios(self) -> Optional[List[Dict]]:
        """Récupère la liste des portfolios"""
        return self._request("GET", "/portfolios")

    def get_portfolio(self, portfolio_id: str) -> Optional[Dict]:
        """Récupère un portfolio spécifique"""
        return self._request("GET", f"/portfolios/{portfolio_id}")

    def create_portfolio(self, portfolio_data: Dict) -> Optional[Dict]:
        """Crée un nouveau portfolio"""
        return self._request("POST", "/portfolios", json=portfolio_data)

    def update_portfolio(self, portfolio_id: str, updates: Dict) -> Optional[Dict]:
        """Met à jour un portfolio"""
        return self._request("PUT", f"/portfolios/{portfolio_id}", json=updates)

    def get_portfolio_positions(self, portfolio_id: str) -> Optional[List[Dict]]:
        """Récupère les positions d'un portfolio"""
        return self._request("GET", f"/portfolios/{portfolio_id}/positions")

    def get_portfolio_performance(self, portfolio_id: str) -> Optional[Dict]:
        """Récupère les performances d'un portfolio"""
        return self._request("GET", f"/portfolios/{portfolio_id}/performance")

    # ===============================
    # Strategy Management
    # ===============================

    def get_strategies(self) -> Optional[List[Dict]]:
        """Récupère la liste des stratégies"""
        return self._request("GET", "/api/v1/strategies")

    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Récupère une stratégie spécifique"""
        return self._request("GET", f"/api/v1/strategies/{strategy_id}")

    def create_strategy(self, strategy_data: Dict) -> Optional[Dict]:
        """Crée une nouvelle stratégie"""
        return self._request("POST", "/api/v1/strategies", json=strategy_data)

    def start_strategy(self, strategy_id: str, config: Dict = None) -> Optional[Dict]:
        """Démarre une stratégie"""
        return self._request("POST", f"/api/v1/strategies/{strategy_id}/start", json=config or {})

    def stop_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Arrête une stratégie"""
        return self._request("POST", f"/api/v1/strategies/{strategy_id}/stop")

    def get_strategy_performance(self, strategy_id: str, period: str = None) -> Optional[Dict]:
        """Récupère les performances d'une stratégie"""
        params = {"period": period} if period else {}
        return self._request("GET", f"/api/v1/strategies/{strategy_id}/performance", params=params)

    # ===============================
    # Order Management
    # ===============================

    def get_orders(self, portfolio_id: str = None, limit: int = 100) -> Optional[List[Dict]]:
        """Récupère la liste des ordres"""
        params = {"per_page": limit}
        if portfolio_id:
            params["portfolio_id"] = portfolio_id
        return self._request("GET", "/api/v1/orders", params=params)

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Récupère un ordre spécifique"""
        return self._request("GET", f"/api/v1/orders/{order_id}")

    def create_order(self, order_data: Dict) -> Optional[Dict]:
        """Crée un nouvel ordre"""
        return self._request("POST", "/api/v1/orders", json=order_data)

    def cancel_order(self, order_id: str) -> Optional[Dict]:
        """Annule un ordre"""
        return self._request("DELETE", f"/api/v1/orders/{order_id}")

    # ===============================
    # Risk Management
    # ===============================

    def get_risk_metrics(self) -> Optional[Dict]:
        """Récupère les métriques de risque actuelles"""
        return self._request("GET", "/api/v1/risk/metrics")

    def get_risk_alerts(self, severity: str = None) -> Optional[List[Dict]]:
        """Récupère les alertes de risque"""
        params = {"severity": severity} if severity else {}
        return self._request("GET", "/api/v1/risk/alerts", params=params)

    def calculate_var(self, confidence_level: float = 0.95, method: str = "monte_carlo") -> Optional[Dict]:
        """Calcule la Value at Risk"""
        params = {
            "confidence_level": confidence_level,
            "method": method
        }
        return self._request("GET", "/api/v1/risk/var", params=params)

    # ===============================
    # Positions Management
    # ===============================

    def get_positions(self, page: int = 1, per_page: int = 20) -> Optional[Dict]:
        """Récupère la liste des positions"""
        params = {"page": page, "per_page": per_page}
        return self._request("GET", "/api/v1/positions", params=params)

    def get_position(self, position_id: str) -> Optional[Dict]:
        """Récupère une position spécifique"""
        return self._request("GET", f"/api/v1/positions/{position_id}")

    def close_position(self, position_id: str, close_price: float = None) -> Optional[Dict]:
        """Ferme une position"""
        params = {"close_price": close_price} if close_price else {}
        return self._request("DELETE", f"/api/v1/positions/{position_id}", params=params)

    def get_portfolio_summary(self) -> Optional[Dict]:
        """Récupère le résumé du portefeuille"""
        return self._request("GET", "/api/v1/positions/portfolio/summary")

    def get_portfolio_allocation(self) -> Optional[Dict]:
        """Récupère l'allocation du portefeuille"""
        return self._request("GET", "/api/v1/positions/portfolio/allocation")

    # ===============================
    # Backtesting
    # ===============================

    def get_backtests(self, strategy_id: str = None, page: int = 1, per_page: int = 20) -> Optional[Dict]:
        """Récupère la liste des backtests"""
        params = {"page": page, "per_page": per_page}
        if strategy_id:
            params["strategy_id"] = strategy_id
        return self._request("GET", "/api/v1/strategies/backtests", params=params)

    def get_backtest(self, backtest_id: str) -> Optional[Dict]:
        """Récupère un backtest spécifique"""
        return self._request("GET", f"/api/v1/strategies/backtests/{backtest_id}")

    def create_backtest(self, backtest_config: Dict) -> Optional[Dict]:
        """Lance un nouveau backtest"""
        return self._request("POST", "/api/v1/strategies/backtests", json=backtest_config)

    def get_backtest_results(self, backtest_id: str) -> Optional[Dict]:
        """Récupère les résultats d'un backtest"""
        return self._request("GET", f"/api/v1/strategies/backtests/{backtest_id}/results")

    def cancel_backtest(self, backtest_id: str) -> Optional[Dict]:
        """Annule un backtest en cours"""
        return self._request("POST", f"/api/v1/strategies/backtests/{backtest_id}/cancel")

    def get_running_strategies(self) -> Optional[Dict]:
        """Récupère les stratégies en cours d'exécution"""
        return self._request("GET", "/api/v1/strategies/running")

    # ===============================
    # Market Data
    # ===============================

    def get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> Optional[Dict]:
        """Récupère les données de marché OHLCV"""
        params = {
            "timeframe": timeframe,
            "limit": limit
        }
        return self._request("GET", f"/api/v1/market-data/ohlcv/{symbol}", params=params)

    def get_available_symbols(self) -> Optional[List[str]]:
        """Récupère la liste des symboles disponibles"""
        return self._request("GET", "/api/v1/market-data/symbols")

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Récupère le prix actuel d'un symbole"""
        return self._request("GET", f"/api/v1/market-data/price/{symbol}")

    def get_multiple_prices(self, symbols: List[str]) -> Optional[List[Dict]]:
        """Récupère les prix de plusieurs symboles"""
        symbols_str = ",".join(symbols)
        return self._request("GET", "/api/v1/market-data/prices", params={"symbols": symbols_str})

    def get_ticker_data(self, symbol: str) -> Optional[Dict]:
        """Récupère les données ticker complètes"""
        return self._request("GET", f"/api/v1/market-data/ticker/{symbol}")

    def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Récupère le carnet d'ordres"""
        return self._request("GET", f"/api/v1/market-data/depth/{symbol}", params={"limit": limit})

    def get_recent_trades(self, symbol: str, limit: int = 50) -> Optional[List[Dict]]:
        """Récupère les trades récents"""
        return self._request("GET", f"/api/v1/market-data/trades/{symbol}", params={"limit": limit})

    def get_supported_exchanges(self) -> Optional[List[Dict]]:
        """Récupère la liste des exchanges supportés"""
        return self._request("GET", "/api/v1/market-data/exchanges")

    def get_market_stats(self, symbol: str, period: str = "24h") -> Optional[Dict]:
        """Récupère les statistiques de marché"""
        return self._request("GET", f"/api/v1/market-data/stats/{symbol}", params={"period": period})

    # ===============================
    # Alerts & Notifications
    # ===============================

    def get_alerts(self, severity: str = None, limit: int = 50) -> Optional[List[Dict]]:
        """Récupère les alertes de risque"""
        params = {"limit": limit}
        if severity:
            params["severity"] = severity
        return self._request("GET", "/api/v1/risk/alerts", params=params)


@st.cache_resource
def get_api_client() -> QFrameAPIClient:
    """
    Récupère une instance mise en cache du client API
    """
    return QFrameAPIClient()


# Fonctions utilitaires pour Streamlit
def check_api_connection() -> bool:
    """
    Vérifie la connexion à l'API et affiche le statut
    """
    client = get_api_client()
    health = client.get_health()

    if health:
        st.success("✅ Connexion API établie")
        return True
    else:
        st.error("❌ Impossible de se connecter à l'API QFrame")
        st.info("Vérifiez que le serveur QFrame est démarré sur http://localhost:8000")
        return False


def display_api_error(response: Optional[Dict], context: str = "API call"):
    """
    Affiche une erreur API avec contexte
    """
    if response is None:
        st.error(f"❌ {context} failed - No response from server")
    elif "error" in response:
        st.error(f"❌ {context} failed: {response['error']}")
    else:
        st.success(f"✅ {context} successful")


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Formate un montant en devise
    """
    return f"{amount:,.2f} {currency}"


def format_percentage(value: float) -> str:
    """
    Formate un pourcentage
    """
    return f"{value:.2%}"


def format_datetime(dt: datetime) -> str:
    """
    Formate une date/heure
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")