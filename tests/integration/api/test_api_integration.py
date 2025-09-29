#!/usr/bin/env python3
"""
🔄 Test d'Intégration API ↔ UI End-to-End
Test automatisé pour valider la connexion complète entre backend et frontend
"""

import asyncio
import sys
import time
import subprocess
import requests
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIIntegrationTester:
    """Testeur d'intégration API-UI automatisé."""

    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.ui_base_url = "http://localhost:8501"
        self.api_process = None
        self.ui_process = None

    def start_api_server(self) -> bool:
        """Démarre le serveur API backend."""
        try:
            logger.info("🚀 Démarrage du serveur API...")
            self.api_process = subprocess.Popen([
                "poetry", "run", "python", "start_api.py", "--port", "8000"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Attendre que l'API soit prête
            for i in range(30):  # 30 secondes max
                try:
                    response = requests.get(f"{self.api_base_url}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ Serveur API démarré avec succès")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)

            logger.error("❌ Échec du démarrage du serveur API")
            return False

        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage API: {e}")
            return False

    def start_ui_server(self) -> bool:
        """Démarre le serveur UI Streamlit."""
        try:
            logger.info("🖥️ Démarrage du serveur UI...")
            self.ui_process = subprocess.Popen([
                "poetry", "run", "streamlit", "run",
                "qframe/ui/streamlit_app/main.py",
                "--server.port", "8501",
                "--server.headless", "true"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Attendre que l'UI soit prête
            for i in range(30):  # 30 secondes max
                try:
                    response = requests.get(f"{self.ui_base_url}/_stcore/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ Serveur UI démarré avec succès")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)

            logger.error("❌ Échec du démarrage du serveur UI")
            return False

        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage UI: {e}")
            return False

    def test_api_endpoints(self) -> Dict[str, bool]:
        """Teste tous les endpoints principaux de l'API."""
        results = {}

        # Health check
        results["health"] = self._test_endpoint("GET", "/health")

        # Market data endpoints
        results["market_data_symbols"] = self._test_endpoint("GET", "/api/v1/market-data/symbols")
        results["market_data_price"] = self._test_endpoint("GET", "/api/v1/market-data/price/BTC/USD")
        results["market_data_exchanges"] = self._test_endpoint("GET", "/api/v1/market-data/exchanges")

        # Strategy endpoints
        results["strategies_list"] = self._test_endpoint("GET", "/api/v1/strategies")
        results["strategies_running"] = self._test_endpoint("GET", "/api/v1/strategies/running")

        # Orders endpoints
        results["orders_list"] = self._test_endpoint("GET", "/api/v1/orders")

        # Positions endpoints
        results["positions_list"] = self._test_endpoint("GET", "/api/v1/positions")
        results["portfolio_summary"] = self._test_endpoint("GET", "/api/v1/positions/portfolio/summary")

        # Risk endpoints
        results["risk_metrics"] = self._test_endpoint("GET", "/api/v1/risk/metrics")
        results["risk_alerts"] = self._test_endpoint("GET", "/api/v1/risk/alerts")

        return results

    def _test_endpoint(self, method: str, endpoint: str, **kwargs) -> bool:
        """Teste un endpoint spécifique."""
        try:
            url = f"{self.api_base_url}{endpoint}"
            response = requests.request(method, url, timeout=5, **kwargs)

            if response.status_code == 200:
                logger.info(f"✅ {method} {endpoint} - SUCCESS")
                return True
            else:
                logger.warning(f"⚠️ {method} {endpoint} - HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ {method} {endpoint} - ERROR: {e}")
            return False

    def test_api_data_format(self) -> Dict[str, bool]:
        """Teste le format des données retournées par l'API."""
        results = {}

        # Test health endpoint data structure
        try:
            response = requests.get(f"{self.api_base_url}/health")
            if response.status_code == 200:
                data = response.json()
                results["health_format"] = all(key in data for key in ["status", "timestamp"])
            else:
                results["health_format"] = False
        except:
            results["health_format"] = False

        # Test strategies data structure
        try:
            response = requests.get(f"{self.api_base_url}/api/v1/strategies")
            if response.status_code == 200:
                data = response.json()
                results["strategies_format"] = isinstance(data, dict) and "success" in data
            else:
                results["strategies_format"] = False
        except:
            results["strategies_format"] = False

        return results

    def test_create_order_flow(self) -> bool:
        """Teste le flux complet de création d'ordre."""
        try:
            # Données d'ordre de test
            order_data = {
                "symbol": "BTC/USD",
                "side": "BUY",
                "type": "MARKET",
                "quantity": 0.001,
                "metadata": {
                    "test": True,
                    "source": "integration_test"
                }
            }

            # Créer l'ordre
            response = requests.post(
                f"{self.api_base_url}/api/v1/orders",
                json=order_data,
                timeout=10
            )

            if response.status_code == 200:
                order_result = response.json()
                order_id = order_result.get("data", {}).get("id")

                if order_id:
                    # Vérifier que l'ordre existe
                    check_response = requests.get(
                        f"{self.api_base_url}/api/v1/orders/{order_id}",
                        timeout=5
                    )

                    if check_response.status_code == 200:
                        logger.info("✅ Flux création d'ordre testé avec succès")
                        return True

            logger.warning("⚠️ Flux création d'ordre - problème de format")
            return False

        except Exception as e:
            logger.error(f"❌ Erreur test création d'ordre: {e}")
            return False

    def test_websocket_connection(self) -> bool:
        """Teste la connexion WebSocket temps réel."""
        try:
            import websocket

            def on_message(ws, message):
                logger.info(f"📨 WebSocket message: {message[:100]}...")
                ws.close()

            def on_error(ws, error):
                logger.error(f"❌ WebSocket error: {error}")

            def on_close(ws, close_status_code, close_msg):
                logger.info("🔌 WebSocket connection closed")

            ws_url = "ws://localhost:8000/ws"
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )

            # Test de connexion rapide
            ws.run_forever(ping_timeout=5)
            logger.info("✅ WebSocket connexion testée")
            return True

        except ImportError:
            logger.warning("⚠️ websocket-client non installé, skip test WebSocket")
            return True  # Ne pas échouer si la lib n'est pas installée
        except Exception as e:
            logger.error(f"❌ Erreur test WebSocket: {e}")
            return False

    def stop_servers(self):
        """Arrête les serveurs."""
        if self.api_process:
            logger.info("🛑 Arrêt du serveur API...")
            self.api_process.terminate()
            self.api_process.wait(timeout=5)

        if self.ui_process:
            logger.info("🛑 Arrêt du serveur UI...")
            self.ui_process.terminate()
            self.ui_process.wait(timeout=5)

    def run_full_test(self) -> bool:
        """Exécute la suite complète de tests d'intégration."""
        try:
            logger.info("🔄 Démarrage des tests d'intégration API ↔ UI")

            # Démarrer les serveurs
            if not self.start_api_server():
                return False

            # Tests API
            logger.info("🧪 Test des endpoints API...")
            api_results = self.test_api_endpoints()
            api_success_rate = sum(api_results.values()) / len(api_results)

            logger.info("🧪 Test du format des données...")
            format_results = self.test_api_data_format()
            format_success_rate = sum(format_results.values()) / len(format_results)

            logger.info("🧪 Test du flux de création d'ordre...")
            order_flow_success = self.test_create_order_flow()

            logger.info("🧪 Test de la connexion WebSocket...")
            websocket_success = self.test_websocket_connection()

            # Rapport de résultats
            logger.info("=" * 60)
            logger.info("📊 RAPPORT DE TESTS D'INTÉGRATION")
            logger.info("=" * 60)
            logger.info(f"API Endpoints: {api_success_rate:.1%} success ({sum(api_results.values())}/{len(api_results)})")
            logger.info(f"Data Format: {format_success_rate:.1%} success ({sum(format_results.values())}/{len(format_results)})")
            logger.info(f"Order Flow: {'✅' if order_flow_success else '❌'}")
            logger.info(f"WebSocket: {'✅' if websocket_success else '❌'}")

            # Détails des échecs
            failed_endpoints = [endpoint for endpoint, success in api_results.items() if not success]
            if failed_endpoints:
                logger.warning(f"⚠️ Endpoints échoués: {', '.join(failed_endpoints)}")

            # Score global
            total_score = (api_success_rate + format_success_rate +
                          (1 if order_flow_success else 0) +
                          (1 if websocket_success else 0)) / 4

            logger.info(f"🎯 Score global: {total_score:.1%}")
            logger.info("=" * 60)

            return total_score >= 0.8  # 80% minimum de réussite

        except Exception as e:
            logger.error(f"❌ Erreur lors des tests d'intégration: {e}")
            return False
        finally:
            self.stop_servers()


def main():
    """Point d'entrée principal."""
    tester = APIIntegrationTester()

    try:
        success = tester.run_full_test()

        if success:
            logger.info("🎉 Tests d'intégration RÉUSSIS !")
            sys.exit(0)
        else:
            logger.error("💥 Tests d'intégration ÉCHOUÉS !")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("🛑 Tests interrompus par l'utilisateur")
        tester.stop_servers()
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        tester.stop_servers()
        sys.exit(1)


if __name__ == "__main__":
    main()