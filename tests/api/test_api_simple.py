#!/usr/bin/env python3
"""
🚀 Test API Simple
Test rapide des endpoints de base pour diagnostiquer les problèmes
"""

import requests
import subprocess
import time
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_endpoints():
    """Test les endpoints de base sans démarrer le serveur complet."""

    # Démarrer l'API en arrière-plan
    logger.info("🚀 Démarrage du serveur API...")
    api_process = subprocess.Popen([
        "poetry", "run", "python", "start_api.py", "--port", "8003"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Attendre le démarrage
    time.sleep(8)

    base_url = "http://localhost:8003"

    try:
        # Test des endpoints de base
        endpoints_to_test = [
            ("/", "Root endpoint"),
            ("/health", "Health check"),
            ("/docs", "API Documentation"),
            ("/api/v1/market-data/symbols", "Market data symbols"),
            ("/api/v1/market-data/exchanges", "Exchanges list"),
        ]

        for endpoint, description in endpoints_to_test:
            try:
                logger.info(f"🧪 Testing {description} - {endpoint}")
                response = requests.get(f"{base_url}{endpoint}", timeout=3)

                if response.status_code == 200:
                    logger.info(f"✅ {description} - SUCCESS")
                    if endpoint == "/":
                        logger.info(f"   Response: {response.json()}")
                else:
                    logger.warning(f"⚠️ {description} - HTTP {response.status_code}")

            except Exception as e:
                logger.error(f"❌ {description} - ERROR: {e}")

        # Test simple création d'ordre (sans attendre la réponse complète)
        try:
            logger.info("🧪 Testing order creation...")
            order_data = {
                "symbol": "BTC/USD",
                "side": "BUY",
                "type": "MARKET",
                "quantity": 0.001
            }

            response = requests.post(f"{base_url}/api/v1/orders", json=order_data, timeout=3)
            if response.status_code in [200, 201]:
                logger.info("✅ Order creation - SUCCESS")
            else:
                logger.warning(f"⚠️ Order creation - HTTP {response.status_code}")
                logger.info(f"   Response: {response.text[:200]}...")

        except Exception as e:
            logger.error(f"❌ Order creation - ERROR: {e}")

    finally:
        # Arrêter le serveur
        logger.info("🛑 Arrêt du serveur API...")
        api_process.terminate()
        api_process.wait(timeout=5)

    logger.info("✅ Test rapide terminé")

if __name__ == "__main__":
    test_basic_endpoints()