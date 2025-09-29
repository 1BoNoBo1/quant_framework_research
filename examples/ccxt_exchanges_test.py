#!/usr/bin/env python3
"""
Test du CCXTProvider avec différents exchanges
=============================================

Exemple démonstrant l'utilisation du CCXTProvider universel
pour connecter à différents exchanges crypto via CCXT.
"""

import asyncio
import logging
from typing import Dict, Any

from qframe.infrastructure.data.ccxt_provider import CCXTProvider, CCXTProviderFactory


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_exchange_connection(exchange_name: str, provider: CCXTProvider) -> Dict[str, Any]:
    """Teste la connexion et capacités d'un exchange."""
    results = {
        "exchange": exchange_name,
        "connected": False,
        "symbols_count": 0,
        "capabilities": {},
        "sample_ticker": None,
        "errors": []
    }

    try:
        # Test de connexion
        logger.info(f"🔗 Connexion à {exchange_name}...")
        connected = await provider.connect()
        results["connected"] = connected

        if not connected:
            results["errors"].append("Failed to connect")
            return results

        # Obtenir les capacités
        capabilities = provider.get_capabilities()
        results["capabilities"] = capabilities
        results["symbols_count"] = capabilities.get("symbols", 0)

        # Test des symboles disponibles
        symbols = await provider.get_symbols()
        if symbols:
            logger.info(f"📊 {exchange_name}: {len(symbols)} symboles disponibles")

        # Tester ticker sur BTC/USDT si disponible
        btc_symbols = [s for s in symbols if "BTC" in s and "USDT" in s]
        if btc_symbols:
            symbol = btc_symbols[0]  # Premier symbole BTC/USDT trouvé

            logger.info(f"📈 Test ticker pour {symbol}...")
            ticker = await provider.get_ticker_24hr(symbol)

            if ticker and ticker.last > 0:
                results["sample_ticker"] = {
                    "symbol": symbol,
                    "price": float(ticker.last),
                    "volume": float(ticker.volume_24h),
                    "change": float(ticker.change_24h)
                }
                logger.info(f"✅ {symbol}: ${ticker.last} (24h: {ticker.change_24h}%)")

        # Test server time
        server_time = await provider.get_server_time()
        logger.info(f"🕐 Server time: {server_time}")

        # Informations sur l'exchange
        exchange_info = await provider.get_exchange_info()
        logger.info(f"ℹ️ Exchange info: {exchange_info.get('name', 'Unknown')}")

    except Exception as e:
        error_msg = f"Error testing {exchange_name}: {str(e)}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    finally:
        # Déconnexion
        try:
            await provider.disconnect()
        except Exception as e:
            logger.warning(f"Warning during disconnect: {e}")

    return results


async def test_multiple_exchanges():
    """Teste plusieurs exchanges en parallèle."""

    logger.info("🚀 Démarrage des tests multi-exchange CCXT")

    # Liste des exchanges à tester (sandbox/testnet mode)
    exchanges_to_test = [
        "binance",
        "coinbase",
        "kraken",
        "okx",
        "bybit"
    ]

    # Créer les providers
    providers = {}
    for exchange in exchanges_to_test:
        try:
            provider = CCXTProviderFactory.create_provider(
                exchange,
                sandbox=True,  # Mode sandbox pour tests
                rate_limit=True
            )
            providers[exchange] = provider
        except Exception as e:
            logger.error(f"❌ Impossible de créer provider pour {exchange}: {e}")

    # Tester tous les exchanges en parallèle
    tasks = []
    for exchange, provider in providers.items():
        task = test_exchange_connection(exchange, provider)
        tasks.append(task)

    # Attendre tous les résultats
    logger.info(f"⏳ Test de {len(providers)} exchanges en parallèle...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Analyser les résultats
    successful_exchanges = []
    failed_exchanges = []

    for i, result in enumerate(results):
        exchange = exchanges_to_test[i]

        if isinstance(result, Exception):
            logger.error(f"❌ {exchange}: Exception - {result}")
            failed_exchanges.append(exchange)
        elif isinstance(result, dict):
            if result["connected"]:
                successful_exchanges.append(exchange)
                logger.info(f"✅ {exchange}: Connecté ({result['symbols_count']} symboles)")

                # Afficher ticker si disponible
                if result["sample_ticker"]:
                    ticker = result["sample_ticker"]
                    logger.info(f"   💰 {ticker['symbol']}: ${ticker['price']}")
            else:
                failed_exchanges.append(exchange)
                logger.error(f"❌ {exchange}: Échec de connexion")
                if result["errors"]:
                    for error in result["errors"]:
                        logger.error(f"      {error}")

    # Rapport final
    logger.info("=" * 60)
    logger.info("📋 RAPPORT FINAL")
    logger.info("=" * 60)
    logger.info(f"✅ Exchanges connectés: {len(successful_exchanges)}")
    if successful_exchanges:
        for exchange in successful_exchanges:
            logger.info(f"   - {exchange}")

    logger.info(f"❌ Exchanges échoués: {len(failed_exchanges)}")
    if failed_exchanges:
        for exchange in failed_exchanges:
            logger.info(f"   - {exchange}")

    success_rate = len(successful_exchanges) / len(exchanges_to_test) * 100
    logger.info(f"📊 Taux de réussite: {success_rate:.1f}%")

    return {
        "total_tested": len(exchanges_to_test),
        "successful": successful_exchanges,
        "failed": failed_exchanges,
        "success_rate": success_rate,
        "detailed_results": [r for r in results if isinstance(r, dict)]
    }


async def demo_factory_methods():
    """Démonstration des méthodes factory."""
    logger.info("\n🏭 DÉMONSTRATION FACTORY METHODS")
    logger.info("=" * 50)

    # Liste des exchanges supportés
    supported = CCXTProviderFactory.get_all_supported_exchanges()
    logger.info(f"📊 Exchanges supportés: {len(supported)}")
    for exchange in supported[:10]:  # Afficher les 10 premiers
        logger.info(f"   - {exchange}")
    if len(supported) > 10:
        logger.info(f"   ... et {len(supported) - 10} autres")

    # Test des méthodes factory spécifiques
    logger.info("\n🧪 Test des factory methods...")

    # Binance
    try:
        binance = CCXTProviderFactory.create_binance(sandbox=True)
        logger.info(f"✅ Binance factory: {binance.name}")
    except Exception as e:
        logger.error(f"❌ Binance factory: {e}")

    # Coinbase
    try:
        coinbase = CCXTProviderFactory.create_coinbase(sandbox=True)
        logger.info(f"✅ Coinbase factory: {coinbase.name}")
    except Exception as e:
        logger.error(f"❌ Coinbase factory: {e}")

    # OKX
    try:
        okx = CCXTProviderFactory.create_okx(sandbox=True)
        logger.info(f"✅ OKX factory: {okx.name}")
    except Exception as e:
        logger.error(f"❌ OKX factory: {e}")


async def main():
    """Point d'entrée principal."""
    logger.info("🎯 QFrame CCXT Multi-Exchange Test")
    logger.info("=" * 50)

    # Démonstration des factory methods
    await demo_factory_methods()

    # Tests multi-exchange
    results = await test_multiple_exchanges()

    logger.info("\n🏁 Tests terminés")

    # Retourner les résultats pour validation
    return results


if __name__ == "__main__":
    # Exécuter les tests
    asyncio.run(main())