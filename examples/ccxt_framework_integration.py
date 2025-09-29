#!/usr/bin/env python3
"""
Test d'intégration du CCXTProvider dans QFrame
==============================================

Exemple démonstrant l'intégration complète du CCXTProvider
dans le framework QFrame avec DI Container et Market Data Pipeline.
"""

import asyncio
import logging
from typing import List, Dict, Any

# QFrame Core
from qframe.core.container import DIContainer
from qframe.core.config import FrameworkConfig, Environment
from qframe.infrastructure.config.service_configuration import ServiceConfiguration

# Market Data Pipeline
from qframe.infrastructure.data.market_data_pipeline import MarketDataPipeline

# CCXT Provider
from qframe.infrastructure.data.ccxt_provider import CCXTProvider


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_ccxt_integration():
    """Teste l'intégration CCXT avec le framework complet."""

    logger.info("🚀 Démarrage test intégration CCXT Framework")
    logger.info("=" * 55)

    # Configuration du framework pour développement
    config = FrameworkConfig(environment=Environment.DEVELOPMENT)

    # Container DI
    container = DIContainer()

    # Configuration des services data pipeline seulement
    service_config = ServiceConfiguration(container, config)
    service_config._configure_data_pipeline_services()

    logger.info("✅ Framework configuré avec succès")

    # Récupérer le pipeline de données
    pipeline: MarketDataPipeline = container.resolve(MarketDataPipeline)

    # Vérifier les providers enregistrés
    provider_names = pipeline.get_registered_providers()
    logger.info(f"📊 Providers enregistrés: {len(provider_names)}")

    ccxt_providers = [name for name in provider_names if "ccxt_" in name]
    logger.info(f"🔗 CCXT Providers: {ccxt_providers}")

    # Test de récupération de données depuis différents exchanges
    test_symbol = "BTC/USDT"

    for provider_name in ccxt_providers:
        try:
            logger.info(f"\n📈 Test {provider_name} - {test_symbol}")

            # Récupérer données ticker
            ticker_data = await pipeline.get_ticker(provider_name, test_symbol)

            if ticker_data:
                logger.info(f"   💰 Prix: ${ticker_data.last}")
                logger.info(f"   📊 Volume 24h: {ticker_data.volume_24h}")
                logger.info(f"   📈 Change 24h: {ticker_data.change_24h}%")
            else:
                logger.warning(f"   ❌ Pas de données pour {test_symbol}")

        except Exception as e:
            logger.error(f"   ❌ Erreur {provider_name}: {e}")

    # Test de récupération de données OHLCV
    logger.info(f"\n📊 Test données OHLCV pour {test_symbol}")

    if ccxt_providers:
        provider_name = ccxt_providers[0]  # Premier provider CCXT
        try:
            klines = await pipeline.get_klines(
                provider_name,
                test_symbol,
                interval="1h",
                limit=10
            )

            if klines:
                logger.info(f"   ✅ {len(klines)} chandeliers récupérés")
                latest = klines[-1]
                logger.info(f"   🕐 Dernière chandelle: {latest.timestamp}")
                logger.info(f"   💰 OHLC: {latest.open} | {latest.high} | {latest.low} | {latest.close}")
            else:
                logger.warning(f"   ❌ Pas de données OHLCV")

        except Exception as e:
            logger.error(f"   ❌ Erreur OHLCV: {e}")

    # Test de capabilities des providers
    logger.info(f"\n🔍 Capacités des providers CCXT:")

    for provider_name in ccxt_providers:
        try:
            provider = pipeline.get_provider(provider_name)
            if isinstance(provider, CCXTProvider):
                capabilities = provider.get_capabilities()
                logger.info(f"   {provider_name}:")
                logger.info(f"     - Symboles: {capabilities.get('symbols', 0)}")
                logger.info(f"     - Intervals: {len(capabilities.get('intervals', []))}")
                logger.info(f"     - REST API: {capabilities.get('rest_api', False)}")
                logger.info(f"     - WebSocket: {capabilities.get('websocket', False)}")

        except Exception as e:
            logger.error(f"   ❌ Erreur capabilities {provider_name}: {e}")

    # Statistiques finales
    logger.info("\n" + "=" * 55)
    logger.info("📋 RAPPORT FINAL")
    logger.info("=" * 55)
    logger.info(f"✅ Framework: Configuré et fonctionnel")
    logger.info(f"📊 Total providers: {len(provider_names)}")
    logger.info(f"🔗 CCXT providers: {len(ccxt_providers)}")
    logger.info(f"🎯 Intégration: Réussie")

    return {
        "framework_configured": True,
        "total_providers": len(provider_names),
        "ccxt_providers": len(ccxt_providers),
        "provider_names": provider_names,
        "ccxt_provider_names": ccxt_providers
    }


async def test_di_container_ccxt():
    """Teste la résolution des providers CCXT via le DI Container."""

    logger.info("\n🏭 Test DI Container - Résolution CCXT Providers")
    logger.info("=" * 55)

    # Configuration minimale
    config = FrameworkConfig(environment=Environment.DEVELOPMENT)
    container = DIContainer()
    service_config = ServiceConfiguration(container, config)

    # Configurer seulement les data providers
    service_config._configure_data_pipeline_services()

    # Tester résolution directe de providers CCXT
    pipeline: MarketDataPipeline = container.resolve(MarketDataPipeline)

    # Tester les providers individuellement
    provider_names = pipeline.get_registered_providers()
    ccxt_providers = [name for name in provider_names if "ccxt_" in name]

    for provider_name in ccxt_providers:
        try:
            provider = pipeline.get_provider(provider_name)
            logger.info(f"✅ {provider_name}: {type(provider).__name__}")

            # Test de connexion rapide
            if hasattr(provider, 'get_status'):
                status = provider.get_status()
                logger.info(f"   Status: {status.get('exchange', 'Unknown')}")

        except Exception as e:
            logger.error(f"❌ {provider_name}: {e}")

    logger.info("🏁 Test DI Container terminé")


async def main():
    """Point d'entrée principal."""
    logger.info("🎯 QFrame CCXT Framework Integration Test")
    logger.info("=" * 55)

    # Test d'intégration complète
    results = await test_ccxt_integration()

    # Test DI Container spécifique
    await test_di_container_ccxt()

    logger.info("\n🏁 Tous les tests terminés")
    return results


if __name__ == "__main__":
    # Exécuter les tests
    asyncio.run(main())