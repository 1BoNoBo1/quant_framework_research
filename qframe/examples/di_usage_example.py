"""
Example: Utilisation du Dependency Injection Container
====================================================

Exemple d'utilisation du container DI refactorisé avec l'architecture hexagonale.
Montre comment résoudre et utiliser les services configurés.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports du framework
from ..core.container import get_container
from ..domain.entities.strategy import StrategyType
from ..application.commands.strategy_commands import create_strategy_command
from ..application.handlers.strategy_command_handler import StrategyCommandHandler
from ..application.queries.signal_queries import active_signals_query
from ..application.handlers.signal_query_handler import SignalQueryHandler


async def example_di_usage():
    """
    Exemple complet d'utilisation du DI container.
    """
    print("🚀 Exemple d'utilisation du DI Container")
    print("=" * 50)

    # 1. Récupérer le container configuré
    container = get_container()
    print(f"✅ Container initialisé avec {len(container.get_registrations())} services")

    # 2. Résoudre les handlers d'application
    print("\n📦 Résolution des handlers...")

    # Utiliser un scope pour les handlers scoped
    with container.create_scope() as scope:

        # Résoudre le handler de commandes de stratégie
        strategy_handler = scope.resolve(StrategyCommandHandler)
        print(f"✅ StrategyCommandHandler résolu: {type(strategy_handler).__name__}")

        # Résoudre le handler de queries de signaux
        signal_handler = scope.resolve(SignalQueryHandler)
        print(f"✅ SignalQueryHandler résolu: {type(signal_handler).__name__}")

        # 3. Créer une nouvelle stratégie
        print("\n🎯 Test de création de stratégie...")

        create_command = create_strategy_command(
            name="Test Strategy DI",
            strategy_type=StrategyType.MEAN_REVERSION,
            universe=["BTC/USDT", "ETH/USDT"],
            description="Stratégie de test pour démontrer le DI",
            max_position_size=0.05,
            max_positions=3
        )

        try:
            strategy_id = await strategy_handler.handle_create_strategy(create_command)
            print(f"✅ Stratégie créée avec succès: {strategy_id}")

            # 4. Récupérer les signaux actifs
            print("\n📊 Test de query de signaux...")

            active_query = active_signals_query(
                max_age_minutes=120,
                limit=10
            )

            active_signals = await signal_handler.handle_get_active_signals(active_query)
            print(f"✅ Signaux actifs trouvés: {len(active_signals)}")

            # 5. Récupérer les statistiques des stratégies
            print("\n📈 Test des statistiques...")

            stats = await strategy_handler.get_strategy_statistics()
            print(f"✅ Statistiques: {stats.get('total_strategies', 0)} stratégies")

        except Exception as e:
            print(f"❌ Erreur lors des tests: {e}")

    print("\n🎉 Exemple terminé avec succès!")


async def example_service_resolution():
    """
    Exemple de résolution directe des services.
    """
    print("\n🔧 Exemple de résolution directe des services")
    print("=" * 50)

    container = get_container()

    # Résoudre les services individuels
    from ..domain.services.signal_service import SignalService
    from ..domain.repositories.strategy_repository import StrategyRepository
    from ..infrastructure.external.market_data_service import MarketDataService
    from ..infrastructure.external.broker_service import BrokerService

    print("\n📦 Résolution des services...")

    # Services domain
    signal_service = container.resolve(SignalService)
    print(f"✅ SignalService: {type(signal_service).__name__}")

    # Repository
    strategy_repo = container.resolve(StrategyRepository)
    print(f"✅ StrategyRepository: {type(strategy_repo).__name__}")

    # Services infrastructure
    market_data = container.resolve(MarketDataService)
    print(f"✅ MarketDataService: {type(market_data).__name__}")

    broker = container.resolve(BrokerService)
    print(f"✅ BrokerService: {type(broker).__name__}")

    # Test des services
    print("\n🧪 Test des services...")

    # Test repository
    strategies = await strategy_repo.find_all()
    print(f"✅ Stratégies dans le repo: {len(strategies)}")

    # Test market data
    price = await market_data.get_current_price("BTC/USDT")
    print(f"✅ Prix BTC/USDT: ${price}")

    # Test broker
    balance = await broker.get_account_balance()
    print(f"✅ Solde USDT: ${balance.get('USDT', {}).get('total', 0)}")


def example_container_statistics():
    """
    Exemple d'affichage des statistiques du container.
    """
    print("\n📊 Statistiques du Container DI")
    print("=" * 50)

    from ..infrastructure.config.service_configuration import get_service_statistics

    container = get_container()
    stats = get_service_statistics(container)

    print(f"📦 Total des services: {stats['total_services']}")
    print(f"🔒 Singletons: {stats['singletons']}")
    print(f"⚡ Transients: {stats['transients']}")
    print(f"🎯 Scoped: {stats['scoped']}")
    print(f"🏭 Avec factories: {stats['with_factories']}")

    print("\n📋 Liste des services:")
    for service in stats['services']:
        lifetime_icon = {
            'singleton': '🔒',
            'transient': '⚡',
            'scoped': '🎯'
        }.get(service['lifetime'], '❓')

        print(f"  {lifetime_icon} {service['interface']} -> {service['implementation']}")


async def example_error_handling():
    """
    Exemple de gestion d'erreurs avec le DI.
    """
    print("\n⚠️ Exemple de gestion d'erreurs")
    print("=" * 50)

    container = get_container()

    try:
        # Essayer de résoudre un service non enregistré
        from typing import Protocol

        class NonExistentService(Protocol):
            def do_something(self) -> str: ...

        service = container.resolve(NonExistentService)

    except ValueError as e:
        print(f"✅ Erreur attendue pour service non enregistré: {e}")

    # Test de dépendance circulaire (à implémenter si nécessaire)
    print("✅ Gestion d'erreurs testée")


async def main():
    """Fonction principale de l'exemple."""
    print("🚀 DÉMONSTRATION DU DEPENDENCY INJECTION CONTAINER")
    print("=" * 60)

    try:
        # Exemple 1: Utilisation complète du DI
        await example_di_usage()

        # Exemple 2: Résolution directe des services
        await example_service_resolution()

        # Exemple 3: Statistiques du container
        example_container_statistics()

        # Exemple 4: Gestion d'erreurs
        await example_error_handling()

        print("\n🎉 TOUS LES EXEMPLES TERMINÉS AVEC SUCCÈS!")

    except Exception as e:
        print(f"\n❌ ERREUR DANS L'EXEMPLE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())