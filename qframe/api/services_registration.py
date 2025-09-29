"""
📦 Services Registration
Enregistrement des services API dans le container DI
"""

from qframe.core.container import get_container
from qframe.api.services.market_data_service import MarketDataService
from qframe.api.services.real_time_service import RealTimeService
from qframe.api.services.order_service import OrderService
from qframe.api.services.position_service import PositionService
from qframe.api.services.portfolio_service import PortfolioService
from qframe.api.services.risk_service import RiskService
from qframe.api.services.strategy_service import StrategyService
from qframe.api.services.backtest_service import BacktestService


def register_api_services():
    """Enregistre tous les services API dans le container DI."""
    container = get_container()

    # Services de données
    container.register_singleton(MarketDataService, MarketDataService)
    container.register_singleton(RealTimeService, RealTimeService)

    # Services de trading
    container.register_singleton(OrderService, OrderService)
    container.register_singleton(PositionService, PositionService)
    container.register_singleton(PortfolioService, PortfolioService)

    # Services de gestion des risques
    container.register_singleton(RiskService, RiskService)

    # Services de stratégies
    container.register_singleton(StrategyService, StrategyService)
    container.register_singleton(BacktestService, BacktestService)

    print("✅ Services API enregistrés dans le container DI")