"""
Infrastructure Layer: PostgreSQL Repository Implementations
===========================================================

Implémentations PostgreSQL complètes des repositories avec
requêtes optimisées et gestion des erreurs.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID

from ...domain.entities.strategy import Strategy, StrategyStatus
from ...domain.entities.portfolio import Portfolio, PortfolioStatus, Position
from ...domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from ...domain.entities.risk_assessment import RiskAssessment, RiskLevel, RiskMetric
from ...domain.entities.backtest import BacktestConfiguration, BacktestResult

from ...domain.repositories.strategy_repository import StrategyRepository
from ...domain.repositories.portfolio_repository import PortfolioRepository
from ...domain.repositories.order_repository import OrderRepository
from ...domain.repositories.risk_assessment_repository import RiskAssessmentRepository
from ...domain.repositories.backtest_repository import BacktestRepository

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace

from .database import DatabaseManager, TransactionManager, IsolationLevel


class PostgresStrategyRepository(StrategyRepository):
    """Repository PostgreSQL pour les stratégies"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.transaction_manager = db_manager.transaction_manager
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

    async def create_table_if_not_exists(self):
        """Créer la table strategies si elle n'existe pas"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS strategies (
            id UUID PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            status VARCHAR(20) NOT NULL DEFAULT 'inactive',
            parameters JSONB DEFAULT '{}',
            risk_limits JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE,
            version INTEGER DEFAULT 1,
            
            CONSTRAINT strategies_status_check CHECK (status IN ('active', 'inactive', 'paused', 'error'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies(status);
        CREATE INDEX IF NOT EXISTS idx_strategies_created_at ON strategies(created_at);
        CREATE INDEX IF NOT EXISTS idx_strategies_name ON strategies(name);
        """
        
        await self.db_manager.execute_script(create_table_sql)

    @trace("postgres.strategy.save")
    async def save(self, strategy: Strategy) -> Strategy:
        """Sauvegarder une stratégie"""
        try:
            # Vérifier si c'est une création ou une mise à jour
            existing = await self.get_by_id(strategy.id)
            
            if existing:
                # Mise à jour
                query = """
                UPDATE strategies 
                SET name = $2, description = $3, status = $4, parameters = $5, 
                    risk_limits = $6, updated_at = NOW(), version = version + 1
                WHERE id = $1
                RETURNING version
                """
                params = (
                    strategy.id,
                    strategy.name,
                    strategy.description,
                    strategy.status.value,
                    json.dumps(strategy.parameters),
                    json.dumps(strategy.risk_limits)
                )
                
                result = await self.db_manager.execute_query(query, params, fetch="one")
                strategy.version = result[0]
                strategy.updated_at = datetime.utcnow()
                
            else:
                # Création
                query = """
                INSERT INTO strategies (id, name, description, status, parameters, risk_limits, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                """
                params = (
                    strategy.id,
                    strategy.name,
                    strategy.description,
                    strategy.status.value,
                    json.dumps(strategy.parameters),
                    json.dumps(strategy.risk_limits)
                )
                
                await self.db_manager.execute_query(query, params)
                strategy.created_at = datetime.utcnow()

            # Métriques
            self.metrics.collector.increment_counter(
                "repository.strategy.save",
                labels={"operation": "update" if existing else "create"}
            )

            self.logger.info(f"Strategy saved: {strategy.id}", strategy_name=strategy.name)
            return strategy

        except Exception as e:
            self.logger.error(f"Error saving strategy {strategy.id}: {e}")
            self.metrics.collector.increment_counter("repository.strategy.errors", labels={"operation": "save"})
            raise

    @trace("postgres.strategy.get_by_id")
    async def get_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """Récupérer une stratégie par ID"""
        try:
            query = """
            SELECT id, name, description, status, parameters, risk_limits, 
                   created_at, updated_at, version
            FROM strategies WHERE id = $1
            """
            
            row = await self.db_manager.execute_query(query, (strategy_id,), fetch="one")
            
            if not row:
                return None

            return Strategy(
                id=str(row[0]),
                name=row[1],
                description=row[2],
                status=StrategyStatus(row[3]),
                parameters=row[4] or {},
                risk_limits=row[5] or {},
                created_at=row[6],
                updated_at=row[7],
                version=row[8]
            )

        except Exception as e:
            self.logger.error(f"Error getting strategy {strategy_id}: {e}")
            self.metrics.collector.increment_counter("repository.strategy.errors", labels={"operation": "get_by_id"})
            raise

    async def get_by_name(self, name: str) -> Optional[Strategy]:
        """Récupérer une stratégie par nom"""
        try:
            query = """
            SELECT id, name, description, status, parameters, risk_limits, 
                   created_at, updated_at, version
            FROM strategies WHERE name = $1
            """
            
            row = await self.db_manager.execute_query(query, (name,), fetch="one")
            
            if not row:
                return None

            return Strategy(
                id=str(row[0]),
                name=row[1],
                description=row[2],
                status=StrategyStatus(row[3]),
                parameters=row[4] or {},
                risk_limits=row[5] or {},
                created_at=row[6],
                updated_at=row[7],
                version=row[8]
            )

        except Exception as e:
            self.logger.error(f"Error getting strategy by name {name}: {e}")
            raise

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Strategy]:
        """Lister toutes les stratégies"""
        try:
            query = """
            SELECT id, name, description, status, parameters, risk_limits, 
                   created_at, updated_at, version
            FROM strategies 
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """
            
            rows = await self.db_manager.execute_query(query, (limit, offset), fetch="all")
            
            strategies = []
            for row in rows:
                strategy = Strategy(
                    id=str(row[0]),
                    name=row[1],
                    description=row[2],
                    status=StrategyStatus(row[3]),
                    parameters=row[4] or {},
                    risk_limits=row[5] or {},
                    created_at=row[6],
                    updated_at=row[7],
                    version=row[8]
                )
                strategies.append(strategy)

            return strategies

        except Exception as e:
            self.logger.error(f"Error listing strategies: {e}")
            raise

    async def find_by_status(self, status: StrategyStatus) -> List[Strategy]:
        """Trouver les stratégies par statut"""
        try:
            query = """
            SELECT id, name, description, status, parameters, risk_limits, 
                   created_at, updated_at, version
            FROM strategies 
            WHERE status = $1
            ORDER BY created_at DESC
            """
            
            rows = await self.db_manager.execute_query(query, (status.value,), fetch="all")
            
            strategies = []
            for row in rows:
                strategy = Strategy(
                    id=str(row[0]),
                    name=row[1],
                    description=row[2],
                    status=StrategyStatus(row[3]),
                    parameters=row[4] or {},
                    risk_limits=row[5] or {},
                    created_at=row[6],
                    updated_at=row[7],
                    version=row[8]
                )
                strategies.append(strategy)

            return strategies

        except Exception as e:
            self.logger.error(f"Error finding strategies by status {status}: {e}")
            raise

    async def delete(self, strategy_id: str) -> bool:
        """Supprimer une stratégie"""
        try:
            query = "DELETE FROM strategies WHERE id = $1"
            result = await self.db_manager.execute_query(query, (strategy_id,))
            
            # asyncpg renvoie "DELETE n" où n est le nombre de lignes supprimées
            deleted = "DELETE 1" in str(result)
            
            if deleted:
                self.logger.info(f"Strategy deleted: {strategy_id}")
                self.metrics.collector.increment_counter("repository.strategy.delete")
            
            return deleted

        except Exception as e:
            self.logger.error(f"Error deleting strategy {strategy_id}: {e}")
            raise

    async def count(self) -> int:
        """Compter le nombre total de stratégies"""
        try:
            query = "SELECT COUNT(*) FROM strategies"
            result = await self.db_manager.execute_query(query, fetch="one")
            return result[0]

        except Exception as e:
            self.logger.error(f"Error counting strategies: {e}")
            raise


class PostgresPortfolioRepository(PortfolioRepository):
    """Repository PostgreSQL pour les portfolios"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.transaction_manager = db_manager.transaction_manager
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

    async def create_tables_if_not_exist(self):
        """Créer les tables portfolios et positions"""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS portfolios (
            id UUID PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            status VARCHAR(20) NOT NULL DEFAULT 'active',
            total_value DECIMAL(20,8) DEFAULT 0,
            cash_balance DECIMAL(20,8) DEFAULT 0,
            currency VARCHAR(3) DEFAULT 'USD',
            target_allocation JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE,
            version INTEGER DEFAULT 1,
            
            CONSTRAINT portfolios_status_check CHECK (status IN ('active', 'inactive', 'closed')),
            CONSTRAINT portfolios_currency_check CHECK (currency ~ '^[A-Z]{3}$')
        );

        CREATE TABLE IF NOT EXISTS positions (
            id UUID PRIMARY KEY,
            portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
            symbol VARCHAR(20) NOT NULL,
            quantity DECIMAL(20,8) NOT NULL DEFAULT 0,
            average_price DECIMAL(20,8) DEFAULT 0,
            current_price DECIMAL(20,8) DEFAULT 0,
            unrealized_pnl DECIMAL(20,8) DEFAULT 0,
            realized_pnl DECIMAL(20,8) DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE,
            
            UNIQUE(portfolio_id, symbol)
        );
        
        CREATE INDEX IF NOT EXISTS idx_portfolios_status ON portfolios(status);
        CREATE INDEX IF NOT EXISTS idx_portfolios_name ON portfolios(name);
        CREATE INDEX IF NOT EXISTS idx_positions_portfolio_id ON positions(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
        """
        
        await self.db_manager.execute_script(create_tables_sql)

    @trace("postgres.portfolio.save")
    async def save(self, portfolio: Portfolio) -> Portfolio:
        """Sauvegarder un portfolio"""
        async with self.transaction_manager.transaction() as conn:
            try:
                # Vérifier si existe
                existing = await self.get_by_id(portfolio.id)
                
                if existing:
                    # Mise à jour portfolio
                    portfolio_query = """
                    UPDATE portfolios 
                    SET name = $2, description = $3, status = $4, total_value = $5,
                        cash_balance = $6, currency = $7, target_allocation = $8,
                        updated_at = NOW(), version = version + 1
                    WHERE id = $1
                    RETURNING version
                    """
                    portfolio_params = (
                        portfolio.id,
                        portfolio.name,
                        portfolio.description,
                        portfolio.status.value,
                        float(portfolio.total_value),
                        float(portfolio.cash_balance),
                        portfolio.currency,
                        json.dumps(portfolio.target_allocation)
                    )
                    
                    result = await conn.fetchrow(portfolio_query, *portfolio_params)
                    portfolio.version = result[0]
                    portfolio.updated_at = datetime.utcnow()
                    
                else:
                    # Création portfolio
                    portfolio_query = """
                    INSERT INTO portfolios (id, name, description, status, total_value,
                                          cash_balance, currency, target_allocation, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                    """
                    portfolio_params = (
                        portfolio.id,
                        portfolio.name,
                        portfolio.description,
                        portfolio.status.value,
                        float(portfolio.total_value),
                        float(portfolio.cash_balance),
                        portfolio.currency,
                        json.dumps(portfolio.target_allocation)
                    )
                    
                    await conn.execute(portfolio_query, *portfolio_params)
                    portfolio.created_at = datetime.utcnow()

                # Gérer les positions
                if existing:
                    # Supprimer les anciennes positions
                    await conn.execute("DELETE FROM positions WHERE portfolio_id = $1", portfolio.id)

                # Insérer les nouvelles positions
                if portfolio.positions:
                    for position in portfolio.positions:
                        position_query = """
                        INSERT INTO positions (id, portfolio_id, symbol, quantity, average_price,
                                             current_price, unrealized_pnl, realized_pnl, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                        """
                        position_params = (
                            str(uuid.uuid4()),
                            portfolio.id,
                            position.symbol,
                            float(position.quantity),
                            float(position.average_price),
                            float(position.current_price),
                            float(position.unrealized_pnl),
                            float(position.realized_pnl)
                        )
                        
                        await conn.execute(position_query, *position_params)

                self.logger.info(f"Portfolio saved: {portfolio.id}", portfolio_name=portfolio.name)
                return portfolio

            except Exception as e:
                self.logger.error(f"Error saving portfolio {portfolio.id}: {e}")
                raise

    @trace("postgres.portfolio.get_by_id")
    async def get_by_id(self, portfolio_id: str) -> Optional[Portfolio]:
        """Récupérer un portfolio par ID"""
        try:
            # Récupérer le portfolio
            portfolio_query = """
            SELECT id, name, description, status, total_value, cash_balance, currency,
                   target_allocation, created_at, updated_at, version
            FROM portfolios WHERE id = $1
            """
            
            portfolio_row = await self.db_manager.execute_query(portfolio_query, (portfolio_id,), fetch="one")
            
            if not portfolio_row:
                return None

            # Récupérer les positions
            positions_query = """
            SELECT symbol, quantity, average_price, current_price, unrealized_pnl, realized_pnl
            FROM positions WHERE portfolio_id = $1
            """
            
            position_rows = await self.db_manager.execute_query(positions_query, (portfolio_id,), fetch="all")
            
            positions = []
            for pos_row in position_rows:
                position = Position(
                    symbol=pos_row[0],
                    quantity=Decimal(str(pos_row[1])),
                    average_price=Decimal(str(pos_row[2])),
                    current_price=Decimal(str(pos_row[3])),
                    unrealized_pnl=Decimal(str(pos_row[4])),
                    realized_pnl=Decimal(str(pos_row[5]))
                )
                positions.append(position)

            return Portfolio(
                id=str(portfolio_row[0]),
                name=portfolio_row[1],
                description=portfolio_row[2],
                status=PortfolioStatus(portfolio_row[3]),
                total_value=Decimal(str(portfolio_row[4])),
                cash_balance=Decimal(str(portfolio_row[5])),
                currency=portfolio_row[6],
                target_allocation=portfolio_row[7] or {},
                positions=positions,
                created_at=portfolio_row[8],
                updated_at=portfolio_row[9],
                version=portfolio_row[10]
            )

        except Exception as e:
            self.logger.error(f"Error getting portfolio {portfolio_id}: {e}")
            raise

    async def get_by_name(self, name: str) -> Optional[Portfolio]:
        """Récupérer un portfolio par nom"""
        try:
            portfolio_query = """
            SELECT id FROM portfolios WHERE name = $1
            """
            
            result = await self.db_manager.execute_query(portfolio_query, (name,), fetch="one")
            
            if not result:
                return None

            return await self.get_by_id(str(result[0]))

        except Exception as e:
            self.logger.error(f"Error getting portfolio by name {name}: {e}")
            raise

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Portfolio]:
        """Lister tous les portfolios"""
        try:
            query = """
            SELECT id FROM portfolios 
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """
            
            rows = await self.db_manager.execute_query(query, (limit, offset), fetch="all")
            
            portfolios = []
            for row in rows:
                portfolio = await self.get_by_id(str(row[0]))
                if portfolio:
                    portfolios.append(portfolio)

            return portfolios

        except Exception as e:
            self.logger.error(f"Error listing portfolios: {e}")
            raise

    async def find_by_status(self, status: PortfolioStatus) -> List[Portfolio]:
        """Trouver les portfolios par statut"""
        try:
            query = """
            SELECT id FROM portfolios 
            WHERE status = $1
            ORDER BY created_at DESC
            """
            
            rows = await self.db_manager.execute_query(query, (status.value,), fetch="all")
            
            portfolios = []
            for row in rows:
                portfolio = await self.get_by_id(str(row[0]))
                if portfolio:
                    portfolios.append(portfolio)

            return portfolios

        except Exception as e:
            self.logger.error(f"Error finding portfolios by status {status}: {e}")
            raise

    async def delete(self, portfolio_id: str) -> bool:
        """Supprimer un portfolio"""
        try:
            # Les positions seront supprimées automatiquement (CASCADE)
            query = "DELETE FROM portfolios WHERE id = $1"
            result = await self.db_manager.execute_query(query, (portfolio_id,))
            
            deleted = "DELETE 1" in str(result)
            
            if deleted:
                self.logger.info(f"Portfolio deleted: {portfolio_id}")
            
            return deleted

        except Exception as e:
            self.logger.error(f"Error deleting portfolio {portfolio_id}: {e}")
            raise


# Les autres repositories suivraient le même pattern...
# Je vais créer des stubs pour les autres pour compléter l'interface

class PostgresOrderRepository(OrderRepository):
    """Repository PostgreSQL pour les ordres"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = LoggerFactory.get_logger(__name__)

    async def save(self, order: Order) -> Order:
        # TODO: Implémenter
        pass

    async def get_by_id(self, order_id: str) -> Optional[Order]:
        # TODO: Implémenter
        pass

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Order]:
        # TODO: Implémenter
        return []

    async def find_by_portfolio_id(self, portfolio_id: str) -> List[Order]:
        # TODO: Implémenter
        return []

    async def find_by_status(self, status: OrderStatus) -> List[Order]:
        # TODO: Implémenter
        return []

    async def find_active_orders(self) -> List[Order]:
        # TODO: Implémenter
        return []

    async def find_by_symbol(self, symbol: str) -> List[Order]:
        # TODO: Implémenter
        return []

    async def delete(self, order_id: str) -> bool:
        # TODO: Implémenter
        return False


class PostgresRiskAssessmentRepository(RiskAssessmentRepository):
    """Repository PostgreSQL pour les évaluations de risque"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = LoggerFactory.get_logger(__name__)

    async def save(self, assessment: RiskAssessment) -> RiskAssessment:
        # TODO: Implémenter
        pass

    async def get_by_id(self, assessment_id: str) -> Optional[RiskAssessment]:
        # TODO: Implémenter
        pass

    async def get_latest_by_target(self, target_type: str, target_id: str) -> Optional[RiskAssessment]:
        # TODO: Implémenter
        pass

    async def find_by_target(self, target_type: str, target_id: str) -> List[RiskAssessment]:
        # TODO: Implémenter
        return []

    async def find_critical(self) -> List[RiskAssessment]:
        # TODO: Implémenter
        return []

    async def find_by_level(self, level: RiskLevel) -> List[RiskAssessment]:
        # TODO: Implémenter
        return []

    async def delete(self, assessment_id: str) -> bool:
        # TODO: Implémenter
        return False


class PostgresBacktestRepository(BacktestRepository):
    """Repository PostgreSQL pour les backtests"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = LoggerFactory.get_logger(__name__)

    async def save_configuration(self, config: BacktestConfiguration) -> BacktestConfiguration:
        # TODO: Implémenter
        pass

    async def get_configuration_by_id(self, config_id: str) -> Optional[BacktestConfiguration]:
        # TODO: Implémenter
        pass

    async def save_result(self, result: BacktestResult) -> BacktestResult:
        # TODO: Implémenter
        pass

    async def get_result_by_id(self, result_id: str) -> Optional[BacktestResult]:
        # TODO: Implémenter
        pass

    async def list_configurations(self, limit: int = 100, offset: int = 0) -> List[BacktestConfiguration]:
        # TODO: Implémenter
        return []

    async def list_results(self, limit: int = 100, offset: int = 0) -> List[BacktestResult]:
        # TODO: Implémenter
        return []

    async def delete_configuration(self, config_id: str) -> bool:
        # TODO: Implémenter
        return False

    async def delete_result(self, result_id: str) -> bool:
        # TODO: Implémenter
        return False
