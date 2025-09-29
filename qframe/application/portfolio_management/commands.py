"""
Application Commands: Portfolio Management
=========================================

Commandes pour les opérations de gestion des portfolios.
Implémente le pattern CQRS pour la séparation des responsabilités.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass

from ..base.command import Command, CommandHandler, CommandResult
from ...domain.entities.portfolio import (
    Portfolio, PortfolioType, PortfolioStatus, PortfolioConstraints, RebalancingFrequency
)
from ...domain.repositories.portfolio_repository import PortfolioRepository
from ...domain.services.portfolio_service import PortfolioService
from ...domain.value_objects.position import Position


@dataclass
class CreatePortfolioCommand(Command):
    """Commande pour créer un nouveau portfolio"""
    name: str
    description: str = ""
    portfolio_type: PortfolioType = PortfolioType.TRADING
    initial_capital: Decimal = Decimal("100000")
    constraints: Optional[PortfolioConstraints] = None
    benchmark_symbol: str = "SPY"
    currency: str = "USD"

    def __post_init__(self):
        super().__init__()


@dataclass
class UpdatePortfolioCommand(Command):
    """Commande pour mettre à jour un portfolio"""
    portfolio_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[PortfolioStatus] = None
    constraints: Optional[PortfolioConstraints] = None
    benchmark_symbol: Optional[str] = None

    def __post_init__(self):
        super().__init__()


@dataclass
class AddPositionCommand(Command):
    """Commande pour ajouter une position au portfolio"""
    portfolio_id: str
    position: Position

    def __post_init__(self):
        super().__init__()


@dataclass
class RemovePositionCommand(Command):
    """Commande pour retirer une position du portfolio"""
    portfolio_id: str
    symbol: str

    def __post_init__(self):
        super().__init__()


@dataclass
class UpdatePositionCommand(Command):
    """Commande pour mettre à jour une position"""
    portfolio_id: str
    symbol: str
    new_quantity: Optional[Decimal] = None
    new_price: Optional[Decimal] = None

    def __post_init__(self):
        super().__init__()


@dataclass
class AdjustCashCommand(Command):
    """Commande pour ajuster le cash du portfolio"""
    portfolio_id: str
    amount: Decimal
    reason: str = "Manual adjustment"

    def __post_init__(self):
        super().__init__()


@dataclass
class SetTargetAllocationCommand(Command):
    """Commande pour définir les allocations cibles"""
    portfolio_id: str
    target_allocations: Dict[str, Decimal]

    def __post_init__(self):
        super().__init__()


@dataclass
class RebalancePortfolioCommand(Command):
    """Commande pour rééquilibrer un portfolio"""
    portfolio_id: str
    target_allocations: Optional[Dict[str, Decimal]] = None
    rebalancing_threshold: Decimal = Decimal("0.05")
    force_rebalance: bool = False

    def __post_init__(self):
        super().__init__()


@dataclass
class OptimizeAllocationCommand(Command):
    """Commande pour optimiser l'allocation d'un portfolio"""
    portfolio_id: str
    optimization_method: str = "equal_weight"  # equal_weight, risk_parity, momentum
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        super().__init__()


@dataclass
class AddStrategyToPortfolioCommand(Command):
    """Commande pour ajouter une stratégie au portfolio"""
    portfolio_id: str
    strategy_id: str

    def __post_init__(self):
        super().__init__()


@dataclass
class RemoveStrategyFromPortfolioCommand(Command):
    """Commande pour retirer une stratégie du portfolio"""
    portfolio_id: str
    strategy_id: str

    def __post_init__(self):
        super().__init__()


@dataclass
class CreateSnapshotCommand(Command):
    """Commande pour créer un snapshot du portfolio"""
    portfolio_id: str

    def __post_init__(self):
        super().__init__()


@dataclass
class ArchivePortfolioCommand(Command):
    """Commande pour archiver un portfolio"""
    portfolio_id: str

    def __post_init__(self):
        super().__init__()


class CreatePortfolioHandler(CommandHandler[CreatePortfolioCommand]):
    """Handler pour créer un nouveau portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, command: CreatePortfolioCommand) -> CommandResult:
        """
        Traite la commande de création de portfolio.

        Args:
            command: Commande de création

        Returns:
            Résultat avec l'ID du portfolio créé
        """
        try:
            # Vérifier que le nom n'existe pas déjà
            existing = await self.repository.find_by_name(command.name)
            if existing:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio name '{command.name}' already exists"
                )

            # Créer le portfolio
            portfolio = Portfolio(
                name=command.name,
                description=command.description,
                portfolio_type=command.portfolio_type,
                initial_capital=command.initial_capital,
                constraints=command.constraints or PortfolioConstraints(),
                benchmark_symbol=command.benchmark_symbol,
                currency=command.currency
            )

            # Sauvegarder
            await self.repository.save(portfolio)

            return CommandResult(
                success=True,
                result_data={
                    "portfolio_id": portfolio.id,
                    "name": portfolio.name,
                    "initial_capital": float(portfolio.initial_capital),
                    "total_value": float(portfolio.total_value)
                },
                message=f"Portfolio '{command.name}' created successfully"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error creating portfolio: {str(e)}"
            )


class UpdatePortfolioHandler(CommandHandler[UpdatePortfolioCommand]):
    """Handler pour mettre à jour un portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, command: UpdatePortfolioCommand) -> CommandResult:
        """
        Traite la commande de mise à jour de portfolio.

        Args:
            command: Commande de mise à jour

        Returns:
            Résultat de la mise à jour
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(command.portfolio_id)
            if not portfolio:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

            # Mettre à jour les champs spécifiés
            if command.name is not None:
                # Vérifier que le nouveau nom n'existe pas déjà
                if command.name != portfolio.name:
                    existing = await self.repository.find_by_name(command.name)
                    if existing:
                        return CommandResult(
                            success=False,
                            error_message=f"Portfolio name '{command.name}' already exists"
                        )
                portfolio.name = command.name

            if command.description is not None:
                portfolio.description = command.description

            if command.status is not None:
                portfolio.status = command.status

            if command.constraints is not None:
                portfolio.constraints = command.constraints

            if command.benchmark_symbol is not None:
                portfolio.benchmark_symbol = command.benchmark_symbol

            portfolio.updated_at = datetime.utcnow()

            # Sauvegarder
            await self.repository.update(portfolio)

            return CommandResult(
                success=True,
                result_data={
                    "portfolio_id": portfolio.id,
                    "updated_fields": [
                        field for field in ["name", "description", "status", "constraints", "benchmark_symbol"]
                        if getattr(command, field) is not None
                    ]
                },
                message="Portfolio updated successfully"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error updating portfolio: {str(e)}"
            )


class AddPositionHandler(CommandHandler[AddPositionCommand]):
    """Handler pour ajouter une position au portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, command: AddPositionCommand) -> CommandResult:
        """
        Traite la commande d'ajout de position.

        Args:
            command: Commande d'ajout de position

        Returns:
            Résultat de l'ajout
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(command.portfolio_id)
            if not portfolio:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

            # Ajouter la position
            portfolio.add_position(command.position)

            # Sauvegarder
            await self.repository.update(portfolio)

            return CommandResult(
                success=True,
                result_data={
                    "portfolio_id": portfolio.id,
                    "symbol": command.position.symbol,
                    "position_value": float(command.position.market_value),
                    "position_weight": float(portfolio.get_position_weight(command.position.symbol))
                },
                message=f"Position {command.position.symbol} added successfully"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error adding position: {str(e)}"
            )


class RemovePositionHandler(CommandHandler[RemovePositionCommand]):
    """Handler pour retirer une position du portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, command: RemovePositionCommand) -> CommandResult:
        """
        Traite la commande de suppression de position.

        Args:
            command: Commande de suppression de position

        Returns:
            Résultat de la suppression
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(command.portfolio_id)
            if not portfolio:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

            # Retirer la position
            removed_position = portfolio.remove_position(command.symbol)

            if not removed_position:
                return CommandResult(
                    success=False,
                    error_message=f"Position {command.symbol} not found in portfolio"
                )

            # Sauvegarder
            await self.repository.update(portfolio)

            return CommandResult(
                success=True,
                result_data={
                    "portfolio_id": portfolio.id,
                    "symbol": command.symbol,
                    "removed_value": float(removed_position.market_value)
                },
                message=f"Position {command.symbol} removed successfully"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error removing position: {str(e)}"
            )


class AdjustCashHandler(CommandHandler[AdjustCashCommand]):
    """Handler pour ajuster le cash du portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, command: AdjustCashCommand) -> CommandResult:
        """
        Traite la commande d'ajustement de cash.

        Args:
            command: Commande d'ajustement de cash

        Returns:
            Résultat de l'ajustement
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(command.portfolio_id)
            if not portfolio:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

            # Ajuster le cash
            old_cash = portfolio.current_cash
            portfolio.adjust_cash(command.amount, command.reason)

            # Sauvegarder
            await self.repository.update(portfolio)

            return CommandResult(
                success=True,
                result_data={
                    "portfolio_id": portfolio.id,
                    "old_cash": float(old_cash),
                    "new_cash": float(portfolio.current_cash),
                    "adjustment": float(command.amount),
                    "reason": command.reason
                },
                message=f"Cash adjusted by {float(command.amount):+.2f}"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error adjusting cash: {str(e)}"
            )


class SetTargetAllocationHandler(CommandHandler[SetTargetAllocationCommand]):
    """Handler pour définir les allocations cibles"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, command: SetTargetAllocationCommand) -> CommandResult:
        """
        Traite la commande de définition d'allocations cibles.

        Args:
            command: Commande de définition d'allocations

        Returns:
            Résultat de la définition
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(command.portfolio_id)
            if not portfolio:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

            # Valider que la somme des allocations fait 1 (100%)
            total_allocation = sum(command.target_allocations.values())
            if not Decimal("0.99") <= total_allocation <= Decimal("1.01"):  # Tolérance de 1%
                return CommandResult(
                    success=False,
                    error_message=f"Target allocations sum to {float(total_allocation):.3f}, must equal 1.0"
                )

            # Définir les allocations cibles
            for symbol, weight in command.target_allocations.items():
                portfolio.set_target_allocation(symbol, weight)

            # Sauvegarder
            await self.repository.update(portfolio)

            return CommandResult(
                success=True,
                result_data={
                    "portfolio_id": portfolio.id,
                    "target_allocations": {symbol: float(weight) for symbol, weight in command.target_allocations.items()},
                    "total_allocation": float(total_allocation)
                },
                message="Target allocations set successfully"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error setting target allocations: {str(e)}"
            )


class RebalancePortfolioHandler(CommandHandler[RebalancePortfolioCommand]):
    """Handler pour rééquilibrer un portfolio"""

    def __init__(self, repository: PortfolioRepository, portfolio_service: PortfolioService):
        self.repository = repository
        self.portfolio_service = portfolio_service

    async def handle(self, command: RebalancePortfolioCommand) -> CommandResult:
        """
        Traite la commande de rééquilibrage.

        Args:
            command: Commande de rééquilibrage

        Returns:
            Résultat du rééquilibrage
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(command.portfolio_id)
            if not portfolio:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

            # Créer le plan de rééquilibrage
            rebalancing_plan = self.portfolio_service.create_rebalancing_plan(
                portfolio,
                command.target_allocations,
                command.rebalancing_threshold
            )

            if not rebalancing_plan and not command.force_rebalance:
                return CommandResult(
                    success=True,
                    result_data={
                        "portfolio_id": portfolio.id,
                        "rebalancing_needed": False
                    },
                    message="Portfolio does not need rebalancing"
                )

            if rebalancing_plan:
                # Exécuter le rééquilibrage (simulation)
                execution_log = self.portfolio_service.execute_rebalancing_plan(portfolio, rebalancing_plan)

                # Sauvegarder
                await self.repository.update(portfolio)

                return CommandResult(
                    success=True,
                    result_data={
                        "portfolio_id": portfolio.id,
                        "rebalancing_plan": {
                            "trades_required": {symbol: float(amount) for symbol, amount in rebalancing_plan.trades_required.items()},
                            "estimated_cost": float(rebalancing_plan.estimated_cost),
                            "reason": rebalancing_plan.reason
                        },
                        "execution_log": execution_log
                    },
                    message="Portfolio rebalanced successfully"
                )
            else:
                return CommandResult(
                    success=False,
                    error_message="Could not create rebalancing plan"
                )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error rebalancing portfolio: {str(e)}"
            )


class OptimizeAllocationHandler(CommandHandler[OptimizeAllocationCommand]):
    """Handler pour optimiser l'allocation d'un portfolio"""

    def __init__(self, repository: PortfolioRepository, portfolio_service: PortfolioService):
        self.repository = repository
        self.portfolio_service = portfolio_service

    async def handle(self, command: OptimizeAllocationCommand) -> CommandResult:
        """
        Traite la commande d'optimisation d'allocation.

        Args:
            command: Commande d'optimisation

        Returns:
            Résultat de l'optimisation
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(command.portfolio_id)
            if not portfolio:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

            # Obtenir les symboles du portfolio
            symbols = list(portfolio.positions.keys())
            if not symbols:
                return CommandResult(
                    success=False,
                    error_message="Portfolio has no positions to optimize"
                )

            # Optimiser selon la méthode choisie
            optimization = None
            parameters = command.parameters or {}

            if command.optimization_method == "equal_weight":
                optimization = self.portfolio_service.optimize_allocation_equal_weight(symbols)

            elif command.optimization_method == "risk_parity":
                # Utiliser des estimations de risque par défaut ou fournies
                risk_estimates = parameters.get("risk_estimates", {symbol: Decimal("0.15") for symbol in symbols})
                optimization = self.portfolio_service.optimize_allocation_risk_parity(symbols, risk_estimates)

            elif command.optimization_method == "momentum":
                # Utiliser des scores de momentum par défaut ou fournis
                momentum_scores = parameters.get("momentum_scores", {symbol: Decimal("0.1") for symbol in symbols})
                optimization = self.portfolio_service.optimize_allocation_momentum(symbols, momentum_scores)

            else:
                return CommandResult(
                    success=False,
                    error_message=f"Unknown optimization method: {command.optimization_method}"
                )

            return CommandResult(
                success=True,
                result_data={
                    "portfolio_id": portfolio.id,
                    "optimization_method": command.optimization_method,
                    "original_allocations": {symbol: float(weight) for symbol, weight in optimization.original_allocations.items()},
                    "optimized_allocations": {symbol: float(weight) for symbol, weight in optimization.optimized_allocations.items()},
                    "expected_return": float(optimization.expected_return),
                    "expected_risk": float(optimization.expected_risk),
                    "sharpe_ratio": float(optimization.sharpe_ratio),
                    "constraints_applied": optimization.constraints_applied
                },
                message=f"Allocation optimized using {command.optimization_method} method"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error optimizing allocation: {str(e)}"
            )


class AddStrategyToPortfolioHandler(CommandHandler[AddStrategyToPortfolioCommand]):
    """Handler pour ajouter une stratégie au portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, command: AddStrategyToPortfolioCommand) -> CommandResult:
        """
        Traite la commande d'ajout de stratégie.

        Args:
            command: Commande d'ajout de stratégie

        Returns:
            Résultat de l'ajout
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(command.portfolio_id)
            if not portfolio:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

            # Ajouter la stratégie
            portfolio.add_strategy(command.strategy_id)

            # Sauvegarder
            await self.repository.update(portfolio)

            return CommandResult(
                success=True,
                result_data={
                    "portfolio_id": portfolio.id,
                    "strategy_id": command.strategy_id,
                    "total_strategies": len(portfolio.strategy_ids)
                },
                message=f"Strategy {command.strategy_id} added to portfolio"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error adding strategy: {str(e)}"
            )


class CreateSnapshotHandler(CommandHandler[CreateSnapshotCommand]):
    """Handler pour créer un snapshot du portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, command: CreateSnapshotCommand) -> CommandResult:
        """
        Traite la commande de création de snapshot.

        Args:
            command: Commande de création de snapshot

        Returns:
            Résultat de la création
        """
        try:
            # Récupérer le portfolio
            portfolio = await self.repository.find_by_id(command.portfolio_id)
            if not portfolio:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

            # Créer et ajouter le snapshot
            snapshot = portfolio.create_snapshot()
            portfolio.add_snapshot(snapshot)

            # Sauvegarder
            await self.repository.update(portfolio)

            return CommandResult(
                success=True,
                result_data={
                    "portfolio_id": portfolio.id,
                    "snapshot": snapshot.to_dict(),
                    "total_snapshots": len(portfolio.snapshots)
                },
                message="Portfolio snapshot created successfully"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error creating snapshot: {str(e)}"
            )


class ArchivePortfolioHandler(CommandHandler[ArchivePortfolioCommand]):
    """Handler pour archiver un portfolio"""

    def __init__(self, repository: PortfolioRepository):
        self.repository = repository

    async def handle(self, command: ArchivePortfolioCommand) -> CommandResult:
        """
        Traite la commande d'archivage.

        Args:
            command: Commande d'archivage

        Returns:
            Résultat de l'archivage
        """
        try:
            # Archiver via le repository
            success = await self.repository.archive(command.portfolio_id)

            if success:
                return CommandResult(
                    success=True,
                    result_data={"portfolio_id": command.portfolio_id},
                    message="Portfolio archived successfully"
                )
            else:
                return CommandResult(
                    success=False,
                    error_message=f"Portfolio not found: {command.portfolio_id}"
                )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Error archiving portfolio: {str(e)}"
            )