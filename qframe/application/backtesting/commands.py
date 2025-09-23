"""
Application Layer: Backtesting Command Handlers
==============================================

Handlers pour les commandes liées au backtesting.
Implémente le pattern CQRS pour les opérations de modification.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any

from ...core.handlers import Command, CommandHandler, CommandResult
from ...domain.entities.backtest import (
    BacktestConfiguration, BacktestResult, BacktestStatus, BacktestType,
    WalkForwardConfig, MonteCarloConfig, RebalanceFrequency
)
from ...domain.services.backtesting_service import BacktestingService
from ...domain.repositories.backtest_repository import BacktestRepository


# Commands

@dataclass
class CreateBacktestConfigurationCommand(Command):
    """Commande pour créer une configuration de backtest"""
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    strategy_ids: List[str]
    strategy_allocations: Optional[Dict[str, Decimal]] = None
    benchmark_symbol: Optional[str] = None
    transaction_cost: Decimal = Decimal("0.001")
    slippage: Decimal = Decimal("0.0005")
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    max_position_size: Decimal = Decimal("0.1")
    max_leverage: Decimal = Decimal("1.0")
    backtest_type: BacktestType = BacktestType.SINGLE_PERIOD
    walk_forward_config: Optional[WalkForwardConfig] = None
    monte_carlo_config: Optional[MonteCarloConfig] = None
    tags: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None


@dataclass
class UpdateBacktestConfigurationCommand(Command):
    """Commande pour mettre à jour une configuration de backtest"""
    configuration_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: Optional[Decimal] = None
    strategy_ids: Optional[List[str]] = None
    strategy_allocations: Optional[Dict[str, Decimal]] = None
    benchmark_symbol: Optional[str] = None
    transaction_cost: Optional[Decimal] = None
    slippage: Optional[Decimal] = None
    rebalance_frequency: Optional[RebalanceFrequency] = None
    max_position_size: Optional[Decimal] = None
    max_leverage: Optional[Decimal] = None
    tags: Optional[Dict[str, Any]] = None


@dataclass
class DeleteBacktestConfigurationCommand(Command):
    """Commande pour supprimer une configuration de backtest"""
    configuration_id: str
    force: bool = False  # Supprimer même s'il y a des résultats associés


@dataclass
class RunBacktestCommand(Command):
    """Commande pour lancer un backtest"""
    configuration_id: str
    name: Optional[str] = None  # Nom personnalisé pour ce run
    tags: Optional[Dict[str, Any]] = None


@dataclass
class StopBacktestCommand(Command):
    """Commande pour arrêter un backtest en cours"""
    result_id: str
    reason: Optional[str] = None


@dataclass
class DeleteBacktestResultCommand(Command):
    """Commande pour supprimer un résultat de backtest"""
    result_id: str


@dataclass
class ArchiveBacktestResultCommand(Command):
    """Commande pour archiver un résultat de backtest"""
    result_id: str


@dataclass
class RestoreBacktestResultCommand(Command):
    """Commande pour restaurer un résultat archivé"""
    result_id: str


@dataclass
class CleanupOldBacktestResultsCommand(Command):
    """Commande pour nettoyer les anciens résultats"""
    days_old: int = 90
    keep_best: int = 10
    dry_run: bool = True


@dataclass
class ExportBacktestResultsCommand(Command):
    """Commande pour exporter des résultats de backtest"""
    result_ids: List[str]
    format: str = "json"  # json, csv, excel


@dataclass
class ImportBacktestResultsCommand(Command):
    """Commande pour importer des résultats de backtest"""
    data: bytes
    format: str = "json"
    overwrite_existing: bool = False


# Command Handlers

class CreateBacktestConfigurationHandler(CommandHandler[CreateBacktestConfigurationCommand]):
    """Handler pour créer une configuration de backtest"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: CreateBacktestConfigurationCommand) -> CommandResult:
        try:
            # Créer la configuration
            config = BacktestConfiguration(
                name=command.name,
                description=command.description,
                start_date=command.start_date,
                end_date=command.end_date,
                initial_capital=command.initial_capital,
                strategy_ids=command.strategy_ids,
                strategy_allocations=command.strategy_allocations or {},
                benchmark_symbol=command.benchmark_symbol,
                transaction_cost=command.transaction_cost,
                slippage=command.slippage,
                rebalance_frequency=command.rebalance_frequency,
                max_position_size=command.max_position_size,
                max_leverage=command.max_leverage,
                backtest_type=command.backtest_type,
                walk_forward_config=command.walk_forward_config,
                monte_carlo_config=command.monte_carlo_config,
                tags=command.tags or {},
                created_by=command.created_by,
                created_at=datetime.utcnow()
            )

            # Valider la configuration
            validation_errors = config.validate()
            if validation_errors:
                return CommandResult(
                    success=False,
                    error_message=f"Configuration validation failed: {', '.join(validation_errors)}"
                )

            # Sauvegarder
            await self.backtest_repository.save_configuration(config)

            return CommandResult(
                success=True,
                data={"configuration_id": config.id, "name": config.name}
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to create backtest configuration: {str(e)}"
            )


class UpdateBacktestConfigurationHandler(CommandHandler[UpdateBacktestConfigurationCommand]):
    """Handler pour mettre à jour une configuration de backtest"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: UpdateBacktestConfigurationCommand) -> CommandResult:
        try:
            # Récupérer la configuration existante
            config = await self.backtest_repository.get_configuration(command.configuration_id)
            if not config:
                return CommandResult(
                    success=False,
                    error_message=f"Configuration {command.configuration_id} not found"
                )

            # Mettre à jour les champs fournis
            if command.name is not None:
                config.name = command.name
            if command.description is not None:
                config.description = command.description
            if command.start_date is not None:
                config.start_date = command.start_date
            if command.end_date is not None:
                config.end_date = command.end_date
            if command.initial_capital is not None:
                config.initial_capital = command.initial_capital
            if command.strategy_ids is not None:
                config.strategy_ids = command.strategy_ids
            if command.strategy_allocations is not None:
                config.strategy_allocations = command.strategy_allocations
            if command.benchmark_symbol is not None:
                config.benchmark_symbol = command.benchmark_symbol
            if command.transaction_cost is not None:
                config.transaction_cost = command.transaction_cost
            if command.slippage is not None:
                config.slippage = command.slippage
            if command.rebalance_frequency is not None:
                config.rebalance_frequency = command.rebalance_frequency
            if command.max_position_size is not None:
                config.max_position_size = command.max_position_size
            if command.max_leverage is not None:
                config.max_leverage = command.max_leverage
            if command.tags is not None:
                config.tags.update(command.tags)

            # Valider la configuration mise à jour
            validation_errors = config.validate()
            if validation_errors:
                return CommandResult(
                    success=False,
                    error_message=f"Updated configuration validation failed: {', '.join(validation_errors)}"
                )

            # Sauvegarder
            await self.backtest_repository.save_configuration(config)

            return CommandResult(
                success=True,
                data={"configuration_id": config.id, "updated_fields": len([f for f in vars(command).values() if f is not None])}
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to update backtest configuration: {str(e)}"
            )


class DeleteBacktestConfigurationHandler(CommandHandler[DeleteBacktestConfigurationCommand]):
    """Handler pour supprimer une configuration de backtest"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: DeleteBacktestConfigurationCommand) -> CommandResult:
        try:
            # Vérifier si la configuration existe
            config = await self.backtest_repository.get_configuration(command.configuration_id)
            if not config:
                return CommandResult(
                    success=False,
                    error_message=f"Configuration {command.configuration_id} not found"
                )

            # Vérifier s'il y a des résultats associés
            if not command.force:
                results = await self.backtest_repository.find_results_by_configuration(command.configuration_id)
                if results:
                    return CommandResult(
                        success=False,
                        error_message=f"Cannot delete configuration with {len(results)} associated results. Use force=True to override."
                    )

            # Supprimer
            success = await self.backtest_repository.delete_configuration(command.configuration_id)

            return CommandResult(
                success=success,
                data={"configuration_id": command.configuration_id} if success else None,
                error_message="Failed to delete configuration" if not success else None
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to delete backtest configuration: {str(e)}"
            )


class RunBacktestHandler(CommandHandler[RunBacktestCommand]):
    """Handler pour lancer un backtest"""

    def __init__(self, backtesting_service: BacktestingService, backtest_repository: BacktestRepository):
        self.backtesting_service = backtesting_service
        self.backtest_repository = backtest_repository

    async def handle(self, command: RunBacktestCommand) -> CommandResult:
        try:
            # Récupérer la configuration
            config = await self.backtest_repository.get_configuration(command.configuration_id)
            if not config:
                return CommandResult(
                    success=False,
                    error_message=f"Configuration {command.configuration_id} not found"
                )

            # Personnaliser le nom si fourni
            if command.name:
                config.name = command.name

            # Ajouter des tags si fournis
            if command.tags:
                config.tags.update(command.tags)

            # Lancer le backtest
            result = await self.backtesting_service.run_backtest(config)

            return CommandResult(
                success=True,
                data={
                    "result_id": result.id,
                    "status": result.status.value,
                    "configuration_id": command.configuration_id
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to run backtest: {str(e)}"
            )


class StopBacktestHandler(CommandHandler[StopBacktestCommand]):
    """Handler pour arrêter un backtest en cours"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: StopBacktestCommand) -> CommandResult:
        try:
            # Récupérer le résultat
            result = await self.backtest_repository.get_result(command.result_id)
            if not result:
                return CommandResult(
                    success=False,
                    error_message=f"Backtest result {command.result_id} not found"
                )

            # Vérifier si le backtest peut être arrêté
            if result.status not in [BacktestStatus.PENDING, BacktestStatus.RUNNING]:
                return CommandResult(
                    success=False,
                    error_message=f"Cannot stop backtest with status {result.status.value}"
                )

            # Mettre à jour le statut
            result.status = BacktestStatus.CANCELLED
            result.end_time = datetime.utcnow()
            if command.reason:
                result.error_message = f"Cancelled: {command.reason}"

            await self.backtest_repository.save_result(result)

            return CommandResult(
                success=True,
                data={"result_id": command.result_id, "status": "cancelled"}
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to stop backtest: {str(e)}"
            )


class DeleteBacktestResultHandler(CommandHandler[DeleteBacktestResultCommand]):
    """Handler pour supprimer un résultat de backtest"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: DeleteBacktestResultCommand) -> CommandResult:
        try:
            success = await self.backtest_repository.delete_result(command.result_id)

            return CommandResult(
                success=success,
                data={"result_id": command.result_id} if success else None,
                error_message="Failed to delete result or result not found" if not success else None
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to delete backtest result: {str(e)}"
            )


class ArchiveBacktestResultHandler(CommandHandler[ArchiveBacktestResultCommand]):
    """Handler pour archiver un résultat de backtest"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: ArchiveBacktestResultCommand) -> CommandResult:
        try:
            success = await self.backtest_repository.archive_result(command.result_id)

            return CommandResult(
                success=success,
                data={"result_id": command.result_id, "status": "archived"} if success else None,
                error_message="Failed to archive result or result not found" if not success else None
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to archive backtest result: {str(e)}"
            )


class RestoreBacktestResultHandler(CommandHandler[RestoreBacktestResultCommand]):
    """Handler pour restaurer un résultat archivé"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: RestoreBacktestResultCommand) -> CommandResult:
        try:
            success = await self.backtest_repository.restore_result(command.result_id)

            return CommandResult(
                success=success,
                data={"result_id": command.result_id, "status": "restored"} if success else None,
                error_message="Failed to restore result or result not found" if not success else None
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to restore backtest result: {str(e)}"
            )


class CleanupOldBacktestResultsHandler(CommandHandler[CleanupOldBacktestResultsCommand]):
    """Handler pour nettoyer les anciens résultats"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: CleanupOldBacktestResultsCommand) -> CommandResult:
        try:
            if command.dry_run:
                # Simulation - compter seulement
                # En réalité, il faudrait implémenter une méthode pour compter
                return CommandResult(
                    success=True,
                    data={"dry_run": True, "would_delete": 0, "days_old": command.days_old}
                )
            else:
                deleted_count = await self.backtest_repository.cleanup_old_results(
                    command.days_old, command.keep_best
                )

                return CommandResult(
                    success=True,
                    data={"deleted_count": deleted_count, "days_old": command.days_old, "kept_best": command.keep_best}
                )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to cleanup old backtest results: {str(e)}"
            )


class ExportBacktestResultsHandler(CommandHandler[ExportBacktestResultsCommand]):
    """Handler pour exporter des résultats de backtest"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: ExportBacktestResultsCommand) -> CommandResult:
        try:
            exported_data = await self.backtest_repository.export_results(
                command.result_ids, command.format
            )

            return CommandResult(
                success=True,
                data={
                    "format": command.format,
                    "result_count": len(command.result_ids),
                    "data_size": len(exported_data),
                    "exported_data": exported_data  # En production, pourrait être un URL de téléchargement
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to export backtest results: {str(e)}"
            )


class ImportBacktestResultsHandler(CommandHandler[ImportBacktestResultsCommand]):
    """Handler pour importer des résultats de backtest"""

    def __init__(self, backtest_repository: BacktestRepository):
        self.backtest_repository = backtest_repository

    async def handle(self, command: ImportBacktestResultsCommand) -> CommandResult:
        try:
            imported_ids = await self.backtest_repository.import_results(
                command.data, command.format
            )

            return CommandResult(
                success=True,
                data={
                    "imported_count": len(imported_ids),
                    "result_ids": imported_ids,
                    "format": command.format
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Failed to import backtest results: {str(e)}"
            )