"""
Application Commands: Strategy Commands
=====================================

Commandes CQRS pour les opérations sur les stratégies.
Encapsulent les données et intentions des use cases.
"""

from typing import List, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal

from ...domain.entities.strategy import StrategyType


@dataclass(frozen=True)  # Immutable command
class CreateStrategyCommand:
    """
    Commande pour créer une nouvelle stratégie.
    """
    name: str
    description: str
    strategy_type: StrategyType
    universe: List[str]  # Symboles à trader
    max_position_size: Decimal
    max_positions: int
    risk_per_trade: Decimal

    # Métadonnées optionnelles
    author: Optional[str] = None
    version: str = "1.0.0"

    def __post_init__(self):
        """Validation des données de la commande."""
        if not self.name or not self.name.strip():
            raise ValueError("Strategy name cannot be empty")

        if not self.universe:
            raise ValueError("Universe cannot be empty")

        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1")

        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")

        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            raise ValueError("risk_per_trade must be between 0 and 0.1")


@dataclass(frozen=True)
class UpdateStrategyCommand:
    """
    Commande pour mettre à jour une stratégie existante.
    """
    strategy_id: str

    # Champs optionnels à mettre à jour
    name: Optional[str] = None
    description: Optional[str] = None
    universe: Optional[List[str]] = None
    max_position_size: Optional[Decimal] = None
    max_positions: Optional[int] = None
    risk_per_trade: Optional[Decimal] = None

    def __post_init__(self):
        """Validation des données de mise à jour."""
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")

        # Validation des champs optionnels si fournis
        if self.name is not None and not self.name.strip():
            raise ValueError("name cannot be empty if provided")

        if self.universe is not None and len(self.universe) == 0:
            raise ValueError("universe cannot be empty if provided")

        if self.max_position_size is not None:
            if self.max_position_size <= 0 or self.max_position_size > 1:
                raise ValueError("max_position_size must be between 0 and 1")

        if self.max_positions is not None and self.max_positions <= 0:
            raise ValueError("max_positions must be positive")

        if self.risk_per_trade is not None:
            if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
                raise ValueError("risk_per_trade must be between 0 and 0.1")


@dataclass(frozen=True)
class ActivateStrategyCommand:
    """
    Commande pour activer une stratégie.
    """
    strategy_id: str
    validate_performance: bool = True  # Valider les performances avant activation
    dry_run: bool = False  # Mode simulation

    def __post_init__(self):
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")


@dataclass(frozen=True)
class DeactivateStrategyCommand:
    """
    Commande pour désactiver une stratégie.
    """
    strategy_id: str
    pause_only: bool = False  # True = pause, False = stop complètement
    close_positions: bool = True  # Fermer les positions ouvertes
    force_close: bool = False  # Forcer la fermeture même en cas d'erreur

    def __post_init__(self):
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")


@dataclass(frozen=True)
class DeleteStrategyCommand:
    """
    Commande pour supprimer une stratégie.
    """
    strategy_id: str
    force: bool = False  # Forcer la suppression même si active/avec positions
    backup_before_delete: bool = True  # Créer une sauvegarde avant suppression

    def __post_init__(self):
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")


@dataclass(frozen=True)
class AddSignalToStrategyCommand:
    """
    Commande pour ajouter un signal à une stratégie.
    """
    strategy_id: str
    symbol: str
    action: str  # "buy", "sell", "hold", "close"
    strength: Decimal
    confidence: str  # "low", "medium", "high", "very_high"
    price: Decimal

    # Optionnel
    quantity: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    expiry: Optional[datetime] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.strength < 0 or self.strength > 1:
            raise ValueError("strength must be between 0 and 1")
        if self.price <= 0:
            raise ValueError("price must be positive")


@dataclass(frozen=True)
class UpdateStrategyPerformanceCommand:
    """
    Commande pour mettre à jour les performances d'une stratégie.
    """
    strategy_id: str
    total_pnl: Decimal
    total_trades: int
    winning_trades: int
    max_drawdown: Decimal
    sharpe_ratio: Optional[Decimal] = None

    def __post_init__(self):
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if self.total_trades < 0:
            raise ValueError("total_trades cannot be negative")
        if self.winning_trades < 0 or self.winning_trades > self.total_trades:
            raise ValueError("winning_trades must be between 0 and total_trades")
        if self.max_drawdown < 0:
            raise ValueError("max_drawdown cannot be negative")


@dataclass(frozen=True)
class BulkActivateStrategiesCommand:
    """
    Commande pour activer plusieurs stratégies en lot.
    """
    strategy_ids: List[str]
    validate_performance: bool = True
    stop_on_first_error: bool = False  # Arrêter au premier échec ou continuer

    def __post_init__(self):
        if not self.strategy_ids:
            raise ValueError("strategy_ids cannot be empty")
        if len(set(self.strategy_ids)) != len(self.strategy_ids):
            raise ValueError("strategy_ids must be unique")


@dataclass(frozen=True)
class BulkDeactivateStrategiesCommand:
    """
    Commande pour désactiver plusieurs stratégies en lot.
    """
    strategy_ids: List[str]
    pause_only: bool = False
    close_positions: bool = True
    stop_on_first_error: bool = False

    def __post_init__(self):
        if not self.strategy_ids:
            raise ValueError("strategy_ids cannot be empty")
        if len(set(self.strategy_ids)) != len(self.strategy_ids):
            raise ValueError("strategy_ids must be unique")


@dataclass(frozen=True)
class BackupStrategyCommand:
    """
    Commande pour sauvegarder une stratégie.
    """
    strategy_id: str
    backup_name: Optional[str] = None
    include_history: bool = True  # Inclure l'historique des signaux

    def __post_init__(self):
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")


@dataclass(frozen=True)
class RestoreStrategyCommand:
    """
    Commande pour restaurer une stratégie depuis une sauvegarde.
    """
    backup_id: str
    new_strategy_name: Optional[str] = None  # Nouveau nom si restoration
    overwrite_existing: bool = False  # Écraser si existe déjà

    def __post_init__(self):
        if not self.backup_id:
            raise ValueError("backup_id cannot be empty")


@dataclass(frozen=True)
class CloneStrategyCommand:
    """
    Commande pour cloner une stratégie existante.
    """
    source_strategy_id: str
    new_name: str
    copy_performance_history: bool = False  # Copier l'historique de performance
    copy_signal_history: bool = False  # Copier l'historique des signaux

    def __post_init__(self):
        if not self.source_strategy_id:
            raise ValueError("source_strategy_id cannot be empty")
        if not self.new_name or not self.new_name.strip():
            raise ValueError("new_name cannot be empty")


# Factory functions pour création simplifiée des commandes

def create_strategy_command(
    name: str,
    strategy_type: StrategyType,
    universe: List[str],
    description: str = "",
    max_position_size: float = 0.02,
    max_positions: int = 5,
    risk_per_trade: float = 0.01
) -> CreateStrategyCommand:
    """Factory pour créer une commande de création de stratégie."""
    return CreateStrategyCommand(
        name=name,
        description=description,
        strategy_type=strategy_type,
        universe=universe,
        max_position_size=Decimal(str(max_position_size)),
        max_positions=max_positions,
        risk_per_trade=Decimal(str(risk_per_trade))
    )


def update_strategy_risk_command(
    strategy_id: str,
    max_position_size: Optional[float] = None,
    max_positions: Optional[int] = None,
    risk_per_trade: Optional[float] = None
) -> UpdateStrategyCommand:
    """Factory pour créer une commande de mise à jour des paramètres de risque."""
    return UpdateStrategyCommand(
        strategy_id=strategy_id,
        max_position_size=Decimal(str(max_position_size)) if max_position_size else None,
        max_positions=max_positions,
        risk_per_trade=Decimal(str(risk_per_trade)) if risk_per_trade else None
    )


def activate_strategy_command(strategy_id: str, dry_run: bool = False) -> ActivateStrategyCommand:
    """Factory pour créer une commande d'activation."""
    return ActivateStrategyCommand(
        strategy_id=strategy_id,
        dry_run=dry_run
    )


def deactivate_strategy_command(
    strategy_id: str,
    pause_only: bool = False,
    close_positions: bool = True
) -> DeactivateStrategyCommand:
    """Factory pour créer une commande de désactivation."""
    return DeactivateStrategyCommand(
        strategy_id=strategy_id,
        pause_only=pause_only,
        close_positions=close_positions
    )