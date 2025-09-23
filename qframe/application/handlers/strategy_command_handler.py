"""
Application Handler: StrategyCommandHandler
==========================================

Handler pour les commandes liées aux stratégies.
Orchestre les use cases de création, modification et suppression des stratégies.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

from ...domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from ...domain.repositories.strategy_repository import StrategyRepository, RepositoryError
from ...domain.services.signal_service import SignalService
from ..commands.strategy_commands import (
    CreateStrategyCommand,
    UpdateStrategyCommand,
    ActivateStrategyCommand,
    DeactivateStrategyCommand,
    DeleteStrategyCommand
)

logger = logging.getLogger(__name__)


class StrategyCommandHandler:
    """
    Handler pour les commandes de stratégie.

    Implémente les use cases de la couche application
    en orchestrant les services du domaine.
    """

    def __init__(
        self,
        strategy_repository: StrategyRepository,
        signal_service: SignalService
    ):
        self.strategy_repository = strategy_repository
        self.signal_service = signal_service

    async def handle_create_strategy(self, command: CreateStrategyCommand) -> str:
        """
        Traite la commande de création d'une stratégie.

        Args:
            command: Commande de création

        Returns:
            ID de la stratégie créée

        Raises:
            ValueError: Si les données sont invalides
            RepositoryError: Si erreur de persistance
        """
        logger.info(f"📝 Création d'une nouvelle stratégie: {command.name}")

        # Vérifier que le nom n'existe pas déjà
        existing = await self.strategy_repository.find_by_name(command.name)
        if existing:
            raise ValueError(f"Strategy with name '{command.name}' already exists")

        # Créer l'entité stratégie
        strategy = Strategy(
            name=command.name,
            description=command.description,
            strategy_type=command.strategy_type,
            universe=set(command.universe),
            max_position_size=command.max_position_size,
            max_positions=command.max_positions,
            risk_per_trade=command.risk_per_trade
        )

        # Valider la stratégie (les invariants sont vérifiés dans __post_init__)
        # Ici on peut ajouter des validations métier supplémentaires
        await self._validate_strategy_business_rules(strategy)

        # Sauvegarder
        await self.strategy_repository.save(strategy)

        logger.info(f"✅ Stratégie créée avec succès: {strategy.id}")
        return strategy.id

    async def handle_update_strategy(self, command: UpdateStrategyCommand) -> None:
        """
        Traite la commande de mise à jour d'une stratégie.

        Args:
            command: Commande de mise à jour

        Raises:
            ValueError: Si la stratégie n'existe pas ou données invalides
            RepositoryError: Si erreur de persistance
        """
        logger.info(f"🔄 Mise à jour de la stratégie: {command.strategy_id}")

        # Récupérer la stratégie existante
        strategy = await self.strategy_repository.find_by_id(command.strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {command.strategy_id} not found")

        # Vérifier que la stratégie peut être modifiée
        if strategy.status == StrategyStatus.ACTIVE:
            logger.warning("⚠️ Modification d'une stratégie active")

        # Créer une nouvelle version avec les modifications
        updated_strategy = Strategy(
            id=strategy.id,
            name=command.name or strategy.name,
            description=command.description or strategy.description,
            strategy_type=strategy.strategy_type,  # Type ne peut pas changer
            version=strategy.version,
            author=strategy.author,
            created_at=strategy.created_at,
            updated_at=datetime.utcnow(),
            status=strategy.status,
            universe=set(command.universe) if command.universe else strategy.universe,
            max_position_size=command.max_position_size or strategy.max_position_size,
            max_positions=command.max_positions or strategy.max_positions,
            risk_per_trade=command.risk_per_trade or strategy.risk_per_trade,
            total_trades=strategy.total_trades,
            winning_trades=strategy.winning_trades,
            total_pnl=strategy.total_pnl,
            max_drawdown=strategy.max_drawdown,
            sharpe_ratio=strategy.sharpe_ratio,
            positions=strategy.positions,
            signal_history=strategy.signal_history
        )

        # Valider les nouvelles données
        await self._validate_strategy_business_rules(updated_strategy)

        # Sauvegarder
        await self.strategy_repository.update(updated_strategy)

        logger.info(f"✅ Stratégie mise à jour: {command.strategy_id}")

    async def handle_activate_strategy(self, command: ActivateStrategyCommand) -> None:
        """
        Traite la commande d'activation d'une stratégie.

        Args:
            command: Commande d'activation

        Raises:
            ValueError: Si la stratégie ne peut pas être activée
        """
        logger.info(f"▶️ Activation de la stratégie: {command.strategy_id}")

        strategy = await self.strategy_repository.find_by_id(command.strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {command.strategy_id} not found")

        # Vérifications préalables à l'activation
        await self._validate_strategy_activation(strategy)

        # Activer la stratégie
        strategy.activate()

        # Sauvegarder
        await self.strategy_repository.update(strategy)

        logger.info(f"✅ Stratégie activée: {command.strategy_id}")

    async def handle_deactivate_strategy(self, command: DeactivateStrategyCommand) -> None:
        """
        Traite la commande de désactivation d'une stratégie.

        Args:
            command: Commande de désactivation
        """
        logger.info(f"⏸️ Désactivation de la stratégie: {command.strategy_id}")

        strategy = await self.strategy_repository.find_by_id(command.strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {command.strategy_id} not found")

        # Gérer les positions ouvertes selon la stratégie de fermeture
        if command.close_positions and strategy.positions:
            await self._close_strategy_positions(strategy, command.force_close)

        # Désactiver selon le mode
        if command.pause_only:
            strategy.pause()
        else:
            strategy.stop()

        # Sauvegarder
        await self.strategy_repository.update(strategy)

        logger.info(f"✅ Stratégie {'mise en pause' if command.pause_only else 'arrêtée'}: {command.strategy_id}")

    async def handle_delete_strategy(self, command: DeleteStrategyCommand) -> None:
        """
        Traite la commande de suppression d'une stratégie.

        Args:
            command: Commande de suppression

        Raises:
            ValueError: Si la stratégie ne peut pas être supprimée
        """
        logger.info(f"🗑️ Suppression de la stratégie: {command.strategy_id}")

        strategy = await self.strategy_repository.find_by_id(command.strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {command.strategy_id} not found")

        # Vérifications de sécurité
        if strategy.status == StrategyStatus.ACTIVE and not command.force:
            raise ValueError("Cannot delete active strategy without force flag")

        if strategy.positions and not command.force:
            raise ValueError("Cannot delete strategy with open positions without force flag")

        # Sauvegarde optionnelle avant suppression
        if command.backup_before_delete:
            backup_id = await self.strategy_repository.backup_strategy(command.strategy_id)
            logger.info(f"💾 Sauvegarde créée: {backup_id}")

        # Fermer toutes les positions si nécessaire
        if strategy.positions:
            await self._close_strategy_positions(strategy, force=True)

        # Supprimer
        deleted = await self.strategy_repository.delete(command.strategy_id)
        if not deleted:
            raise ValueError(f"Failed to delete strategy {command.strategy_id}")

        logger.info(f"✅ Stratégie supprimée: {command.strategy_id}")

    async def _validate_strategy_business_rules(self, strategy: Strategy) -> None:
        """Valide les règles métier pour une stratégie."""

        # Vérification de l'univers de trading
        if not strategy.universe:
            raise ValueError("Strategy must have at least one symbol in universe")

        # Vérification des paramètres de risque
        if strategy.max_position_size * strategy.max_positions > 1.0:
            raise ValueError("Total possible exposure exceeds 100%")

        # Vérifications spécifiques par type de stratégie
        if strategy.strategy_type == StrategyType.ARBITRAGE:
            if len(strategy.universe) < 2:
                raise ValueError("Arbitrage strategy requires at least 2 symbols")

        # Autres validations métier...

    async def _validate_strategy_activation(self, strategy: Strategy) -> None:
        """Valide qu'une stratégie peut être activée."""

        # Vérifier l'état
        if strategy.status == StrategyStatus.ERROR:
            raise ValueError("Cannot activate strategy in error state")

        # Vérifier la configuration
        if not strategy.universe:
            raise ValueError("Cannot activate strategy without trading universe")

        # Vérifier les performances historiques (optionnel)
        if strategy.total_trades > 10 and strategy.sharpe_ratio and strategy.sharpe_ratio < -1:
            logger.warning("⚠️ Activating strategy with poor historical performance")

        # Autres vérifications...

    async def _close_strategy_positions(self, strategy: Strategy, force: bool = False) -> None:
        """Ferme toutes les positions d'une stratégie."""
        logger.info(f"🔒 Fermeture de {len(strategy.positions)} positions")

        for symbol, position in strategy.positions.items():
            try:
                # Ici on utiliserait un service d'exécution pour fermer la position
                # Pour l'instant, on simule la fermeture
                closed_position = position.close_position(position.current_price)
                logger.info(f"✅ Position fermée: {symbol}, PnL: {closed_position.realized_pnl}")

                # Mettre à jour les métriques de la stratégie
                strategy._update_trade_metrics(closed_position)

            except Exception as e:
                if not force:
                    raise ValueError(f"Failed to close position {symbol}: {e}")
                else:
                    logger.error(f"❌ Erreur fermeture forcée {symbol}: {e}")

        # Vider les positions
        strategy.positions.clear()

    async def get_strategy_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques globales des stratégies."""
        return await self.strategy_repository.get_statistics()

    async def bulk_activate_strategies(self, strategy_ids: List[str]) -> Dict[str, Any]:
        """Active plusieurs stratégies en lot."""
        results = {"success": [], "errors": []}

        for strategy_id in strategy_ids:
            try:
                command = ActivateStrategyCommand(strategy_id=strategy_id)
                await self.handle_activate_strategy(command)
                results["success"].append(strategy_id)
            except Exception as e:
                results["errors"].append({"strategy_id": strategy_id, "error": str(e)})

        return results

    async def bulk_deactivate_strategies(
        self,
        strategy_ids: List[str],
        pause_only: bool = True
    ) -> Dict[str, Any]:
        """Désactive plusieurs stratégies en lot."""
        results = {"success": [], "errors": []}

        for strategy_id in strategy_ids:
            try:
                command = DeactivateStrategyCommand(
                    strategy_id=strategy_id,
                    pause_only=pause_only
                )
                await self.handle_deactivate_strategy(command)
                results["success"].append(strategy_id)
            except Exception as e:
                results["errors"].append({"strategy_id": strategy_id, "error": str(e)})

        return results