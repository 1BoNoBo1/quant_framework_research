"""
Infrastructure Layer: Memory Backtest Repository
===============================================

Implémentation en mémoire du repository des backtests.
Utilisée pour le développement et les tests.
"""

import json
import pickle
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from collections import defaultdict

from qframe.domain.entities.backtest import (
    BacktestConfiguration, BacktestResult, BacktestStatus, BacktestType, BacktestMetrics
)
from qframe.domain.repositories.backtest_repository import BacktestRepository


class MemoryBacktestRepository(BacktestRepository):
    """Repository en mémoire pour les backtests"""

    def __init__(self):
        # Stockage principal
        self._configurations: Dict[str, BacktestConfiguration] = {}
        self._results: Dict[str, BacktestResult] = {}

        # Index pour performance
        self._configs_by_name: Dict[str, List[str]] = defaultdict(list)
        self._results_by_config: Dict[str, List[str]] = defaultdict(list)
        self._results_by_status: Dict[BacktestStatus, List[str]] = defaultdict(list)
        self._results_by_strategy: Dict[str, List[str]] = defaultdict(list)

        # Archive
        self._archived_results: Dict[str, BacktestResult] = {}

    # Configuration Management

    async def save_configuration(self, config: BacktestConfiguration) -> None:
        """Sauvegarde une configuration de backtest"""
        old_config = self._configurations.get(config.id)

        # Mettre à jour les index
        if old_config:
            self._remove_config_from_indexes(old_config)

        self._configurations[config.id] = config
        self._add_config_to_indexes(config)

    async def get_configuration(self, config_id: str) -> Optional[BacktestConfiguration]:
        """Récupère une configuration par son ID"""
        return self._configurations.get(config_id)

    async def find_configurations_by_name(self, name: str) -> List[BacktestConfiguration]:
        """Trouve des configurations par nom"""
        # Recherche exacte et partielle
        matching_configs = []

        for config in self._configurations.values():
            if name.lower() in config.name.lower():
                matching_configs.append(config)

        return matching_configs

    async def get_all_configurations(self) -> List[BacktestConfiguration]:
        """Récupère toutes les configurations"""
        return list(self._configurations.values())

    async def delete_configuration(self, config_id: str) -> bool:
        """Supprime une configuration"""
        config = self._configurations.get(config_id)
        if not config:
            return False

        self._remove_config_from_indexes(config)
        del self._configurations[config_id]
        return True

    # Result Management

    async def save_result(self, result: BacktestResult) -> None:
        """Sauvegarde un résultat de backtest"""
        old_result = self._results.get(result.id)

        # Mettre à jour les index
        if old_result:
            self._remove_result_from_indexes(old_result)

        self._results[result.id] = result
        self._add_result_to_indexes(result)

    async def get_result(self, result_id: str) -> Optional[BacktestResult]:
        """Récupère un résultat par son ID"""
        return self._results.get(result_id)

    async def find_results_by_configuration(self, config_id: str) -> List[BacktestResult]:
        """Trouve tous les résultats pour une configuration"""
        result_ids = self._results_by_config.get(config_id, [])
        return [self._results[result_id] for result_id in result_ids if result_id in self._results]

    async def find_results_by_status(self, status: BacktestStatus) -> List[BacktestResult]:
        """Trouve tous les résultats par statut"""
        result_ids = self._results_by_status.get(status, [])
        return [self._results[result_id] for result_id in result_ids if result_id in self._results]

    async def find_results_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[BacktestResult]:
        """Trouve les résultats créés dans une plage de dates"""
        return [
            result for result in self._results.values()
            if start_date <= result.created_at <= end_date
        ]

    async def find_results_by_strategy(self, strategy_id: str) -> List[BacktestResult]:
        """Trouve tous les résultats qui utilisent une stratégie donnée"""
        result_ids = self._results_by_strategy.get(strategy_id, [])
        return [self._results[result_id] for result_id in result_ids if result_id in self._results]

    async def get_latest_results(self, limit: int = 10) -> List[BacktestResult]:
        """Récupère les derniers résultats"""
        results = sorted(
            self._results.values(),
            key=lambda r: r.created_at,
            reverse=True
        )
        return results[:limit]

    async def delete_result(self, result_id: str) -> bool:
        """Supprime un résultat"""
        result = self._results.get(result_id)
        if not result:
            return False

        self._remove_result_from_indexes(result)
        del self._results[result_id]
        return True

    # Advanced Queries

    async def find_best_performing_results(
        self,
        metric: str = "sharpe_ratio",
        limit: int = 10,
        min_trades: int = 10
    ) -> List[BacktestResult]:
        """Trouve les backtests les plus performants selon une métrique"""
        valid_results = [
            result for result in self._results.values()
            if (result.metrics and
                result.metrics.total_trades >= min_trades and
                hasattr(result.metrics, metric))
        ]

        # Trier par métrique
        if metric == "max_drawdown":
            # Pour le drawdown, moins négatif est mieux
            valid_results.sort(key=lambda r: getattr(r.metrics, metric), reverse=True)
        else:
            # Pour les autres métriques, plus grand est mieux
            valid_results.sort(key=lambda r: getattr(r.metrics, metric), reverse=True)

        return valid_results[:limit]

    async def find_results_by_metrics_criteria(
        self,
        min_sharpe_ratio: Optional[Decimal] = None,
        max_drawdown: Optional[Decimal] = None,
        min_win_rate: Optional[Decimal] = None,
        min_return: Optional[Decimal] = None
    ) -> List[BacktestResult]:
        """Trouve les résultats selon des critères de métriques"""
        matching_results = []

        for result in self._results.values():
            if not result.metrics:
                continue

            metrics = result.metrics

            # Vérifier tous les critères
            if min_sharpe_ratio and metrics.sharpe_ratio < min_sharpe_ratio:
                continue
            if max_drawdown and metrics.max_drawdown < max_drawdown:  # max_drawdown est négatif
                continue
            if min_win_rate and metrics.win_rate < min_win_rate:
                continue
            if min_return and metrics.total_return < min_return:
                continue

            matching_results.append(result)

        return matching_results

    async def get_performance_comparison(
        self,
        result_ids: List[str]
    ) -> Dict[str, BacktestMetrics]:
        """Compare les métriques de plusieurs backtests"""
        comparison = {}

        for result_id in result_ids:
            result = self._results.get(result_id)
            if result and result.metrics:
                comparison[result_id] = result.metrics

        return comparison

    async def search_results(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[BacktestResult]:
        """Recherche textuelle dans les backtests"""
        query_lower = query.lower()
        matching_results = []

        for result in self._results.values():
            # Rechercher dans le nom et les tags
            if (query_lower in result.name.lower() or
                any(query_lower in str(tag).lower() for tag in result.tags.values())):

                # Appliquer les filtres si fournis
                if filters:
                    if not self._matches_filters(result, filters):
                        continue

                matching_results.append(result)

        return matching_results

    # Statistics and Analytics

    async def get_backtest_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques globales sur les backtests"""
        total_configs = len(self._configurations)
        total_results = len(self._results)

        # Statistiques par statut
        status_counts = {
            status.value: len(result_ids)
            for status, result_ids in self._results_by_status.items()
        }

        # Métriques moyennes
        valid_results = [r for r in self._results.values() if r.metrics]

        avg_metrics = {}
        if valid_results:
            avg_metrics = {
                "avg_sharpe_ratio": float(sum(r.metrics.sharpe_ratio for r in valid_results) / len(valid_results)),
                "avg_return": float(sum(r.metrics.total_return for r in valid_results) / len(valid_results)),
                "avg_max_drawdown": float(sum(r.metrics.max_drawdown for r in valid_results) / len(valid_results)),
                "avg_win_rate": float(sum(r.metrics.win_rate for r in valid_results) / len(valid_results)),
            }

        return {
            "total_configurations": total_configs,
            "total_results": total_results,
            "status_distribution": status_counts,
            "average_metrics": avg_metrics,
            "completed_results": len([r for r in self._results.values() if r.status == BacktestStatus.COMPLETED]),
            "failed_results": len([r for r in self._results.values() if r.status == BacktestStatus.FAILED]),
            "archived_results": len(self._archived_results)
        }

    async def get_strategy_performance_summary(
        self,
        strategy_id: str
    ) -> Dict[str, Any]:
        """Retourne un résumé de performance pour une stratégie"""
        strategy_results = await self.find_results_by_strategy(strategy_id)
        completed_results = [r for r in strategy_results if r.status == BacktestStatus.COMPLETED and r.metrics]

        if not completed_results:
            return {"strategy_id": strategy_id, "total_backtests": 0}

        # Calculer les statistiques
        returns = [float(r.metrics.total_return) for r in completed_results]
        sharpe_ratios = [float(r.metrics.sharpe_ratio) for r in completed_results]
        max_drawdowns = [float(r.metrics.max_drawdown) for r in completed_results]

        return {
            "strategy_id": strategy_id,
            "total_backtests": len(strategy_results),
            "completed_backtests": len(completed_results),
            "avg_return": sum(returns) / len(returns),
            "best_return": max(returns),
            "worst_return": min(returns),
            "avg_sharpe": sum(sharpe_ratios) / len(sharpe_ratios),
            "best_sharpe": max(sharpe_ratios),
            "avg_max_drawdown": sum(max_drawdowns) / len(max_drawdowns),
            "worst_drawdown": min(max_drawdowns)
        }

    async def get_monthly_performance_summary(
        self,
        year: int
    ) -> Dict[str, Any]:
        """Retourne un résumé mensuel des performances"""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)

        year_results = await self.find_results_by_date_range(start_date, end_date)
        completed_results = [r for r in year_results if r.status == BacktestStatus.COMPLETED and r.metrics]

        # Grouper par mois
        monthly_stats = {}
        for month in range(1, 13):
            month_results = [
                r for r in completed_results
                if r.created_at.month == month
            ]

            if month_results:
                monthly_stats[month] = {
                    "count": len(month_results),
                    "avg_return": float(sum(r.metrics.total_return for r in month_results) / len(month_results)),
                    "avg_sharpe": float(sum(r.metrics.sharpe_ratio for r in month_results) / len(month_results))
                }
            else:
                monthly_stats[month] = {"count": 0, "avg_return": 0, "avg_sharpe": 0}

        return {
            "year": year,
            "total_backtests": len(year_results),
            "completed_backtests": len(completed_results),
            "monthly_breakdown": monthly_stats
        }

    async def find_similar_configurations(
        self,
        config: BacktestConfiguration,
        similarity_threshold: float = 0.8
    ) -> List[BacktestConfiguration]:
        """Trouve des configurations similaires"""
        similar_configs = []

        for other_config in self._configurations.values():
            if other_config.id == config.id:
                continue

            similarity = self._calculate_config_similarity(config, other_config)
            if similarity >= similarity_threshold:
                similar_configs.append(other_config)

        return similar_configs

    # Data Management

    async def cleanup_old_results(
        self,
        days_old: int = 90,
        keep_best: int = 10
    ) -> int:
        """Nettoie les anciens résultats en gardant les meilleurs"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        # Identifier les anciens résultats
        old_results = [
            r for r in self._results.values()
            if r.created_at < cutoff_date and r.status == BacktestStatus.COMPLETED
        ]

        if len(old_results) <= keep_best:
            return 0

        # Trier par Sharpe ratio et garder les meilleurs
        old_results.sort(
            key=lambda r: float(r.metrics.sharpe_ratio) if r.metrics else -999,
            reverse=True
        )

        to_delete = old_results[keep_best:]
        deleted_count = 0

        for result in to_delete:
            await self.delete_result(result.id)
            deleted_count += 1

        return deleted_count

    async def get_storage_usage(self) -> Dict[str, Any]:
        """Retourne l'utilisation de l'espace de stockage"""
        # Estimation approximative de la taille
        config_size = len(pickle.dumps(self._configurations))
        result_size = len(pickle.dumps(self._results))
        archived_size = len(pickle.dumps(self._archived_results))

        return {
            "total_size_bytes": config_size + result_size + archived_size,
            "configurations_size_bytes": config_size,
            "results_size_bytes": result_size,
            "archived_size_bytes": archived_size,
            "total_configurations": len(self._configurations),
            "total_results": len(self._results),
            "archived_results": len(self._archived_results)
        }

    async def export_results(
        self,
        result_ids: List[str],
        format: str = "json"
    ) -> bytes:
        """Exporte des résultats dans un format donné"""
        results_to_export = []

        for result_id in result_ids:
            result = self._results.get(result_id)
            if result:
                # Sérialiser le résultat
                result_dict = {
                    "id": result.id,
                    "configuration_id": result.configuration_id,
                    "name": result.name,
                    "status": result.status.value,
                    "start_time": result.start_time.isoformat() if result.start_time else None,
                    "end_time": result.end_time.isoformat() if result.end_time else None,
                    "initial_capital": float(result.initial_capital),
                    "final_capital": float(result.final_capital),
                    "created_at": result.created_at.isoformat(),
                    "tags": result.tags
                }

                if result.metrics:
                    result_dict["metrics"] = {
                        "total_return": float(result.metrics.total_return),
                        "sharpe_ratio": float(result.metrics.sharpe_ratio),
                        "max_drawdown": float(result.metrics.max_drawdown),
                        "win_rate": float(result.metrics.win_rate),
                        "total_trades": result.metrics.total_trades
                    }

                results_to_export.append(result_dict)

        if format == "json":
            return json.dumps(results_to_export, indent=2).encode('utf-8')
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def import_results(
        self,
        data: bytes,
        format: str = "json"
    ) -> List[str]:
        """Importe des résultats depuis des données"""
        if format == "json":
            results_data = json.loads(data.decode('utf-8'))
            imported_ids = []

            for result_data in results_data:
                # Créer un objet BacktestResult à partir des données
                # (Version simplifiée pour l'exemple)
                result = BacktestResult(
                    id=result_data.get("id", ""),
                    configuration_id=result_data.get("configuration_id", ""),
                    name=result_data.get("name", ""),
                    status=BacktestStatus(result_data.get("status", "completed")),
                    initial_capital=Decimal(str(result_data.get("initial_capital", 0))),
                    final_capital=Decimal(str(result_data.get("final_capital", 0)))
                )

                await self.save_result(result)
                imported_ids.append(result.id)

            return imported_ids
        else:
            raise ValueError(f"Unsupported import format: {format}")

    # Batch Operations

    async def batch_save_results(self, results: List[BacktestResult]) -> None:
        """Sauvegarde plusieurs résultats en lot"""
        for result in results:
            await self.save_result(result)

    async def batch_update_status(
        self,
        result_ids: List[str],
        status: BacktestStatus
    ) -> int:
        """Met à jour le statut de plusieurs résultats"""
        updated_count = 0

        for result_id in result_ids:
            result = self._results.get(result_id)
            if result:
                old_status = result.status
                result.status = status

                # Mettre à jour les index
                self._results_by_status[old_status].remove(result_id)
                self._results_by_status[status].append(result_id)

                updated_count += 1

        return updated_count

    async def get_results_count_by_status(self) -> Dict[BacktestStatus, int]:
        """Compte les résultats par statut"""
        return {
            status: len(result_ids)
            for status, result_ids in self._results_by_status.items()
        }

    async def get_results_count_by_type(self) -> Dict[BacktestType, int]:
        """Compte les résultats par type de backtest"""
        type_counts = defaultdict(int)

        for result in self._results.values():
            # Récupérer le type depuis la configuration
            config = self._configurations.get(result.configuration_id)
            if config:
                type_counts[config.backtest_type] += 1

        return dict(type_counts)

    # Archive and Restore

    async def archive_result(self, result_id: str) -> bool:
        """Archive un résultat"""
        result = self._results.get(result_id)
        if not result:
            return False

        self._archived_results[result_id] = result
        await self.delete_result(result_id)
        return True

    async def restore_result(self, result_id: str) -> bool:
        """Restaure un résultat archivé"""
        result = self._archived_results.get(result_id)
        if not result:
            return False

        await self.save_result(result)
        del self._archived_results[result_id]
        return True

    async def get_archived_results(self) -> List[BacktestResult]:
        """Récupère tous les résultats archivés"""
        return list(self._archived_results.values())

    # Validation and Health

    async def validate_result_integrity(self, result_id: str) -> List[str]:
        """Valide l'intégrité d'un résultat et retourne les erreurs"""
        errors = []
        result = self._results.get(result_id)

        if not result:
            errors.append(f"Result {result_id} not found")
            return errors

        # Vérifications de base
        if result.status == BacktestStatus.COMPLETED:
            if not result.metrics:
                errors.append("Completed result missing metrics")
            if result.final_capital <= 0:
                errors.append("Invalid final capital")

        # Vérifier la configuration associée
        if not self._configurations.get(result.configuration_id):
            errors.append("Associated configuration not found")

        return errors

    async def get_repository_health(self) -> Dict[str, Any]:
        """Retourne l'état de santé du repository"""
        total_results = len(self._results)
        completed_results = len([r for r in self._results.values() if r.status == BacktestStatus.COMPLETED])
        failed_results = len([r for r in self._results.values() if r.status == BacktestStatus.FAILED])

        health_score = completed_results / total_results if total_results > 0 else 1.0

        return {
            "status": "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.5 else "unhealthy",
            "total_configurations": len(self._configurations),
            "total_results": total_results,
            "completed_results": completed_results,
            "failed_results": failed_results,
            "success_rate": health_score,
            "storage_usage": await self.get_storage_usage()
        }

    # Helper methods

    def _add_config_to_indexes(self, config: BacktestConfiguration) -> None:
        """Ajoute une configuration aux index"""
        self._configs_by_name[config.name.lower()].append(config.id)

    def _remove_config_from_indexes(self, config: BacktestConfiguration) -> None:
        """Supprime une configuration des index"""
        try:
            self._configs_by_name[config.name.lower()].remove(config.id)
        except ValueError:
            pass

    def _add_result_to_indexes(self, result: BacktestResult) -> None:
        """Ajoute un résultat aux index"""
        self._results_by_config[result.configuration_id].append(result.id)
        self._results_by_status[result.status].append(result.id)

        # Indexer par stratégies
        config = self._configurations.get(result.configuration_id)
        if config:
            for strategy_id in config.strategy_ids:
                self._results_by_strategy[strategy_id].append(result.id)

    def _remove_result_from_indexes(self, result: BacktestResult) -> None:
        """Supprime un résultat des index"""
        try:
            self._results_by_config[result.configuration_id].remove(result.id)
        except ValueError:
            pass

        try:
            self._results_by_status[result.status].remove(result.id)
        except ValueError:
            pass

        # Supprimer des index de stratégies
        config = self._configurations.get(result.configuration_id)
        if config:
            for strategy_id in config.strategy_ids:
                try:
                    self._results_by_strategy[strategy_id].remove(result.id)
                except ValueError:
                    pass

    def _matches_filters(self, result: BacktestResult, filters: Dict[str, Any]) -> bool:
        """Vérifie si un résultat correspond aux filtres"""
        for key, value in filters.items():
            if key == "status" and result.status.value != value:
                return False
            elif key == "min_sharpe" and (not result.metrics or result.metrics.sharpe_ratio < Decimal(str(value))):
                return False
            # Ajouter d'autres filtres selon les besoins
        return True

    def _calculate_config_similarity(
        self,
        config1: BacktestConfiguration,
        config2: BacktestConfiguration
    ) -> float:
        """Calcule la similarité entre deux configurations"""
        score = 0.0
        total_checks = 0

        # Comparer les stratégies
        if set(config1.strategy_ids) == set(config2.strategy_ids):
            score += 0.3
        total_checks += 0.3

        # Comparer les paramètres financiers
        if abs(config1.transaction_cost - config2.transaction_cost) < Decimal("0.001"):
            score += 0.1
        if abs(config1.slippage - config2.slippage) < Decimal("0.001"):
            score += 0.1
        if config1.max_position_size == config2.max_position_size:
            score += 0.1
        total_checks += 0.3

        # Comparer les types et fréquences
        if config1.backtest_type == config2.backtest_type:
            score += 0.2
        if config1.rebalance_frequency == config2.rebalance_frequency:
            score += 0.2
        total_checks += 0.4

        return score / total_checks if total_checks > 0 else 0.0

    async def clear_all(self) -> None:
        """Supprime toutes les données (pour les tests)"""
        self._configurations.clear()
        self._results.clear()
        self._archived_results.clear()
        self._configs_by_name.clear()
        self._results_by_config.clear()
        self._results_by_status.clear()
        self._results_by_strategy.clear()