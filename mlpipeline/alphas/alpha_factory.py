"""
Alpha Factory - Interface Unifiée
Point d'entrée principal pour tous les types d'alphas:
- Alphas traditionnels (DMN, Mean Reversion, Funding)
- Alphas générés par RL
- Combinaisons synergiques
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import json

# Alphas traditionnels
from mlpipeline.alphas.dmn_model import DMNPredictor
from mlpipeline.alphas.mean_reversion import AdaptiveMeanReversion
from mlpipeline.alphas.funding_strategy import AdvancedFundingStrategy

# Alphas RL
from mlpipeline.alphas.rl_alpha_pipeline import RLAlphaPipeline
from mlpipeline.alphas.rl_alpha_generator import AlphaFormula
from mlpipeline.alphas.synergistic_combiner import AlphaCombination

logger = logging.getLogger(__name__)

class AlphaFactory:
    """
    Factory centralisée pour la génération et gestion de tous les alphas
    """

    def __init__(self, market_data: pd.DataFrame, symbol: str = "BTCUSDT"):
        self.data = market_data
        self.symbol = symbol

        # Alphas traditionnels
        self.traditional_alphas = {
            'dmn_lstm': DMNPredictor(symbol),
            'mean_reversion': AdaptiveMeanReversion(),
            'funding_strategy': AdvancedFundingStrategy(symbol) if symbol else None
        }

        # Pipeline RL
        self.rl_pipeline = RLAlphaPipeline(market_data, symbol)

        # Stockage des résultats
        self.alpha_results = {}
        self.rl_discoveries = {}
        self.performance_metrics = {}

        logger.info(f"🏭 Alpha Factory initialisée pour {symbol}")

    async def generate_all_alphas(self, include_rl: bool = True) -> Dict:
        """
        Génère tous les types d'alphas disponibles
        """
        logger.info("🚀 Génération complète de tous les alphas")

        results = {
            "symbol": self.symbol,
            "traditional_alphas": {},
            "rl_discoveries": {},
            "performance_summary": {},
            "recommendations": []
        }

        try:
            # Phase 1: Alphas traditionnels
            logger.info("📊 Phase 1: Alphas traditionnels")
            traditional_results = await self._generate_traditional_alphas()
            results["traditional_alphas"] = traditional_results

            # Phase 2: Découverte RL (optionnel)
            if include_rl:
                logger.info("🤖 Phase 2: Découverte RL d'alphas")
                rl_results = await self._generate_rl_alphas()
                results["rl_discoveries"] = rl_results

            # Phase 3: Évaluation comparative
            logger.info("⚖️ Phase 3: Évaluation comparative")
            comparison = await self._compare_alpha_performance()
            results["performance_summary"] = comparison

            # Phase 4: Recommandations
            logger.info("💡 Phase 4: Recommandations")
            recommendations = self._generate_recommendations(traditional_results, rl_results if include_rl else {})
            results["recommendations"] = recommendations

            # Sauvegarde
            await self._save_factory_results(results)

            logger.info("✅ Génération complète terminée")
            return results

        except Exception as e:
            logger.error(f"❌ Erreur génération alphas: {e}")
            return results

    async def _generate_traditional_alphas(self) -> Dict:
        """Génère les signaux des alphas traditionnels"""
        results = {}

        for alpha_name, alpha_instance in self.traditional_alphas.items():
            if alpha_instance is None:
                continue

            try:
                logger.info(f"   🎯 {alpha_name}")

                if alpha_name == 'dmn_lstm':
                    signals = await alpha_instance.predict(self.data)
                    metrics = self._calculate_signal_metrics(signals, alpha_name)

                elif alpha_name == 'mean_reversion':
                    signals_df = await alpha_instance.generate_signals(
                        self.data, None, None
                    )
                    signals = signals_df['signal_raw'].fillna(0) if not signals_df.empty else pd.Series(0, index=self.data.index)
                    metrics = self._calculate_signal_metrics(signals, alpha_name)

                elif alpha_name == 'funding_strategy':
                    try:
                        signals_df = await alpha_instance.generate_signals(self.data)
                        signals = signals_df['signal_raw'].fillna(0) if not signals_df.empty else pd.Series(0, index=self.data.index)
                        metrics = self._calculate_signal_metrics(signals, alpha_name)
                    except Exception:
                        # Funding strategy peut échouer sur certaines données
                        signals = pd.Series(0, index=self.data.index)
                        metrics = {"status": "error", "ic": 0.0, "active_signals": 0}

                results[alpha_name] = {
                    "signals_generated": len(signals) if hasattr(signals, '__len__') else 0,
                    "metrics": metrics,
                    "status": "success" if metrics.get("ic", 0) != 0 else "low_performance"
                }

                logger.info(f"      ✅ IC: {metrics.get('ic', 0):.4f}")

            except Exception as e:
                logger.warning(f"      ❌ Erreur {alpha_name}: {e}")
                results[alpha_name] = {
                    "signals_generated": 0,
                    "metrics": {"status": "error", "ic": 0.0},
                    "status": "error"
                }

        return results

    async def _generate_rl_alphas(self) -> Dict:
        """Lance la découverte RL d'alphas"""
        try:
            # Découverte rapide pour commencer
            rl_results = await self.rl_pipeline.quick_discovery(15)

            # Enrichir avec des métriques détaillées
            enriched_results = {
                **rl_results,
                "best_alphas": [],
                "best_combination": None
            }

            # Meilleurs alphas individuels
            best_alphas = self.rl_pipeline.get_best_alphas(5)
            for alpha in best_alphas:
                enriched_results["best_alphas"].append({
                    "formula": alpha.formula,
                    "ic": alpha.ic,
                    "rank_ic": alpha.rank_ic,
                    "complexity": alpha.complexity
                })

            # Meilleure combinaison
            best_combo = self.rl_pipeline.get_best_combination()
            if best_combo:
                enriched_results["best_combination"] = {
                    "total_score": best_combo.total_score,
                    "combined_ic": best_combo.combined_ic,
                    "num_alphas": len(best_combo.alphas),
                    "diversification": best_combo.diversification_score
                }

            return enriched_results

        except Exception as e:
            logger.error(f"❌ Erreur découverte RL: {e}")
            return {
                "total_alphas_generated": 0,
                "valid_alphas": 0,
                "best_combinations": 0,
                "status": "error"
            }

    def _calculate_signal_metrics(self, signals: Union[pd.Series, np.ndarray], alpha_name: str) -> Dict:
        """Calcule les métriques pour un signal d'alpha"""
        try:
            if isinstance(signals, np.ndarray):
                signals = pd.Series(signals, index=self.data.index[:len(signals)])

            # Métriques de base
            metrics = {
                "signal_count": len(signals),
                "active_signals": (np.abs(signals) > 0.01).sum(),
                "signal_strength": np.abs(signals).mean(),
                "signal_std": signals.std(),
                "ic": 0.0,
                "rank_ic": 0.0
            }

            # IC si nous avons des returns
            if 'ret' in self.data.columns and len(signals) > 1:
                future_returns = self.data['ret'].shift(-1).fillna(0)
                aligned_signals = signals.reindex(future_returns.index).fillna(0)

                if len(aligned_signals) > 0 and aligned_signals.std() > 1e-8:
                    ic = aligned_signals.corr(future_returns)
                    rank_ic = aligned_signals.corr(future_returns, method='spearman')

                    metrics["ic"] = ic if not np.isnan(ic) else 0.0
                    metrics["rank_ic"] = rank_ic if not np.isnan(rank_ic) else 0.0

            return metrics

        except Exception as e:
            logger.debug(f"Erreur calcul métriques {alpha_name}: {e}")
            return {"status": "error", "ic": 0.0, "active_signals": 0}

    async def _compare_alpha_performance(self) -> Dict:
        """Compare les performances de tous les alphas"""
        comparison = {
            "traditional_ranking": [],
            "rl_vs_traditional": {},
            "overall_best": None,
            "performance_summary": {}
        }

        try:
            # Ranking des alphas traditionnels
            traditional_ics = {}
            for alpha_name, results in self.alpha_results.get("traditional_alphas", {}).items():
                ic = results.get("metrics", {}).get("ic", 0.0)
                traditional_ics[alpha_name] = ic

            # Trier par IC
            sorted_traditional = sorted(traditional_ics.items(), key=lambda x: x[1], reverse=True)
            comparison["traditional_ranking"] = [
                {"alpha": name, "ic": ic} for name, ic in sorted_traditional
            ]

            # Comparaison RL vs Traditional
            rl_results = self.rl_discoveries
            if rl_results.get("best_alphas"):
                best_rl_ic = max(alpha["ic"] for alpha in rl_results["best_alphas"])
                best_traditional_ic = max(traditional_ics.values()) if traditional_ics else 0.0

                comparison["rl_vs_traditional"] = {
                    "best_rl_ic": best_rl_ic,
                    "best_traditional_ic": best_traditional_ic,
                    "rl_advantage": best_rl_ic - best_traditional_ic,
                    "winner": "RL" if best_rl_ic > best_traditional_ic else "Traditional"
                }

            # Meilleur overall
            all_performers = []

            # Ajouter traditionnels
            for name, ic in traditional_ics.items():
                all_performers.append({"type": "traditional", "name": name, "ic": ic})

            # Ajouter RL
            if rl_results.get("best_alphas"):
                for i, alpha in enumerate(rl_results["best_alphas"]):
                    all_performers.append({
                        "type": "rl_generated",
                        "name": f"RL_Alpha_{i+1}",
                        "ic": alpha["ic"]
                    })

            if all_performers:
                best_overall = max(all_performers, key=lambda x: x["ic"])
                comparison["overall_best"] = best_overall

            # Summary
            comparison["performance_summary"] = {
                "total_alphas_evaluated": len(all_performers),
                "traditional_count": len(traditional_ics),
                "rl_generated_count": len(rl_results.get("best_alphas", [])),
                "best_ic_overall": best_overall["ic"] if all_performers else 0.0
            }

        except Exception as e:
            logger.error(f"❌ Erreur comparaison: {e}")

        return comparison

    def _generate_recommendations(self, traditional_results: Dict, rl_results: Dict) -> List[str]:
        """Génère des recommandations basées sur les résultats"""
        recommendations = []

        try:
            # Analyser les performances traditionnelles
            traditional_ics = {}
            for alpha_name, results in traditional_results.items():
                ic = results.get("metrics", {}).get("ic", 0.0)
                traditional_ics[alpha_name] = ic

            best_traditional = max(traditional_ics.values()) if traditional_ics else 0.0

            # Analyser les performances RL
            best_rl = 0.0
            if rl_results.get("best_alphas"):
                best_rl = max(alpha["ic"] for alpha in rl_results["best_alphas"])

            # Recommandations basées sur les performances
            if best_traditional > 0.05:
                recommendations.append("✅ Alphas traditionnels performants - utilisation recommandée")
            elif best_traditional > 0.02:
                recommendations.append("⚠️ Alphas traditionnels moyens - optimisation recommandée")
            else:
                recommendations.append("❌ Alphas traditionnels faibles - révision nécessaire")

            if best_rl > best_traditional:
                recommendations.append("🚀 Alphas RL supérieurs - intégration prioritaire")
            elif best_rl > 0.03:
                recommendations.append("🎯 Alphas RL prometteurs - tests supplémentaires recommandés")

            # Recommandations spécifiques
            if rl_results.get("best_combination"):
                combo_score = rl_results["best_combination"]["total_score"]
                if combo_score > 0.1:
                    recommendations.append("🔬 Combinaison synergique excellente - déploiement recommandé")

            # Recommandations opérationnelles
            if len(traditional_ics) < 3:
                recommendations.append("📈 Diversifier avec plus d'alphas traditionnels")

            if rl_results.get("valid_alphas", 0) < 5:
                recommendations.append("🤖 Augmenter la génération RL d'alphas")

        except Exception as e:
            logger.error(f"❌ Erreur recommandations: {e}")
            recommendations.append("⚠️ Erreur génération recommandations")

        return recommendations

    async def _save_factory_results(self, results: Dict):
        """Sauvegarde les résultats de l'Alpha Factory"""
        try:
            output_dir = Path("data/artifacts/alpha_factory")
            output_dir.mkdir(parents=True, exist_ok=True)

            results_file = output_dir / f"alpha_factory_results_{self.symbol}.json"

            # Convertir les objets non-JSON en données sérialisables
            serializable_results = self._make_json_serializable(results)

            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"💾 Résultats Alpha Factory sauvegardés: {results_file}")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde Factory: {e}")

    def _make_json_serializable(self, obj):
        """Convertit les objets en format JSON sérialisable"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

    async def quick_alpha_scan(self) -> Dict:
        """Scan rapide de tous les alphas pour évaluation rapide"""
        logger.info("⚡ Scan rapide des alphas")

        # Version allégée de la génération complète
        results = await self.generate_all_alphas(include_rl=False)

        # Ajouter un petit échantillon RL
        try:
            rl_sample = await self.rl_pipeline.quick_discovery(5)
            results["rl_sample"] = rl_sample
        except Exception as e:
            logger.warning(f"⚠️ Erreur échantillon RL: {e}")
            results["rl_sample"] = {"status": "error"}

        return results

async def test_alpha_factory():
    """Test de l'Alpha Factory complète"""
    print("🧪 Test de l'Alpha Factory")

    # Données de test plus conséquentes
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')

    data = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.cumsum(np.random.randn(200) * 100),
        'high': 50100 + np.cumsum(np.random.randn(200) * 100),
        'low': 49900 + np.cumsum(np.random.randn(200) * 100),
        'close': 50000 + np.cumsum(np.random.randn(200) * 100),
        'volume': np.random.randint(1000, 10000, 200),
        'ret': np.random.randn(200) * 0.02
    }, index=dates)

    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3

    # Créer l'Alpha Factory
    factory = AlphaFactory(data, "TESTUSDT")

    # Test scan rapide
    print("⚡ Test scan rapide...")
    quick_results = await factory.quick_alpha_scan()

    print(f"✅ Scan terminé:")
    print(f"   Alphas traditionnels: {len(quick_results.get('traditional_alphas', {}))}")
    print(f"   Échantillon RL: {quick_results.get('rl_sample', {}).get('valid_alphas', 0)}")

    # Recommandations
    recommendations = quick_results.get('recommendations', [])
    if recommendations:
        print(f"💡 Recommandations:")
        for rec in recommendations[:3]:  # Top 3
            print(f"   {rec}")

    return factory

if __name__ == "__main__":
    asyncio.run(test_alpha_factory())