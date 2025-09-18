"""
Pipeline Intégré RL Alpha Generator
Interface entre le générateur RL et le framework existant
"""

import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path
import json

from mlpipeline.alphas.rl_alpha_generator import RLAlphaGenerator, AlphaFormula
from mlpipeline.alphas.synergistic_combiner import SynergisticAlphaEngine, AlphaCombination

logger = logging.getLogger(__name__)

class RLAlphaPipeline:
    """Pipeline principal pour la génération RL d'alphas"""

    def __init__(self, market_data: pd.DataFrame, symbol: str = "BTCUSDT"):
        self.data = market_data
        self.symbol = symbol

        # Composants principaux
        self.rl_generator = RLAlphaGenerator(market_data)
        self.synergistic_engine = SynergisticAlphaEngine(market_data)

        # Stockage des résultats
        self.generated_alphas: List[AlphaFormula] = []
        self.best_combinations: List[AlphaCombination] = []

        # Configuration
        self.config = {
            "batch_size": 20,           # Nombre d'alphas par batch
            "num_batches": 3,           # Nombre de batches
            "max_combinations": 5,      # Combinaisons synergiques max
            "min_ic_threshold": 0.02,   # IC minimum pour garder un alpha
            "max_alphas_total": 100     # Limite totale d'alphas
        }

        logger.info(f"✅ RL Alpha Pipeline initialisé pour {symbol}")

    async def run_full_discovery(self) -> Dict:
        """
        Lance le processus complet de découverte d'alphas
        """
        logger.info("🚀 Début découverte automatique d'alphas")

        results = {
            "symbol": self.symbol,
            "total_alphas_generated": 0,
            "valid_alphas": 0,
            "best_combinations": 0,
            "discovery_stats": {},
            "execution_time": 0
        }

        start_time = pd.Timestamp.now()

        try:
            # Phase 1: Génération d'alphas par batches
            logger.info("📡 Phase 1: Génération d'alphas RL")
            await self._generate_alpha_batches()

            # Phase 2: Filtrage et sélection
            logger.info("🔍 Phase 2: Filtrage et sélection")
            valid_alphas = self._filter_and_select_alphas()

            # Phase 3: Combinaisons synergiques
            logger.info("🔬 Phase 3: Combinaisons synergiques")
            if len(valid_alphas) >= 2:
                combinations = self._generate_synergistic_combinations(valid_alphas)
                self.best_combinations = combinations

            # Phase 4: Évaluation finale
            logger.info("📊 Phase 4: Évaluation finale")
            final_evaluation = await self._evaluate_discoveries()

            # Mise à jour des résultats
            end_time = pd.Timestamp.now()
            execution_time = (end_time - start_time).total_seconds()

            results.update({
                "total_alphas_generated": len(self.generated_alphas),
                "valid_alphas": len(valid_alphas),
                "best_combinations": len(self.best_combinations),
                "discovery_stats": final_evaluation,
                "execution_time": execution_time
            })

            # Sauvegarde
            await self._save_discoveries()

            logger.info(f"✅ Découverte terminée en {execution_time:.1f}s")
            logger.info(f"📊 {results['valid_alphas']}/{results['total_alphas_generated']} alphas valides")

            return results

        except Exception as e:
            logger.error(f"❌ Erreur découverte d'alphas: {e}")
            return results

    async def _generate_alpha_batches(self):
        """Génère des alphas par batches"""
        total_generated = 0

        for batch_num in range(self.config["num_batches"]):
            logger.info(f"🎯 Batch {batch_num + 1}/{self.config['num_batches']}")

            try:
                # Générer un batch d'alphas
                batch_alphas = self.rl_generator.generate_alpha_batch(
                    self.config["batch_size"]
                )

                self.generated_alphas.extend(batch_alphas)
                total_generated += len(batch_alphas)

                logger.info(f"   📈 {len(batch_alphas)} alphas générés dans ce batch")

                # Limitation du nombre total
                if total_generated >= self.config["max_alphas_total"]:
                    logger.info("🛑 Limite maximale d'alphas atteinte")
                    break

            except Exception as e:
                logger.warning(f"⚠️ Erreur batch {batch_num + 1}: {e}")

        logger.info(f"🎉 Total généré: {len(self.generated_alphas)} alphas")

    def _filter_and_select_alphas(self) -> List[AlphaFormula]:
        """Filtre et sélectionne les meilleurs alphas"""
        if not self.generated_alphas:
            return []

        # Filtrer par IC minimum
        valid_alphas = [
            alpha for alpha in self.generated_alphas
            if alpha.ic >= self.config["min_ic_threshold"]
        ]

        logger.info(f"🔍 {len(valid_alphas)}/{len(self.generated_alphas)} alphas passent le filtre IC")

        if not valid_alphas:
            return []

        # Trier par IC décroissant
        valid_alphas.sort(key=lambda x: x.ic, reverse=True)

        # Analyse de la diversité
        diversity_stats = self._analyze_alpha_diversity(valid_alphas)
        logger.info(f"📊 Diversité: {diversity_stats}")

        return valid_alphas

    def _generate_synergistic_combinations(self, alphas: List[AlphaFormula]) -> List[AlphaCombination]:
        """Génère des combinaisons synergiques"""
        try:
            combinations = self.synergistic_engine.generate_synergistic_alphas(
                alphas,
                self.config["max_combinations"]
            )

            logger.info(f"🔬 {len(combinations)} combinaisons synergiques créées")

            return combinations

        except Exception as e:
            logger.error(f"❌ Erreur combinaisons synergiques: {e}")
            return []

    async def _evaluate_discoveries(self) -> Dict:
        """Évalue les découvertes finales"""
        evaluation = {
            "generation_stats": self.rl_generator.get_generation_stats(),
            "combination_stats": self.synergistic_engine.get_combination_stats(),
            "best_alpha_ic": 0.0,
            "best_combination_score": 0.0,
            "alpha_types_discovered": []
        }

        # Meilleur alpha individuel
        if self.generated_alphas:
            best_alpha = max(self.generated_alphas, key=lambda x: x.ic)
            evaluation["best_alpha_ic"] = best_alpha.ic

        # Meilleure combinaison
        if self.best_combinations:
            best_combination = max(self.best_combinations, key=lambda x: x.total_score)
            evaluation["best_combination_score"] = best_combination.total_score

        # Types d'alphas découverts
        operator_usage = {}
        for alpha in self.generated_alphas:
            for op in alpha.operators:
                operator_usage[op] = operator_usage.get(op, 0) + 1

        evaluation["alpha_types_discovered"] = list(operator_usage.keys())
        evaluation["operator_usage"] = operator_usage

        return evaluation

    def _analyze_alpha_diversity(self, alphas: List[AlphaFormula]) -> Dict:
        """Analyse la diversité des alphas générés"""
        if not alphas:
            return {"diversity_score": 0.0}

        # Comptage des opérateurs utilisés
        operator_counts = {}
        for alpha in alphas:
            for op in alpha.operators:
                operator_counts[op] = operator_counts.get(op, 0) + 1

        # Score de diversité basé sur l'entropie
        total_operators = sum(operator_counts.values())
        if total_operators == 0:
            return {"diversity_score": 0.0}

        entropy = 0
        for count in operator_counts.values():
            prob = count / total_operators
            if prob > 0:
                entropy -= prob * np.log2(prob)

        # Normaliser par le nombre d'opérateurs possibles
        max_entropy = np.log2(len(self.rl_generator.search_space.operators))
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0

        return {
            "diversity_score": diversity_score,
            "unique_operators": len(operator_counts),
            "total_operators_used": total_operators,
            "operator_distribution": operator_counts
        }

    async def _save_discoveries(self):
        """Sauvegarde les découvertes"""
        try:
            # Créer le dossier de sortie
            output_dir = Path("data/artifacts/rl_alphas")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Sauvegarder les alphas générés
            alphas_data = []
            for alpha in self.generated_alphas:
                alphas_data.append({
                    "formula": alpha.formula,
                    "operators": alpha.operators,
                    "ic": alpha.ic,
                    "rank_ic": alpha.rank_ic,
                    "complexity": alpha.complexity,
                    "generation": alpha.generation
                })

            alphas_file = output_dir / f"generated_alphas_{self.symbol}.json"
            with open(alphas_file, 'w') as f:
                json.dump(alphas_data, f, indent=2)

            # Sauvegarder les combinaisons
            combinations_data = []
            for combo in self.best_combinations:
                combinations_data.append({
                    "num_alphas": len(combo.alphas),
                    "weights": combo.weights.tolist(),
                    "combined_ic": combo.combined_ic,
                    "total_score": combo.total_score,
                    "diversification_score": combo.diversification_score
                })

            combinations_file = output_dir / f"synergistic_combinations_{self.symbol}.json"
            with open(combinations_file, 'w') as f:
                json.dump(combinations_data, f, indent=2)

            logger.info(f"💾 Découvertes sauvegardées dans {output_dir}")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")

    def get_best_alphas(self, top_k: int = 5) -> List[AlphaFormula]:
        """Retourne les meilleurs alphas découverts"""
        if not self.generated_alphas:
            return []

        return sorted(self.generated_alphas, key=lambda x: x.ic, reverse=True)[:top_k]

    def get_best_combination(self) -> Optional[AlphaCombination]:
        """Retourne la meilleure combinaison synergique"""
        if not self.best_combinations:
            return None

        return max(self.best_combinations, key=lambda x: x.total_score)

    async def quick_discovery(self, num_alphas: int = 10) -> Dict:
        """Lance une découverte rapide avec moins d'alphas"""
        logger.info(f"⚡ Découverte rapide de {num_alphas} alphas")

        # Configuration temporaire pour découverte rapide
        original_config = self.config.copy()
        self.config.update({
            "batch_size": max(5, num_alphas // 2),
            "num_batches": 2,
            "max_combinations": 3
        })

        try:
            results = await self.run_full_discovery()
            return results
        finally:
            # Restaurer la configuration
            self.config = original_config

async def test_rl_pipeline():
    """Test du pipeline RL complet"""
    print("🧪 Test du pipeline RL complet")

    # Données de test
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')  # Plus petit pour test

    data = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.cumsum(np.random.randn(100) * 100),
        'high': 50100 + np.cumsum(np.random.randn(100) * 100),
        'low': 49900 + np.cumsum(np.random.randn(100) * 100),
        'close': 50000 + np.cumsum(np.random.randn(100) * 100),
        'volume': np.random.randint(1000, 10000, 100),
        'ret': np.random.randn(100) * 0.02
    }, index=dates)

    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3

    # Créer le pipeline
    pipeline = RLAlphaPipeline(data, "TESTUSDT")

    # Test découverte rapide
    print("⚡ Test découverte rapide...")
    results = await pipeline.quick_discovery(8)

    print(f"✅ Résultats:")
    print(f"   Alphas générés: {results['total_alphas_generated']}")
    print(f"   Alphas valides: {results['valid_alphas']}")
    print(f"   Combinaisons: {results['best_combinations']}")
    print(f"   Temps: {results['execution_time']:.1f}s")

    # Meilleurs résultats
    best_alphas = pipeline.get_best_alphas(3)
    if best_alphas:
        print(f"🏆 Top 3 alphas:")
        for i, alpha in enumerate(best_alphas):
            print(f"   {i+1}. {alpha.formula} (IC: {alpha.ic:.4f})")

    best_combo = pipeline.get_best_combination()
    if best_combo:
        print(f"🔬 Meilleure combinaison:")
        print(f"   Score: {best_combo.total_score:.4f}")
        print(f"   IC: {best_combo.combined_ic:.4f}")
        print(f"   Alphas: {len(best_combo.alphas)}")

    return pipeline

if __name__ == "__main__":
    asyncio.run(test_rl_pipeline())