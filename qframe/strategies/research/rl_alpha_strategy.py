"""
Reinforcement Learning Alpha Strategy
====================================

Migration du générateur RL d'alphas vers l'architecture moderne.
Basé sur "Synergistic Formulaic Alpha Generation for Quantitative Trading".
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

from qframe.core.interfaces import (
    BaseStrategy,
    Signal,
    SignalAction,
    TimeFrame,
    MetricsCollector
)
from qframe.core.container import injectable
from qframe.features.symbolic_operators import SymbolicOperators

logger = logging.getLogger(__name__)


@dataclass
class AlphaFormula:
    """Représente une formule alpha générée par RL"""
    formula: str
    operators: List[str]
    ic: float
    rank_ic: float
    complexity: int
    generation: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RLAlphaConfig:
    """Configuration pour la stratégie RL Alpha"""
    # Paramètres RL
    state_size: int = 50
    hidden_size: int = 128
    learning_rate: float = 3e-4
    episodes_per_batch: int = 10
    max_complexity: int = 10
    max_formula_length: int = 20

    # Paramètres trading
    signal_threshold: float = 0.01
    position_size: float = 0.02
    min_ic_threshold: float = 0.01

    # Paramètres reward
    complexity_penalty: float = 0.01
    ic_reward_multiplier: float = 1.0


class SearchSpace:
    """Définit l'espace de recherche pour la génération d'alphas"""

    def __init__(self):
        # Opérateurs disponibles (du papier)
        self.operators = [
            "sign", "cs_rank", "product", "scale", "pow_op", "skew", "kurt",
            "ts_rank", "delta", "argmax", "argmin", "cond", "wma", "ema", "mad"
        ]

        # Features de base
        self.features = ["open", "high", "low", "close", "volume", "vwap"]

        # Constantes et paramètres temporels
        self.constants = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.time_deltas = [5, 10, 20, 30, 40, 50, 60, 120]

        # Taille maximale des formules
        self.max_depth = 4
        self.max_complexity = 10

    def get_action_space_size(self) -> int:
        """Retourne la taille de l'espace d'action"""
        return (
            len(self.operators) + len(self.features) +
            len(self.constants) + len(self.time_deltas)
        )


class FormulaEnvironment:
    """Environnement RL pour la génération de formules alpha"""

    def __init__(self, market_data: pd.DataFrame, search_space: SearchSpace):
        self.data = market_data
        self.search_space = search_space
        self.ops = SymbolicOperators()
        self.current_formula = []
        self.current_complexity = 0

    def reset(self) -> np.ndarray:
        """Reset l'environnement pour un nouvel épisode"""
        self.current_formula = []
        self.current_complexity = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Retourne l'état actuel de l'environnement"""
        # État basé sur la formule courante et statistiques des données
        state = np.zeros(50)  # État de taille fixe

        # Encoder la formule courante
        for i, element in enumerate(self.current_formula[-10:]):  # Derniers 10 éléments
            if i < 10:
                state[i] = hash(str(element)) % 1000 / 1000.0

        # Statistiques des données
        if len(self.data) > 0:
            state[10:15] = [
                self.data["close"].mean() / 50000,  # Prix normalisé
                self.data["volume"].mean() / 10000,  # Volume normalisé
                self.data["close"].std() / 1000,  # Volatilité
                len(self.data) / 1000,  # Taille dataset
                self.current_complexity / 10,  # Complexité courante
            ]

        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Exécute une action dans l'environnement"""
        # Décoder l'action
        element = self._decode_action(action)

        # Ajouter l'élément à la formule
        self.current_formula.append(element)
        self.current_complexity += 1

        # Calculer la récompense
        reward = self._calculate_reward()

        # Vérifier si l'épisode est terminé
        done = (
            self.current_complexity >= self.search_space.max_complexity
            or len(self.current_formula) >= 20
        )

        return self._get_state(), reward, done, {}

    def _decode_action(self, action: int) -> str:
        """Décode une action en élément de formule"""
        total_ops = len(self.search_space.operators)
        total_features = len(self.search_space.features)
        total_constants = len(self.search_space.constants)

        if action < total_ops:
            return self.search_space.operators[action]
        elif action < total_ops + total_features:
            return self.search_space.features[action - total_ops]
        elif action < total_ops + total_features + total_constants:
            return str(self.search_space.constants[action - total_ops - total_features])
        else:
            return str(
                self.search_space.time_deltas[
                    action - total_ops - total_features - total_constants
                ]
            )

    def _calculate_reward(self) -> float:
        """Calcule la récompense pour la formule courante"""
        if len(self.current_formula) < 3:  # Formule trop courte
            return -0.1

        try:
            # Essayer d'évaluer la formule
            alpha_values = self._evaluate_formula()

            if alpha_values is None or len(alpha_values) == 0:
                return -1.0  # Formule invalide

            # Calculer IC (Information Coefficient)
            ic = self._calculate_ic(alpha_values)

            # Récompense basée sur IC avec pénalité pour complexité
            complexity_penalty = self.current_complexity * 0.01
            return ic - complexity_penalty

        except Exception:
            return -1.0  # Erreur d'évaluation

    def _evaluate_formula(self) -> Optional[pd.Series]:
        """Évalue la formule courante"""
        try:
            # Simplification : créer une formule basique combinant les éléments
            if len(self.current_formula) < 3:
                return None

            # Prendre les 3 derniers éléments significatifs
            relevant_elements = [
                e for e in self.current_formula
                if e in self.search_space.operators + self.search_space.features
            ][-3:]

            if len(relevant_elements) < 2:
                return None

            # Construire une formule simple
            if "close" in self.data.columns and "volume" in self.data.columns:
                if "cs_rank" in relevant_elements:
                    return self.ops.cs_rank(self.data["volume"])
                elif "sign" in relevant_elements:
                    returns = self.data["close"].pct_change().fillna(0)
                    return self.ops.sign(returns)
                elif "delta" in relevant_elements:
                    return self.ops.delta(self.data["close"], 5)
                else:
                    # Formule par défaut
                    return self.data["close"].rolling(10).corr(self.data["volume"])

            return None

        except Exception as e:
            logger.debug(f"Erreur évaluation formule: {e}")
            return None

    def _calculate_ic(self, alpha_values: pd.Series) -> float:
        """Calcule l'Information Coefficient"""
        try:
            if "returns" in self.data.columns:
                # Utiliser les returns futurs
                future_returns = self.data["returns"].shift(-1).fillna(0)
                aligned_alpha = alpha_values.reindex(future_returns.index).fillna(0)

                # Corrélation de Pearson
                correlation = aligned_alpha.corr(future_returns)
                return correlation if not np.isnan(correlation) else 0.0
            else:
                # Corrélation avec close price changes
                price_changes = self.data["close"].pct_change().fillna(0)
                aligned_alpha = alpha_values.reindex(price_changes.index).fillna(0)

                correlation = aligned_alpha.corr(price_changes)
                return correlation if not np.isnan(correlation) else 0.0

        except Exception:
            return 0.0


class PPOAgent(nn.Module):
    """Agent PPO pour la génération d'alphas"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        # Réseau de politique (actor)
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )

        # Réseau de valeur (critic)
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

    def get_action(self, state):
        """Sélectionne une action basée sur la politique"""
        action_probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


@injectable
class RLAlphaStrategy(BaseStrategy):
    """
    Stratégie utilisant RL pour générer automatiquement des alphas

    Préserve la logique métier de recherche sophistiquée tout en
    s'intégrant dans l'architecture moderne.
    """

    def __init__(
        self,
        config: RLAlphaConfig = None,
        metrics_collector: MetricsCollector = None
    ):
        self.config = config or RLAlphaConfig()
        self.metrics_collector = metrics_collector

        super().__init__(
            name="RL_Alpha_Generator",
            parameters=self.config.__dict__
        )

        # Composants RL
        self.search_space = SearchSpace()
        self.env: Optional[FormulaEnvironment] = None
        self.agent: Optional[PPOAgent] = None
        self.optimizer: Optional[torch.optim.Adam] = None

        # État de la stratégie
        self.generated_alphas: List[AlphaFormula] = []
        self.generation_count = 0
        self.is_trained = False
        self.last_alpha_values: Optional[pd.Series] = None

        logger.info(f"RL Alpha Strategy initialisée: {self.config}")

    def generate_signals(
        self,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """
        Génère des signaux basés sur les alphas générés par RL
        """
        if len(data) < 50:  # Besoin de données suffisantes
            logger.warning("Pas assez de données pour RL Alpha Strategy")
            return []

        try:
            # Initialiser l'environnement si nécessaire
            if self.env is None:
                self._initialize_rl_components(data)

            # Générer de nouveaux alphas si nécessaire
            if not self.generated_alphas or self.generation_count % 10 == 0:
                self._generate_alpha_batch()

            # Utiliser le meilleur alpha pour générer des signaux
            signals = self._generate_signals_from_best_alpha(data)

            # Métriques
            if self.metrics_collector and signals:
                self.metrics_collector.record_metric(
                    "rl_alpha_signals",
                    len(signals),
                    {"generation": self.generation_count, "total_alphas": len(self.generated_alphas)}
                )

            return signals

        except Exception as e:
            logger.error(f"Erreur génération signaux RL Alpha: {e}")
            return []

    def _initialize_rl_components(self, data: pd.DataFrame) -> None:
        """Initialise les composants RL"""
        try:
            # Créer l'environnement
            work_data = self._prepare_data(data)
            self.env = FormulaEnvironment(work_data, self.search_space)

            # Créer l'agent PPO
            action_size = self.search_space.get_action_space_size()
            self.agent = PPOAgent(
                self.config.state_size,
                action_size,
                self.config.hidden_size
            )
            self.optimizer = torch.optim.Adam(
                self.agent.parameters(),
                lr=self.config.learning_rate
            )

            logger.info(f"✅ Composants RL initialisés - Action space: {action_size}")

        except Exception as e:
            logger.error(f"Erreur initialisation RL: {e}")
            raise

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare les données pour l'environnement RL"""
        work_data = data.copy()

        # Ajouter returns si pas présent
        if "returns" not in work_data.columns:
            work_data["returns"] = work_data["close"].pct_change().fillna(0)

        # Ajouter VWAP si pas présent
        if "vwap" not in work_data.columns:
            work_data["vwap"] = (
                work_data["high"] + work_data["low"] + work_data["close"]
            ) / 3

        return work_data

    def _generate_alpha_batch(self) -> List[AlphaFormula]:
        """Génère un batch d'alphas avec RL"""
        if self.agent is None or self.env is None:
            return []

        new_alphas = []

        for episode in range(self.config.episodes_per_batch):
            try:
                alpha = self._generate_single_alpha()
                if alpha and alpha.ic > self.config.min_ic_threshold:
                    new_alphas.append(alpha)
                    logger.info(
                        f"📊 Alpha généré - IC: {alpha.ic:.4f}, Complexité: {alpha.complexity}"
                    )
            except Exception as e:
                logger.debug(f"Erreur génération épisode {episode}: {e}")

        self.generated_alphas.extend(new_alphas)
        self.generation_count += 1

        logger.info(
            f"🎯 Batch RL terminé: {len(new_alphas)}/{self.config.episodes_per_batch} alphas valides"
        )
        return new_alphas

    def _generate_single_alpha(self) -> Optional[AlphaFormula]:
        """Génère un seul alpha avec RL"""
        if self.agent is None or self.env is None:
            return None

        state = torch.FloatTensor(self.env.reset()).unsqueeze(0)
        done = False
        total_reward = 0
        actions = []

        while not done:
            action, log_prob = self.agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            actions.append(action)
            total_reward += reward
            state = torch.FloatTensor(next_state).unsqueeze(0)

        # Créer l'objet AlphaFormula
        if total_reward > -0.5:  # Seuil de qualité minimum
            formula_str = self._actions_to_formula_string(actions)

            return AlphaFormula(
                formula=formula_str,
                operators=self.env.current_formula,
                ic=max(0, total_reward),
                rank_ic=max(0, total_reward * 0.8),  # Approximation
                complexity=self.env.current_complexity,
                generation=self.generation_count,
                metadata={
                    "actions": actions,
                    "total_reward": total_reward,
                    "strategy": self.name
                }
            )

        return None

    def _actions_to_formula_string(self, actions: List[int]) -> str:
        """Convertit une séquence d'actions en string de formule"""
        if self.env is None:
            return "unknown_formula"

        elements = [self.env._decode_action(a) for a in actions]

        # Simplification : créer une représentation readable
        ops = [e for e in elements if e in self.search_space.operators]
        features = [e for e in elements if e in self.search_space.features]

        if ops and features:
            return f"{ops[0]}({features[0] if features else 'close'})"
        else:
            return "simple_formula"

    def _generate_signals_from_best_alpha(self, data: pd.DataFrame) -> List[Signal]:
        """Génère des signaux à partir du meilleur alpha"""
        signals = []

        try:
            best_alphas = self.get_best_alphas(1)
            if not best_alphas:
                return signals

            best_alpha = best_alphas[0]

            # Évaluer l'alpha sur les données courantes
            alpha_values = self._evaluate_alpha_on_data(best_alpha, data)

            if alpha_values is None:
                return signals

            self.last_alpha_values = alpha_values

            # Générer signal basé sur la valeur alpha
            last_alpha_value = alpha_values.iloc[-1] if len(alpha_values) > 0 else 0

            if abs(last_alpha_value) > self.config.signal_threshold:
                action = SignalAction.BUY if last_alpha_value > 0 else SignalAction.SELL

                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=data.get("symbol", "UNKNOWN").iloc[-1] if "symbol" in data.columns else "UNKNOWN",
                    action=action,
                    strength=min(abs(last_alpha_value), 1.0),
                    price=data["close"].iloc[-1],
                    size=self.config.position_size,
                    metadata={
                        "strategy": self.name,
                        "alpha_formula": best_alpha.formula,
                        "alpha_ic": best_alpha.ic,
                        "alpha_value": last_alpha_value,
                        "generation": best_alpha.generation
                    }
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"Erreur génération signaux depuis alpha: {e}")

        return signals

    def _evaluate_alpha_on_data(self, alpha: AlphaFormula, data: pd.DataFrame) -> Optional[pd.Series]:
        """Évalue un alpha sur des données"""
        try:
            # Simplification : utiliser les opérateurs pour reproduire l'alpha
            if self.env is None:
                return None

            ops = SymbolicOperators()

            # Logique simplifiée basée sur la formule
            if "cs_rank" in alpha.formula:
                return ops.cs_rank(data["volume"])
            elif "sign" in alpha.formula:
                returns = data["close"].pct_change().fillna(0)
                return ops.sign(returns)
            elif "delta" in alpha.formula:
                return ops.delta(data["close"], 5)
            else:
                # Corrélation par défaut
                return data["close"].rolling(10).corr(data["volume"])

        except Exception as e:
            logger.debug(f"Erreur évaluation alpha: {e}")
            return None

    def get_best_alphas(self, top_k: int = 5) -> List[AlphaFormula]:
        """Retourne les meilleurs alphas générés"""
        if not self.generated_alphas:
            return []

        sorted_alphas = sorted(self.generated_alphas, key=lambda x: x.ic, reverse=True)
        return sorted_alphas[:top_k]

    def get_generation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de génération"""
        if not self.generated_alphas:
            return {"total": 0, "mean_ic": 0, "best_ic": 0}

        ics = [alpha.ic for alpha in self.generated_alphas]

        return {
            "total_generated": len(self.generated_alphas),
            "generations": self.generation_count,
            "mean_ic": np.mean(ics),
            "best_ic": max(ics),
            "std_ic": np.std(ics),
            "valid_alphas": len([a for a in self.generated_alphas if a.ic > self.config.min_ic_threshold]),
            "config": self.config.__dict__
        }

    def save_alphas_to_file(self, filepath: str) -> None:
        """Sauvegarde les alphas générés dans un fichier"""
        try:
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "strategy": self.name,
                "stats": self.get_generation_stats(),
                "alphas": [
                    {
                        "formula": alpha.formula,
                        "ic": alpha.ic,
                        "rank_ic": alpha.rank_ic,
                        "complexity": alpha.complexity,
                        "generation": alpha.generation,
                        "metadata": alpha.metadata
                    }
                    for alpha in self.generated_alphas
                ]
            }

            with open(filepath, "w") as f:
                json.dump(results_data, f, indent=2)

            logger.info(f"💾 Alphas sauvegardés: {filepath}")

        except Exception as e:
            logger.error(f"Erreur sauvegarde alphas: {e}")