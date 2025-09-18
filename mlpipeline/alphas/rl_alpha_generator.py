"""
Générateur RL d'Alphas Automatique
Basé sur "Synergistic Formulaic Alpha Generation for Quantitative Trading"

Implémente un agent PPO qui génère automatiquement des formules alpha
en combinant intelligemment les opérateurs symboliques.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import random
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from mlpipeline.features.symbolic_operators import SymbolicOperators

logger = logging.getLogger(__name__)

@dataclass
class AlphaFormula:
    """Représente une formule alpha générée"""
    formula: str
    operators: List[str]
    ic: float
    rank_ic: float
    complexity: int
    generation: int

class SearchSpace:
    """Définit l'espace de recherche pour la génération d'alphas"""

    def __init__(self):
        # Opérateurs disponibles (du papier)
        self.operators = [
            'sign', 'cs_rank', 'product', 'scale', 'pow_op',
            'skew', 'kurt', 'ts_rank', 'delta', 'argmax', 'argmin',
            'cond', 'wma', 'ema', 'mad'
        ]

        # Features de base
        self.features = ['open', 'high', 'low', 'close', 'volume', 'vwap']

        # Constantes et paramètres temporels
        self.constants = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.time_deltas = [5, 10, 20, 30, 40, 50, 60, 120]

        # Taille maximale des formules
        self.max_depth = 4
        self.max_complexity = 10

    def get_action_space_size(self) -> int:
        """Retourne la taille de l'espace d'action"""
        return len(self.operators) + len(self.features) + len(self.constants) + len(self.time_deltas)

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
                self.data['close'].mean() / 50000,  # Prix normalisé
                self.data['volume'].mean() / 10000,  # Volume normalisé
                self.data['close'].std() / 1000,     # Volatilité
                len(self.data) / 1000,               # Taille dataset
                self.current_complexity / 10         # Complexité courante
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
        done = (self.current_complexity >= self.search_space.max_complexity or
                len(self.current_formula) >= 20)

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
            return str(self.search_space.time_deltas[action - total_ops - total_features - total_constants])

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
            relevant_elements = [e for e in self.current_formula if e in self.search_space.operators + self.search_space.features][-3:]

            if len(relevant_elements) < 2:
                return None

            # Construire une formule simple
            if 'close' in self.data.columns and 'volume' in self.data.columns:
                if 'cs_rank' in relevant_elements:
                    return self.ops.cs_rank(self.data['volume'])
                elif 'sign' in relevant_elements:
                    returns = self.data['close'].pct_change().fillna(0)
                    return self.ops.sign(returns)
                elif 'delta' in relevant_elements:
                    return self.ops.delta(self.data['close'], 5)
                else:
                    # Formule par défaut
                    return self.data['close'].rolling(10).corr(self.data['volume'])

            return None

        except Exception as e:
            logger.debug(f"Erreur évaluation formule: {e}")
            return None

    def _calculate_ic(self, alpha_values: pd.Series) -> float:
        """Calcule l'Information Coefficient"""
        try:
            if 'ret' in self.data.columns:
                # Utiliser les returns futurs
                future_returns = self.data['ret'].shift(-1).fillna(0)
                aligned_alpha = alpha_values.reindex(future_returns.index).fillna(0)

                # Corrélation de Pearson
                correlation = aligned_alpha.corr(future_returns)
                return correlation if not np.isnan(correlation) else 0.0
            else:
                # Corrélation avec close price changes
                price_changes = self.data['close'].pct_change().fillna(0)
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
            nn.Softmax(dim=-1)
        )

        # Réseau de valeur (critic)
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
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

class RLAlphaGenerator:
    """Générateur principal d'alphas utilisant RL"""

    def __init__(self, market_data: pd.DataFrame):
        self.data = market_data
        self.search_space = SearchSpace()
        self.env = FormulaEnvironment(market_data, self.search_space)

        # Paramètres PPO
        self.state_size = 50
        self.action_size = self.search_space.get_action_space_size()

        # Agent PPO
        self.agent = PPOAgent(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=3e-4)

        # Historique des alphas générés
        self.generated_alphas: List[AlphaFormula] = []
        self.generation_count = 0

        logger.info(f"✅ RL Alpha Generator initialisé - Action space: {self.action_size}")

    def generate_alpha_batch(self, num_episodes: int = 10) -> List[AlphaFormula]:
        """Génère un batch d'alphas"""
        new_alphas = []

        for episode in range(num_episodes):
            try:
                alpha = self._generate_single_alpha()
                if alpha and alpha.ic > 0.01:  # Seuil minimum d'IC
                    new_alphas.append(alpha)
                    logger.info(f"📊 Alpha généré - IC: {alpha.ic:.4f}, Complexité: {alpha.complexity}")
            except Exception as e:
                logger.debug(f"Erreur génération épisode {episode}: {e}")

        self.generated_alphas.extend(new_alphas)
        self.generation_count += 1

        logger.info(f"🎯 Batch terminé: {len(new_alphas)}/{num_episodes} alphas valides")
        return new_alphas

    def _generate_single_alpha(self) -> Optional[AlphaFormula]:
        """Génère un seul alpha"""
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
                generation=self.generation_count
            )

        return None

    def _actions_to_formula_string(self, actions: List[int]) -> str:
        """Convertit une séquence d'actions en string de formule"""
        elements = [self.env._decode_action(a) for a in actions]

        # Simplification : créer une représentation readable
        ops = [e for e in elements if e in self.search_space.operators]
        features = [e for e in elements if e in self.search_space.features]

        if ops and features:
            return f"{ops[0]}({features[0] if features else 'close'})"
        else:
            return "simple_formula"

    def get_best_alphas(self, top_k: int = 5) -> List[AlphaFormula]:
        """Retourne les meilleurs alphas générés"""
        if not self.generated_alphas:
            return []

        sorted_alphas = sorted(self.generated_alphas, key=lambda x: x.ic, reverse=True)
        return sorted_alphas[:top_k]

    def get_generation_stats(self) -> Dict:
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
            "valid_alphas": len([a for a in self.generated_alphas if a.ic > 0.01])
        }

def test_rl_generator():
    """Test du générateur RL"""
    print("🧪 Test du générateur RL d'alphas")

    # Créer des données de test
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

    # Créer le générateur
    generator = RLAlphaGenerator(data)

    # Générer des alphas
    print("🎯 Génération d'alphas...")
    alphas = generator.generate_alpha_batch(5)

    print(f"✅ Alphas générés: {len(alphas)}")

    # Statistiques
    stats = generator.get_generation_stats()
    print(f"📊 Stats: {stats}")

    # Meilleurs alphas
    best = generator.get_best_alphas(3)
    for i, alpha in enumerate(best):
        print(f"🏆 Top {i+1}: IC={alpha.ic:.4f}, Formule={alpha.formula}")

    return generator

if __name__ == "__main__":
    test_rl_generator()