# 🧠 Stratégies de Trading

QFrame intègre 4 stratégies de recherche sophistiquées pour la génération d'alphas.

## Stratégies Disponibles

### 1. DMN LSTM Strategy
Architecture Deep Market Networks avec LSTM et attention optionnelle.
- **Modèle** : PyTorch avec 64+ hidden units
- **Features** : Séries temporelles avec sliding windows
- **Performance** : Prédiction de returns futurs

### 2. Adaptive Mean Reversion
Mean reversion adaptatif avec détection de régimes de marché.
- **Régimes** : Low volatility, Normal, High volatility
- **Optimisation** : ML pour seuils d'entrée/sortie
- **Position sizing** : Kelly Criterion

### 3. Funding Arbitrage
Arbitrage de taux de financement avec prédiction ML.
- **Multi-exchanges** : Collecte rates de financement
- **ML Prediction** : Random Forest pour taux futurs
- **Risk Management** : Gestion risque de contrepartie

### 4. RL Alpha Generation
Génération automatique d'alphas via Reinforcement Learning.
- **Agent** : PPO (Proximal Policy Optimization)
- **Environnement** : 42 actions possibles
- **Reward** : Information Coefficient (IC)

## Configuration

```python
from qframe.core.container import get_container

container = get_container()
strategy = container.resolve(AdaptiveMeanReversionStrategy)
signals = strategy.generate_signals(market_data)
```

## Voir aussi

- [DMN LSTM](dmn-lstm.md) - Deep Learning avancé
- [Mean Reversion](mean-reversion.md) - Stratégie adaptative
- [RL Alpha](rl-alpha.md) - Génération automatique