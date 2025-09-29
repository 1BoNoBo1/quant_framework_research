# 🤖 RL Alpha Generation Strategy

Génération automatique d'alphas via Reinforcement Learning basé sur le papier "Synergistic Formulaic Alpha Generation".

## Architecture RL

```python
class PPOAgent:
    def __init__(self, state_dim=50, action_dim=42, learning_rate=3e-4):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, 1)
        self.optimizer = torch.optim.Adam(lr=learning_rate)
```

## Search Space

### Opérateurs Disponibles (15)
```python
operators = [
    "sign", "cs_rank", "product", "scale", "pow_op",
    "skew", "kurt", "ts_rank", "delta", "argmax",
    "argmin", "cond", "wma", "ema", "mad"
]
```

### Features (6)
```python
features = ["open", "high", "low", "close", "volume", "vwap"]
```

### Constantes (8)
```python
constants = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0]
```

## Environnement

- **État** : 50 dimensions (structure formule + stats marché)
- **Actions** : 42 actions possibles (opérateurs + features + constantes)
- **Reward** : Information Coefficient (IC) avec pénalité complexité

## Formules Alpha Générées

### Exemples Générés
```python
# Alpha simple
"sign(delta(close, 5))"

# Alpha complexe
"cs_rank(ts_rank(close, 10) * scale(volume))"

# Alpha avec conditions
"cond(close > ma(close, 20), delta(volume, 5), 0)"
```

## Training Loop

```python
for episode in range(1000):
    alpha_formula = agent.generate_alpha()
    ic_score = evaluate_alpha(alpha_formula, market_data)
    reward = ic_score - complexity_penalty(alpha_formula)
    agent.update_policy(reward)
```

## Évaluation

- **Information Coefficient** : Corrélation prédictions/réalité
- **Rank IC** : IC sur rangs pour robustesse
- **Turnover** : Coût de transaction implicite
- **Complexity** : Pénalité pour formules trop complexes

## Voir aussi

- [Symbolic Operators](../features/symbolic-operators.md) - Opérateurs utilisés
- [Backtesting](../backtesting/engine.md) - Validation performance