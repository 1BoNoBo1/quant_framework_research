# 🧬 DMN LSTM Strategy

Deep Market Networks avec architecture LSTM avancée pour prédiction de séries temporelles financières.

## Architecture

```python
class DMNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           dropout=dropout, batch_first=True)
        self.attention = AttentionLayer(hidden_size) if use_attention else None
        self.output = nn.Linear(hidden_size, 1)
```

## Configuration

```python
DMNConfig(
    window_size=64,          # Taille fenêtre temporelle
    hidden_size=64,          # Unités cachées LSTM
    num_layers=2,            # Couches LSTM
    dropout=0.2,             # Régularisation
    use_attention=False,     # Mécanisme d'attention
    learning_rate=0.001,     # Taux d'apprentissage
    signal_threshold=0.1     # Seuil de génération signal
)
```

## Entraînement

- **Dataset** : TimeSeriesDataset avec sliding windows
- **Validation** : TimeSeriesSplit pour éviter data leakage
- **Loss** : MSE avec régularisation L2
- **Activation** : Tanh pour prédictions normalisées

## Performance

- **Métriques** : IC (Information Coefficient), Sharpe ratio
- **Backtesting** : Walk-forward analysis
- **GPU** : Support CUDA pour accélération

## Voir aussi

- [Configuration](../architecture/configuration.md) - Paramètres avancés
- [Backtesting](../backtesting/engine.md) - Validation performance