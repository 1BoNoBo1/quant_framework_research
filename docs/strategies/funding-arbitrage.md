# 💰 Funding Arbitrage Strategy

Arbitrage sophistiqué des taux de financement avec prédiction ML et gestion multi-exchanges.

## Principe

Capture des écarts de taux de financement entre exchanges crypto avec prédiction des mouvements futurs.

```python
# Collecte taux multi-exchanges
funding_rates = {
    'binance': get_funding_rate('BTCUSDT'),
    'bybit': get_funding_rate('BTCUSDT'),
    'okex': get_funding_rate('BTC-USDT')
}

# Calcul spreads
spread = max(funding_rates.values()) - min(funding_rates.values())
```

## Prédiction ML

### Features Engineering
- Taux de financement historiques (7 jours)
- Open Interest et volume
- Volatilité implicite
- Sentiment de marché (fear/greed index)
- Corrélations cross-assets

### Modèle Random Forest
```python
class FundingPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

    def predict_funding_rate(self, features):
        return self.model.predict(features.reshape(1, -1))[0]
```

## Stratégie d'Execution

### Opportunités
1. **Spread Detection** : Écart > seuil minimum (ex: 0.01%)
2. **ML Confirmation** : Prédiction direction favorable
3. **Risk Check** : Exposure limits et corrélations

### Positions
```python
if predicted_spread > threshold:
    # Long exchange avec funding rate faible
    # Short exchange avec funding rate élevé
    position_size = kelly_sizing(win_prob, avg_profit, avg_loss)
```

## Gestion des Risques

- **Counterparty Risk** : Diversification exchanges
- **Basis Risk** : Monitoring écarts de prix
- **Liquidation Risk** : Margin buffer > 50%
- **Funding Risk** : Hedge des positions avant reset

## Performance Attendue

- **Sharpe** : 2-4 (stratégie market-neutral)
- **Volatilité** : < 5% annualisé
- **Capacity** : Limitée par liquidité exchanges

## Voir aussi

- [Risk Management](../portfolio/risk.md) - Gestion risques
- [Data Providers](../architecture/interfaces.md) - Sources données