# 📈 Adaptive Mean Reversion Strategy

Stratégie de mean reversion avec détection automatique de régimes de marché et optimisation ML.

## Logique Core

```python
# Calcul Z-score adaptatif
z_score = (price - rolling_mean) / rolling_std

# Détection de régime
regime = detect_volatility_regime(volatility, regime_window=252)

# Seuils adaptatifs
if regime == "low_vol":
    entry_threshold = z_entry_base * 0.8
elif regime == "high_vol":
    entry_threshold = z_entry_base * 1.5
```

## Configuration

```python
MeanReversionConfig(
    lookback_short=10,         # Moyenne courte
    lookback_long=50,          # Moyenne longue
    z_entry_base=1.0,          # Seuil d'entrée base
    z_exit_base=0.2,           # Seuil de sortie base
    regime_window=252,         # Fenêtre détection régime
    use_ml_optimization=True   # Optimisation ML des seuils
)
```

## Détection de Régimes

### Régimes Supportés
- **Low Volatility** : Seuils réduits pour plus de sensibilité
- **Normal** : Seuils standards
- **High Volatility** : Seuils augmentés pour éviter faux signaux

### Calcul Volatility
```python
volatility = returns.rolling(window=20).std() * np.sqrt(252)
volatility_percentile = volatility.rolling(regime_window).rank(pct=True)
```

## Position Sizing

- **Kelly Criterion** : Optimal sizing basé sur probabilité de succès
- **Risk Management** : Stop-loss et take-profit adaptatifs
- **Max Exposure** : Limitation exposition totale

## Performance

- **Sharpe** : Généralement > 1.5 en marché sideways
- **Drawdown** : Contrôlé via détection de régimes
- **Hit Rate** : 60-70% sur signaux filtrés

## Voir aussi

- [Risk Management](../portfolio/risk.md) - Gestion des risques
- [Position Sizing](../portfolio/sizing.md) - Optimisation taille