# 🔣 Opérateurs Symboliques

Implémentation complète des opérateurs du papier "Synergistic Formulaic Alpha Generation for Quantitative Trading".

## Opérateurs Temporels

### ts_rank(x, t)
Rang temporel sur les t dernières périodes.
```python
def ts_rank(x: pd.Series, t: int) -> pd.Series:
    return x.rolling(window=t).rank(pct=True)
```

### delta(x, t)
Différence avec la valeur t périodes avant.
```python
def delta(x: pd.Series, t: int) -> pd.Series:
    return x - x.shift(t)
```

### argmax/argmin(x, t)
Index du maximum/minimum sur t périodes.
```python
def argmax(x: pd.Series, t: int) -> pd.Series:
    return x.rolling(window=t).apply(lambda w: w.argmax())
```

## Opérateurs Statistiques

### skew(x, window)
Asymétrie de la distribution sur une fenêtre.
```python
def skew(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window).skew()
```

### kurt(x, window)
Kurtosis (peakedness) de la distribution.
```python
def kurt(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window).kurt()
```

### mad(x, window)
Mean Absolute Deviation.
```python
def mad(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window).apply(
        lambda w: np.abs(w - w.mean()).mean()
    )
```

## Opérateurs Cross-Sectionnels

### cs_rank(x)
Rang cross-sectionnel (simulé avec rolling rank).
```python
def cs_rank(x: pd.Series) -> pd.Series:
    return x.rolling(window=20).rank(pct=True)
```

### scale(x)
Normalisation par somme absolue.
```python
def scale(x: pd.Series) -> pd.Series:
    abs_sum = x.rolling(window=20).apply(lambda w: np.abs(w).sum())
    return x / abs_sum
```

## Formules Alpha du Papier

### alpha_006
```python
alpha_006 = -1 * Corr(open, volume, 10)
```

### alpha_061
```python
alpha_061 = Less(
    CSRank((vwap - Min(vwap, 16))),
    CSRank(Corr(vwap, Mean(volume, 180), 17))
)
```

## SymbolicFeatureProcessor

```python
@injectable
class SymbolicFeatureProcessor(FeatureProcessor):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=data.index)

        # Génère 18+ features symboliques
        features['ts_rank_close_20'] = ts_rank(data['close'], 20)
        features['delta_volume_5'] = delta(data['volume'], 5)
        features['skew_returns_30'] = skew(returns, 30)
        # ... plus de features

        return features
```

## Voir aussi

- [RL Alpha Strategy](../strategies/rl-alpha.md) - Utilisation en RL
- [Feature Pipeline](pipeline.md) - Pipeline de processing