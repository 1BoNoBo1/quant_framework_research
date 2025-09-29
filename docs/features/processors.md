# 🧮 Feature Processors

Processeurs de features pour transformation et enrichissement des données de marché.

## SymbolicFeatureProcessor

Processeur principal utilisant les opérateurs symboliques du papier académique.

```python
@injectable
class SymbolicFeatureProcessor(FeatureProcessor):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=data.index)
        
        # Opérateurs temporels
        features['ts_rank_close_20'] = ts_rank(data['close'], 20)
        features['delta_volume_5'] = delta(data['volume'], 5)
        features['argmax_high_10'] = argmax(data['high'], 10)
        
        # Opérateurs statistiques
        returns = data['close'].pct_change()
        features['skew_returns_30'] = skew(returns, 30)
        features['kurt_returns_30'] = kurt(returns, 30)
        features['mad_close_20'] = mad(data['close'], 20)
        
        return features
```

## TechnicalIndicatorProcessor

Processeur pour indicateurs techniques classiques.

```python
class TechnicalIndicatorProcessor(FeatureProcessor):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=data.index)
        
        # Moyennes mobiles
        features['sma_20'] = data['close'].rolling(20).mean()
        features['ema_20'] = data['close'].ewm(span=20).mean()
        
        # Oscillateurs
        features['rsi'] = calculate_rsi(data['close'], 14)
        features['macd'] = calculate_macd(data['close'])
        
        return features
```

## Voir aussi

- [Symbolic Operators](symbolic-operators.md) - Opérateurs de base
- [Feature Pipeline](pipeline.md) - Pipeline de processing