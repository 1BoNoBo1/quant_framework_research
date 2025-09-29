# 🔗 Interfaces & Protocols

QFrame utilise les Protocols Python pour définir des contrats clairs et modernes.

## Protocols Principaux

### Strategy Protocol
```python
class Strategy(Protocol):
    def generate_signals(self, data: pd.DataFrame, features: Optional[pd.DataFrame]) -> List[Signal]: ...
    def get_config(self) -> StrategyConfig: ...
```

### DataProvider Protocol
```python
class DataProvider(Protocol):
    async def fetch_ohlcv(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame: ...
    async def fetch_orderbook(self, symbol: str) -> Dict: ...
```

### FeatureProcessor Protocol
```python
class FeatureProcessor(Protocol):
    def process(self, data: pd.DataFrame) -> pd.DataFrame: ...
    def get_feature_names(self) -> List[str]: ...
```

## Avantages des Protocols

- **Duck typing** : Interface flexible
- **Type safety** : Validation MyPy
- **Modularité** : Composants interchangeables

## Voir aussi

- [DI Container](di-container.md) - Injection de dépendances
- [Stratégies](../strategies/index.md) - Implémentations concrètes