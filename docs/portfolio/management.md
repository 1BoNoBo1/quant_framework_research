# 💼 Portfolio Management

Système complet de gestion de portfolios avec suivi, archivage et analytics avancés.

## Entités Core

### Portfolio
```python
@dataclass
class Portfolio:
    id: str
    name: str
    initial_capital: Decimal
    base_currency: str = "USD"
    created_time: datetime = field(default_factory=datetime.now)
```

### Position
```python
@dataclass
class Position:
    symbol: str
    quantity: Decimal
    average_price: Decimal
    side: PositionSide  # LONG, SHORT
    unrealized_pnl: Decimal
```

## Repository Pattern

```python
class MemoryPortfolioRepository(PortfolioRepository):
    def create(self, portfolio: Portfolio) -> Portfolio: ...
    def get_by_id(self, portfolio_id: str) -> Optional[Portfolio]: ...
    def update(self, portfolio: Portfolio) -> Portfolio: ...
    def archive(self, portfolio_id: str) -> bool: ...
    def get_portfolio_statistics(self, portfolio_id: str) -> Dict: ...
```

## Opérations Supportées

### Création et Gestion
- **Création** : Nouveau portfolio avec capital initial
- **Modification** : Update nom, description, paramètres
- **Archivage** : Soft delete avec conservation historique
- **Restauration** : Réactivation portfolios archivés

### Analytics
```python
portfolio_stats = {
    'total_value': Decimal('12500.00'),
    'total_pnl': Decimal('2500.00'),
    'pnl_percentage': 25.0,
    'positions_count': 5,
    'performance_history': [...],
    'risk_metrics': {
        'var_95': 0.025,
        'sharpe_ratio': 1.85
    }
}
```

## Multi-Portfolio Support

- **Isolation** : Portfolios indépendants avec leurs propres positions
- **Comparaison** : Analytics comparatifs entre portfolios
- **Consolidation** : Vue agrégée multi-portfolios
- **Risk Management** : Limites par portfolio et globales

## Intégration Strategy

```python
# Usage avec stratégies
container = get_container()
portfolio_service = container.resolve(PortfolioService)

# Création portfolio pour stratégie
portfolio = portfolio_service.create_portfolio(
    name="Mean Reversion Strategy",
    initial_capital=Decimal("10000.00")
)

# Génération ordres
strategy = container.resolve(AdaptiveMeanReversionStrategy)
signals = strategy.generate_signals(data)
orders = portfolio_service.execute_signals(portfolio.id, signals)
```

## Voir aussi

- [Risk Assessment](risk.md) - Évaluation des risques
- [Position Sizing](sizing.md) - Optimisation des tailles
- [Order Management](../backtesting/engine.md) - Exécution ordres