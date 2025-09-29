# qframe.infrastructure.data.market_data_pipeline


Infrastructure Layer: Market Data Pipeline
=========================================

Pipeline complet pour ingestion, normalisation et distribution
des données de marché en temps réel depuis multiple providers.


::: qframe.infrastructure.data.market_data_pipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      members_order: alphabetical
      filters:
        - "!^_"
        - "!^__"
      group_by_category: true
      show_category_heading: true

## Composants

### Classes

- `DataType`
- `DataQuality`
- `MarketDataPoint`
- `TickerData`
- `OrderBookData`
- `DataProvider`
- `MockDataProvider`
- `DataValidator`
- `DataNormalizer`
- `MarketDataCache`
- `MarketDataPipeline`

### Fonctions

- `get_market_data_pipeline`

