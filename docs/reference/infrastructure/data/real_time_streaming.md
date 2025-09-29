# qframe.infrastructure.data.real_time_streaming


Infrastructure Layer: Real-Time Market Data Streaming
====================================================

Service de streaming en temps réel pour distribuer les données de marché
vers différents consommateurs avec gestion des subscriptions et backpressure.


::: qframe.infrastructure.data.real_time_streaming
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

- `StreamingProtocol`
- `SubscriptionLevel`
- `StreamingSubscription`
- `StreamingConsumer`
- `CallbackConsumer`
- `AsyncGeneratorConsumer`
- `StreamingStatistics`
- `RealTimeStreamingService`

### Fonctions

- `get_streaming_service`

