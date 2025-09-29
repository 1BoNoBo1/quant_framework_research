# qframe.infrastructure.events.event_store


Infrastructure Layer: Event Store
=================================

Store pour persister et récupérer les événements.
Support de l'Event Sourcing avec snapshots et streaming.


::: qframe.infrastructure.events.event_store
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

- `EventStoreError`
- `ConcurrencyError`
- `EventStream`
- `EventSnapshot`
- `StoredEvent`
- `EventStoreStatistics`
- `EventStore`
- `InMemoryEventStore`

### Fonctions

- `get_event_store`

