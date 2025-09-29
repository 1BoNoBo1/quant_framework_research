# qframe.infrastructure.events.core


Infrastructure Layer: Core Event System
=======================================

Système d'événements de base pour l'architecture événementielle.
Event, EventHandler, EventBus pour découpler les composants.


::: qframe.infrastructure.events.core
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

- `EventType`
- `EventMetadata`
- `Event`
- `DomainEvent`
- `SystemEvent`
- `EventHandler`
- `AsyncEventHandler`
- `EventHandlerRegistry`
- `EventBus`

### Fonctions

- `get_event_bus`

