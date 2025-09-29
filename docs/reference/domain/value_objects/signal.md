# qframe.domain.value_objects.signal


Value Object: Signal
==================

Signal de trading représentant une recommandation d'achat/vente.
Les value objects sont immutables et définis par leurs valeurs.


::: qframe.domain.value_objects.signal
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

- `SignalAction`
- `SignalConfidence`
- `Signal`

### Fonctions

- `create_buy_signal`
- `create_sell_signal`
- `create_close_signal`

