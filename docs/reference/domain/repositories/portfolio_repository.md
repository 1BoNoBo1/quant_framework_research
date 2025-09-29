# qframe.domain.repositories.portfolio_repository


Repository Interface: Portfolio Repository
=========================================

Interface de repository pour la persistance des portfolios.
Définit le contrat pour l'accès aux données selon l'architecture hexagonale.


::: qframe.domain.repositories.portfolio_repository
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

- `PortfolioRepository`
- `PortfolioQuery`
- `PortfolioAggregateQuery`
- `RepositoryError`
- `PortfolioNotFoundError`
- `PortfolioAlreadyExistsError`
- `DuplicatePortfolioNameError`

