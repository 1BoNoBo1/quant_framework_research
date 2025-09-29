# qframe.application.backtesting.queries


Application Layer: Backtesting Query Handlers
============================================

Handlers pour les requêtes liées au backtesting.
Implémente le pattern CQRS pour les opérations de lecture.


::: qframe.application.backtesting.queries
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

- `GetBacktestConfigurationQuery`
- `GetAllBacktestConfigurationsQuery`
- `FindBacktestConfigurationsByNameQuery`
- `GetBacktestResultQuery`
- `FindBacktestResultsByConfigurationQuery`
- `FindBacktestResultsByStatusQuery`
- `FindBacktestResultsByDateRangeQuery`
- `FindBacktestResultsByStrategyQuery`
- `GetLatestBacktestResultsQuery`
- `FindBestPerformingBacktestsQuery`
- `FindBacktestsByMetricsCriteriaQuery`
- `GetBacktestPerformanceComparisonQuery`
- `SearchBacktestResultsQuery`
- `GetBacktestStatisticsQuery`
- `GetStrategyPerformanceSummaryQuery`
- `GetMonthlyPerformanceSummaryQuery`
- `GetBacktestDashboardQuery`
- `GetArchivedBacktestResultsQuery`
- `GetBacktestStorageUsageQuery`
- `GetBacktestConfigurationHandler`
- `GetAllBacktestConfigurationsHandler`
- `FindBacktestConfigurationsByNameHandler`
- `GetBacktestResultHandler`
- `FindBestPerformingBacktestsHandler`
- `GetBacktestPerformanceComparisonHandler`
- `GetBacktestStatisticsHandler`
- `GetBacktestDashboardHandler`

