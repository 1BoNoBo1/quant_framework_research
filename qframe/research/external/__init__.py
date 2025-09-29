"""
External Integrations
====================

Connections to external academic and research platforms.
"""

from .academic_data_connector import (
    AcademicDataConnector,
    DataSource,
    ResearchDataProvider,
    DataSourceType,
    DataQuery,
    DataResult
)
from .research_api_manager import (
    ResearchAPIManager,
    APIProvider,
    APIEndpoint,
    APICredentials,
    RateLimiter
)
from .paper_alert_system import (
    PaperAlertSystem,
    AlertRule,
    AlertTrigger,
    AlertChannel,
    PaperAlert
)

__all__ = [
    # Academic Data Connector
    "AcademicDataConnector",
    "DataSource",
    "ResearchDataProvider",
    "DataSourceType",
    "DataQuery",
    "DataResult",

    # Research API Manager
    "ResearchAPIManager",
    "APIProvider",
    "APIEndpoint",
    "APICredentials",
    "RateLimiter",

    # Paper Alert System
    "PaperAlertSystem",
    "AlertRule",
    "AlertTrigger",
    "AlertChannel",
    "PaperAlert"
]