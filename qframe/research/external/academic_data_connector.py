"""
Academic Data Connector
=======================

Connector for academic data sources and research repositories.
Access to financial datasets, research papers, and academic databases.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, AsyncIterator
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
import asyncio
import aiohttp
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import zipfile
import io
import re

from qframe.core.container import injectable
from qframe.core.config import FrameworkConfig

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types de sources de données académiques"""
    FINANCIAL_DATABASE = "financial_database"      # Bases financières (CRSP, Compustat, etc.)
    RESEARCH_REPOSITORY = "research_repository"    # Répertoires de recherche (arXiv, SSRN, etc.)
    CENTRAL_BANK = "central_bank"                  # Banques centrales (Fed, ECB, etc.)
    GOVERNMENT_AGENCY = "government_agency"        # Agences gouvernementales (SEC, etc.)
    ACADEMIC_INSTITUTION = "academic_institution"  # Institutions académiques
    COMMERCIAL_PROVIDER = "commercial_provider"    # Fournisseurs commerciaux (Bloomberg, etc.)


class DataFormat(Enum):
    """Formats de données supportés"""
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    EXCEL = "excel"
    PICKLE = "pickle"


@dataclass
class DataCredentials:
    """Identifiants pour accès aux données"""
    provider: str
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    additional_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataQuery:
    """Requête de données"""
    source_id: str
    query_type: str  # "papers", "timeseries", "cross_section", "panel"
    parameters: Dict[str, Any]

    # Filtres temporels
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Filtres de contenu
    keywords: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    journals: List[str] = field(default_factory=list)

    # Format et limites
    format: DataFormat = DataFormat.JSON
    limit: Optional[int] = None
    offset: int = 0


@dataclass
class DataResult:
    """Résultat d'une requête de données"""
    query: DataQuery
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    metadata: Dict[str, Any]

    # Informations sur le résultat
    total_records: int
    returned_records: int
    has_more: bool = False
    next_offset: Optional[int] = None

    # Métadonnées d'accès
    source: str = ""
    query_time: datetime = field(default_factory=datetime.utcnow)
    execution_time: float = 0.0
    cache_hit: bool = False


@dataclass
class DataSource:
    """Source de données académique"""
    source_id: str
    name: str
    description: str
    source_type: DataSourceType

    # Configuration d'accès
    base_url: str
    api_version: Optional[str] = None
    authentication_type: str = "api_key"  # "api_key", "oauth", "basic", "none"
    rate_limit: int = 100  # Requêtes par minute

    # Capacités
    supported_queries: List[str] = field(default_factory=list)
    supported_formats: List[DataFormat] = field(default_factory=list)
    max_records_per_query: Optional[int] = None

    # Métadonnées
    documentation_url: Optional[str] = None
    data_coverage: Dict[str, Any] = field(default_factory=dict)
    update_frequency: Optional[str] = None
    cost: str = "free"  # "free", "subscription", "per_query"


class ResearchDataProvider:
    """Interface pour fournisseurs de données de recherche"""

    def __init__(self, source: DataSource, credentials: Optional[DataCredentials] = None):
        self.source = source
        self.credentials = credentials
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Établit la connexion avec la source"""
        raise NotImplementedError

    async def disconnect(self):
        """Ferme la connexion"""
        if self.session:
            await self.session.close()

    async def query(self, query: DataQuery) -> DataResult:
        """Exécute une requête"""
        raise NotImplementedError

    async def validate_credentials(self) -> bool:
        """Valide les identifiants"""
        raise NotImplementedError


class ArXivProvider(ResearchDataProvider):
    """Fournisseur pour arXiv.org"""

    def __init__(self, source: DataSource, credentials: Optional[DataCredentials] = None):
        super().__init__(source, credentials)
        self.base_api_url = "http://export.arxiv.org/api/query"

    async def connect(self) -> bool:
        """Connexion à arXiv (pas d'authentification nécessaire)"""
        self.session = aiohttp.ClientSession()
        return True

    async def query(self, query: DataQuery) -> DataResult:
        """Requête arXiv"""
        if not self.session:
            await self.connect()

        start_time = datetime.utcnow()

        # Construire la requête arXiv
        search_query = self._build_arxiv_query(query)
        params = {
            "search_query": search_query,
            "start": query.offset,
            "max_results": query.limit or 100,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        try:
            async with self.session.get(self.base_api_url, params=params) as response:
                xml_content = await response.text()

                # Parser le XML de réponse
                papers = self._parse_arxiv_response(xml_content)

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                return DataResult(
                    query=query,
                    data=papers,
                    metadata={"source": "arxiv", "query_params": params},
                    total_records=len(papers),
                    returned_records=len(papers),
                    source="arXiv",
                    execution_time=execution_time
                )

        except Exception as e:
            logger.error(f"ArXiv query failed: {e}")
            raise

    def _build_arxiv_query(self, query: DataQuery) -> str:
        """Construit une requête arXiv"""
        search_terms = []

        # Keywords dans titre ou abstract
        if query.keywords:
            keyword_queries = []
            for keyword in query.keywords:
                keyword_queries.append(f'ti:"{keyword}" OR abs:"{keyword}"')
            search_terms.append(f"({' OR '.join(keyword_queries)})")

        # Auteurs
        if query.authors:
            author_queries = [f'au:"{author}"' for author in query.authors]
            search_terms.append(f"({' OR '.join(author_queries)})")

        # Catégories par défaut pour finance/économie
        if not search_terms:
            search_terms.append("cat:q-fin.* OR cat:econ.* OR cat:stat.ML")

        return " AND ".join(search_terms)

    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse la réponse XML d'arXiv"""
        root = ET.fromstring(xml_content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        for entry in root.findall("atom:entry", namespace):
            paper = {
                "id": entry.find("atom:id", namespace).text if entry.find("atom:id", namespace) is not None else "",
                "title": entry.find("atom:title", namespace).text.strip() if entry.find("atom:title", namespace) is not None else "",
                "summary": entry.find("atom:summary", namespace).text.strip() if entry.find("atom:summary", namespace) is not None else "",
                "published": entry.find("atom:published", namespace).text if entry.find("atom:published", namespace) is not None else "",
                "updated": entry.find("atom:updated", namespace).text if entry.find("atom:updated", namespace) is not None else "",
                "authors": [],
                "categories": [],
                "links": []
            }

            # Auteurs
            for author in entry.findall("atom:author", namespace):
                name_elem = author.find("atom:name", namespace)
                if name_elem is not None:
                    paper["authors"].append(name_elem.text)

            # Catégories
            for category in entry.findall("atom:category", namespace):
                term = category.get("term")
                if term:
                    paper["categories"].append(term)

            # Liens
            for link in entry.findall("atom:link", namespace):
                paper["links"].append({
                    "href": link.get("href"),
                    "rel": link.get("rel"),
                    "type": link.get("type")
                })

            papers.append(paper)

        return papers

    async def validate_credentials(self) -> bool:
        """arXiv ne nécessite pas d'authentification"""
        return True


class FREDProvider(ResearchDataProvider):
    """Fournisseur pour FRED (Federal Reserve Economic Data)"""

    def __init__(self, source: DataSource, credentials: Optional[DataCredentials] = None):
        super().__init__(source, credentials)
        self.base_api_url = "https://api.stlouisfed.org/fred"

    async def connect(self) -> bool:
        """Connexion à FRED"""
        if not self.credentials or not self.credentials.api_key:
            logger.error("FRED API key required")
            return False

        self.session = aiohttp.ClientSession()
        return await self.validate_credentials()

    async def query(self, query: DataQuery) -> DataResult:
        """Requête FRED"""
        if not self.session:
            await self.connect()

        start_time = datetime.utcnow()

        if query.query_type == "series_search":
            return await self._search_series(query, start_time)
        elif query.query_type == "series_data":
            return await self._get_series_data(query, start_time)
        else:
            raise ValueError(f"Unsupported query type: {query.query_type}")

    async def _search_series(self, query: DataQuery, start_time: datetime) -> DataResult:
        """Recherche de séries FRED"""
        params = {
            "api_key": self.credentials.api_key,
            "file_type": "json",
            "search_text": " ".join(query.keywords) if query.keywords else "GDP",
            "limit": query.limit or 100,
            "offset": query.offset
        }

        url = f"{self.base_api_url}/series/search"

        async with self.session.get(url, params=params) as response:
            data = await response.json()

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            series_list = data.get("seriess", [])

            return DataResult(
                query=query,
                data=series_list,
                metadata={"source": "fred", "query_params": params},
                total_records=data.get("count", len(series_list)),
                returned_records=len(series_list),
                source="FRED",
                execution_time=execution_time
            )

    async def _get_series_data(self, query: DataQuery, start_time: datetime) -> DataResult:
        """Récupère les données d'une série FRED"""
        series_id = query.parameters.get("series_id")
        if not series_id:
            raise ValueError("series_id required for series_data query")

        params = {
            "api_key": self.credentials.api_key,
            "file_type": "json",
            "series_id": series_id
        }

        if query.start_date:
            params["observation_start"] = query.start_date.strftime("%Y-%m-%d")
        if query.end_date:
            params["observation_end"] = query.end_date.strftime("%Y-%m-%d")

        url = f"{self.base_api_url}/series/observations"

        async with self.session.get(url, params=params) as response:
            data = await response.json()

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            observations = data.get("observations", [])

            # Convertir en DataFrame
            if observations:
                df = pd.DataFrame(observations)
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.set_index("date")
            else:
                df = pd.DataFrame()

            return DataResult(
                query=query,
                data=df,
                metadata={"source": "fred", "series_id": series_id},
                total_records=len(observations),
                returned_records=len(observations),
                source="FRED",
                execution_time=execution_time
            )

    async def validate_credentials(self) -> bool:
        """Valide la clé API FRED"""
        try:
            params = {
                "api_key": self.credentials.api_key,
                "file_type": "json",
                "limit": 1
            }

            url = f"{self.base_api_url}/series/search"
            async with self.session.get(url, params=params) as response:
                return response.status == 200
        except Exception:
            return False


class YahooFinanceProvider(ResearchDataProvider):
    """Fournisseur pour Yahoo Finance (données gratuites)"""

    def __init__(self, source: DataSource, credentials: Optional[DataCredentials] = None):
        super().__init__(source, credentials)
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"

    async def connect(self) -> bool:
        """Connexion Yahoo Finance (pas d'auth nécessaire)"""
        self.session = aiohttp.ClientSession()
        return True

    async def query(self, query: DataQuery) -> DataResult:
        """Requête Yahoo Finance"""
        if not self.session:
            await self.connect()

        start_time = datetime.utcnow()

        symbol = query.parameters.get("symbol", "SPY")
        interval = query.parameters.get("interval", "1d")

        params = {
            "interval": interval,
            "includePrePost": "false"
        }

        if query.start_date and query.end_date:
            params["period1"] = int(query.start_date.timestamp())
            params["period2"] = int(query.end_date.timestamp())
        else:
            params["range"] = query.parameters.get("range", "1y")

        url = f"{self.base_url}/{symbol}"

        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Parser les données Yahoo
                df = self._parse_yahoo_response(data, symbol)

                return DataResult(
                    query=query,
                    data=df,
                    metadata={"source": "yahoo_finance", "symbol": symbol},
                    total_records=len(df),
                    returned_records=len(df),
                    source="Yahoo Finance",
                    execution_time=execution_time
                )

        except Exception as e:
            logger.error(f"Yahoo Finance query failed: {e}")
            raise

    def _parse_yahoo_response(self, data: Dict[str, Any], symbol: str) -> pd.DataFrame:
        """Parse la réponse Yahoo Finance"""
        try:
            chart = data["chart"]["result"][0]
            timestamps = chart["timestamp"]
            quotes = chart["indicators"]["quote"][0]

            df_data = {
                "timestamp": [datetime.fromtimestamp(ts) for ts in timestamps],
                "open": quotes["open"],
                "high": quotes["high"],
                "low": quotes["low"],
                "close": quotes["close"],
                "volume": quotes["volume"]
            }

            df = pd.DataFrame(df_data)
            df = df.set_index("timestamp")
            df = df.dropna()

            return df

        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse Yahoo Finance response: {e}")
            return pd.DataFrame()

    async def validate_credentials(self) -> bool:
        """Yahoo Finance ne nécessite pas d'authentification"""
        return True


@injectable
class AcademicDataConnector:
    """
    Connecteur central pour sources de données académiques.

    Fonctionnalités:
    - Connexions multiples sources (arXiv, FRED, Yahoo Finance, etc.)
    - Cache intelligent et rate limiting
    - Requêtes fédérées
    - Transformation et normalisation des données
    - Monitoring et métriques d'usage
    """

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.sources: Dict[str, DataSource] = {}
        self.providers: Dict[str, ResearchDataProvider] = {}
        self.credentials: Dict[str, DataCredentials] = {}

        # Cache et rate limiting
        self.query_cache: Dict[str, DataResult] = {}
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}

        # Métriques
        self.query_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "failed_queries": 0,
            "by_source": {}
        }

        # Initialiser les sources par défaut
        self._initialize_default_sources()

        logger.info("Academic Data Connector initialized")

    def _initialize_default_sources(self):
        """Initialise les sources de données par défaut"""

        # arXiv
        arxiv_source = DataSource(
            source_id="arxiv",
            name="arXiv",
            description="Open access archive of scholarly articles",
            source_type=DataSourceType.RESEARCH_REPOSITORY,
            base_url="http://export.arxiv.org/api",
            authentication_type="none",
            rate_limit=1,  # 1 query per 3 seconds
            supported_queries=["papers", "search"],
            supported_formats=[DataFormat.XML, DataFormat.JSON],
            documentation_url="https://arxiv.org/help/api/",
            cost="free"
        )
        self.register_source(arxiv_source)

        # FRED
        fred_source = DataSource(
            source_id="fred",
            name="FRED Economic Data",
            description="Federal Reserve Economic Data",
            source_type=DataSourceType.CENTRAL_BANK,
            base_url="https://api.stlouisfed.org/fred",
            api_version="v1",
            authentication_type="api_key",
            rate_limit=120,  # 120 calls per minute
            supported_queries=["series_search", "series_data", "categories"],
            supported_formats=[DataFormat.JSON, DataFormat.XML],
            documentation_url="https://fred.stlouisfed.org/docs/api/",
            cost="free"
        )
        self.register_source(fred_source)

        # Yahoo Finance
        yahoo_source = DataSource(
            source_id="yahoo_finance",
            name="Yahoo Finance",
            description="Free financial data from Yahoo",
            source_type=DataSourceType.COMMERCIAL_PROVIDER,
            base_url="https://query1.finance.yahoo.com",
            authentication_type="none",
            rate_limit=2000,  # 2000 calls per hour
            supported_queries=["price_data", "fundamentals"],
            supported_formats=[DataFormat.JSON],
            documentation_url="https://finance.yahoo.com/",
            cost="free"
        )
        self.register_source(yahoo_source)

    def register_source(self, source: DataSource):
        """Enregistre une nouvelle source de données"""
        self.sources[source.source_id] = source

        # Créer le provider approprié
        if source.source_id == "arxiv":
            provider = ArXivProvider(source)
        elif source.source_id == "fred":
            provider = FREDProvider(source, self.credentials.get("fred"))
        elif source.source_id == "yahoo_finance":
            provider = YahooFinanceProvider(source)
        else:
            # Provider générique (à implémenter)
            provider = None

        if provider:
            self.providers[source.source_id] = provider

        logger.info(f"Registered data source: {source.source_id}")

    def set_credentials(self, source_id: str, credentials: DataCredentials):
        """Configure les identifiants pour une source"""
        self.credentials[source_id] = credentials

        # Mettre à jour le provider si il existe
        if source_id in self.providers:
            self.providers[source_id].credentials = credentials

    async def query(self, query: DataQuery) -> DataResult:
        """Exécute une requête sur une source de données"""

        # Vérifier le cache
        cache_key = self._generate_cache_key(query)
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_result.cache_hit = True
            self.query_stats["cache_hits"] += 1
            return cached_result

        # Vérifier rate limiting
        if not await self._check_rate_limit(query.source_id):
            raise Exception(f"Rate limit exceeded for source: {query.source_id}")

        # Exécuter la requête
        try:
            if query.source_id not in self.providers:
                raise ValueError(f"Unknown source: {query.source_id}")

            provider = self.providers[query.source_id]

            # Se connecter si nécessaire
            if not provider.session:
                await provider.connect()

            # Exécuter la requête
            result = await provider.query(query)

            # Mettre en cache
            self.query_cache[cache_key] = result

            # Mettre à jour les statistiques
            self.query_stats["total_queries"] += 1
            source_stats = self.query_stats["by_source"].setdefault(query.source_id, 0)
            self.query_stats["by_source"][query.source_id] = source_stats + 1

            logger.info(f"Query executed successfully: {query.source_id}")
            return result

        except Exception as e:
            self.query_stats["failed_queries"] += 1
            logger.error(f"Query failed for {query.source_id}: {e}")
            raise

    async def multi_source_query(
        self,
        queries: List[DataQuery],
        merge_strategy: str = "concat"
    ) -> DataResult:
        """Exécute des requêtes sur plusieurs sources et combine les résultats"""

        # Exécuter toutes les requêtes en parallèle
        tasks = [self.query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrer les erreurs
        valid_results = [r for r in results if isinstance(r, DataResult)]
        errors = [r for r in results if isinstance(r, Exception)]

        if errors:
            logger.warning(f"Some queries failed: {len(errors)} errors")

        if not valid_results:
            raise Exception("All queries failed")

        # Combiner les résultats
        combined_data = self._merge_results(valid_results, merge_strategy)

        # Créer un résultat combiné
        combined_result = DataResult(
            query=queries[0],  # Query de référence
            data=combined_data,
            metadata={
                "source": "multi_source",
                "sources": [r.source for r in valid_results],
                "merge_strategy": merge_strategy,
                "errors": len(errors)
            },
            total_records=sum(r.total_records for r in valid_results),
            returned_records=len(combined_data) if isinstance(combined_data, (list, pd.DataFrame)) else 1,
            source="Multi-source"
        )

        return combined_result

    def _merge_results(
        self,
        results: List[DataResult],
        strategy: str
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Combine les résultats de plusieurs sources"""

        if strategy == "concat":
            # Concaténer les DataFrames ou listes
            if all(isinstance(r.data, pd.DataFrame) for r in results):
                return pd.concat([r.data for r in results], ignore_index=True)
            elif all(isinstance(r.data, list) for r in results):
                combined = []
                for result in results:
                    combined.extend(result.data)
                return combined
            else:
                # Types mixtes - convertir en liste de dicts
                combined = []
                for result in results:
                    if isinstance(result.data, pd.DataFrame):
                        combined.extend(result.data.to_dict('records'))
                    elif isinstance(result.data, list):
                        combined.extend(result.data)
                    else:
                        combined.append(result.data)
                return combined

        elif strategy == "join":
            # Joindre les DataFrames par index
            if all(isinstance(r.data, pd.DataFrame) for r in results):
                merged = results[0].data
                for result in results[1:]:
                    merged = merged.join(result.data, how="outer", rsuffix=f"_{result.source}")
                return merged
            else:
                raise ValueError("Join strategy only supports DataFrames")

        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

    async def search_papers(
        self,
        keywords: List[str],
        sources: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Recherche de papiers académiques"""

        if sources is None:
            sources = ["arxiv"]  # Source par défaut

        queries = []
        for source_id in sources:
            if source_id in self.sources:
                query = DataQuery(
                    source_id=source_id,
                    query_type="papers",
                    parameters={},
                    keywords=keywords,
                    start_date=date_range[0] if date_range else None,
                    end_date=date_range[1] if date_range else None,
                    limit=limit
                )
                queries.append(query)

        if not queries:
            return []

        # Exécuter requêtes multi-sources
        result = await self.multi_source_query(queries, merge_strategy="concat")

        return result.data if isinstance(result.data, list) else []

    async def get_economic_data(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime,
        source: str = "fred"
    ) -> pd.DataFrame:
        """Récupère des données économiques"""

        query = DataQuery(
            source_id=source,
            query_type="series_data",
            parameters={"series_id": series_id},
            start_date=start_date,
            end_date=end_date,
            format=DataFormat.JSON
        )

        result = await self.query(query)

        if isinstance(result.data, pd.DataFrame):
            return result.data
        else:
            return pd.DataFrame()

    async def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        source: str = "yahoo_finance"
    ) -> pd.DataFrame:
        """Récupère des données de marché"""

        query = DataQuery(
            source_id=source,
            query_type="price_data",
            parameters={"symbol": symbol, "interval": interval},
            start_date=start_date,
            end_date=end_date
        )

        result = await self.query(query)

        if isinstance(result.data, pd.DataFrame):
            return result.data
        else:
            return pd.DataFrame()

    def get_available_sources(self) -> Dict[str, DataSource]:
        """Retourne les sources disponibles"""
        return self.sources.copy()

    def get_query_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation"""
        return self.query_stats.copy()

    async def disconnect_all(self):
        """Ferme toutes les connexions"""
        for provider in self.providers.values():
            await provider.disconnect()

        logger.info("All data source connections closed")

    def _generate_cache_key(self, query: DataQuery) -> str:
        """Génère une clé de cache pour une requête"""
        key_data = {
            "source": query.source_id,
            "type": query.query_type,
            "params": query.parameters,
            "keywords": sorted(query.keywords),
            "start": query.start_date.isoformat() if query.start_date else None,
            "end": query.end_date.isoformat() if query.end_date else None,
            "limit": query.limit,
            "offset": query.offset
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))

    async def _check_rate_limit(self, source_id: str) -> bool:
        """Vérifie les limites de taux"""
        if source_id not in self.sources:
            return False

        source = self.sources[source_id]
        current_time = datetime.utcnow()

        # Initialiser rate limiter si nécessaire
        if source_id not in self.rate_limiters:
            self.rate_limiters[source_id] = {
                "requests": [],
                "limit": source.rate_limit
            }

        rate_limiter = self.rate_limiters[source_id]

        # Nettoyer les anciennes requêtes (> 1 minute)
        cutoff_time = current_time - timedelta(minutes=1)
        rate_limiter["requests"] = [
            req_time for req_time in rate_limiter["requests"]
            if req_time > cutoff_time
        ]

        # Vérifier la limite
        if len(rate_limiter["requests"]) >= rate_limiter["limit"]:
            return False

        # Ajouter cette requête
        rate_limiter["requests"].append(current_time)
        return True