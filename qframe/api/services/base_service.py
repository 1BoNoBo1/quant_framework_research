"""
🔧 Base Service
Classe de base pour tous les services API
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Classe de base pour tous les services."""

    def __init__(self):
        self._is_running = False
        self._start_time: Optional[datetime] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def start(self):
        """Démarre le service."""
        pass

    @abstractmethod
    async def stop(self):
        """Arrête le service."""
        pass

    def is_healthy(self) -> bool:
        """Vérifie si le service est en bonne santé."""
        return self._is_running

    def get_uptime(self) -> Optional[float]:
        """Retourne le temps de fonctionnement en secondes."""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return None

    def get_status(self) -> dict:
        """Retourne le statut du service."""
        return {
            "name": self.__class__.__name__,
            "running": self._is_running,
            "healthy": self.is_healthy(),
            "uptime": self.get_uptime(),
            "start_time": self._start_time.isoformat() if self._start_time else None
        }