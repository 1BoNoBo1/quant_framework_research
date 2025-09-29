#!/usr/bin/env python3
"""
🚀 QFrame API Starter
Script de démarrage pour l'API backend QFrame
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

from qframe.api.main import run_server

def setup_logging(level: str = "INFO"):
    """Configure le logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("qframe_api.log")
        ]
    )

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="QFrame API Backend Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_api.py                     # Démarrage standard
  python start_api.py --port 8080         # Port personnalisé
  python start_api.py --reload --debug    # Mode développement
  python start_api.py --host 0.0.0.0      # Accessible depuis le réseau
        """
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Adresse IP d'écoute (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port d'écoute (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Mode rechargement automatique (développement)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mode debug avec logs détaillés"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Niveau de logging (default: INFO)"
    )

    args = parser.parse_args()

    # Configuration du logging
    if args.debug:
        setup_logging("DEBUG")
    else:
        setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    # Affichage des informations de démarrage
    print("=" * 60)
    print("🚀 QFrame API Backend Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Reload: {args.reload}")
    print(f"Log Level: {args.log_level}")
    print("=" * 60)

    logger.info(f"Starting QFrame API on {args.host}:{args.port}")

    try:
        # Démarrer le serveur
        run_server(
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()