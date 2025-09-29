#!/usr/bin/env python3
"""
QFrame CLI Entry Point
=====================

Point d'entrée principal pour le CLI QFrame.
"""

import sys
import asyncio
from pathlib import Path

# Ajouter le répertoire racine au path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from qframe.applications.cli.interactive_cli import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()