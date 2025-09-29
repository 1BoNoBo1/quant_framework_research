#!/usr/bin/env python3
"""
Générateur de rapports de coverage intégrés dans MkDocs.
"""

import subprocess
import sys
from pathlib import Path

def generate_coverage_report():
    """Génère le rapport de coverage HTML."""
    project_root = Path(__file__).parent.parent.parent

    try:
        # Exécuter pytest avec coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=qframe",
            "--cov-report=html:docs/coverage/html",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term"
        ], cwd=project_root, capture_output=True, text=True)

        print("Coverage Report Generated!")
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Error generating coverage: {e}")
        return False

if __name__ == "__main__":
    generate_coverage_report()