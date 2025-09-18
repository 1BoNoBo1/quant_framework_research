#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Pipeline - Quant Stack Production
Pipeline quantitatif sophistiqué avec validation stricte données réelles
"""

__version__ = "1.0.0"
__author__ = "Quant Stack Production Team"

# Import des modules principaux
from . import utils
from . import data_sources
from . import features  
from . import alphas
from . import selection

__all__ = [
    'utils',
    'data_sources', 
    'features',
    'alphas',
    'selection'
]