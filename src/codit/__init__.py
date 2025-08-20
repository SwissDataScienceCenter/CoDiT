"""
CoDiT - Prediction of trajectories in ocean
"""

__version__ = "0.0.1"

# Import main modules to make them available at package level
from . import codit_module
from . import data_module  
from . import model_module

__all__ = [
    "codit_module",
    "data_module", 
    "model_module",
]
