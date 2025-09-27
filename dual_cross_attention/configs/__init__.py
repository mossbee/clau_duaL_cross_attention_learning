"""
Configuration files for different tasks and datasets.

This module contains configuration classes that define all hyperparameters
and settings for FGVC and Re-ID experiments as described in the paper.
"""

from .fgvc_config import FGVCConfig, CUBConfig, CarsConfig, AircraftConfig
from .reid_config import ReIDConfig, Market1501Config, DukeConfig, MSMT17Config, VeRi776Config

__all__ = [
    'FGVCConfig', 'CUBConfig', 'CarsConfig', 'AircraftConfig',
    'ReIDConfig', 'Market1501Config', 'DukeConfig', 'MSMT17Config', 'VeRi776Config'
]

