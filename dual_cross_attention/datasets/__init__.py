"""
Dataset implementations for FGVC and Re-ID tasks.

This module provides data loaders for fine-grained visual categorization
and object re-identification datasets with appropriate preprocessing
and augmentation strategies.
"""

from .fgvc_datasets import CUBDataset, StanfordCarsDataset, FGVCAircraftDataset
from .reid_datasets import Market1501Dataset, DukeMTMCDataset, MSMT17Dataset, VeRi776Dataset
from .transforms import FGVCTransforms, ReIDTransforms

__all__ = [
    'CUBDataset', 'StanfordCarsDataset', 'FGVCAircraftDataset',
    'Market1501Dataset', 'DukeMTMCDataset', 'MSMT17Dataset', 'VeRi776Dataset',
    'FGVCTransforms', 'ReIDTransforms'
]

