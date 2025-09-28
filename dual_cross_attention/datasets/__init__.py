"""
Dataset implementations for FGVC and Re-ID tasks.

This module provides data loaders for fine-grained visual categorization
and object re-identification datasets with appropriate preprocessing
and augmentation strategies.
"""

from .fgvc_datasets import CUBDataset, FGVCDataLoader, PairBatchSampler
from .reid_datasets import Market1501Dataset, ReIDDataLoader
from .transforms import FGVCTransforms, ReIDTransforms, get_transform_factory

__all__ = [
    'CUBDataset', 'FGVCDataLoader', 'PairBatchSampler',
    'Market1501Dataset', 'ReIDDataLoader',
    'FGVCTransforms', 'ReIDTransforms', 'get_transform_factory'
]

