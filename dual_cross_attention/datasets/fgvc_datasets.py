"""
Fine-Grained Visual Categorization (FGVC) Datasets

This module implements data loaders for FGVC benchmarks used in the paper:
1. CUB-200-2011: 200 bird species, 5,994 training images, 5,794 test images
2. Stanford Cars: 196 car models, ~8,000 training images, ~8,000 test images  
3. FGVC-Aircraft: 100 aircraft variants, ~6,600 training images, ~3,300 test images

All datasets use the same preprocessing pipeline:
- Resize to 550x550 
- Random crop to 448x448 for training
- Center crop to 448x448 for testing
- Standard ImageNet normalization
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from typing import Tuple, Optional, Dict, List
import json


class CUBDataset(Dataset):
    """
    CUB-200-2011 Bird Species Dataset
    
    Loads the Caltech-UCSD Birds-200-2011 dataset as described in CUB_200_2011.md.
    The dataset contains 200 bird species with fine-grained visual differences.
    
    Dataset structure (from CUB_200_2011.md):
    - images/: Images organized by species subdirectories
    - images.txt: List of all image files  
    - train_test_split.txt: Train/test split indicators
    - classes.txt: Class names (bird species)
    - image_class_labels.txt: Ground truth labels for each image
    - bounding_boxes.txt: Bounding box annotations (optional)
    
    Args:
        root_dir: Path to CUB dataset root directory
        split: "train" or "test" 
        transform: Image transformations to apply
        use_bounding_box: Whether to crop images using bounding box annotations
        
    Returns:
        image: Preprocessed image tensor [3, 448, 448]
        label: Class label (0-199) 
        image_id: Unique image identifier for tracking
        metadata: Additional information (species name, etc.)
    """
    
    def __init__(self, root_dir: str, split: str = "train", 
                 transform: Optional[object] = None, use_bounding_box: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_bounding_box = use_bounding_box
        
        # Load all annotations
        self._load_annotations()
        
        # Filter for current split
        self.samples = []
        for img_id in self.image_ids:
            if self.splits[img_id] == (1 if split == "train" else 0):
                self.samples.append(img_id)
        
        print(f"CUB Dataset: {len(self.samples)} {split} samples loaded")
    
    def _load_annotations(self):
        """
        Load all annotation files as described in CUB_200_2011.md
        
        Loads:
        - images.txt: image_id -> image_name mapping
        - train_test_split.txt: train/test split
        - classes.txt: class_id -> class_name mapping  
        - image_class_labels.txt: image_id -> class_id mapping
        - bounding_boxes.txt: bounding box coordinates (if used)
        """
        # Load images.txt: image_id -> image_name
        images_file = os.path.join(self.root_dir, 'images.txt')
        self.image_names = {}
        self.image_ids = []
        with open(images_file, 'r') as f:
            for line in f:
                img_id, img_name = line.strip().split(' ', 1)
                img_id = int(img_id)
                self.image_names[img_id] = img_name
                self.image_ids.append(img_id)
        
        # Load train_test_split.txt: image_id -> is_training_image
        split_file = os.path.join(self.root_dir, 'train_test_split.txt')
        self.splits = {}
        with open(split_file, 'r') as f:
            for line in f:
                img_id, is_training = line.strip().split()
                self.splits[int(img_id)] = int(is_training)
        
        # Load classes.txt: class_id -> class_name
        classes_file = os.path.join(self.root_dir, 'classes.txt')
        self.class_names = {}
        with open(classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                class_id = int(parts[0])
                class_name = parts[1]
                self.class_names[class_id] = class_name
        
        # Load image_class_labels.txt: image_id -> class_id
        labels_file = os.path.join(self.root_dir, 'image_class_labels.txt')
        self.image_labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                img_id, class_id = line.strip().split()
                self.image_labels[int(img_id)] = int(class_id) - 1  # Convert to 0-based indexing
        
        # Load bounding_boxes.txt if needed
        if self.use_bounding_box:
            bbox_file = os.path.join(self.root_dir, 'bounding_boxes.txt')
            self.bounding_boxes = {}
            with open(bbox_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    img_id = int(parts[0])
                    x, y, width, height = map(float, parts[1:5])
                    self.bounding_boxes[img_id] = (int(x), int(y), int(width), int(height))
    
    def _load_bounding_box(self, image_id: int) -> Tuple[int, int, int, int]:
        """Load bounding box for given image ID"""
        if self.use_bounding_box and image_id in self.bounding_boxes:
            return self.bounding_boxes[image_id]
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, Dict]:
        img_id = self.samples[idx]
        
        # Load image
        img_name = self.image_names[img_id]
        img_path = os.path.join(self.root_dir, 'images', img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Apply bounding box crop if requested
        if self.use_bounding_box:
            bbox = self._load_bounding_box(img_id)
            if bbox is not None:
                x, y, width, height = bbox
                image = image.crop((x, y, x + width, y + height))
        
        # Get label
        label = self.image_labels[img_id]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Metadata
        metadata = {
            'image_id': img_id,
            'image_name': img_name,
            'class_name': self.class_names[label + 1],  # Convert back to 1-based for class name
            'bbox': self._load_bounding_box(img_id) if self.use_bounding_box else None
        }
        
        return image, label, img_id, metadata


class StanfordCarsDataset(Dataset):
    """
    Stanford Cars Dataset
    
    196 car models with fine-grained differences (e.g., 2012 Tesla Model S vs 2012 BMW M3).
    Similar structure to CUB but with car-specific annotations.
    
    Args:
        root_dir: Path to Stanford Cars dataset root
        split: "train" or "test"
        transform: Image transformations
        
    Returns:
        image: Preprocessed image tensor [3, 448, 448]
        label: Car model class label (0-195)
        image_id: Unique image identifier
        metadata: Car make, model, year information
    """
    
    def __init__(self, root_dir: str, split: str = "train", 
                 transform: Optional[object] = None):
        super().__init__()
        pass
    
    def _load_annotations(self):
        """Load Stanford Cars annotation files"""
        pass
    
    def __len__(self) -> int:
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, Dict]:
        pass


class FGVCAircraftDataset(Dataset):
    """
    FGVC-Aircraft Dataset
    
    100 aircraft variants with subtle visual differences between similar aircraft types.
    
    Args:
        root_dir: Path to FGVC-Aircraft dataset root
        split: "train" or "test"  
        transform: Image transformations
        
    Returns:
        image: Preprocessed image tensor [3, 448, 448]
        label: Aircraft variant class label (0-99)
        image_id: Unique image identifier
        metadata: Aircraft variant and family information
    """
    
    def __init__(self, root_dir: str, split: str = "train",
                 transform: Optional[object] = None):
        super().__init__()
        pass
    
    def _load_annotations(self):
        """Load FGVC-Aircraft annotation files"""
        pass
    
    def __len__(self) -> int:
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, Dict]:
        pass


class PairBatchSampler:
    """
    Custom batch sampler for pair-wise cross-attention training.
    
    Ensures each batch has good diversity for creating meaningful 
    distractor pairs for PWCA training.
    """
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group samples by class for better pairing
        self.class_to_indices = {}
        for idx in range(len(dataset)):
            _, label, _, _ = dataset[idx]
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.num_classes = len(self.class_to_indices)
        self.length = len(dataset) // batch_size
    
    def __iter__(self):
        if self.shuffle:
            # Shuffle indices within each class
            for indices in self.class_to_indices.values():
                np.random.shuffle(indices)
        
        # Create batches with diverse classes
        all_indices = []
        for indices in self.class_to_indices.values():
            all_indices.extend(indices)
        
        if self.shuffle:
            np.random.shuffle(all_indices)
        
        for i in range(0, len(all_indices) - self.batch_size + 1, self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            yield batch
    
    def __len__(self):
        return self.length


class FGVCDataLoader:
    """
    Unified data loader factory for all FGVC datasets.
    
    Provides consistent interface for loading different FGVC datasets
    with appropriate transforms and batch sampling strategies.
    
    For pair-wise cross-attention training, implements batch sampling
    that ensures diverse image pairs within each batch.
    
    Args:
        dataset_name: "cub", "cars", or "aircraft"
        root_dirs: Dictionary mapping dataset names to root directories
        batch_size: Training batch size (default: 16 as in paper)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        train_loader: Training data loader with pair sampling
        test_loader: Test data loader  
        num_classes: Number of classes in the dataset
        class_names: List of class names for visualization
    """
    
    def __init__(self, dataset_name: str, root_dirs: Dict[str, str],
                 batch_size: int = 16, num_workers: int = 4, pin_memory: bool = True):
        self.dataset_name = dataset_name
        self.root_dirs = root_dirs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Import transforms here to avoid circular import
        from .transforms import get_transform_factory
        self.train_transform = get_transform_factory("fgvc", dataset_name, is_training=True)
        self.test_transform = get_transform_factory("fgvc", dataset_name, is_training=False)
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, int, List[str]]:
        """
        Create train and test data loaders.
        
        Training loader uses custom batch sampler for pair-wise cross-attention
        to ensure good diversity of image pairs within each batch.
        
        Returns:
            train_loader: Training data loader with pair sampling
            test_loader: Test data loader
            num_classes: Number of classes  
            class_names: Class name list
        """
        # Create datasets
        if self.dataset_name == "cub":
            train_dataset = CUBDataset(
                root_dir=self.root_dirs["cub"],
                split="train",
                transform=self.train_transform
            )
            test_dataset = CUBDataset(
                root_dir=self.root_dirs["cub"],
                split="test", 
                transform=self.test_transform
            )
            num_classes = 200
            class_names = list(train_dataset.class_names.values())
            
        else:
            raise ValueError(f"Dataset {self.dataset_name} not implemented yet")
        
        # Create batch sampler for training
        train_sampler = PairBatchSampler(
            train_dataset, 
            self.batch_size, 
            shuffle=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
        
        return train_loader, test_loader, num_classes, class_names
    
    def _collate_fn(self, batch):
        """Custom collate function to handle metadata"""
        images, labels, image_ids, metadata = zip(*batch)
        
        # Stack images and labels
        images = torch.stack(images)
        labels = torch.tensor(labels)
        image_ids = torch.tensor(image_ids)
        
        return {
            'images': images,
            'labels': labels, 
            'image_ids': image_ids,
            'metadata': metadata
        }
    
    def create_pair_batch_sampler(self, dataset: Dataset) -> PairBatchSampler:
        """
        Create batch sampler that ensures diverse image pairs.
        
        For PWCA training, we need batches with good diversity to create
        meaningful distractor relationships between images.
        
        Args:
            dataset: FGVC dataset instance
            
        Returns:
            batch_sampler: Custom sampler for pair-wise training
        """
        return PairBatchSampler(dataset, self.batch_size, shuffle=True)


def get_fgvc_dataset_stats() -> Dict[str, Dict]:
    """
    Get dataset statistics for all FGVC datasets.
    
    Returns comprehensive statistics including number of classes, images,
    and dataset-specific properties for proper model configuration.
    
    Returns:
        stats: Dictionary with statistics for each dataset
    """
    stats = {
        "cub": {
            "num_classes": 200,
            "num_train_images": 5994,
            "num_test_images": 5794,
            "input_size": (448, 448),
            "description": "Bird species classification"
        },
        "cars": {
            "num_classes": 196, 
            "num_train_images": 8144,
            "num_test_images": 8041,
            "input_size": (448, 448),
            "description": "Car model classification"
        },
        "aircraft": {
            "num_classes": 100,
            "num_train_images": 6667,
            "num_test_images": 3333,
            "input_size": (448, 448), 
            "description": "Aircraft variant classification"
        }
    }
    return stats

