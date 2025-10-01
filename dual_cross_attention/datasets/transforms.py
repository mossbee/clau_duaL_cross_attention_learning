"""
Data Transforms for FGVC and Re-ID Tasks

This module implements the specific data preprocessing and augmentation strategies
described in the paper for different tasks:

FGVC (Fine-Grained Visual Categorization):
- Resize to 550x550, random crop to 448x448 for training
- Center crop to 448x448 for testing
- Standard ImageNet normalization

Re-ID (Object Re-Identification):  
- Person datasets: Resize to 256x128
- Vehicle datasets: Resize to 256x256
- Standard data augmentation for Re-ID
- ImageNet normalization
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from typing import Tuple, Optional, Union
import random
from PIL import Image
import numpy as np


class FGVCTransforms:
    """
    Data transforms for Fine-Grained Visual Categorization.
    
    Implements the exact preprocessing pipeline described in the paper:
    - Training: Resize(550x550) -> RandomCrop(448x448) -> RandomHorizontalFlip
    - Testing: Resize(550x550) -> CenterCrop(448x448)
    - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    
    Args:
        is_training: Whether to apply training augmentations
        input_size: Final input size (default: 448x448)
        resize_size: Intermediate resize size (default: 550x550)  
        mean: Normalization mean values
        std: Normalization standard deviation values
        use_extra_augmentations: Whether to use ColorJitter and RandomRotation (default: False)
                                 Set to True for potential performance improvements, but not in paper
    """
    
    def __init__(self, is_training: bool = True, input_size: int = 448, 
                 resize_size: int = 550, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 use_extra_augmentations: bool = False):
        self.is_training = is_training
        self.input_size = input_size
        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.use_extra_augmentations = use_extra_augmentations
    
    def get_transforms(self) -> transforms.Compose:
        """
        Get the composed transforms for FGVC.
        
        Training transforms (as specified in paper):
        - Resize to 550x550
        - Random crop to 448x448
        - Random horizontal flip
        - Normalization
        
        Optional extra augmentations (if use_extra_augmentations=True):
        - Color jittering 
        - Random rotation
        
        Test transforms include:
        - Resize to 550x550  
        - Center crop to 448x448
        - Normalization
        
        Returns:
            transform: Composed transforms
        """
        if self.is_training:
            return self.get_train_transforms()
        else:
            return self.get_test_transforms()
    
    def get_train_transforms(self) -> transforms.Compose:
        """Get training-specific transforms with augmentation"""
        transform_list = [
            transforms.Resize((self.resize_size, self.resize_size)),
            transforms.RandomCrop(self.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        # Add extra augmentations if enabled (not specified in paper)
        if self.use_extra_augmentations:
            transform_list.extend([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(degrees=10),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return transforms.Compose(transform_list)
    
    def get_test_transforms(self) -> transforms.Compose:
        """Get test-specific transforms without augmentation"""
        transform_list = [
            transforms.Resize((self.resize_size, self.resize_size)),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]
        return transforms.Compose(transform_list)


class ReIDTransforms:
    """
    Data transforms for Object Re-Identification.
    
    Implements preprocessing for both person and vehicle Re-ID:
    - Person Re-ID: Resize to 256x128
    - Vehicle Re-ID: Resize to 256x256
    - Standard Re-ID augmentations (horizontal flip, padding, cropping)
    - ImageNet normalization
    
    Args:
        is_training: Whether to apply training augmentations
        input_size: Input size tuple (height, width)
        task_type: "person_reid" or "vehicle_reid"
        mean: Normalization mean values
        std: Normalization standard deviation values
        random_erase_prob: Probability for random erasing augmentation
    """
    
    def __init__(self, is_training: bool = True, 
                 input_size: Tuple[int, int] = (256, 128),
                 task_type: str = "person_reid",
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 random_erase_prob: float = 0.5):
        self.is_training = is_training
        self.input_size = input_size
        self.task_type = task_type
        self.mean = mean
        self.std = std
        self.random_erase_prob = random_erase_prob
    
    def get_transforms(self) -> transforms.Compose:
        """
        Get the composed transforms for Re-ID.
        
        Training transforms include:
        - Resize to target size
        - Random horizontal flip
        - Padding and random cropping
        - Random erasing (popular in Re-ID)
        - Normalization
        
        Test transforms include:
        - Resize to target size
        - Normalization
        
        Returns:
            transform: Composed transforms
        """
        if self.is_training:
            return self.get_train_transforms()
        else:
            return self.get_test_transforms()
    
    def get_train_transforms(self) -> transforms.Compose:
        """Get training-specific transforms with Re-ID augmentations"""
        transform_list = [
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),  # Padding for random crop
            transforms.RandomCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            RandomErasing(probability=self.random_erase_prob)
        ]
        return transforms.Compose(transform_list)
    
    def get_test_transforms(self) -> transforms.Compose:
        """Get test-specific transforms for Re-ID evaluation"""
        transform_list = [
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]
        return transforms.Compose(transform_list)


class RandomErasing(object):
    """
    Random Erasing augmentation for Re-ID.
    
    Randomly erases rectangular regions in the image to improve generalization.
    This is a common augmentation technique in Re-ID that helps models focus
    on different body parts and reduces overfitting.
    
    Args:
        probability: Probability of applying random erasing
        sl: Minimum proportion of erased area 
        sh: Maximum proportion of erased area
        r1: Minimum aspect ratio of erased area
        mean: Fill value for erased region
    """
    
    def __init__(self, probability: float = 0.5, sl: float = 0.02, sh: float = 0.4,
                 r1: float = 0.3, mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random erasing to input image.
        
        Args:
            img: Input image tensor [C, H, W]
            
        Returns:
            img: Image with random rectangular regions erased
        """
        if random.uniform(0, 1) >= self.probability:
            return img
        
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img
        
        return img


class PairWiseTransform:
    """
    Special transform for Pair-Wise Cross-Attention training.
    
    Creates pairs of images with consistent transforms for PWCA training.
    Ensures both images in a pair undergo similar augmentations while
    maintaining the randomness needed for effective training.
    
    Args:
        base_transform: Base transform to apply to both images
        sync_augmentation: Whether to synchronize augmentations across pairs
    """
    
    def __init__(self, base_transform: transforms.Compose, sync_augmentation: bool = False):
        pass
    
    def __call__(self, img1: Image.Image, img2: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms to image pair.
        
        Args:
            img1: First image in pair
            img2: Second image in pair (distractor)
            
        Returns:
            transformed_img1: Transformed first image
            transformed_img2: Transformed second image
        """
        pass


def get_transform_factory(task_type: str, dataset_name: str, is_training: bool = True, 
                          use_extra_augmentations: bool = False) -> transforms.Compose:
    """
    Factory function to get appropriate transforms for task and dataset.
    
    Automatically selects the correct transform configuration based on:
    - Task type: "fgvc" or "reid"  
    - Dataset: Specific dataset requirements
    - Training phase: Training vs evaluation transforms
    
    Args:
        task_type: "fgvc" or "reid"
        dataset_name: Specific dataset name
        is_training: Whether to get training or test transforms
        use_extra_augmentations: For FGVC only - whether to use extra augmentations
                                 (ColorJitter, RandomRotation) not specified in paper
        
    Returns:
        transform: Appropriate transform composition
    """
    
    if task_type == "fgvc":
        # FGVC datasets use 448x448 input
        fgvc_transforms = FGVCTransforms(is_training=is_training, input_size=448, resize_size=550,
                                         use_extra_augmentations=use_extra_augmentations)
        return fgvc_transforms.get_transforms()
    
    elif task_type == "reid":
        # Person Re-ID: 256x128, Vehicle Re-ID: 256x256
        if dataset_name in ["market1501", "duke", "msmt17"]:
            input_size = (256, 128)  # Person Re-ID
            task_subtype = "person_reid"
        elif dataset_name == "veri776":
            input_size = (256, 256)  # Vehicle Re-ID  
            task_subtype = "vehicle_reid"
        else:
            raise ValueError(f"Unknown Re-ID dataset: {dataset_name}")
            
        reid_transforms = ReIDTransforms(is_training=is_training, input_size=input_size, task_type=task_subtype)
        return reid_transforms.get_transforms()
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def visualize_transforms(transform: object, image_path: str, save_path: str = None):
    """
    Visualize the effect of transforms on a sample image.
    
    Useful for debugging and understanding what augmentations are being applied.
    
    Args:
        transform: Transform object to visualize
        image_path: Path to sample image
        save_path: Path to save visualization (optional)
    """
    pass

