"""
Configuration for Object Re-Identification (Re-ID) Tasks

This module defines all hyperparameters and settings for Re-ID experiments
as specified in the paper:

Training Settings:
- Person datasets: Resize to 256x128  
- Vehicle datasets: Resize to 256x256
- Batch size: 64 with 4 images per ID
- Epochs: 120
- Optimizer: SGD with momentum 0.9, weight decay 1e-4
- Learning rate: 0.008, cosine decay
- Local query ratio R: 30% for all Re-ID datasets
- Loss: Cross-entropy + triplet loss

Datasets: Market1501, DukeMTMC-ReID, MSMT17, VeRi-776
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional


@dataclass
class ReIDConfig:
    """
    Base configuration for Object Re-Identification tasks.
    
    Contains all common settings shared across Re-ID datasets with
    dataset-specific values overridden in subclasses.
    """
    
    # Dataset settings
    task_type: str = "reid"
    dataset_name: str = "market1501"
    num_identities: int = 1501
    num_cameras: int = 6
    input_size: Tuple[int, int] = (256, 128)  # Person Re-ID default
    
    # Model architecture (same as FGVC)
    embed_dim: int = 768
    num_sa_layers: int = 12      # L=12 SA blocks
    num_glca_layers: int = 1     # M=1 GLCA block
    num_pwca_layers: int = 12    # T=12 PWCA blocks
    num_heads: int = 12
    hidden_dim: int = 3072
    patch_size: int = 16
    dropout: float = 0.1
    
    # Cross-attention settings  
    top_k_ratio: float = 0.3     # R=30% for Re-ID local queries
    use_pwca_training: bool = True
    share_pwca_weights: bool = True
    
    # Training hyperparameters (from paper)
    batch_size: int = 64
    gradient_accumulation_steps: int = 1  # No accumulation by default (paper uses full batch size)
    images_per_id: int = 4       # 4 images per ID in each batch
    num_epochs: int = 120
    learning_rate: float = 0.008
    momentum: float = 0.9        # SGD momentum
    weight_decay: float = 1e-4
    optimizer: str = "sgd"       # SGD for Re-ID vs Adam for FGVC
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 10
    
    # Loss settings
    use_uncertainty_weighting: bool = True
    use_triplet_loss: bool = True
    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    use_center_loss: bool = False  # Optional center loss
    center_loss_weight: float = 0.0005
    
    # Re-ID specific settings
    re_ranking: bool = False     # Post-processing re-ranking
    distance_metric: str = "euclidean"
    
    # Data augmentation (Re-ID specific)
    random_flip_prob: float = 0.5
    random_crop_prob: float = 0.5
    random_erase_prob: float = 0.5
    color_jitter: bool = True
    
    # Evaluation settings
    eval_frequency: int = 10  # Evaluate every 10 epochs
    save_frequency: int = 20
    
    # Pretrained model
    pretrained_model: Optional[str] = None
    freeze_backbone: bool = False
    
    # Hardware settings
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Logging
    log_frequency: int = 100
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Debug and wandb settings
    debug: bool = False
    use_wandb: bool = False
    wandb_project: str = "dual-cross-attention-reid"
    wandb_entity: Optional[str] = None
    experiment_name: Optional[str] = None
    wandb_tags: List[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Calculate sequence length for patch embeddings
        h_patches = self.input_size[0] // self.patch_size
        w_patches = self.input_size[1] // self.patch_size
        self.num_patches = h_patches * w_patches
        self.sequence_length = self.num_patches + 1  # +1 for CLS token
        
        # Validate batch sampling
        assert self.batch_size % self.images_per_id == 0, \
            f"Batch size {self.batch_size} must be divisible by images_per_id {self.images_per_id}"
        
        self.num_ids_per_batch = self.batch_size // self.images_per_id
    
    def get_model_config(self) -> Dict:
        """Get model configuration dictionary"""
        return {
            "img_size": self.input_size,
            "patch_size": self.patch_size,
            "num_classes": self.num_identities,  # Use identities as classes
            "embed_dim": self.embed_dim,
            "num_sa_layers": self.num_sa_layers,
            "num_glca_layers": self.num_glca_layers,
            "num_pwca_layers": self.num_pwca_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "top_k_ratio": self.top_k_ratio,
            "task_type": self.task_type
        }
    
    def get_optimizer_config(self) -> Dict:
        """Get optimizer configuration dictionary"""
        return {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "warmup_epochs": self.warmup_epochs,
            "num_epochs": self.num_epochs
        }
    
    def get_loss_config(self) -> Dict:
        """Get loss configuration dictionary"""
        return {
            "use_uncertainty_weighting": self.use_uncertainty_weighting,
            "use_triplet_loss": self.use_triplet_loss,
            "triplet_margin": self.triplet_margin,
            "triplet_weight": self.triplet_weight,
            "use_center_loss": self.use_center_loss,
            "center_loss_weight": self.center_loss_weight,
            "num_classes": self.num_identities
        }


@dataclass
class Market1501Config(ReIDConfig):
    """
    Configuration for Market-1501 Person Re-ID dataset.
    
    1501 identities, 32,668 images from 6 cameras.
    Standard person Re-ID benchmark.
    """
    
    dataset_name: str = "market1501" 
    num_identities: int = 1501
    num_cameras: int = 6
    input_size: Tuple[int, int] = (256, 128)  # Person Re-ID size
    
    # Dataset paths
    data_root: str = "/path/to/Market-1501-v15.09.15"
    
    def __post_init__(self):
        super().__post_init__()
        
        # Market1501 statistics
        self.dataset_mean = [0.485, 0.456, 0.406]
        self.dataset_std = [0.229, 0.224, 0.225]


@dataclass
class DukeConfig(ReIDConfig):
    """
    Configuration for DukeMTMC-ReID Person Re-ID dataset.
    
    1404 identities, 36,411 images from 8 cameras.
    """
    
    dataset_name: str = "duke"
    num_identities: int = 1404
    num_cameras: int = 8
    input_size: Tuple[int, int] = (256, 128)
    
    # Dataset paths
    data_root: str = "/path/to/DukeMTMC-reID"
    
    def __post_init__(self):
        super().__post_init__()
        
        self.dataset_mean = [0.485, 0.456, 0.406]
        self.dataset_std = [0.229, 0.224, 0.225]


@dataclass 
class MSMT17Config(ReIDConfig):
    """
    Configuration for MSMT17 Large-Scale Person Re-ID dataset.
    
    4101 identities, 126,441 images from 15 cameras.
    More challenging due to larger scale and diversity.
    
    Paper results: Swin-T achieves 55.7% → 56.7% mAP with dual cross-attention.
    """
    
    dataset_name: str = "msmt17"
    num_identities: int = 4101
    num_cameras: int = 15
    input_size: Tuple[int, int] = (256, 128)
    
    # MSMT17 specific settings (larger dataset)
    num_epochs: int = 150  # More epochs for larger dataset
    eval_frequency: int = 15
    
    # Dataset paths
    data_root: str = "/path/to/MSMT17_V1"
    
    def __post_init__(self):
        super().__post_init__()
        
        self.dataset_mean = [0.485, 0.456, 0.406]
        self.dataset_std = [0.229, 0.224, 0.225]


@dataclass
class VeRi776Config(ReIDConfig):
    """
    Configuration for VeRi-776 Vehicle Re-ID dataset.
    
    776 vehicles, 49,357 images from 20 cameras as described in VeRi_776.md.
    Vehicle Re-ID uses 256x256 input size unlike person Re-ID.
    
    Dataset structure from VeRi_776.md:
    - image_train/: 37,778 training images
    - image_query/: 1,678 query images
    - image_test/: 11,579 gallery images  
    - XML labels with vehicle ID, camera ID, color, type
    """
    
    dataset_name: str = "veri776"
    num_identities: int = 776
    num_cameras: int = 20
    input_size: Tuple[int, int] = (256, 256)  # Square input for vehicles
    
    # Vehicle-specific settings
    use_color_attributes: bool = True  # Use color/type from XML
    use_vehicle_attributes: bool = True
    
    # Dataset paths
    data_root: str = "/path/to/VeRi"
    
    def __post_init__(self):
        super().__post_init__()
        
        self.dataset_mean = [0.485, 0.456, 0.406]
        self.dataset_std = [0.229, 0.224, 0.225]
        
        # Adjust patches for square input
        h_patches = self.input_size[0] // self.patch_size  # 256/16 = 16
        w_patches = self.input_size[1] // self.patch_size  # 256/16 = 16
        self.num_patches = h_patches * w_patches  # 256 patches
        self.sequence_length = self.num_patches + 1


def get_reid_config(dataset_name: str) -> ReIDConfig:
    """
    Factory function to get appropriate Re-ID configuration.
    
    Args:
        dataset_name: "market1501", "duke", "msmt17", or "veri776"
        
    Returns:
        config: Appropriate configuration instance
    """
    
    config_map = {
        "market1501": Market1501Config,
        "duke": DukeConfig,
        "msmt17": MSMT17Config, 
        "veri776": VeRi776Config
    }
    
    if dataset_name not in config_map:
        raise ValueError(f"Unknown Re-ID dataset: {dataset_name}")
    
    return config_map[dataset_name]()


def create_reid_experiment_configs() -> List[ReIDConfig]:
    """
    Create configurations for all Re-ID experiments in the paper.
    
    Returns:
        configs: List of all Re-ID experiment configurations
    """
    
    configs = []
    
    # Standard configurations for each dataset
    for dataset in ["market1501", "duke", "msmt17", "veri776"]:
        config = get_reid_config(dataset)
        configs.append(config)
    
    return configs


def get_reid_ablation_configs() -> List[ReIDConfig]:
    """
    Create configurations for Re-ID ablation studies.
    
    Tests different combinations on Market1501 as representative dataset:
    1. SA only (baseline)
    2. SA + GLCA
    3. SA + PWCA  
    4. SA + GLCA + PWCA (full model)
    
    Returns:
        ablation_configs: List of ablation study configurations
    """
    
    base_config = Market1501Config()
    
    ablation_configs = []
    
    # SA only
    config_sa = base_config
    config_sa.num_glca_layers = 0
    config_sa.num_pwca_layers = 0
    config_sa.experiment_name = "reid_sa_only"
    ablation_configs.append(config_sa)
    
    # SA + GLCA
    config_sa_glca = base_config
    config_sa_glca.num_pwca_layers = 0
    config_sa_glca.experiment_name = "reid_sa_glca"
    ablation_configs.append(config_sa_glca)
    
    # SA + PWCA
    config_sa_pwca = base_config
    config_sa_pwca.num_glca_layers = 0
    config_sa_pwca.experiment_name = "reid_sa_pwca"
    ablation_configs.append(config_sa_pwca)
    
    # Full model
    config_full = base_config
    config_full.experiment_name = "reid_full_model"
    ablation_configs.append(config_full)
    
    return ablation_configs


def compare_task_configs() -> Dict[str, Dict]:
    """
    Compare key differences between FGVC and Re-ID configurations.
    
    Returns:
        comparison: Dictionary highlighting key differences
    """
    
    from .fgvc_config import CUBConfig
    
    fgvc_config = CUBConfig()
    reid_config = Market1501Config()
    
    comparison = {
        "input_size": {
            "fgvc": fgvc_config.input_size,  # (448, 448)
            "reid": reid_config.input_size   # (256, 128)
        },
        "top_k_ratio": {
            "fgvc": fgvc_config.top_k_ratio,  # 10%
            "reid": reid_config.top_k_ratio   # 30%
        },
        "optimizer": {
            "fgvc": fgvc_config.optimizer,    # Adam
            "reid": reid_config.optimizer     # SGD
        },
        "learning_rate": {
            "fgvc": fgvc_config.scaled_lr,    # Scaled Adam LR
            "reid": reid_config.learning_rate  # Fixed SGD LR
        },
        "loss_functions": {
            "fgvc": "Cross-entropy only",
            "reid": "Cross-entropy + Triplet"
        },
        "batch_sampling": {
            "fgvc": "Random sampling",
            "reid": "Identity-based sampling (4 images per ID)"
        }
    }
    
    return comparison

