"""
Configuration for Fine-Grained Visual Categorization (FGVC) Tasks

This module defines all hyperparameters and settings for FGVC experiments
as specified in the paper:

Training Settings:
- Resize to 550x550, random crop to 448x448  
- Batch size: 16
- Epochs: 100
- Optimizer: Adam with weight decay 0.05
- Learning rate: lr_scaled = 5e-4 / 512 * batch_size, cosine decay
- Local query ratio R: 10% for all FGVC datasets
- Architecture: L=12 SA blocks, M=1 GLCA block, T=12 PWCA blocks

Datasets: CUB-200-2011, Stanford Cars, FGVC-Aircraft
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional


@dataclass
class FGVCConfig:
    """
    Base configuration for Fine-Grained Visual Categorization tasks.
    
    Contains all common settings shared across FGVC datasets with
    dataset-specific values overridden in subclasses.
    """
    
    # Dataset settings
    task_type: str = "fgvc"
    dataset_name: str = "cub"
    num_classes: int = 200
    input_size: Tuple[int, int] = (448, 448)
    resize_size: int = 550
    
    # Model architecture (from paper)
    embed_dim: int = 768
    num_sa_layers: int = 12      # L=12 SA blocks
    num_glca_layers: int = 1     # M=1 GLCA block  
    num_pwca_layers: int = 12    # T=12 PWCA blocks
    num_heads: int = 12
    hidden_dim: int = 3072
    patch_size: int = 16
    dropout: float = 0.1
    
    # Cross-attention settings
    top_k_ratio: float = 0.1     # R=10% for FGVC local queries
    use_pwca_training: bool = True
    share_pwca_weights: bool = True  # PWCA shares weights with SA
    
    # Training hyperparameters (from paper)
    batch_size: int = 16  # Physical batch size
    gradient_accumulation_steps: int = 2  # Accumulate gradients over 2 steps to maintain effective batch size of 32
    num_epochs: int = 100
    learning_rate: float = 5e-4  # Will be scaled: lr * effective_batch_size / 512
    weight_decay: float = 0.05
    optimizer: str = "adam"
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0  # Gradient clipping (from ViT-pytorch reference)
    use_gradient_checkpointing: bool = False  # DISABLE for speed (not mentioned in paper; trades memory for compute)
    
    # Loss settings
    use_uncertainty_weighting: bool = True
    label_smoothing: float = 0.0
    
    # Data augmentation
    use_stochastic_depth: bool = True
    stochastic_depth_prob: float = 0.1
    use_mixup: bool = False  # Not mentioned in paper for FGVC
    
    # Evaluation settings
    eval_frequency: int = 5  # Evaluate every 5 epochs
    save_frequency: int = 10
    
    # Pretrained model
    pretrained_model: Optional[str] = "/kaggle/input/cub-200-2011/ViT-B_16.npz"  # Auto-detect ViT-B_16 weights if not supplied
    require_pretrained: bool = True  # Enforce pretrained weights per paper for fair reproduction
    freeze_backbone: bool = False
    
    # Hardware settings
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Logging and checkpoints
    log_frequency: int = 50  # Log every 50 batches (reduced overhead)
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Debug and wandb settings
    debug: bool = False
    use_wandb: bool = False
    wandb_project: str = "dual-cross-attention"
    wandb_entity: Optional[str] = None
    experiment_name: Optional[str] = None
    wandb_tags: List[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Calculate effective batch size from physical batch size and gradient accumulation
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        # Scale learning rate based on effective batch size as in paper
        # Paper formula: lr_scaled = 5e-4 / 512 * batch_size
        self.scaled_lr = self.learning_rate * self.effective_batch_size / 512
        
        # Calculate sequence length for patch embeddings
        h_patches = self.input_size[0] // self.patch_size
        w_patches = self.input_size[1] // self.patch_size  
        self.num_patches = h_patches * w_patches
        self.sequence_length = self.num_patches + 1  # +1 for CLS token
    
    def get_model_config(self) -> Dict:
        """Get model configuration dictionary"""
        return {
            "img_size": self.input_size,
            "patch_size": self.patch_size,
            "num_classes": self.num_classes,
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
            "lr": self.scaled_lr,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "warmup_epochs": self.warmup_epochs,
            "num_epochs": self.num_epochs
        }


@dataclass  
class CUBConfig(FGVCConfig):
    """
    Configuration for CUB-200-2011 Bird Species dataset.
    
    Paper results: 
    - DeiT-T: ~88.5% → 89.7% with dual cross-attention
    - 200 bird species, 5,994 training images, 5,794 test images
    """
    
    dataset_name: str = "cub"
    num_classes: int = 200
    
    # Dataset paths (to be set by user)
    data_root: str = "/kaggle/input/cub-200-2011"
    
    # CUB-specific settings
    use_bounding_box: bool = False  # Whether to use provided bounding boxes
    
    def __post_init__(self):
        super().__post_init__()
        
        # CUB dataset statistics for normalization
        self.dataset_mean = [0.485, 0.456, 0.406]  # ImageNet stats
        self.dataset_std = [0.229, 0.224, 0.225]


@dataclass
class CarsConfig(FGVCConfig):
    """
    Configuration for Stanford Cars dataset.
    
    196 car models with fine-grained differences.
    ~8,000 training images, ~8,000 test images.
    """
    
    dataset_name: str = "cars"
    num_classes: int = 196
    
    # Dataset paths
    data_root: str = "/path/to/stanford_cars"
    
    def __post_init__(self):
        super().__post_init__()
        
        self.dataset_mean = [0.485, 0.456, 0.406]
        self.dataset_std = [0.229, 0.224, 0.225]


@dataclass
class AircraftConfig(FGVCConfig):
    """
    Configuration for FGVC-Aircraft dataset.
    
    100 aircraft variants with subtle visual differences.
    ~6,600 training images, ~3,300 test images.
    """
    
    dataset_name: str = "aircraft"
    num_classes: int = 100
    
    # Dataset paths
    data_root: str = "/path/to/fgvc_aircraft"
    
    def __post_init__(self):
        super().__post_init__()
        
        self.dataset_mean = [0.485, 0.456, 0.406]
        self.dataset_std = [0.229, 0.224, 0.225]


def get_fgvc_config(dataset_name: str) -> FGVCConfig:
    """
    Factory function to get appropriate FGVC configuration.
    
    Args:
        dataset_name: "cub", "cars", or "aircraft"
        
    Returns:
        config: Appropriate configuration instance
    """
    
    config_map = {
        "cub": CUBConfig,
        "cars": CarsConfig,  
        "aircraft": AircraftConfig
    }
    
    if dataset_name not in config_map:
        raise ValueError(f"Unknown FGVC dataset: {dataset_name}")
    
    return config_map[dataset_name]()


def create_experiment_configs() -> List[FGVCConfig]:
    """
    Create configurations for all FGVC experiments in the paper.
    
    Returns:
        configs: List of all FGVC experiment configurations
    """
    
    configs = []
    
    # Standard configurations for each dataset
    for dataset in ["cub", "cars", "aircraft"]:
        config = get_fgvc_config(dataset)
        configs.append(config)
    
    return configs


def get_ablation_configs() -> List[FGVCConfig]:
    """
    Create configurations for ablation studies.
    
    Tests different combinations of attention mechanisms:
    1. SA only (baseline)
    2. SA + GLCA  
    3. SA + PWCA
    4. SA + GLCA + PWCA (full model)
    
    Returns:
        ablation_configs: List of ablation study configurations
    """
    
    base_config = CUBConfig()  # Use CUB as default for ablation
    
    ablation_configs = []
    
    # SA only
    config_sa = base_config
    config_sa.num_glca_layers = 0
    config_sa.num_pwca_layers = 0
    config_sa.experiment_name = "sa_only"
    ablation_configs.append(config_sa)
    
    # SA + GLCA
    config_sa_glca = base_config  
    config_sa_glca.num_pwca_layers = 0
    config_sa_glca.experiment_name = "sa_glca"
    ablation_configs.append(config_sa_glca)
    
    # SA + PWCA
    config_sa_pwca = base_config
    config_sa_pwca.num_glca_layers = 0
    config_sa_pwca.experiment_name = "sa_pwca" 
    ablation_configs.append(config_sa_pwca)
    
    # Full model (SA + GLCA + PWCA)
    config_full = base_config
    config_full.experiment_name = "full_model"
    ablation_configs.append(config_full)
    
    return ablation_configs

