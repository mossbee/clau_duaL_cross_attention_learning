"""
Example Usage of Dual Cross-Attention Learning

This script demonstrates how to use the dual cross-attention model 
for both FGVC and Re-ID tasks, showing the complete pipeline from
model creation to training and evaluation.
"""

import torch
import torch.nn as nn
from dual_cross_attention.models import DualCrossAttentionViT
from dual_cross_attention.utils import DualCrossAttentionLoss
from dual_cross_attention.configs import CUBConfig, Market1501Config

def example_fgvc():
    """Example usage for Fine-Grained Visual Categorization"""
    print("="*50)
    print("FGVC Example - CUB-200-2011")
    print("="*50)
    
    # Load configuration
    config = CUBConfig()
    
    # Create model
    model = DualCrossAttentionViT(
        img_size=(448, 448),
        num_classes=200,  # CUB has 200 bird species
        embed_dim=768,
        num_sa_layers=12,  # L=12 SA blocks
        num_glca_layers=1,  # M=1 GLCA block
        num_pwca_layers=12,  # T=12 PWCA blocks
        top_k_ratio=0.1,  # 10% local queries for FGVC
        task_type="fgvc"
    )
    
    # Create loss function
    criterion = DualCrossAttentionLoss(
        task_type="fgvc",
        num_classes=200,
        label_smoothing=0.0
    )
    
    # Example forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 448, 448)
    labels = torch.randint(0, 200, (batch_size,))
    
    # Training mode - with pair-wise cross-attention
    model.train()
    outputs = model(images, images)  # Use same batch as pairs for demo
    
    print(f"Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Compute loss
    targets = {"labels": labels}
    total_loss, loss_dict, metrics_dict = criterion(outputs, targets)
    
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nMetrics:")
    for key, value in metrics_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluation mode - no PWCA
    model.eval()
    with torch.no_grad():
        eval_outputs = model(images)
    
    print(f"\nEvaluation outputs (no PWCA):")
    for key, value in eval_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

def example_reid():
    """Example usage for Object Re-Identification"""
    print("\n" + "="*50)
    print("Re-ID Example - Market1501")
    print("="*50)
    
    # Load configuration  
    config = Market1501Config()
    
    # Create model
    model = DualCrossAttentionViT(
        img_size=(256, 128),  # Person Re-ID resolution
        num_classes=1501,  # Market1501 has 1501 identities
        embed_dim=768,
        num_sa_layers=12,
        num_glca_layers=1,
        num_pwca_layers=12,
        top_k_ratio=0.3,  # 30% local queries for Re-ID
        task_type="reid"
    )
    
    # Create loss function with triplet loss
    criterion = DualCrossAttentionLoss(
        task_type="reid",
        num_classes=1501,
        triplet_margin=0.3,
        triplet_weight=1.0
    )
    
    # Example forward pass
    batch_size = 8  # 2 identities x 4 images each (typical Re-ID sampling)
    images = torch.randn(batch_size, 3, 256, 128)
    identities = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # 4 images per identity
    
    # Training mode
    model.train()
    outputs = model(images, images)  # Pair-wise training
    
    print(f"Re-ID Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Compute loss (includes triplet loss)
    targets = {"labels": identities}
    total_loss, loss_dict, metrics_dict = criterion(outputs, targets)
    
    print(f"\nRe-ID Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")


def example_attention_rollout():
    """Example of attention rollout computation"""
    print("\n" + "="*50)
    print("Attention Rollout Example")
    print("="*50)
    
    from dual_cross_attention.models.attention_modules import AttentionRollout
    
    # Simulate attention weights from multiple layers
    batch_size, num_heads, seq_len = 2, 12, 197  # 196 patches + 1 CLS token
    
    # Create fake attention weights for 12 layers
    attention_weights = []
    for layer in range(12):
        # Random attention weights
        attn = torch.randn(batch_size, num_heads, seq_len, seq_len)
        attn = torch.softmax(attn, dim=-1)
        attention_weights.append(attn)
    
    # Compute attention rollout
    rollout = AttentionRollout(discard_ratio=0.9, head_fusion="mean")
    cls_attention = rollout.rollout(attention_weights)
    
    print(f"Attention rollout shape: {cls_attention.shape}")
    print(f"CLS attention to patches: {cls_attention[0][:10]}")  # First 10 patches


def example_model_configurations():
    """Show different model configurations for tasks"""
    print("\n" + "="*50)
    print("Model Configuration Examples")
    print("="*50)
    
    configs = {
        "CUB (FGVC)": {
            "img_size": (448, 448),
            "num_classes": 200,
            "top_k_ratio": 0.1,
            "task_type": "fgvc"
        },
        "Market1501 (Re-ID)": {
            "img_size": (256, 128), 
            "num_classes": 1501,
            "top_k_ratio": 0.3,
            "task_type": "reid"
        },
        "VeRi-776 (Vehicle Re-ID)": {
            "img_size": (256, 256),
            "num_classes": 776,
            "top_k_ratio": 0.3,
            "task_type": "reid"
        }
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Input size: {config['img_size']}")
        print(f"  Classes: {config['num_classes']}")
        print(f"  Local queries: {config['top_k_ratio']*100:.0f}%")
        print(f"  Task type: {config['task_type']}")


def main():
    """Main function demonstrating all examples"""
    print("Dual Cross-Attention Learning - Example Usage")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run examples
    try:
        example_fgvc()
        example_reid()
        example_attention_rollout()
        example_model_configurations()
        
        print("\n" + "="*50)
        print("All examples completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
